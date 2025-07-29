"""
优化的训练流程模块
包含自监督预训练、微调和评估功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Data, DataLoader, Batch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

from enhanced_gnn import EnhancedWDMPNN, create_enhanced_graph_from_smiles
from data_augmentation import load_and_augment_train_data, load_test_data, clean_data
from feature_engineering import process_molecular_features_parallel, USELESS_COLS

class SelfSupervisedTrainer:
    """自监督训练器"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
    def mask_data(self, data, node_mask_ratio=0.15, edge_mask_ratio=0.15):
        """增强的掩码函数（论文Section 2.3.1）"""
        data = deepcopy(data)
        
        # 节点掩码
        num_nodes = data.x.size(0)
        node_mask = torch.rand(num_nodes) < node_mask_ratio
        data.masked_nodes = node_mask
        data.original_x = data.x.clone()
        data.x[node_mask] = 0
        
        # 边掩码
        num_edges = data.edge_index.size(1)
        edge_mask = torch.rand(num_edges) < edge_mask_ratio
        data.masked_edges = edge_mask
        data.original_edge_attr = data.edge_attr.clone()
        data.edge_attr[edge_mask] = 0
        
        return data
    
    def pretrain(self, graph_data, epochs=100, patience=10):
        """自监督预训练"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['pretrain_lr'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        best_loss = float('inf')
        no_improve = 0
        
        print("🚀 开始自监督预训练...")
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # 掩码并批处理
            masked_data = [self.mask_data(g) for g in graph_data]
            
            # 验证数据有效性
            valid_data = []
            for i, data in enumerate(masked_data):
                if data.x.size(0) > 0 and data.edge_index.size(1) > 0:
                    valid_data.append(data)
            
            if len(valid_data) == 0:
                print("❌ 错误: 没有有效的图数据")
                break
                
            try:
                batch_obj = Batch.from_data_list(valid_data)
                
                # 前向传播
                node_pred, edge_pred, _ = self.model(
                    batch_obj.x, 
                    batch_obj.edge_index, 
                    batch_obj.edge_attr, 
                    batch_obj.batch
                )
                
                # 收集掩码信息
                all_masked_nodes = []
                all_masked_edges = []
                all_original_x = []
                all_original_edge_attr = []
                
                node_offset = 0
                edge_offset = 0
                
                for data in valid_data:
                    # 节点掩码
                    masked_node_indices = torch.where(data.masked_nodes)[0] + node_offset
                    all_masked_nodes.append(masked_node_indices)
                    all_original_x.append(data.original_x[data.masked_nodes])
                    
                    # 边掩码
                    masked_edge_indices = torch.where(data.masked_edges)[0] + edge_offset
                    all_masked_edges.append(masked_edge_indices)
                    all_original_edge_attr.append(data.original_edge_attr[data.masked_edges])
                    
                    node_offset += data.x.size(0)
                    edge_offset += data.edge_index.size(1)
                
                # 计算损失
                node_loss = torch.tensor(0.0, requires_grad=True)
                edge_loss = torch.tensor(0.0, requires_grad=True)
                
                if all_masked_nodes and any(len(mask) > 0 for mask in all_masked_nodes):
                    masked_node_indices = torch.cat([mask for mask in all_masked_nodes if len(mask) > 0])
                    original_node_features = torch.cat([feat for feat in all_original_x if len(feat) > 0])
                    node_loss = F.mse_loss(node_pred[masked_node_indices], original_node_features)
                
                if all_masked_edges and any(len(mask) > 0 for mask in all_masked_edges):
                    masked_edge_indices = torch.cat([mask for mask in all_masked_edges if len(mask) > 0])
                    original_edge_features = torch.cat([feat for feat in all_original_edge_attr if len(feat) > 0])
                    edge_loss = F.mse_loss(edge_pred[masked_edge_indices], original_edge_features)
                
                # 总损失
                loss = node_loss + 0.5 * edge_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                
                # 早停机制
                if loss < best_loss:
                    best_loss = loss
                    no_improve = 0
                    torch.save(self.model.state_dict(), 'best_pretrain.pt')
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"⏹️ 早停于第 {epoch+1} 轮")
                        break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Node Loss: {node_loss.item():.4f} | Edge Loss: {edge_loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
                
            except Exception as e:
                print(f"❌ Epoch {epoch+1} 出错: {e}")
                continue
        
        print("✅ 预训练完成!")

class SupervisedTrainer:
    """监督训练器"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
    def finetune(self, graph_data, targets, target_name, epochs=200, patience=15):
        """监督微调函数"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['finetune_lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        best_loss = float('inf')
        no_improve = 0
        
        print(f"🎯 开始微调 {target_name} 模型...")
        
        if len(graph_data) != len(targets):
            raise ValueError(f"图数据数量({len(graph_data)})与目标数量({len(targets)})不匹配")
        
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(-1)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            try:
                valid_data = []
                valid_targets = []
                for i, (data, target) in enumerate(zip(graph_data, targets)):
                    if data.x.size(0) == 0 or data.edge_index.size(1) == 0:
                        continue
                    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                        continue
                    if torch.isnan(target).any() or torch.isinf(target).any():
                        continue
                    valid_data.append(data)
                    valid_targets.append(target)

                if len(valid_data) == 0:
                    print("❌ 错误: 没有有效的图数据")
                    break

                valid_targets = torch.stack(valid_targets) if len(valid_targets) > 1 else valid_targets[0].unsqueeze(0)

                batch_obj = Batch.from_data_list(valid_data)
                _, _, graph_pred = self.model(
                    batch_obj.x,
                    batch_obj.edge_index,
                    batch_obj.edge_attr,
                    batch_obj.batch
                )

                loss = F.mse_loss(graph_pred, valid_targets)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if loss < best_loss:
                    best_loss = loss
                    no_improve = 0
                    torch.save(self.model.state_dict(), f'{target_name}_best_finetune.pt')
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"⏹️ 早停于第 {epoch+1} 轮")
                        break

                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            except Exception as e:
                print(f"❌ Epoch {epoch+1} 出错: {e}")
                continue

        print(f"✅ {target_name} 微调完成!")
        print(f"最佳损失: {best_loss:.4f}")

class Evaluator:
    """评估器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate(self, graph_data, targets, target_name):
        """评估模型性能"""
        self.model.eval()
        
        with torch.no_grad():
            valid_data = []
            valid_targets = []
            
            for data, target in zip(graph_data, targets):
                if data.x.size(0) > 0 and data.edge_index.size(1) > 0:
                    valid_data.append(data)
                    valid_targets.append(target)
            
            if len(valid_data) == 0:
                return None
            
            valid_targets = torch.stack(valid_targets) if len(valid_targets) > 1 else valid_targets[0].unsqueeze(0)
            batch_obj = Batch.from_data_list(valid_data)
            
            _, _, predictions = self.model(
                batch_obj.x,
                batch_obj.edge_index,
                batch_obj.edge_attr,
                batch_obj.batch
            )
            
            # 计算指标
            targets_np = valid_targets.cpu().numpy().flatten()
            preds_np = predictions.cpu().numpy().flatten()
            
            mse = mean_squared_error(targets_np, preds_np)
            mae = mean_absolute_error(targets_np, preds_np)
            r2 = r2_score(targets_np, preds_np)
            
            results = {
                'target': target_name,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': preds_np,
                'targets': targets_np
            }
            
            print(f"📊 {target_name} 评估结果:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²:  {r2:.4f}")
            
            return results
    
    def plot_results(self, results_dict):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('模型性能评估结果', fontsize=16)
        
        for i, (target, results) in enumerate(results_dict.items()):
            if results is None:
                continue
                
            row = i // 3
            col = i % 3
            
            # 散点图
            axes[row, col].scatter(results['targets'], results['predictions'], alpha=0.6)
            axes[row, col].plot([results['targets'].min(), results['targets'].max()], 
                              [results['targets'].min(), results['targets'].max()], 'r--', lw=2)
            axes[row, col].set_xlabel('真实值')
            axes[row, col].set_ylabel('预测值')
            axes[row, col].set_title(f'{target} (R² = {results["r2"]:.3f})')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

class TrainingPipeline:
    """完整的训练流程"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 初始化模型
        self.model = EnhancedWDMPNN(
            in_channels=10,  # 增强的原子特征维度
            hidden_channels=config['hidden_channels'],
            out_channels=1,
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(self.device)
        
        # 初始化训练器
        self.pretrainer = SelfSupervisedTrainer(self.model, self.device, config)
        self.finetuner = SupervisedTrainer(self.model, self.device, config)
        self.evaluator = Evaluator(self.model, self.device)
        
    def prepare_data(self):
        """准备数据"""
        print("📂 准备数据...")
        
        # 加载和增强训练数据
        train_data = load_and_augment_train_data()
        train_data = clean_data(train_data)
        
        # 加载测试数据
        test_data = load_test_data()
        test_data = clean_data(test_data)
        
        # 创建图数据
        print("🔄 创建分子图...")
        train_graphs = [create_enhanced_graph_from_smiles(s).to(self.device) for s in train_data['SMILES']]
        train_graphs = [g for g in train_graphs if g is not None]
        
        test_graphs = [create_enhanced_graph_from_smiles(s).to(self.device) for s in test_data['SMILES']]
        test_graphs = [g for g in test_graphs if g is not None]
        
        print(f"✅ 训练图数量: {len(train_graphs)}")
        print(f"✅ 测试图数量: {len(test_graphs)}")
        
        return train_data, test_data, train_graphs, test_graphs
    
    def run_pretraining(self, train_graphs):
        """运行预训练"""
        print("🚀 开始预训练阶段...")
        self.pretrainer.pretrain(train_graphs, epochs=self.config['pretrain_epochs'])
    
    def run_finetuning(self, train_data, train_graphs, target_columns):
        """运行微调"""
        print("🎯 开始微调阶段...")
        
        results = {}
        for target in target_columns:
            print(f"\n{'='*50}")
            print(f"训练 {target} 模型")
            print(f"{'='*50}")
            
            # 清洗数据
            target_data = train_data.dropna(subset=[target])
            target_data = target_data[~target_data[target].isin([float('inf'), -float('inf')])]
            
            # 重新生成图数据
            target_graphs = [create_enhanced_graph_from_smiles(s).to(self.device) for s in target_data['SMILES']]
            target_graphs = [g for g in target_graphs if g is not None]
            
            # 提取目标值
            target_values = torch.tensor(target_data[target].values, dtype=torch.float32).view(-1, 1).to(self.device)
            
            # 筛选有效数据
            valid_graphs = []
            valid_targets = []
            for g, t in zip(target_graphs, target_values):
                if g.x.size(0) > 0 and g.edge_index.size(1) > 0:
                    valid_graphs.append(g)
                    valid_targets.append(t)
            
            if len(valid_graphs) == 0:
                print(f"❌ 没有有效的图数据用于训练 {target}")
                continue
            
            # 加载预训练权重
            try:
                self.model.load_state_dict(torch.load('best_pretrain.pt'))
                print(f"✅ 加载预训练权重")
            except:
                print(f"⚠️ 无法加载预训练权重，使用随机初始化")
            
            # 微调
            self.finetuner.finetune(
                valid_graphs, 
                torch.stack(valid_targets), 
                target,
                epochs=self.config['finetune_epochs']
            )
            
            # 评估
            results[target] = self.evaluator.evaluate(valid_graphs, torch.stack(valid_targets), target)
        
        return results
    
    def run_full_pipeline(self):
        """运行完整训练流程"""
        print("🚀 开始完整训练流程")
        print("="*60)
        
        # 1. 准备数据
        train_data, test_data, train_graphs, test_graphs = self.prepare_data()
        
        # 2. 预训练
        self.run_pretraining(train_graphs)
        
        # 3. 微调
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        results = self.run_finetuning(train_data, train_graphs, target_columns)
        
        # 4. 评估和可视化
        self.evaluator.plot_results(results)
        
        # 5. 保存结果
        self.save_results(results)
        
        return results
    
    def save_results(self, results):
        """保存结果"""
        print("💾 保存结果...")
        
        # 保存评估结果
        results_df = pd.DataFrame([
            {
                'target': target,
                'mse': res['mse'],
                'mae': res['mae'],
                'r2': res['r2']
            }
            for target, res in results.items() if res is not None
        ])
        
        results_df.to_csv('model_evaluation_results.csv', index=False)
        print("✅ 结果已保存到 model_evaluation_results.csv")
        
        # 计算平均性能
        avg_mse = results_df['mse'].mean()
        avg_mae = results_df['mae'].mean()
        avg_r2 = results_df['r2'].mean()
        
        print(f"\n📊 平均性能:")
        print(f"  平均 MSE: {avg_mse:.4f}")
        print(f"  平均 MAE: {avg_mae:.4f}")
        print(f"  平均 R²:  {avg_r2:.4f}")

# 配置参数
DEFAULT_CONFIG = {
    'hidden_channels': 256,
    'num_layers': 4,
    'num_heads': 8,
    'dropout': 0.1,
    'pretrain_lr': 0.001,
    'finetune_lr': 0.0005,
    'pretrain_epochs': 100,
    'finetune_epochs': 200
}

if __name__ == "__main__":
    # 运行完整训练流程
    pipeline = TrainingPipeline(DEFAULT_CONFIG)
    results = pipeline.run_full_pipeline()
