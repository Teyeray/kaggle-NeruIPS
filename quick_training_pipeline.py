#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速调试版本训练pipeline
参照参考文献实现自监督预训练和针对每个属性的单独微调
"""

import os
import pickle
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import logging
from datetime import datetime

# 导入自定义模块
from enhanced_gnn import EnhancedWDMPNN, create_enhanced_graph_from_smiles
from data_augmentation import load_and_augment_train_data

# 忽略警告
warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quick_training.log'),
        logging.StreamHandler()
    ]
)

class QuickTrainingConfig:
    """快速训练配置类"""
    def __init__(self):
        # 训练参数（调试版本，最小最快）
        self.BATCH_SIZE = 32  # 小批量
        self.LEARNING_RATE = 1e-3
        self.EPOCHS = 10  # 少轮次
        self.HIDDEN_DIM = 64  # 小维度
        self.NUM_LAYERS = 2  # 少层数
        self.DROPOUT = 0.1
        self.SEED = 42
        
        # 自监督预训练参数
        self.PRETRAIN_EPOCHS = 5
        self.MASK_RATIO = 0.15  # 节点和边掩码比例
        
        # 目标变量
        self.TARGETS = ['Tg', 'Tc', 'Rg', 'FFV', 'Density']
        
        # 路径配置
        self.FEATURE_PATH = 'feature_engineering_output/'
        self.MODEL_SAVE_PATH = 'quick_models/'
        self.RESULTS_PATH = 'quick_results/'
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.RESULTS_PATH, exist_ok=True)

class SelfSupervisedTrainer:
    """自监督预训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def create_pretraining_data(self, smiles_list):
        """创建预训练数据（节点和边掩码）"""
        print("🔄 创建自监督预训练数据...")
        
        # 简化的预训练数据创建
        # 在实际应用中，这里应该创建图数据并进行掩码
        # 为了快速调试，我们使用简化的特征掩码
        
        # 加载特征数据
        train_features = pd.read_pickle(self.config.FEATURE_PATH + 'train_features_complete.pkl')
        
        # 选择有SMILES的样本
        valid_data = train_features[train_features['SMILES'].notnull()].copy()
        
        # 提取分子特征列（排除基础列）
        base_cols = ['id', 'SMILES'] + self.config.TARGETS
        feature_cols = [col for col in valid_data.columns if col not in base_cols]
        
        # 创建掩码数据
        masked_data = []
        original_data = []
        
        for idx in tqdm(range(min(1000, len(valid_data)))):  # 限制样本数用于快速调试
            features = valid_data.iloc[idx][feature_cols].values
            
            # 确保数据类型为float
            features = features.astype(np.float32)
            
            # 处理NaN值和无穷大值
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            features = np.clip(features, -1e6, 1e6)
            
            # 随机掩码一些特征
            mask = np.random.random(len(features)) < self.config.MASK_RATIO
            masked_features = features.copy()
            masked_features[mask] = 0  # 简单掩码为0
            
            masked_data.append(masked_features)
            original_data.append(features)
        
        return np.array(masked_data, dtype=np.float32), np.array(original_data, dtype=np.float32)
    
    def pretrain(self, smiles_list):
        """自监督预训练"""
        print("🚀 开始自监督预训练...")
        
        # 创建预训练数据
        masked_data, original_data = self.create_pretraining_data(smiles_list)
        
        # 转换为tensor
        masked_tensor = torch.FloatTensor(masked_data).to(self.device)
        original_tensor = torch.FloatTensor(original_data).to(self.device)
        
        # 创建简单的重建模型（用于快速调试）
        input_dim = masked_data.shape[1]
        self.pretrain_model = nn.Sequential(
            nn.Linear(input_dim, self.config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM, input_dim)
        ).to(self.device)
        
        # 训练
        optimizer = optim.Adam(self.pretrain_model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        print(f"📊 预训练数据形状: {masked_data.shape}")
        
        for epoch in range(self.config.PRETRAIN_EPOCHS):
            self.pretrain_model.train()
            optimizer.zero_grad()
            
            # 前向传播
            reconstructed = self.pretrain_model(masked_tensor)
            loss = criterion(reconstructed, original_tensor)
            
            # 检查损失是否为NaN或无穷大
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 预训练 Epoch {epoch+1} 损失异常: {loss.item()}")
                # 跳过这个epoch
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.pretrain_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"预训练 Epoch {epoch+1}/{self.config.PRETRAIN_EPOCHS}, Loss: {loss.item():.4f}")
        
        print("✅ 自监督预训练完成!")
        return self.pretrain_model

class PropertySpecificTrainer:
    """针对特定属性的训练器"""
    def __init__(self, config, target_property):
        self.config = config
        self.target_property = target_property
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self):
        """准备特定属性的训练数据"""
        print(f"📂 准备 {self.target_property} 的训练数据...")
        
        # 加载特征数据
        train_features = pd.read_pickle(self.config.FEATURE_PATH + 'train_features_complete.pkl')
        
        # 选择有目标值的样本
        valid_data = train_features[train_features[self.target_property].notnull()].copy()
        
        print(f"📊 {self.target_property} 有效样本数: {len(valid_data)}")
        
        # 提取特征和目标
        base_cols = ['id', 'SMILES'] + self.config.TARGETS
        feature_cols = [col for col in valid_data.columns if col not in base_cols]
        
        X = valid_data[feature_cols].values
        y = valid_data[self.target_property].values
        
        # 处理NaN值和无穷大值
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 检查并处理异常值
        X = np.clip(X, -1e6, 1e6)
        y = np.clip(y, -1e6, 1e6)
        
        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=self.config.SEED
        )
        
        # 转换为tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, scaler_X, scaler_y
    
    def create_model(self, input_dim):
        """创建针对特定属性的模型"""
        # 为了保持一致性，总是创建新的模型结构
        model = nn.Sequential(
            nn.Linear(input_dim, self.config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT),
            nn.Linear(self.config.HIDDEN_DIM // 2, 1)
        ).to(self.device)
        
        # 如果使用预训练模型，可以初始化权重（可选）
        if hasattr(self, 'pretrain_model'):
            # 这里可以添加权重初始化逻辑
            pass
        
        return model
    
    def train(self, pretrain_model=None):
        """训练特定属性的模型"""
        print(f"🚀 开始训练 {self.target_property} 模型...")
        
        # 设置预训练模型
        if pretrain_model is not None:
            self.pretrain_model = pretrain_model
        
        # 准备数据
        train_loader, val_loader, scaler_X, scaler_y = self.prepare_data()
        
        # 创建模型
        input_dim = train_loader.dataset[0][0].shape[0]
        model = self.create_model(input_dim)
        
        # 训练设置
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # 训练循环
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.EPOCHS):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).flatten()
                loss = criterion(outputs, batch_y)
                
                # 检查损失是否为NaN或无穷大
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ 训练损失异常: {loss.item()}")
                    continue
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).flatten()
                    loss = criterion(outputs, batch_y)
                    
                    # 检查损失是否为NaN或无穷大
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ 验证损失异常: {loss.item()}")
                        continue
                    
                    val_loss += loss.item()
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            print(f"  训练损失: {train_losses[-1]:.4f}")
            print(f"  验证损失: {val_losses[-1]:.4f}")
        
        # 计算最终指标
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        # 处理预测结果中的NaN值
        val_predictions = np.nan_to_num(val_predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        val_targets = np.nan_to_num(val_targets, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 反标准化
        val_predictions_original = scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
        val_targets_original = scaler_y.inverse_transform(val_targets.reshape(-1, 1)).flatten()
        
        # 再次处理反标准化后的NaN值
        val_predictions_original = np.nan_to_num(val_predictions_original, nan=0.0, posinf=1e6, neginf=-1e6)
        val_targets_original = np.nan_to_num(val_targets_original, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 计算指标
        mse = mean_squared_error(val_targets_original, val_predictions_original)
        mae = mean_absolute_error(val_targets_original, val_predictions_original)
        r2 = r2_score(val_targets_original, val_predictions_original)
        
        print(f"\n📊 {self.target_property} 最终结果:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # 保存模型
        model_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{self.target_property}_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
            'train_losses': train_losses,
            'val_losses': val_losses
        }, model_path)
        
        print(f"💾 模型已保存: {model_path}")
        
        return model, {'mse': mse, 'mae': mae, 'r2': r2}, (train_losses, val_losses)

class TestPredictor:
    """测试数据预测器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self):
        """加载所有训练好的模型"""
        models = {}
        scalers = {}
        
        for target in self.config.TARGETS:
            model_path = os.path.join(self.config.MODEL_SAVE_PATH, f'{target}_model.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # 重建模型结构
                input_dim = checkpoint['scaler_X'].n_features_in_
                model = nn.Sequential(
                    nn.Linear(input_dim, self.config.HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Dropout(self.config.DROPOUT),
                    nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM // 2),
                    nn.ReLU(),
                    nn.Dropout(self.config.DROPOUT),
                    nn.Linear(self.config.HIDDEN_DIM // 2, 1)
                ).to(self.device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                models[target] = model
                scalers[target] = {
                    'X': checkpoint['scaler_X'],
                    'y': checkpoint['scaler_y']
                }
                
                print(f"✅ 已加载 {target} 模型")
            else:
                print(f"⚠️ 未找到 {target} 模型文件")
        
        return models, scalers
    
    def predict_test_data(self):
        """预测测试数据"""
        print("🔮 开始预测测试数据...")
        
        # 加载模型
        models, scalers = self.load_models()
        
        if not models:
            print("❌ 没有可用的模型进行预测")
            return None
        
        # 加载测试特征
        test_features_path = os.path.join(self.config.FEATURE_PATH, 'test_features_complete.pkl')
        if not os.path.exists(test_features_path):
            print(f"❌ 测试特征文件不存在: {test_features_path}")
            return None
        
        test_features = pd.read_pickle(test_features_path)
        
        # 准备预测结果
        predictions = {}
        
        for target in self.config.TARGETS:
            if target not in models:
                print(f"⚠️ 跳过 {target}，模型不可用")
                continue
            
            print(f"🔮 预测 {target}...")
            
            # 提取特征
            base_cols = ['id', 'SMILES'] + self.config.TARGETS
            feature_cols = [col for col in test_features.columns if col not in base_cols]
            
            X_test = test_features[feature_cols].values
            X_test = np.nan_to_num(X_test, nan=0.0)
            
            # 标准化
            X_test_scaled = scalers[target]['X'].transform(X_test)
            
            # 预测
            model = models[target]
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            
            with torch.no_grad():
                predictions_scaled = model(X_test_tensor).cpu().numpy().flatten()
            
            # 反标准化
            predictions_original = scalers[target]['y'].inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            predictions[target] = predictions_original
        
        # 创建提交文件
        if predictions:
            submission = pd.DataFrame({'id': test_features['id']})
            for target, preds in predictions.items():
                submission[target] = preds
            
            # 保存预测结果
            submission_path = os.path.join(self.config.RESULTS_PATH, 'quick_submission.csv')
            submission.to_csv(submission_path, index=False)
            print(f"💾 预测结果已保存: {submission_path}")
            
            return submission
        
        return None

class ModelEvaluator:
    """模型评估器"""
    def __init__(self, config):
        self.config = config
        
    def plot_training_curves(self, results):
        """绘制训练曲线"""
        print("📊 绘制训练曲线...")
        
        n_targets = len(results)
        fig, axes = plt.subplots(2, n_targets, figsize=(5*n_targets, 10))
        if n_targets == 1:
            axes = axes.reshape(2, 1)
        
        for i, (target, data) in enumerate(results.items()):
            train_losses, val_losses = data['losses']
            
            # 损失曲线
            axes[0, i].plot(train_losses, label='训练损失', color='blue')
            axes[0, i].plot(val_losses, label='验证损失', color='red')
            axes[0, i].set_title(f'{target} - 训练损失')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('损失')
            axes[0, i].legend()
            axes[0, i].grid(True)
            
            # 指标柱状图
            metrics = data['metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[1, i].bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[1, i].set_title(f'{target} - 评估指标')
            axes[1, i].set_ylabel('值')
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, metric_values):
                axes[1, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.RESULTS_PATH, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 训练曲线已保存: {plot_path}")
    
    def generate_report(self, results, submission=None):
        """生成评估报告"""
        print("📋 生成评估报告...")
        
        report = []
        report.append("# 快速训练Pipeline评估报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 模型性能汇总
        report.append("## 模型性能汇总")
        report.append("")
        
        for target, data in results.items():
            metrics = data['metrics']
            report.append(f"### {target}")
            report.append(f"- MSE: {metrics['mse']:.4f}")
            report.append(f"- MAE: {metrics['mae']:.4f}")
            report.append(f"- R²: {metrics['r2']:.4f}")
            report.append("")
        
        # 平均性能
        avg_mse = np.mean([data['metrics']['mse'] for data in results.values()])
        avg_mae = np.mean([data['metrics']['mae'] for data in results.values()])
        avg_r2 = np.mean([data['metrics']['r2'] for data in results.values()])
        
        report.append("## 平均性能")
        report.append(f"- 平均MSE: {avg_mse:.4f}")
        report.append(f"- 平均MAE: {avg_mae:.4f}")
        report.append(f"- 平均R²: {avg_r2:.4f}")
        report.append("")
        
        # 预测信息
        if submission is not None:
            report.append("## 预测信息")
            report.append(f"- 测试样本数: {len(submission)}")
            report.append(f"- 预测属性: {', '.join(self.config.TARGETS)}")
            report.append("")
        
        # 保存报告
        report_path = os.path.join(self.config.RESULTS_PATH, 'evaluation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📋 评估报告已保存: {report_path}")

def main():
    """主函数"""
    print("🎯 快速调试版本训练Pipeline")
    print("="*70)
    
    # 初始化配置
    config = QuickTrainingConfig()
    
    try:
        # 步骤1: 自监督预训练
        print("\n" + "="*50)
        print("🏋️ 自监督预训练阶段")
        print("="*50)
        
        # 加载所有SMILES用于预训练
        train_data = load_and_augment_train_data()
        all_smiles = train_data['SMILES'].dropna().tolist()
        
        pretrainer = SelfSupervisedTrainer(config)
        pretrain_model = pretrainer.pretrain(all_smiles)
        
        # 步骤2: 针对每个属性单独训练
        print("\n" + "="*50)
        print("🎯 属性特定微调阶段")
        print("="*50)
        
        results = {}
        
        for target in config.TARGETS:
            print(f"\n--- 训练 {target} 模型 ---")
            trainer = PropertySpecificTrainer(config, target)
            model, metrics, losses = trainer.train(pretrain_model)
            results[target] = {
                'metrics': metrics,
                'losses': losses
            }
        
        # 步骤3: 测试数据预测
        print("\n" + "="*50)
        print("🔮 测试数据预测阶段")
        print("="*50)
        
        predictor = TestPredictor(config)
        submission = predictor.predict_test_data()
        
        # 步骤4: 模型评估和可视化
        print("\n" + "="*50)
        print("📊 模型评估阶段")
        print("="*50)
        
        evaluator = ModelEvaluator(config)
        evaluator.plot_training_curves(results)
        evaluator.generate_report(results, submission)
        
        # 步骤5: 结果总结
        print("\n" + "="*70)
        print("🎉 训练完成!")
        print("="*70)
        
        print("📊 各属性结果汇总:")
        for target, data in results.items():
            metrics = data['metrics']
            print(f"  {target}:")
            print(f"    MSE: {metrics['mse']:.4f}")
            print(f"    MAE: {metrics['mae']:.4f}")
            print(f"    R²: {metrics['r2']:.4f}")
        
        # 保存总体结果
        results_path = os.path.join(config.MODEL_SAVE_PATH, 'training_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 结果已保存: {results_path}")
        
        if submission is not None:
            print(f"📤 提交文件已生成: {os.path.join(config.RESULTS_PATH, 'quick_submission.csv')}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        logging.error(f"训练错误: {e}", exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 