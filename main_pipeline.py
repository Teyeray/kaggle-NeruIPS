"""
主流程文件
提供便捷的接口来运行完整的训练和评估流程
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from enhanced_gnn import EnhancedWDMPNN, create_enhanced_graph_from_smiles
from training_pipeline import TrainingPipeline, DEFAULT_CONFIG
from data_augmentation import load_and_augment_train_data, load_test_data, clean_data

class PolymerPropertyPredictor:
    """聚合物性质预测器 - 主接口类"""
    
    def __init__(self, config=None):
        """
        初始化预测器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or DEFAULT_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 初始化模型
        self.model = EnhancedWDMPNN(
            in_channels=10,  # 增强的原子特征维度
            hidden_channels=self.config['hidden_channels'],
            out_channels=1,
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        self.pipeline = None
        self.train_data = None
        self.test_data = None
        self.train_graphs = None
        self.test_graphs = None
        
    def load_data(self, train_path=None, test_path=None):
        """
        加载和预处理数据
        
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径
        """
        print("📂 加载数据...")
        
        # 加载和增强训练数据
        self.train_data = load_and_augment_train_data()
        self.train_data = clean_data(self.train_data)
        
        # 加载测试数据
        self.test_data = load_test_data()
        self.test_data = clean_data(self.test_data)
        
        print(f"✅ 训练数据形状: {self.train_data.shape}")
        print(f"✅ 测试数据形状: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def prepare_graphs(self):
        """准备图数据"""
        print("🔄 创建分子图...")
        
        # 创建训练图数据
        self.train_graphs = [create_enhanced_graph_from_smiles(s).to(self.device) for s in self.train_data['SMILES']]
        self.train_graphs = [g for g in self.train_graphs if g is not None]
        
        # 创建测试图数据
        self.test_graphs = [create_enhanced_graph_from_smiles(s).to(self.device) for s in self.test_data['SMILES']]
        self.test_graphs = [g for g in self.test_graphs if g is not None]
        
        print(f"✅ 训练图数量: {len(self.train_graphs)}")
        print(f"✅ 测试图数量: {len(self.test_graphs)}")
        
        return self.train_graphs, self.test_graphs
    
    def train(self, target_columns=None):
        """
        训练模型
        
        Args:
            target_columns: 目标列列表，默认为所有5个属性
        """
        if target_columns is None:
            target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        print("🚀 开始训练流程...")
        
        # 初始化训练流程
        self.pipeline = TrainingPipeline(self.config)
        
        # 准备数据
        if self.train_data is None:
            self.load_data()
        if self.train_graphs is None:
            self.prepare_graphs()
        
        # 运行完整训练流程
        results = self.pipeline.run_full_pipeline()
        
        return results
    
    def predict(self, smiles_list, target_name):
        """
        对新的SMILES进行预测
        
        Args:
            smiles_list: SMILES字符串列表
            target_name: 目标属性名称
            
        Returns:
            预测结果数组
        """
        print(f"🔮 预测 {target_name}...")
        
        # 加载训练好的模型
        try:
            self.model.load_state_dict(torch.load(f'{target_name}_best_finetune.pt'))
            print(f"✅ 加载 {target_name} 模型权重")
        except:
            print(f"❌ 无法加载 {target_name} 模型权重，请先训练模型")
            return None
        
        # 创建图数据
        graphs = [create_enhanced_graph_from_smiles(s).to(self.device) for s in smiles_list]
        graphs = [g for g in graphs if g is not None]
        
        if len(graphs) == 0:
            print("❌ 没有有效的图数据")
            return None
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            batch = torch.utils.data.DataLoader(graphs, batch_size=32, shuffle=False)
            predictions = []
            
            for batch_graphs in batch:
                _, _, pred = self.model(
                    batch_graphs.x,
                    batch_graphs.edge_index,
                    batch_graphs.edge_attr,
                    batch_graphs.batch
                )
                predictions.extend(pred.cpu().numpy().flatten())
        
        return np.array(predictions)
    
    def evaluate_single_target(self, target_name, test_size=0.2):
        """
        评估单个目标属性的性能
        
        Args:
            target_name: 目标属性名称
            test_size: 测试集比例
        """
        print(f"📊 评估 {target_name} 模型性能...")
        
        # 准备数据
        target_data = self.train_data.dropna(subset=[target_name])
        target_data = target_data[~target_data[target_name].isin([float('inf'), -float('inf')])]
        
        # 分割数据
        train_df, test_df = train_test_split(target_data, test_size=test_size, random_state=42)
        
        # 创建图数据
        train_graphs = [create_enhanced_graph_from_smiles(s).to(self.device) for s in train_df['SMILES']]
        train_graphs = [g for g in train_graphs if g is not None]
        
        test_graphs = [create_enhanced_graph_from_smiles(s).to(self.device) for s in test_df['SMILES']]
        test_graphs = [g for g in test_graphs if g is not None]
        
        # 提取目标值
        train_targets = torch.tensor(train_df[target_name].values, dtype=torch.float32).view(-1, 1).to(self.device)
        test_targets = torch.tensor(test_df[target_name].values, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # 训练模型
        self.pipeline = TrainingPipeline(self.config)
        
        # 预训练
        self.pipeline.run_pretraining(train_graphs)
        
        # 微调
        self.pipeline.finetuner.finetune(
            train_graphs, 
            train_targets, 
            target_name,
            epochs=self.config['finetune_epochs']
        )
        
        # 评估
        results = self.pipeline.evaluator.evaluate(test_graphs, test_targets, target_name)
        
        return results
    
    def get_model_summary(self):
        """获取模型摘要"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("📋 模型摘要:")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  隐藏层维度: {self.config['hidden_channels']}")
        print(f"  层数: {self.config['num_layers']}")
        print(f"  注意力头数: {self.config['num_heads']}")
        print(f"  Dropout率: {self.config['dropout']}")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'config': self.config
        }

def run_quick_demo():
    """运行快速演示"""
    print("🎯 运行快速演示...")
    
    # 创建预测器
    predictor = PolymerPropertyPredictor()
    
    # 获取模型摘要
    predictor.get_model_summary()
    
    # 加载数据
    train_data, test_data = predictor.load_data()
    
    # 准备图数据
    train_graphs, test_graphs = predictor.prepare_graphs()
    
    # 评估单个目标（以Tg为例）
    print("\n" + "="*60)
    print("评估 Tg 模型性能")
    print("="*60)
    results = predictor.evaluate_single_target('Tg', test_size=0.2)
    
    if results:
        print(f"\n📊 Tg 模型性能:")
        print(f"  MSE: {results['mse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  R²:  {results['r2']:.4f}")
    
    # 对新SMILES进行预测
    test_smiles = ["CCO", "c1ccccc1", "CC(C)(C)c1ccc(N)cc1"]
    predictions = predictor.predict(test_smiles, 'Tg')
    
    if predictions is not None:
        print(f"\n🔮 Tg 预测结果:")
        for i, (smiles, pred) in enumerate(zip(test_smiles, predictions)):
            print(f"  {smiles}: {pred:.2f}")

def run_full_training():
    """运行完整训练流程"""
    print("🚀 运行完整训练流程...")
    
    # 创建预测器
    predictor = PolymerPropertyPredictor()
    
    # 运行完整训练
    results = predictor.train()
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    
    # 显示结果摘要
    if results:
        print("\n📊 训练结果摘要:")
        for target, result in results.items():
            if result:
                print(f"  {target}: MSE={result['mse']:.4f}, MAE={result['mae']:.4f}, R²={result['r2']:.4f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        run_full_training()
    else:
        run_quick_demo() 