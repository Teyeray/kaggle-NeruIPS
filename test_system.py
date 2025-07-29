"""
系统功能测试脚本
测试所有模块的基本功能
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Batch

def test_basic_imports():
    """测试基础导入"""
    print("🔍 测试基础导入...")
    
    try:
        from data_augmentation import make_smile_canonical, clean_data
        from feature_engineering import clean_features_and_compute_means, USELESS_COLS
        from enhanced_gnn import EnhancedWDMPNN, create_enhanced_graph_from_smiles
        from training_pipeline import TrainingPipeline, DEFAULT_CONFIG
        from main_pipeline import PolymerPropertyPredictor
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_graph_creation():
    """测试图创建功能"""
    print("\n🔍 测试图创建功能...")
    
    try:
        from enhanced_gnn import create_enhanced_graph_from_smiles
        
        # 测试单个图创建
        graph = create_enhanced_graph_from_smiles('CCO')
        print(f"✅ 单个图创建成功: 节点数={graph.x.shape[0]}, 边数={graph.edge_index.shape[1]}")
        
        # 测试多个图创建
        smiles_list = ['CCO', 'c1ccccc1', 'CC(C)(C)c1ccc(N)cc1']
        graphs = [create_enhanced_graph_from_smiles(s) for s in smiles_list]
        graphs = [g for g in graphs if g is not None]
        print(f"✅ 多个图创建成功: {len(graphs)}个有效图")
        
        return True
    except Exception as e:
        print(f"❌ 图创建失败: {e}")
        return False

def test_model_forward():
    """测试模型前向传播"""
    print("\n🔍 测试模型前向传播...")
    
    try:
        from enhanced_gnn import EnhancedWDMPNN, create_enhanced_graph_from_smiles
        
        device = torch.device('cpu')
        model = EnhancedWDMPNN(10, 64, 1).to(device)
        
        # 测试单个图
        graph = create_enhanced_graph_from_smiles('CCO').to(device)
        node_out, edge_out, graph_out = model(graph.x, graph.edge_index, graph.edge_attr)
        print(f"✅ 单个图前向传播成功: 输出形状={graph_out.shape}")
        
        # 测试批处理
        graphs = [create_enhanced_graph_from_smiles(s).to(device) for s in ['CCO', 'c1ccccc1']]
        batch = Batch.from_data_list(graphs).to(device)
        node_out, edge_out, graph_out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        print(f"✅ 批处理前向传播成功: 输出形状={graph_out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 模型前向传播失败: {e}")
        return False

def test_data_augmentation():
    """测试数据增强功能"""
    print("\n🔍 测试数据增强功能...")
    
    try:
        from data_augmentation import make_smile_canonical
        from feature_engineering import clean_features_and_compute_means
        
        # 测试SMILES标准化
        test_smiles = ['CCO', 'c1ccccc1', 'CC(C)(C)c1ccc(N)cc1']
        canonical_smiles = [make_smile_canonical(s) for s in test_smiles]
        print(f"✅ SMILES标准化成功: {canonical_smiles}")
        
        # 测试数据清洗
        test_df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [5, np.inf, 7, 8]
        })
        cleaned_df, means = clean_features_and_compute_means(test_df, ['feature1', 'feature2'])
        print(f"✅ 数据清洗成功: 均值={means}")
        
        return True
    except Exception as e:
        print(f"❌ 数据增强失败: {e}")
        return False

def test_training_components():
    """测试训练组件"""
    print("\n🔍 测试训练组件...")
    
    try:
        from enhanced_gnn import EnhancedWDMPNN, create_enhanced_graph_from_smiles
        from training_pipeline import SelfSupervisedTrainer, SupervisedTrainer, Evaluator
        
        device = torch.device('cpu')
        model = EnhancedWDMPNN(10, 64, 1).to(device)
        
        # 测试自监督训练器
        trainer = SelfSupervisedTrainer(model, device, {'pretrain_lr': 0.001})
        graph = create_enhanced_graph_from_smiles('CCO').to(device)
        masked_graph = trainer.mask_data(graph)
        print(f"✅ 自监督训练器测试成功: 掩码节点数={masked_graph.masked_nodes.sum().item()}")
        
        # 测试监督训练器
        finetuner = SupervisedTrainer(model, device, {'finetune_lr': 0.0005})
        print("✅ 监督训练器初始化成功")
        
        # 测试评估器
        evaluator = Evaluator(model, device)
        graphs = [create_enhanced_graph_from_smiles(s).to(device) for s in ['CCO', 'c1ccccc1']]
        targets = torch.tensor([[1.0], [2.0]]).to(device)
        print("✅ 评估器初始化成功")
        
        return True
    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        return False

def test_main_interface():
    """测试主接口"""
    print("\n🔍 测试主接口...")
    
    try:
        from main_pipeline import PolymerPropertyPredictor
        
        # 创建预测器
        predictor = PolymerPropertyPredictor()
        
        # 获取模型摘要
        summary = predictor.get_model_summary()
        print(f"✅ 模型摘要获取成功: 参数数量={summary['total_params']:,}")
        
        # 测试自定义配置
        custom_config = {
            'hidden_channels': 128,
            'num_layers': 3,
            'num_heads': 4,
            'dropout': 0.2
        }
        predictor_custom = PolymerPropertyPredictor(custom_config)
        print("✅ 自定义配置测试成功")
        
        return True
    except Exception as e:
        print(f"❌ 主接口测试失败: {e}")
        return False

def test_end_to_end():
    """端到端测试"""
    print("\n🔍 端到端测试...")
    
    try:
        from main_pipeline import PolymerPropertyPredictor
        from enhanced_gnn import create_enhanced_graph_from_smiles
        
        # 创建预测器
        predictor = PolymerPropertyPredictor()
        
        # 测试图创建
        test_smiles = ["CCO", "c1ccccc1"]
        graphs = [create_enhanced_graph_from_smiles(s) for s in test_smiles]
        graphs = [g for g in graphs if g is not None]
        print(f"✅ 端到端图创建成功: {len(graphs)}个图")
        
        # 测试模型预测（不训练，只测试前向传播）
        device = torch.device('cpu')
        model = predictor.model.to(device)
        
        for i, graph in enumerate(graphs):
            graph = graph.to(device)
            with torch.no_grad():
                node_out, edge_out, graph_out = model(graph.x, graph.edge_index, graph.edge_attr)
            print(f"✅ 图{i+1}预测成功: 输出={graph_out.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始系统功能测试")
    print("="*60)
    
    tests = [
        test_basic_imports,
        test_graph_creation,
        test_model_forward,
        test_data_augmentation,
        test_training_components,
        test_main_interface,
        test_end_to_end
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "="*60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统功能正常")
    else:
        print("⚠️  部分测试失败，请检查相关模块")
    
    return passed == total

if __name__ == "__main__":
    main() 