"""
使用示例
展示如何使用优化后的聚合物性质预测系统
"""

import torch
import pandas as pd
import numpy as np
from main_pipeline import PolymerPropertyPredictor, run_quick_demo, run_full_training

def example_1_basic_usage():
    """示例1: 基本使用"""
    print("="*60)
    print("示例1: 基本使用")
    print("="*60)
    
    # 创建预测器
    predictor = PolymerPropertyPredictor()
    
    # 获取模型摘要
    predictor.get_model_summary()
    
    # 加载数据
    train_data, test_data = predictor.load_data()
    
    # 准备图数据
    train_graphs, test_graphs = predictor.prepare_graphs()
    
    print("✅ 基本使用示例完成")

def example_2_single_target_training():
    """示例2: 单目标训练"""
    print("\n" + "="*60)
    print("示例2: 单目标训练")
    print("="*60)
    
    # 创建预测器
    predictor = PolymerPropertyPredictor()
    
    # 加载数据
    predictor.load_data()
    
    # 训练单个目标（Tg）
    results = predictor.evaluate_single_target('Tg', test_size=0.2)
    
    if results:
        print(f"\n📊 Tg 模型性能:")
        print(f"  MSE: {results['mse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  R²:  {results['r2']:.4f}")
    
    print("✅ 单目标训练示例完成")

def example_3_prediction():
    """示例3: 预测新SMILES"""
    print("\n" + "="*60)
    print("示例3: 预测新SMILES")
    print("="*60)
    
    # 创建预测器
    predictor = PolymerPropertyPredictor()
    
    # 加载数据
    predictor.load_data()
    
    # 准备图数据
    predictor.prepare_graphs()
    
    # 训练Tg模型
    predictor.evaluate_single_target('Tg', test_size=0.2)
    
    # 新SMILES列表
    new_smiles = [
        "CCO",  # 乙醇
        "c1ccccc1",  # 苯
        "CC(C)(C)c1ccc(N)cc1",  # 对叔丁基苯胺
        "C1=CC=C(C=C1)C(=O)O",  # 苯甲酸
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # 布洛芬
    ]
    
    # 预测Tg
    predictions = predictor.predict(new_smiles, 'Tg')
    
    if predictions is not None:
        print(f"\n🔮 Tg 预测结果:")
        for i, (smiles, pred) in enumerate(zip(new_smiles, predictions)):
            print(f"  {i+1}. {smiles}: {pred:.2f}°C")
    
    print("✅ 预测示例完成")

def example_4_custom_config():
    """示例4: 自定义配置"""
    print("\n" + "="*60)
    print("示例4: 自定义配置")
    print("="*60)
    
    # 自定义配置
    custom_config = {
        'hidden_channels': 512,  # 更大的隐藏层
        'num_layers': 6,         # 更多层
        'num_heads': 16,         # 更多注意力头
        'dropout': 0.2,          # 更高的dropout
        'pretrain_lr': 0.0005,   # 更小的学习率
        'finetune_lr': 0.0001,
        'pretrain_epochs': 150,
        'finetune_epochs': 300
    }
    
    # 创建预测器
    predictor = PolymerPropertyPredictor(custom_config)
    
    # 获取模型摘要
    predictor.get_model_summary()
    
    print("✅ 自定义配置示例完成")

def example_5_full_training():
    """示例5: 完整训练流程"""
    print("\n" + "="*60)
    print("示例5: 完整训练流程")
    print("="*60)
    
    # 注意：这会运行完整的训练流程，可能需要较长时间
    print("⚠️  注意：完整训练流程可能需要较长时间...")
    
    # 运行完整训练
    run_full_training()
    
    print("✅ 完整训练流程示例完成")

def example_6_quick_demo():
    """示例6: 快速演示"""
    print("\n" + "="*60)
    print("示例6: 快速演示")
    print("="*60)
    
    # 运行快速演示
    run_quick_demo()
    
    print("✅ 快速演示完成")

def example_7_batch_prediction():
    """示例7: 批量预测"""
    print("\n" + "="*60)
    print("示例7: 批量预测")
    print("="*60)
    
    # 创建预测器
    predictor = PolymerPropertyPredictor()
    
    # 加载数据
    predictor.load_data()
    
    # 准备图数据
    predictor.prepare_graphs()
    
    # 训练所有目标
    results = predictor.train()
    
    # 批量SMILES
    batch_smiles = [
        "CCO", "c1ccccc1", "CC(C)(C)c1ccc(N)cc1",
        "C1=CC=C(C=C1)C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "C1=CC=C(C=C1)C2=CC=CC=C2", "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O"
    ]
    
    # 预测所有属性
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print(f"\n🔮 批量预测结果:")
    print(f"{'SMILES':<30} {'Tg':<8} {'FFV':<8} {'Tc':<8} {'Density':<8} {'Rg':<8}")
    print("-" * 80)
    
    for smiles in batch_smiles:
        row = [smiles[:29]]
        for target in targets:
            pred = predictor.predict([smiles], target)
            if pred is not None:
                row.append(f"{pred[0]:.2f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<30} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8}")
    
    print("✅ 批量预测示例完成")

if __name__ == "__main__":
    print("🎯 聚合物性质预测系统使用示例")
    print("="*60)
    
    # 运行示例
    example_1_basic_usage()
    example_2_single_target_training()
    example_3_prediction()
    example_4_custom_config()
    
    # 注释掉完整训练示例，因为需要较长时间
    # example_5_full_training()
    
    example_6_quick_demo()
    example_7_batch_prediction()
    
    print("\n" + "="*60)
    print("🎉 所有示例运行完成！")
    print("="*60) 