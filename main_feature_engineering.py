#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主特征工程脚本
参照金牌pipeline实现完整的特征工程流程
"""

import os
import pickle
import gc
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# 导入自定义模块
from data_augmentation import load_and_augment_train_data, load_test_data, clean_data
from feature_engineering import (
    process_molecular_features_parallel, 
    clean_features_and_compute_means,
    apply_imputation_to_test,
    validate_features,
    USELESS_COLS
)

# 忽略警告
warnings.filterwarnings("ignore")

class FeatureEngineeringConfig:
    """特征工程配置类"""
    def __init__(self):
        # 路径配置
        self.BASE_PATH = 'neurips-open-polymer-prediction-2025/'
        self.OUTPUT_PATH = 'feature_engineering_output/'
        
        # 文件配置
        self.TRAIN_FEATURES_FILE = 'train_features_complete.pkl'
        self.TEST_FEATURES_FILE = 'test_features_complete.pkl'
        self.TRAIN_MEANS_FILE = 'feature_means.pkl'
        self.FEATURE_METADATA_FILE = 'feature_metadata.pkl'
        
        # 处理参数
        self.N_JOBS = 4  # 并行处理进程数
        self.SEED = 42
        
        # 目标变量
        self.TARGETS = ['Tg', 'Tc', 'Rg', 'FFV', 'Density']
        
        # 创建输出目录
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)

def prepare_train_features(cfg):
    """
    准备训练特征的主函数
    参照金牌pipeline的完整流程
    """
    print("🚀 开始训练特征准备")
    print("="*70)
    
    # 步骤1: 加载和增强训练数据
    print("\n📂 步骤1: 加载和增强训练数据...")
    train = load_and_augment_train_data(cfg.BASE_PATH)
    print(f"✅ 训练数据加载完成，形状: {train.shape}")
    
    # 步骤2: 生成分子特征
    print("\n🧬 步骤2: 生成分子特征...")
    print("🔄 开始计算分子描述符、图特征、指纹特征...")
    
    molecular_features = process_molecular_features_parallel(
        train, 
        USELESS_COLS, 
        n_jobs=cfg.N_JOBS
    )
    
    if not molecular_features.empty:
        print(f"✅ 分子特征计算完成，形状: {molecular_features.shape}")
        print(f"📊 特征类型统计:")
        print(f"   - RDKit描述符: {len([col for col in molecular_features.columns if col not in ['graph_diameter', 'avg_shortest_path', 'num_cycles', 'betweenness_mean', 'betweenness_std', 'eigenvector_mean', 'ring_4', 'max_degree', 'closeness_mean', 'katz_centrality_std', 'heteroatom_ratio', 'AMW', 'TIC2', 'naRing', 'MPC3', 'MACCS_Key130', 'MACCS_Key142', 'MACCS_Key066', 'MACCS_Key153', 'TopologicalTorsion_Bit0512', 'TopologicalTorsion_Bit1296', 'AtomPair_B512_Bit0138', 'AtomPair_B512_Bit0448', 'AtomPair_B512_Bit0408']])}")
        print(f"   - 图拓扑特征: 11个")
        print(f"   - Mordred特征: 4个")
        print(f"   - MACCS指纹: 4个")
        print(f"   - TopologicalTorsion指纹: 2个")
        print(f"   - AtomPair指纹: 3个")
    else:
        print("⚠️ 没有生成分子特征，使用基础特征")
    
    # 步骤3: 合并所有特征
    print("\n🔗 步骤3: 合并所有特征...")
    if not molecular_features.empty:
        train_complete = pd.concat([train, molecular_features], axis=1)
    else:
        train_complete = train
        molecular_features = pd.DataFrame()
    
    print(f"✅ 特征合并完成，最终形状: {train_complete.shape}")
    
    # 步骤4: 识别特征列
    print("\n📋 步骤4: 识别特征列...")
    molecular_feature_names = molecular_features.columns.tolist() if not molecular_features.empty else []
    all_computed_features = molecular_feature_names
    
    print(f"📊 特征统计:")
    print(f"   - 分子特征: {len(molecular_feature_names)}个")
    print(f"   - 总计算特征: {len(all_computed_features)}个")
    
    # 步骤5: 清理特征并计算均值
    print("\n🧹 步骤5: 清理特征并计算均值...")
    train_complete, feature_means = clean_features_and_compute_means(
        train_complete, 
        all_computed_features
    )
    
    # 步骤6: 生成元数据
    print("\n📝 步骤6: 生成元数据...")
    feature_metadata = {
        'molecular_features': molecular_feature_names,
        'all_computed_features': all_computed_features,
        'original_columns': ['id', 'SMILES'] + cfg.TARGETS,
        'useless_cols_excluded': USELESS_COLS,
        'generation_config': {
            'n_jobs': cfg.N_JOBS,
            'seed': cfg.SEED
        }
    }
    
    # 步骤7: 保存所有文件
    print("\n💾 步骤7: 保存训练文件...")
    
    # 保存完整训练特征
    train_features_path = os.path.join(cfg.OUTPUT_PATH, cfg.TRAIN_FEATURES_FILE)
    train_complete.to_pickle(train_features_path)
    print(f"✅ 训练特征已保存: {train_features_path}")
    
    # 保存特征均值用于测试填充
    means_path = os.path.join(cfg.OUTPUT_PATH, cfg.TRAIN_MEANS_FILE)
    with open(means_path, 'wb') as f:
        pickle.dump(feature_means, f)
    print(f"✅ 特征均值已保存: {means_path}")
    
    # 保存元数据
    metadata_path = os.path.join(cfg.OUTPUT_PATH, cfg.FEATURE_METADATA_FILE)
    with open(metadata_path, 'wb') as f:
        pickle.dump(feature_metadata, f)
    print(f"✅ 元数据已保存: {metadata_path}")
    
    # 最终报告
    print(f"\n✨ 训练特征准备完成! ✨")
    print(f"📊 生成了{len(all_computed_features)}个特征:")
    print(f"   • 分子特征: {len(molecular_feature_names)}个")
    print(f"📁 文件保存到: {cfg.OUTPUT_PATH}")
    
    # 清理内存
    gc.collect()
    
    return train_complete, feature_means, feature_metadata

def prepare_test_features(cfg):
    """
    准备测试特征的主函数
    使用保存的训练信息
    """
    print("🚀 开始测试特征准备")
    print("="*70)
    
    # 步骤1: 加载训练准备文件
    print("📂 步骤1: 加载训练准备文件...")
    
    # 加载特征均值
    means_path = os.path.join(cfg.OUTPUT_PATH, cfg.TRAIN_MEANS_FILE)
    try:
        with open(means_path, 'rb') as f:
            feature_means = pickle.load(f)
        print(f"✅ 已加载特征均值: {len(feature_means)}个特征")
    except FileNotFoundError:
        raise FileNotFoundError(f"训练均值文件未找到: {means_path}. 请先运行prepare_train_features()")
    
    # 加载元数据
    metadata_path = os.path.join(cfg.OUTPUT_PATH, cfg.FEATURE_METADATA_FILE)
    try:
        with open(metadata_path, 'rb') as f:
            feature_metadata = pickle.load(f)
        print(f"✅ 已加载特征元数据")
    except FileNotFoundError:
        raise FileNotFoundError(f"元数据文件未找到: {metadata_path}. 请先运行prepare_train_features()")
    
    # 步骤2: 加载测试数据
    print("\n📂 步骤2: 加载测试数据...")
    test = load_test_data(cfg.BASE_PATH)
    print(f"✅ 测试数据加载完成，形状: {test.shape}")
    
    # 步骤3: 生成分子特征
    print("\n🧬 步骤3: 生成分子特征...")
    molecular_features = process_molecular_features_parallel(
        test, 
        USELESS_COLS, 
        n_jobs=cfg.N_JOBS
    )
    
    if not molecular_features.empty:
        print(f"✅ 测试分子特征计算完成，形状: {molecular_features.shape}")
    else:
        print("⚠️ 没有生成测试分子特征")
    
    # 步骤4: 合并所有特征
    print("\n🔗 步骤4: 合并所有特征...")
    if not molecular_features.empty:
        test_complete = pd.concat([test, molecular_features], axis=1)
    else:
        test_complete = test
        molecular_features = pd.DataFrame()
    
    print(f"✅ 测试特征合并完成，形状: {test_complete.shape}")
    
    # 步骤5: 应用清理和填充
    print("\n🧹 步骤5: 应用清理和填充...")
    all_computed_features = feature_metadata['all_computed_features']
    test_complete = apply_imputation_to_test(
        test_complete, 
        all_computed_features, 
        feature_means
    )
    
    # 步骤6: 验证特征
    print("\n✅ 步骤6: 验证特征...")
    validate_features(test_complete, all_computed_features, phase='test')
    
    # 步骤7: 保存测试特征
    print("\n💾 步骤7: 保存测试特征...")
    test_features_path = os.path.join(cfg.OUTPUT_PATH, cfg.TEST_FEATURES_FILE)
    test_complete.to_pickle(test_features_path)
    print(f"✅ 测试特征已保存: {test_features_path}")
    
    # 最终报告
    print(f"\n✨ 测试特征准备完成! ✨")
    print(f"📊 应用了{len(all_computed_features)}个特征，使用训练均值填充")
    
    # 清理内存
    gc.collect()
    
    return test_complete

def main():
    """主函数"""
    print("🎯 聚合物性质预测 - 特征工程Pipeline")
    print("="*70)
    
    # 初始化配置
    cfg = FeatureEngineeringConfig()
    
    try:
        # 准备训练特征
        print("\n" + "="*50)
        print("🏋️ 准备训练特征")
        print("="*50)
        train_complete, feature_means, feature_metadata = prepare_train_features(cfg)
        
        # 准备测试特征
        print("\n" + "="*50)
        print("🧪 准备测试特征")
        print("="*50)
        test_complete = prepare_test_features(cfg)
        
        # 最终总结
        print("\n" + "="*70)
        print("🎉 特征工程Pipeline完成!")
        print("="*70)
        print(f"📊 训练数据: {train_complete.shape}")
        print(f"📊 测试数据: {test_complete.shape}")
        print(f"🔧 特征数量: {len(feature_metadata['all_computed_features'])}")
        print(f"📁 输出目录: {cfg.OUTPUT_PATH}")
        
        # 显示目标变量统计
        print(f"\n📈 目标变量统计:")
        for target in cfg.TARGETS:
            if target in train_complete.columns:
                train_count = train_complete[target].notnull().sum()
                print(f"   {target}: {train_count}个训练样本")
        
    except Exception as e:
        print(f"❌ 特征工程过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 