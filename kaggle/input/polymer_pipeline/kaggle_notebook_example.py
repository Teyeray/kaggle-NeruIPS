# Kaggle Notebook 使用示例
# 这个文件展示了如何在Kaggle notebook中使用polymer_pipeline

# ============================================================================
# 1. 导入必要的库和函数
# ============================================================================

import os
import pandas as pd
import numpy as np

# 导入pipeline函数
from data_preparation import get_data_paths, load_and_split_data
from train_stage1 import setup_stage1_data, objective_stage1
from train_stage2 import (
    train_stage2, prepare_stage2_data, load_stage1_models, 
    create_stage2_model, train_stage2_epoch, save_stage2_models
)
from train_stage3 import main as stage3_main
from predict import main as predict_main

# ============================================================================
# 2. 设置数据路径（可选）
# ============================================================================

# 方法1: 使用环境变量（推荐）
os.environ['NEURIPS_DATA_PATH'] = '/kaggle/input/neurips-open-polymer-prediction-2025'
os.environ['EXTRA_DATA_BASE'] = '/kaggle/input/smiles-extra-data'
os.environ['TC_DATA_BASE'] = '/kaggle/input/tc-smiles'

# 方法2: 直接定义路径字典
custom_paths = {
    'train_csv': '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv',
    'test_csv': '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv',
    'tc_data': '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    'tg_jcim_data': '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    'tg_excel_data': '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    'density_data': '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
    'ffv_data': '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv',
    'supplement_dir': '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement',
    'dataset1': '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv',
    'dataset2': '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv',
    'dataset3': '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv',
}

# ============================================================================
# 3. 检查数据文件是否存在
# ============================================================================

def check_data_files(paths):
    """检查数据文件是否存在"""
    print("检查数据文件...")
    for key, path in paths.items():
        if os.path.exists(path):
            print(f"✅ {key}: {path}")
        else:
            print(f"❌ {key}: {path} (文件不存在)")
    print()

# 使用默认路径检查
default_paths = get_data_paths()
check_data_files(default_paths)

# 或者使用自定义路径检查
# check_data_files(custom_paths)

# ============================================================================
# 4. 数据加载和预处理
# ============================================================================

print("=" * 60)
print("数据加载和预处理")
print("=" * 60)

# 加载数据（使用默认路径）
train_df, val_df, test_df = load_and_split_data()

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(val_df)}")
print(f"测试集大小: {len(test_df)}")

# 或者使用自定义路径
# train_df, val_df, test_df = load_and_split_data(custom_paths)

# ============================================================================
# 5. Stage1: 节点/边级自监督学习
# ============================================================================

print("\n" + "=" * 60)
print("Stage1: 节点/边级自监督学习")
print("=" * 60)

# 设置Stage1数据
train_df, base_graph_ds, dataset_nodeedge, nodeedge_loader = setup_stage1_data()

# 运行Optuna优化（可选，如果已经有最佳参数可以跳过）
# import optuna
# study = optuna.create_study(direction="minimize")
# study.optimize(objective_stage1, n_trials=5)

print("Stage1数据准备完成")

# ============================================================================
# 6. Stage2: 图级自监督学习
# ============================================================================

print("\n" + "=" * 60)
print("Stage2: 图级自监督学习")
print("=" * 60)

# 运行Stage2训练
# 方法1: 使用完整函数
stage2_result = train_stage2()  # 使用默认路径
# 或者使用自定义路径: stage2_result = train_stage2(custom_paths)

# 方法2: 分步骤执行（更灵活）
# train_df, base_graph_ds, graph_loader = prepare_stage2_data()
# encoder, best_params = load_stage1_models()
# model, optimizer = create_stage2_model(encoder, best_params)
# for epoch in range(1, 21):
#     loss = train_stage2_epoch(model, optimizer, graph_loader)
#     print(f"Epoch {epoch}: loss={loss:.4f}")
# save_stage2_models(encoder, model)

# ============================================================================
# 7. Stage3: 下游任务微调
# ============================================================================

print("\n" + "=" * 60)
print("Stage3: 下游任务微调")
print("=" * 60)

# 运行Stage3训练
stage3_main()  # 使用默认路径
# 或者使用自定义路径: stage3_main(custom_paths)

# ============================================================================
# 8. 预测
# ============================================================================

print("\n" + "=" * 60)
print("预测阶段")
print("=" * 60)

# 运行预测
predictions = predict_main()  # 使用默认路径
# 或者使用自定义路径: predictions = predict_main(custom_paths)

# 查看预测结果
print("\n预测结果预览:")
print(predictions.head())

# ============================================================================
# 9. 高级用法示例
# ============================================================================

print("\n" + "=" * 60)
print("高级用法示例")
print("=" * 60)

# 示例1: 只训练特定属性
def train_specific_property(property_name, paths=None):
    """只训练特定属性"""
    from train_stage3 import prepare_property_datasets, finetune_property
    
    if paths is None:
        paths = get_data_paths()
    
    # 准备数据集
    property_datasets = prepare_property_datasets([property_name], paths)
    datasets = property_datasets[property_name]
    
    # 训练
    finetune_property(
        train_df=datasets['train'],
        val_df=datasets['val'],
        property_name=property_name
    )

# 示例2: 自定义预测
def custom_predict(smiles_list, paths=None):
    """对自定义SMILES列表进行预测"""
    from predict import load_trained_models, predict_single_smiles
    
    if paths is None:
        paths = get_data_paths()
    
    # 加载模型
    properties = ['Tg', 'Tc', 'Density', 'FFV', 'Rg']
    models = load_trained_models(properties)
    
    # 预测
    results = []
    for smiles in smiles_list:
        pred = predict_single_smiles(smiles, models)
        results.append(pred)
    
    return pd.DataFrame(results)

# 示例3: 批量预测自定义数据
def predict_custom_data(custom_csv_path, paths=None, output_path="custom_predictions.csv"):
    """对自定义CSV文件进行预测"""
    from predict import load_trained_models, predict_test_set
    
    if paths is None:
        paths = get_data_paths()
    
    # 加载模型
    properties = ['Tg', 'Tc', 'Density', 'FFV', 'Rg']
    models = load_trained_models(properties)
    
    # 预测
    predictions = predict_test_set(custom_csv_path, models)
    
    # 保存结果
    predictions.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")
    
    return predictions

# 示例4: 自定义Stage2训练流程
def custom_stage2_training(paths=None, n_epochs=15, batch_size=32, lr=1e-4):
    """自定义Stage2训练流程"""
    
    # 1. 准备数据
    train_df, base_graph_ds, graph_loader = prepare_stage2_data(paths, batch_size)
    print(f"数据准备完成，样本数: {len(train_df)}")
    
    # 2. 加载Stage1模型
    encoder, best_params = load_stage1_models()
    print("Stage1模型加载完成")
    
    # 3. 创建Stage2模型（使用自定义学习率）
    model, optimizer = create_stage2_model(encoder, best_params)
    # 修改学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Stage2模型创建完成，学习率: {lr}")
    
    # 4. 自定义训练循环
    print(f"开始训练，共{n_epochs}轮...")
    losses = []
    for epoch in range(1, n_epochs + 1):
        loss = train_stage2_epoch(model, optimizer, graph_loader)
        losses.append(loss)
        print(f"Epoch {epoch:02d}/{n_epochs}  loss={loss:.4f}")
        
        # 可以在这里添加早停逻辑
        if len(losses) > 5 and losses[-1] > losses[-5]:
            print("损失上升，提前停止训练")
            break
    
    # 5. 保存模型
    save_stage2_models(encoder, model)
    
    return encoder, model, losses

# ============================================================================
# 10. 使用示例
# ============================================================================

# 示例1: 只训练Tg属性
# train_specific_property('Tg')

# 示例2: 对单个SMILES进行预测
# custom_smiles = ['CCO', 'CCCC', 'c1ccccc1']
# results = custom_predict(custom_smiles)
# print(results)

# 示例3: 对自定义数据进行预测
# custom_predictions = predict_custom_data('/path/to/custom_data.csv')

# 示例4: 自定义Stage2训练
# encoder, model, losses = custom_stage2_training(n_epochs=10, batch_size=32, lr=1e-4)

print("\n🎉 所有示例代码已准备就绪！")
print("您可以根据需要取消注释相应的代码块来运行。") 