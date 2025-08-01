# 快速开始指南

## 在Kaggle Notebook中使用

### 1. 基本使用（推荐）

```python
# 导入函数
from data_preparation import get_data_paths, load_and_split_data
from train_stage2 import main as stage2_main
from train_stage3 import main as stage3_main
from predict import main as predict_main

# 设置路径（可选，如果不设置会使用默认Kaggle路径）
import os
os.environ['NEURIPS_DATA_PATH'] = '/kaggle/input/neurips-open-polymer-prediction-2025'
os.environ['EXTRA_DATA_BASE'] = '/kaggle/input/smiles-extra-data'
os.environ['TC_DATA_BASE'] = '/kaggle/input/tc-smiles'

# 运行完整pipeline
stage2_main()  # Stage2训练
stage3_main()  # Stage3训练
predictions = predict_main()  # 预测
```

### 2. 使用自定义路径

```python
# 定义路径字典
custom_paths = {
    'train_csv': '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv',
    'test_csv': '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv',
    'tc_data': '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    'tg_jcim_data': '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    'tg_excel_data': '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    'density_data': '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
    'ffv_data': '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv',
}

# 使用自定义路径运行
stage2_main(custom_paths)
stage3_main(custom_paths)
predictions = predict_main(custom_paths)
```

### 3. 分阶段使用

```python
# 只运行特定阶段
from train_stage2 import train_stage2
from train_stage3 import main as stage3_main
from predict import main as predict_main

train_stage2()  # 只运行Stage2
stage3_main()  # 只运行Stage3
predictions = predict_main()  # 只运行预测

# 自定义输出路径
predictions = predict_main(output_path="my_predictions.csv")
```

### 4. 自定义Stage2训练

```python
# 使用解构后的函数进行更灵活的训练
from train_stage2 import (
    prepare_stage2_data, load_stage1_models, create_stage2_model,
    train_stage2_epoch, save_stage2_models
)

# 分步骤执行
train_df, base_graph_ds, graph_loader = prepare_stage2_data()
encoder, best_params = load_stage1_models()
model, optimizer = create_stage2_model(encoder, best_params)

# 自定义训练循环
for epoch in range(1, 11):
    loss = train_stage2_epoch(model, optimizer, graph_loader)
    print(f"Epoch {epoch}: loss={loss:.4f}")

save_stage2_models(encoder, model)
```

### 5. 检查数据文件

```python
# 检查数据文件是否存在
paths = get_data_paths()
for key, path in paths.items():
    print(f"{key}: {os.path.exists(path)}")
```

## 关键特性

- ✅ **无相对路径依赖**: 所有路径都使用绝对路径
- ✅ **Kaggle环境兼容**: 默认配置适用于Kaggle环境
- ✅ **灵活配置**: 支持环境变量和参数传入
- ✅ **向后兼容**: 不传参数时使用默认路径
- ✅ **完整pipeline**: 从训练到预测的完整流程

## 注意事项

1. 确保所有数据文件都在正确的位置
2. 如果GPU内存不足，可以调整batch_size
3. 训练完成后会生成模型文件，预测时需要用到
4. 在Kaggle环境中，默认路径会自动配置 