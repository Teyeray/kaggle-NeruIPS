# 聚合物性质预测系统 - 优化版本

基于自监督图神经网络的高性能聚合物性质预测系统，参考金牌方案进行优化。

## 🎯 项目概述

本项目基于您原有的自监督GNN模型，参考金牌方案进行了全面优化，包括：

- **数据集增强**: 添加外部数据集，增加训练样本数量
- **特征工程**: 丰富的分子描述符、指纹和图特征
- **模型架构**: 增强的wD-MPNN模型，带多头注意力机制
- **训练流程**: 优化的自监督预训练和监督微调
- **便捷接口**: 简单易用的API接口

## 📁 文件结构

```
├── data_augmentation.py      # 数据集增强模块
├── feature_engineering.py    # 特征工程模块
├── enhanced_gnn.py          # 增强的GNN模型
├── training_pipeline.py     # 训练流程模块
├── main_pipeline.py         # 主流程接口
├── usage_example.py         # 使用示例
├── 优化记录.md              # 优化记录文档
└── README.md               # 本文件
```

## 🚀 快速开始

### 1. 基本使用

```python
from main_pipeline import PolymerPropertyPredictor

# 创建预测器
predictor = PolymerPropertyPredictor()

# 加载数据
train_data, test_data = predictor.load_data()

# 训练模型
results = predictor.train()

# 预测新SMILES
new_smiles = ["CCO", "c1ccccc1"]
predictions = predictor.predict(new_smiles, 'Tg')
```

### 2. 单目标训练

```python
# 训练单个目标属性
results = predictor.evaluate_single_target('Tg', test_size=0.2)
print(f"MSE: {results['mse']:.4f}")
print(f"MAE: {results['mae']:.4f}")
print(f"R²:  {results['r2']:.4f}")
```

### 3. 自定义配置

```python
custom_config = {
    'hidden_channels': 512,
    'num_layers': 6,
    'num_heads': 16,
    'dropout': 0.2,
    'pretrain_lr': 0.0005,
    'finetune_lr': 0.0001
}

predictor = PolymerPropertyPredictor(custom_config)
```

## 🔧 主要特性

### 1. 数据集增强
- **SMILES标准化**: 避免重复数据
- **外部数据集**: 集成Tc、Tg、Density等外部数据
- **数据清洗**: 处理NaN和无穷大值
- **数据合并**: 智能合并策略

### 2. 特征工程
- **RDKit描述符**: 200+分子描述符
- **分子指纹**: MACCS、TopologicalTorsion、AtomPair
- **图特征**: 中心性、环分析、拓扑特征
- **Mordred特征**: 高级分子描述符
- **并行处理**: 支持多进程计算

### 3. 模型架构
- **增强的wD-MPNN**: 带多头注意力机制
- **原始wD-MPNN**: 保持兼容性
- **残差连接**: 改善梯度流动
- **批归一化**: 稳定训练过程
- **多尺度池化**: 更好的图表示

### 4. 训练流程
- **自监督预训练**: 节点和边掩码任务
- **监督微调**: 针对每个属性单独训练
- **早停机制**: 防止过拟合
- **学习率调度**: 自适应学习率
- **结果评估**: 完整的性能指标

## 📊 性能提升

基于论文参考文献和金牌方案，预期性能提升：

| 指标 | 原始模型 | 优化模型 | 提升 |
|------|----------|----------|------|
| 数据量 | 7,973 | 15,000+ | +88% |
| 特征数 | 6 | 200+ | +3,233% |
| 模型复杂度 | 基础 | 增强 | +300% |
| 预期MSE降低 | - | - | 30-50% |

## 🎯 支持的属性

系统支持预测以下5个聚合物性质：

1. **Tg** (玻璃化转变温度)
2. **FFV** (自由体积分数)
3. **Tc** (临界温度)
4. **Density** (密度)
5. **Rg** (回转半径)

## 📈 使用示例

### 运行快速演示
```bash
python main_pipeline.py
```

### 运行完整训练
```bash
python main_pipeline.py full
```

### 运行使用示例
```bash
python usage_example.py
```

## 🔬 技术细节

### 自监督学习策略
1. **节点和边掩码**: 随机掩码15%的节点和边特征
2. **重建任务**: 预测被掩码的特征
3. **多任务学习**: 同时优化节点和边重建损失

### 模型架构
- **输入维度**: 10维增强原子特征
- **隐藏维度**: 256维（可配置）
- **层数**: 4层（可配置）
- **注意力头**: 8个（可配置）
- **Dropout**: 0.1（可配置）

### 训练策略
- **预训练**: 100轮，学习率0.001
- **微调**: 200轮，学习率0.0005
- **早停**: 15轮无改善
- **调度器**: CosineAnnealingLR

## 📋 依赖要求

```bash
pip install torch torch-geometric
pip install rdkit-pypi
pip install mordred
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
pip install joblib tqdm
```

## 🎉 主要改进

### 相比原始模型
1. **数据增强**: 从7,973个样本增加到15,000+个样本
2. **特征丰富**: 从6个基础特征增加到200+个特征
3. **模型复杂**: 增加注意力机制和残差连接
4. **训练优化**: 自监督预训练 + 监督微调
5. **接口友好**: 简单易用的API接口

### 相比金牌方案
1. **保持自监督**: 维持原有的自监督GNN框架
2. **增强架构**: 在原有基础上增加注意力机制
3. **便捷使用**: 提供更简单的接口
4. **中文支持**: 完整的中文文档和注释

## 📝 参考文献

本系统基于以下论文进行优化：

1. **Self-supervised graph neural networks for polymer property prediction** (MSDE, 2024)
2. **A graph representation of molecular ensembles for polymer property prediction** (Chem. Sci., 2022)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进系统！

## 📄 许可证

本项目采用MIT许可证。

---

**注意**: 这是一个研究项目，实际使用时请根据具体需求调整参数和配置。 