import os
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

# -----------------------------
# 数据加载与划分
# -----------------------------
import random
import torch
from torch_geometric.data import InMemoryDataset, Data

class NodeEdgeMaskDataset(InMemoryDataset):
    """
    Node/Edge-level SSL 数据集：每个样本随机 mask 2 个节点和 2 条边，
    输出 (x_masked, x_orig, edge_attr_masked, edge_attr_orig, edge_index, batch)
    对应 Gao et al., MSDE 2024 节点/边级伪任务§2.3.1。
    """
    def __init__(self, base_dataset, device=None):
        super().__init__(None, None)
        self.base = base_dataset
        self.device = device or torch.device('cpu')

    def len(self):
        return len(self.base)

    def get(self, idx):
        data = self.base.get(idx)  # Data(x, edge_index, edge_attr, [y])
        # 原始特征
        x_orig = data.x.clone().to(self.device)
        e_orig = data.edge_attr.clone().to(self.device)

        # 1) 节点 Mask：随机挑 2 个不同节点
        x_masked = x_orig.clone()
        N = x_masked.size(0)
        if N >= 2:
            nodes = random.sample(range(N), 2)
            x_masked[nodes] = 0.0

        # 2) 边 Mask：随机挑 2 条不同边
        e_masked = e_orig.clone()
        E = e_masked.size(0)
        if E >= 2:
            edges = random.sample(range(E), 2)
            e_masked[edges] = 0.0

        # 3) 返回一个新的 Data 对象
        return Data(
            x_masked=x_masked,
            x_orig=x_orig,
            edge_index=data.edge_index,
            edge_attr_masked=e_masked,
            edge_attr_orig=e_orig,
            batch=getattr(data, 'batch', None)
        )


def make_smile_canonical(smile):
    """将 SMILES 转换为标准形式，避免重复"""
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return np.nan


def add_extra_data(df_train, df_extra, target):
    """
    将外部数据 df_extra 根据 target 列添加到 df_train
    1. 标准化 SMILES
    2. 根据 SMILES 聚合外部数据
    3. 填充训练集中缺失的样本
    4. 追加外部独有样本
    """
    print(f"  → 正在增强 {target} 数据，共 {len(df_extra)} 条")
    df_train = df_train.copy()
    df_extra = df_extra.copy()
    df_extra['SMILES'] = df_extra['SMILES'].apply(make_smile_canonical)
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    print(f'cross_smiles: {len(cross_smiles)}')
    existing = set(df_train[df_train[target].notnull()]['SMILES'])
    cross_smiles -= existing

    for smi in cross_smiles:
        val = df_extra.loc[df_extra['SMILES'] == smi, target].values[0]
        df_train.loc[df_train['SMILES'] == smi, target] = val

    unique_extra = df_extra[~df_extra['SMILES'].isin(df_train['SMILES'])]
    print(f"    填充已有样本 {len(cross_smiles)} 条，新增样本 {len(unique_extra)} 条")
    df_train = pd.concat([df_train, unique_extra], ignore_index=True)
    return df_train


def load_and_split_data(
    base_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    1. 加载主训练集 train.csv
    2. 标准化 SMILES
    3. 分别加载 Tc, Tg, Density 外部数据并增强
    4. 划分 train/val/test，保留缺失值
    """
    print("👉 加载主训练数据")
    train = pd.read_csv(os.path.join(base_path, 'train.csv'))
    train['SMILES'] = train['SMILES'].apply(make_smile_canonical)
    print(f"  原始训练样本数: {len(train)}")

    # 外部数据目录
    extra_dir = os.path.join(base_path, 'smiles-extra-data')

    # 2. 增强 Tc
    tc_path = os.path.join(base_path, 'Tc_SMILES.csv')
    if os.path.exists(tc_path):
        df_tc = pd.read_csv(tc_path).rename(columns={'TC_mean': 'Tc'})
        train = add_extra_data(train, df_tc, 'Tc')
    else:
        print("  ⚠️ 未找到 Tc 外部数据")

    # 3. 增强 Tg 来源1
    tg1_path = os.path.join(extra_dir, 'JCIM_sup_bigsmiles.csv')
    if os.path.exists(tg1_path):
        df_tg1 = pd.read_csv(tg1_path, usecols=['SMILES', 'Tg (C)']).rename(columns={'Tg (C)': 'Tg'})
        train = add_extra_data(train, df_tg1, 'Tg')
    else:
        print("  ⚠️ 未找到 Tg 来源1 数据")

    # 4. 增强 Tg 来源2
    tg2_path = os.path.join(extra_dir, 'data_tg3.xlsx')
    if os.path.exists(tg2_path):
        df_tg2 = pd.read_excel(tg2_path).rename(columns={'Tg [K]': 'Tg'})
        df_tg2['Tg'] = df_tg2['Tg'] - 273.15
        train = add_extra_data(train, df_tg2, 'Tg')
    else:
        print("  ⚠️ 未找到 Tg 来源2 数据")

    # 5. 增强 Density
    d_path = os.path.join(extra_dir, 'data_dnst1.xlsx')
    if os.path.exists(d_path):
        df_den = pd.read_excel(d_path)
        df_den = df_den.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
        df_den['Density'] = pd.to_numeric(df_den['Density'], errors='coerce') - 0.118
        train = add_extra_data(train, df_den, 'Density')
    else:
        print("  ⚠️ 未找到 Density 外部数据")

    # 6. 划分数据集
    print("👉 划分 train / validation / test")
    train_df, temp_df = train_test_split(train, test_size=test_size, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)
    print(f"  划分结果: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------
# PyG 数据集定义
# -----------------------------

def create_graph_from_smiles(smiles):
    """
    将 SMILES 转为 PyG Data 输入格式
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    x = [[atom.GetAtomicNum(), int(atom.GetIsAromatic())] for atom in mol.GetAtoms()]
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        edge_attr += [[bond.GetBondTypeAsDouble()]] * 2
    return (
        torch.tensor(x, dtype=torch.float),
        torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_attr, dtype=torch.float)
    )

class PolymerDataset(InMemoryDataset):
    """基于 DataFrame 的 PyG InMemoryDataset"""
    def __init__(self, df: pd.DataFrame, y_cols: list, transform=None):
        super().__init__(None, transform)
        print(f"📦 构建 PolymerDataset，样本数={len(df)}")
        self.data_list = []
        for idx, row in df.iterrows():
            x, ei, ea = create_graph_from_smiles(row['SMILES'])
            if x is None: continue
            y = torch.tensor([row[c] for c in y_cols], dtype=torch.float)
            data = Data(x=x, edge_index=ei, edge_attr=ea, y=y)
            self.data_list.append(data)
        print(f"   成功转换为图数据: {len(self.data_list)} 条")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# -----------------------------
# 构造数据集 & DataLoader
# -----------------------------

if __name__ == "__main__":
    base_path = "neurips-open-polymer-prediction-2025"
    train_df, val_df, test_df = load_and_split_data(base_path)

    targets = ['Tg', 'Tc', 'Density']
    train_dataset = PolymerDataset(train_df, y_cols=targets)
    val_dataset   = PolymerDataset(val_df,   y_cols=targets)
    test_dataset  = PolymerDataset(test_df,  y_cols=targets)

    print("👉 构建 DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    print(f"🔢 样本总计 train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
