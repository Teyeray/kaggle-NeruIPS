import os
import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

# -----------------------------
# 数据加载与划分
# -----------------------------
import random
import torch
from torch_geometric.data import InMemoryDataset, Data


# -----------------------------
# 环境变量配置
# -----------------------------
def get_data_paths():
    """
    从环境变量获取数据路径，提供默认值
    """
    # 主数据目录
    BASE_PATH = os.getenv('NEURIPS_DATA_PATH', '/kaggle/input/neurips-open-polymer-prediction-2025')
    EXTRA_BASE = os.getenv('EXTRA_DATA_BASE', '/kaggle/input/smiles-extra-data')
    TC_BASE = os.getenv('TC_DATA_BASE', '/kaggle/input/tc-smiles')
    
    # 各个数据文件路径
    paths = {
        'train_csv': os.getenv('TRAIN_CSV_PATH', os.path.join(BASE_PATH, 'train.csv')),
        'test_csv': os.getenv('TEST_CSV_PATH', os.path.join(BASE_PATH, 'test.csv')),
        'sample_submission': os.getenv('SAMPLE_SUBMISSION_PATH', os.path.join(BASE_PATH, 'sample_submission.csv')),
        
        # Tc 数据
        'tc_data': os.getenv('TC_DATA_PATH', os.path.join(TC_BASE, 'Tc_SMILES.csv')),
        
        # 外部数据文件
        'tg_jcim_data': os.getenv('TG_JCIM_PATH', os.path.join(EXTRA_BASE, 'JCIM_sup_bigsmiles.csv')),
        'tg_excel_data': os.getenv('TG_EXCEL_PATH', os.path.join(EXTRA_BASE, 'data_tg3.xlsx')),
        'density_data': os.getenv('DENSITY_PATH', os.path.join(EXTRA_BASE, 'data_dnst1.xlsx')),
        
        # 补充数据目录
        'supplement_dir': os.getenv('SUPPLEMENT_DIR', os.path.join(BASE_PATH, 'train_supplement')),
        'ffv_data': os.getenv('FFV_DATA_PATH', os.path.join(BASE_PATH, 'train_supplement', 'dataset4.csv')),
        
        # 其他补充数据
        'dataset1': os.getenv('DATASET1_PATH', os.path.join(BASE_PATH, 'train_supplement', 'dataset1.csv')),
        'dataset2': os.getenv('DATASET2_PATH', os.path.join(BASE_PATH, 'train_supplement', 'dataset2.csv')),
        'dataset3': os.getenv('DATASET3_PATH', os.path.join(BASE_PATH, 'train_supplement', 'dataset3.csv')),
    }
    
    return paths

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
        data_out = Data(
                        x=x_masked,
                        x_masked=x_masked,
                        x_orig=x_orig,
                        edge_index=data.edge_index,
                        edge_attr_masked=e_masked,
                        edge_attr_orig=e_orig,
                        batch=getattr(data, 'batch', None)
                    )
        return data_out


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
    paths: dict = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    1. 加载主训练集 train.csv
    2. 标准化 SMILES
    3. 分别加载 Tc, Tg, Density 外部数据并增强
    4. 划分 train/val/test，保留缺失值
    
    Args:
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
        test_size: 测试集比例
        random_state: 随机种子
    """
    if paths is None:
        paths = get_data_paths()
    
    print("👉 加载主训练数据")
    train = pd.read_csv(paths['train_csv'])
    train['SMILES'] = train['SMILES'].apply(make_smile_canonical)
    print(f"  原始训练样本数: {len(train)}")

    # 2. 增强 Tc
    if os.path.exists(paths['tc_data']):
        df_tc = pd.read_csv(paths['tc_data']).rename(columns={'TC_mean': 'Tc'})
        train = add_extra_data(train, df_tc, 'Tc')
    else:
        print("  ⚠️ 未找到 Tc 外部数据")

    # 3. 增强 Tg 来源1
    if os.path.exists(paths['tg_jcim_data']):
        df_tg1 = pd.read_csv(paths['tg_jcim_data'], usecols=['SMILES', 'Tg (C)']).rename(columns={'Tg (C)': 'Tg'})
        train = add_extra_data(train, df_tg1, 'Tg')
    else:
        print("  ⚠️ 未找到 Tg 来源1 数据")

    # 4. 增强 Tg 来源2
    if os.path.exists(paths['tg_excel_data']):
        df_tg2 = pd.read_excel(paths['tg_excel_data']).rename(columns={'Tg [K]': 'Tg'})
        df_tg2['Tg'] = df_tg2['Tg'] - 273.15
        train = add_extra_data(train, df_tg2, 'Tg')
    else:
        print("  ⚠️ 未找到 Tg 来源2 数据")

    # 5. 增强 Density
    if os.path.exists(paths['density_data']):
        df_den = pd.read_excel(paths['density_data'])
        df_den = df_den.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
        df_den['Density'] = pd.to_numeric(df_den['Density'], errors='coerce') - 0.118
        train = add_extra_data(train, df_den, 'Density')
    else:
        print("  ⚠️ 未找到 Density 外部数据")

    # 6. 增强 FFV
    if os.path.exists(paths['ffv_data']):
        df_ffv = pd.read_csv(paths['ffv_data'])
        df_ffv = df_ffv.rename(columns={'FFV': 'FFV'})[['SMILES', 'FFV']]
        train = add_extra_data(train, df_ffv, 'FFV')
        print(f'add dataset4: {len(train)}')    
    else:
        print("  ⚠️ 未找到 FFV 外部数据")

    # 7. 划分数据集
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

def smiles_to_data(smiles, device=None):
    x, edge_index, edge_attr = create_graph_from_smiles(smiles)
    if x is None:
        raise ValueError("Invalid SMILES")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_weight = torch.ones(edge_attr.size(0))
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data.to(device or torch.device('cpu'))


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
    # 使用默认路径配置
    paths = get_data_paths()
    train_df, val_df, test_df = load_and_split_data(paths)

    targets = ['Tg', 'Tc', 'Density']
    train_dataset = PolymerDataset(train_df, y_cols=targets)
    val_dataset   = PolymerDataset(val_df,   y_cols=targets)
    test_dataset  = PolymerDataset(test_df,  y_cols=targets)

    print("👉 构建 DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    print(f"🔢 样本总计 train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
