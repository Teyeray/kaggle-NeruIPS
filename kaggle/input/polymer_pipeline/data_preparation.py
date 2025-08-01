import os
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdPartialCharges
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

import sys
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
    BASE_PATH = os.getenv('NEURIPS_DATA_PATH', 'kaggle/input/neurips-open-polymer-prediction-2025')
    EXTRA_BASE = os.getenv('EXTRA_DATA_BASE', 'kaggle/input/smiles-extra-data')
    TC_BASE = os.getenv('TC_DATA_BASE', 'kaggle/input/tc-smiles')
    
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

# def clean_smiles(smiles: str) -> str:
#     """清洗非法字符：[*], [R], (*), * 等占位符"""
#     if not isinstance(smiles, str):
#         return None
#     s = re.sub(r"\[[^\]]*\]", "", smiles)         # 删除所有方括号内容
#     s = re.sub(r"\([^)]*\*[^)]*\)", "", s)        # 删除括号中包含 * 的片段
#     s = s.replace("*", "")                          # 删除孤立星号
#     s = re.sub(r"\.+", ".", s).strip(".")         # 清理多余点号
#     return s or None

def make_smile_canonical(smile):
    """清洗并标准化 SMILES"""
    try:
        #smile = clean_smiles(smile)  # 🔥先清洗非法字符
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
    cross = set(df_extra['SMILES']) & set(df_train['SMILES'])
    existing = set(df_train[df_train[target].notnull()]['SMILES'])
    fill = cross - existing
    print(f"cross_smiles: {len(cross)} | 填充: {len(fill)}")
    for smi in fill:
        val = df_extra.loc[df_extra['SMILES']==smi, target].item()
        df_train.loc[df_train['SMILES']==smi, target] = val
    extra = df_extra[~df_extra['SMILES'].isin(df_train['SMILES'])]
    print(f"新增样本: {len(extra)} 条")
    df_train = pd.concat([df_train, extra], ignore_index=True)
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
    train = pd.read_csv(paths['train_csv'])
    train['SMILES'] = train['SMILES'].apply(make_smile_canonical)
    print(f"原始训练: {len(train)} 条")
    # 增强
    if os.path.exists(paths['tc_data']):       train = add_extra_data(train, pd.read_csv(paths['tc_data']).rename(columns={'TC_mean':'Tc'}), 'Tc')
    if os.path.exists(paths['tg_jcim_data']):  train = add_extra_data(train, pd.read_csv(paths['tg_jcim_data'],usecols=['SMILES','Tg (C)']).rename(columns={'Tg (C)':'Tg'}), 'Tg')
    if os.path.exists(paths['tg_excel_data']): train = add_extra_data(train, pd.read_excel(paths['tg_excel_data']).rename(columns={'Tg [K]':'Tg'}).assign(Tg=lambda df:df['Tg']-273.15), 'Tg')
    if os.path.exists(paths['density_data']):  train = add_extra_data(train, pd.read_excel(paths['density_data']).rename(columns={'density(g/cm3)':'Density'}).assign(Density=lambda df:pd.to_numeric(df['Density'],errors='coerce')-0.118), 'Density')
    if os.path.exists(paths['ffv_data']):      train = add_extra_data(train, pd.read_csv(paths['ffv_data']).rename(columns={'FFV':'FFV'}), 'FFV')
    
    # 划分
    t, tmp = train_test_split(train, test_size=test_size, random_state=random_state)
    v, te  = train_test_split(tmp,   test_size=0.5, random_state=random_state)
    return t.reset_index(drop=True), v.reset_index(drop=True), te.reset_index(drop=True)


# -----------------------------
# PyG 数据集定义
# -----------------------------

from rdkit.Chem import rdPartialCharges
import numpy as np

def create_graph_from_smiles(smiles):
    """
    SMILES -> Graph with extended atom & bond features
    如果解析失败，抛 ValueError
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES, MolFromSmiles 返回 None: {smiles}")

    # 计算 Gasteiger 电荷（可选）
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    # --- 构建节点特征 ---
    x_feats = []
    for atom in mol.GetAtoms():
        x_feats.append([
            atom.GetAtomicNum(),
            int(atom.GetIsAromatic()),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            atom.GetMass(),
            int(atom.GetChiralTag()!=Chem.CHI_UNSPECIFIED),
            #float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
        ])
    x = torch.tensor(x_feats, dtype=torch.float)

    # --- 构建边特征与索引 ---
    e_idx, e_feats = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        feats = [
            b.GetBondTypeAsDouble(),
            int(b.GetIsConjugated()),
            int(b.IsInRing()),
            float(b.GetStereo())
        ]
        # 双向边
        e_idx += [[i, j], [j, i]]
        e_feats += [feats, feats]

    edge_index = torch.tensor(e_idx, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(e_feats, dtype=torch.float)

    # --- 可选：立即自检，确保节点/边数量正确 ---
    # 1. 原子数 vs 节点数
    n_atoms = mol.GetNumAtoms()
    if x.size(0) != n_atoms:
        raise AssertionError(f"节点数量不匹配 ({x.size(0)}) vs 原子数 ({n_atoms}): {smiles}")

    # 2. 键数 vs 边数
    n_bonds = mol.GetNumBonds()
    if edge_attr.size(0) != 2 * n_bonds:
        raise AssertionError(f"边数量不匹配 ({edge_attr.size(0)}) vs 2×键数 ({2*n_bonds}): {smiles}")

    # 3. 邻接矩阵对比
    from rdkit.Chem.rdmolops import GetAdjacencyMatrix
    adj_rd    = GetAdjacencyMatrix(mol)
    adj_graph = np.zeros_like(adj_rd)
    ei_np     = edge_index.cpu().numpy()
    for u, v in ei_np.T:
        adj_graph[u, v] = 1
    if not np.array_equal(adj_rd, adj_graph):
        raise AssertionError(f"邻接矩阵不匹配: {smiles}")

    return x, edge_index, edge_attr

def smiles_to_data(smiles, device=None):
    x, edge_index, edge_attr = create_graph_from_smiles(smiles)
    if x is None:
        raise ValueError("Invalid SMILES")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_weight = torch.ones(edge_attr.size(0))
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data.to(device or torch.device('cpu'))
    
class PolymerDataset(InMemoryDataset):
    def __init__(self, df, y_cols, transform=None):
        super().__init__(None, transform)
        print(f"📦 构建 PolymerDataset，样本数={len(df)}")
        self.data_list = []
        for _, row in df.iterrows():
            smi = row['SMILES']
            try:
                x, ei, ea = create_graph_from_smiles(smi)
                validate_graph(smi, x, ei, ea)
            except Exception as e:
                print(f"[Validation FAILED] {smi} → {e}")
                continue        # 跳过这条
            y = torch.tensor([row[c] for c in y_cols], dtype=torch.float)
            data = Data(x=x, edge_index=ei, edge_attr=ea, y=y, smiles=smi)
            self.data_list.append(data)
        print(f"   成功转换为图数据: {len(self.data_list)} 条")
    def len(self): return len(self.data_list)
    def get(self,idx): return self.data_list[idx]


# -----------------------------
# 构造数据集 & DataLoader
# -----------------------------
def validate_graph(smiles, x, edge_index, edge_attr):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    n_atoms = mol.GetNumAtoms()
    if x.size(0) != n_atoms:
        raise AssertionError(f"Nodes ({x.size(0)}) ≠ atoms ({n_atoms}) for {smiles}")
    
    n_bonds = mol.GetNumBonds()
    if edge_attr.size(0) != 2*n_bonds:
        raise AssertionError(f"Edges ({edge_attr.size(0)}) ≠ 2×bonds ({2*n_bonds}) for {smiles}")
    
    from rdkit.Chem.rdmolops import GetAdjacencyMatrix
    adj_rd = GetAdjacencyMatrix(mol)
    adj_graph = np.zeros_like(adj_rd)
    ei = edge_index.cpu().numpy()
    for u,v in ei.T:
        adj_graph[u,v] = 1
    if not np.array_equal(adj_rd, adj_graph):
        raise AssertionError(f"Adjacency mismatch for {smiles}")
    

if __name__ == "__main__":
    # 使用默认路径配置
    paths = get_data_paths()
    train_df, val_df, test_df = load_and_split_data(paths)

    targets = ['Tg','Tc','Density','FFV','Rg']
    train_dataset = PolymerDataset(train_df, y_cols=targets)
    val_dataset   = PolymerDataset(val_df,   y_cols=targets)
    test_dataset  = PolymerDataset(test_df,  y_cols=targets)

    print("👉 构建 DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    print(f"🔢 样本总计 train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        # 1) 取一个 NodeEdgeMaskDataset 对应的 loader
    _, _, nodeedge_loader, device = setup_stage1_data(train_df)

    # 2) 抽一个 batch
    batch = next(iter(nodeedge_loader))

    # 3) 根据这个 batch 自动推断节点/边维度
    node_dim = batch.x_masked.size(-1)
    edge_dim = batch.edge_attr_masked.size(-1)
    print(f"Sample batch: node_feat_dim={node_dim}, edge_feat_dim={edge_dim}")

    # 4) 构造一个小模型（用最小 hidden_dim）
    params = {"hidden_dim": 16, "num_edge_layers": 2}
    encoder, model = create_stage1_model(params, device, sample_batch=batch)

    # 5) 前向一次
    node_pred, edge_pred = model(
        batch.x_masked.to(device),
        batch.edge_index.to(device),
        batch.edge_attr_masked.to(device),
        batch.batch.to(device),
    )

    # 6) 验证 shape
    assert node_pred.shape == batch.x_orig.shape, \
        f"node_pred {node_pred.shape} vs x_orig {batch.x_orig.shape}"
    assert edge_pred.shape == batch.edge_attr_orig.shape, \
        f"edge_pred {edge_pred.shape} vs edge_attr_orig {batch.edge_attr_orig.shape}"

    print("✅ Shape check passed!")
