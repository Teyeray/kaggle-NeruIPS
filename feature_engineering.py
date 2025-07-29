"""
特征工程模块
实现分子描述符、指纹、图特征的计算
"""

import numpy as np
import pandas as pd
import warnings
from collections import Counter
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# RDKit相关
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdmolops, MACCSkeys, rdMolDescriptors

# Mordred相关
from mordred import Calculator, descriptors

# 图分析相关
import networkx as nx

# 忽略警告
warnings.filterwarnings("ignore")

# 定义要排除的无用特征列（参照金牌pipeline）
USELESS_COLS = [    
    # NaN数据
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
    
    # 常数数据
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan',
    'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan',
    
    # 高相关数据 >0.95
    'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1',
    'LabuteASA', 'HeavyAtomCount', 'MolMR', 'Chi3n', 'BertzCT', 'Chi2v',
    'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile', 'SlogP_VSA6',
    'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6',
    'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors', 'fr_amide',
    'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2',
    'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester'
]

def compute_molecular_features_for_smiles(smiles, useless_cols):
    """为单个SMILES计算所有分子特征"""
    try:
        # 初始化结果字典
        results = {}
        
        # 计算RDKit描述符
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
            results.update({name: None for name in desc_names})
        else:
            desc_results = [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]
            desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
            results.update(dict(zip(desc_names, desc_results)))
        
        # 计算图特征
        graph_defaults = {
            'graph_diameter': 0, 'avg_shortest_path': 0, 'num_cycles': 0,
            'betweenness_mean': 0, 'betweenness_std': 0, 'eigenvector_mean': 0, 
            'ring_4': 0, 'max_degree': 0, 'closeness_mean': 0, 'katz_centrality_std': 0,
            'heteroatom_ratio': 0
        }
        
        if mol is not None:
            try:
                adj = rdmolops.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)
                
                # 图直径和最短路径
                if nx.is_connected(G):
                    results['graph_diameter'] = nx.diameter(G)
                    results['avg_shortest_path'] = nx.average_shortest_path_length(G)
                else:
                    results['graph_diameter'] = 0
                    results['avg_shortest_path'] = 0
                
                # 最大度数
                n_nodes = G.number_of_nodes()
                results['max_degree'] = max([d for n, d in G.degree()]) if n_nodes > 0 else 0
                
                # 接近中心性
                if nx.is_connected(G):
                    closeness = list(nx.closeness_centrality(G).values())
                    results['closeness_mean'] = np.mean(closeness) if closeness else 0
                else:
                    results['closeness_mean'] = 0
                
                # Katz中心性
                try:
                    katz = list(nx.katz_centrality(G, max_iter=1000).values())
                    results['katz_centrality_std'] = np.std(katz) if len(katz) > 1 else 0
                except:
                    results['katz_centrality_std'] = 0
                
                # 环数量
                cycles = list(nx.cycle_basis(G))
                results['num_cycles'] = len(cycles)
                
                # 介数中心性
                betweenness = list(nx.betweenness_centrality(G).values())
                if betweenness:
                    betweenness = [b for b in betweenness if np.isfinite(b)]
                    results['betweenness_mean'] = np.mean(betweenness) if betweenness else 0
                    results['betweenness_std'] = np.std(betweenness) if len(betweenness) > 1 else 0
                else:
                    results['betweenness_mean'] = 0
                    results['betweenness_std'] = 0
                
                # 特征向量中心性
                try:
                    eigenvector = list(nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6).values())
                    eigenvector = [e for e in eigenvector if np.isfinite(e)]
                    results['eigenvector_mean'] = np.mean(eigenvector) if eigenvector else 0
                except:
                    results['eigenvector_mean'] = 0
                
                # 环分析
                cycle_lengths = [len(cycle) for cycle in cycles]
                results['ring_4'] = sum(1 for length in cycle_lengths if length == 4)
                
                # 原子特定特征
                try:
                    # 原子类型分布
                    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
                    atom_counts = Counter(atom_types)
                    
                    # 杂原子比例
                    total_atoms = len(atom_types)
                    hetero_atoms = total_atoms - atom_counts.get('C', 0)
                    results['heteroatom_ratio'] = hetero_atoms / total_atoms if total_atoms > 0 else 0
                except Exception:
                    results['heteroatom_ratio'] = 0
                
            except Exception as e:
                results.update(graph_defaults)
        else:
            results.update(graph_defaults)
        
        # 计算Mordred特征（关键特征）
        mordred_defaults = {'AMW': 0, 'TIC2': 0, 'naRing': 0, 'MPC3': 0}
        
        if mol is not None:
            try:
                calc = Calculator([
                    descriptors.Weight,
                    descriptors.InformationContent,
                    descriptors.RingCount,
                    descriptors.PathCount
                ])
                
                mordred_result = calc(mol)
                desc_dict = dict(zip(calc.descriptors, mordred_result))
                
                # 提取特定描述符
                amw = next((v for k, v in desc_dict.items() if 'AMW' in str(k)), 0)
                tic2 = next((v for k, v in desc_dict.items() if 'TIC2' in str(k)), 0)
                naring = next((v for k, v in desc_dict.items() if 'naRing' in str(k)), 0)
                mpc3 = next((v for k, v in desc_dict.items() if 'MPC3' in str(k)), 0)
                
                results['AMW'] = float(amw) if amw is not None and not isinstance(amw, type(None)) else 0
                results['TIC2'] = float(tic2) if tic2 is not None and not isinstance(tic2, type(None)) else 0
                results['naRing'] = float(naring) if naring is not None and not isinstance(naring, type(None)) else 0
                results['MPC3'] = float(mpc3) if mpc3 is not None and not isinstance(mpc3, type(None)) else 0
                
            except Exception as e:
                results.update(mordred_defaults)
        else:
            results.update(mordred_defaults)
        
        # 计算特定MACCS指纹特征
        maccs_defaults = {
            'MACCS_Key130': 0, 'MACCS_Key142': 0, 'MACCS_Key066': 0, 'MACCS_Key153': 0
        }
        
        if mol is not None:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fingerprint = [int(x) for x in fp.ToBitString()]
                
                # 提取特定MACCS keys（转换为0基索引）
                if len(fingerprint) >= 167:
                    results['MACCS_Key130'] = fingerprint[129] if 129 < len(fingerprint) else 0  # 130-1
                    results['MACCS_Key142'] = fingerprint[141] if 141 < len(fingerprint) else 0  # 142-1
                    results['MACCS_Key066'] = fingerprint[65] if 65 < len(fingerprint) else 0   # 66-1
                    results['MACCS_Key153'] = fingerprint[152] if 152 < len(fingerprint) else 0  # 153-1
                else:
                    results.update(maccs_defaults)
                    
            except Exception as e:
                results.update(maccs_defaults)
        else:
            results.update(maccs_defaults)
        
        # 计算特定TopologicalTorsion指纹特征
        torsion_defaults = {
            'TopologicalTorsion_Bit0512': 0, 'TopologicalTorsion_Bit1296': 0
        }
        
        if mol is not None:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
                    fingerprint = [int(x) for x in fp.ToBitString()]
                    
                    # 提取特定TopologicalTorsion bits
                    if len(fingerprint) >= 2048:
                        results['TopologicalTorsion_Bit0512'] = fingerprint[512] if 512 < len(fingerprint) else 0
                        results['TopologicalTorsion_Bit1296'] = fingerprint[1296] if 1296 < len(fingerprint) else 0
                    else:
                        results.update(torsion_defaults)
                    
            except Exception as e:
                results.update(torsion_defaults)
        else:
            results.update(torsion_defaults)

        # 计算特定AtomPair指纹特征
        atom_pair_defaults = {
            'AtomPair_B512_Bit0138': 0, 
            'AtomPair_B512_Bit0448': 0, 
            'AtomPair_B512_Bit0408': 0
        }

        if mol is not None:
            try:
                fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512)
                fingerprint = [int(x) for x in fp.ToBitString()]

                # 提取特定AtomPair bits
                if len(fingerprint) == 512:
                    results['AtomPair_B512_Bit0138'] = fingerprint[138]
                    results['AtomPair_B512_Bit0448'] = fingerprint[448]
                    results['AtomPair_B512_Bit0408'] = fingerprint[408]
                else:
                    results.update(atom_pair_defaults)

            except Exception as e:
                results.update(atom_pair_defaults)
        else:
            results.update(atom_pair_defaults)
        
        return results
        
    except Exception as e:
        print(f"处理SMILES时出现严重错误 {smiles}: {e}")
        return None

def process_molecular_features_parallel(df, useless_cols, n_jobs=4):
    """使用joblib并行处理分子特征"""
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    print(f"🔄 使用joblib并行处理，{n_jobs}个进程")
    
    smiles_list = df['SMILES'].tolist()
    
    # 并行处理
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(compute_molecular_features_for_smiles)(smiles, useless_cols) for smiles in smiles_list
    )
    
    # 过滤None结果
    results = [r for r in results if r is not None]
    
    if not results:
        print("⚠️ 没有计算到有效的分子特征")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    result_df = result_df.replace([-np.inf, np.inf], np.nan)
    
    return result_df

def clean_features_and_compute_means(df, feature_columns):
    """清理特征并计算均值用于训练数据"""
    print("🧹 清理特征并计算均值...")
    
    # 复制避免修改原始数据
    df = df.copy()
    
    # 替换无穷大值为NaN
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].replace([-np.inf, np.inf], np.nan)
    
    # 计算非NaN值的均值
    feature_means = {}
    for col in feature_columns:
        if col in df.columns:
            mean_val = df[col].mean()
            feature_means[col] = mean_val if not np.isnan(mean_val) else 0
        else:
            feature_means[col] = 0
    
    print(f"✅ 为{len(feature_means)}个特征计算了均值")
    return df, feature_means

def apply_imputation_to_test(df, feature_columns, feature_means):
    """对测试数据应用特征清理和填充"""
    print("🧹 清理和填充测试特征...")
    
    # 复制避免修改原始数据
    df = df.copy()
    
    # 替换无穷大值为NaN
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].replace([-np.inf, np.inf], np.nan)
    
    # 用训练均值替换NaN
    for col in feature_columns:
        if col in df.columns and col in feature_means:
            df[col] = df[col].fillna(feature_means[col])
        elif col not in df.columns and col in feature_means:
            # 添加缺失列并填充均值
            df[col] = feature_means[col]
    
    print(f"✅ 对{len(feature_columns)}个特征应用了填充")
    return df

def validate_features(df, expected_features, phase='train'):
    """验证所有预期特征是否存在"""
    missing_features = []
    for feat in expected_features:
        if feat not in df.columns:
            missing_features.append(feat)
    
    if missing_features:
        print(f"⚠️ 警告: {phase}数据中缺少{len(missing_features)}个特征:")
        print(f"   {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
    else:
        print(f"✅ {phase}数据中包含所有预期特征")
    
    return missing_features

if __name__ == "__main__":
    # 测试特征计算
    test_smiles = ["CCO", "c1ccccc1", "CC(C)(C)c1ccc(N)cc1"]
    test_df = pd.DataFrame({'SMILES': test_smiles})
    
    features = process_molecular_features_parallel(test_df, USELESS_COLS, n_jobs=2)
    print(f"计算的特征形状: {features.shape}")
    print(f"特征列: {list(features.columns[:10])}...") 