"""
数据集增强模块
参考金牌方案，实现外部数据集的加载和合并
"""

import pandas as pd
import numpy as np
from rdkit import Chem
import os

def make_smile_canonical(smile):
    """将SMILES转换为标准形式以避免重复"""
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

def add_extra_data(df_train, df_extra, target):
    """为特定目标添加外部数据"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    # 复制数据避免修改原始数据
    df_train = df_train.copy()
    df_extra = df_extra.copy()
    
    # 标准化SMILES
    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    # 找出交叉和独特的SMILES
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # 优先使用竞赛数据中的目标值
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)

    # 为竞赛SMILES填充缺失值
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]
    
    # 添加来自外部数据的新SMILES
    new_data = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]
    df_train = pd.concat([df_train, new_data], axis=0, ignore_index=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'  {target}: 添加了 {n_samples_after-n_samples_before} 个新样本')
    return df_train

def load_and_augment_train_data(base_path='neurips-open-polymer-prediction-2025/'):
    """加载训练数据并使用外部数据集进行增强"""
    print("📂 加载和增强训练数据...")
    
    # 加载基础训练数据
    train = pd.read_csv(base_path + 'train.csv')
    train['SMILES'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
    
    # 加载外部数据集
    print("📂 加载外部数据集...")
    
    try:
        # Tc数据
        data_tc = pd.read_csv('Tc_SMILES.csv')
        data_tc = data_tc.rename(columns={'TC_mean': 'Tc'})
        train = add_extra_data(train, data_tc, 'Tc')
    except Exception as e:
        print(f"⚠️ 无法加载Tc数据: {e}")
    
    try:
        # Tg数据来源1
        data_tg2 = pd.read_csv('smiles-extra-data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
        data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})
        train = add_extra_data(train, data_tg2, 'Tg')
    except Exception as e:
        print(f"⚠️ 无法加载Tg数据2: {e}")
    
    try:
        # Tg数据来源2
        data_tg3 = pd.read_excel('smiles-extra-data/data_tg3.xlsx')
        data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
        data_tg3['Tg'] = data_tg3['Tg'] - 273.15  # 转换为摄氏度
        train = add_extra_data(train, data_tg3, 'Tg')
    except Exception as e:
        print(f"⚠️ 无法加载Tg数据3: {e}")
    
    try:
        # Density数据
        data_dnst = pd.read_excel('smiles-extra-data/data_dnst1.xlsx')
        data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
        data_dnst['SMILES'] = data_dnst['SMILES'].apply(lambda s: make_smile_canonical(s))
        data_dnst = data_dnst[(data_dnst['SMILES'].notnull())&(data_dnst['Density'].notnull())&(data_dnst['Density'] != 'nylon')]
        data_dnst['Density'] = data_dnst['Density'].astype('float64')
        data_dnst['Density'] -= 0.118  # 调整密度值
        train = add_extra_data(train, data_dnst, 'Density')
    except Exception as e:
        print(f"⚠️ 无法加载密度数据: {e}")
    
    print(f"📊 最终训练样本数量:")
    targets = ['Tg', 'Tc', 'Rg', 'FFV', 'Density']
    for t in targets:
        print(f'  {t}: {len(train[train[t].notnull()])} 个样本')
    
    return train

def load_test_data(base_path='neurips-open-polymer-prediction-2025/'):
    """加载测试数据"""
    print("📂 加载测试数据...")
    test = pd.read_csv(base_path + 'test.csv')
    test['SMILES'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))
    return test

def clean_data(df):
    """清理数据，处理缺失值和异常值"""
    print("🧹 清理数据...")
    
    # 处理无穷大值
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # 处理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'id':  # 不处理ID列
            df[col] = df[col].fillna(df[col].mean())
    
    return df 