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
            data = Data(x=x, edge_index=ei, edge_attr=ea, y=y, smiles=smi)  # ← Add smiles here!
            self.data_list.append(data)
        print(f"   成功转换为图数据: {len(self.data_list)} 条")
    def len(self): return len(self.data_list)
    def get(self,idx): return self.data_list[idx]

def add_pseudo_label(dataset):
    """为数据集添加graph-level伪标签（分子量）"""
    for data in dataset:
        smiles = getattr(data, 'smiles', None)
        
        # Check for both None and string 'None'
        if smiles and smiles != 'None':
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol_weight = Descriptors.MolWt(mol) if mol else None
            except:
                mol_weight = None
        else:
            mol_weight = None
        
        # 回退方案：使用原子质量求和
        if mol_weight is None:
            mol_weight = data.x[:, 0].sum().item()  # Using atomic numbers as fallback
            if smiles:  # Only print warning if there was an actual SMILES string
                print(f"Warning: Invalid SMILES '{smiles}', using atom mass sum as pseudo label.")
        
        data.pseudo = torch.tensor([mol_weight], dtype=torch.float)
    return dataset