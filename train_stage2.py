# stage2_graph_pretrain.py

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import WDMPNN, GraphSSLModel
from data_preparation import load_and_split_data, PolymerDataset

from rdkit import Chem
from rdkit.Chem import Descriptors

def add_pseudo_label(dataset):
    """
    对每个 Data 计算 graph-level pseudo label：ensemble molecular weight
    并动态添加 data.pseudo 属性
    """
    for data in dataset:
        smiles = data.smiles if hasattr(data, 'smiles') else None
        if smiles is None:
            # 临时 fallback，用原子质量粗略估算
            mol_weight = data.x[:, 0].sum().item()
        else:
            mol = Chem.MolFromSmiles(smiles)
            mol_weight = Descriptors.MolWt(mol) if mol is not None else data.x[:, 0].sum().item()
        data.pseudo = torch.tensor([mol_weight], dtype=torch.float)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载并划分，只取 train part，用于 SSL 预训练
    base_path = "neurips-open-polymer-prediction-2025"
    train_df, _, _ = load_and_split_data(base_path)
    base_graph_ds = PolymerDataset(train_df, y_cols=[])   # 不需要 y

    # 在构建 Data 时，务必将 ensemble pseudo label 存到 Data.pseudo
    # 添加 pseudo label（ensemble molecular weight）
    add_pseudo_label(base_graph_ds)

    graph_loader = DataLoader(base_graph_ds, batch_size=64, shuffle=True)

    # 2) 从 Stage1 读取最佳超参 & encoder 权重
    # 假设 study1.best_trial.params 已经存在
    best = torch.load("stage1_best_params.pt")  # 存好的 dict
    encoder = WDMPNN(
        node_feat_dim=2,
        edge_feat_dim=1,
        hidden_dim=best["hidden_dim"],
        num_edge_layers=best["num_edge_layers"]
    ).to(device)
    encoder.load_state_dict(torch.load("stage1_encoder_best.pt"))

    # 3) 构建 GraphSSLModel（不 freeze encoder）
    model2 = GraphSSLModel(encoder, mlp_hidden_dim=best["hidden_dim"] // 2).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=best["lr"])

    # 4) 训练循环
    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        model2.train()
        total_loss = 0
        for data in graph_loader:
            data = data.to(device)
            pseudo = data.pseudo.to(device).view(-1, 1)               # [batch_size,1]
            pred   = model2(
                data.x,
                data.edge_index,
                data.edge_attr,
                torch.ones(data.edge_attr.size(0), device=device),
                data.batch
            )
            loss = F.mse_loss(pred, pseudo)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            total_loss += loss.item() * data.num_graphs

        avg = total_loss / len(graph_loader.dataset)
        print(f"[Stage2] Epoch {epoch:02d}/{n_epochs}  loss={avg:.4f}")

    # 5) 保存 Stage2 学得的 encoder
    torch.save(encoder.state_dict(), "stage2_encoder.pt")
    # 如需保存 Stage2 predictor，也一并保存
    torch.save(model2.predictor.state_dict(), "stage2_predictor.pt")


if __name__ == "__main__":
    main()