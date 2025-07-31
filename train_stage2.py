# stage2_graph_pretrain.py

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import WDMPNN, GraphSSLModel
from data_preparation import load_and_split_data, PolymerDataset

def compute_ensemble_M(data):
    """
    计算论文 §2.3.2 中的 ensemble molecular weight，
    这里假设 data.y 存着每个 node 所属 monomer 的分子量，
    data.batch 存图索引，data.x 里没有分子量，我们需要自己预先
    在 Data 对象里添加 pseudo 属性。此处仅示意：
    """
    # pseudo = torch.randn((data.num_graphs, 1), device=data.x.device)
    # 实际用法要根据你的数据结构来写
    return data.pseudo  # 在构建 Dataset 时就把 pseudo label 放入 Data.pseudo

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载并划分，只取 train part，用于 SSL 预训练
    base_path = "neurips-open-polymer-prediction-2025"
    train_df, _, _ = load_and_split_data(base_path)
    base_graph_ds = PolymerDataset(train_df, y_cols=[])   # 不需要 y

    # 在构建 Data 时，务必将 ensemble pseudo label 存到 Data.pseudo
    # 这里假设 PolymerDataset 已经在 Data 对象上加了 pseudo

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
    encoder.load_state_dict(torch.load(f"stage1_encoder_trial{best['trial_number']}.pt"))

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
            pseudo = compute_ensemble_M(data)               # [batch_size,1]
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