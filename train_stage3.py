# stage3_finetune.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import WDMPNN, GraphPredictor
from data_preparation import load_and_split_data, PolymerDataset

def freeze_parameters(module):
    for p in module.parameters():
        p.requires_grad = False

def main(transfer_strategy: str = "c"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载有标签的 train/val
    base_path = "neurips-open-polymer-prediction-2025"
    train_df, val_df, _ = load_and_split_data(base_path)
    y_cols = ['Tg','Tc','Density']
    train_ds = PolymerDataset(train_df, y_cols=y_cols)
    val_ds   = PolymerDataset(val_df,   y_cols=y_cols)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # 2) 恢复 Stage1 最佳超参及 Stage2 权重
    best = torch.load("stage1_best_params.pt")  # dict 包含 hidden_dim, num_edge_layers, lr
    encoder = WDMPNN(
        node_feat_dim=2,
        edge_feat_dim=1,
        hidden_dim=best["hidden_dim"],
        num_edge_layers=best["num_edge_layers"]
    ).to(device)
    encoder.load_state_dict(torch.load("stage2_encoder.pt"))

    predictor = GraphPredictor(
        hidden_dim=encoder.hidden_dim,
        mlp_hidden_dim=best["hidden_dim"]//2,
        output_dim=encoder.hidden_dim
    ).to(device)
    if transfer_strategy in ("b","c"):
        predictor.load_state_dict(torch.load("stage2_predictor.pt"))

    # 3) 根据策略冻结
    freeze_parameters(encoder)
    if transfer_strategy=="a":
        freeze_parameters(predictor)
    elif transfer_strategy=="b":
        # 只开放最后一层
        for name,p in predictor.named_parameters():
            if "2" not in name:
                p.requires_grad=False
    elif transfer_strategy=="c":
        freeze_parameters(predictor)
    else:
        raise ValueError

    # 4) 新建下游 head
    downstream = nn.Sequential(
        nn.Linear(encoder.hidden_dim, 128),
        nn.ReLU(),
        nn.Linear(128, len(y_cols))
    ).to(device)

    # 5) optimizer 只优化 downstream
    optim = torch.optim.Adam(downstream.parameters(), lr=1e-3)

    # 6) 训练
    for epoch in range(1, 101):
        downstream.train()
        tot=0.0
        for data in train_loader:
            data = data.to(device)
            # 先 encoder -> predictor (可选)
            h = encoder(
                data.x, data.edge_index, data.edge_attr,
                torch.ones(data.edge_attr.size(0),device=device),
                data.batch
            )
            if transfer_strategy in ("b","c"):
                h = predictor(h)
            out = downstream(h)
            y = torch.stack([data[c] for c in y_cols], dim=1)
            loss = F.mse_loss(out, y)
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item()*data.num_graphs

        rmse = torch.sqrt(torch.tensor(tot/len(train_loader.dataset)))
        print(f"[Stage3:{transfer_strategy}] epoch={epoch:03d} train_RMSE={rmse:.4f}")

    # 7) 保存
    torch.save(downstream.state_dict(), f"stage3_head_{transfer_strategy}.pt")

if __name__=="__main__":
    main(transfer_strategy="c")