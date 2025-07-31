import optuna
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import WDMPNN, NodeEdgeSSLModel
from data_preparation import load_and_split_data, PolymerDataset, NodeEdgeMaskDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 加载并划分原始数据（只用 SMILES，不需要 y）
base_path = "neurips-open-polymer-prediction-2025"
train_df, _, _ = load_and_split_data(base_path)
base_graph_ds = PolymerDataset(train_df, y_cols=[])                # y_cols=[] 表示不生成 y

# 2) 构造 Node/Edge SSL 数据集和 DataLoader
dataset_nodeedge = NodeEdgeMaskDataset(base_graph_ds, device=device)
nodeedge_loader   = DataLoader(dataset_nodeedge, batch_size=64, shuffle=True)

# 3) 定义 Optuna 目标函数
def objective_stage1(trial):
    # 超参空间
    lr         = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    n_layers   = trial.suggest_int("num_edge_layers", 2, 4)

    # 模型 & SSL head
    encoder = WDMPNN(
        node_feat_dim=2,        # 与 create_graph_from_smiles 中的 node_feat_dim 对应
        edge_feat_dim=1,        # 与 edge_feat_dim 对应
        hidden_dim=hidden_dim,
        num_edge_layers=n_layers
    ).to(device)
    model = NodeEdgeSSLModel(encoder, node_feat_dim=2, edge_feat_dim=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练若干 epoch
    for epoch in range(5):
        total_loss = 0
        for batch in nodeedge_loader:
            batch = batch.to(device)
            node_pred, edge_pred = model(
                batch.x_masked,
                batch.edge_index,
                batch.edge_attr_masked,
                batch.batch
            )
            loss = (
                F.mse_loss(node_pred, batch.x_orig) +
                F.mse_loss(edge_pred, batch.edge_attr_orig)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(nodeedge_loader.dataset)
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # 保存最优 encoder 权重
    torch.save(encoder.state_dict(), f"stage1_encoder_trial{trial.number}.pt")
    return avg_loss

# 4) 启动 Optuna
storage_uri = "sqlite:///stage1_optuna.db"
study1 = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
    storage=storage_uri
)
study1.optimize(objective_stage1, n_trials=20)

print("Stage1 best params:", study1.best_trial.params)