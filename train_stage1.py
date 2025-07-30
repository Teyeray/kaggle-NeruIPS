import optuna
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import WDMPNN, NodeEdgeSSLModel
from data_preparation import NodeEdgeMaskDataset, PolymerDataset, load_and_split_data
def train_se_mask_epoch(model, loader, optimizer):
    model.train()
    total = 0
    for data in loader:
        data = data.to(device)
        node_pred, edge_pred = model(
            data.x_masked, data.edge_index, data.edge_attr_masked, data.batch
        )
        loss = F.mse_loss(node_pred, data.x_orig) + F.mse_loss(edge_pred, data.edge_attr_orig)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item() * data.num_graphs
    return total / len(loader.dataset)

def objective_stage1(trial):
    # 超参空间
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    
    # 模型、优化器
    encoder = WDMPNN(node_feat_dim, edge_feat_dim, hidden_dim, num_edge_layers=num_layers).to(device)
    model = NodeEdgeSSLModel(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(5):
        total = 0
        for batch in nodeedge_loader:
            batch = batch.to(device)
            node_pred, edge_pred = model(
                batch.x_masked, batch.edge_index,
                batch.edge_attr_masked, batch.batch
            )
            loss = (
                F.mse_loss(node_pred, batch.x_orig) +
                F.mse_loss(edge_pred, batch.edge_attr_orig)
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * batch.num_graphs
        avg = total / len(nodeedge_loader.dataset)
        trial.report(avg, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    torch.save(encoder.state_dict(), f"stage1_enc_best_{trial.number}.pt")
    return avg


base_path = "neurips-open-polymer-prediction-2025"
train_df, val_df, test_df = load_and_split_data(base_path)

# 1) 构造只含结构（不含 y）的 base_dataset
base_graph_ds = PolymerDataset(train_df, y_cols=[])

# 2) 包装 Node/Edge SSL 数据集
dataset_nodeedge = NodeEdgeMaskDataset(base_graph_ds, device=device)
nodeedge_loader = DataLoader(dataset_nodeedge, batch_size=64, shuffle=True)

# 启动 Stage1 的 Optuna 研究
storage_uri = "sqlite:///stage1_optuna.db"
study1 = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), storage=storage_uri)
study1.optimize(objective_stage1, n_trials=20)
print("Stage1 best params:", study1.best_trial.params)


