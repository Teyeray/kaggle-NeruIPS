import optuna
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

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
    
    # 数据加载
    loader = DataLoader(dataset_nodeedge, batch_size=64, shuffle=True)
    
    # 简化训练：跑固定若干 epoch
    best_loss = float("inf")
    for epoch in range(5):
        loss = train_se_mask_epoch(model, loader, optimizer)
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_loss = min(best_loss, loss)
    
    # 保存最优 encoder
    torch.save(encoder.state_dict(), f"stage1_best_enc_trial{trial.number}.pt")
    return best_loss

# 启动 Stage1 的 Optuna 研究
study1 = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study1.optimize(objective_stage1, n_trials=20)
print("Stage1 best params:", study1.best_trial.params)


