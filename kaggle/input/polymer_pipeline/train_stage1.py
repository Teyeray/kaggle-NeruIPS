import optuna
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os
from model import WDMPNN, NodeEdgeSSLModel
from data_preparation import load_and_split_data, PolymerDataset, NodeEdgeMaskDataset, get_data_paths

def setup_stage1_data(paths=None):
    """
    设置Stage1训练所需的数据
    
    Args:
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
    
    Returns:
        tuple: (train_df, base_graph_ds, dataset_nodeedge, nodeedge_loader)
    """
    if paths is None:
        paths = get_data_paths()
    
    # 1) 加载并划分原始数据（只用 SMILES，不需要 y）
    train_df, _, _ = load_and_split_data(paths)
    base_graph_ds = PolymerDataset(train_df, y_cols=[])                # y_cols=[] 表示不生成 y

    # 2) 构造 Node/Edge SSL 数据集和 DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_nodeedge = NodeEdgeMaskDataset(base_graph_ds, device=device)
    nodeedge_loader = DataLoader(dataset_nodeedge, batch_size=64, shuffle=True)
    
    return train_df, base_graph_ds, dataset_nodeedge, nodeedge_loader

# 默认设置（向后兼容）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paths = get_data_paths()
train_df, _, _ = load_and_split_data(paths)
base_graph_ds = PolymerDataset(train_df, y_cols=[])
dataset_nodeedge = NodeEdgeMaskDataset(base_graph_ds, device=device)
nodeedge_loader = DataLoader(dataset_nodeedge, batch_size=64, shuffle=True)

def objective_stage1(
    trial,
    nodeedge_loader,
    device,
    out_dir: str = "stage1_checkpoints",
    prefix: str = "encoder"
):
    # 超参空间
    lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    n_layers   = trial.suggest_int("num_edge_layers", 2, 4)

    # 早停参数
    patience = 15
    min_delta = 1e-3
    best_loss = float('inf')
    no_improve = 0

    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)

    # 模型 & SSL head
    encoder = WDMPNN(2, 1, hidden_dim, n_layers).to(device)
    model   = NodeEdgeSSLModel(encoder, node_feat_dim=2, edge_feat_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch = 0
    while True:
        epoch += 1
        total_loss = 0.0
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

        # 如果性能提升，保存 checkpoint
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            no_improve = 0
            ckpt_path = os.path.join(
                out_dir,
                f"{prefix}_trial{trial.number}_epoch{epoch}.pt"
            )
            torch.save(encoder.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # 最后再保存一次最优权重（以 trial 号为标识）
    final_ckpt = os.path.join(out_dir, f"{prefix}_best_trial{trial.number}.pt")
    torch.save(encoder.state_dict(), final_ckpt)
    return best_loss


def init_sudy1(
        storage_uri="sqlite:///stage1_optuna.db",
        study_name="stage1_nodeedge_ssl"
):
    """
    初始化 Stage1 的 Optuna Study
    """
    # 确保输出目录存在
    os.makedirs("stage1_checkpoints", exist_ok=True)

    # 创建或加载 Study
    study = optuna.create_study(
        study_name= study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage_uri,
        load_if_exists=True,
    )
    return study
#study1.optimize(objective_stage1, show_progress_bar=True, n_trials=20)
print("Stage1 best params:", study1.best_trial.params)
best_trial = study1.best_trial
best_params = best_trial.params
best_params["trial_number"] = best_trial.number

# 保存权重为统一名称（而不是 trial 编号）
torch.save(
    torch.load(f"stage1_encoder_trial{best_trial.number}.pt"),
    "stage1_encoder_best.pt"
)

# 保存超参数 dict，包括 trial_number
torch.save(best_params, "stage1_best_params.pt")
print("✅ 已保存最佳 encoder 权重 → stage1_encoder_best.pt")
print("✅ 已保存最佳参数字典    → stage1_best_params.pt")