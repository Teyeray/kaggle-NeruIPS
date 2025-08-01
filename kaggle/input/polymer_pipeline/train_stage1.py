import optuna
import torch
import shutil
from typing import Dict, Tuple
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os

from model import WDMPNN, NodeEdgeSSLModel
from data_preparation import PolymerDataset, NodeEdgeMaskDataset

def setup_stage1_data(train_df):
    """
    设置Stage1训练所需的数据
    
    Args:
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
    
    Returns:
        tuple: (base_graph_ds, dataset_nodeedge, nodeedge_loader, device)
    """

    base_graph_ds = PolymerDataset(train_df, y_cols=[])                # y_cols=[] 表示不生成 y

    # 2) 构造 Node/Edge SSL 数据集和 DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_nodeedge = NodeEdgeMaskDataset(base_graph_ds, device=device)
    nodeedge_loader = DataLoader(dataset_nodeedge, batch_size=64, shuffle=True)
    
    return base_graph_ds, dataset_nodeedge, nodeedge_loader, device

def create_stage1_model(params, device):
    """
    根据给定的参数创建Stage1模型
    
    Args:
        params: 包含模型超参的字典
        device: 计算设备
    
    Returns:
        tuple: (encoder, model)
    """
    hidden_dim = params["hidden_dim"]
    n_layers = params["num_edge_layers"]
    
    encoder = WDMPNN(2, 1, hidden_dim, n_layers).to(device)
    model = NodeEdgeSSLModel(encoder, node_feat_dim=2, edge_feat_dim=1).to(device)

    return encoder, model

def train_stage1_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device,
    lr: float,
    max_epochs: int = 100,
    patience: int = 15,
    min_delta: float = 1e-3,
    trial: optuna.Trial = None,
) -> Tuple[float, torch.nn.Module]:
    """
    训练Stage1模型
    
    Args:
        model: 要训练的模型
        dataloader: 数据加载器
        device: 计算设备
        lr: 学习率
        max_epochs: 最大训练轮次
        patience: 早停耐心值
        min_delta: 最小改进阈值
        trial: Optuna Trial对象(可选)
    
    Returns:
        tuple: (best_loss, best_model)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_model = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
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

        avg_loss = total_loss / len(dataloader.dataset)
        
        # 报告给Optuna(如果使用)
        if trial:
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 早停和保存逻辑
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_model = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_model:
        model.load_state_dict(best_model)
    
    return best_loss, model

def optimize_stage1(
    train_df,
    study_name: str = "stage1_nodeedge_ssl",
    storage_uri: str = "sqlite:///stage1_optuna.db",
    n_trials: int = 50,
    patience: int = 15,
    tmp_dir: str = "tmp_stage1",
    output_dir: str = "stage1_artifacts"
) -> optuna.Study:
    """
    执行Stage1的超参数优化
    
    Args:
        train_df: 训练数据
        study_name: Optuna study名称
        storage_uri: Optuna存储URI
        n_trials: 优化试验次数
        patience: 早停耐心值
        tmp_dir: 临时目录
        output_dir: 输出目录
    
    Returns:
        optuna.Study: 完成的study对象
    """
    # 初始化数据和目录
    _, _, nodeedge_loader, device = setup_stage1_data(train_df)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义Optuna目标函数
    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 256]),
            "num_edge_layers": trial.suggest_int("num_edge_layers", 2, 8)
        }
        
        # 创建和训练模型
        encoder, model = create_stage1_model(params, device)
        best_loss, _ = train_stage1_model(
            model=model,
            dataloader=nodeedge_loader,
            device=device,
            lr=params["lr"],
            max_epochs=200,
            patience=patience,
            trial=trial
        )
        
        # 保存当前trial的最佳encoder
        if best_loss < float('inf'):
            torch.save(
                encoder.state_dict(),
                os.path.join(tmp_dir, f"encoder_trial{trial.number}.pt")
            )
        
        return best_loss
    
    # 运行Optuna优化
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage_uri,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    
    # 保存最佳结果
    save_stage1_artifacts(study, tmp_dir, output_dir)
    
    return study

def save_stage1_artifacts(
    study: optuna.Study,
    tmp_dir: str = "tmp_stage1",
    output_dir: str = "stage1_artifacts"
):
    """
    保存最佳trial的结果
    
    Args:
        study: Optuna study对象
        tmp_dir: 临时文件目录
        output_dir: 输出目录
    """
    n = study.best_trial.number
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 保存最佳参数
        best_params = study.best_trial.params.copy()
        best_params["trial_number"] = n
        param_path = os.path.join(output_dir, f"stage1_best_params_trial{n}.pt")
        torch.save(best_params, param_path)

        # 移动最佳encoder（确保源文件存在）
        src = os.path.join(tmp_dir, f"encoder_trial{n}.pt")
        dst = os.path.join(output_dir, f"stage1_encoder_best_trial{n}.pt")
        
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source encoder file not found: {src}")
            
        shutil.move(src, dst)
        print(f"✅ Saved best artifacts to {output_dir}:")
        print(f"   - Parameters: {param_path}")
        print(f"   - Encoder: {dst}")

    except Exception as e:
        print(f"❌ Error saving artifacts: {str(e)}")
        raise

    finally:
        # 确保清理临时目录（即使前面出错）
        if os.path.exists(tmp_dir):
            try:
                # 先检查是否还有其他需要保留的文件
                remaining_files = os.listdir(tmp_dir)
                if remaining_files:
                    print(f"⚠️ Found {len(remaining_files)} remaining files in {tmp_dir}, cleaning up...")
                
                shutil.rmtree(tmp_dir)
                print(f"✅ Cleaned up temporary directory: {tmp_dir}")
                
            except Exception as cleanup_error:
                print(f"⚠️ Failed to clean up {tmp_dir}: {str(cleanup_error)}")

def train_final_stage1_model(
    train_df,
    params: Dict = None,
    model_path: str = None,
    output_path: str = "final_stage1_model.pth",
    n_epochs: int = 200,
    patience: int = 20
) -> torch.nn.Module:
    """
    使用最佳参数训练最终模型
    
    Args:
        train_df: 训练数据
        params: 参数字典(如果None则从保存的文件加载)
        model_path: 预训练模型路径(可选)
        output_path: 最终模型保存路径
        n_epochs: 训练轮次
        patience: 早停耐心值
    
    Returns:
        训练好的模型
    """
    # 加载数据和参数
    _, _, nodeedge_loader, device = setup_stage1_data(train_df)
    
    if params is None:
        # 查找最新的参数文件
        param_files = [f for f in os.listdir("stage1_artifacts") if f.startswith("stage1_best_params")]
        if not param_files:
            raise FileNotFoundError("No parameter files found in stage1_artifacts")
        latest_file = sorted(param_files)[-1]
        params = torch.load(os.path.join("stage1_artifacts", latest_file))
    
    # 创建和训练模型
    encoder, model = create_stage1_model(params, device)
    
    if model_path:
        encoder.load_state_dict(torch.load(model_path))
    
    best_loss, model = train_stage1_model(
        model=model,
        dataloader=nodeedge_loader,
        device=device,
        lr=params["lr"],
        max_epochs=n_epochs,
        patience=patience
    )
    
    # 保存最终模型
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "model_state_dict": model.state_dict(),
        "params": params
    }, output_path)
    
    print(f"✅ Final model trained with loss {best_loss:.4f}, saved to {output_path}")
    return model