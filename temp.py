# stage2_graph_pretrain.py

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict, Tuple, Optional
from rdkit import Chem
import shutil
import optuna
from rdkit.Chem import Descriptors

# 复用Stage1的组件
from model import WDMPNN, GraphSSLModel
from data_preparation import load_and_split_data, PolymerDataset, get_data_paths


def weighted_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算加权 MAE：mean(abs(pred - target)) 或者 sum(weight * abs(pred-target)) / sum(weight)
    """
    diff = torch.abs(pred - target)
    if weight is None:
        return diff.mean()
    return (weight * diff).sum() / weight.sum()


def add_pseudo_label(dataset):
    """为数据集添加graph-level伪标签（分子量）"""
    for data in dataset:
        # 如果Data对象中包含'smiles'属性，则使用它；否则，无法从data对象获取smiles，推荐在构造时添加
        smiles = getattr(data, 'smiles', None)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            mol_weight = Descriptors.MolWt(mol) if mol else None
        else:
            mol_weight = None

        # 回退方案：使用原子质量求和
        if mol_weight is None:
            mol_weight = data.x[:, 0].sum().item()
            print(f"Warning: Invalid SMILES '{smiles}', using atom mass sum as pseudo label.")
        data.pseudo = torch.tensor([mol_weight], dtype=torch.float)
    return dataset


def prepare_stage2_data(
    train_df,
    batch_size: int = 64,
    shuffle: bool = True
) -> Tuple[PolymerDataset, DataLoader]:
    """准备Stage2数据（复用Stage1的数据结构）"""
    dataset = PolymerDataset(train_df, y_cols=[])
    # 如果需要Data对象携带SMILES，可在PolymerDataset中添加
    dataset = add_pseudo_label(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader


def create_stage2_model_from_stage1(
    stage1_model_path: str,
    predictor_params: Dict,
    freeze_encoder: bool = True,
    device: torch.device = None
) -> Tuple[GraphSSLModel, Dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(stage1_model_path, map_location=device)
    stage1_params = checkpoint.get("params", {})

    # 保持与Stage1相同的节点/边特征维度
    node_dim = 10
    edge_dim = 4
    hidden_dim = stage1_params.get("hidden_dim")
    num_edge_layers = stage1_params.get("num_edge_layers")

    encoder = WDMPNN(
        node_feat_dim=node_dim,
        edge_feat_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_edge_layers=num_edge_layers
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if freeze_encoder:
        encoder.eval()
    else:
        encoder.train()

    # 构造GraphSSL模型
    mlp_hidden_dims = [predictor_params["predictor_hidden_dim"]] * predictor_params["predictor_num_layers"]
    model = GraphSSLModel(encoder=encoder, mlp_hidden_dims=mlp_hidden_dims).to(device)
    return model, stage1_params


def train_stage2_model(
    model,
    optimizer,
    loader,
    device,
    trial=None,
    max_epochs=100,
    patience=10,
    min_delta=1e-4,
    verbose=True
):
    """
    训练Stage2模型，使用加权MAE作为损失函数
    """
    best_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            # 预测图级伪标签
            pred = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                torch.ones(batch.edge_attr.size(0), device=device),
                batch.batch
            )
            # 使用加权MAE
            target = batch.pseudo.view(-1, 1)
            loss = weighted_mae(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(loader.dataset)
        if trial:
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if verbose:
            print(f"Epoch {epoch}: avg_loss={avg_loss:.4e}, best_loss={best_loss:.4e} (no_improve={epochs_no_improve})")

    # 加载最佳权重
    if best_model:
        model.load_state_dict(best_model)
    return best_loss, model


def save_stage2_artifacts_optuna(study, tmp_dir: str, output_dir: str):
    best_params = study.best_trial.user_attrs.get("full_params", {})
    n = study.best_trial.number
    os.makedirs(output_dir, exist_ok=True)

    # 保存超参和权重
    torch.save(best_params, os.path.join(output_dir, f"stage2_best_params_trial{n}.pt"))
    shutil.move(os.path.join(tmp_dir, f"encoder_trial{n}.pt"), os.path.join(output_dir, "stage2_encoder.pt"))
    shutil.move(os.path.join(tmp_dir, f"predictor_trial{n}.pt"), os.path.join(output_dir, "stage2_predictor.pt"))
    shutil.rmtree(tmp_dir)
    print(f"✅ Best Stage2 model (trial {n}) saved to {output_dir}")


def optimize_stage2(
    train_df,
    stage1_model_path: str,
    study_name="stage2_graph_ssl",
    storage_uri="sqlite:///stage2_optuna.db",
    n_trials=50,
    tmp_dir="tmp_stage2",
    output_dir="stage2_artifacts"
):
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    _, loader = prepare_stage2_data(train_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device: {device}')

    def objective(trial):
        predictor_params = {
            "predictor_hidden_dim": trial.suggest_categorical("predictor_hidden_dim", [64, 128, 256]),
            "predictor_num_layers": trial.suggest_int("predictor_num_layers", 1, 3)
        }

        model, stage1_params = create_stage2_model_from_stage1(
            stage1_model_path=stage1_model_path,
            predictor_params=predictor_params,
            freeze_encoder=True,
            device=device
        )

        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        full_params = dict(stage1_params)
        full_params.update(predictor_params)
        full_params["lr"] = lr
        trial.set_user_attr("full_params", full_params)

        best_loss, _ = train_stage2_model(
            model, optimizer, loader, device,
            trial=trial, max_epochs=100, patience=10
        )

        torch.save(model.encoder.state_dict(), os.path.join(tmp_dir, f"encoder_trial{trial.number}.pt"))
        torch.save(model.predictor.state_dict(), os.path.join(tmp_dir, f"predictor_trial{trial.number}.pt"))
        return best_loss

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        storage=storage_uri,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    save_stage2_artifacts_optuna(study, tmp_dir, output_dir)
    return study


def train_final_stage2_model(
    train_df,
    stage1_model_path: str = "stage1_artifacts/final_stage1_model.pth",
    best_params_path: str = None,
    output_dir: str = "stage2_final_model",
    freeze_encoder: bool = True,
    n_epochs: int = 150,
    patience: int = 15,
    min_delta: float = 1e-4
) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    if best_params_path is None:
        param_files = [f for f in os.listdir("stage2_artifacts") if f.startswith("stage2_best_params")]
        if not param_files:
            raise FileNotFoundError("No stage2_best_params file found in stage2_artifacts/")
        best_params_path = os.path.join("stage2_artifacts", sorted(param_files)[-1])

    params = torch.load(best_params_path)
    print(f"✅ Loaded best stage2 params from {best_params_path}")

    model, _ = create_stage2_model_from_stage1(
        stage1_model_path=stage1_model_path,
        predictor_params=params,
        freeze_encoder=freeze_encoder,
        device=device
    )
    _, loader = prepare_stage2_data(train_df)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    best_loss, model = train_stage2_model(
        model=model,
        optimizer=optimizer,
        loader=loader,
        device=device,
        max_epochs=n_epochs,
        patience=patience,
        min_delta=min_delta,
        trial=None,
        verbose=True
    )

    # 保存模型和超参
    torch.save(params, os.path.join(output_dir, "final_stage2_params.pt"))
    torch.save(model.encoder.state_dict(), os.path.join(output_dir, "final_stage2_encoder.pt"))
    torch.save(model.predictor.state_dict(), os.path.join(output_dir, "final_stage2_predictor.pt"))

    print(f"\n✅ Final Stage2 model trained. Best loss: {best_loss:.4f}")
    print(f"📦 Saved to: {output_dir}")

    return model
