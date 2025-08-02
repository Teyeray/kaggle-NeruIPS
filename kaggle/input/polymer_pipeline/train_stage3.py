#stage3 -fine tune
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Optional
from model import WDMPNN, GraphPredictor
from data_preparation import PolymerDataset

from data_preparation import load_and_split_data, smiles_to_data, get_data_paths

def load_stage3_models(prop: str):
    """加载某个属性的 encoder, predictor, downstream"""
    enc = WDMPNN(9, 4, BEST_PARAMS["hidden_dim"], BEST_PARAMS["num_edge_layers"]).to(DEVICE)
    enc.load_state_dict(torch.load(f"{STAGE3_DIR}/encoder_ft_{prop}.pt", map_location=DEVICE))
    enc.eval()

    pred = GraphPredictor(BEST_PARAMS["hidden_dim"], [BEST_PARAMS["hidden_dim"] // 2], 1).to(DEVICE)
    pred.load_state_dict(torch.load(f"{STAGE3_DIR}/predictor_ft_{prop}.pt", map_location=DEVICE))
    pred.eval()

    down = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(DEVICE)
    down.load_state_dict(torch.load(f"{STAGE3_DIR}/downstream_{prop}.pt", map_location=DEVICE))
    down.eval()

    return enc, pred, down
def prepare_property_datasets(properties, train_df, val_df, test_df):
    """
    加载原始 train/val/test，并为每个属性返回清洗好的 DataFrame。
    返回值示例：
    {
      "Tg": {"train": train_clean_df, "val": val_clean_df, "test": test_clean_df},
      "FFV": { ... },
      ...
    }
    
    Args:
        properties: 要处理的属性列表
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
    """
    result = {}
    for prop in properties:
        train_clean = train_df[["SMILES", prop]].dropna().reset_index(drop=True)
        val_clean   = val_df[  ["SMILES", prop]].dropna().reset_index(drop=True)
        test_clean  = test_df[ ["SMILES", prop]].dropna().reset_index(drop=True)
        result[prop] = {
            "train": train_clean,
            "val":   val_clean,
            "test":  test_clean
        }
    return result
    
def finetune_property(
    train_df,
    val_df,
    property_name: str,
    best_params_path: str,
    stage2_encoder_path: str,
    stage2_predictor_path: str,
    output_dir: str = "stage3_heads",
    device: torch.device = None,
    num_epochs: int = 50,
    batch_size: int = 64,
    patience: int = 10
):
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
    # 设备选择
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # 准备数据集
    train_ds = PolymerDataset(train_df, y_cols=[property_name])
    val_ds   = PolymerDataset(val_df,   y_cols=[property_name])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # 加载最佳超参 + Stage2 encoder/predictor
    best = torch.load(best_params_path)
    encoder = WDMPNN(9, 4, best["hidden_dim"], best["num_edge_layers"]).to(device)
    encoder.load_state_dict(torch.load(stage2_encoder_path, map_location=device))
    predictor = GraphPredictor(best["hidden_dim"], [best["hidden_dim"] // 2], 1).to(device)
    predictor.load_state_dict(torch.load(stage2_predictor_path, map_location=device))

    # 下游 head（图级回归）
    downstream = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    ).to(device)

    # 设置优化器
    optim = torch.optim.Adam(
        list(encoder.parameters()) +
        list(predictor.parameters()) +
        list(downstream.parameters()),
        lr=best.get("lr", 1e-3)
    )

    best_val_mae = float('inf')
    no_improve = 0

    for epoch in range(1, num_epochs+1):
        # —— 训练一步 —— 
        encoder.train(); predictor.train(); downstream.train()

        total_mae = 0.0
        total_mse = 0.0
        for data in train_loader:
            data = data.to(device)
            h = encoder(data.x, data.edge_index, data.edge_attr,
                        torch.ones(data.edge_attr.size(0),device=device),
                        data.batch)
            h = predictor(h).view(-1,1)
            out = downstream(h)
            y = data.y.view(-1,1)

            mae_loss = weighted_mae(out, y)
            mse_loss = F.mse_loss(out, y)

            optim.zero_grad()
            mae_loss.backward()
            optim.step()

            total_mae += mae_loss.item() * data.num_graphs
            total_mse += mse_loss.item() * data.num_graphs

        train_mae = total_mae / len(train_loader.dataset)
        train_mse = total_mse / len(train_loader.dataset)

        # —— 验证集评估 —— 
        encoder.eval(); predictor.eval(); downstream.eval()
        val_total_mae = 0.0
        val_total_mse = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                h = predictor(encoder(data.x, data.edge_index, data.edge_attr,
                                      torch.ones(data.edge_attr.size(0),device=device),
                                      data.batch)).view(-1,1)
                out = downstream(h)
                y = data.y.view(-1,1)

                mae_loss = weighted_mae(out, y)
                mse_loss = F.mse_loss(out, y)

                val_total_mae += mae_loss.item() * data.num_graphs
                val_total_mse += mse_loss.item() * data.num_graphs

        val_mae = val_total_mae / len(val_loader.dataset)
        val_mse = val_total_mse / len(val_loader.dataset)

        print(f"[{property_name}] Epoch {epoch:03d} Train MSE={train_mse:.4f}, MAE={train_mae:.4f}, Val   MSE={val_mse:.4f}, MAE={val_mae:.4f}")

        # —— 早停判断 —— 
        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            no_improve = 0
            # 保存当前最优 head
            best_state = downstream.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement in {patience} epochs)")
                break

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    torch.save(best_state, os.path.join(output_dir, f"downstream_{property_name}.pt"))
    torch.save(encoder.state_dict(), os.path.join(output_dir, f"encoder_ft_{property_name}.pt"))
    torch.save(predictor.state_dict(), os.path.join(output_dir, f"predictor_ft_{property_name}.pt"))
    print(f"✅ Saved head & encoder & predictor for {property_name} in {output_dir}")


def main(paths=None):
    """
    主函数：为所有属性训练微调模型
    
    Args:
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
    """
    # 获取数据路径
    if paths is None:
        paths = get_data_paths()
    
    # 要处理的属性
    properties = ['Tg', 'Tc', 'Density', 'FFV', 'Rg']
    
    # 为每个属性准备数据集
    property_datasets = prepare_property_datasets(properties, paths)
    
    # 为每个属性训练微调模型
    for prop in properties:
        print(f"\n{'='*50}")
        print(f"开始训练 {prop} 属性")
        print(f"{'='*50}")
        
        datasets = property_datasets[prop]
        finetune_property(
            train_df=datasets['train'],
            val_df=datasets['val'],
            property_name=prop
        )
    
    print("\n✅ 所有属性的微调训练完成！")


if __name__ == "__main__":
    main()