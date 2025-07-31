# predict_stage3.py

import os
import torch
import pandas as pd

from data_preparation import smiles_to_data
from model import WDMPNN, GraphPredictor

# 五个属性
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]

# 全局超参和设备
BEST_PARAMS = torch.load("stage1_best_params.pt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stage 3 微调后权重目录
STAGE3_DIR  = "stage3_heads"

# 缓存每个属性的三部分模型
_model_cache = {}

def load_stage3_models(prop: str):
    if prop in _model_cache:
        return _model_cache[prop]

    # 1) encoder
    enc = WDMPNN(
        node_feat_dim=2,
        edge_feat_dim=1,
        hidden_dim=BEST_PARAMS["hidden_dim"],
        num_edge_layers=BEST_PARAMS["num_edge_layers"]
    ).to(DEVICE)
    enc.load_state_dict(torch.load(
        os.path.join(STAGE3_DIR, f"encoder_ft_{prop}.pt"),
        map_location=DEVICE
    ))
    enc.eval()

    # 2) predictor
    pred = GraphPredictor(
        hidden_dim=enc.hidden_dim,
        mlp_hidden_dim=BEST_PARAMS["hidden_dim"] // 2,
        output_dim=1
    ).to(DEVICE)
    pred.load_state_dict(torch.load(
        os.path.join(STAGE3_DIR, f"predictor_ft_{prop}.pt"),
        map_location=DEVICE
    ))
    pred.eval()

    # 3) downstream head
    head = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    ).to(DEVICE)
    head.load_state_dict(torch.load(
        os.path.join(STAGE3_DIR, f"downstream_{prop}.pt"),
        map_location=DEVICE
    ))
    head.eval()

    _model_cache[prop] = (enc, pred, head)
    return enc, pred, head


def predict_for_smiles(smiles: str):
    """
    对单条 SMILES 分别用各属性对应的微调模型预测值。
    返回 dict[prop -> float].
    """
    data = smiles_to_data(smiles, DEVICE)
    results = {}
    with torch.no_grad():
        for prop in PROPERTIES:
            enc, pred, head = load_stage3_models(prop)
            h = enc(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.edge_weight.to(DEVICE),
                data.batch.to(DEVICE)
            )
            h = pred(h).view(-1, 1)
            results[prop] = head(h).item()
    return results


if __name__ == "__main__":
    # 1) 读取 test.csv
    test_csv = os.path.join("neurips-open-polymer-prediction-2025", "test.csv")
    df_test  = pd.read_csv(test_csv, dtype={"id": str})

    # 2) 对每条 SMILES 预测
    out_records = []
    for _, row in df_test.iterrows():
        _id, smi = row["id"], row["SMILES"]
        try:
            preds = predict_for_smiles(smi)
        except Exception:
            # 若解析或推理失败，则填 NaN
            preds = {p: float("nan") for p in PROPERTIES}
        rec = {"id": _id}
        rec.update(preds)
        out_records.append(rec)

    # 3) 输出 submission.csv
    df_out = pd.DataFrame(out_records, columns=["id"] + PROPERTIES)
    df_out.to_csv("submission.csv", index=False)
    print("Saved submission.csv with", len(df_out), "rows.")