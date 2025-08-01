# stage2_graph_pretrain.py

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import WDMPNN, GraphSSLModel
from data_preparation import load_and_split_data, PolymerDataset, get_data_paths

from rdkit import Chem
from rdkit.Chem import Descriptors

def add_pseudo_label(dataset):
    """
    对每个 Data 计算 graph-level pseudo label：ensemble molecular weight
    并动态添加 data.pseudo 属性
    """
    for data in dataset:
        smiles = data.smiles if hasattr(data, 'smiles') else None
        if smiles is None:
            # 临时 fallback，用原子质量粗略估算
            mol_weight = data.x[:, 0].sum().item()
        else:
            mol = Chem.MolFromSmiles(smiles)
            mol_weight = Descriptors.MolWt(mol) if mol is not None else data.x[:, 0].sum().item()
        data.pseudo = torch.tensor([mol_weight], dtype=torch.float)

def load_stage1_models(
    best_params_path: str = "stage1_best_params.pt",
    encoder_weights_path: str = None,
    device: torch.device = None
):
    """
    加载 Stage1 训练好的模型和超参。

    Args:
        best_params_path: 包含 best 参数的 .pt 文件路径。
        encoder_weights_path: encoder 权重文件路径，若为 None，则从 best_params 中拼接。
        device: 计算设备，默认自动选择 cuda/CPU。

    Returns:
        encoder: 加载好的 WDMPNN 模型（eval 模式）。
        best_params: 从 best_params_path 读取的超参 dict。
    """
    # 设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取超参
    best_params = torch.load(best_params_path, map_location=device)

    # 如果没有单独提供 encoder 权重路径，就拼一个默认名字
    if encoder_weights_path is None:
        trial = best_params.get("trial_number")
        if trial is not None:
            encoder_weights_path = f"stage1_encoder_trial{trial}.pt"
        else:
            encoder_weights_path = "stage1_encoder_best.pt"

    # 检查文件存在
    if not os.path.exists(encoder_weights_path):
        raise FileNotFoundError(f"Encoder weights not found: {encoder_weights_path}")

    # 构建模型并加载
    encoder = WDMPNN(
        node_feat_dim=2,
        edge_feat_dim=1,
        hidden_dim=best_params["hidden_dim"],
        num_edge_layers=best_params["num_edge_layers"]
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    encoder.eval()

    return encoder, best_params

def prepare_stage2_data(paths=None, batch_size=64):
    """
    准备Stage2训练所需的数据
    
    Args:
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
        batch_size: 批处理大小
    
    Returns:
        tuple: (train_df, base_graph_ds, graph_loader)
    """
    if paths is None:
        paths = get_data_paths()
    
    # 加载并划分，只取 train part，用于 SSL 预训练
    train_df, _, _ = load_and_split_data(paths)
    base_graph_ds = PolymerDataset(train_df, y_cols=[])   # 不需要 y

    # 添加 pseudo label（ensemble molecular weight）
    add_pseudo_label(base_graph_ds)

    graph_loader = DataLoader(base_graph_ds, batch_size=batch_size, shuffle=True)
    
    return train_df, base_graph_ds, graph_loader

def create_stage2_model(encoder, best_params, device=None):
    """
    创建Stage2模型

    Args:
        encoder: Stage1训练好的encoder
        best_params: Stage1最佳参数
        device: 计算设备
    
    Returns:
        tuple: (model, optimizer)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建 GraphSSLModel（不 freeze encoder）
    model = GraphSSLModel(encoder, mlp_hidden_dim=best_params["hidden_dim"] // 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    
    return model, optimizer

def train_stage2_epoch(model, optimizer, graph_loader, device=None):
    """
    训练一个epoch
    
    Args:
        model: Stage2模型
        optimizer: 优化器
        graph_loader: 数据加载器
        device: 计算设备
    
    Returns:
        float: 平均损失
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.train()
    total_loss = 0
    
    for data in graph_loader:
        data = data.to(device)
        pseudo = data.pseudo.to(device).view(-1, 1)  # [batch_size,1]
        pred = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            torch.ones(data.edge_attr.size(0), device=device),
            data.batch
        )
        loss = F.mse_loss(pred, pseudo)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(graph_loader.dataset)
    return avg_loss

def save_stage2_models(encoder, model, encoder_path="stage2_encoder.pt", predictor_path="stage2_predictor.pt"):
    """
    保存Stage2训练好的模型
    
    Args:
        encoder: 训练好的encoder
        model: 训练好的模型
        encoder_path: encoder保存路径
        predictor_path: predictor保存路径
    """
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(model.predictor.state_dict(), predictor_path)
    print(f"✅ Stage2模型已保存: {encoder_path}, {predictor_path}")

def train_stage2(paths=None, n_epochs=20, batch_size=64, device=None, 
                encoder_path="stage2_encoder.pt", predictor_path="stage2_predictor.pt"):
    """
    完整的Stage2训练流程
    
    Args:
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
        n_epochs: 训练轮数
        batch_size: 批处理大小
        device: 计算设备
        encoder_path: encoder保存路径
        predictor_path: predictor保存路径
    
    Returns:
        tuple: (encoder, model, best_params)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 开始Stage2训练，设备: {device}")
    
    # 1. 准备数据
    print("📦 准备数据...")
    train_df, base_graph_ds, graph_loader = prepare_stage2_data(paths, batch_size)
    print(f"   数据准备完成，样本数: {len(train_df)}")
    
    # 2. 加载Stage1模型
    print("🔧 加载Stage1模型...")
    encoder, best_params = load_stage1_models(device)
    print("   Stage1模型加载完成")
    
    # 3. 创建Stage2模型
    print("🏗️ 创建Stage2模型...")
    model, optimizer = create_stage2_model(encoder, best_params, device)
    print("   Stage2模型创建完成")
    
    # 4. 训练循环
    print(f"🎯 开始训练，共{n_epochs}轮...")
    for epoch in range(1, n_epochs + 1):
        avg_loss = train_stage2_epoch(model, optimizer, graph_loader, device)
        print(f"   [Stage2] Epoch {epoch:02d}/{n_epochs}  loss={avg_loss:.4f}")
    
    # 5. 保存模型
    print("💾 保存模型...")
    save_stage2_models(encoder, model, encoder_path, predictor_path)
    
    print("✅ Stage2训练完成！")
    return encoder, model, best_params

def main(paths=None, n_epochs=20):
    """
    主函数（向后兼容）
    
    Args:
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
        n_epochs: 训练轮数
    """
    return train_stage2(paths, n_epochs)

if __name__ == "__main__":
    main()