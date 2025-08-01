import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

from model import WDMPNN, GraphPredictor
from data_preparation import PolymerDataset, get_data_paths, smiles_to_data


def load_trained_models(properties, model_dir="stage3_heads"):
    """
    加载所有训练好的模型
    
    Args:
        properties: 要预测的属性列表
        model_dir: 模型文件目录
    
    Returns:
        dict: 包含所有模型的字典
    """
    models = {}
    
    for prop in properties:
        # 加载stage1最佳参数
        best_params = torch.load("stage1_best_params.pt")
        
        # 创建encoder
        encoder = WDMPNN(
            node_feat_dim=2,
            edge_feat_dim=1,
            hidden_dim=best_params["hidden_dim"],
            num_edge_layers=best_params["num_edge_layers"]
        )
        
        # 加载stage2 encoder权重
        encoder.load_state_dict(torch.load("stage2_encoder.pt"))
        
        # 创建predictor
        predictor = GraphPredictor(
            hidden_dim=encoder.hidden_dim,
            mlp_hidden_dim=best_params["hidden_dim"] // 2,
            output_dim=1
        )
        
        # 加载stage2 predictor权重
        predictor.load_state_dict(torch.load("stage2_predictor.pt"))
        
        # 加载stage3微调后的模型
        encoder.load_state_dict(torch.load(os.path.join(model_dir, f"encoder_ft_{prop}.pt")))
        predictor.load_state_dict(torch.load(os.path.join(model_dir, f"predictor_ft_{prop}.pt")))
        
        # 加载downstream head
        downstream = torch.load(os.path.join(model_dir, f"downstream_{prop}.pt"))
        
        models[prop] = {
            'encoder': encoder,
            'predictor': predictor,
            'downstream': downstream
        }
    
    return models


def predict_single_smiles(smiles, models, device=None):
    """
    对单个SMILES进行预测
    
    Args:
        smiles: SMILES字符串
        models: 训练好的模型字典
        device: 计算设备
    
    Returns:
        dict: 预测结果
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 转换为图数据
    try:
        data = smiles_to_data(smiles, device)
    except:
        # 如果SMILES无效，返回NaN
        return {prop: np.nan for prop in models.keys()}
    
    results = {}
    
    for prop, model_dict in models.items():
        encoder = model_dict['encoder'].to(device).eval()
        predictor = model_dict['predictor'].to(device).eval()
        downstream = model_dict['downstream'].to(device).eval()
        
        with torch.no_grad():
            # 前向传播
            h = encoder(
                data.x, 
                data.edge_index, 
                data.edge_attr,
                torch.ones(data.edge_attr.size(0), device=device),
                data.batch
            )
            h = predictor(h).view(-1, 1)
            pred = downstream(h)
            
            results[prop] = pred.item()
    
    return results


def predict_test_set(test_csv_path, models, device=None, batch_size=64):
    """
    对整个测试集进行预测
    
    Args:
        test_csv_path: 测试集CSV文件路径
        models: 训练好的模型字典
        device: 计算设备
        batch_size: 批处理大小
    
    Returns:
        pd.DataFrame: 包含预测结果的DataFrame
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载测试数据
    test_df = pd.read_csv(test_csv_path)
    print(f"加载测试集，共 {len(test_df)} 个样本")
    
    # 创建测试数据集
    test_dataset = PolymerDataset(test_df, y_cols=[])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化预测结果
    predictions = {prop: [] for prop in models.keys()}
    
    # 批量预测
    for batch in test_loader:
        batch = batch.to(device)
        
        for prop, model_dict in models.items():
            encoder = model_dict['encoder'].to(device).eval()
            predictor = model_dict['predictor'].to(device).eval()
            downstream = model_dict['downstream'].to(device).eval()
            
            with torch.no_grad():
                h = encoder(
                    batch.x, 
                    batch.edge_index, 
                    batch.edge_attr,
                    torch.ones(batch.edge_attr.size(0), device=device),
                    batch.batch
                )
                h = predictor(h).view(-1, 1)
                pred = downstream(h)
                
                predictions[prop].extend(pred.cpu().numpy().flatten())
    
    # 创建结果DataFrame
    result_df = test_df[['SMILES']].copy()
    for prop in models.keys():
        result_df[prop] = predictions[prop]
    
    return result_df


def main(paths=None, output_path="submission.csv"):
    """
    主函数：加载模型并进行预测
    
    Args:
        paths: 数据路径字典，如果为None则使用get_data_paths()获取
        output_path: 预测结果保存路径
    """
    # 获取数据路径
    if paths is None:
        paths = get_data_paths()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 要预测的属性
    properties = ['Tg', 'Tc', 'Density', 'FFV', 'Rg']
    
    # 加载训练好的模型
    print("加载训练好的模型...")
    models = load_trained_models(properties)
    print("✅ 模型加载完成")
    
    # 对测试集进行预测
    print("开始预测测试集...")
    predictions = predict_test_set(paths['test_csv'], models, device)
    
    # 保存预测结果
    predictions.to_csv(output_path, index=False)
    print(f"✅ 预测结果已保存到 {output_path}")
    
    # 显示预测结果统计
    print("\n预测结果统计:")
    for prop in properties:
        if prop in predictions.columns:
            print(f"{prop}: 均值={predictions[prop].mean():.4f}, 标准差={predictions[prop].std():.4f}")
    
    return predictions


if __name__ == "__main__":
    main() 