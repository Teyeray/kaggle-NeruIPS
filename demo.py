"""
聚合物性质预测系统演示
展示系统的主要功能
"""

import torch
import numpy as np
from main_pipeline import PolymerPropertyPredictor
from enhanced_gnn import create_enhanced_graph_from_smiles

def demo_basic_functionality():
    """演示基本功能"""
    print("🎯 聚合物性质预测系统演示")
    print("="*60)
    
    # 1. 创建预测器
    print("1️⃣ 创建预测器...")
    predictor = PolymerPropertyPredictor()
    print("✅ 预测器创建成功")
    
    # 2. 显示模型信息
    print("\n2️⃣ 模型信息...")
    predictor.get_model_summary()
    
    # 3. 测试图创建
    print("\n3️⃣ 测试分子图创建...")
    test_smiles = [
        "CCO",           # 乙醇
        "c1ccccc1",      # 苯
        "CC(C)(C)c1ccc(N)cc1",  # 对叔丁基苯胺
        "C1=CC=C(C=C1)C(=O)O",  # 苯甲酸
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # 布洛芬
    ]
    
    for i, smiles in enumerate(test_smiles, 1):
        graph = create_enhanced_graph_from_smiles(smiles)
        if graph is not None:
            print(f"   {i}. {smiles}: {graph.x.shape[0]}个原子, {graph.edge_index.shape[1]}条边")
        else:
            print(f"   {i}. {smiles}: 创建失败")
    
    # 4. 测试模型预测
    print("\n4️⃣ 测试模型预测...")
    device = torch.device('cpu')
    model = predictor.model.to(device)
    
    for i, smiles in enumerate(test_smiles, 1):
        graph = create_enhanced_graph_from_smiles(smiles)
        if graph is not None:
            graph = graph.to(device)
            with torch.no_grad():
                node_out, edge_out, graph_out = model(graph.x, graph.edge_index, graph.edge_attr)
            print(f"   {i}. {smiles}: 预测值 = {graph_out.item():.4f}")
    
    print("\n✅ 演示完成！")
    print("\n📝 使用说明:")
    print("   - 运行 'python main_pipeline.py' 进行快速演示")
    print("   - 运行 'python main_pipeline.py full' 进行完整训练")
    print("   - 运行 'python usage_example.py' 查看更多示例")
    print("   - 运行 'python test_system.py' 进行系统测试")

if __name__ == "__main__":
    demo_basic_functionality() 