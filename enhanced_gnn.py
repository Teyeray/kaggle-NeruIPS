"""
增强的GNN模型
在保持自监督框架的基础上优化网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch):
        # 简化的注意力机制
        # 计算Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 计算注意力分数（简化版本）
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, v)
        out = self.out_proj(out)
        
        return out

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(x + self.layers(x))

class EnhancedWDMPNN(MessagePassing):
    """增强的wD-MPNN模型"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__(aggr='mean')
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 边编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(5, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多层消息传递
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            # 消息传递层
            self.message_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            
            # 节点更新层
            self.update_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            
            # 注意力层
            self.attention_layers.append(MultiHeadAttention(hidden_channels, num_heads, dropout))
            
            # 残差块
            self.residual_blocks.append(ResidualBlock(hidden_channels, dropout))
        
        # 输出头
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, in_channels)
        )
        
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 5)
        )
        
        # 多尺度图池化
        self.graph_pool_mean = global_mean_pool
        self.graph_pool_max = global_max_pool
        self.graph_pool_add = global_add_pool
        
        self.graph_decoder = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_channels * num_layers, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        # 编码
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # 存储每层的输出用于特征融合
        layer_outputs = []
        
        # 多层消息传递
        for i in range(self.num_layers):
            # 设置当前层索引
            self.current_layer = i
            
            # 消息传递
            x_message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            
            # 注意力机制
            x_attn = self.attention_layers[i](x, edge_index, batch)
            
            # 更新节点特征
            x_updated = self.update_layers[i](torch.cat([x, x_message], dim=-1))
            
            # 残差连接
            x = self.residual_blocks[i](x_updated + x_attn)
            
            # 存储当前层输出
            layer_outputs.append(x)
        
        # 特征融合
        if len(layer_outputs) > 1:
            x_fused = torch.cat(layer_outputs, dim=-1)
            x_final = self.feature_fusion(x_fused)
        else:
            x_final = layer_outputs[0]
        
        # 多任务输出
        node_out = self.node_decoder(x_final)
        
        # 边特征重建
        edge_out = self.edge_decoder(torch.cat([
            x_final[edge_index[0]], 
            x_final[edge_index[1]]
        ], dim=-1))
        
        # 图级别预测
        if batch is not None:
            # 多尺度池化
            x_mean = self.graph_pool_mean(x_final, batch)
            x_max = self.graph_pool_max(x_final, batch)
            x_add = self.graph_pool_add(x_final, batch)
            
            # 拼接多尺度特征
            graph_features = torch.cat([x_mean, x_max, x_add], dim=-1)
            graph_out = self.graph_decoder(graph_features)
            return node_out, edge_out, graph_out
        else:
            # 单个图的情况
            x_mean = x_final.mean(dim=0, keepdim=True)
            x_max = x_final.max(dim=0, keepdim=True)[0]
            x_add = x_final.sum(dim=0, keepdim=True)
            
            graph_features = torch.cat([x_mean, x_max, x_add], dim=-1)
            graph_out = self.graph_decoder(graph_features)
            return node_out, edge_out, graph_out

    def message(self, x_j, edge_attr):
        # x_j: 邻居节点特征
        # edge_attr: 边特征
        return self.message_layers[self.current_layer](torch.cat([x_j, edge_attr], dim=-1))

class GraphTransformer(nn.Module):
    """图Transformer模型作为备选"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(5, hidden_channels)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=num_heads,
                dim_feedforward=hidden_channels * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 输出头
        self.graph_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        # 编码
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # 构建邻接矩阵
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # 应用Transformer层
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=None)
        
        # 图级别预测
        if batch is not None:
            # 池化
            graph_features = global_mean_pool(x, batch)
        else:
            graph_features = x.mean(dim=0, keepdim=True)
        
        return self.graph_decoder(graph_features)

def create_enhanced_graph_from_smiles(smiles):
    """创建增强的图数据"""
    from rdkit import Chem
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    
    # 增强的原子特征
    def get_enhanced_atom_features(atom):
        features = [
            atom.GetAtomicNum(),
            int(atom.GetIsAromatic()),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumImplicitHs(),
            int(atom.IsInRing()),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.GetTotalNumHs()
        ]
        return features
    
    # 增强的边特征
    def get_enhanced_bond_features(bond):
        return [
            bond.GetBondTypeAsDouble(),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
            bond.GetBondDir().real
        ]
    
    # 节点特征
    x = torch.tensor([get_enhanced_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    
    # 边索引和特征
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # 双向边
        edge_index.extend([[i, j], [j, i]])  
        edge_attr.extend([get_enhanced_bond_features(bond)] * 2)
    
    return Data(
        x=x,
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    test_smiles = ["CCO", "c1ccccc1"]
    graphs = [create_enhanced_graph_from_smiles(s) for s in test_smiles]
    graphs = [g for g in graphs if g is not None]
    
    if graphs:
        batch = Batch.from_data_list(graphs).to(device)
        
        # 测试增强模型
        model = EnhancedWDMPNN(
            in_channels=10,  # 增强的原子特征维度
            hidden_channels=256,
            out_channels=1,
            num_layers=4,
            num_heads=8
        ).to(device)
        
        node_out, edge_out, graph_out = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        
        print(f"节点输出形状: {node_out.shape}")
        print(f"边输出形状: {edge_out.shape}")
        print(f"图输出形状: {graph_out.shape}") 