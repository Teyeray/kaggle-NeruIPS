# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool

class WDMPNN(MessagePassing):
    """
    Weighted Directed Message Passing Neural Network (wD-MPNN)
    as in MSDE paper §2.2.
    Performs edge-centered message passing and node update.
    """
    def __init__(self,
                 node_feat_dim: int,
                 edge_feat_dim: int,
                 hidden_dim: int,
                 num_edge_layers: int = 3):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.num_edge_layers = num_edge_layers

        # 初始变换
        self.W_i = nn.Linear(node_feat_dim + edge_feat_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(node_feat_dim + hidden_dim, hidden_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_weight: torch.Tensor,
                batch: torch.Tensor = None):
        """
        x: [N, node_feat_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_feat_dim]
        edge_weight: [E]  # stochastic weights
        batch: [N] node->graph index
        """
        src, dst = edge_index

        # 1) 初始化边特征 h^0_vu
        h = torch.cat([x[src], edge_attr], dim=-1)             # [E, node+edge]
        h = F.relu(self.W_i(h))                                # [E, hidden]

        # 2) 边级消息传递 L 层
        for _ in range(self.num_edge_layers):
            m = torch.zeros_like(h)
            # 将来自同一 dst 的所有边消息累加
            m = m.index_add(0, dst, edge_weight.unsqueeze(-1) * self.W_h(h))
            h = F.relu(h + m)

        # 3) 节点更新：对每个节点 v，聚合所有入边 h_{*->v}
        agg = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        agg = agg.index_add(0, dst, edge_weight.unsqueeze(-1) * h)
        node_input = torch.cat([x, agg], dim=-1)
        node_hidden = F.relu(self.W_o(node_input))            # [N, hidden]

        # 4) 图级池化
        if batch is not None:
            graph_repr = global_add_pool(node_hidden, batch)  # [batch_size, hidden]
            return graph_repr
        else:
            return node_hidden.sum(dim=0, keepdim=True)      # 单图情况

class NodeEdgeSSLModel(nn.Module):
    """
    Node- & Edge-level SSL 模型 (§2.3.1)。
    使用 wD-MPNN 的消息传递部分来生成节点/边的隐藏向量，
    然后用两个 MLP head 重建被 mask 的原始特征。
    """
    def __init__(self,
                 encoder: WDMPNN,
                 node_feat_dim: int,
                 edge_feat_dim: int):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.hidden_dim
        # 用于节点特征重建的 head
        self.node_predictor = nn.Linear(hidden, node_feat_dim)
        # 用于边特征重建的 head
        self.edge_predictor = nn.Linear(hidden, edge_feat_dim)

    def forward(self,
                x_masked: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr_masked: torch.Tensor,
                batch: torch.Tensor):
        # 如果没有 edge_weight，默认全 1
        E = edge_attr_masked.size(0)
        edge_weight = torch.ones(E, device=x_masked.device)

        # 重用 WDMPNN 的消息传递逻辑
        src, dst = edge_index
        h = torch.cat([x_masked[src], edge_attr_masked], dim=-1)
        h = F.relu(self.encoder.W_i(h))

        for _ in range(self.encoder.num_edge_layers):
            m = torch.zeros_like(h)
            m = m.index_add(0, dst, edge_weight.unsqueeze(-1) * self.encoder.W_h(h))
            h = F.relu(h + m)

        agg = torch.zeros(x_masked.size(0), self.encoder.hidden_dim, device=x_masked.device)
        agg = agg.index_add(0, dst, edge_weight.unsqueeze(-1) * h)
        node_input = torch.cat([x_masked, agg], dim=-1)
        node_hidden = F.relu(self.encoder.W_o(node_input))

        # 预测被 mask 的原始特征
        node_pred = self.node_predictor(node_hidden)    # [N, node_feat_dim]
        edge_pred = self.edge_predictor(h)              # [E, edge_feat_dim]
        return node_pred, edge_pred

class GraphPredictor(nn.Module):
    """
    图级 MLP 预测头，结构可调用于 Optuna 超参搜索。
    
    Args:
        hidden_dim (int): 输入维度（来自 encoder）
        mlp_hidden_dims (List[int]): 每一层的隐藏维度列表
        output_dim (int): 输出维度（默认 1）
    """
    def __init__(self,
                 hidden_dim: int,
                 mlp_hidden_dims: list,
                 output_dim: int = 1):
        super().__init__()

        layers = []
        in_dim = hidden_dim
        for hidden in mlp_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden

        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, graph_repr: torch.Tensor):
        return self.mlp(graph_repr)

class GraphSSLModel(nn.Module):
    """
    Graph-level 自监督模型 (§2.3.2)。
    1) 用 WDMPNN 得到图表征
    2) 用 GraphPredictor 预测伪标签（e.g. ensemble molecular weight）
    """
    def __init__(self,
                 encoder: WDMPNN,
                 mlp_hidden_dims: list):
        super().__init__()
        self.encoder = encoder
        self.predictor = GraphPredictor(
            hidden_dim=encoder.hidden_dim,
            mlp_hidden_dims=mlp_hidden_dims,  
            output_dim=1
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_weight: torch.Tensor,
                batch: torch.Tensor):
        graph_repr = self.encoder(x, edge_index, edge_attr, edge_weight, batch)
        return self.predictor(graph_repr)