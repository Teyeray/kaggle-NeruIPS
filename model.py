import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool

class WDMPNN(MessagePassing):
    """
    Weighted Directed Message Passing Neural Network (wD-MPNN) as in MSDE paper.
    Implements edge-centered message passing and node update.
    """
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_edge_layers=3):
        super().__init__(aggr='add')  # aggregate messages by weighted sum
        self.hidden_dim = hidden_dim
        self.num_edge_layers = num_edge_layers
        
        # Initial transforms
        self.W_i = nn.Linear(node_feat_dim + edge_feat_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(node_feat_dim + hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, edge_weight, batch=None):
        """
        x: [N, node_feat_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_feat_dim]
        edge_weight: [E] stochastic weights w_{vu}
        batch: assignment of nodes to graphs
        """
        # Prepare initial edge representations h0_vu
        src, dst = edge_index
        h = torch.cat([x[src], edge_attr], dim=-1)
        h = F.relu(self.W_i(h))  # [E, hidden_dim]
        
        # Edge-centered message passing
        for _ in range(self.num_edge_layers):
            # message: weighted sum over incoming edges to each dst node, excluding self-edge
            m = torch.zeros_like(h)
            # aggregate messages: for each edge (k->v), use h[k->v]*w_{kv}
            m = m.index_add(0, dst, edge_weight.unsqueeze(-1) * self.W_h(h))
            # update edge features
            h = F.relu(h + m)
        
        # Node update: for each node v, sum incoming edges
        agg = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        agg = agg.index_add(0, dst, edge_weight.unsqueeze(-1) * h)
        node_input = torch.cat([x, agg], dim=-1)
        node_hidden = F.relu(self.W_o(node_input))
        
        # Graph pooling
        if batch is not None:
            graph_repr = global_add_pool(node_hidden, batch)
            return graph_repr
        else:
            # single graph case: sum all nodes
            return node_hidden.sum(dim=0, keepdim=True)

class GraphPredictor(nn.Module):
    """
    MLP predictor for graph-level task.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, graph_repr):
        return self.mlp(graph_repr)

# Example usage:
# encoder = WDMPNN(node_feat_dim=..., edge_feat_dim=..., hidden_dim=300, num_edge_layers=3)
# predictor = GraphPredictor(input_dim=300, hidden_dim=128)
# graph_repr = encoder(x, edge_index, edge_attr, edge_weight, batch)
# out = predictor(graph_repr)
