import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import Set2Set

class ConditionIntegrationModule(nn.Module):
    def __init__(self, graph_dim, cond_dim, out_dim):
        super().__init__()
        
        # Project graph features to match output dimension
        self.graph_proj = nn.Linear(graph_dim, out_dim)
        
        # Condition encoder
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim * 2, out_dim)
        )
        
        # Cross attention (now both inputs are out_dim)
        self.cross_attention = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)
        
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, graph_feat, cond_feat):
        # Project graph features
        graph_proj = self.graph_proj(graph_feat)  # [batch, out_dim]
        
        # Encode conditions
        cond_encoded = self.cond_net(cond_feat)  # [batch, out_dim]
        
        # Cross attention: graph attends to conditions
        graph_attended, _ = self.cross_attention(
            graph_proj.unsqueeze(1),      # query: [batch, 1, out_dim]
            cond_encoded.unsqueeze(1),    # key: [batch, 1, out_dim]
            cond_encoded.unsqueeze(1)     # value: [batch, 1, out_dim]
        )
        
        graph_attended = graph_attended.squeeze(1)  # [batch, out_dim]
        
        # Residual connection and layer norm
        out = self.layer_norm(graph_proj + graph_attended)
        
        return out
    
class SuzukiYieldGNN(nn.Module):
    def __init__(self, node_dim, edge_dim=6, cond_dim=32, hidden_dim=256, dropout=0.2):
        super().__init__()
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # GAT layers (now 4 layers)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim)
        self.gat3 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim)
        self.gat4 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim)



        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 256
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        
        # Pooling
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        
        # Condition integration
        self.cond_integration = ConditionIntegrationModule(
            hidden_dim * 2, cond_dim, hidden_dim
        )
        
        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        conditions = data.conditions
        
        # Encode nodes
        x = self.node_encoder(x)
        
        # Encode edges
        if edge_attr is not None and edge_attr.shape[1] > 0:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None
        
        # GAT layers with residual connections
        x1 = F.elu(self.bn1(self.gat1(x, edge_index, edge_attr)))
        x2 = F.elu(self.bn2(self.gat2(x1, edge_index, edge_attr)))
        x3 = F.elu(self.bn3(self.gat3(x2, edge_index, edge_attr)))
        x4 = self.bn4(self.gat4(x3, edge_index, edge_attr))

        # Skip connection (from initial x to final x4)
        x = x + x4
        
        # Pooling
        x_pooled = self.set2set(x, batch)
        
        # Integrate conditions
        x_final = self.cond_integration(x_pooled, conditions)
        
        # Predict
        out = self.predictor(x_final)
        return out