import torch
import torch.nn as nn
import torch.nn.functional as F

class EGNNLayer(nn.Module):
    def __init__(self, feature_dim, hidden_dim, edge_attribnute_dim, vector_dim):
        super(EGNNLayer, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.edge_attribnute_dim = edge_attribnute_dim
        self.coord_scaling = nn.Parameter(torch.tensor(1.0))
        self.vector_dim = vector_dim

        self.edge_function = nn.Sequential(
            nn.Linear(2 * feature_dim + 1 + edge_attribnute_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.SiLU()
        )

        self.coord_function = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.SiLU(),
            nn.Linear(1, 1)  # scalar output per edge
        )

        self.vector_function = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vector_dim)
        )

        self.node_function = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, h, x, v_init, edge_index, edge_attr=None):
        # Edge message passing
        src, dst = edge_index
        x_diff = x[dst] - x[src]
        x_diff_sq = (x_diff**2).sum(dim=1, keepdim=True)

        if edge_attr is None and self.edge_attribnute_dim > 0:
            edge_attr = torch.zeros(x_diff_sq.shape[0], self.edge_attribnute_dim, device=x.device, dtype=x.dtype)
        if self.edge_attribnute_dim > 0:
            edge_input = torch.cat([h[src], h[dst], x_diff_sq, edge_attr], dim=1)
        else:
            edge_input = torch.cat([h[src], h[dst], x_diff_sq], dim=1)
        m_ij = self.edge_function(edge_input)

        # Updating coordinates
        coord_update = self.coord_scaling * self.coord_function(m_ij) * x_diff

        # Vector updates
        v_update = torch.zeros_like(x).index_add(0, dst, coord_update)
        v_new = self.vector_function(h) * v_init + v_update

        # Aggregating messages from other nodes
        m_i = torch.zeros_like(h).index_add(0, dst, m_ij)

        # Updating node features with residual connection
        h_input = torch.cat([h, m_i], dim=1)
        h_new = self.node_function(h_input) + h

        x_new = x + v_new

        return h_new, x_new, v_new

class EGNN(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, vector_dim, n_layers, edge_attribnute_dim=0):
        super(EGNN, self).__init__()
        self.embedding = nn.Linear(input_dim, feature_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(feature_dim, hidden_dim, edge_attribnute_dim, vector_dim) for _ in range(n_layers)
        ])

    def forward(self, x, v_init, edge_index, edge_attr=None):
        velocity_norm = torch.sqrt(torch.sum(v_init**2, dim=1, keepdim=True))
        h = self.embedding(velocity_norm)
        for layer in self.layers:
            h, x, v_init = layer(h, x, v_init, edge_index, edge_attr)
        return h, x, v_init
