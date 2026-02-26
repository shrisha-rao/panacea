import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_add_pool
from torch_geometric.nn.models import MLP

class GraphEncoder(nn.Module):
    def __init__(self, node_feature_dim, disease_dim, hidden_dim, latent_dim,
                 num_layers=3, gnn_type='gin', dropout=0.1):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.disease_dim = disease_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        # Input dimension after concatenating disease vector to node features
        input_dim = node_feature_dim + disease_dim

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            if gnn_type == 'gin':
                nn_mlp = MLP([in_dim, hidden_dim, out_dim], norm=None, dropout=dropout)
                conv = GINConv(nn_mlp, train_eps=True)
            elif gnn_type == 'gcn':
                conv = GCNConv(in_dim, out_dim, improved=True)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            self.convs.append(conv)

        # Global pooling and output layers for mean and log variance
        self.pool = global_add_pool
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data, disease_vec):
        """
        Args:
            data: PyG Data object (with x, edge_index, batch)
            disease_vec: (batch_size, disease_dim) tensor
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch


        # print(f"x shape: {x.shape}")
        # print(f"disease_vec shape: {disease_vec.shape}")
        # print(f"batch shape: {batch.shape}")        

        # Expand disease_vec to each node: disease_vec[batch] gives (num_nodes, disease_dim)
        disease_expanded = disease_vec[batch]  # batch is a tensor of node batch indices

        # print(f"disease_expanded shape: {disease_expanded.shape}")
        
        # Concatenate node features with disease vector
        x = torch.cat([x, disease_expanded], dim=-1)

        # print(f"After concat shape: {x.shape}")

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        graph_emb = self.pool(x, batch)  # (batch_size, hidden_dim)

        mu = self.fc_mu(graph_emb)
        logvar = self.fc_logvar(graph_emb)
        return mu, logvar
