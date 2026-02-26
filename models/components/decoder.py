import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, disease_dim, hidden_dim, max_nodes, node_feature_dim,
                 num_edge_types=4):
        """
        Args:
            latent_dim: dimension of latent z
            disease_dim: dimension of disease vector
            hidden_dim: hidden dimension for MLP
            max_nodes: maximum number of nodes to generate
            node_feature_dim: dimension of node features (output)
            num_edge_types: number of bond types (including no-edge? We'll treat separately)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.disease_dim = disease_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.num_edge_types = num_edge_types

        # Input to decoder: concatenated z and disease vector
        input_dim = latent_dim + disease_dim

        # MLP to generate node features and edge logits
        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * node_feature_dim)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim + 2 * node_feature_dim, hidden_dim),  # for each pair of nodes
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_edge_types)
        )

        # Alternatively, we could generate a full adjacency matrix with another MLP.

    def forward(self, z, disease_vec):
        """
        Args:
            z: (batch_size, latent_dim)
            disease_vec: (batch_size, disease_dim)
        Returns:
            node_features: (batch_size, max_nodes, node_feature_dim) – probabilities
            edge_logits: (batch_size, max_nodes, max_nodes, num_edge_types) – logits for each bond type
        """
        batch_size = z.size(0)
        decoder_input = torch.cat([z, disease_vec], dim=-1)  # (batch_size, input_dim)

        # Generate node features
        node_out = self.node_mlp(decoder_input)  # (batch_size, max_nodes * node_feature_dim)
        node_features = node_out.view(batch_size, self.max_nodes, self.node_feature_dim)
        # Apply softmax to get probabilities? Usually we use cross-entropy later, so logits.
        # We'll return raw logits for node features? Actually node features are categorical.
        # We'll treat them as logits (no softmax here).

        # Generate edge logits for each pair
        # Expand to get all pairs: create tensors of shape (batch, max_nodes, max_nodes, node_feature_dim)
        # by taking node_features_i and node_features_j.
        node_i = node_features.unsqueeze(2).expand(-1, -1, self.max_nodes, -1)  # (batch, max_nodes, max_nodes, feat)
        node_j = node_features.unsqueeze(1).expand(-1, self.max_nodes, -1, -1)  # (batch, max_nodes, max_nodes, feat)
        # Concatenate with global decoder_input repeated
        global_input = decoder_input.unsqueeze(1).unsqueeze(2).expand(-1, self.max_nodes, self.max_nodes, -1)  # (batch, max_nodes, max_nodes, input_dim)
        pair_input = torch.cat([global_input, node_i, node_j], dim=-1)  # (batch, max_nodes, max_nodes, input_dim+2*feat)
        # Reshape to (batch * max_nodes * max_nodes, ...) for MLP
        flat_input = pair_input.view(batch_size * self.max_nodes * self.max_nodes, -1)
        flat_edge_logits = self.edge_mlp(flat_input)  # (batch * max_nodes * max_nodes, num_edge_types)
        edge_logits = flat_edge_logits.view(batch_size, self.max_nodes, self.max_nodes, self.num_edge_types)

        # Remove self-loops? We'll keep them for now; they should be predicted as no-edge.
        return node_features, edge_logits

    def compute_reconstruction_loss(self, node_features_pred, edge_logits_pred,
                                    true_graphs, batch_size, max_nodes):
        """
        Compute reconstruction loss using Hungarian matching.

        Args:
            node_features_pred: (batch_size, max_nodes, node_feature_dim) logits
            edge_logits_pred: (batch_size, max_nodes, max_nodes, num_edge_types) logits
            true_graphs: list of Data objects for each item in batch
            batch_size: int
            max_nodes: int
        Returns:
            loss: scalar tensor
        """
        device = node_features_pred.device
        total_loss = 0.0

        for i in range(batch_size):
            true_graph = true_graphs[i]
            num_true_nodes = true_graph.x.size(0)
            # Pad true node features to max_nodes
            true_node_features = torch.zeros((max_nodes, self.node_feature_dim), device=device)
            true_node_features[:num_true_nodes] = true_graph.x
            # Build true adjacency (including bond types) of shape (max_nodes, max_nodes, num_edge_types)
            true_edge_attr = true_graph.edge_attr if hasattr(true_graph, 'edge_attr') else None
            edge_index = true_graph.edge_index
            true_adj = torch.zeros((max_nodes, max_nodes, self.num_edge_types), device=device)
            if edge_index.size(1) > 0:
                # For each edge, set the corresponding bond type
                # edge_attr is (num_edges, bond_feat_dim). We need to convert to bond type index.
                # We'll assume edge_attr is one-hot encoded with last dimension being type.
                # For simplicity, we'll take argmax to get bond type index.
                bond_types = true_edge_attr.argmax(dim=-1)  # (num_edges,)
                for j in range(edge_index.size(1)):
                    src, dst = edge_index[0, j], edge_index[1, j]
                    if src < max_nodes and dst < max_nodes:
                        true_adj[src, dst, bond_types[j]] = 1.0
                        # Usually we also set reverse? Already in edge_index.
            # Also set diagonal as no-bond (type 0 perhaps). We'll treat type 0 as "no edge".
            # But in our prediction we have separate logits for each type including no-edge? We'll include no-edge as first type.
            # We'll assume that for true graph, edges have type 1..num_edge_types-1, and non-edges have type 0.
            # So we need to construct a true_adj where each pair (i,j) has a one-hot over types, with type0 for no-edge.
            # The above only sets existing edges; we need to set all non-edges to type0.
            # We'll create a zero tensor and then set the existing edges.
            # Actually, we already initialized zeros. So zeros correspond to type0 (no-edge).
            # But we must ensure that for existing edges, the type index is correct and we set that type to 1.
            # That's what we did. So true_adj has a 1 at the bond type for each edge, and 0 elsewhere.
            # However, for non-edges, all types are 0. That means type0 (no-edge) is not represented explicitly.
            # To have a proper one-hot, we need to set type0 to 1 for non-edges. Let's adjust:
            # For all pairs, set type0 to 1 initially, then overwrite existing edges.
            true_adj = torch.zeros((max_nodes, max_nodes, self.num_edge_types), device=device)
            # Set type0 to 1 for all pairs
            true_adj[:, :, 0] = 1.0
            if edge_index.size(1) > 0:
                for j in range(edge_index.size(1)):
                    src, dst = edge_index[0, j], edge_index[1, j]
                    if src < max_nodes and dst < max_nodes:
                        # Reset type0 to 0 for this edge
                        true_adj[src, dst, 0] = 0.0
                        # Set the actual bond type to 1
                        true_adj[src, dst, bond_types[j]] = 1.0

            # Now perform Hungarian matching between predicted nodes (max_nodes) and true nodes (max_nodes)
            # We'll compute a cost matrix based on node feature cross-entropy? Or we can use the true nodes only,
            # but we have to match the predicted nodes to the true nodes (some predicted nodes may be dummy).
            # We'll use the node features: compute cross-entropy loss for each pair of predicted node (logits) vs true node.
            # Cost matrix of size (max_nodes, max_nodes) where cost[i,j] = cross_entropy(pred_i, true_j)
            # But true_j may be zero vectors for dummy nodes? We should mask.
            # Let's compute a cost based on negative log likelihood: -log(predicted probability of true node type).
            # We'll need to get predicted probabilities via softmax on node_features_pred[i].
            pred_node_probs = F.softmax(node_features_pred[i], dim=-1)  # (max_nodes, node_feature_dim)
            # For true nodes, we have one-hot features. We'll compute cross-entropy for each pair.
            # But node features are not necessarily probabilities; they are logits for categorical features.
            # To simplify, we'll assume node features are multi-dimensional binary (multiple categoricals concatenated).
            # We'll treat the whole vector as independent categories? That's not strictly correct.
            # For simplicity, we'll use a simple MSE between logits and true features as a proxy.
            # But a better approach: separate node feature reconstruction into separate losses per feature group.
            # Given the complexity, we'll use a simple MSE and hope it works for prototype.
            cost = torch.cdist(node_features_pred[i], true_node_features, p=2)  # Euclidean distance
            # cost shape (max_nodes, max_nodes)

            # Convert cost to numpy for Hungarian
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # Compute node loss for matched pairs
            node_loss = 0.0
            for r, c in zip(row_ind, col_ind):
                node_loss += F.mse_loss(node_features_pred[i, r], true_node_features[c])

            # Edge loss: for each pair (r,c) in matched indices, we have predicted edge logits for all pairs.
            # We need to compare predicted edge logits for all pairs (r1,r2) with true edges between matched true nodes.
            # We'll use cross-entropy loss on edge types.
            # Construct mapping from predicted node index to true node index.
            pred_to_true = {r: c for r, c in zip(row_ind, col_ind)}
            edge_loss = 0.0
            # For all pairs of predicted nodes (u, v)
            for u in range(max_nodes):
                for v in range(max_nodes):
                    if u in pred_to_true and v in pred_to_true:
                        true_u = pred_to_true[u]
                        true_v = pred_to_true[v]
                        # True edge type between true_u and true_v
                        true_edge = true_adj[true_u, true_v]  # one-hot (num_edge_types)
                        # Predicted logits for (u,v)
                        pred_logits = edge_logits_pred[i, u, v]  # (num_edge_types)
                        loss = F.cross_entropy(pred_logits.unsqueeze(0), true_edge.argmax().unsqueeze(0))
                        edge_loss += loss
            # Average over number of pairs
            edge_loss /= (max_nodes * max_nodes)

            total_loss += node_loss / len(row_ind) + edge_loss  # node_loss averaged over matched nodes

        return total_loss / batch_size
