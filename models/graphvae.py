import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseConditionalGenerator
from .components.encoder import GraphEncoder
from .components.decoder import GraphDecoder
from .components.latent import reparameterize

class ConditionalGraphVAE(BaseConditionalGenerator):
    def __init__(self, node_feature_dim, disease_dim, hidden_dim, latent_dim, max_nodes,
                 num_encoder_layers=3, gnn_type='gin', dropout=0.1):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.disease_dim = disease_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes

        self.encoder = GraphEncoder(
            node_feature_dim=node_feature_dim,
            disease_dim=disease_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_encoder_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            disease_dim=disease_dim,
            hidden_dim=hidden_dim,
            max_nodes=max_nodes,
            node_feature_dim=node_feature_dim,
            num_edge_types=4  # SINGLE, DOUBLE, TRIPLE, AROMATIC (we'll treat no-edge separately)
        )

    def encode(self, graph, disease_vec):
        """Return mu and logvar."""
        return self.encoder(graph, disease_vec)

    def decode(self, z, disease_vec):
        """Return node features and edge logits."""
        return self.decoder(z, disease_vec)

    def forward(self, graph, disease_vec):
        """Encode, reparameterize, decode."""
        mu, logvar = self.encode(graph, disease_vec)
        z = reparameterize(mu, logvar)
        node_features, edge_logits = self.decode(z, disease_vec)
        return node_features, edge_logits, mu, logvar

    def loss(self, graph, disease_vec, kl_beta=1.0):
        """
        Compute VAE loss: reconstruction + KL.
        graph: batch of PyG Data objects (list)
        disease_vec: (batch_size, disease_dim)
        """
        batch_size = len(graph)
        # Pad graphs to max_nodes? Not needed; decoder expects max_nodes.
        node_features, edge_logits, mu, logvar = self.forward(graph, disease_vec)
        # Reconstruction loss using decoder's helper
        recon_loss = self.decoder.compute_reconstruction_loss(
            node_features, edge_logits, graph, batch_size, self.max_nodes
        )
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total_loss = recon_loss + kl_beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def sample(self, disease_vec, num_samples=1):
        """
        Generate molecules for given disease vector.
        disease_vec: (disease_dim) or (1, disease_dim)
        Returns: list of (node_features, edge_logits) for each sample.
        """
        if disease_vec.dim() == 1:
            disease_vec = disease_vec.unsqueeze(0)
        batch_size = disease_vec.size(0)
        # Sample z from prior N(0,1)
        z = torch.randn(batch_size, self.latent_dim, device=disease_vec.device)
        node_features, edge_logits = self.decode(z, disease_vec)
        # Convert to graphs (post-processing needed)
        return node_features, edge_logits
