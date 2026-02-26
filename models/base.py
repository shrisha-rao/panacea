import torch.nn as nn
from abc import ABC, abstractmethod

class BaseConditionalGenerator(nn.Module, ABC):
    """Abstract base class for all conditional molecule generators."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, graph, disease_vec):
        """Forward pass: encode and decode (for training)."""
        pass

    @abstractmethod
    def loss(self, graph, disease_vec):
        """Compute loss for a batch."""
        pass

    @abstractmethod
    def sample(self, disease_vec, num_samples=1):
        """Generate molecules conditioned on disease vector."""
        pass

    @abstractmethod
    def encode(self, graph, disease_vec):
        """Encode graph to latent code (optional)."""
        pass

    @abstractmethod
    def decode(self, z, disease_vec):
        """Decode latent code to graph."""
        pass
