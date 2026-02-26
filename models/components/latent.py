import torch

def reparameterize(mu, logvar):
    """Reparameterization trick to sample from N(mu, var) using N(0,1)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
