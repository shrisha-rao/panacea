import torch
import torch.nn as nn

def vae_loss(model, graph_batch, disease_vec_batch, kl_beta=1.0):
    """
    Compute VAE loss for a batch.

    Args:
        model: ConditionalGraphVAE instance.
        graph_batch: list of PyG Data objects.
        disease_vec_batch: tensor (batch_size, disease_dim).
        kl_beta: weight for KL divergence.

    Returns:
        total_loss: scalar tensor.
        recon_loss: scalar tensor.
        kl_loss: scalar tensor.
    """
    total_loss, recon_loss, kl_loss = model.loss(graph_batch, disease_vec_batch, kl_beta)
    return total_loss, recon_loss, kl_loss
