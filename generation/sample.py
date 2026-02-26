import torch
import numpy as np
from typing import List, Dict, Any
from ..models.graphvae import ConditionalGraphVAE
from ..generation.postprocess import graphs_to_smiles, filter_and_rank

@torch.no_grad()
def generate_for_disease(model: ConditionalGraphVAE, disease_vec: torch.Tensor,
                         num_samples: int = 1000, device='cpu',
                         filters_config: Dict[str, Any] = None,
                         scorer = None,
                         known_smiles: List[str] = None):
    """
    Generate molecules for a single disease.

    Args:
        model: trained model.
        disease_vec: (disease_dim,) tensor.
        num_samples: number of molecules to generate.
        device: device.
        filters_config: config for chemical filters.
        scorer: WeightedScorer instance for ranking.
        known_smiles: list of known SMILES (for novelty).

    Returns:
        List of (smiles, score) tuples, sorted by score descending.
    """
    model.eval()
    disease_vec = disease_vec.to(device).unsqueeze(0).repeat(num_samples, 1)
    node_feats, edge_logits = model.sample(disease_vec)
    smiles_list = graphs_to_smiles(node_feats, edge_logits, model.max_nodes)

    # Filter and rank
    ranked = filter_and_rank(smiles_list, filters_config, scorer, known_smiles)
    return ranked
