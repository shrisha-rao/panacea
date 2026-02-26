"""
Some functions, like graph_to_smiles in postprocess.py, are placeholders because converting a graph with predicted logits to a valid SMILES is a non-trivial task. In practice, we'd need a more sophisticated decoder that directly outputs SMILES strings or uses a graph-to-molecule algorithm (e.g., using RDKit's MolFromGraph or a fragment-based approach). For this prototype, we assume you'll eventually replace that with a proper implementation

"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Any, Optional, Tuple
from evaluation.filters import combined_filters
from evaluation.scoring import WeightedScorer

def logits_to_graph(node_feats_logits: torch.Tensor, edge_logits: torch.Tensor, max_nodes: int):
    """
    Convert logits to a concrete graph (discrete nodes and edges).
    Returns node types (indices) and edge types (indices).
    This is a simple greedy decoding – for better results, you might sample.
    """
    # Node features: we assume each node feature is a one-hot vector. Take argmax.
    node_types = node_feats_logits.argmax(dim=-1).cpu().numpy()  # (max_nodes,)
    # Edge logits: shape (max_nodes, max_nodes, num_edge_types). Take argmax for each pair.
    edge_types = edge_logits.argmax(dim=-1).cpu().numpy()  # (max_nodes, max_nodes)
    return node_types, edge_types

def graph_to_smiles(node_types, edge_types):
    """
    Convert discrete node types and edge types to RDKit molecule.
    This is highly non-trivial because we need to reconstruct atoms and bonds from indices.
    For simplicity, we'll assume node_types are atomic numbers (1-10) and edge_types are bond types (0: none, 1: single, 2: double, 3: triple, 4: aromatic).
    But our node features were one-hot over many dimensions, not just atomic number.
    A proper implementation would need to map feature vectors back to atom objects.
    For this prototype, we'll return a placeholder SMILES.
    """
    # Placeholder: return a dummy SMILES
    # In practice, you'd need to reconstruct the molecule from the predicted graph.
    # This is a complex task; many generative models use a separate decoder that directly outputs SMILES or uses a graph-to-molecule algorithm.
    # For now, we'll just return a string indicating failure.
    return "C"  # dummy

def graphs_to_smiles(node_feats_list, edge_logits_list, max_nodes):
    """
    Convert a batch of generated graphs to SMILES.
    node_feats_list: (batch, max_nodes, node_feat_dim) logits.
    edge_logits_list: (batch, max_nodes, max_nodes, num_edge_types) logits.
    Returns list of SMILES strings.
    """
    batch_size = node_feats_list.size(0)
    smiles_list = []
    for i in range(batch_size):
        node_types, edge_types = logits_to_graph(node_feats_list[i], edge_logits_list[i], max_nodes)
        smi = graph_to_smiles(node_types, edge_types)
        smiles_list.append(smi)
    return smiles_list

def filter_and_rank(smiles_list: List[str],
                    filters_config: Optional[Dict[str, Any]] = None,
                    scorer: Optional[WeightedScorer] = None,
                    known_smiles: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    """
    Apply filters, deduplicate, compute scores, and return ranked list.

    Args:
        smiles_list: list of SMILES strings.
        filters_config: config for combined_filters.
        scorer: WeightedScorer instance.
        known_smiles: list of known SMILES (for novelty).

    Returns:
        List of (smiles, score) tuples sorted descending.
    """
    # Convert to canonical SMILES and validate
    valid_mols = []
    valid_smiles_canon = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            can = Chem.MolToSmiles(mol)
            if can not in valid_smiles_canon:
                valid_mols.append(mol)
                valid_smiles_canon.append(can)
        except:
            continue

    # Apply chemical filters
    if filters_config:
        filtered_mols = []
        filtered_smiles = []
        for mol, can in zip(valid_mols, valid_smiles_canon):
            if combined_filters(mol, filters_config):
                filtered_mols.append(mol)
                filtered_smiles.append(can)
    else:
        filtered_mols = valid_mols
        filtered_smiles = valid_smiles_canon

    if not filtered_mols:
        return []

    # Compute scores if scorer provided
    if scorer:
        # Prepare known fingerprints
        known_fps = []
        if known_smiles:
            for smi in known_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    known_fps.append(fp)
        scores = scorer.score_list(filtered_mols, known_fps=known_fps)
    else:
        scores = [1.0] * len(filtered_mols)  # default score

    # Sort by score descending
    ranked = sorted(zip(filtered_smiles, scores), key=lambda x: x[1], reverse=True)
    return ranked


