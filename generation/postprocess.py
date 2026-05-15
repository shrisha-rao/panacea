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

ATOM_VOCAB = [6, 7, 8, 9, 15, 16, 17, 35, 53, 1]
BOND_VOCAB = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}
MAX_VALENCE = {
    1: 1,
    6: 4,
    7: 3,
    8: 2,
    9: 1,
    15: 5,
    16: 6,
    17: 1,
    35: 1,
    53: 1,
}

def logits_to_graph(node_feats_logits: torch.Tensor, edge_logits: torch.Tensor, max_nodes: int,
                    node_threshold: float = 0.2, max_decode_nodes: int = 30):
    """
    Convert logits to a concrete graph (discrete nodes and edges).
    Returns node types (indices) and edge types (indices).
    This is a simple greedy decoding – for better results, you might sample.
    """
    atom_logits = node_feats_logits[:, :len(ATOM_VOCAB)]
    atom_probs = torch.softmax(atom_logits, dim=-1)
    atom_conf, atom_idx = atom_probs.max(dim=-1)
    active = (atom_conf > node_threshold).nonzero(as_tuple=False).flatten().tolist()
    if not active:
        active = [int(atom_conf.argmax().item())]
    active = active[:min(max_nodes, max_decode_nodes)]
    atomic_nums = [ATOM_VOCAB[int(atom_idx[i].item())] for i in active]
    edge_types = edge_logits.argmax(dim=-1).cpu().numpy()  # (max_nodes, max_nodes)
    return active, atomic_nums, edge_types

def graph_to_smiles(active_nodes, atomic_nums, edge_types):
    """
    Convert discrete node and edge predictions to a sanitized canonical SMILES.
    """
    if not active_nodes or not atomic_nums:
        return None

    rw_mol = Chem.RWMol()
    valence_used = []
    old_to_new = {}
    for old_idx, atomic_num in zip(active_nodes, atomic_nums):
        try:
            atom = Chem.Atom(int(atomic_num))
            new_idx = rw_mol.AddAtom(atom)
            old_to_new[old_idx] = new_idx
            valence_used.append(0)
        except Exception:
            return None

    for pos_i, old_i in enumerate(active_nodes):
        for pos_j, old_j in enumerate(active_nodes):
            if pos_j <= pos_i:
                continue
            edge_type = int(edge_types[old_i, old_j])
            bond_type = BOND_VOCAB.get(edge_type)
            if bond_type is None:
                continue
            order = int(bond_type == Chem.rdchem.BondType.SINGLE) or 1
            if bond_type == Chem.rdchem.BondType.DOUBLE:
                order = 2
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                order = 3
            atom_i = atomic_nums[pos_i]
            atom_j = atomic_nums[pos_j]
            if valence_used[pos_i] + order > MAX_VALENCE.get(atom_i, 4):
                continue
            if valence_used[pos_j] + order > MAX_VALENCE.get(atom_j, 4):
                continue
            try:
                rw_mol.AddBond(old_to_new[old_i], old_to_new[old_j], bond_type)
                valence_used[pos_i] += order
                valence_used[pos_j] += order
            except Exception:
                continue

    try:
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles if smiles else None
    except Exception:
        return None

def decode_graph_batch(node_feats_list, edge_logits_list, max_nodes):
    """Decode a batch and keep both valid and invalid attempts for reporting."""
    records = []
    batch_size = node_feats_list.size(0)
    for i in range(batch_size):
        active_nodes, atomic_nums, edge_types = logits_to_graph(node_feats_list[i], edge_logits_list[i], max_nodes)
        smiles = graph_to_smiles(active_nodes, atomic_nums, edge_types)
        records.append({
            'sample_index': i,
            'smiles': smiles,
            'valid': smiles is not None,
            'num_nodes': len(active_nodes),
        })
    return records

def graphs_to_smiles(node_feats_list, edge_logits_list, max_nodes):
    """
    Convert a batch of generated graphs to SMILES.
    node_feats_list: (batch, max_nodes, node_feat_dim) logits.
    edge_logits_list: (batch, max_nodes, max_nodes, num_edge_types) logits.
    Returns list of SMILES strings.
    """
    return [record['smiles'] for record in decode_graph_batch(node_feats_list, edge_logits_list, max_nodes) if record['valid']]

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
        if not smi:
            continue
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

