import torch
import numpy as np
from rdkit import Chem
# from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem, DataStructs
from typing import List, Dict, Optional
import warnings

def compute_validity(smiles_list: List[str]) -> float:
    """Fraction of SMILES that are valid."""
    valid = 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid += 1
            except:
                pass
    return valid / len(smiles_list) if smiles_list else 0.0

def compute_uniqueness(smiles_list: List[str]) -> float:
    """Fraction of valid SMILES that are unique."""
    valid_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid_smiles.append(Chem.MolToSmiles(mol))  # canonical
            except:
                pass
    if not valid_smiles:
        return 0.0
    unique = set(valid_smiles)
    return len(unique) / len(valid_smiles)

def compute_novelty(generated_smiles: List[str], known_smiles: List[str]) -> float:
    """
    Fraction of generated molecules that are not in the known set.
    Uses canonical SMILES.
    """
    known_set = set()
    for s in known_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                known_set.add(Chem.MolToSmiles(mol))
            except:
                pass
    novel = 0
    total_valid = 0
    for s in generated_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                can = Chem.MolToSmiles(mol)
                if can not in known_set:
                    novel += 1
                total_valid += 1
            except:
                pass
    return novel / total_valid if total_valid > 0 else 0.0

def average_qed(smiles_list: List[str]) -> float:
    """Average QED over valid molecules."""
    qeds = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                qeds.append(QED.qed(mol))
            except:
                pass
    return np.mean(qeds) if qeds else 0.0

def average_sa_score(smiles_list: List[str]) -> float:
    """
    Average synthetic accessibility score (1=easy, 10=hard).
    Requires the SA score module from RDKit Contrib.
    If not available, return a placeholder.
    """
    try:
        from rdkit.Chem import rdMolDescriptors
        # This is a placeholder; you need to implement SA score properly.
        # For now, we'll return a dummy value.
        return 5.0
    except ImportError:
        warnings.warn("SA score module not available, returning 0.")
        return 0.0

def diversity(smiles_list: List[str]) -> float:
    """
    Average pairwise Tanimoto distance (1 - similarity) among valid molecules.
    """
    valid_mols = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                valid_mols.append(mol)
            except:
                pass
    if len(valid_mols) < 2:
        return 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in valid_mols]
    n = len(fps)
    sim_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim_sum += sim
            count += 1
    avg_sim = sim_sum / count if count > 0 else 0.0
    return 1.0 - avg_sim

def compute_all_metrics(generated_smiles: List[str], known_smiles: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute all metrics and return as dict."""
    metrics = {}
    metrics['validity'] = compute_validity(generated_smiles)
    metrics['uniqueness'] = compute_uniqueness(generated_smiles)
    if known_smiles is not None:
        metrics['novelty'] = compute_novelty(generated_smiles, known_smiles)
    metrics['avg_qed'] = average_qed(generated_smiles)
    metrics['avg_sa'] = average_sa_score(generated_smiles)
    metrics['diversity'] = diversity(generated_smiles)
    return metrics
