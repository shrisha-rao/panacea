from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from typing import Optional, Dict, Any

def qed_filter(mol: Chem.Mol, threshold: float = 0.5) -> bool:
    """Return True if QED >= threshold."""
    try:
        return QED.qed(mol) >= threshold
    except:
        return False

def lipinski_filter(mol: Chem.Mol, max_violations: int = 1) -> bool:
    """
    Lipinski's rule of five:
    - MW <= 500
    - LogP <= 5
    - HBD <= 5
    - HBA <= 10
    Returns True if violations <= max_violations.
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    return violations <= max_violations

def sa_score_filter(mol: Chem.Mol, threshold: float = 6.0) -> bool:
    """
    Synthetic accessibility score (1=easy, 10=hard). Requires SA score implementation.
    Return True if score <= threshold.
    """
    # Placeholder – you need to implement actual SA score.
    # For now, always return True.
    return True

def combined_filters(mol: Chem.Mol, config: Dict[str, Any]) -> bool:
    """Apply multiple filters based on config."""
    if not config.get('use_filters', True):
        return True
    # QED
    if config.get('qed_threshold', 0) > 0:
        if not qed_filter(mol, config['qed_threshold']):
            return False
    # Lipinski
    if config.get('lipinski_max_violations', 1) < 5:
        if not lipinski_filter(mol, config['lipinski_max_violations']):
            return False
    # SA score
    if config.get('sa_threshold', 10) < 10:
        if not sa_score_filter(mol, config['sa_threshold']):
            return False
    return True
