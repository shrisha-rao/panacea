import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from typing import List, Optional

def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
    """Convert SMILES to RDKit Mol object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except:
            return None
    return mol

def mol_to_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Generate ECFP fingerprint."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two binary fingerprints."""
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    if union == 0:
        return 0.0
    return intersection / union

def is_valid_molecule(mol: Chem.Mol) -> bool:
    """Check if molecule is valid (sanitizable)."""
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

def compute_qed(mol: Chem.Mol) -> float:
    """Compute QED (quantitative estimate of drug-likeness)."""
    from rdkit.Chem.QED import qed
    return qed(mol)

def compute_sa_score(mol: Chem.Mol) -> float:
    """Compute synthetic accessibility score (1 easy, 10 hard)."""
    from rdkit.Chem import rdMolDescriptors
    # Implementation from RDKit Contrib SA_score
    # We'll use a simplified version or call external function
    # For simplicity, return a placeholder. In practice, use the SA score module.
    return 5.0  # placeholder

def lipinski_violations(mol: Chem.Mol) -> int:
    """Count number of Lipinski rule-of-5 violations."""
    from rdkit.Chem import Descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    return violations
