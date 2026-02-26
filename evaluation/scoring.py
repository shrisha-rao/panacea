import numpy as np
from rdkit.Chem import QED, Descriptors
from typing import List, Dict, Any

class WeightedScorer:
    def __init__(self, weights: Dict[str, float]):
        """
        weights: dict with keys like 'qed', 'sa', 'lipinski', 'novelty', etc.
        """
        self.weights = weights

    def score_molecule(self, mol, reference_fingerprint=None, known_fps=None):
        """
        Compute a composite score for a single molecule.
        reference_fingerprint: fingerprint of the known drug (for similarity) – optional.
        known_fps: list of fingerprints of known drugs (for novelty) – optional.
        Returns a float.
        """
        score = 0.0
        # QED
        if 'qed' in self.weights:
            q = QED.qed(mol)
            score += self.weights['qed'] * q

        # Lipinski (inverse violations)
        if 'lipinski' in self.weights:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            violations = sum([mw>500, logp>5, hbd>5, hba>10])
            lip_score = 1.0 - violations / 4.0  # 1 if 0 violations, 0 if all 4
            score += self.weights['lipinski'] * lip_score

        # Similarity to reference
        if 'similarity' in self.weights and reference_fingerprint is not None:
            from rdkit.Chem import AllChem, DataStructs
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            sim = DataStructs.TanimotoSimilarity(fp, reference_fingerprint)
            score += self.weights['similarity'] * sim

        # Novelty: max similarity to known set (inverse)
        if 'novelty' in self.weights and known_fps is not None:
            from rdkit.Chem import AllChem, DataStructs
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            max_sim = max([DataStructs.TanimotoSimilarity(fp, kfp) for kfp in known_fps]) if known_fps else 0
            novelty = 1.0 - max_sim
            score += self.weights['novelty'] * novelty

        # SA score (inverse)
        if 'sa' in self.weights:
            # Placeholder SA score – implement properly
            sa = 5.0  # dummy
            # Normalize SA to 0-1 (1 is good, 0 is bad) assuming SA range 1-10
            sa_norm = max(0, 1 - (sa - 1) / 9)
            score += self.weights['sa'] * sa_norm

        return score

    def score_list(self, mols, reference_fp=None, known_fps=None):
        """Return list of scores for a list of molecules."""
        return [self.score_molecule(m, reference_fp, known_fps) for m in mols]
