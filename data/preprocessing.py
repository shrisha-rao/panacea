import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from typing import Optional, Dict, List
import os

# Atom feature dimensions (example)
ATOM_FEATURE_DIMS = {
    'atomic_num': 10,   # one-hot for atomic numbers up to 10 (H, C, N, O, F, P, S, Cl, Br, I)
    'chirality': 4,      # one-hot for chiral tags (CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW, CHI_OTHER)
    'degree': 5,         # one-hot for degree 0-4
    'formal_charge': 5,  # one-hot for charges -2,-1,0,1,2
    'num_hs': 5,         # one-hot for number of hydrogens 0-4
    'hybridization': 5,  # one-hot: SP, SP2, SP3, SP3D, SP3D2
    'aromatic': 1,       # binary
    'in_ring': 1,        # binary
}
TOTAL_ATOM_FEATURES = sum(ATOM_FEATURE_DIMS.values())

# Bond feature dimensions
BOND_FEATURE_DIMS = {
    'bond_type': 4,      # one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC
    'stereo': 3,         # one-hot: STEREONONE, STEREOZ, STEREOE
    'conjugated': 1,     # binary
}
TOTAL_BOND_FEATURES = sum(BOND_FEATURE_DIMS.values())

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract atom features as a numpy array."""
    features = []
    # Atomic number (one-hot up to 10)
    atomic_num = atom.GetAtomicNum()
    atomic_num_vec = [0] * ATOM_FEATURE_DIMS['atomic_num']
    if atomic_num <= 10:
        atomic_num_vec[atomic_num-1] = 1
    else:
        atomic_num_vec[-1] = 1  # fallback to last index
    features.extend(atomic_num_vec)

    # Chirality
    chirality = atom.GetChiralTag()
    chirality_vec = [0] * ATOM_FEATURE_DIMS['chirality']
    if chirality == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        chirality_vec[1] = 1
    elif chirality == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        chirality_vec[2] = 1
    elif chirality != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        chirality_vec[3] = 1
    else:
        chirality_vec[0] = 1
    features.extend(chirality_vec)

    # Degree
    degree = atom.GetDegree()
    degree_vec = [0] * ATOM_FEATURE_DIMS['degree']
    if degree < len(degree_vec):
        degree_vec[degree] = 1
    else:
        degree_vec[-1] = 1
    features.extend(degree_vec)

    # Formal charge
    charge = atom.GetFormalCharge()
    charge_idx = charge + 2  # map -2..2 to 0..4
    charge_vec = [0] * ATOM_FEATURE_DIMS['formal_charge']
    if 0 <= charge_idx < len(charge_vec):
        charge_vec[charge_idx] = 1
    else:
        charge_vec[-1] = 1
    features.extend(charge_vec)

    # Number of hydrogens
    num_hs = atom.GetTotalNumHs()
    hs_vec = [0] * ATOM_FEATURE_DIMS['num_hs']
    if num_hs < len(hs_vec):
        hs_vec[num_hs] = 1
    else:
        hs_vec[-1] = 1
    features.extend(hs_vec)

    # Hybridization
    hybridization = atom.GetHybridization()
    hyb_vec = [0] * ATOM_FEATURE_DIMS['hybridization']
    hyb_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    if hybridization in hyb_map:
        hyb_vec[hyb_map[hybridization]] = 1
    else:
        hyb_vec[0] = 1  # default SP
    features.extend(hyb_vec)

    # Aromatic
    features.append(1 if atom.GetIsAromatic() else 0)

    # In ring
    features.append(1 if atom.IsInRing() else 0)

    return np.array(features, dtype=np.float32)

def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Extract bond features as a numpy array."""
    features = []
    # Bond type
    bond_type = bond.GetBondType()
    bond_type_vec = [0] * BOND_FEATURE_DIMS['bond_type']
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_type_vec[0] = 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_type_vec[1] = 1
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_type_vec[2] = 1
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_type_vec[3] = 1
    else:
        bond_type_vec[0] = 1
    features.extend(bond_type_vec)

    # Stereochemistry
    stereo = bond.GetStereo()
    stereo_vec = [0] * BOND_FEATURE_DIMS['stereo']
    if stereo == Chem.rdchem.BondStereo.STEREONONE:
        stereo_vec[0] = 1
    elif stereo == Chem.rdchem.BondStereo.STEREOZ:
        stereo_vec[1] = 1
    elif stereo == Chem.rdchem.BondStereo.STEREOE:
        stereo_vec[2] = 1
    else:
        stereo_vec[0] = 1
    features.extend(stereo_vec)

    # Conjugated
    features.append(1 if bond.GetIsConjugated() else 0)

    return np.array(features, dtype=np.float32)

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(np.array(atom_features), dtype=torch.float)

    # Edge indices and features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)  # same for reverse
    if len(edge_indices) == 0:
        # No bonds (single atom) – add self-loop? Usually not needed.
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0, TOTAL_BOND_FEATURES), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def load_disease_vectors(vector_path: str) -> Dict[str, torch.Tensor]:
    """Load precomputed disease vectors from a .pt file."""
    return torch.load(vector_path)

# Placeholder for computing disease vectors from scratch.
# In practice, you'd implement methods like protein-based embedding or ontology embedding.
