import torch
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
from .preprocessing import smiles_to_graph, load_disease_vectors
from . import utils
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


class MoleculeOnlyDataset(TorchDataset):
    """In-memory molecule dataset for graph AE/VAE smoke and POC training."""

    def __init__(self, csv_path, smiles_column='smiles'):
        self.csv_path = csv_path
        self.smiles_column = smiles_column
        self.df = pd.read_csv(csv_path)
        if smiles_column not in self.df.columns:
            raise KeyError(f"Column '{smiles_column}' not found in {csv_path}")

        self.examples = []
        for _, row in self.df.iterrows():
            smiles = row[smiles_column]
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue
            graph.smiles = smiles
            self.examples.append((graph, torch.empty(0, dtype=torch.float), '', smiles))

        if not self.examples:
            raise ValueError(f"No valid SMILES found in {csv_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DrugDiseaseInMemoryDataset(TorchDataset):
    """In-memory drug-disease dataset with explicit disease-vector validation."""

    def __init__(self, csv_path, disease_vectors, smiles_column='smiles', disease_id_column='disease_id'):
        self.csv_path = csv_path
        self.smiles_column = smiles_column
        self.disease_id_column = disease_id_column
        self.df = pd.read_csv(csv_path)
        for column in (smiles_column, disease_id_column):
            if column not in self.df.columns:
                raise KeyError(f"Column '{column}' not found in {csv_path}")
        if not disease_vectors:
            raise ValueError("Disease vectors are required for conditional training")

        first_vec = next(iter(disease_vectors.values()))
        self.disease_dim = int(first_vec.numel())
        self.examples = []
        missing = []
        wrong_shape = []
        for _, row in self.df.iterrows():
            smiles = row[smiles_column]
            disease_id = row[disease_id_column]
            if disease_id not in disease_vectors:
                missing.append(disease_id)
                continue
            disease_vec = disease_vectors[disease_id].float().flatten()
            if disease_vec.numel() != self.disease_dim:
                wrong_shape.append(disease_id)
                continue
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue
            graph.smiles = smiles
            self.examples.append((graph, disease_vec, disease_id, smiles))

        if missing:
            unique_missing = sorted(set(missing))[:10]
            raise ValueError(f"Missing disease vectors for disease IDs: {unique_missing}")
        if wrong_shape:
            unique_wrong = sorted(set(wrong_shape))[:10]
            raise ValueError(f"Wrong disease vector shape for disease IDs: {unique_wrong}")
        if not self.examples:
            raise ValueError(f"No valid drug-disease examples found in {csv_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class DrugDiseaseDataset(Dataset):
    def __init__(self, root, csv_path, disease_vector_path=None,
                 smiles_column='smiles', disease_id_column='disease_id',
                 transform=None, pre_transform=None):
        """
        Args:
            root: Root directory where the dataset should be stored.
            csv_path: Path to CSV file with drug-disease pairs.
            disease_vector_path: Path to precomputed disease vectors (.pt).
            smiles_column: Column name for SMILES.
            disease_id_column: Column name for disease identifier.
        """
        self.root = root
        self.csv_path = csv_path
        self.disease_vector_path = disease_vector_path
        self.smiles_column = smiles_column
        self.disease_id_column = disease_id_column
        self.df = pd.read_csv(csv_path)
        self.disease_vectors = None
        if disease_vector_path and os.path.exists(disease_vector_path):
            self.disease_vectors = load_disease_vectors(disease_vector_path)
        else:
            # If no vectors, we'll use learnable embeddings (handled by model)
            pass

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_path)]

    @property
    def processed_file_names(self):
        # We'll create one processed file per row
        return [f'data_{i}.pt' for i in range(len(self.df))]

    def download(self):
        # No download, just check existence
        pass

    def process(self):
        idx = 0
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
            smiles = row[self.smiles_column]
            disease_id = row[self.disease_id_column]
            graph = smiles_to_graph(smiles)
            if graph is None:
                # Skip invalid SMILES
                continue
            graph.smiles = smiles
            # Get disease vector if available
            if self.disease_vectors is not None:
                if disease_id in self.disease_vectors:
                    disease_vec = self.disease_vectors[disease_id]
                else:
                    # Fallback: zero vector (or raise error)
                    disease_vec = torch.zeros(self.disease_vectors[list(self.disease_vectors.keys())[0]].size(0))
            else:
                disease_vec = torch.tensor([], dtype=torch.float)  # placeholder
            # Save graph and disease_id in a tuple? We'll save a combined dict.
            data = {
                'graph': graph,
                'disease_id': disease_id,
                'disease_vec': disease_vec
            }
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Allowlist the PyTorch Geometric classes needed for safe loading
        with torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr]):
            data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'),
                              weights_only=False)
        graph = data['graph']
        disease_vec = data['disease_vec']
        disease_id = data['disease_id']
        return graph, disease_vec, disease_id

    def get_0(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'),
                          weights_only=False )
        graph = data['graph']
        disease_vec = data['disease_vec']
        disease_id = data['disease_id']
        return graph, disease_vec, disease_id
