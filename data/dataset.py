import torch
import pandas as pd
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
from .preprocessing import smiles_to_graph, load_disease_vectors
from . import utils
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

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
