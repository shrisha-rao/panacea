import torch
import numpy as np
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Any
import logging

from models.graphvae import ConditionalGraphVAE
from data.dataset import DrugDiseaseDataset
from generation.postprocess import graphs_to_smiles
import training.metrics as metric_utils
from evaluation import scoring


logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_on_test_set(model: ConditionalGraphVAE, test_loader, device,
                         known_smiles: List[str] = None,
                         num_samples_per_disease: int = 100,
                         scorer: scoring.WeightedScorer = None):
    """
    Evaluate model on test set.

    Args:
        model: trained model.
        test_loader: DataLoader for test set (should return batches of (graph_batch, disease_vec_batch, disease_id_list, smiles_list)).
        device: torch device.
        known_smiles: list of known drug SMILES (for novelty).
        num_samples_per_disease: number of molecules to generate per disease.
        scorer: optional WeightedScorer to compute scores.

    Returns:
        dict of metrics.
    """
    model.eval()
    all_generated_smiles = []
    all_known_smiles = []
    disease_ids = []

    for batch in test_loader:
        # Unpack batch (assuming collate_fn returns 4 elements)
        graph_batch, disease_vec_batch, disease_id_list, smiles_list = batch
        graph_batch = graph_batch.to(device)
        disease_vec_batch = disease_vec_batch.to(device)

        # Iterate over each disease in the batch
        for i in range(disease_vec_batch.size(0)):
            disease_vec = disease_vec_batch[i]                # shape: [disease_dim]
            disease_id = disease_id_list[i]
            true_smiles = smiles_list[i]                      # SMILES of the known drug

            # Repeat disease vector for num_samples
            disease_vec_exp = disease_vec.unsqueeze(0).repeat(num_samples_per_disease, 1)  # [num_samples, disease_dim]

            # Generate molecules
            node_feats, edge_logits = model.sample(disease_vec_exp)
            generated_smiles = graphs_to_smiles(node_feats, edge_logits, model.max_nodes)

            all_generated_smiles.extend(generated_smiles)
            disease_ids.extend([disease_id] * num_samples_per_disease)
            all_known_smiles.append(true_smiles)

    # Compute metrics on generated set
    metrics = metric_utils.compute_all_metrics(all_generated_smiles, known_smiles=known_smiles)

    # If scorer provided, compute scores and average top scores per disease
    if scorer is not None:
        # Build fingerprints for known SMILES (for novelty calculation in scorer)
        known_fps = []
        if known_smiles:
            from rdkit.Chem import AllChem
            for smi in known_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    known_fps.append(fp)

        # Group generated SMILES by disease
        disease_to_generated = {}
        for disease_id, smiles in zip(disease_ids, all_generated_smiles):
            disease_to_generated.setdefault(disease_id, []).append(smiles)

        avg_top_scores = []
        for disease_id, smiles_list in disease_to_generated.items():
            mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None]
            if not mols:
                continue
            scores = scorer.score_list(mols, known_fps=known_fps)
            scores_sorted = sorted(scores, reverse=True)
            k = max(1, len(scores_sorted) // 10)   # top 10%
            avg_top = np.mean(scores_sorted[:k])
            avg_top_scores.append(avg_top)

        metrics['avg_top_score'] = np.mean(avg_top_scores) if avg_top_scores else 0.0

    return metrics

@torch.no_grad()
def evaluate_on_test_set____(model: ConditionalGraphVAE, test_loader, device,
                         known_smiles: List[str] = None,
                         num_samples_per_disease: int = 100,
                         scorer: scoring.WeightedScorer = None):
    """
    Evaluate model on test set.

    Args:
        model: trained model.
        test_loader: DataLoader for test set.
        device: torch device.
        known_smiles: list of known drug SMILES (for novelty).
        num_samples_per_disease: number of molecules to generate per disease.
        scorer: optional WeightedScorer to compute scores.

    Returns:
        dict of metrics.
    """
    model.eval()
    all_generated_smiles = []
    all_known_smiles = []
    disease_ids = []


    for batch in test_loader:
        graph_batch, disease_vec_batch, disease_id_list = batch
        graph_batch = graph_batch.to(device)
        disease_vec_batch = disease_vec_batch.to(device)

        # Now iterate over batch dimension
        for i in range(graph_batch.num_graphs):
            # Extract individual graph? But we need disease_vec for each sample.
            # The batch already contains all graphs. For generation, we might need to handle each disease separately.
            # We'll keep the previous approach: generate for each disease in the batch.
            # But now disease_vec_batch is a tensor of shape (batch_size, disease_dim)
            # and graph_batch contains all graphs concatenated.
            # For generating new molecules, we don't need the graphs, only disease vectors.
            # So we can still iterate over disease_vec_batch.    

    # for batch in test_loader:
    #     graph_list, disease_vec_list, disease_id_list = batch
    #     disease_vec_batch = torch.stack(disease_vec_list).to(device)

    #     # Generate samples for each disease in batch
    #     for i, disease_vec in enumerate(disease_vec_batch):
            # Repeat disease_vec for num_samples
            disease_vec_exp = disease_vec.unsqueeze(0).repeat(num_samples_per_disease, 1)
            # Sample
            node_feats, edge_logits = model.sample(disease_vec_exp)
            # Convert to SMILES
            generated_smiles = graphs_to_smiles(node_feats, edge_logits, model.max_nodes)
            all_generated_smiles.extend(generated_smiles)
            # For each generated molecule, we also need to know which disease it's for
            disease_ids.extend([disease_id_list[i]] * num_samples_per_disease)
            # The true drug for this disease (there might be multiple, but we only have one per entry)
            # We'll collect true SMILES from the original graph
            true_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(graph_list[i].smiles))  # need to store smiles in graph
            all_known_smiles.append(true_smiles)

    # Compute metrics on generated set
    metrics = metric_utils.compute_all_metrics(all_generated_smiles, known_smiles=known_smiles)

    # If scorer provided, compute scores for each generated molecule and aggregate per disease
    if scorer is not None:
        # We need fingerprints for known drugs
        known_fps = []
        for smi in known_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                known_fps.append(fp)

        # Group by disease
        disease_to_generated = {}
        for disease_id, smiles in zip(disease_ids, all_generated_smiles):
            if disease_id not in disease_to_generated:
                disease_to_generated[disease_id] = []
            disease_to_generated[disease_id].append(smiles)

        # For each disease, compute average score of top-k molecules
        avg_top_scores = []
        for disease_id, smiles_list in disease_to_generated.items():
            mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None]
            if not mols:
                continue
            # Get reference fingerprint from the true drug? Not necessary for scoring, but we might want similarity.
            # We'll compute scores without reference.
            scores = scorer.score_list(mols, known_fps=known_fps)
            # Sort descending
            scores_sorted = sorted(scores, reverse=True)
            # Average of top 10%
            k = max(1, len(scores_sorted) // 10)
            avg_top = np.mean(scores_sorted[:k])
            avg_top_scores.append(avg_top)
        metrics['avg_top_score'] = np.mean(avg_top_scores) if avg_top_scores else 0.0

    return metrics
