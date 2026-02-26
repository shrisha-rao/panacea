import os
import yaml
import torch
from torch.utils.data import random_split, DataLoader
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

import sys
sys.path.append('../')

from data.dataset import DrugDiseaseDataset
from models.graphvae import ConditionalGraphVAE
from training.trainer import Trainer
from evaluation.benchmark import evaluate_on_test_set


def collate_fn(batch):
    graphs = [item[0] for item in batch]
    disease_vecs = torch.stack([item[1] for item in batch])
    disease_ids = [item[2] for item in batch]
    graph_batch = Batch.from_data_list(graphs)
    smiles_list = [g.smiles for g in graphs]
    return graph_batch, disease_vecs, disease_ids, smiles_list

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configs
    data_config = load_config('config/data.yaml')
    model_config = load_config('config/model.yaml')
    train_config = load_config('config/train.yaml')

    # Prepare dataset
    dataset = DrugDiseaseDataset(
        root='data/processed',
        csv_path=data_config['csv_path'],
        disease_vector_path=data_config.get('disease_vector_path'),
        smiles_column=data_config['smiles_column'],
        disease_id_column=data_config['disease_id_column']
    )

    if train_config.get('quick_run', False):
        dataset = torch.utils.data.Subset(dataset, range(11))

    # Split into train/val/test (80/10/10)
    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        collate_fn=collate_fn
    )


    # train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True,
    #                           num_workers=train_config['num_workers'])
    # val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False,
    #                         num_workers=train_config['num_workers'])
    # test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False,
    #                          num_workers=train_config['num_workers'])

    # Model
    model = ConditionalGraphVAE(
        node_feature_dim=model_config['node_feature_dim'],
        disease_dim=model_config['disease_dim'],
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        max_nodes=model_config['max_nodes'],
        num_encoder_layers=model_config['num_encoder_layers'],
        gnn_type=model_config['gnn_type'],
        dropout=model_config['dropout']
    )

    device = torch.device(train_config['device'] if torch.cuda.is_available() else 'cpu')

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        checkpoint_dir=train_config['checkpoint_dir']
    )

    # Train
    trainer.fit()

    # Load best model
    best_checkpoint = os.path.join(train_config['checkpoint_dir'], 'best_model.pt')
    if os.path.exists(best_checkpoint):
        trainer.load_checkpoint(best_checkpoint)

    # Evaluate on test set
    # Collect known SMILES from training set for novelty calculation
    known_smiles = []
    for idx in train_dataset.indices:
        graph, _, _ = dataset[idx]
        # Need to store SMILES in graph? We didn't. For now, we'll assume we have a separate file.
        # This is a placeholder.
        pass

    # We'll skip novelty for now
    metrics = evaluate_on_test_set(model, test_loader, device, num_samples_per_disease=100)
    print("Test metrics:", metrics)

if __name__ == '__main__':
    main()
