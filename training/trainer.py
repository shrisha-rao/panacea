import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import yaml
import logging
from typing import Dict, Any, Optional

from models.graphvae import ConditionalGraphVAE
from data.dataset import DrugDiseaseDataset
from .loss import vae_loss
# from . import metrics as metric_utils

def collate_fn(batch):
    graphs = [item[0] for item in batch]
    disease_vecs = torch.stack([item[1] for item in batch])
    disease_ids = [item[2] for item in batch]
    # Assuming each graph has a 'smiles' attribute (set during dataset processing)
    smiles_list = [g.smiles for g in graphs]
    graph_batch = Batch.from_data_list(graphs)
    return graph_batch, disease_vecs, disease_ids, smiles_list

def collate_fn0(batch):
    """Custom collate function for DataLoader."""
    graphs = [item[0] for item in batch]
    disease_vecs = torch.stack([item[1] for item in batch])
    disease_ids = [item[2] for item in batch]
    graph_batch = Batch.from_data_list(graphs)
    return graph_batch, disease_vecs, disease_ids

class Trainer:
    def __init__(self, model: ConditionalGraphVAE, train_loader: DataLoader, val_loader: DataLoader,
                 config: Dict[str, Any], device: torch.device, checkpoint_dir: str):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
        self.scheduler = None
        if config.get('use_scheduler', False):
            self.scheduler = StepLR(self.optimizer,
                                    step_size=config.get('scheduler_step_size', 30),
                                    gamma=config.get('scheduler_gamma', 0.5))
        self.kl_beta = config.get('kl_beta', 1.0)
        self.epochs = config.get('epochs', 100)
        self.log_interval = config.get('log_interval', 10)
        self.val_interval = config.get('val_interval', 1)

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # batch is a tuple (graph_list, disease_vec_list, disease_ids)
            # We need to collate graphs into a Batch, and disease vectors into a tensor.
            # graph_list, disease_vec_list, _ = batch
            # Move graphs to device: graphs are already on CPU? We'll move later.
            # But we need to create a Batch object for the graphs.
            # graph_batch = Batch.from_data_list(graph_list).to(self.device)
            # disease_vec_batch = torch.stack(disease_vec_list).to(self.device)


            graph_batch, disease_vec_batch, _ , _ = batch  # unpack directly
            graph_batch = graph_batch.to(self.device)
            disease_vec_batch = disease_vec_batch.to(self.device)            
            

            self.optimizer.zero_grad()
            total, recon, kl = vae_loss(self.model, graph_batch, disease_vec_batch, self.kl_beta)
            total.backward()
            self.optimizer.step()

            total_loss += total.item()
            total_recon += recon.item()
            total_kl += kl.item()
            num_batches += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': total.item(),
                    'recon': recon.item(),
                    'kl': kl.item()
                })

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        self.logger.info(f"Epoch {epoch} Train: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")
        return avg_loss, avg_recon, avg_kl

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        # For metrics, we need to generate molecules and compute validity, etc.
        # We'll also collect generated SMILES for each disease? That could be heavy.
        # For simplicity, we'll just compute loss on validation set.
        # for batch_idx, batch in enumerate(self.val_loader):
        #     graph_list, disease_vec_list, _ = batch
        #     graph_batch = Batch.from_data_list(graph_list).to(self.device)
        #     disease_vec_batch = torch.stack(disease_vec_list).to(self.device)

        #     total, recon, kl = vae_loss(self.model, graph_batch, disease_vec_batch, self.kl_beta)

        for batch_idx, batch in enumerate(self.val_loader):
            graph_batch, disease_vec_batch, _, _ = batch
            graph_batch = graph_batch.to(self.device)
            disease_vec_batch = disease_vec_batch.to(self.device)

            total, recon, kl = vae_loss(self.model, graph_batch, disease_vec_batch, self.kl_beta)        

            total_loss += total.item()
            total_recon += recon.item()
            total_kl += kl.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        self.logger.info(f"Epoch {epoch} Val: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")
        return avg_loss, avg_recon, avg_kl

    def fit(self):
        best_val_loss = float('inf')
        for epoch in range(1, self.epochs + 1):
            train_loss, _, _ = self.train_epoch(epoch)
            if epoch % self.val_interval == 0:
                val_loss, _, _ = self.validate(epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
            if self.scheduler:
                self.scheduler.step()

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Loaded checkpoint from {path}")
        return checkpoint
