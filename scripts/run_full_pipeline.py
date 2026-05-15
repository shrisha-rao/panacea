import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import DrugDiseaseInMemoryDataset, MoleculeOnlyDataset
from generation.postprocess import decode_graph_batch, filter_and_rank
from models.graphvae import ConditionalGraphVAE
import training.metrics as metric_utils

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    PLOT_IMPORT_ERROR = exc
else:
    PLOT_IMPORT_ERROR = None


DEFAULT_SAMPLE_MOLECULES = 'data/samples/sample_molecules.csv'
DEFAULT_SAMPLE_PAIRS = 'data/samples/sample_drug_disease_pairs.csv'


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def ensure_writable_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    test_file = path / '.write_test'
    try:
        test_file.write_text('ok')
        test_file.unlink()
    except OSError as exc:
        raise RuntimeError(f"Output directory is not writable: {path}") from exc
    return path


def create_run_dir(base_dir, mode):
    base = ensure_writable_dir(base_dir)
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{mode}"
    run_dir = base / run_id
    for subdir in ('checkpoints', 'candidates', 'configs', 'metrics', 'plots', 'reconstructions'):
        (run_dir / subdir).mkdir(parents=True, exist_ok=False)
    return run_dir


def validate_csv(path, required_columns):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path, nrows=5)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def collate_examples(batch):
    graphs = [item[0] for item in batch]
    disease_vecs = torch.stack([item[1] for item in batch])
    disease_ids = [item[2] for item in batch]
    smiles = [item[3] for item in batch]
    return Batch.from_data_list(graphs), disease_vecs, disease_ids, smiles


def split_dataset(dataset, seed=13):
    if len(dataset) < 3:
        return dataset, dataset, dataset
    train_len = max(1, int(0.8 * len(dataset)))
    val_len = max(1, int(0.1 * len(dataset)))
    test_len = len(dataset) - train_len - val_len
    if test_len <= 0:
        test_len = 1
        train_len = len(dataset) - val_len - test_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def build_model(model_config, disease_dim, device, smoke=False):
    hidden_dim = 16 if smoke else model_config.get('hidden_dim', 64)
    latent_dim = 8 if smoke else model_config.get('latent_dim', 32)
    max_nodes = 12 if smoke else model_config.get('max_nodes', 100)
    model = ConditionalGraphVAE(
        node_feature_dim=model_config.get('node_feature_dim', 36),
        disease_dim=disease_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_nodes=max_nodes,
        num_encoder_layers=model_config.get('num_encoder_layers', 3),
        gnn_type=model_config.get('gnn_type', 'gin'),
        dropout=model_config.get('dropout', 0.1),
    )
    return model.to(device)


def train_model(model, train_loader, val_loader, device, epochs, lr, kl_beta, checkpoint_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for graph_batch, disease_vecs, _, _ in train_loader:
            graph_batch = graph_batch.to(device)
            disease_vecs = disease_vecs.to(device)
            optimizer.zero_grad()
            total, recon, kl = model.loss(graph_batch, disease_vecs, kl_beta)
            total.backward()
            optimizer.step()
            train_losses.append((float(total.item()), float(recon.item()), float(kl.item())))

        val_losses = []
        model.eval()
        with torch.no_grad():
            for graph_batch, disease_vecs, _, _ in val_loader:
                graph_batch = graph_batch.to(device)
                disease_vecs = disease_vecs.to(device)
                total, recon, kl = model.loss(graph_batch, disease_vecs, kl_beta)
                val_losses.append((float(total.item()), float(recon.item()), float(kl.item())))

        train_avg = average_losses(train_losses)
        val_avg = average_losses(val_losses)
        history.append({
            'epoch': epoch,
            'train_loss': train_avg[0],
            'train_recon': train_avg[1],
            'train_kl': train_avg[2],
            'val_loss': val_avg[0],
            'val_recon': val_avg[1],
            'val_kl': val_avg[2],
        })
        if val_avg[0] < best_val:
            best_val = val_avg[0]
            torch.save({'model_state_dict': model.state_dict(), 'history': history}, checkpoint_path)
        print(f"epoch={epoch} train_loss={train_avg[0]:.4f} val_loss={val_avg[0]:.4f}")
    return history


def average_losses(losses):
    if not losses:
        return (0.0, 0.0, 0.0)
    cols = list(zip(*losses))
    return tuple(sum(col) / len(col) for col in cols)


def reconstruct_examples(model, loader, device, output_csv, limit=16):
    records = []
    model.eval()
    with torch.no_grad():
        for graph_batch, disease_vecs, disease_ids, smiles_list in loader:
            graph_batch = graph_batch.to(device)
            disease_vecs = disease_vecs.to(device)
            node_feats, edge_logits, _, _ = model(graph_batch, disease_vecs)
            decoded = decode_graph_batch(node_feats, edge_logits, model.max_nodes)
            for input_smiles, disease_id, record in zip(smiles_list, disease_ids, decoded):
                records.append({
                    'input_smiles': input_smiles,
                    'disease_id': disease_id,
                    'decoded_smiles': record['smiles'] or '',
                    'valid': record['valid'],
                    'num_nodes': record['num_nodes'],
                })
                if len(records) >= limit:
                    break
            if len(records) >= limit:
                break
    write_csv(output_csv, records)
    return records


def generate_candidates(model, disease_vecs, disease_ids, device, output_csv, samples_per_context=16,
                        filters_config=None):
    all_records = []
    model.eval()
    with torch.no_grad():
        for disease_id, disease_vec in zip(disease_ids, disease_vecs):
            disease_batch = disease_vec.to(device).unsqueeze(0).repeat(samples_per_context, 1)
            node_feats, edge_logits = model.sample(disease_batch)
            decoded = decode_graph_batch(node_feats, edge_logits, model.max_nodes)
            valid_smiles = [record['smiles'] for record in decoded if record['valid']]
            ranked = dict(filter_and_rank(valid_smiles, filters_config=filters_config)) if valid_smiles else {}
            for record in decoded:
                smiles = record['smiles'] or ''
                all_records.append({
                    'disease_id': disease_id,
                    'smiles': smiles,
                    'valid': record['valid'],
                    'passed_filters': bool(smiles and smiles in ranked),
                    'score': ranked.get(smiles, 0.0),
                    'filters_applied': bool(filters_config),
                    'score_components': 'default' if smiles in ranked else '',
                    'num_nodes': record['num_nodes'],
                })
    write_csv(output_csv, all_records)
    return all_records


def write_csv(path, records):
    if not records:
        Path(path).write_text('')
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def compute_generation_metrics(candidate_records, known_smiles=None):
    attempted = len(candidate_records)
    valid_smiles = [record['smiles'] for record in candidate_records if record.get('valid') and record.get('smiles')]
    metrics = metric_utils.compute_all_metrics(valid_smiles, known_smiles=known_smiles)
    metrics.update({
        'attempted_decodes': attempted,
        'valid_decodes': len(valid_smiles),
        'invalid_decodes': attempted - len(valid_smiles),
        'decode_validity_rate': len(valid_smiles) / attempted if attempted else 0.0,
        'filter_pass_rate': sum(1 for record in candidate_records if record.get('passed_filters')) / attempted if attempted else 0.0,
    })
    return metrics


def plot_history(history, path):
    if not history:
        return
    if plt is None:
        Path(path).with_suffix('.txt').write_text(f"Plot unavailable: {PLOT_IMPORT_ERROR}\n")
        return
    epochs = [row['epoch'] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row['train_loss'] for row in history], label='train loss')
    plt.plot(epochs, [row['val_loss'] for row in history], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_validity(metrics_by_stage, path):
    if plt is None:
        Path(path).with_suffix('.txt').write_text(f"Plot unavailable: {PLOT_IMPORT_ERROR}\n")
        return
    stages = list(metrics_by_stage.keys())
    rates = [metrics_by_stage[stage].get('decode_validity_rate', 0.0) for stage in stages]
    plt.figure(figsize=(8, 5))
    plt.bar(stages, rates)
    plt.ylim(0, 1)
    plt.ylabel('valid decode rate')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scores(candidate_records, path):
    if plt is None:
        Path(path).with_suffix('.txt').write_text(f"Plot unavailable: {PLOT_IMPORT_ERROR}\n")
        return
    scores = [float(record.get('score', 0.0)) for record in candidate_records if record.get('valid')]
    if not scores:
        scores = [0.0]
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=10)
    plt.xlabel('candidate score')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def disease_vectors_from_csv(csv_path, disease_id_column, dim, output_path):
    df = pd.read_csv(csv_path)
    vectors = {disease_id: torch.randn(dim) for disease_id in sorted(df[disease_id_column].dropna().unique())}
    torch.save(vectors, output_path)
    return vectors


def load_disease_vectors(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing disease vector file: {path}")
    vectors = torch.load(path, map_location='cpu')
    if not isinstance(vectors, dict) or not vectors:
        raise ValueError(f"Disease vector file must contain a non-empty dict: {path}")
    return {key: value.float().flatten() for key, value in vectors.items()}


def copy_configs(run_dir):
    for path in ('config/data.yaml', 'config/model.yaml', 'config/train.yaml'):
        if os.path.exists(path):
            shutil.copy(path, run_dir / 'configs' / Path(path).name)


def write_report(path, mode, run_dir, metrics, limitations, metadata):
    lines = [
        '# Panacea Proof-of-Concept Run',
        '',
        f'- Mode: `{mode}`',
        f'- Run directory: `{run_dir}`',
        '',
        '## Metrics',
        '',
    ]
    for key, value in metadata.items():
        lines.append(f'- {key}: `{value}`')
    lines.append('')
    for stage, stage_metrics in metrics.items():
        lines.append(f'### {stage}')
        for key, value in stage_metrics.items():
            lines.append(f'- {key}: {value}')
        lines.append('')
    lines.extend(['## Limitations', ''])
    lines.extend([f'- {item}' for item in limitations])
    Path(path).write_text('\n'.join(lines) + '\n')


def make_manifest(run_dir, mode, metrics):
    artifacts = []
    for root, _, files in os.walk(run_dir):
        for file_name in files:
            path = Path(root) / file_name
            artifacts.append(str(path.relative_to(run_dir)))
    manifest = {
        'mode': mode,
        'created_at': datetime.now().isoformat(),
        'run_dir': str(run_dir),
        'metrics': metrics,
        'artifacts': sorted(artifacts),
    }
    write_json(run_dir / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description='Run the Panacea graph-native POC pipeline.')
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--drive-output-base', default='/content/drive/MyDrive/panacea-runs')
    parser.add_argument('--molecule-csv', default=None)
    parser.add_argument('--drug-disease-csv', default=None)
    parser.add_argument('--disease-vector-path', default=None)
    parser.add_argument('--allow-random-disease-vectors', action='store_true')
    parser.add_argument('--molecule-objective', choices=['vae', 'ae'], default='vae')
    parser.add_argument('--target-disease-id', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--samples-per-context', type=int, default=16)
    args = parser.parse_args()

    smoke = args.mode == 'smoke'
    molecule_csv = args.molecule_csv or (DEFAULT_SAMPLE_MOLECULES if smoke else 'data/raw/molecules.csv')
    drug_disease_csv = args.drug_disease_csv or (DEFAULT_SAMPLE_PAIRS if smoke else 'data/raw/drug_disease_pairs.csv')
    data_config = load_yaml('config/data.yaml')
    model_config = load_yaml('config/model.yaml')
    train_config = load_yaml('config/train.yaml')

    run_dir = create_run_dir(args.drive_output_base, args.mode)
    copy_configs(run_dir)
    print(f"Run directory: {run_dir}")

    validate_csv(molecule_csv, ['smiles'])
    validate_csv(drug_disease_csv, [data_config.get('smiles_column', 'smiles'), data_config.get('disease_id_column', 'disease_id')])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs if args.epochs is not None else (1 if smoke else train_config.get('epochs', 10))
    batch_size = args.batch_size if args.batch_size is not None else (4 if smoke else train_config.get('batch_size', 16))
    lr = train_config.get('learning_rate', 0.001)
    kl_beta = 0.0 if args.molecule_objective == 'ae' else train_config.get('kl_beta', 1.0)
    filters_config = {'use_filters': True, 'qed_threshold': 0.0, 'lipinski_max_violations': 4, 'sa_threshold': 10}

    metrics = {}
    limitations = [
        'Graph decoding is conservative and may reject many generated graphs.',
        'Random disease vectors are non-semantic and only validate conditional plumbing.',
        'SELFIES/SMILES decoders are intentionally deferred to a future comparison phase.',
    ]

    molecule_dataset = MoleculeOnlyDataset(molecule_csv)
    train_ds, val_ds, _ = split_dataset(molecule_dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_examples)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    molecule_model = build_model(model_config, disease_dim=0, device=device, smoke=smoke)
    molecule_ckpt = run_dir / 'checkpoints' / 'molecule_graphvae.pt'
    molecule_history = train_model(molecule_model, train_loader, val_loader, device, epochs, lr, kl_beta, molecule_ckpt)
    write_json(run_dir / 'metrics' / 'molecule_history.json', molecule_history)
    plot_history(molecule_history, run_dir / 'plots' / 'molecule_training_loss.png')
    recon_records = reconstruct_examples(molecule_model, val_loader, device, run_dir / 'reconstructions' / 'molecule_reconstructions.csv')
    molecule_candidates = generate_candidates(
        molecule_model,
        [torch.empty(0)],
        ['unconditional'],
        device,
        run_dir / 'candidates' / 'molecule_candidates.csv',
        samples_per_context=args.samples_per_context,
        filters_config=filters_config,
    )
    metrics['molecule_generation'] = compute_generation_metrics(
        molecule_candidates,
        known_smiles=[record['input_smiles'] for record in recon_records],
    )

    disease_vector_path = args.disease_vector_path or data_config.get('disease_vector_path')
    if smoke or args.allow_random_disease_vectors:
        disease_vector_path = run_dir / 'configs' / 'random_disease_vectors.pt'
        disease_vectors = disease_vectors_from_csv(
            drug_disease_csv,
            data_config.get('disease_id_column', 'disease_id'),
            model_config.get('disease_dim', 128),
            disease_vector_path,
        )
        limitations.append('Conditional stage used random disease vectors and does not prove disease relevance.')
    else:
        disease_vectors = load_disease_vectors(disease_vector_path)

    conditional_dataset = DrugDiseaseInMemoryDataset(
        drug_disease_csv,
        disease_vectors,
        smiles_column=data_config.get('smiles_column', 'smiles'),
        disease_id_column=data_config.get('disease_id_column', 'disease_id'),
    )
    cond_train_ds, cond_val_ds, _ = split_dataset(conditional_dataset)
    cond_train_loader = DataLoader(cond_train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_examples)
    cond_val_loader = DataLoader(cond_val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    disease_dim = conditional_dataset.disease_dim
    cond_model = build_model(model_config, disease_dim=disease_dim, device=device, smoke=smoke)
    cond_ckpt = run_dir / 'checkpoints' / 'conditional_graphvae.pt'
    cond_history = train_model(cond_model, cond_train_loader, cond_val_loader, device, epochs, lr, kl_beta, cond_ckpt)
    write_json(run_dir / 'metrics' / 'conditional_history.json', cond_history)
    plot_history(cond_history, run_dir / 'plots' / 'conditional_training_loss.png')
    reconstruct_examples(cond_model, cond_val_loader, device, run_dir / 'reconstructions' / 'conditional_reconstructions.csv')

    if args.target_disease_id:
        if args.target_disease_id not in disease_vectors:
            raise ValueError(f"Target disease ID has no vector: {args.target_disease_id}")
        selected_ids = [args.target_disease_id]
    else:
        selected_ids = list(disease_vectors.keys())[: min(4, len(disease_vectors))]
    selected_vecs = [disease_vectors[disease_id] for disease_id in selected_ids]
    conditional_candidates = generate_candidates(
        cond_model,
        selected_vecs,
        selected_ids,
        device,
        run_dir / 'candidates' / 'conditional_candidates.csv',
        samples_per_context=args.samples_per_context,
        filters_config=filters_config,
    )
    known_smiles = pd.read_csv(drug_disease_csv)[data_config.get('smiles_column', 'smiles')].dropna().tolist()
    metrics['conditional_generation'] = compute_generation_metrics(conditional_candidates, known_smiles=known_smiles)
    write_json(run_dir / 'metrics' / 'summary_metrics.json', metrics)
    plot_validity(metrics, run_dir / 'plots' / 'decode_validity.png')
    plot_scores(molecule_candidates + conditional_candidates, run_dir / 'plots' / 'candidate_scores.png')
    metadata = {
        'device': device,
        'epochs': epochs,
        'batch_size': batch_size,
        'molecule_objective': args.molecule_objective,
        'random_disease_vectors': smoke or args.allow_random_disease_vectors,
    }
    write_report(run_dir / 'report.md', args.mode, run_dir, metrics, limitations, metadata)
    make_manifest(run_dir, args.mode, metrics)
    print(f"Pipeline complete. Results saved to: {run_dir}")


if __name__ == '__main__':
    main()
