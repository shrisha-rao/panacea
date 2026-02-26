"""
python scripts/compute_disease_vecs.py \
    --csv data/raw/drug_disease_pairs.csv \
    --method random \
    --output data/processed/disease_vectors.pt
"""


import torch
import pandas as pd
import argparse
from tqdm import tqdm
import os

def compute_protein_embeddings(disease_to_proteins, protein_sequences, model_name='esm2_t12_35M'):
    """
    Compute protein embeddings using ESM-2.
    This is a placeholder; you need to install fair-esm and implement properly.
    """
    # Placeholder: return random vectors
    import numpy as np
    vectors = {}
    for disease, prot_list in disease_to_proteins.items():
        # Dummy: average of random vectors
        emb = np.random.randn(128)  # 128-dim for demonstration
        vectors[disease] = torch.tensor(emb, dtype=torch.float)
    return vectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='CSV with disease_id column')
    parser.add_argument('--method', choices=['protein', 'ontology', 'random'], required=True)
    parser.add_argument('--output', required=True, help='Output .pt file')
    parser.add_argument('--protein_map', help='CSV mapping disease_id to protein sequences (if method=protein)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    disease_ids = df['disease_id'].unique()

    if args.method == 'random':
        # Random vectors
        vectors = {did: torch.randn(128) for did in disease_ids}
    elif args.method == 'protein':
        # Load mapping
        map_df = pd.read_csv(args.protein_map)
        # Build dictionary disease -> list of protein sequences
        disease_to_proteins = {}
        protein_sequences = {}
        for _, row in map_df.iterrows():
            did = row['disease_id']
            prot_id = row['protein_id']
            seq = row['sequence']
            if did not in disease_to_proteins:
                disease_to_proteins[did] = []
            disease_to_proteins[did].append(prot_id)
            protein_sequences[prot_id] = seq
        vectors = compute_protein_embeddings(disease_to_proteins, protein_sequences)
    else:
        # ontology not implemented
        raise NotImplementedError

    torch.save(vectors, args.output)
    print(f"Saved {len(vectors)} disease vectors to {args.output}")

if __name__ == '__main__':
    main()
