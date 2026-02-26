import os
import pandas as pd
import requests
import argparse
from tqdm import tqdm

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

def prepare_hcdt(output_path):
    """Download and prepare HCDT 2.0 dataset."""
    # HCDT 2.0 URL (example – need actual URL)
    url = "https://example.com/hcdt2.0.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    download_file(url, output_path)
    # Possibly process: extract SMILES and disease IDs
    df = pd.read_csv(output_path)
    # Assume columns: 'SMILES', 'DiseaseID'
    df = df[['SMILES', 'DiseaseID']].rename(columns={'SMILES': 'smiles', 'DiseaseID': 'disease_id'})
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def prepare_tdc(output_path):
    """Use Therapeutics Data Commons to get drug-disease pairs."""
    from tdc.single_pred import ADMET
    # This is just an example – TDC has many datasets.
    # For drug-disease, we might use the DrugRepurposing dataset.
    # But TDC's DrugRepurposing is for prediction, not generation pairs.
    # So we need a dataset with explicit drug-disease associations.
    # For now, we'll raise an error.
    raise NotImplementedError("TDC preparation not implemented yet.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['hcdt', 'tdc', 'repodb'], required=True)
    parser.add_argument('--output', required=True, help='Output CSV path')
    args = parser.parse_args()

    if args.dataset == 'hcdt':
        prepare_hcdt(args.output)
    elif args.dataset == 'tdc':
        prepare_tdc(args.output)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

if __name__ == '__main__':
    main()
