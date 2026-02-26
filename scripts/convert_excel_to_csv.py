"""
python scripts/convert_excel_to_csv.py Drug-Disease.xlsx data/raw/drug_disease_pairs.csv --smiles-col SMILES --disease-col DiseaseID
"""

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert Excel drug-disease file to CSV.')
    parser.add_argument('input', help='Input Excel file path (e.g., Drug-Disease.xlsx)')
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--smiles-col', default='SMILES', help='Name of column containing SMILES')
    parser.add_argument('--disease-col', default='DiseaseID', help='Name of column containing disease ID')
    args = parser.parse_args()

    # Read Excel
    df = pd.read_excel(args.input)
    # Keep only required columns and rename to standard names
    df = df[[args.smiles_col, args.disease_col]].rename(columns={
        args.smiles_col: 'smiles',
        args.disease_col: 'disease_id'
    })
    # Drop rows with missing values
    df.dropna(subset=['smiles', 'disease_id'], inplace=True)
    # Save CSV
    df.to_csv(args.output, index=False)
    print(f"Converted {len(df)} rows to {args.output}")

if __name__ == '__main__':
    main()
