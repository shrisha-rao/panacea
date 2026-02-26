"""
python scripts/prepare_hcdt_data.py \
    data/raw/HCDT2.0/DRUG_GENE/DRUG.tsv \
    data/raw/HCDT2.0/Drug-Disease.xlsx \
    data/raw/drug_disease_pairs.csv \
    --disease-id-col MeSH
"""




import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Prepare HCDT dataset for conditional molecule generation.')
    parser.add_argument('drug_tsv', help='Path to DRUG.tsv file')
    parser.add_argument('drug_disease_xlsx', help='Path to Drug-Disease.xlsx file')
    parser.add_argument('output_csv', help='Output CSV path')
    parser.add_argument('--disease-id-col', default='MeSH', choices=['MeSH', 'OMIM', 'ICD-11'],
                        help='Disease identifier column to use')
    parser.add_argument('--drug-name-col-drug-tsv', default='DRUG_NAME',
                        help='Column name in DRUG.tsv containing drug name')
    parser.add_argument('--drug-name-col-disease', default='Drug_Name',
                        help='Column name in Drug-Disease.xlsx containing drug name')
    parser.add_argument('--smiles-col', default='canonicalsmiles',
                        help='Column name in DRUG.tsv containing SMILES')
    args = parser.parse_args()

    # Read DRUG.tsv
    drug_df = pd.read_csv(args.drug_tsv, sep='\t')
    required_drug_cols = [args.drug_name_col_drug_tsv, args.smiles_col]
    for col in required_drug_cols:
        if col not in drug_df.columns:
            raise KeyError(f"Column '{col}' not found in {args.drug_tsv}")
    drug_df = drug_df[required_drug_cols].dropna()
    drug_df = drug_df.rename(columns={args.drug_name_col_drug_tsv: 'drug_name', args.smiles_col: 'smiles'})

    # Read Drug-Disease.xlsx
    disease_df = pd.read_excel(args.drug_disease_xlsx)
    if args.drug_name_col_disease not in disease_df.columns:
        raise KeyError(f"Column '{args.drug_name_col_disease}' not found in {args.drug_disease_xlsx}")
    if args.disease_id_col not in disease_df.columns:
        raise KeyError(f"Column '{args.disease_id_col}' not found in {args.drug_disease_xlsx}")
    disease_df = disease_df[[args.drug_name_col_disease, args.disease_id_col]].dropna()
    disease_df = disease_df.rename(columns={args.drug_name_col_disease: 'drug_name', args.disease_id_col: 'disease_id'})

    # Merge
    merged = pd.merge(disease_df, drug_df, on='drug_name', how='inner')
    merged = merged[['smiles', 'disease_id']].drop_duplicates()

    # Save
    merged.to_csv(args.output_csv, index=False)
    print(f"Saved {len(merged)} rows to {args.output_csv}")

if __name__ == '__main__':
    main()
