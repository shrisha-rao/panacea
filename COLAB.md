# Colab Quickstart

Use this workflow after selecting a GPU runtime in Colab.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
git clone https://github.com/shrisha-rao/panacea.git
cd panacea
pip install -r requirements.txt
```

Run the complete smoke pipeline with one Python command:

```bash
python scripts/run_full_pipeline.py \
  --mode smoke \
  --drive-output-base /content/drive/MyDrive/panacea-runs
```

Run a full configured pipeline after placing data in Drive or the repo:

```bash
python scripts/run_full_pipeline.py \
  --mode full \
  --molecule-csv data/samples/sample_molecules.csv \
  --drug-disease-csv data/samples/sample_drug_disease_pairs.csv \
  --drive-output-base /content/drive/MyDrive/panacea-runs \
  --allow-random-disease-vectors
```

Each run creates a new timestamped folder under the Drive output base. The run folder contains checkpoints, metrics, candidates, plots, configs, a report, and a manifest.

Expected run folder shape:

```text
panacea-runs/
└── 20260515-143022-smoke/
    ├── checkpoints/
    ├── candidates/
    ├── configs/
    ├── metrics/
    ├── plots/
    ├── reconstructions/
    ├── manifest.json
    └── report.md
```

Notes:

- `--mode smoke` uses bundled tiny sample data and is intended to verify the environment before spending Colab credits.
- `--mode full` uses the configured or provided dataset paths.
- Random disease vectors are only for plumbing/smoke tests and are labeled as non-semantic conditioning in reports.
