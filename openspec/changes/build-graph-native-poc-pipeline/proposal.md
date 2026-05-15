## Why

The current project has a strong graph-native molecular generation concept, but the implementation is not yet ready for a credible end-to-end proof of concept: dependencies, data setup, training flow, graph decoding, and evaluation are incomplete or placeholder-based. This change establishes a staged path from runnable Colab training to a defensible graph-based disease-conditioned molecule generation pipeline.

## What Changes

- Add a reproducible Colab-oriented setup for installing dependencies, preparing data, running smoke tests, training, and saving artifacts.
- Add a single-command Colab runner that executes the full proof-of-concept pipeline and writes all outputs to a new run folder in Google Drive.
- Introduce a molecule-only graph autoencoder/VAE milestone before disease conditioning so graph decoding can be validated independently.
- Replace placeholder graph-to-SMILES behavior with a real graph-to-RDKit decoding and validation path suitable for proof-of-concept outputs.
- Extend the graph VAE into a disease-conditioned model after molecule-only reconstruction/generation is working.
- Add evaluation/reporting outputs that distinguish reconstruction, unconditional generation, and disease-conditioned generation results.
- Add generated visualizations for training curves, reconstruction/generation validity, candidate score distributions, and proof-of-concept summaries.
- Clarify demo expectations so metrics are computed from actual decoded molecules rather than placeholders.

## Capabilities

### New Capabilities
- `colab-training-workflow`: Defines the reproducible Colab workflow for installing dependencies, validating data, running training, and persisting outputs.
- `molecule-graph-autoencoding`: Defines molecule-only graph autoencoder/VAE training, reconstruction, checkpointing, and smoke-test behavior.
- `graph-molecule-decoding`: Defines conversion from predicted graph tensors into RDKit-valid molecules and canonical SMILES.
- `disease-conditioned-generation`: Defines disease-vector-conditioned graph generation after molecule-only decoding is validated.
- `generation-evaluation-reporting`: Defines metrics and output artifacts for reconstruction, generated molecules, filtering, ranking, and proof-of-concept reporting.

### Modified Capabilities

None.

## Impact

- Affects `README.md`, configuration files under `config/`, and dependency/environment documentation.
- Affects scripts under `scripts/`, especially the end-to-end experiment path and any new Colab/smoke-test entrypoints.
- Affects data handling under `data/` for dataset validation and molecule-only training support.
- Affects `models/`, `training/`, `generation/`, and `evaluation/` to support staged graph-native training, real decoding, and honest metrics.
- Adds dependencies required for Colab execution, PyTorch Geometric, RDKit, data loading, and evaluation.
