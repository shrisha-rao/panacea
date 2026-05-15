## 1. Environment And Colab Workflow

- [x] 1.1 Add a repository dependency file for the Python packages required by preprocessing, training, graph decoding, and evaluation.
- [x] 1.2 Add Colab setup documentation or notebook cells that clone/use the repo, install dependencies, select GPU, mount Drive, and configure output paths.
- [x] 1.3 Add a single Python full-pipeline runner command for Colab that orchestrates validation, training, evaluation, saving, and visualization generation.
- [x] 1.4 Create a unique timestamped Google Drive run folder for each full-pipeline execution.
- [x] 1.5 Add a smoke-test workflow that runs in Colab on a tiny dataset before full training.
- [x] 1.6 Add data readiness checks for required CSV columns, configured paths, disease vector files, Drive mount state, and writable output directories.

## 2. Molecule-Only Graph Training

- [x] 2.1 Add molecule-only dataset loading from SMILES without requiring disease vectors.
- [x] 2.2 Add or adapt a graph autoencoder/VAE model path that can train without disease conditioning.
- [x] 2.3 Add a molecule-only training entrypoint with config support for smoke, autoencoder, and VAE modes.
- [x] 2.4 Save molecule-only checkpoints and validation reconstruction samples.
- [ ] 2.5 Verify molecule-only training completes at least one smoke-test epoch and writes expected artifacts.

## 3. Graph-To-Molecule Decoding

- [x] 3.1 Replace placeholder generated SMILES behavior with graph tensor discretization that maps node and edge predictions to atom and bond candidates.
- [x] 3.2 Construct RDKit molecules from decoded atom and bond candidates.
- [x] 3.3 Sanitize decoded RDKit molecules and return canonical SMILES for valid decodes.
- [x] 3.4 Record invalid decode counts and avoid substituting hardcoded dummy molecules.
- [x] 3.5 Add tests or smoke checks for valid, invalid, and empty graph decoding cases.

## 4. Evaluation And Reporting

- [x] 4.1 Report reconstruction metrics separately from generated molecule metrics.
- [x] 4.2 Compute generation metrics only from actual decoded candidates, including validity, uniqueness, novelty when available, diversity, QED, and filter pass rate.
- [x] 4.3 Write candidate outputs with canonical SMILES, disease ID when available, scores, filter metadata, and validity metadata.
- [x] 4.4 Write a concise proof-of-concept report summarizing data, model stage, checkpoints, metrics, outputs, and limitations.
- [x] 4.5 Generate visualizations for training curves, validity rates, molecule metric distributions, candidate scores, and top candidate summaries.
- [x] 4.6 Write a results manifest that lists checkpoints, reports, metrics, candidates, plots, config, and run metadata.

## 5. Disease-Conditioned Generation

- [x] 5.1 Validate disease vector coverage and dimensionality before conditional training or generation.
- [x] 5.2 Adapt the conditional GraphVAE training path to use the validated graph decoding and reporting pipeline.
- [x] 5.3 Add disease-conditioned sampling for a requested disease vector.
- [x] 5.4 Label random-vector conditional runs as smoke/plumbing runs in reports.
- [ ] 5.5 Verify conditional training and sampling complete on a small drug-disease smoke dataset.

## 6. Documentation And Verification

- [x] 6.1 Update README guidance to distinguish smoke tests, molecule-only graph generation, and disease-conditioned proof-of-concept runs.
- [x] 6.2 Document the single Colab Python command and the expected Google Drive results directory structure.
- [x] 6.3 Document required data formats for molecule-only and drug-disease runs.
- [ ] 6.4 Run local or Colab-equivalent smoke tests for setup, molecule-only training, decoding, evaluation, visualization, and conditional training.
- [x] 6.5 Record known limitations and next steps, including future SELFIES/SMILES decoder comparison.
