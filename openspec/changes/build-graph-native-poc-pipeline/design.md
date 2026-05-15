## Context

Panacea currently contains a conditional GraphVAE-oriented codebase with preprocessing, training, generation, and evaluation modules, but several parts are prototype scaffolding. The dependency declarations are incomplete, Colab execution is not documented, disease embeddings can be random placeholders, and graph decoding currently returns a dummy SMILES value rather than converting predicted graph tensors into molecules.

The project should remain graph-native for the first proof of concept. SELFIES/SMILES sequence decoders are useful future alternatives, but introducing them now would change the core project premise and avoid the main graph decoding problem rather than solving it.

The implementation should therefore proceed in stages: make the environment reproducible, validate molecule-only graph reconstruction/generation, then add disease conditioning once the molecule decoder can produce real RDKit-valid outputs.

## Goals / Non-Goals

**Goals:**

- Provide a reproducible Colab workflow for running smoke tests and training jobs.
- Establish a molecule-only graph autoencoder/VAE path before disease conditioning.
- Decode model graph outputs into RDKit molecules and canonical SMILES without placeholder values.
- Add disease conditioning only after molecule-only decoding has measurable validity.
- Produce honest evaluation outputs for reconstruction, generation, filtering, ranking, and demo reporting.

**Non-Goals:**

- Do not replace the graph-native decoder with SELFIES or SMILES in this change.
- Do not claim scientific disease relevance from random disease vectors.
- Do not add docking, wet-lab validation, or production serving infrastructure.
- Do not optimize for large-scale distributed training in the initial Colab proof of concept.

## Decisions

### Stage the model before conditioning

The first trainable target will be molecule-only graph autoencoding/VAE training. This isolates the core decoding challenge from disease representation quality.

Alternatives considered:

- Train the conditional GraphVAE immediately. Rejected because failures would be ambiguous across graph decoding, conditioning, data, and embedding quality.
- Train only the decoder. Rejected because the decoder needs a meaningful latent space and supervised reconstruction target.

### Keep graph tensors as the primary generated representation

The decoder will continue to output atom/node and bond/edge predictions. Postprocessing will discretize these predictions and construct RDKit molecules.

Alternatives considered:

- SELFIES/SMILES decoder. Deferred to a future comparison phase because it improves validity but changes the project from graph-native generation to sequence generation.
- Fragment-based decoder. Deferred because it can improve chemical validity but requires a larger design and fragment vocabulary.

### Validate decoded molecules with RDKit before reporting metrics

Generated candidates and metrics will be based only on attempted graph-to-RDKit decoding, sanitization, and canonicalization. Placeholder SMILES values must not be used for proof-of-concept metrics.

Alternatives considered:

- Report loss-only training. Useful for smoke tests but insufficient for generation claims.
- Keep dummy molecules for pipeline testing. Acceptable only in explicit unit tests, not in reported generation results.

### Treat disease embeddings as a staged dependency

The first conditional implementation may support random vectors for plumbing tests, but proof-of-concept reporting must distinguish random-vector smoke tests from meaningful disease embeddings.

Alternatives considered:

- Require protein/ontology embeddings immediately. Rejected for the first implementation because it would delay validation of the core graph generator.

### Favor small, runnable scripts over notebook-only logic

Colab should execute repository scripts rather than embedding the main logic in a notebook. The notebook or Colab instructions should orchestrate setup, data placement, config changes, and script execution.

Alternatives considered:

- Put training logic directly in a notebook. Rejected because it makes local/Colab behavior diverge and reduces reproducibility.

### Provide a single full-pipeline command

The Colab proof of concept should expose one Python command that validates inputs, runs the staged pipeline, saves checkpoints, writes candidate outputs, computes metrics, and creates visualizations in a new Google Drive-backed run directory. Lower-level scripts can still exist for development, but the demo path should not require manually chaining multiple commands.

Alternatives considered:

- Manual step-by-step notebook execution. Rejected for the main demo because it is error-prone and makes results harder to reproduce.
- Shell-only orchestration. Rejected as the primary interface because a Python runner can share config parsing, Drive output directory creation, validation, and reporting utilities with the rest of the project.

### Use timestamped Google Drive run folders

The full-pipeline Colab command should create a new timestamped run folder under a user-configurable Google Drive base path, such as `/content/drive/MyDrive/panacea-runs/<run-id>/`. This prevents accidental overwrites and makes it easy to retrieve checkpoints, reports, plots, and candidates after the Colab runtime disconnects.

Alternatives considered:

- Save only to Colab local storage. Rejected because Colab local storage is ephemeral.
- Reuse a fixed Drive output folder. Rejected because repeated runs could overwrite artifacts and make comparisons harder.

## Risks / Trade-offs

- Graph decoding may produce low validity even after molecule-only training -> mitigate by reporting validity honestly, adding constraints/repair incrementally, and keeping sequence decoders as a future fallback.
- Colab GPU dependency installation can be brittle -> mitigate with pinned dependency instructions and a smoke-test cell before full training.
- A single full-pipeline command can hide failures inside a long run -> mitigate with explicit stage logging, early validation, and saved partial artifacts.
- Google Drive may not be mounted or writable -> mitigate with an early output-path validation step and a clear error before training starts.
- Random disease vectors can create misleading demos -> mitigate by labeling random-vector runs as plumbing tests and separating them from disease-relevance claims.
- Small datasets may cause overfitting or memorization -> mitigate by reporting reconstruction, novelty, uniqueness, and train/test split behavior separately.
- RDKit sanitization can reject many generated graphs -> mitigate by preserving invalid decode counts and reasons where practical.

## Migration Plan

1. Add reproducible dependency and Colab setup artifacts.
2. Add data validation and smoke-test paths using a tiny molecule dataset.
3. Implement molecule-only graph AE/VAE training and checkpointing.
4. Implement graph-to-RDKit decoding and canonical SMILES output.
5. Add disease-conditioned training once molecule-only decoding is validated.
6. Add evaluation/reporting that clearly separates reconstruction, unconditional generation, and conditional generation results.

Existing placeholder behavior should be removed or gated so it cannot be mistaken for model output in proof-of-concept reports.

## Open Questions

- Which real drug-disease dataset should be the default full POC dataset after smoke tests?
- What minimum molecule validity rate is acceptable before moving from molecule-only VAE to disease-conditioned generation?
- Should chemical repair rules be conservative only, or should invalid generated graphs be aggressively corrected for demo usability?
- Which disease embedding source should be used for the first meaningful conditional run: ontology, protein/gene associations, or text embeddings?
