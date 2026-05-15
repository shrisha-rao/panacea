## ADDED Requirements

### Requirement: Molecule-only dataset mode
The system SHALL support training on molecular graphs without requiring disease vectors.

#### Scenario: Molecule-only data is loaded
- **WHEN** a CSV with SMILES values is provided for molecule-only training
- **THEN** the system converts valid SMILES into PyTorch Geometric graph examples without disease conditioning

### Requirement: Graph autoencoder training
The system SHALL provide a molecule-only graph autoencoder training path that reconstructs input molecular graphs.

#### Scenario: Autoencoder training runs
- **WHEN** the user runs molecule-only autoencoder training
- **THEN** the system trains an encoder-decoder model, logs reconstruction loss, and writes a checkpoint

### Requirement: Graph VAE training
The system SHALL provide a molecule-only graph VAE training path with latent sampling and KL regularization.

#### Scenario: VAE training runs
- **WHEN** the user runs molecule-only VAE training
- **THEN** the system trains with reconstruction and KL losses and writes the best checkpoint

### Requirement: Reconstruction outputs
The system SHALL produce reconstruction outputs for a sample of validation molecules.

#### Scenario: Reconstructions are generated
- **WHEN** validation runs after molecule-only training
- **THEN** the system saves input SMILES, decoded SMILES when valid, and reconstruction validity status

### Requirement: Stage gate for conditioning
The system SHALL expose molecule-only decoding metrics before disease-conditioned training is treated as proof-of-concept output.

#### Scenario: Molecule-only metrics are available
- **WHEN** molecule-only training completes
- **THEN** the system reports reconstruction validity and generation validity metrics that can be used to decide whether to proceed to disease conditioning
