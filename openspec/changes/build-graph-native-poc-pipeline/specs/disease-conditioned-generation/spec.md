## ADDED Requirements

### Requirement: Conditional graph model training
The system SHALL support graph VAE training conditioned on disease vectors after molecule-only graph training is available.

#### Scenario: Conditional training runs
- **WHEN** a drug-disease dataset and matching disease vectors are configured
- **THEN** the system trains a disease-conditioned graph VAE and writes a checkpoint

### Requirement: Disease vector validation
The system SHALL validate that disease vectors match configured dimensionality and cover disease IDs used for training or generation.

#### Scenario: Disease vector is missing
- **WHEN** a disease ID has no usable vector
- **THEN** the system reports the missing disease ID and applies the configured failure behavior before training or generation continues

#### Scenario: Disease vector has wrong shape
- **WHEN** a disease vector does not match the configured disease dimension
- **THEN** the system reports the mismatch and stops before model execution

### Requirement: Disease-conditioned sampling
The system SHALL generate molecule candidates from a trained conditional model for a requested disease vector.

#### Scenario: Samples are generated for disease
- **WHEN** the user requests candidates for a disease with a valid vector
- **THEN** the system samples latent vectors, conditions the decoder on the disease vector, decodes molecules, and writes candidate outputs

### Requirement: Distinguish random-vector runs
The system SHALL label random disease-vector runs as plumbing or smoke-test runs rather than disease-relevance proof.

#### Scenario: Random vectors are used
- **WHEN** disease vectors are generated with a random method
- **THEN** reports identify the run as non-semantic conditioning and avoid disease-relevance claims
