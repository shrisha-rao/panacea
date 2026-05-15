## ADDED Requirements

### Requirement: Decode graph tensors to molecule candidates
The system SHALL convert predicted node and edge tensors into discrete molecule candidates without using placeholder SMILES values.

#### Scenario: Predicted graph is decoded
- **WHEN** the decoder produces node and edge predictions
- **THEN** the system discretizes atom and bond predictions and attempts to construct an RDKit molecule

### Requirement: Validate RDKit molecules
The system SHALL validate decoded molecule candidates using RDKit sanitization and canonical SMILES generation.

#### Scenario: Decoded molecule is valid
- **WHEN** RDKit successfully sanitizes a decoded molecule
- **THEN** the system returns its canonical SMILES and marks the decode as valid

#### Scenario: Decoded molecule is invalid
- **WHEN** RDKit cannot construct or sanitize a decoded molecule
- **THEN** the system marks the decode as invalid and excludes it from valid generated candidate metrics

### Requirement: Preserve invalid decode accounting
The system SHALL track the number of attempted, valid, and invalid graph decodes.

#### Scenario: Evaluation includes invalid decodes
- **WHEN** generation or reconstruction evaluation completes
- **THEN** the reported metrics include attempted decode count, valid decode count, invalid decode count, and validity rate

### Requirement: No dummy candidate reporting
The system MUST NOT report hardcoded or placeholder molecules as generated model candidates.

#### Scenario: Decoder cannot produce a valid molecule
- **WHEN** graph decoding fails for a generated sample
- **THEN** the system records an invalid decode instead of substituting a dummy SMILES
