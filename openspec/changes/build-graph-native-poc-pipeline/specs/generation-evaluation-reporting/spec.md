## ADDED Requirements

### Requirement: Reconstruction metrics
The system SHALL report reconstruction metrics separately from generation metrics.

#### Scenario: Reconstruction evaluation completes
- **WHEN** the system evaluates validation reconstructions
- **THEN** it reports reconstruction loss, valid reconstruction rate, and decoded reconstruction examples

### Requirement: Generation metrics
The system SHALL report generated molecule metrics based on actual decoded candidates.

#### Scenario: Generation evaluation completes
- **WHEN** the system evaluates generated molecules
- **THEN** it reports validity, uniqueness, novelty when known molecules are provided, diversity, QED, and filter pass rate

### Requirement: Candidate output file
The system SHALL write generated candidate molecules and associated metadata to a machine-readable file.

#### Scenario: Candidates are generated
- **WHEN** valid candidate molecules are produced
- **THEN** the system writes canonical SMILES, disease ID when available, score fields, and validity/filter metadata to an output file

### Requirement: Honest proof-of-concept report
The system SHALL produce a concise report that distinguishes implemented behavior, placeholder-free outputs, and known limitations.

#### Scenario: Report is generated
- **WHEN** a training/evaluation run completes
- **THEN** the report summarizes data used, model stage, checkpoints, generated outputs, metrics, and limitations

### Requirement: Visualization outputs
The system SHALL create visualizations summarizing training progress, decoding validity, molecule metrics, and candidate ranking results.

#### Scenario: Visualizations are generated
- **WHEN** a full pipeline run completes
- **THEN** the results directory contains plots for training losses, reconstruction or generation validity, candidate score distributions, and top candidate summaries when data is available

### Requirement: Results directory manifest
The system SHALL write a manifest describing all generated artifacts from a full pipeline run.

#### Scenario: Manifest is written
- **WHEN** the full pipeline command completes
- **THEN** the Google Drive run folder contains a manifest listing checkpoints, metrics, candidate files, reports, visualizations, config files, and run metadata

### Requirement: Filter and ranking transparency
The system SHALL make molecule filtering and ranking criteria visible in outputs.

#### Scenario: Filters are applied
- **WHEN** generated molecules are filtered or ranked
- **THEN** the output records which filters and score components were applied
