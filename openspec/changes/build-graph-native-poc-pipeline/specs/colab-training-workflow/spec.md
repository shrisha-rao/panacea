## ADDED Requirements

### Requirement: Reproducible Colab setup
The system SHALL provide a Colab-compatible workflow that installs all required runtime dependencies for training and evaluation.

#### Scenario: Dependencies install in Colab
- **WHEN** a user follows the Colab setup workflow in a fresh GPU runtime
- **THEN** PyTorch, PyTorch Geometric, RDKit, pandas, PyYAML, tqdm, scipy, and evaluation dependencies are available for repository scripts

### Requirement: Data readiness validation
The system SHALL validate required input data paths and file formats before starting training.

#### Scenario: Required data is missing
- **WHEN** the configured drug-disease CSV or disease vector file is required but missing
- **THEN** the workflow reports the missing path and stops before model training begins

#### Scenario: Required data is present
- **WHEN** the configured data files exist and contain required columns or tensors
- **THEN** the workflow proceeds to preprocessing or training

### Requirement: Smoke test before full training
The system SHALL support a small smoke-test mode that verifies setup, preprocessing, model construction, one training pass, decoding, and artifact writing.

#### Scenario: Smoke test succeeds
- **WHEN** the user runs the smoke-test workflow
- **THEN** the system completes a minimal training/evaluation pass and writes a checkpoint or report artifact

### Requirement: Single-command full pipeline
The system SHALL provide one Python command suitable for Colab that runs the full configured proof-of-concept pipeline.

#### Scenario: Full pipeline command runs
- **WHEN** the user runs the documented full-pipeline Python command in Colab
- **THEN** the system validates inputs, trains the configured model stages, evaluates outputs, saves checkpoints, writes result files, and creates visualizations without requiring additional manual commands

### Requirement: Persisted Colab outputs
The system SHALL save checkpoints, generated candidates, metrics, reports, and visualizations to a new run folder in a user-controlled Google Drive directory.

#### Scenario: Drive output base is configured
- **WHEN** the user configures a Google Drive output base directory
- **THEN** the full pipeline creates a new run folder under that base directory and writes training and evaluation artifacts under the run folder

#### Scenario: Drive output is not writable
- **WHEN** the configured Google Drive output base directory cannot be created or written
- **THEN** the workflow reports the output-path problem and stops before model training begins

### Requirement: Unique run folders
The system SHALL create a distinct output folder for each full-pipeline run.

#### Scenario: Pipeline is run multiple times
- **WHEN** the user runs the full-pipeline command more than once
- **THEN** each run writes to a different run folder and does not overwrite previous results
