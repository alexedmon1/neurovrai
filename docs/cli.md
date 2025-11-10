# Command-Line Interface (CLI)

**Status**: 🚧 To be implemented in Phase 9

This document describes the command-line interface for the MRI preprocessing pipeline.

---

## Installation

```bash
cd human-mri-preprocess
uv sync
source .venv/bin/activate
```

The CLI tool `mri-preprocess` will be available after installation.

---

## Global Options

Available for all commands:

```bash
--config PATH          Path to study configuration YAML file [required for most commands]
--verbose, -v          Enable verbose output (DEBUG level logging)
--dry-run              Show what would be done without executing
--help, -h             Show help message
```

---

## Commands

### `mri-preprocess config`

Configuration management commands.

#### `config init` - Auto-generate configuration from DICOMs

```bash
mri-preprocess config init \
  --dicom-dir /path/to/subject/dicoms \
  --output configs/my_study.yaml \
  --study-name "My Study" \
  --study-code "STUDY01"
```

**Options**:
- `--dicom-dir PATH` - Directory containing DICOM files [required]
- `--output PATH` - Output YAML file path [required]
- `--study-name TEXT` - Full study name [required]
- `--study-code TEXT` - Short study code [required]
- `--subjects-list PATH` - Optional file with subject IDs (one per line)

**Output**: Complete study configuration YAML with auto-detected sequences

#### `config validate` - Validate configuration file

```bash
mri-preprocess config validate --config configs/my_study.yaml
```

**Checks**:
- YAML syntax
- Required parameters present
- File paths exist
- Sequence mappings non-empty
- Parameter values valid

---

### `mri-preprocess convert`

Convert DICOM to NIfTI (BIDS format).

```bash
mri-preprocess convert \
  --config configs/my_study.yaml \
  --dicom-dir /path/to/dicom \
  --output-dir /path/to/rawdata \
  --subject sub-001 \
  --anonymize
```

**Options**:
- `--config PATH` - Configuration file [required]
- `--dicom-dir PATH` - DICOM directory [required]
- `--output-dir PATH` - Output directory for NIfTI files [required]
- `--subject TEXT` - Subject ID [required]
- `--session TEXT` - Session ID (optional)
- `--anonymize` - Strip patient info from headers [recommended]

---

### `mri-preprocess anat`

Anatomical preprocessing (T1w/T2w).

```bash
mri-preprocess anat \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bids-dir /path/to/rawdata \
  --output-dir /path/to/derivatives
```

**Options**:
- `--config PATH` - Configuration file [required]
- `--subject TEXT` - Subject ID [required]
- `--bids-dir PATH` - BIDS rawdata directory [required]
- `--output-dir PATH` - Output directory [required]
- `--session TEXT` - Session ID (optional)
- `--override KEY=VALUE` - Override config parameter (e.g., `anatomical.skull_strip.frac=0.3`)

---

### `mri-preprocess dwi`

Diffusion preprocessing (DTI/DWI).

```bash
mri-preprocess dwi \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bids-dir /path/to/rawdata \
  --output-dir /path/to/derivatives
```

**Options**:
- `--config PATH` - Configuration file [required]
- `--subject TEXT` - Subject ID [required]
- `--bids-dir PATH` - BIDS rawdata directory [required]
- `--output-dir PATH` - Output directory [required]
- `--run-bedpostx` - Enable BEDPOSTX (slow, requires GPU)
- `--run-probtrackx` - Enable tractography (requires BEDPOSTX)

---

### `mri-preprocess func`

Functional preprocessing (fMRI).

```bash
mri-preprocess func \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bids-dir /path/to/rawdata \
  --output-dir /path/to/derivatives
```

**Options**:
- `--config PATH` - Configuration file [required]
- `--subject TEXT` - Subject ID [required]
- `--bids-dir PATH` - BIDS rawdata directory [required]
- `--output-dir PATH` - Output directory [required]
- `--force-single-echo` - Disable TEDANA (even for multi-echo data)
- `--skip-tedana` - Same as above

---

### `mri-preprocess myelin`

Myelin mapping (T1w/T2w ratio).

```bash
mri-preprocess myelin \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bids-dir /path/to/rawdata \
  --output-dir /path/to/derivatives
```

**Options**:
- `--config PATH` - Configuration file [required]
- `--subject TEXT` - Subject ID [required]
- `--bids-dir PATH` - BIDS rawdata directory [required]
- `--output-dir PATH` - Output directory [required]

---

### `mri-preprocess run`

Run full pipeline (orchestrator).

```bash
mri-preprocess run \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --steps all
```

**Options**:
- `--config PATH` - Configuration file [required]
- `--subject TEXT` - Subject ID [required]
- `--subjects-file PATH` - File with subject IDs (one per line)
- `--steps TEXT` - Steps to run: `all`, `convert`, `anat`, `dwi`, `func`, `myelin` (comma-separated)
- `--force` - Rerun even if outputs exist
- `--parallel` - Run subjects in parallel (use with `--subjects-file`)
- `--n-jobs INT` - Number of parallel jobs

**Examples**:

```bash
# Full pipeline, single subject
mri-preprocess run --config configs/study.yaml --subject sub-001 --steps all

# Specific steps
mri-preprocess run --config configs/study.yaml --subject sub-001 --steps anat,func

# Batch processing
mri-preprocess run --config configs/study.yaml --subjects-file subjects.txt --steps all --parallel --n-jobs 4
```

---

## Usage Patterns

### Quick Start

1. **Generate config from DICOMs**:
   ```bash
   mri-preprocess config init \
     --dicom-dir /data/pilot/sub-001/dicom \
     --output configs/my_study.yaml \
     --study-name "My Study" \
     --study-code "MYSTUDY"
   ```

2. **Edit config** to customize parameters

3. **Validate config**:
   ```bash
   mri-preprocess config validate --config configs/my_study.yaml
   ```

4. **Run full pipeline**:
   ```bash
   mri-preprocess run --config configs/my_study.yaml --subject sub-001 --steps all
   ```

### Step-by-Step Processing

1. **Convert DICOMs**:
   ```bash
   mri-preprocess convert \
     --config configs/study.yaml \
     --dicom-dir /data/sub-001/dicom \
     --output-dir /data/rawdata \
     --subject sub-001 \
     --anonymize
   ```

2. **Anatomical** (runs first - creates transforms):
   ```bash
   mri-preprocess anat \
     --config configs/study.yaml \
     --subject sub-001 \
     --bids-dir /data/rawdata \
     --output-dir /data/derivatives
   ```

3. **Other modalities** (reuse transforms):
   ```bash
   mri-preprocess dwi ...
   mri-preprocess func ...
   mri-preprocess myelin ...
   ```

### Batch Processing

Create `subjects.txt`:
```
sub-001
sub-002
sub-003
```

Run batch:
```bash
mri-preprocess run \
  --config configs/study.yaml \
  --subjects-file subjects.txt \
  --steps all \
  --parallel \
  --n-jobs 4
```

---

## Environment Variables

- `FSLDIR` - FSL installation directory (required)
- `SUBJECTS_DIR` - FreeSurfer subjects directory (if using FreeSurfer)
- `MRI_PREPROCESS_CONFIG` - Default config file path

---

## Logging

Logs are saved to `{output_dir}/logs/{subject}/`:
- `mri-preprocess.log` - Main log file
- `workflow_*.log` - Nipype workflow logs
- `crash_*.pklz` - Crash dumps (for debugging)

Set logging level in config:
```yaml
logging:
  level: DEBUG  # or INFO, WARNING, ERROR
```

Or via CLI:
```bash
mri-preprocess anat --config ... --verbose
```

---

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Configuration error
- `3` - Missing required files
- `4` - Workflow execution failure

---

**Note**: This documentation will be finalized when CLI is implemented in Phase 9.
