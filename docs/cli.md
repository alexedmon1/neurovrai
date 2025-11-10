# Command-Line Interface (CLI)

**Status**: ✅ IMPLEMENTED - Production Ready

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

### `mri-preprocess run`

Main preprocessing command group for running workflows.

#### `run anatomical` - Anatomical preprocessing (T1w)

```bash
mri-preprocess run anatomical \
  --config configs/my_study.yaml \
  --subject sub-001
```

**Options**:
- `--config, -c PATH` - Configuration file [required]
- `--subject, -s TEXT` - Subject ID [required]
- `--session TEXT` - Session ID (optional)
- `--t1w-file PATH` - Override T1w file path (optional, auto-detected from config)

**Features**:
- Reorientation to standard space
- Skull stripping with BET
- Bias correction (ANTs N4)
- Linear registration to MNI152 (FLIRT)
- Nonlinear registration to MNI152 (FNIRT)
- Saves transforms to TransformRegistry for reuse

---

#### `run diffusion` - Diffusion preprocessing (DWI/DTI)

```bash
mri-preprocess run diffusion \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bedpostx
```

**Options**:
- `--config, -c PATH` - Configuration file [required]
- `--subject, -s TEXT` - Subject ID [required]
- `--session TEXT` - Session ID (optional)
- `--bedpostx / --no-bedpostx` - Enable BEDPOSTX fiber estimation (default: False)

**Features**:
- Denoising (dwidenoise)
- Gibbs unringing
- Eddy current and motion correction
- DTI model fitting (FA, MD, etc.)
- Optional BEDPOSTX for fiber distributions
- Reuses T1w→MNI transforms from anatomical workflow

**Note**: Additional workflow commands (functional, myelin) are planned for future implementation.

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

2. **Edit config** to customize parameters (set paths, adjust processing parameters)

3. **Validate config**:
   ```bash
   mri-preprocess config validate --config configs/my_study.yaml
   ```

4. **Run anatomical preprocessing** (required first):
   ```bash
   mri-preprocess run anatomical \
     --config configs/my_study.yaml \
     --subject sub-001
   ```

5. **Run diffusion preprocessing** (optional, reuses anatomical transforms):
   ```bash
   mri-preprocess run diffusion \
     --config configs/my_study.yaml \
     --subject sub-001
   ```

### Step-by-Step Processing

1. **Convert DICOMs** (if starting from DICOM):
   ```bash
   mri-preprocess convert \
     --config configs/study.yaml \
     --subject sub-001 \
     --anonymize
   ```

2. **Anatomical preprocessing** (runs first - creates transforms):
   ```bash
   mri-preprocess run anatomical \
     --config configs/study.yaml \
     --subject sub-001
   ```

3. **Diffusion preprocessing** (reuses T1w→MNI transforms):
   ```bash
   mri-preprocess run diffusion \
     --config configs/study.yaml \
     --subject sub-001
   ```

### Batch Processing

For multiple subjects, use a shell loop:

```bash
# Process multiple subjects
for subject in sub-001 sub-002 sub-003; do
  echo "Processing ${subject}..."
  mri-preprocess run anatomical --config configs/study.yaml --subject ${subject}
  mri-preprocess run diffusion --config configs/study.yaml --subject ${subject}
done
```

Or use GNU Parallel for true parallel execution:
```bash
# Create subject list
echo -e "sub-001\nsub-002\nsub-003" > subjects.txt

# Parallel anatomical preprocessing (run first for all subjects)
cat subjects.txt | parallel -j 4 \
  mri-preprocess run anatomical --config configs/study.yaml --subject {}

# Then parallel diffusion preprocessing
cat subjects.txt | parallel -j 4 \
  mri-preprocess run diffusion --config configs/study.yaml --subject {}
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
mri-preprocess run anatomical --config ... --verbose
```

---

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Configuration error
- `3` - Missing required files
- `4` - Workflow execution failure
