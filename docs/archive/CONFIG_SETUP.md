# Configuration Setup Guide

Complete guide for creating and customizing `config.yaml` for your study.

## Quick Start

### Option 1: Auto-Generate (Recommended)

```bash
# Create config.yaml for your study (creates /mnt/bytopia/IRC805/config.yaml)
python create_config.py --study-root /mnt/bytopia/IRC805
```

This creates a complete `config.yaml` in the study root directory with all required sections and sensible defaults.

### Option 2: Interactive Mode

```bash
# Prompts you for study root directory
python create_config.py
```

### Option 3: Manual Creation

Copy and edit the template in this document (see "Complete Template" section below).

## Before You Start

### Required Information

Before creating your config, gather:

1. **Study root directory**
   - Where all study data will live
   - Example: `/mnt/bytopia/IRC805`

2. **DICOM location**
   - Where raw DICOM files are stored
   - Should be: `{study_root}/raw/dicom/`
   - Each subject in: `{study_root}/raw/dicom/{subject_id}/`

3. **Scanner parameters** (check your MRI protocol!)
   - Functional TR (repetition time)
   - Functional TE (echo time/times for multi-echo)
   - DWI readout time (for TOPUP)

4. **System resources**
   - Number of CPU cores available
   - GPU available? (for eddy_cuda)

## Directory Structure

The pipeline expects this structure:

```
{study_root}/                    # e.g., /mnt/bytopia/IRC805/
├── raw/
│   └── dicom/                   # ← Place DICOM files here
│       ├── SUBJECT-001/
│       ├── SUBJECT-002/
│       └── ...
├── bids/                        # Created by pipeline (NIfTI files)
├── derivatives/                 # Created by pipeline (preprocessed data)
├── work/                        # Created by pipeline (temporary files)
├── qc/                          # Created by pipeline (QC reports)
├── logs/                        # Pipeline logs
└── transforms/                  # Spatial transformation matrices
```

**Before running:** Only `raw/dicom/` needs to exist with subject data!

## Configuration Sections

### 1. Project Paths (Required)

```yaml
project_dir: /mnt/bytopia/IRC805
dicom_dir: /mnt/bytopia/IRC805/raw/dicom
bids_dir: /mnt/bytopia/IRC805/bids
derivatives_dir: /mnt/bytopia/IRC805/derivatives
work_dir: /mnt/bytopia/IRC805/work
```

**What to change:**
- `project_dir`: Your study root directory
- Other paths: Usually fine as defaults

### 2. Execution Settings

```yaml
execution:
  plugin: MultiProc        # Use multiprocessing
  n_procs: 6               # Number of parallel processes
```

**What to change:**
- `n_procs`: Set to number of CPU cores available (typically 4-16)
  - Too high: May run out of memory
  - Too low: Slow processing
  - Good starting point: 6-8 cores

### 3. Templates (Usually Don't Change)

```yaml
templates:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
  mni152_t1_1mm: /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
  fmrib58_fa: /usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz
```

**Only change if:** Your FSL installation is in a different location

### 4. Anatomical Settings

```yaml
anatomical:
  bet:
    frac: 0.5              # Brain extraction threshold (0.1-0.9)
    reduce_bias: true      # Bias field reduction
    robust: true           # Robust brain center estimation
  segmentation_method: ants  # 'ants' or 'fsl'
  registration_method: fsl   # 'fsl' or 'ants'
  run_qc: true
```

**What to change:**
- `bet.frac`: If brain extraction is too aggressive (0.3) or conservative (0.7)
- Usually defaults are good

### 5. Diffusion Settings ⚠️ Check Your Protocol!

```yaml
diffusion:
  topup:
    enabled: auto          # Auto-detect reverse PE images
    readout_time: 0.05     # ← CHECK YOUR PROTOCOL!
  eddy_config:
    use_cuda: true         # Set false if no GPU
  advanced_models:
    fit_dki: true          # Diffusion Kurtosis Imaging
    fit_noddi: true        # NODDI via AMICO (recommended)
    use_amico: true        # 100x faster than DIPY
```

**CRITICAL: `readout_time`**
- This is the total EPI readout time in seconds
- Check your MRI protocol document
- Typical values: 0.03-0.08 seconds
- Wrong value → incorrect TOPUP correction!

**How to find readout time:**
1. Check MRI protocol PDF
2. Or calculate: `(EPI factor - 1) × echo spacing`
3. Or extract from DICOM headers (can use config_generator.py)

**GPU Settings:**
- `use_cuda: true` - If NVIDIA GPU with CUDA available
- `use_cuda: false` - If no GPU (will use CPU eddy, slower)

### 6. Functional Settings ⚠️ Check Your Protocol!

```yaml
functional:
  tr: 1.029                # ← CHECK YOUR PROTOCOL!
  te: [10.0, 30.0, 50.0]  # ← CHECK YOUR PROTOCOL!
  highpass: 0.001          # 1000s filter
  lowpass: 0.08            # 12.5s filter
  fwhm: 6                  # 6mm spatial smoothing
  tedana:
    enabled: true          # For multi-echo
    tedpca: 225            # ← IMPORTANT FOR CONVERGENCE!
    tree: kundu
  aroma:
    enabled: auto          # Auto for single-echo
```

**CRITICAL Parameters:**

**TR (Repetition Time):**
- Time between volumes in seconds
- Example: 1.029s, 2.0s, 3.0s
- Check your protocol!

**TE (Echo Time):**
- Single echo: `te: 30.0` (one value)
- Multi-echo: `te: [10.0, 30.0, 50.0]` (list)
- Check your protocol for all echo times!

**TEDANA PCA Components (tedpca):**
- Controls PCA dimensionality reduction before ICA
- **Recommended**: `num_volumes / 2` (e.g., 225 for 450 volumes)
  - Improves ICA convergence
  - Reduces computation time
  - Prevents overfitting
- **Options**:
  - Integer (e.g., `225`): Exact number of components
  - Float 0-1 (e.g., `0.95`): Retain this much variance
  - String `'kundu'`: Auto-selection (may have convergence issues)
  - String `'aic'`, `'kic'`, `'mdl'`: Other auto-selection methods
- **Note**: If TEDANA fails to converge with `'kundu'`, switch to manual (half volumes)

**TEDANA vs AROMA:**
- Multi-echo data → TEDANA (set `tedana.enabled: true`)
- Single-echo data → AROMA (set `aroma.enabled: true`)
- Pipeline auto-detects if `enabled: auto`

### 7. ASL Settings

```yaml
asl:
  labeling_type: pcasl
  labeling_duration: 1.932    # Auto-extracted from DICOM
  post_labeling_delay: 2.031  # Auto-extracted from DICOM
  apply_m0_calibration: true
  wm_cbf_reference: 25.0
```

**Note:** Most ASL parameters are auto-extracted from DICOM headers.
Only change if auto-extraction fails or you need to override.

## Validation

After creating config.yaml:

### 1. Check Syntax

```bash
# Verify YAML is valid
uv run python -c "from mri_preprocess.config import load_config; load_config('config.yaml'); print('✓ Config valid')"
```

### 2. Verify Paths

```bash
# Check all paths exist or can be created
python << 'EOF'
from pathlib import Path
from mri_preprocess.config import load_config

config = load_config('config.yaml')

print("Checking paths...")
for key in ['project_dir', 'dicom_dir']:
    path = Path(config.get(key, ''))
    if path.exists():
        print(f"✓ {key}: {path}")
    else:
        print(f"✗ {key}: {path} (does not exist)")

print("\nDIRECOM directory should exist with subject data!")
print(f"Check: ls {config.get('dicom_dir')}/*")
EOF
```

### 3. Verify Templates

```bash
# Check FSL templates exist
for template in /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz \
                /usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz; do
    if [ -f "$template" ]; then
        echo "✓ $template"
    else
        echo "✗ $template not found"
    fi
done
```

## Complete Template

```yaml
# MRI Preprocessing Pipeline Configuration

# Project paths
project_dir: /path/to/study
dicom_dir: /path/to/study/raw/dicom
bids_dir: /path/to/study/bids
derivatives_dir: /path/to/study/derivatives
work_dir: /path/to/study/work

# Subdirectories
paths:
  logs: /path/to/study/logs
  transforms: /path/to/study/transforms
  qc: /path/to/study/qc

# Execution settings
execution:
  plugin: MultiProc
  n_procs: 6

# Template files (FSL standard)
templates:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
  mni152_t1_1mm: /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
  fmrib58_fa: /usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz

# Anatomical preprocessing
anatomical:
  bet:
    frac: 0.5
    reduce_bias: true
    robust: true
  segmentation_method: ants
  atropos:
    n_iterations: 5
    convergence_threshold: 0.001
    mrf_smoothing_factor: 0.1
  registration_method: fsl
  run_qc: true

# Diffusion preprocessing
diffusion:
  denoise_method: dwidenoise
  topup:
    enabled: auto
    readout_time: 0.05  # CHECK YOUR PROTOCOL!
  eddy_config:
    flm: linear
    slm: linear
    use_cuda: true
  advanced_models:
    enabled: auto
    fit_dki: true
    fit_noddi: true
    fit_sandi: false
    fit_activeax: false
    use_amico: true
  run_qc: true

# Functional preprocessing
functional:
  tr: 1.029  # CHECK YOUR PROTOCOL!
  te: [10.0, 30.0, 50.0]  # CHECK YOUR PROTOCOL!
  highpass: 0.001
  lowpass: 0.08
  fwhm: 6
  normalize_to_mni: true
  tedana:
    enabled: true
    tedpca: 225  # Recommended: num_volumes / 2 (improves ICA convergence)
    tree: kundu
  aroma:
    enabled: auto
  acompcor:
    enabled: true
    num_components: 5
    variance_threshold: 0.5
  run_qc: true

# ASL preprocessing
asl:
  labeling_type: pcasl
  labeling_duration: 1.932
  post_labeling_delay: 2.031
  labeling_efficiency: 0.85
  t1_blood: 1.65
  blood_brain_partition: 0.9
  label_control_order: control_first
  background_suppression_pulses: 1
  apply_m0_calibration: true
  wm_cbf_reference: 25.0
  apply_pvc: false
  normalize_to_mni: false
  run_qc: true

# FreeSurfer (EXPERIMENTAL - NOT PRODUCTION READY)
freesurfer:
  enabled: false
  subjects_dir: /path/to/study/freesurfer
  use_for_tractography: false
  use_for_masks: false
```

## Common Issues

### Issue: "DICOM directory not found"

**Cause:** `dicom_dir` doesn't exist or has wrong path

**Solution:**
```bash
# Check current setting
grep dicom_dir config.yaml

# Verify directory exists
ls /mnt/bytopia/IRC805/raw/dicom/

# Update config if needed
```

### Issue: "Template file not found"

**Cause:** FSL not installed or in different location

**Solution:**
```bash
# Find FSL directory
echo $FSLDIR

# Update template paths in config
templates:
  mni152_t1_2mm: $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz
```

### Issue: "Invalid readout_time"

**Cause:** Wrong or missing readout time for DWI

**Solution:**
1. Check MRI protocol document
2. Contact MRI physicist
3. Or use typical value: 0.05 (but verify!)

## Next Steps

After creating and validating config.yaml:

1. **Place DICOM files**
   ```bash
   # Organize by subject
   /mnt/bytopia/IRC805/raw/dicom/
   ├── SUBJECT-001/
   ├── SUBJECT-002/
   └── ...
   ```

2. **Test on single subject**
   ```bash
   uv run python run_simple_pipeline.py \
       --subject SUBJECT-001 \
       --dicom-dir /mnt/bytopia/IRC805/raw/dicom/SUBJECT-001 \
       --config /mnt/bytopia/IRC805/config.yaml
   ```

3. **Check outputs**
   ```bash
   ls /mnt/bytopia/IRC805/derivatives/SUBJECT-001/
   ```

4. **Run batch if successful**
   ```bash
   uv run python run_batch_simple.py --config /mnt/bytopia/IRC805/config.yaml
   ```

## Additional Resources

- **Main README**: Overview and installation
- **SETUP_GUIDE.md**: Environment setup
- **SIMPLE_PIPELINE_GUIDE.md**: Pipeline usage
- **DEPENDENCIES.md**: Package reference
