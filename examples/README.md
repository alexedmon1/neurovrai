# MRI Preprocessing Examples

This directory contains example scripts and configurations for using the MRI preprocessing pipeline.

## Files

### `example.py`
Complete end-to-end demonstration of the preprocessing pipeline using test subject 0580101.

**What it demonstrates:**
1. Config auto-generation from DICOM headers
2. DICOM to BIDS conversion
3. Data anonymization for safe sharing
4. Anatomical preprocessing (computes T1w→MNI transforms)
5. Diffusion preprocessing (reuses transforms from TransformRegistry)

**Requirements:**
- Test DICOM data at `/mnt/bytopia/IRC805/raw/dicom/IRC805-0580101`
- Output directory at `/mnt/bytopia/development/mri-preprocess/`

**Usage:**
```bash
cd /path/to/mri-preprocess
python examples/example.py
```

### Key Features Demonstrated

#### 1. TransformRegistry (Compute Once, Reuse)
The anatomical workflow computes T1w→MNI152 transformations and saves them:
```python
registry.save_nonlinear_transform(
    warp_file=warp,
    affine_file=affine,
    source_space='T1w',
    target_space='MNI152'
)
```

The diffusion workflow then REUSES these transforms:
```python
if registry.has_transform('T1w', 'MNI152', 'nonlinear'):
    warp, affine = registry.get_nonlinear_transform('T1w', 'MNI152')
    # Apply to FA/MD maps without recomputing!
```

#### 2. Tissue Segmentation for ACompCor
The anatomical workflow outputs tissue probability maps:
```python
outputs = {
    'csf_prob': ...,  # For CSF nuisance regression
    'gm_prob': ...,   # Gray matter
    'wm_prob': ...,   # For WM nuisance regression
}
```

These can be used in functional preprocessing for ACompCor.

#### 3. Config Auto-Generation
```python
auto_generate_config(
    dicom_dir=DICOM_DIR,
    study_name="IRC805 Test Data",
    study_code="IRC805_TEST",
    base_dir=BASE_DIR,
    output_path=CONFIG_FILE
)
```

Automatically detects sequences from DICOM headers and creates a study-specific configuration.

#### 4. Anonymization
```python
anonymize_subject_data(
    rawdata_dir=RAWDATA_DIR,
    subject=SUBJECT_BIDS,
    anonymize_nifti=True
)
```

Removes PHI from JSON sidecars for safe use as example data.

## Quick Start for Other Data

To process your own data:

1. **Create a config file:**
```bash
mri-preprocess config init \
    --dicom-dir /your/dicom/path \
    --study-name "My Study" \
    --study-code MYSTUDY \
    --output configs/mystudy.yaml
```

2. **Review and edit the config:**
```bash
vi configs/mystudy.yaml
```

3. **Convert DICOM to BIDS:**
```bash
mri-preprocess convert \
    --config configs/mystudy.yaml \
    --subject sub-001
```

4. **Run preprocessing:**
```bash
# Just anatomical
mri-preprocess run anatomical \
    --config configs/mystudy.yaml \
    --subject sub-001

# Just diffusion (after anatomical)
mri-preprocess run diffusion \
    --config configs/mystudy.yaml \
    --subject sub-001

# Complete pipeline
mri-preprocess run all \
    --config configs/mystudy.yaml \
    --subject sub-001
```

## Python API

You can also use the Python API directly:

```python
from pathlib import Path
from mri_preprocess.config import load_config
from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing
from mri_preprocess.workflows.dwi_preprocess import run_dwi_preprocessing

# Load config
config = load_config("configs/mystudy.yaml")

# Run anatomical (computes transforms)
anat_results = run_anat_preprocessing(
    config=config,
    subject="sub-001",
    t1w_file=Path("/data/rawdata/sub-001/anat/sub-001_T1w.nii.gz"),
    output_dir=Path("/data/derivatives"),
    work_dir=Path("/tmp/work")
)

# Run diffusion (reuses transforms)
dwi_results = run_dwi_preprocessing(
    config=config,
    subject="sub-001",
    dwi_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.nii.gz"),
    bval_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.bval"),
    bvec_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.bvec"),
    output_dir=Path("/data/derivatives"),
    work_dir=Path("/tmp/work"),
    warp_to_mni=True  # Uses TransformRegistry!
)

# Use tissue masks for functional preprocessing
from mri_preprocess.workflows.func_preprocess import run_func_preprocessing

func_results = run_func_preprocessing(
    config=config,
    subject="sub-001",
    func_file=Path("/data/rawdata/sub-001/func/sub-001_task-rest_bold.nii.gz"),
    output_dir=Path("/data/derivatives"),
    work_dir=Path("/tmp/work"),
    csf_mask=anat_results['csf_prob'],  # For ACompCor
    wm_mask=anat_results['wm_prob']     # For ACompCor
)
```

## Output Structure

After processing, you'll have:

```
/your/base/directory/
├── rawdata/                    # BIDS-organized raw data
│   └── sub-001/
│       ├── anat/
│       │   ├── sub-001_T1w.nii.gz
│       │   └── sub-001_T1w.json
│       └── dwi/
│           ├── sub-001_dwi.nii.gz
│           ├── sub-001_dwi.bval
│           ├── sub-001_dwi.bvec
│           └── sub-001_dwi.json
├── derivatives/                # Preprocessed outputs
│   └── mri-preprocess/
│       └── sub-001/
│           ├── anat/
│           │   ├── brain.nii.gz
│           │   ├── mask.nii.gz
│           │   ├── bias_corrected.nii.gz
│           │   ├── segmentation/    # Tissue masks
│           │   │   ├── pve_0.nii.gz  # CSF
│           │   │   ├── pve_1.nii.gz  # GM
│           │   │   └── pve_2.nii.gz  # WM
│           │   ├── transforms/
│           │   │   ├── affine.mat
│           │   │   └── warp.nii.gz
│           │   └── mni_space/
│           └── dwi/
│               ├── eddy_corrected.nii.gz
│               └── dti/
│                   ├── FA.nii.gz
│                   ├── MD.nii.gz
│                   ├── FA_mni.nii.gz  # Warped using TransformRegistry
│                   └── MD_mni.nii.gz
└── transforms/                 # TransformRegistry
    └── sub-001/
        ├── transforms.json     # Registry metadata
        ├── T1w_to_MNI152_affine.mat
        └── T1w_to_MNI152_warp.nii.gz
```

## Notes

- **TransformRegistry** is in a separate directory so transforms can be shared across different derivative outputs
- **Tissue masks** from anatomical workflow are automatically saved for use in functional preprocessing
- All intermediate files are in the `work/` directory (can be deleted after successful completion)
- Logs are in the `logs/` directory

## Troubleshooting

### "No T1w files found"
Make sure your sequence_mappings in the config includes patterns that match your T1w sequence names:
```yaml
sequence_mappings:
  t1w:
    - "MPRAGE"
    - ".*T1.*3D.*"
```

### "Transform not found in registry"
Run anatomical preprocessing first - it computes and saves the transforms that other workflows need.

### "FSLDIR not set"
Make sure FSL is installed and FSLDIR environment variable is set:
```bash
export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh
```
