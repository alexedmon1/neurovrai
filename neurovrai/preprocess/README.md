# neurovrai.preprocess

**Subject-level MRI preprocessing workflows built on Nipype, FSL, and ANTs.**

## Overview

This module provides production-ready preprocessing pipelines for four MRI modalities:

| Modality | Workflow | Key Features |
|----------|----------|--------------|
| **Anatomical** | `anat_preprocess.py` | N4 bias correction, BET, Atropos segmentation, MNI registration |
| **Diffusion** | `dwi_preprocess.py` | TOPUP, GPU eddy, DTI/DKI/NODDI, FMRIB58 normalization |
| **Functional** | `func_preprocess.py` | TEDANA (multi-echo), ICA-AROMA (single-echo), ACompCor, bandpass |
| **ASL** | `asl_preprocess.py` | CBF quantification, M0 calibration, PVC |

## Module Structure

```
preprocess/
├── workflows/              # Main preprocessing pipelines
│   ├── anat_preprocess.py     # Anatomical T1w/T2w
│   ├── dwi_preprocess.py      # Diffusion MRI
│   ├── func_preprocess.py     # Functional fMRI
│   ├── asl_preprocess.py      # Arterial spin labeling
│   ├── advanced_diffusion.py  # DKI, NODDI (DIPY)
│   └── amico_models.py        # AMICO-accelerated models
│
├── utils/                  # Helper functions
│   ├── topup_helper.py        # TOPUP parameter files
│   ├── dwi_normalization.py   # DWI spatial normalization
│   ├── func_normalization.py  # fMRI MNI normalization
│   ├── func_registration.py   # Functional registration
│   ├── acompcor_helper.py     # ACompCor confound extraction
│   ├── asl_cbf.py             # CBF quantification
│   ├── dicom_asl_params.py    # ASL parameter extraction
│   ├── freesurfer_utils.py    # FreeSurfer integration
│   └── freesurfer_transforms.py # FreeSurfer transforms
│
├── qc/                     # Quality control modules
│   ├── anat/                  # Anatomical QC
│   │   ├── skull_strip_qc.py
│   │   ├── segmentation_qc.py
│   │   └── registration_qc.py
│   ├── dwi/                   # Diffusion QC
│   │   ├── motion_qc.py
│   │   ├── topup_qc.py
│   │   ├── dti_qc.py
│   │   └── skull_strip_qc.py
│   ├── func_qc.py             # Functional QC
│   └── asl_qc.py              # ASL QC
│
├── dicom/                  # DICOM conversion
│   ├── converter.py           # dcm2niix wrapper
│   ├── bids_converter.py      # BIDS organization
│   └── anonymize.py           # DICOM anonymization
│
├── cli.py                  # Command-line interface
└── config_generator.py     # Config file generation
```

---

## Anatomical Preprocessing

**Pipeline**: Reorient → N4 Bias Correction → BET Skull Strip → Atropos Segmentation → FLIRT/FNIRT Registration

### Usage

```python
from pathlib import Path
from neurovrai.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing

results = run_anatomical_preprocessing(
    config=config,
    subject='sub-001',
    t1w_file=Path('T1w.nii.gz'),
    t2w_file=Path('T2w.nii.gz'),  # Optional
    output_dir=Path('/derivatives')
)
```

### Configuration

```yaml
anatomical:
  bet:
    frac: 0.5
    reduce_bias: true
    robust: true
  segmentation:
    method: atropos  # ANTs Atropos (fast)
    n_classes: 3     # CSF, GM, WM
  registration_method: fsl  # or 'ants'
  run_qc: true
```

### Outputs

```
derivatives/{subject}/anat/
├── brain.nii.gz              # Skull-stripped brain
├── brain_mask.nii.gz         # Binary brain mask
├── bias_corrected.nii.gz     # N4 bias-corrected T1w
├── segmentation/
│   ├── pve_0.nii.gz          # CSF probability
│   ├── pve_1.nii.gz          # GM probability
│   └── pve_2.nii.gz          # WM probability
├── transforms/
│   ├── anat2mni_affine.mat   # FLIRT affine
│   ├── anat2mni_warp.nii.gz  # FNIRT warp
│   └── mni2anat_warp.nii.gz  # Inverse warp
└── qc/
    └── skull_strip_qc.html   # QC report
```

---

## Diffusion Preprocessing

**Pipeline**: TOPUP Distortion Correction → GPU Eddy → DTI Fitting → Advanced Models → Spatial Normalization

### Usage

```python
from pathlib import Path
from neurovrai.preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject='sub-001',
    dwi_files=[Path('dwi.nii.gz')],
    bval_files=[Path('dwi.bval')],
    bvec_files=[Path('dwi.bvec')],
    rev_phase_files=[Path('SE_EPI_PA.nii.gz')],  # Optional TOPUP
    output_dir=Path('/derivatives'),
    run_advanced_models=True  # Enable DKI/NODDI
)
```

### Configuration

```yaml
diffusion:
  denoise_method: dwidenoise
  topup:
    readout_time: 0.05
    pe_direction: AP  # or PA, LR, RL
  eddy_config:
    flm: linear
    slm: linear
    use_cuda: true    # GPU acceleration
  dti:
    fit_method: WLS   # Weighted least squares
  advanced_models:
    dki: true         # Multi-shell only
    noddi: true       # Multi-shell only
    amico: true       # Use AMICO (100x faster)
  normalize_to_template: true
  run_qc: true
```

### Outputs

```
derivatives/{subject}/dwi/
├── eddy_corrected.nii.gz     # Preprocessed DWI
├── eddy_corrected.bval       # b-values
├── eddy_corrected.bvec       # Rotated b-vectors
├── dwi_mask.nii.gz           # Brain mask
├── dti/
│   ├── FA.nii.gz             # Fractional anisotropy
│   ├── MD.nii.gz             # Mean diffusivity
│   ├── AD.nii.gz             # Axial diffusivity
│   └── RD.nii.gz             # Radial diffusivity
├── dki/                      # Multi-shell only
│   ├── MK.nii.gz             # Mean kurtosis
│   ├── AK.nii.gz             # Axial kurtosis
│   └── RK.nii.gz             # Radial kurtosis
├── noddi/                    # Multi-shell only
│   ├── ficvf.nii.gz          # Neurite density
│   ├── odi.nii.gz            # Orientation dispersion
│   └── fiso.nii.gz           # Isotropic fraction
├── normalized/
│   ├── FA_FMRIB58.nii.gz     # FA in FMRIB58 space
│   └── ...
└── qc/
    ├── motion_qc.html        # Motion parameters
    ├── topup_qc.html         # Distortion correction
    └── dti_qc.html           # DTI metrics
```

### AMICO-Accelerated Models

For 100x faster NODDI fitting:

```python
from neurovrai.preprocess.workflows.amico_models import fit_noddi_amico

results = fit_noddi_amico(
    dwi_file=Path('eddy_corrected.nii.gz'),
    bval_file=Path('dwi.bval'),
    bvec_file=Path('dwi_rotated.bvec'),
    mask_file=Path('dwi_mask.nii.gz'),
    output_dir=Path('noddi_output')
)
# Runtime: ~30 seconds (vs 20-25 min with DIPY)
```

---

## Functional Preprocessing

**Pipeline**: Motion Correction → Denoising → ACompCor → Bandpass Filter → Smoothing → Registration

### Multi-Echo vs Single-Echo

| Feature | Multi-Echo | Single-Echo |
|---------|------------|-------------|
| **Denoising** | TEDANA (ICA-based) | ICA-AROMA |
| **Motion artifacts** | T2* decay separation | Independent components |
| **Requirements** | Multiple echo times | ICA-AROMA installed |

### Usage

```python
from pathlib import Path
from neurovrai.preprocess.workflows.func_preprocess import run_functional_preprocessing

# Multi-echo (auto-detected)
results = run_functional_preprocessing(
    config=config,
    subject='sub-001',
    func_files=[
        Path('echo1.nii.gz'),
        Path('echo2.nii.gz'),
        Path('echo3.nii.gz')
    ],
    output_dir=Path('/derivatives'),
    t1w_brain=Path('derivatives/sub-001/anat/brain.nii.gz'),
    normalize_to_mni=True
)

# Single-echo
results = run_functional_preprocessing(
    config=config,
    subject='sub-001',
    func_files=[Path('bold.nii.gz')],
    output_dir=Path('/derivatives'),
    t1w_brain=Path('derivatives/sub-001/anat/brain.nii.gz'),
    normalize_to_mni=True
)
```

### Configuration

```yaml
functional:
  tr: 2.0
  te: [14.0, 29.0, 44.0]  # Multi-echo TEs (ms)
  highpass: 0.001         # Hz
  lowpass: 0.08           # Hz
  fwhm: 6                 # Smoothing (mm)
  tedana:
    enabled: true         # Auto for multi-echo
    tedpca: kundu
    tree: kundu
  aroma:
    enabled: true         # Auto for single-echo
  acompcor:
    enabled: true
    num_components: 5
    variance_threshold: 0.5
  run_qc: true
```

### Outputs

```
derivatives/{subject}/func/
├── preprocessed_bold.nii.gz  # Final preprocessed fMRI
├── brain_mask.nii.gz         # Functional brain mask
├── mean_bold.nii.gz          # Temporal mean
├── tsnr.nii.gz               # Temporal SNR map
├── tedana/                   # Multi-echo outputs
│   ├── desc-optcom_bold.nii.gz
│   └── ...
├── transforms/
│   ├── func2anat.mat         # Functional → T1w
│   └── func2mni_warp.nii.gz  # Functional → MNI
└── qc/
    ├── motion_qc.html        # Motion parameters
    └── tsnr_qc.html          # tSNR maps
```

---

## ASL Preprocessing

**Pipeline**: Motion Correction → Label-Control Separation → CBF Quantification → M0 Calibration → PVC

### Usage

```python
from pathlib import Path
from neurovrai.preprocess.workflows.asl_preprocess import run_asl_preprocessing

results = run_asl_preprocessing(
    config=config,
    subject='sub-001',
    asl_file=Path('asl.nii.gz'),
    output_dir=Path('/derivatives'),
    t1w_brain=Path('derivatives/sub-001/anat/brain.nii.gz'),
    gm_mask=Path('derivatives/sub-001/anat/segmentation/pve_1.nii.gz'),
    wm_mask=Path('derivatives/sub-001/anat/segmentation/pve_2.nii.gz'),
    dicom_dir=Path('rawdata/sub-001/asl'),  # Auto parameter extraction
    normalize_to_mni=True
)
```

### Configuration

```yaml
asl:
  labeling_type: pCASL          # or PASL, CASL
  labeling_duration: 1.8        # seconds
  post_labeling_delay: 2.0      # seconds
  label_control_order: control-label
  m0_scale: 1.0
  partition_coefficient: 0.9    # Blood-brain
  t1_blood: 1.65                # seconds at 3T
  labeling_efficiency: 0.85
  run_pvc: true                 # Partial volume correction
  run_qc: true
```

### Outputs

```
derivatives/{subject}/asl/
├── cbf.nii.gz                # Cerebral blood flow map
├── cbf_pvc.nii.gz            # PVC-corrected CBF
├── m0.nii.gz                 # M0 calibration image
├── cbf_stats.json            # Tissue-specific CBF
├── transforms/
│   └── asl2anat.mat          # ASL → T1w
└── qc/
    ├── motion_qc.html        # Motion parameters
    └── cbf_qc.html           # CBF distributions
```

---

## Quality Control

All workflows generate HTML QC reports with:

| Modality | QC Metrics |
|----------|------------|
| **Anatomical** | Skull stripping overlay, tissue segmentation, MNI registration |
| **Diffusion** | Motion parameters (eddy), TOPUP correction, DTI metrics, skull stripping |
| **Functional** | Motion parameters (MCFLIRT), tSNR maps, DVARS, skull stripping |
| **ASL** | Motion parameters, CBF distributions, tSNR, skull stripping |

### QC Module Usage

```python
from neurovrai.preprocess.qc.dwi.motion_qc import DWIMotionQC

qc = DWIMotionQC(
    subject='sub-001',
    dwi_dir=Path('/derivatives/sub-001/dwi'),
    qc_dir=Path('/qc/sub-001/dwi/motion')
)
results = qc.run_qc()
```

---

## Utilities

### TOPUP Parameter Generation

```python
from neurovrai.preprocess.utils.topup_helper import create_topup_files_for_multishell

acqparams, index = create_topup_files_for_multishell(
    dwi_files=[Path('b1000.nii.gz'), Path('b2000.nii.gz')],
    pe_direction='AP',
    readout_time=0.05,
    output_dir=Path('/dwi_params')
)
```

### DWI Normalization

```python
from neurovrai.preprocess.utils.dwi_normalization import normalize_dti_to_template

normalize_dti_to_template(
    fa_file=Path('FA.nii.gz'),
    template=Path('$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz'),
    output_dir=Path('normalized'),
    other_metrics=['MD.nii.gz', 'AD.nii.gz', 'RD.nii.gz']
)
```

### ACompCor Confounds

```python
from neurovrai.preprocess.utils.acompcor_helper import run_acompcor

confounds = run_acompcor(
    func_file=Path('preprocessed_bold.nii.gz'),
    wm_mask=Path('wm_mask.nii.gz'),
    csf_mask=Path('csf_mask.nii.gz'),
    n_components=5,
    variance_threshold=0.5
)
```

---

## Command-Line Interface

```bash
# Main pipeline runner
uv run python run_simple_pipeline.py \
    --subject sub-001 \
    --dicom-dir /rawdata/sub-001 \
    --config config.yaml

# Skip specific modalities
uv run python run_simple_pipeline.py \
    --subject sub-001 \
    --dicom-dir /rawdata/sub-001 \
    --config config.yaml \
    --skip-func --skip-asl

# Generate config file
uv run python create_config.py --study-root /mnt/data/my_study
```

---

## Performance

| Process | CPU | GPU | Speedup |
|---------|-----|-----|---------|
| Eddy correction | ~45 min | ~4 min | 10x |
| NODDI (DIPY) | 20-25 min | - | - |
| NODDI (AMICO) | 30 sec | - | 100x |
| TEDANA | 60-90 min | - | - |
| BET | ~2 min | - | - |
| FNIRT | ~15-20 min | - | - |

**Recommended Hardware**: 8+ cores, 32+ GB RAM, NVIDIA GPU with CUDA

---

## Dependencies

- **FSL 6.0+**: BET, FLIRT, FNIRT, TOPUP, eddy, FAST
- **ANTs**: N4BiasFieldCorrection, Atropos
- **Nipype**: Workflow engine
- **DIPY**: DTI, DKI, NODDI fitting
- **TEDANA**: Multi-echo denoising
- **AMICO**: Accelerated microstructure models
- **ICA-AROMA**: Single-echo motion artifact removal (optional)
