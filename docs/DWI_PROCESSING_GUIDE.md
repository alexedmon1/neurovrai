# DWI Processing Guide

This guide describes the enhanced DWI preprocessing pipeline with TOPUP distortion correction, advanced diffusion models (DKI, NODDI), and probabilistic tractography.

## Overview

The pipeline implements a comprehensive multi-shell DWI analysis workflow:

1. **TOPUP Distortion Correction** - Susceptibility distortion correction using reverse phase-encoding images
2. **Multi-shell Merging** - Intelligent concatenation of DWI shells
3. **Eddy Correction** - Motion and eddy current correction with TOPUP integration (GPU-accelerated)
4. **DTI Fitting** - Standard diffusion tensor metrics (FA, MD, etc.)
5. **BEDPOSTX** - Probabilistic fiber orientation modeling (GPU-accelerated)
6. **Advanced Models** - DKI and NODDI for microstructural analysis
7. **Probabilistic Tractography** - Atlas-based connectivity analysis with GPU acceleration

## Key Features

### 1. TOPUP Workflow

**Merge-First Approach (Standard FSL Method):**
- Merges DWI shells BEFORE distortion correction
- Ensures homogeneous correction across all volumes
- Runs TOPUP once on merged b0 volumes
- Eddy uses TOPUP outputs for improved correction

**Benefits:**
- ✅ Consistent correction across all shells
- ✅ Follows FSL best practices
- ✅ Efficient (one TOPUP run)
- ✅ Compatible with eddy TOPUP integration

### 2. GPU Acceleration

**Accelerated Components:**
- `eddy_cuda` - 5-10x faster than CPU version
- `bedpostx_gpu` - 20-50x faster than CPU version
- `probtrackx2_gpu` - 10-50x faster than CPU version

**Requirements:**
- NVIDIA GPU with CUDA support
- FSL GPU tools installed
- CUDA 9.0+ (check with `module load cuda`)

### 3. Advanced Diffusion Models

**DKI (Diffusion Kurtosis Imaging):**
- Extends DTI with kurtosis tensor
- Captures non-Gaussian diffusion
- Metrics: MK, AK, RK, KFA
- Requires: Multi-shell data (≥2 b-values)

**NODDI (Neurite Orientation Dispersion and Density Imaging):**
- Biophysical tissue model
- Metrics: ODI (orientation dispersion), FICVF (neurite density), FISO (free water)
- Requires: Multi-shell data with ≥2 non-zero b-values
- Note: This implementation uses DIPY approximations; for precise NODDI, consider NODDI MATLAB toolbox or AMICO

## Usage

### Step 1: Generate Acquisition Parameter Files

The pipeline requires `acqparams.txt` and `index.txt` for TOPUP/eddy:

```python
from pathlib import Path
from mri_preprocess.utils.topup_helper import create_topup_files_for_multishell

# Example: Two-shell DWI with AP phase encoding
dwi_files = [
    Path('DTI_2shell_b1000_b2000.nii.gz'),
    Path('DTI_1shell_b3000.nii.gz')
]

acqparams, index = create_topup_files_for_multishell(
    dwi_files=dwi_files,
    pe_direction='AP',  # or 'PA', 'LR', 'RL'
    readout_time=0.05,  # Check your protocol (typically 0.04-0.08s)
    output_dir=Path('/study/dwi_params')
)
```

**acqparams.txt format:**
```
0 -1 0 0.05  # AP direction (main DWI)
0 1 0 0.05   # PA direction (reverse PE for TOPUP)
```

**index.txt:**
- One entry per DWI volume
- Maps each volume to line in acqparams.txt
- Example: `1 1 1 ... 1` (all volumes use line 1)

### Step 2: Run DWI Preprocessing with TOPUP

```python
from pathlib import Path
from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing
from mri_preprocess.utils.workflow import load_config

config = load_config('configs/default.yaml')

results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject='sub-001',
    dwi_files=[
        Path('DTI_2shell_b1000_b2000.nii.gz'),
        Path('DTI_1shell_b3000.nii.gz')
    ],
    bval_files=[
        Path('DTI_2shell_b1000_b2000.bval'),
        Path('DTI_1shell_b3000.bval')
    ],
    bvec_files=[
        Path('DTI_2shell_b1000_b2000.bvec'),
        Path('DTI_1shell_b3000.bvec')
    ],
    rev_phase_files=[
        Path('SE_EPI_b1000_b2000_PA.nii.gz'),  # Reverse PE for shell 1
        Path('SE_EPI_b3000_PA.nii.gz')         # Reverse PE for shell 2
    ],
    output_dir=Path('/data/derivatives'),
    work_dir=Path('/data/work'),
    run_bedpostx=True  # Set to True for tractography
)

print(f"FA map: {results['fa']}")
print(f"MD map: {results['md']}")
print(f"Eddy-corrected: {results['eddy_corrected']}")
```

### Step 3: Advanced Diffusion Models (DKI & NODDI)

```python
from pathlib import Path
from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models

# Run after eddy correction
advanced_results = run_advanced_diffusion_models(
    dwi_file=results['eddy_corrected'],
    bval_file=results['merged_bval'],
    bvec_file=results['merged_bvec'],
    mask_file=results['mask'],
    output_dir=Path('/data/derivatives/sub-001/advanced_models'),
    fit_dki=True,
    fit_noddi=True
)

# DKI outputs
print(f"Mean Kurtosis: {advanced_results['dki']['mk']}")
print(f"Axial Kurtosis: {advanced_results['dki']['ak']}")
print(f"Radial Kurtosis: {advanced_results['dki']['rk']}")

# NODDI outputs
print(f"Neurite Density (FICVF): {advanced_results['noddi']['ficvf']}")
print(f"Orientation Dispersion (ODI): {advanced_results['noddi']['odi']}")
print(f"Free Water (FISO): {advanced_results['noddi']['fiso']}")
```

### Step 4: Probabilistic Tractography with Atlas-Based ROIs

```python
from pathlib import Path
from mri_preprocess.workflows.tractography import run_atlas_based_tractography

# Run after BEDPOSTX completes
tract_results = run_atlas_based_tractography(
    config=config,
    subject='sub-001',
    bedpostx_dir=Path('/data/derivatives/sub-001/dwi/bedpostx.bedpostX'),
    dwi_reference=results['fa'],  # Use FA map as reference
    output_dir=Path('/data/derivatives/sub-001/tractography'),
    seed_regions=['hippocampus_l', 'hippocampus_r'],
    target_regions=['thalamus_l', 'thalamus_r', 'frontal_pole_l'],
    atlas='HarvardOxford-subcortical',
    n_samples=5000,
    use_gpu=True  # Use GPU acceleration
)

# Results include connectivity matrices
print(f"Connectivity: {tract_results['connectivity']}")
```

**Available Atlases:**
- `HarvardOxford-cortical` - 48 cortical regions
- `HarvardOxford-subcortical` - 21 subcortical structures
- `JHU-ICBM-tracts-2mm` - 48 white matter tracts

**Common Seed/Target Regions:**
```python
# Subcortical structures
['hippocampus_l', 'hippocampus_r', 'amygdala_l', 'amygdala_r',
 'thalamus_l', 'thalamus_r', 'caudate_l', 'caudate_r',
 'putamen_l', 'putamen_r', 'pallidum_l', 'pallidum_r']

# White matter tracts (JHU atlas)
['genu_of_corpus_callosum', 'body_of_corpus_callosum',
 'splenium_of_corpus_callosum', 'fornix',
 'corticospinal_tract_r', 'corticospinal_tract_l',
 'cingulum_hippocampus_r', 'cingulum_hippocampus_l',
 'superior_longitudinal_fasciculus_r', 'superior_longitudinal_fasciculus_l']
```

## Configuration

Update `configs/default.yaml` with your acquisition parameters:

```yaml
diffusion:
  topup:
    encoding_file: /study/dwi_params/acqparams.txt
    config: b02b0.cnf  # Default TOPUP config (optional)

  eddy:
    acqp_file: /study/dwi_params/acqparams.txt
    index_file: /study/dwi_params/index.txt
    method: jac
    repol: true
    use_cuda: true  # Enable GPU

  bedpostx:
    n_fibres: 2
    n_jumps: 1250
    burn_in: 1000
    use_gpu: true  # Enable GPU
```

## Pipeline Outputs

### Standard DTI Metrics
```
derivatives/mri-preprocess/sub-001/dwi/
├── dti/
│   ├── FA.nii.gz              # Fractional Anisotropy
│   ├── MD.nii.gz              # Mean Diffusivity
│   ├── L1.nii.gz              # Axial Diffusivity
│   ├── L2.nii.gz              # Radial Diffusivity (1)
│   ├── L3.nii.gz              # Radial Diffusivity (2)
│   └── tensor.nii.gz          # Full diffusion tensor
├── eddy_corrected.nii.gz      # Eddy-corrected DWI
├── rotated_bvec               # Rotated b-vectors
└── mask.nii.gz                # Brain mask
```

### DKI Outputs
```
derivatives/mri-preprocess/sub-001/advanced_models/dki/
├── mean_kurtosis.nii.gz       # Mean Kurtosis (MK)
├── axial_kurtosis.nii.gz      # Axial Kurtosis (AK)
├── radial_kurtosis.nii.gz     # Radial Kurtosis (RK)
├── kurtosis_fa.nii.gz         # Kurtosis FA (KFA)
├── kurtosis_tensor.nii.gz     # Full kurtosis tensor
├── fractional_anisotropy.nii.gz
├── mean_diffusivity.nii.gz
├── axial_diffusivity.nii.gz
└── radial_diffusivity.nii.gz
```

### NODDI Outputs
```
derivatives/mri-preprocess/sub-001/advanced_models/noddi/
├── orientation_dispersion_index.nii.gz  # ODI
├── intracellular_volume_fraction.nii.gz # FICVF (neurite density)
├── isotropic_volume_fraction.nii.gz     # FISO (free water)
└── fiber_directions.nii.gz              # Principal fiber orientations
```

### Tractography Outputs
```
derivatives/mri-preprocess/sub-001/tractography/
├── rois/
│   ├── seeds/
│   │   ├── hippocampus_l.nii.gz
│   │   └── hippocampus_r.nii.gz
│   └── targets/
│       ├── thalamus_l.nii.gz
│       └── thalamus_r.nii.gz
└── tractography/
    ├── hippocampus_l/
    │   ├── fdt_paths.nii.gz           # Connectivity distribution
    │   ├── waytotal                    # Total streamlines
    │   └── matrix_seeds_to_all_targets # Connectivity matrix
    └── hippocampus_r/
        └── ...
```

## Troubleshooting

### TOPUP Errors

**Error: "Unable to read encoding file"**
- Check that acqparams.txt exists and has correct format
- Ensure 2 lines minimum (forward and reverse PE)

**Error: "Number of input volumes doesn't match encoding file"**
- For merged b0 (2 volumes), acqparams.txt needs exactly 2 lines
- Check b0 extraction: should have 1 b0 from DWI + 1 b0 from reverse PE

### GPU Errors

**Error: "CUDA not available"**
```bash
# Load CUDA module
module load cuda/11.0  # Or appropriate version

# Check GPU
nvidia-smi

# Test eddy_cuda
which eddy_cuda
```

**Fallback to CPU:**
- Set `use_cuda: false` in config for eddy
- Set `use_gpu: false` for BEDPOSTX and probtrackx2

### Memory Issues

**DKI/NODDI running out of memory:**
- Process subset of slices
- Use brain mask to reduce computation
- Consider using computing cluster

## Future Enhancements

### FreeSurfer Integration (TODO)

In future versions, FreeSurfer-based ROIs will be supported:

```python
# Future API (not yet implemented)
tract_results = run_atlas_based_tractography(
    ...
    use_freesurfer=True,
    subjects_dir='/data/freesurfer',
    atlas='aparc+aseg'  # Use FreeSurfer segmentation
)
```

This will provide:
- Subject-specific anatomical ROIs
- Higher accuracy than atlas-based ROIs
- Full cortical and subcortical parcellation

## References

1. **TOPUP**: Andersson, J. L., et al. (2003). "How to correct susceptibility distortions in spin-echo echo-planar images." Neuroimage.

2. **Eddy**: Andersson, J. L., & Sotiropoulos, S. N. (2016). "An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging." Neuroimage.

3. **DKI**: Jensen, J. H., et al. (2005). "Diffusional kurtosis imaging." MRM.

4. **NODDI**: Zhang, H., et al. (2012). "NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain." Neuroimage.

5. **probtrackx**: Behrens, T. E., et al. (2007). "Probabilistic diffusion tractography with multiple fibre orientations." MRM.

## Support

For questions or issues:
- Check existing code in `mri_preprocess/workflows/`
- Review FSL documentation: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
- DIPY documentation: https://dipy.org/
