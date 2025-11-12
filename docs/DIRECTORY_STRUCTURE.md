# Directory Structure

All preprocessing workflows now use a standardized directory hierarchy with clear separation between raw data, conversions, derivatives, and working files.

## Standard Structure

```
{study_root}/                          # e.g., /mnt/bytopia/development/IRC805/
├── dicoms/                            # Raw DICOM files
│   └── {subject}/
│       ├── scan_001/
│       └── scan_002/
│
├── nifti/                             # Converted NIfTI files (from DICOM)
│   └── {subject}/
│       ├── anat/
│       │   ├── sub-{subject}_T1w.nii.gz
│       │   └── sub-{subject}_T2w.nii.gz
│       ├── dwi/
│       │   ├── sub-{subject}_dwi-b1000-b2000.nii.gz
│       │   ├── sub-{subject}_dwi-b1000-b2000.bval
│       │   ├── sub-{subject}_dwi-b1000-b2000.bvec
│       │   └── sub-{subject}_dwi-b3000.nii.gz
│       └── func/
│           └── sub-{subject}_task-rest_bold.nii.gz
│
├── qc/                                # Quality Control outputs (centralized)
│   ├── dwi/                           # DWI QC
│   │   └── {subject}/
│   │       ├── reports/               # HTML/PDF reports
│   │       ├── metrics/               # JSON/CSV metrics
│   │       ├── images/                # Visualization images
│   │       ├── motion/                # Motion QC
│   │       ├── topup/                 # TOPUP QC
│   │       └── comparisons/           # Before/after
│   ├── rest/                          # Resting-state fMRI QC
│   │   └── {subject}/
│   │       ├── reports/
│   │       ├── metrics/
│   │       ├── images/
│   │       ├── motion/
│   │       ├── carpet_plots/
│   │       └── ica_aroma/
│   ├── anat/                          # Anatomical QC
│   │   └── {subject}/
│   │       ├── reports/
│   │       ├── metrics/
│   │       ├── images/
│   │       ├── segmentation/
│   │       └── registration/
│   └── cross_modality/                # Cross-modality registration QC
│       └── {subject}/
│           ├── dwi_to_t1w/
│           ├── rest_to_t1w/
│           └── reports/
│
├── derivatives/                       # Preprocessed outputs
│   ├── anat_preproc/                 # Anatomical preprocessing
│   │   └── {subject}/
│   │       └── anat/
│   │           ├── *_brain.nii.gz
│   │           ├── *_brain_mask.nii.gz
│   │           ├── *_bias_corrected.nii.gz
│   │           ├── segmentation/
│   │           │   ├── *_pve_0.nii.gz  # CSF
│   │           │   ├── *_pve_1.nii.gz  # GM
│   │           │   └── *_pve_2.nii.gz  # WM
│   │           └── transforms/
│   │               ├── *_to_MNI.mat
│   │               └── *_to_MNI_warp.nii.gz
│   │
│   ├── dwi_topup/                    # DWI preprocessing with TOPUP
│   │   └── {subject}/
│   │       ├── eddy_corrected/
│   │       │   └── eddy_corrected.nii.gz
│   │       ├── dti/
│   │       │   ├── dtifit__FA.nii.gz
│   │       │   ├── dtifit__MD.nii.gz
│   │       │   ├── dtifit__L1.nii.gz
│   │       │   ├── dtifit__L2.nii.gz
│   │       │   └── dtifit__L3.nii.gz
│   │       ├── mask/
│   │       │   └── dwi_merged_roi_brain_mask.nii.gz
│   │       ├── rotated_bvec/
│   │       │   └── eddy_corrected.eddy_rotated_bvecs
│   │       └── bedpostx/             # Optional
│   │           └── bedpostx.bedpostX/
│   │
│   ├── advanced_diffusion/           # DKI and NODDI
│   │   └── {subject}/
│   │       ├── dki/
│   │       │   ├── mean_kurtosis.nii.gz
│   │       │   ├── axial_kurtosis.nii.gz
│   │       │   └── radial_kurtosis.nii.gz
│   │       └── noddi/
│   │           ├── orientation_dispersion_index.nii.gz
│   │           ├── intracellular_volume_fraction.nii.gz
│   │           └── isotropic_volume_fraction.nii.gz
│   │
│   └── tractography/                 # Probabilistic tractography
│       └── {subject}/
│           ├── rois/
│           │   ├── seeds/
│           │   └── targets/
│           └── tractography/
│               └── {seed_name}/
│                   ├── fdt_paths.nii.gz
│                   └── matrix_seeds_to_all_targets
│
├── work/                              # Temporary Nipype working files
│   └── {subject}/
│       ├── anat_preproc/
│       │   └── workflow/
│       └── dwi_topup/
│           ├── dwi_merged.nii.gz
│           ├── topup_results_fieldcoef.nii.gz
│           └── workflow/
│
└── dwi_params/                        # Acquisition parameter files
    ├── acqparams.txt
    └── index.txt
```

## Usage

### Anatomical Preprocessing

```python
from pathlib import Path
from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing

# Study root directory
study_root = Path('/mnt/bytopia/development/IRC805')

results = run_anat_preprocessing(
    config=config,
    subject='IRC805-0580101',
    t1w_file=Path('/path/to/T1w.nii.gz'),
    output_dir=study_root,  # Study root
    work_dir=None  # Auto: {study_root}/work/{subject}/anat_preproc/
)

# Outputs in: {study_root}/derivatives/anat_preproc/IRC805-0580101/anat/
```

### DWI Preprocessing with TOPUP

```python
from pathlib import Path
from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

# Study root directory
study_root = Path('/mnt/bytopia/development/IRC805')

results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject='IRC805-0580101',
    dwi_files=[...],
    bval_files=[...],
    bvec_files=[...],
    rev_phase_files=[...],
    output_dir=study_root,  # Study root
    work_dir=None  # Auto: {study_root}/work/{subject}/dwi_topup/
)

# Outputs in: {study_root}/derivatives/dwi_topup/IRC805-0580101/
```

## Benefits

1. **Clear Organization**: Separate directories for raw data, conversions, derivatives, and QC
2. **Centralized QC**: All quality control outputs in one location for easy review
3. **Modality-Specific QC**: DWI, anatomical, and functional QC organized separately
4. **Cross-Modality QC**: Dedicated space for registration QC across modalities
5. **Easy Cleanup**: Temporary `work/` directory can be deleted after successful preprocessing
6. **Consistent Structure**: All workflows follow the same pattern
7. **BIDS-Compatible**: Structure aligns with BIDS derivatives specification
8. **Subject-Level**: Each subject has isolated derivative directories
9. **Group QC**: Easy to aggregate QC metrics across subjects from centralized location

## Migration from Old Structure

Old structure used:
```
/mnt/bytopia/IRC805/
├── subjects/{subject}/nifti/...
└── derivatives/mri-preprocess/sub-{subject}/...
```

New structure uses:
```
/mnt/bytopia/development/IRC805/
├── nifti/{subject}/...
└── derivatives/{workflow}/{subject}/...
```

The new structure is cleaner and allows for better organization of multiple preprocessing pipelines.
