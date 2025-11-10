# Workflow Documentation

This document describes the preprocessing workflows for each MRI modality.

**Note**: This documentation will be expanded as each workflow is implemented.

---

## Table of Contents

1. [Anatomical (T1w/T2w)](#anatomical-preprocessing)
2. [Diffusion (DWI/DTI)](#diffusion-preprocessing)
3. [Functional (fMRI)](#functional-preprocessing)
4. [Myelin Mapping](#myelin-mapping)

---

## Anatomical Preprocessing

**Status**: 🚧 To be implemented in Phase 5

### Overview
Preprocesses T1-weighted (and optionally T2-weighted) structural images for:
- Bias field correction
- Skull stripping
- Registration to MNI152 standard space
- Segmentation (optional, for nuisance regression)

### Input Requirements

**Mandatory**:
- T1w image (NIfTI format)

**Optional**:
- T2w image (for improved processing)

### Output Files

- `{subject}_T1w_biascorrected.nii.gz` - Bias-corrected T1w
- `{subject}_T1w_brain.nii.gz` - Skull-stripped brain
- `{subject}_T1w_to_MNI_affine.mat` - Affine transformation matrix
- `{subject}_T1w_to_MNI_warp.nii.gz` - Nonlinear warp field
- `{subject}_T1w_MNI.nii.gz` - T1w in MNI space
- Segmentation files (if enabled)

### Configuration Parameters

See `docs/configuration.md` under "Anatomical Preprocessing"

### Pipeline Steps

1. Reorient to standard orientation
2. FAST bias field correction
3. BET skull stripping
4. FLIRT affine registration to MNI152
5. FNIRT nonlinear registration to MNI152
6. FAST tissue segmentation (if enabled)

### Transform Registry

**Saves to TransformRegistry**:
- T1w→MNI affine matrix
- T1w→MNI warp field

**Used by other workflows**: Diffusion, Functional, Myelin

### Usage

```bash
mri-preprocess anat \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bids-dir /path/to/rawdata \
  --output-dir /path/to/derivatives
```

---

## Diffusion Preprocessing

**Status**: 🚧 To be implemented in Phase 6

### Overview
Preprocesses diffusion-weighted imaging (DWI) data for:
- Eddy current and motion correction
- DTI fitting (FA, MD, etc.)
- Fiber orientation estimation (BEDPOSTX)
- Probabilistic tractography (probtrackx2)

### Input Requirements

**Mandatory**:
- DWI image(s) (NIfTI format)
- bval file (b-values)
- bvec file (gradient directions)
- Acquisition parameters file (acqp.txt)
- Index file (index.txt)

**Optional**:
- Multiple shells (for multi-shell DTI)
- Fieldmaps (for distortion correction)

### Output Files

- `{subject}_dwi_corrected.nii.gz` - Eddy-corrected DWI
- `{subject}_dtifit_FA.nii.gz` - Fractional anisotropy map
- `{subject}_dtifit_MD.nii.gz` - Mean diffusivity
- `{subject}_dtifit_RD.nii.gz` - Radial diffusivity
- `{subject}_dtifit_AD.nii.gz` - Axial diffusivity
- BEDPOSTX outputs (if enabled)
- Connectivity matrices (if probtrackx2 enabled)

### Configuration Parameters

See `docs/configuration.md` under "Diffusion Preprocessing"

### Pipeline Steps

1. Merge multiple shells (if multi-shell)
2. Extract b0 volume
3. BET on b0
4. Eddy current and motion correction (GPU-accelerated)
5. DTIFit for tensor metrics
6. BEDPOSTX for fiber distributions (optional)
7. probtrackx2 for tractography (optional)

### Transform Usage

**Loads from TransformRegistry**:
- T1w→MNI transformations (from anatomical workflow)

**Computes**:
- DWI→T1w transformation
- Concatenates: DWI→T1w→MNI

### Usage

```bash
mri-preprocess dwi \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bids-dir /path/to/rawdata \
  --output-dir /path/to/derivatives
```

---

## Functional Preprocessing

**Status**: 🚧 To be implemented in Phase 7

### Overview
Preprocesses resting-state fMRI data with:
- Multi-echo support with TEDANA denoising
- Motion correction
- Spatial normalization to MNI
- Temporal filtering
- Nuisance regression (ICA-AROMA, ACompCor)

### Input Requirements

**Mandatory**:
- Functional image(s) (NIfTI format)
  - Single-echo: 1 file
  - Multi-echo: 3+ files

**Optional**:
- Fieldmaps (for distortion correction)

### Output Files

- `{subject}_func_corrected.nii.gz` - Motion-corrected
- `{subject}_func_tedana_denoised.nii.gz` - TEDANA output (multi-echo only)
- `{subject}_func_MNI.nii.gz` - In MNI space
- `{subject}_func_smoothed.nii.gz` - Spatially smoothed
- `{subject}_func_preprocessed.nii.gz` - Fully preprocessed
- Motion parameters, ICA-AROMA classifications, ACompCor components

### Configuration Parameters

See `docs/configuration.md` under "Functional Preprocessing"

### Pipeline Steps

**Multi-echo path**:
1. Reorient each echo
2. Skull strip (middle echo)
3. Motion correction (middle echo as reference)
4. Apply motion transforms to all echoes
5. **TEDANA denoising** (critical step!)
6. Coregister to T1w
7. Apply concatenated transform to MNI
8. Spatial smoothing
9. ICA-AROMA
10. ACompCor nuisance regression
11. Temporal filtering (bandpass)

**Single-echo path**:
Similar but skips TEDANA (steps 1-4, 6-11)

### Transform Usage

**Loads from TransformRegistry**:
- T1w→MNI transformations (from anatomical workflow)
- T1w segmentation (for ACompCor masks)

**Computes**:
- Func→T1w transformation
- Concatenates: Func→T1w→MNI

### Usage

```bash
mri-preprocess func \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bids-dir /path/to/rawdata \
  --output-dir /path/to/derivatives
```

---

## Myelin Mapping

**Status**: 🚧 To be implemented in Phase 8

### Overview
Generates T1w/T2w ratio maps as a proxy for myelin content.

### Input Requirements

**Mandatory**:
- T1w image (from anatomical preprocessing)
- T2w image

### Output Files

- `{subject}_T2w_to_T1w.nii.gz` - T2w coregistered to T1w
- `{subject}_myelin_map.nii.gz` - T1w/T2w ratio (native space)
- `{subject}_myelin_map_MNI.nii.gz` - Myelin map in MNI space

### Configuration Parameters

Minimal parameters needed (uses anatomical defaults)

### Pipeline Steps

1. Coregister T2w to T1w space
2. Compute T1w/T2w ratio
3. Apply T1w→MNI transform to myelin map

### Transform Usage

**Loads from TransformRegistry**:
- T1w→MNI transformations (from anatomical workflow - no recomputation!)

### Usage

```bash
mri-preprocess myelin \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --bids-dir /path/to/rawdata \
  --output-dir /path/to/derivatives
```

---

## Full Pipeline

Run all workflows in sequence:

```bash
mri-preprocess run \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --steps all
```

Or specific steps:

```bash
mri-preprocess run \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --steps anat,dwi,func
```

---

## Quality Control

After preprocessing, perform visual QC:

```bash
# View in FSLeyes
fsleyes \
  derivatives/sub-001/anat/sub-001_T1w_MNI.nii.gz \
  $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz
```

Check workflow graphs:

```bash
# View workflow visualization
eog derivatives/sub-001/workflow_graphs/workflow_graph.png
```

---

## Troubleshooting

### Common Issues

**Preprocessing fails**:
1. Check logs in `{output_dir}/logs/`
2. Inspect Nipype crash files
3. Enable DEBUG logging in config

**Poor registration quality**:
- Adjust BET frac parameter for better skull stripping
- Try different cost functions
- Check for motion artifacts in original data

**TEDANA fails**:
- Verify all echoes have same dimensions
- Check echo times are correctly ordered
- Try different tedpca methods

---

**Note**: This documentation will be updated with detailed examples and screenshots as workflows are implemented and tested.
