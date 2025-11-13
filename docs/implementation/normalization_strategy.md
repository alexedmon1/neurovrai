# Spatial Normalization Strategy

## Overview

This document describes how spatial normalization will be implemented across modalities while **reusing existing anatomical transforms** to avoid duplication.

## Current Status

### Anatomical Preprocessing (✅ Implemented)
- **Affine registration**: FLIRT (T1w → MNI152)
  - Saved to: `derivatives/{subject}/anat/transforms/*_flirt.mat`
- **Nonlinear registration**: FNIRT (T1w → MNI152)
  - Generated in: `work/{subject}/anat_preprocess/reorient/*_warpcoef.nii.gz`
  - **Issue**: Not currently saved to derivatives directory
  - **Action needed**: Modify anatomical workflow to copy warp to transforms/

### DWI Preprocessing (❌ Not Implemented)
- No normalization currently implemented
- Need to add: FA → FMRIB58_FA template

### Functional Preprocessing (❌ Not Implemented)
- No normalization currently implemented
- Need to add: func → anat → MNI152

---

## Normalization Strategy

### 1. Anatomical: T1w → MNI152 (Current Implementation)

**Template**: MNI152_T1_2mm (or 1mm for high-resolution)

**Method**:
- FLIRT for affine (12 DOF)
- FNIRT for nonlinear warping

**Outputs** (to save):
```
derivatives/{subject}/anat/transforms/
├── {subject}_T1w_to_MNI152_affine.mat        # FLIRT affine matrix
└── {subject}_T1w_to_MNI152_warp.nii.gz       # FNIRT warp coefficients
```

**Current Issue**: Warp coefficient exists in work directory but not saved to derivatives

**Fix Required**:
```python
# In anat_preprocess.py, add to DataSink connections:
wf.connect(nonlinear_reg, 'fieldcoeff_file', datasink, 'transforms.@warp')
```

---

### 2. DWI: FA → FMRIB58_FA (To Implement)

**Why separate from anatomical?**
- Different template (FMRIB58_FA is FA-specific, not T1w)
- Different modality contrast
- Cannot reuse anatomical transforms directly

**Template**: `$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz`

**Method**:
```bash
# 1. Affine registration
flirt -in FA.nii.gz \
      -ref $FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz \
      -omat FA_to_FMRIB58_affine.mat \
      -out FA_to_FMRIB58_affine.nii.gz

# 2. Nonlinear registration
fnirt --in=FA.nii.gz \
      --ref=$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz \
      --aff=FA_to_FMRIB58_affine.mat \
      --cout=FA_to_FMRIB58_warp.nii.gz \
      --iout=FA_to_FMRIB58_warped.nii.gz
```

**Outputs** (to save):
```
derivatives/{subject}/dwi/transforms/
├── {subject}_FA_to_FMRIB58_affine.mat        # Affine transform
├── {subject}_FA_to_FMRIB58_warp.nii.gz       # Forward warp (for group analyses)
└── {subject}_FMRIB58_to_FA_warp.nii.gz       # Inverse warp (for tractography ROIs)
```

**Usage**:

*Forward warp (for TBSS, group analyses)*:
```bash
# Apply to FA
applywarp --in=FA.nii.gz \
          --ref=$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz \
          --warp=FA_to_FMRIB58_warp.nii.gz \
          --out=FA_normalized.nii.gz

# Apply to other metrics (MD, AD, RD, MK, AK, RK, ODI, FICVF, etc.)
applywarp --in=MD.nii.gz \
          --ref=$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz \
          --warp=FA_to_FMRIB58_warp.nii.gz \
          --out=MD_normalized.nii.gz
```

*Inverse warp (for tractography ROIs)*:
```bash
# Compute inverse warp
invwarp --ref=FA.nii.gz \
        --warp=FA_to_FMRIB58_warp.nii.gz \
        --out=FMRIB58_to_FA_warp.nii.gz

# Warp atlas ROIs to native DWI space
applywarp --in=atlas_roi.nii.gz \
          --ref=FA.nii.gz \
          --warp=FMRIB58_to_FA_warp.nii.gz \
          --interp=nn \
          --out=atlas_roi_in_dwi.nii.gz
```

**Rationale for FMRIB58_FA**:
- Specifically designed for FA images (better contrast)
- Standard template for TBSS analysis
- Better alignment quality than using T1w→MNI152 transforms
- Maintains subject-specific anatomy for tractography via inverse warp

---

### 3. Functional: func → anat → MNI152 (To Implement, Reuses Anatomical Transforms)

**Why reuse anatomical transforms?**
- Anatomical T1w → MNI152 already computed
- Only need func → anat registration
- Concatenate transforms for final func → MNI152

**IMPORTANT OPTIMIZATION**: The BBR transform (func→anat) is **already computed during ACompCor** for tissue mask registration. We save this transform to the transforms directory and reuse it for normalization, avoiding duplicate computation.

**Method**:

*Step 1: Reuse BBR transform from ACompCor (already computed)*
```bash
# BBR transform already computed during ACompCor step
# Saved to: derivatives/{subject}/func/transforms/{subject}_func_to_anat_bbr.mat
# No need to recompute!
```

*Step 2: Concatenate with anatomical transforms*
```bash
# Combine: func→anat (affine) + anat→MNI152 (affine) + anat→MNI152 (warp)
convertwarp --ref=$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz \
            --premat=func_to_anat.mat \
            --warp1=anat_to_MNI152_warp.nii.gz \
            --out=func_to_MNI152_warp.nii.gz
```

*Step 3: Apply to functional data*
```bash
applywarp --in=func_preprocessed.nii.gz \
          --ref=$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz \
          --warp=func_to_MNI152_warp.nii.gz \
          --out=func_normalized.nii.gz \
          --interp=spline
```

**Outputs** (to save):
```
derivatives/{subject}/func/transforms/
├── {subject}_func_to_anat_bbr.mat            # BBR affine (func→anat)
└── {subject}_func_to_MNI152_warp.nii.gz      # Combined warp (func→MNI152)
```

**Reused from anatomical**:
```
derivatives/{subject}/anat/transforms/
├── {subject}_T1w_to_MNI152_affine.mat        # Reused for concatenation
└── {subject}_T1w_to_MNI152_warp.nii.gz       # Reused for concatenation
```

**Rationale**:
- **No duplication**: Reuses both BBR (from ACompCor) AND anatomical→MNI152 transforms
- **Better alignment**: BBR optimized for EPI→T1w registration
- **Standard approach**: Matches fMRIPrep methodology
- **Two-step strategy**: func→anat→MNI152 is more robust than direct func→MNI152
- **Efficient**: BBR computed once during ACompCor, saved for reuse in normalization

---

## Implementation Plan

### Phase 1: Fix Anatomical Warp Saving (Immediate)

**File**: `mri_preprocess/workflows/anat_preprocess.py`

**Current Issue**: Warp coefficient generated but not saved to derivatives

**Fix**: Add DataSink connection for warp coefficient
```python
# After line ~503 where FNIRT outputs are connected to datasink
wf.connect(nonlinear_reg, 'fieldcoeff_file', datasink, 'transforms.@warp')
```

**Test**:
```bash
# After fix, verify warp is saved
ls -lh /mnt/bytopia/IRC805/derivatives/IRC805-0580101/anat/transforms/*_warp.nii.gz
```

---

### Phase 2: Implement DWI Normalization (✅ COMPLETED)

**File**: `mri_preprocess/utils/dwi_normalization.py` (NEW)

**Functions implemented**:
- `normalize_dwi_to_fmrib58()`: Main normalization function
- `apply_warp_to_metrics()`: Apply warp to all DWI metrics

**Integration**: `mri_preprocess/workflows/dwi_preprocess.py` lines 940-1008

**Outputs**:
```
derivatives/{subject}/dwi/
├── transforms/
│   ├── FA_to_FMRIB58_affine.mat
│   ├── FA_to_FMRIB58_warp.nii.gz       # Forward warp
│   └── FMRIB58_to_FA_warp.nii.gz       # Inverse warp
└── normalized/
    ├── FA_normalized.nii.gz
    ├── MD_normalized.nii.gz
    ├── mean_kurtosis_normalized.nii.gz  # DKI metrics
    ├── ficvf_normalized.nii.gz          # NODDI metrics
    └── ... (all available metrics)
```

**Tested**: IRC805-0580101 - Successfully normalized 12 metrics (FA + 4 DTI + 4 DKI + 3 NODDI)

---

### Phase 3: Implement Functional Normalization (✅ COMPLETED)

**File**: `mri_preprocess/utils/func_normalization.py` (NEW)

**Function implemented**: `normalize_func_to_mni152()`

**Integration**: `mri_preprocess/workflows/func_preprocess.py` lines 834-890

**Transform Reuse Strategy** (zero redundant computation):
1. BBR (func→anat) - REUSED from ACompCor step
2. Affine & warp (anat→MNI152) - REUSED from anatomical preprocessing
3. Concatenate via `convertwarp` for efficient single-step normalization

**Configuration**: Added `normalize_to_mni: true` flag to `config.yaml`

**Outputs**:
```
derivatives/{subject}/func/
├── transforms/
│   ├── {subject}_func_to_anat_bbr.mat       # BBR (saved during ACompCor)
│   └── {subject}_func_to_MNI152_warp.nii.gz # Concatenated warp
├── {subject}_bold_preprocessed.nii.gz        # Native space
└── normalized/
    └── bold_normalized.nii.gz                # MNI152 space
```

**Status**: Implemented, ready for testing on IRC805-0580101

---

## Testing Strategy

### Test Subject: IRC805-0580101

**Anatomical** (already preprocessed):
1. Verify warp file copied to derivatives after fix
2. Check warp quality with FSLeyes overlay

**DWI** (pending):
1. Run dwi_preprocess with normalization enabled
2. Verify FA_normalized overlays on FMRIB58_FA template
3. Test inverse warp: bring JHU atlas to native DWI space
4. Verify all metrics normalized (FA, MD, MK, ODI, etc.)

**Functional** (preprocessing in progress):
1. After functional preprocessing completes, run normalization
2. Verify func_normalized overlays on MNI152 template
3. Check alignment quality at ventricles, cortex boundaries
4. Compare with direct func→MNI152 (expect better with two-step)

---

## Quality Control

**Anatomical**:
- Visual check: T1w_warped.nii.gz overlay on MNI152 template
- Edge alignment at ventricles, cortical sulci

**DWI**:
- Visual check: FA_normalized overlay on FMRIB58_FA template
- White matter tract alignment (corpus callosum, corticospinal tracts)
- Inverse warp check: Harvard-Oxford atlas in native DWI space

**Functional**:
- Visual check: func_normalized overlay on MNI152 template
- BOLD contrast preserved after normalization
- Check time series integrity (no interpolation artifacts)

---

## File Organization

After implementing all normalization:

```
derivatives/{subject}/
├── anat/
│   ├── transforms/
│   │   ├── {subject}_T1w_to_MNI152_affine.mat
│   │   └── {subject}_T1w_to_MNI152_warp.nii.gz
│   └── normalized/
│       └── {subject}_T1w_normalized.nii.gz
├── dwi/
│   ├── transforms/
│   │   ├── {subject}_FA_to_FMRIB58_affine.mat
│   │   ├── {subject}_FA_to_FMRIB58_warp.nii.gz      # Forward
│   │   └── {subject}_FMRIB58_to_FA_warp.nii.gz      # Inverse
│   └── normalized/
│       ├── FA_normalized.nii.gz
│       ├── MD_normalized.nii.gz
│       ├── MK_normalized.nii.gz
│       └── ODI_normalized.nii.gz
└── func/
    ├── transforms/
    │   ├── {subject}_func_to_anat_bbr.mat
    │   └── {subject}_func_to_MNI152_warp.nii.gz
    └── normalized/
        └── {subject}_bold_normalized.nii.gz
```

---

## Key Decisions Summary

1. **Anatomical**: Use existing FLIRT/FNIRT to MNI152 (just need to save warp)
2. **DWI**: Separate FA→FMRIB58_FA normalization (cannot reuse anatomical due to different template/contrast)
3. **Functional**: **Reuse anatomical transforms** via two-step concatenation (func→anat→MNI152)
4. **Tractography**: Use inverse DWI warp to bring atlases to native space (maintains subject anatomy)
5. **Storage**: Save both native and normalized versions for flexibility

This strategy **minimizes duplication** by reusing anatomical transforms for functional normalization while maintaining modality-appropriate templates for DWI.
