# MRI Preprocessing Pipeline - Session Status

**Last Updated:** 2025-11-11

## Current Session Summary

### Completed: Anatomical QC Framework ✅

Implemented a complete Quality Control suite for anatomical preprocessing with three modules:

#### 1. Skull Strip QC (`mri_preprocess/qc/anat/skull_strip_qc.py`)
- ✅ Brain mask statistics (volume, voxel count, bounding box)
- ✅ Quality checks (contrast ratio, intensity variance)
- ✅ Visual overlays (mask contours on T1w)
- ✅ JSON metrics export
- ✅ Test script: `test_anat_qc.py`

#### 2. Segmentation QC (`mri_preprocess/qc/anat/segmentation_qc.py`)
- ✅ Tissue volume calculations (GM/WM/CSF)
- ✅ Tissue fraction validation against physiological ranges
- ✅ GM/WM ratio assessment
- ✅ Bar charts and pie charts for tissue volumes
- ✅ JSON metrics export
- ✅ Test script: `test_segmentation_qc.py`

#### 3. Registration QC (`mri_preprocess/qc/anat/registration_qc.py`)
- ✅ Alignment metrics (Pearson correlation, NCC, Dice coefficient)
- ✅ Error metrics (MAD, RMSE)
- ✅ Edge overlay visualization (red=registered, green=template)
- ✅ Checkerboard overlay for visual inspection
- ✅ Auto-detection of registered files and MNI152 templates
- ✅ JSON metrics export
- ✅ Test script: `test_registration_qc.py`

#### 4. Integrated QC Runner
- ✅ Complete test suite: `test_anat_qc_complete.py`
- ✅ Data finder utility: `find_anat_data.py`
- ✅ Automatic file detection across standard directory structures
- ✅ Combined JSON results output
- ✅ Comprehensive summary reporting

### Test Results

All QC modules validated with synthetic data:
- **Skull Strip QC**: Brain volume calculation, mask quality checks ✅
- **Segmentation QC**: Tissue volume validation, physiological range checks ✅
- **Registration QC**: Intentional misalignment detected correctly (correlation: 0.74, Dice: 0.94) ✅

---

## Next Session Plan

### PRIORITY 1: Complete Anatomical Pipeline with Integrated QC

**Goal:** Generate anatomical preprocessing data and validate QC integration

#### Step 1: Run Anatomical Preprocessing
- Use `mri_preprocess/workflows/anat_preprocess.py`
- Process at least one IRC805 subject to generate:
  - T1w bias-corrected
  - Brain-extracted (BET)
  - Brain mask
  - Tissue segmentations (FAST: CSF, GM, WM)
  - Registered to MNI152 (FLIRT/FNIRT)

#### Step 2: Integrate QC into Anatomical Workflow
- **File to modify:** `mri_preprocess/workflows/anat_preprocess.py`
- Add automatic QC execution after each preprocessing stage:
  ```python
  # After BET
  from mri_preprocess.qc.anat.skull_strip_qc import SkullStripQualityControl
  skull_qc = SkullStripQualityControl(subject, anat_dir, qc_dir)
  skull_qc.run_qc()

  # After FAST
  from mri_preprocess.qc.anat.segmentation_qc import SegmentationQualityControl
  seg_qc = SegmentationQualityControl(subject, anat_dir, qc_dir)
  seg_qc.run_qc()

  # After FNIRT
  from mri_preprocess.qc.anat.registration_qc import RegistrationQualityControl
  reg_qc = RegistrationQualityControl(subject, anat_dir, qc_dir)
  reg_qc.run_qc()
  ```
- Add workflow parameter: `run_qc=True` (default)

#### Step 3: Test Complete Pipeline
- Run integrated workflow on IRC805 subject
- Validate all QC outputs are generated
- Review QC visualizations and metrics
- Ensure quality flags are appropriate

**Estimated Time:** 1-2 hours (including preprocessing runtime)

---

### PRIORITY 2: Complete DWI Pipeline with Advanced Models

**Goal:** Finish DWI processing with advanced diffusion models before moving to fMRI

#### Step 1: Advanced DWI Analyses (DKI & NODDI)
- **Files:**
  - `mri_preprocess/workflows/advanced_diffusion.py` (already implemented, needs testing)
  - `mri_preprocess/workflows/tractography.py` (already implemented, needs testing)

- **Tasks:**
  1. Test DKI (Diffusion Kurtosis Imaging) on real multi-shell data
     - Requires ≥2 non-zero b-values
     - Outputs: MK, AK, RK, KFA metrics

  2. Test NODDI (Neurite Orientation Dispersion and Density)
     - Requires multi-shell data
     - Outputs: ODI, FICVF, FISO

  3. Test GPU-accelerated probabilistic tractography
     - Requires BEDPOSTX outputs
     - Atlas-based ROI extraction (Harvard-Oxford, JHU)
     - Connectivity matrix generation

#### Step 2: Integrate Advanced Models into DWI Workflow
- **File to modify:** `mri_preprocess/workflows/dwi_preprocess.py`
- Add optional advanced model fitting:
  ```python
  # After eddy correction and DTI fitting
  if config.get('run_advanced_models', False):
      from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models
      advanced_results = run_advanced_diffusion_models(
          dwi_file=eddy_corrected,
          bval_file=bval_merged,
          bvec_file=rotated_bvecs,
          mask_file=dwi_mask,
          output_dir=output_dir / 'advanced_diffusion',
          fit_dki=True,
          fit_noddi=True
      )

  if config.get('run_tractography', False):
      from mri_preprocess.workflows.tractography import run_atlas_based_tractography
      tract_results = run_atlas_based_tractography(...)
  ```

#### Step 3: DWI QC Enhancement
- Review existing DWI QC modules:
  - `mri_preprocess/qc/dwi/topup_qc.py` ✅
  - `mri_preprocess/qc/dwi/motion_qc.py` ✅
  - `mri_preprocess/qc/dwi/dti_qc.py` ✅
- Add QC for advanced models (DKI, NODDI metrics)
- Add tractography QC (connectivity matrices, fiber counts)

**Estimated Time:** 2-3 hours

---

### PRIORITY 3: Resting-State fMRI Pipeline

**Goal:** Complete functional preprocessing with tissue segmentation integration

#### Step 1: Implement Tissue Segmentation for fMRI
- **New module needed:** `mri_preprocess/utils/tissue_segmentation.py`
- **Purpose:** Extract WM/CSF time series for nuisance regression
- **Implementation:**
  ```python
  def extract_tissue_timeseries(func_data, tissue_masks):
      """
      Extract mean time series from WM/CSF for nuisance regression.

      - Register anatomical tissue masks to functional space
      - Extract mean signals from WM and CSF
      - Return timeseries for ACompCor/nuisance regression
      """
  ```

#### Step 2: Integrate with Resting-State Workflow
- **File to modify:** `mri_preprocess/workflows/func_preprocess.py`
- Add tissue segmentation integration:
  1. Co-register anatomical to functional
  2. Warp tissue masks (WM/CSF) to functional space
  3. Extract tissue time series
  4. Use for ACompCor nuisance regression

#### Step 3: Functional QC Module
- **New module:** `mri_preprocess/qc/func/rest_qc.py`
- QC metrics:
  - Motion parameters (FD, DVARS)
  - TSNR (temporal signal-to-noise ratio)
  - Carpet plots (voxel intensity over time)
  - Tissue signal validation
  - Connectivity matrix QC

**Estimated Time:** 3-4 hours

---

## File Structure Overview

### Completed Modules
```
mri_preprocess/
├── qc/
│   ├── anat/
│   │   ├── skull_strip_qc.py       ✅ Complete
│   │   ├── segmentation_qc.py      ✅ Complete
│   │   └── registration_qc.py      ✅ Complete
│   └── dwi/
│       ├── topup_qc.py             ✅ Complete
│       ├── motion_qc.py            ✅ Complete
│       └── dti_qc.py               ✅ Complete
├── workflows/
│   ├── anat_preprocess.py          ✅ Complete (needs QC integration)
│   ├── dwi_preprocess.py           ✅ Complete (needs advanced model integration)
│   ├── advanced_diffusion.py       ✅ Complete (needs testing on real data)
│   ├── tractography.py             ✅ Complete (needs testing on real data)
│   └── func_preprocess.py          ✅ Complete (needs tissue segmentation)
└── utils/
    ├── topup_helper.py             ✅ Complete
    └── atlas_rois.py               ✅ Complete
```

### To Be Created
```
mri_preprocess/
├── qc/
│   ├── dwi/
│   │   ├── advanced_diffusion_qc.py  ⏳ TODO
│   │   └── tractography_qc.py        ⏳ TODO
│   └── func/
│       └── rest_qc.py                 ⏳ TODO
└── utils/
    └── tissue_segmentation.py         ⏳ TODO (PRIORITY for fMRI)
```

---

## Test Scripts Available

1. **Anatomical QC:**
   - `test_anat_qc.py` - Skull strip QC only
   - `test_segmentation_qc.py` - Segmentation QC only
   - `test_registration_qc.py` - Registration QC only
   - `test_anat_qc_complete.py` - All anatomical QC modules ✅
   - `find_anat_data.py` - Find available anatomical data

2. **DWI QC:**
   - `test_topup_qc.py` - TOPUP distortion correction QC
   - `test_motion_qc.py` - Eddy motion QC
   - `test_dti_qc.py` - DTI metrics QC

---

## Key Decisions & Notes

### Directory Structure Standard
All workflows now use consistent structure:
```
{study_root}/
├── derivatives/{workflow}/{subject}/  # Outputs
├── work/{subject}/{workflow}/          # Temporary Nipype files
└── qc/{modality}/{subject}/            # QC outputs
```

### Quality Thresholds
**Anatomical QC:**
- Correlation (registration): ≥0.85
- Dice coefficient: ≥0.85
- MAD (mean absolute difference): <0.15
- GM fraction: 30-60%
- WM fraction: 25-55%
- CSF fraction: 5-35%

**DWI QC:**
- Motion threshold: <2mm displacement
- SNR threshold: >10
- FA range: 0-1 (with physiological checks)

### GPU Acceleration
- Eddy: `eddy_cuda` (10x faster)
- BEDPOSTX: `bedpostx_gpu` (50x faster)
- Tractography: `probtrackx2_gpu` (10-50x faster)

---

## Commands for Next Session

### 1. Start with Anatomical Preprocessing
```bash
# Find a subject with anatomical data
find /mnt/bytopia -name "*IRC805*" -type d | grep nifti

# Run anatomical preprocessing (once workflow is updated with QC)
python -c "
from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing
from pathlib import Path

results = run_anat_preprocessing(
    subject='IRC805-XXXXX',
    t1w_file=Path('...'),
    output_dir=Path('/mnt/bytopia/development/IRC805'),
    run_qc=True  # New parameter
)
"
```

### 2. Test Advanced DWI Models
```bash
# Test DKI on existing DWI data
python -c "
from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models
from pathlib import Path

subject = 'IRC805-0580101'
dwi_dir = Path(f'/mnt/bytopia/development/IRC805/derivatives/dwi_topup/{subject}')

results = run_advanced_diffusion_models(
    dwi_file=dwi_dir / 'dwi_eddy_corrected.nii.gz',
    bval_file=dwi_dir / 'dwi_merged.bval',
    bvec_file=dwi_dir / 'dwi_rotated.bvec',
    mask_file=dwi_dir / 'dwi_mask.nii.gz',
    output_dir=dwi_dir / 'advanced_models',
    fit_dki=True,
    fit_noddi=True
)
"
```

### 3. Find and Test QC
```bash
# Find available data
python find_anat_data.py /mnt/bytopia/development/IRC805 --detail

# Run complete QC
python test_anat_qc_complete.py \
    --subject IRC805-XXXXX \
    --study-root /mnt/bytopia/development/IRC805
```

---

## Session Goals Summary

### Immediate Next Steps (Priority Order)
1. ✅ **DONE:** Complete anatomical QC implementation
2. ⏳ **NEXT:** Generate anatomical preprocessing data + integrate QC
3. ⏳ **NEXT:** Test and complete advanced DWI analyses (DKI/NODDI/tractography)
4. ⏳ **FUTURE:** Implement tissue segmentation for fMRI
5. ⏳ **FUTURE:** Complete resting-state fMRI pipeline

### End Goal
Complete, validated, QC-integrated preprocessing pipeline for:
- ✅ Anatomical (T1w/T2w) - QC complete
- ⏳ DWI/DTI/DKI/NODDI - Advanced models need testing
- ⏳ Resting-state fMRI - Needs tissue segmentation

---

## Questions to Address Next Session
1. Which IRC805 subject has complete raw anatomical data?
2. Should we process multiple subjects or validate pipeline with one first?
3. What are the actual b-values in IRC805 DWI data? (needed for DKI/NODDI validation)
4. Do we want HTML QC reports in addition to PNG/JSON outputs?

---

**Status:** Ready for next session - anatomical QC framework complete, ready to integrate and test on real data.
