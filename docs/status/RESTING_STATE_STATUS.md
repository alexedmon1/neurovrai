# Resting State fMRI Preprocessing - Implementation Status

**Last Updated**: 2025-11-12
**Subject**: IRC805-0580101
**Status**: Ready for anatomical preprocessing (prerequisite for ACompCor)

---

## Current Implementation Status

### ✅ COMPLETED

#### 1. Core Functional Preprocessing Workflow
**File**: `mri_preprocess/workflows/func_preprocess.py`

**Implementation**:
- Multi-echo TEDANA denoising (lines 42-115)
- Motion correction BEFORE TEDANA (lines 331-450)
  - MCFLIRT on middle echo
  - Apply transforms to all echoes using applyxfm4D
  - Proper echo alignment before optimal combination
- Bandpass filtering (0.001-0.08 Hz)
- Spatial smoothing (6mm FWHM)
- Optional ICA-AROMA (disabled by default for multi-echo)

**Pipeline Order**:
```
Multi-echo: MCFLIRT (per-echo) → TEDANA → Bandpass → Smooth → QC
Single-echo: MCFLIRT → (optional AROMA) → Bandpass → Smooth → QC
```

#### 2. Quality Control (QC) Module
**File**: `mri_preprocess/qc/func_qc.py` (~558 lines)

**Features**:
- `compute_motion_qc()`: Framewise displacement, outlier detection, motion plots
- `compute_tsnr()`: Temporal SNR calculation with histograms
- `generate_func_qc_report()`: HTML report with color-coded quality indicators

**Thresholds**:
- Mean FD < 0.2 mm: GOOD
- Mean FD 0.2-0.5 mm: ACCEPTABLE
- Mean FD > 0.5 mm: POOR

#### 3. ACompCor Helper Functions
**File**: `mri_preprocess/utils/acompcor_helper.py` (~440 lines)

**Functions**:
- `run_fast_segmentation()`: FSL FAST tissue segmentation
- `register_masks_to_functional()`: BBR registration T1w ↔ functional
- `prepare_acompcor_masks()`: Threshold (p>0.9) and erode masks
- `extract_acompcor_components()`: PCA on CSF/WM signals
- `regress_out_components()`: Nuisance regression

**Note**: These are imported into func_preprocess.py but NOT YET integrated into the workflow.

#### 4. Test Scripts
- `test_rest_preprocessing.py`: Functional preprocessing test (has ACompCor disabled)
- `test_anat_preprocess_0580101.py`: Anatomical preprocessing test (NOT YET RUN)

#### 5. Documentation
- `RESTING_STATE_PLAN.md`: Complete implementation plan
- `TEDANA_VS_AROMA.md`: Design decision to disable AROMA for multi-echo
- `ACOMPCOR_IMPLEMENTATION.md`: Full ACompCor integration guide

---

## ⚠️ BLOCKING ISSUE - MUST RESOLVE FIRST

### Anatomical Preprocessing Required

**Problem**: ACompCor requires tissue segmentations (CSF, GM, WM) from anatomical preprocessing, but:
1. IRC805-0580101 has NO anatomical derivatives yet
2. The anatomical workflow function signature is unclear

**Last Error**:
```
TypeError: run_anat_preprocessing() got an unexpected keyword argument 't2w_file'
```

**Root Cause**: `test_anat_preprocess_0580101.py` calls:
```python
results = run_anat_preprocessing(
    config=config,
    subject=subject,
    t1w_file=t1w_file,
    t2w_file=t2w_file,  # ← This parameter doesn't exist
    output_dir=study_root
)
```

But the actual function signature in `mri_preprocess/workflows/anat_preprocess.py` needs to be checked (only read first 100 lines).

---

## 🎯 IMMEDIATE NEXT STEPS

### Step 1: Fix Anatomical Test Script

1. **Read** `mri_preprocess/workflows/anat_preprocess.py` to find the correct function signature for `run_anat_preprocessing()`
2. **Update** `test_anat_preprocess_0580101.py` with correct parameters
3. **Run** anatomical preprocessing for IRC805-0580101

**Expected outputs** (in `/mnt/bytopia/IRC805/derivatives/anat_preproc/IRC805-0580101/`):
```
├── t1w_brain.nii.gz           # Brain-extracted T1w
├── t1w_brain_restore.nii.gz   # Bias-corrected T1w
└── segmentation/
    ├── fast_seg_0.nii.gz      # CSF probability map (for ACompCor)
    ├── fast_seg_1.nii.gz      # Grey matter probability map
    └── fast_seg_2.nii.gz      # White matter probability map (for ACompCor)
```

**Command**:
```bash
source .venv/bin/activate
python test_anat_preprocess_0580101.py
```

---

### Step 2: Integrate ACompCor into Functional Workflow

After anatomical preprocessing completes, integrate ACompCor into `func_preprocess.py`:

**Location**: Between bandpass filtering and smoothing (lines ~550-600)

**Integration Pattern**:
```python
# After bandpass filtering
if config.get('acompcor', {}).get('enabled', False):
    logger.info("Running ACompCor nuisance regression...")

    # 1. Register anatomical masks to functional space
    csf_func, wm_func = register_masks_to_functional(
        t1w_brain=anat_derivatives / 't1w_brain.nii.gz',
        func_ref=mean_func,
        csf_mask=anat_derivatives / 'segmentation/fast_seg_0.nii.gz',
        wm_mask=anat_derivatives / 'segmentation/fast_seg_2.nii.gz',
        output_dir=work_dir / 'acompcor'
    )

    # 2. Prepare masks (threshold + erode)
    csf_eroded, wm_eroded = prepare_acompcor_masks(
        csf_mask=csf_func,
        wm_mask=wm_func,
        output_dir=work_dir / 'acompcor',
        csf_threshold=0.9,
        wm_threshold=0.9
    )

    # 3. Extract components
    acompcor_results = extract_acompcor_components(
        func_file=bandpass_output,
        csf_mask=csf_eroded,
        wm_mask=wm_eroded,
        output_dir=work_dir / 'acompcor',
        num_components=config.get('acompcor', {}).get('num_components', 5)
    )

    # 4. Regress out components
    cleaned_file = regress_out_components(
        func_file=bandpass_output,
        components_file=acompcor_results['components_file'],
        output_file=derivatives_dir / f'{subject}_desc-acompcor_bold.nii.gz'
    )

    # Use cleaned file as input to smoothing
    smooth_input = cleaned_file
else:
    smooth_input = bandpass_output
```

**Changes to `run_func_preprocessing()` function signature**:
```python
def run_func_preprocessing(
    config: Dict[str, Any],
    subject: str,
    func_file: Union[Path, List[Path]],
    output_dir: Path,
    work_dir: Optional[Path] = None,
    anat_derivatives: Optional[Path] = None  # ← ADD THIS
) -> Dict[str, Any]:
```

---

### Step 3: Update Test Script

**File**: `test_rest_preprocessing.py`

**Changes**:
```python
# After anatomical preprocessing completes, add:
from pathlib import Path

study_root = Path('/mnt/bytopia/IRC805')
anat_derivatives = study_root / 'derivatives/anat_preproc/IRC805-0580101'

# Enable ACompCor in config
config = {
    'tr': 1.029,
    'te': [10.0, 30.0, 50.0],
    'highpass': 0.001,
    'lowpass': 0.08,
    'fwhm': 6,
    'n_procs': 6,
    'tedana': {'enabled': True, 'tedpca': 'kundu', 'tree': 'kundu'},
    'aroma': {'enabled': False},
    'acompcor': {
        'enabled': True,        # ← ENABLE
        'num_components': 5,
        'variance_threshold': 0.5
    },
    'run_qc': True
}

# Pass anatomical derivatives to workflow
results = run_func_preprocessing(
    config=config,
    subject=subject,
    func_file=func_files,
    output_dir=study_root,
    anat_derivatives=anat_derivatives  # ← ADD THIS
)
```

---

### Step 4: Run Complete Pipeline

```bash
# 1. Run anatomical preprocessing (generates tissue segmentations)
source .venv/bin/activate
python test_anat_preprocess_0580101.py

# Expected runtime: ~15-20 minutes
# Expected output: /mnt/bytopia/IRC805/derivatives/anat_preproc/IRC805-0580101/

# 2. Verify segmentation outputs exist
ls -lh /mnt/bytopia/IRC805/derivatives/anat_preproc/IRC805-0580101/segmentation/

# 3. Run functional preprocessing with ACompCor
python test_rest_preprocessing.py

# Expected runtime: ~45-60 minutes
# Expected output: /mnt/bytopia/IRC805/derivatives/func_preproc/IRC805-0580101/
```

---

## Dataset Information

### Subject: IRC805-0580101

**Anatomical Data**:
- T1w: `/mnt/bytopia/IRC805/subjects/IRC805-0580101/nifti/anat/201_IRC805-0580101_WIP_3D_T1_TFE_SAG_CS3.nii.gz` (65.5 MB)
- T2w: `/mnt/bytopia/IRC805/subjects/IRC805-0580101/nifti/anat/1201_IRC805-0580101_WIP_T2W_CS5_OF1_TR2500.nii.gz` (6.3 MB)

**Functional Data** (Multi-echo):
- Echo 1: `/mnt/bytopia/IRC805/subjects/IRC805-0580101/nifti/rest/501_IRC805-0580101_WIP_RESTING_ME3_MB3_SENSE3_e1.nii.gz`
- Echo 2: `/mnt/bytopia/IRC805/subjects/IRC805-0580101/nifti/rest/501_IRC805-0580101_WIP_RESTING_ME3_MB3_SENSE3_e2.nii.gz`
- Echo 3: `/mnt/bytopia/IRC805/subjects/IRC805-0580101/nifti/rest/501_IRC805-0580101_WIP_RESTING_ME3_MB3_SENSE3_e3.nii.gz`

**Acquisition Parameters**:
- TR: 1.029 s
- TE: 10, 30, 50 ms (3 echoes)
- Volumes: 450
- Duration: ~7.7 minutes
- Multiband: 3
- SENSE: 3

---

## Configuration

### Recommended Functional Config

```python
config = {
    'tr': 1.029,
    'te': [10.0, 30.0, 50.0],
    'highpass': 0.001,  # Hz (1000s period)
    'lowpass': 0.08,    # Hz (12.5s period) - standard resting state
    'fwhm': 6,          # mm smoothing kernel
    'n_procs': 6,
    'tedana': {
        'enabled': True,
        'tedpca': 'kundu',
        'tree': 'kundu'
    },
    'aroma': {
        'enabled': False  # Redundant with TEDANA for multi-echo
    },
    'acompcor': {
        'enabled': True,
        'num_components': 5,
        'variance_threshold': 0.5,
        'erode_mm': 2.0,
        'csf_threshold': 0.9,
        'wm_threshold': 0.9
    },
    'run_qc': True
}
```

### Recommended Anatomical Config

```python
config = {
    'bet': {
        'frac': 0.5,
        'reduce_bias': True,
        'robust': True
    },
    'fast': {
        'bias_iters': 4,
        'bias_lowpass': 10
    },
    'n_procs': 6,
    'run_qc': True
}
```

---

## Expected Processing Pipeline

### Complete Workflow:
```
ANATOMICAL PREPROCESSING (IRC805-0580101)
  ↓
  1. Reorient to standard
  2. BET skull stripping
  3. FAST bias correction
  4. FAST tissue segmentation → CSF, GM, WM probability maps
  5. FLIRT/FNIRT to MNI152
  ↓
  Outputs: t1w_brain.nii.gz, segmentation/fast_seg_*.nii.gz
  Runtime: ~15-20 min

FUNCTIONAL PREPROCESSING (IRC805-0580101)
  ↓
  Multi-echo input (3 echoes: TE 10/30/50 ms)
  ↓
  1. MCFLIRT motion correction (middle echo)
  2. Apply transforms to all echoes (applyxfm4D)
  3. Brain extraction (BET on middle echo)
  4. TEDANA multi-echo denoising
     → Optimal combination (tedana_desc-optcom_bold.nii.gz)
  5. Bandpass filter (0.001-0.08 Hz)
  6. ACompCor nuisance regression
     a. Register anatomical masks to functional space (BBR)
     b. Threshold and erode CSF/WM masks (p>0.9, 2mm erosion)
     c. Extract 5 principal components from CSF/WM
     d. Regress components from bandpassed data
  7. Spatial smoothing (6mm FWHM)
  8. Quality control
     → Motion metrics (FD, outliers)
     → tSNR calculation
     → HTML QC report
  ↓
  Outputs: preprocessed_bold.nii.gz, qc_report.html
  Runtime: ~45-60 min
```

---

## Key Design Decisions

### 1. Motion Correction Timing
**Decision**: Apply motion correction BEFORE TEDANA (not after)
**Rationale**: Ensures proper echo alignment for optimal combination
**Reference**: fMRIPrep documentation, Mao et al. (2024)

### 2. ICA-AROMA for Multi-echo
**Decision**: Disabled by default for multi-echo data
**Rationale**: Redundant with TEDANA's T2* decay-based denoising
**Reference**: TEDANA documentation, fMRIPrep behavior

### 3. ACompCor Position
**Decision**: Between bandpass filtering and smoothing
**Rationale**:
- After bandpass: Remove non-physiological frequencies first
- Before smoothing: Need sharp tissue boundaries for accurate extraction
- After TEDANA: Complementary denoising approaches

### 4. Anatomical Workflow Reuse
**Decision**: Use existing anatomical workflow for tissue segmentation
**Rationale**:
- Avoid duplicate segmentation code
- Maintain consistency across modalities
- Enable reuse of anatomical derivatives

---

## Literature Support

### TEDANA
- Kundu et al. (2012): "Differentiating BOLD and non-BOLD signals"
- DuPre et al. (2021): "TE-dependent analysis of multi-echo fMRI"

### ACompCor
- Behzadi et al. (2007): Original CompCor paper - 50% reduction in physiological noise
- Muschelli et al. (2014): ACompCor superior to global signal regression
- Ciric et al. (2017): Benchmark study - TEDANA + ACompCor best combination

### Processing Order
- Mao et al. (2024): "Multi-echo fMRI denoising does not remove global motion"
- fMRIPrep documentation: Motion → TEDANA → ACompCor workflow

---

## TODO Summary

- [ ] **IMMEDIATE**: Check `run_anat_preprocessing()` function signature in anat_preprocess.py
- [ ] **IMMEDIATE**: Fix `test_anat_preprocess_0580101.py` to use correct parameters
- [ ] **IMMEDIATE**: Run anatomical preprocessing for IRC805-0580101
- [ ] **NEXT**: Integrate ACompCor into func_preprocess.py (after bandpass, before smooth)
- [ ] **NEXT**: Update `run_func_preprocessing()` to accept `anat_derivatives` parameter
- [ ] **NEXT**: Update `test_rest_preprocessing.py` to enable ACompCor and pass anatomical path
- [ ] **THEN**: Run complete functional preprocessing pipeline with ACompCor
- [ ] **VERIFY**: Check QC report for quality metrics (FD, tSNR)
- [ ] **VERIFY**: Confirm ACompCor variance removal (~10% expected improvement)

---

## Files Modified/Created This Session

### Created:
- `RESTING_STATE_PLAN.md` - Complete implementation plan
- `TEDANA_VS_AROMA.md` - Design decision documentation
- `ACOMPCOR_IMPLEMENTATION.md` - Integration guide
- `RESTING_STATE_STATUS.md` - This file
- `mri_preprocess/qc/func_qc.py` - QC module (~558 lines)
- `mri_preprocess/utils/acompcor_helper.py` - ACompCor functions (~440 lines)
- `test_rest_preprocessing.py` - Functional test script
- `test_anat_preprocess_0580101.py` - Anatomical test script (needs fix)

### Modified:
- `mri_preprocess/workflows/func_preprocess.py` - Complete rewrite with TEDANA, motion correction order, optional AROMA
  - Major refactoring: Motion correction before TEDANA (lines 331-450)
  - Imported ACompCor functions (not yet integrated into workflow)

### To Be Modified:
- `test_anat_preprocess_0580101.py` - Fix function call parameters
- `mri_preprocess/workflows/func_preprocess.py` - Add ACompCor integration (lines ~550-600)
- `test_rest_preprocessing.py` - Enable ACompCor, add anatomical derivatives path

---

**RESTART FROM HERE**: Fix anatomical test script and run anatomical preprocessing first.
