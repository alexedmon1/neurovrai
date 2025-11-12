# Quick Start Guide - Next Session

## What We Accomplished Today ✅
- **Complete Anatomical QC Framework** - All 3 modules implemented and tested
  - Skull Strip QC
  - Segmentation QC
  - Registration QC

## What's Next - Priority Order

### 1. RUN ANATOMICAL PREPROCESSING (30-60 min)
Generate test data by running anatomical workflow on IRC805 subject:

```bash
# Activate environment
source .venv/bin/activate

# Find anatomical raw data
find /mnt/bytopia -name "*IRC805*" | grep -i "t1\|anat"

# Run preprocessing (example - adjust paths)
python -m mri_preprocess.workflows.anat_preprocess \
    --subject IRC805-XXXXX \
    --t1w /path/to/T1w.nii.gz \
    --output-dir /mnt/bytopia/development/IRC805
```

### 2. INTEGRATE QC INTO WORKFLOW (30 min)
Edit `mri_preprocess/workflows/anat_preprocess.py` to add automatic QC after each stage.

### 3. TEST COMPLETE QC SUITE (15 min)
```bash
# After preprocessing completes
python test_anat_qc_complete.py \
    --subject IRC805-XXXXX \
    --study-root /mnt/bytopia/development/IRC805
```

### 4. ADVANCED DWI ANALYSES (1-2 hours)
Test DKI, NODDI, and tractography on existing DWI data:

```bash
# Test advanced diffusion models
python -c "
from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models
from pathlib import Path

results = run_advanced_diffusion_models(
    dwi_file=Path('derivatives/dwi_topup/IRC805-0580101/dwi_eddy_corrected.nii.gz'),
    bval_file=Path('derivatives/dwi_topup/IRC805-0580101/dwi_merged.bval'),
    bvec_file=Path('derivatives/dwi_topup/IRC805-0580101/dwi_rotated.bvec'),
    mask_file=Path('derivatives/dwi_topup/IRC805-0580101/dwi_mask.nii.gz'),
    output_dir=Path('derivatives/dwi_topup/IRC805-0580101/advanced_models'),
    fit_dki=True,
    fit_noddi=True
)
print('DKI results:', results['dki'])
print('NODDI results:', results['noddi'])
"
```

## Files to Reference
- **Status Document:** `SESSION_STATUS.md` - Detailed plan and context
- **Test Scripts:** `test_anat_qc_complete.py`, `find_anat_data.py`
- **QC Modules:** `mri_preprocess/qc/anat/*.py`

## Key Paths
- Study root: `/mnt/bytopia/development/IRC805`
- Derivatives: `/mnt/bytopia/development/IRC805/derivatives/`
- QC outputs: `/mnt/bytopia/development/IRC805/qc/`

## Order of Operations
1. Anatomical preprocessing → QC integration → Testing
2. Advanced DWI (DKI/NODDI/tractography)
3. Tissue segmentation for fMRI
4. Complete resting-state pipeline
