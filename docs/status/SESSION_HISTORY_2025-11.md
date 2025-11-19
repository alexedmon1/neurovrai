# Development Session History - November 2025

This document archives detailed implementation history from November 2025 development sessions.

## Session: 2025-11-13 to 2025-11-14

### Bug Fixes (2025-11-14)
- Fixed DWI work directory hierarchy (now correctly uses `work/{subject}/dwi_preprocess/`)
- Fixed TEDANA NumPy 2.0 compatibility (upgraded 23.0.2 → 25.1.0)
- Fixed multi-echo DICOM conversion and workflow routing
- Relaxed dependency version constraints in pyproject.toml

### Spatial Normalization Implementation
#### DWI → FMRIB58_FA
- Complete normalization pipeline implemented and tested
- Created `neurovrai/preprocess/utils/dwi_normalization.py`
- Integrated into `dwi_preprocess.py` workflow
- Generates forward warp (group analyses) + inverse warp (tractography ROIs)
- Tested on IRC805-0580101: Successfully normalized 12 metrics (FA + DTI + DKI + NODDI)

#### Functional → MNI152
- Transform reuse strategy implemented
- Created `neurovrai/preprocess/utils/func_normalization.py`
- Integrated into `func_preprocess.py` workflow
- Reuses BBR transform from ACompCor (zero redundant computation)
- Reuses anatomical→MNI152 transforms from anatomical preprocessing
- Concatenates transforms via `convertwarp` for single-step normalization

#### Performance Improvements
- **BBR Performance Fix**: Switched from BBR to correlation ratio (20x speedup: 54s vs 10+ min)
- **Transform Management**: All transforms saved to standardized `transforms/` directories

### TEDANA Multi-Echo Denoising (IRC805-0580101)
- Motion correction: MCFLIRT + applyxfm4D (~40 minutes)
- TEDANA with 225 PCA components (~1h 18min)
- ICA converged after 10 attempts (seed 51)
- Component selection: 208 accepted, 17 rejected
- 82.55% variance explained
- All outputs generated: denoised BOLD, component maps, confounds, HTML report

## Session: 2025-11-12
- DWI preprocessing with TOPUP
- DKI/NODDI metrics validation
- Functional QC module enhancements

## Session: 2025-11-11
- ASL preprocessing implementation with M0 calibration and PVC
- Automated DICOM parameter extraction

## Session: 2025-11-10
- AMICO integration (NODDI/SANDI/ActiveAx models with 100x speedup)
- Performance benchmarking and validation

## Session: 2025-11-09
- Anatomical preprocessing validation
- Tissue segmentation QC implementation
