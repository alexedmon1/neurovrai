# Session Report: 2025-11-13

## Overview
Comprehensive fixes and enhancements to the MRI preprocessing pipeline, focusing on directory standardization, transform management, and advanced diffusion integration.

## Issues Identified and Resolved

### 1. QC Directory Structure Issues
**Problem**: QC outputs were going to non-standard locations
- ASL QC: `derivatives/{subject}/asl/qc/` → Should be `qc/{subject}/asl/`
- Anatomical QC: `qc/anat/{subject}/` → Should be `qc/{subject}/anat/`

**Fix**:
- `mri_preprocess/workflows/anat_preprocess.py:656` - Fixed QC directory hierarchy
- `mri_preprocess/workflows/asl_preprocess.py:617` - Fixed QC directory hierarchy

**Result**: All QC outputs now follow standardized `{study_root}/qc/{subject}/{modality}/` pattern

### 2. Transform Management Centralization
**Problem**: Transforms were being saved to modality-specific directories instead of centralized location

**Fix**: Updated workflows to use TransformRegistry for cross-modality transforms
- `mri_preprocess/workflows/asl_preprocess.py` (lines 48, 517-545, 569-578, 585-608)
  - Added TransformRegistry import
  - Save ASL→anat transform to TransformRegistry
  - Read anatomical transforms from TransformRegistry
  - Save concatenated ASL→MNI warp to registry

- `mri_preprocess/workflows/func_preprocess.py` (lines 53, 673-684, 848-862)
  - Added TransformRegistry import
  - Save BBR transform to TransformRegistry
  - Read anatomical transforms from TransformRegistry

**Result**: All cross-modality transforms now centralized in `{study_root}/transforms/{subject}/`

### 3. DWI Work Directory Structure
**Problem**: `dwi_preprocess/` folder not being created properly when DWI workflow begins

**Root Cause**: TOPUP runs before the Nipype workflow, and the workflow name didn't match expected directory structure

**Fix**: `mri_preprocess/workflows/dwi_preprocess.py:639-644`
- Changed workflow name from 'dwi_eddy_dtifit' to 'dwi_preprocess'
- Passed `work_dir.parent` instead of `work_dir` to ensure correct subdirectory creation

**Result**: Work directory now correctly structured as `{study_root}/work/{subject}/dwi_preprocess/`

### 4. Functional Workflow Parameter Error
**Problem**: `run_func_preprocessing()` called with incorrect parameter `t1w_brain`

**Error**: `TypeError: run_func_preprocessing() got an unexpected keyword argument 't1w_brain'`

**Fix**: `run_continuous_pipeline.py:474`
- Changed from `t1w_brain=t1w_brain` to `anat_derivatives=self.derivatives_dir / self.subject / 'anat'`

**Result**: Functional workflow now receives correct anatomical derivatives directory path

## New Features Implemented

### 5. Advanced Diffusion Models Integration
**Implementation**: Fully integrated DKI and NODDI models into DWI preprocessing pipeline

**Config Updates** (`config.yaml:49-58`):
```yaml
diffusion:
  advanced_models:
    enabled: auto  # Auto-enables for multi-shell data (≥2 b-values)
    fit_dki: true  # Diffusion Kurtosis Imaging (DIPY)
    fit_noddi: true  # NODDI via AMICO (100x faster) or DIPY
    fit_sandi: false  # SANDI via AMICO (requires ≥3 shells)
    fit_activeax: false  # ActiveAx via AMICO (requires specific protocol)
    use_amico: true  # Use AMICO for NODDI/SANDI/ActiveAx (recommended)
```

**Code Integration** (`mri_preprocess/workflows/dwi_preprocess.py`):
- Line 44: Added import for `run_advanced_diffusion_models`
- Lines 680-733: Integrated as Step 7.5 (after DTI fitting, before normalization)

**Features**:
- **Auto-detection**: Automatically detects multi-shell data (≥2 unique non-zero b-values)
- **Config-driven**: Respects `enabled: true/false/auto` setting
- **Proper hierarchy**: Creates `{derivatives}/{subject}/dwi/dki/` and `{derivatives}/{subject}/dwi/noddi/`
- **Graceful failure**: Falls back to standard DTI if advanced models fail
- **Automatic normalization**: DKI/NODDI metrics automatically included in spatial normalization (Step 8)

**Expected Output**:
```
/mnt/bytopia/IRC805/derivatives/IRC805-0580101/dwi/
├── dti/                    # Standard DTI metrics
│   ├── FA.nii.gz
│   ├── MD.nii.gz, AD.nii.gz, RD.nii.gz
├── dki/                    # DKI metrics (NEW)
│   ├── mean_kurtosis.nii.gz
│   ├── axial_kurtosis.nii.gz
│   ├── radial_kurtosis.nii.gz
│   └── kurtosis_fa.nii.gz
└── noddi/                  # NODDI metrics (NEW)
    ├── ficvf.nii.gz       # Neurite density
    ├── odi.nii.gz         # Orientation dispersion
    └── fiso.nii.gz        # Isotropic fraction
```

## Files Modified

### Configuration
1. `config.yaml` - Added advanced diffusion model configuration

### Core Workflows
2. `mri_preprocess/workflows/anat_preprocess.py` - Fixed QC directory
3. `mri_preprocess/workflows/asl_preprocess.py` - Fixed QC + TransformRegistry integration
4. `mri_preprocess/workflows/func_preprocess.py` - TransformRegistry integration
5. `mri_preprocess/workflows/dwi_preprocess.py` - Work directory fix + advanced diffusion integration

### Pipeline Runner
6. `run_continuous_pipeline.py` - Fixed functional workflow parameter

## Testing Status

### Completed
- Pipeline restarted with all fixes applied
- Error monitoring active (auto-stop on errors)
- Anatomical workflow started successfully
- DICOM conversion completed (anat: 5, dwi: 4, func: 3, asl: 3 files)

### In Progress
- Full pipeline execution for subject IRC805-0580101
- Pipeline set to run overnight for complete validation

## Directory Structure Validation

All workflows now follow standardized hierarchy:

```
{study_root}/                          # e.g., /mnt/bytopia/IRC805/
├── derivatives/                       # Preprocessed outputs
│   └── {subject}/                     # e.g., IRC805-0580101
│       ├── anat/                      # Anatomical preprocessing
│       ├── dwi/                       # DWI preprocessing
│       │   ├── dti/                   # DTI metrics
│       │   ├── dki/                   # DKI metrics (if multi-shell)
│       │   └── noddi/                 # NODDI metrics (if multi-shell)
│       ├── func/                      # Functional preprocessing
│       └── asl/                       # ASL preprocessing
├── work/{subject}/{workflow}/         # Temporary Nipype files
├── qc/{subject}/{modality}/           # Quality control reports
└── transforms/{subject}/              # Cross-modality spatial transforms
```

## Impact Summary

### Reliability
- ✅ Fixed 4 critical bugs preventing pipeline execution
- ✅ Centralized transform management for better reproducibility
- ✅ Improved error handling with graceful fallbacks

### Features
- ✅ Advanced diffusion models (DKI, NODDI) now integrated
- ✅ Auto-detection of multi-shell data
- ✅ Config-driven advanced model selection

### Maintainability
- ✅ Standardized directory hierarchy across all workflows
- ✅ Consistent use of TransformRegistry for transforms
- ✅ Clear documentation in config.yaml

## Next Steps

1. **Validation**: Complete overnight pipeline run for IRC805-0580101
2. **Testing**: Verify DKI/NODDI outputs for multi-shell data
3. **Monitoring**: Review error logs and QC reports
4. **Iteration**: Address any issues discovered during overnight run

## Commands for Next Session

To check pipeline status:
```bash
# Check if pipeline is still running
ps aux | grep run_continuous_pipeline

# View latest log output
tail -50 logs/continuous_pipeline_all_fixes_0580101.log

# Check for errors
grep -E "ERROR|Failed" logs/continuous_pipeline_all_fixes_0580101.log
```

To restart if needed:
```bash
# Clean restart
rm -rf /mnt/bytopia/IRC805/work/IRC805-0580101
source .venv/bin/activate
python run_continuous_pipeline.py \
  --subject IRC805-0580101 \
  --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-0580101 \
  --study-root /mnt/bytopia/IRC805 \
  --config config.yaml \
  > logs/continuous_pipeline_all_fixes_0580101.log 2>&1 &
```

## Session Metrics

- **Duration**: Full session
- **Files Modified**: 6
- **Bugs Fixed**: 4
- **Features Added**: 1 (Advanced Diffusion Integration)
- **Lines Changed**: ~150+
- **Pipeline Status**: Running (overnight validation)
