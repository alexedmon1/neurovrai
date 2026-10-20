# MRI Preprocessing Pipeline - Project Status
**Last Updated**: 2026-10-20

## 🎯 Project Overview

Production-ready MRI preprocessing pipeline for anatomical, diffusion, functional, and ASL data with comprehensive QC and standardized outputs.

**Project Architecture** - **neurovrai** with three-part architecture:
- `neurovrai.preprocess` ✅ (complete - preprocessing for all modalities)
- `neurovrai.analysis` ✅ (complete - VBM, TBSS, ReHo/fALFF, MELODIC)
- `neurovrai.connectome` ✅ (complete - functional & structural connectivity, graph metrics, NBS)


---

## 📝 Latest Updates (2026-10-20)

### Minimum Detectable Effect on Tabular Tests ✅
**Known issues found doing this, not yet fixed**: see
`docs/issues/KNOWN_ISSUES_2026-10-20.md` (TBSS GLM path calls undefined functions,
undeclared `statsmodels`, wrong connectome README examples, a colliding SE filename,
the WMH script's group coding).

**Goal**: Every tabular inferential result says what difference its design could
have detected, so a null reads against a number rather than only a p-value.

**Completed**:
1. **`neurovrai/analysis/stats/power.py`**: MDE at 80% power, two-sided alpha 0.05
   (nominal, no multiple-comparison correction): `(t.ppf(1-alpha/2, df) + t.ppf(power, df)) * se`,
   normal quantiles when df is missing or infinite. SE/df helpers for Student, Welch
   and paired t-tests; `mde_fields()` returns the shared columns `se`, `df`, `mde`,
   `mde_percent`, `mde_power`, `mde_alpha`, `mde_note` (same names as neurofaune and
   source-analytics).
2. **`scripts/wmh_group_comparison.py`**: `wmh_group_comparison.csv` gains
   `mean_difference`, `t_stat` and the MDE columns for the t-test it reports;
   `mde_percent` is relative to the good-control mean.
3. **`connectome/group_analysis.py`**: `compute_group_difference` writes
   `{prefix}_se_matrix.npy` and `{prefix}_mde_matrix.npy` and records the MDE settings in
   `{prefix}_summary.json`; the NBS records `mde_note: "not computed: network-based
   statistic component"`.
4. **Voxelwise maps** (randomise/TFCE, nilearn GLM): no MDE, documented as
   "not computed: voxelwise map" in `docs/GLM_USAGE.md`.

**Tests**: `tests/unit/test_power.py` (reference values shared across the three tools).

---

## 📝 Updates (2026-01-20)

### Study Initialization System ✅
**Goal**: Implement comprehensive study setup workflow similar to neurofaune, adapted for human MRI data

**Completed**:
1. **Study Initialization Module** (`neurovrai/study_initialization.py`):
   - ✅ `setup_study()` - Main entry point for complete study initialization
   - ✅ `discover_bids_data()` - BIDS directory scanning with subject/session/modality inventory
   - ✅ `discover_dicom_data()` - DICOM directory scanning with automatic modality classification
   - ✅ `create_study_directories()` - Standardized directory structure creation
   - ✅ `generate_config()` - Config.yaml generation with study-specific settings
   - ✅ `get_study_subjects()` - List subjects ready for processing with filtering
   - ✅ `print_study_summary()` - Study status overview

2. **CLI Script** (`scripts/init_study.py`):
   - ✅ Full command-line interface with all options
   - ✅ `--discover-only` mode for data inventory without directory creation
   - ✅ `--summary` mode for viewing study status
   - ✅ `--list-subjects` mode for listing available subjects
   - ✅ FreeSurfer integration support (`--freesurfer-dir`)
   - ✅ JSON export (`--output-json`)

3. **Data Discovery Features**:
   - ✅ BIDS: T1w, T2w, FLAIR, DWI, BOLD, ASL, fieldmap detection
   - ✅ DICOM: Automatic modality classification from series descriptions
   - ✅ Multi-session support with subject/session hierarchy
   - ✅ Issue tracking (missing data, incomplete scans)

4. **Documentation Updates**:
   - ✅ README.md - Study initialization as first step in Quick Start
   - ✅ CLAUDE.md - Added study initialization section
   - ✅ Output structure updated to reflect study init hierarchy

**Files Created**:
- `neurovrai/study_initialization.py` - Core module
- `scripts/init_study.py` - CLI script

**Usage**:
```bash
# Initialize new study
uv run python scripts/init_study.py /path/to/study --name "My Study" --code STUDY01

# With existing data
uv run python scripts/init_study.py /path/to/study --name "My Study" --code STUDY01 \
    --dicom-root /path/to/dicom --bids-root /path/to/bids

# Just discover data
uv run python scripts/init_study.py --discover-only /path/to/data
```

**Impact**: New studies now have a standardized setup workflow that creates directory structure, discovers data, generates configuration, and provides clear next steps for preprocessing.

---

## 📝 Updates (2025-12-16 PM)

### Structural Connectivity Pipeline Optimization ✅
**Goal**: Optimize FreeSurfer atlas transformation and resolve probtrackx2_gpu compatibility issues

**Completed**:
1. **FreeSurfer Atlas Transformation Optimization**:
   - ✅ Replaced slow FLIRT (FS→T1w) with FreeSurfer's `mri_vol2vol --regheader`
   - ✅ Performance improvement: ~5 minutes → ~5 seconds for atlas transformation
   - ✅ Better accuracy using FreeSurfer's built-in registration

2. **DWI↔T1w Registration Strategy Improvement**:
   - ✅ Implemented DWI→T1w registration with mutual information (better for cross-modality)
   - ✅ Compute inverse for T1w→DWI transform (more stable than registering high-res to low-res)
   - ✅ Reduced search ranges (±10° vs ±30°) for faster, more reliable registration

3. **probtrackx2_gpu Compatibility Fixes**:
   - ✅ Fixed command syntax: GPU version requires `=` format (`--option=value` not `--option value`)
   - ✅ Fixed option names: `--cthr` (not `--curvthresh`) for GPU version
   - ✅ Added required `--targetmasks` parameter for network mode
   - ✅ Verified ventricle avoidance and white matter waypoints working correctly

4. **Successful Test Run**:
   - ✅ STUDY-0580101 structural connectivity running with probtrackx2_gpu
   - ✅ Desikan-Killiany atlas (107 ROIs) successfully transformed to DWI space
   - ✅ Ventricle avoidance mask active
   - ✅ White matter waypoints enabled
   - ✅ 5,000 samples/voxel, estimated 2-4 hours for full 107×107 connectivity matrix

**Files Modified**:
- `neurovrai/preprocess/utils/freesurfer_transforms.py`:
  - Updated `compute_fs_to_t1w_transform()` to use `mri_vol2vol`
  - Updated `compute_t1w_to_dwi_transform()` to register DWI→T1w + inverse
- `neurovrai/connectome/atlas_dwi_transform.py`:
  - Updated `transform_fs_atlas_to_dwi()` to use `mri_vol2vol` for FS→T1w step
- `neurovrai/connectome/structural_connectivity.py`:
  - Fixed probtrackx2_gpu command syntax (use `=` format)
  - Fixed GPU-specific option names (`--cthr` vs `--curvthresh`)
  - Added `--targetmasks` parameter for network mode

**Known Issues**:
- ⚠️ `mri_surf2vol` failing for GM/WM interface extraction (falling back to volume method)
  - Impact: Using volume-based GMWMI instead of surface-based (less anatomically precise)
  - Follow-up: Investigate FreeSurfer surface extraction for better seeding

**Impact**: Structural connectivity pipeline now fully functional with optimized FreeSurfer integration and GPU acceleration. Atlas transformation time reduced by >95%, and probtrackx2_gpu running successfully with anatomical constraints.

---

## 📝 Updates (2025-12-16 AM)

### Structural Connectivity Pipeline ✅
**Goal**: Implement tractography-based structural connectivity with advanced anatomical constraints

**Completed**:
1. **Core Structural Connectivity Module**:
   - ✅ FSL probtrackx2 integration with network mode
   - ✅ GPU support (probtrackx2_gpu) for 5-10x speedup
   - ✅ Connectivity matrix construction with waytotal normalization
   - ✅ Graph metrics computation (degree, clustering, efficiency, betweenness)
   - ✅ Multiple atlas support (Schaefer 100/200/400, Desikan-Killiany)

2. **Anatomical Constraints (FreeSurfer Integration)**:
   - ✅ Ventricle avoidance mask (CSF exclusion using FreeSurfer labels)
   - ✅ White matter waypoint mask (ACT-style constraints)
   - ✅ Gray matter termination mask (optional)
   - ✅ GMWMI seeding (Gray-White Matter Interface) from FreeSurfer surfaces or volume
   - ✅ Subcortical waypoints (thalamus, basal ganglia)

3. **Config-Driven Architecture**:
   - ✅ All tractography parameters configurable via config.yaml
   - ✅ Added `structural_connectivity` section to configs
   - ✅ FreeSurfer options configurable per study

4. **Batch Processing**:
   - ✅ `batch_structural_connectivity.py` for multi-subject processing
   - ✅ `run_structural_connectivity.py` CLI with full config support
   - ✅ Automatic BEDPOSTX detection and validation

**Configuration Options Added**:
```yaml
structural_connectivity:
  tractography:
    use_gpu: true
    n_samples: 5000
    step_length: 0.5
    curvature_threshold: 0.2
    loop_check: true
  anatomical_constraints:
    avoid_ventricles: true
    use_wm_mask: true
    terminate_at_gm: false
    wm_source: auto
  freesurfer_options:
    use_gmwmi_seeding: true
    gmwmi_method: surface
    use_subcortical_waypoints: false
```

**Files Created/Modified**:
- `neurovrai/connectome/structural_connectivity.py` - Core functions with anatomical constraints
- `neurovrai/connectome/batch_structural_connectivity.py` - Batch processing
- `neurovrai/connectome/run_structural_connectivity.py` - CLI runner
- `neurovrai/connectome/graph_metrics.py` - Added `compute_graph_metrics()` wrapper
- `configs/config.yaml` - Added structural_connectivity section
- `/path/to/study/config.yaml` - Study-specific config with FreeSurfer enabled

**Documentation Updated**:
- `README.md` - Added Structural Connectivity section
- `neurovrai/connectome/README.md` - Added complete structural connectivity documentation

**Impact**: Structural connectivity now production-ready with configurable FreeSurfer integration for anatomically constrained probabilistic tractography.

---

## 📝 Updates (2025-11-24)

### Resting-State fMRI Analysis & TBSS Investigation ✅
**Goal**: Implement ReHo/fALFF analysis and investigate missing DTI subjects in TBSS

**Completed**:
1. **Resting-State fMRI Analysis Module**:
   - ✅ Implemented ReHo (Regional Homogeneity) with Kendall's coefficient of concordance
   - ✅ Implemented ALFF/fALFF (Amplitude of Low-Frequency Fluctuations)
   - ✅ Z-score normalization for group-level comparison
   - ✅ Integrated workflow with error handling and JSON summary output
   - ✅ Performance: 7 min ReHo + 22 sec fALFF for 136k voxels
   - ✅ Renamed connectivity_workflow → resting_workflow (reserved connectivity for neurovrai.connectome)
   - ⏳ MELODIC (group ICA) planned for future enhancement

2. **TBSS Missing Subjects Investigation** (6 subjects missing FA data):
   - ✅ **Root cause identified**: Scanner-processed derivative maps blocking preprocessing
     - BIDS directories contained both raw DWI + scanner derivatives (ADC, dWIP, facWIP, isoWIP)
     - File discovery pattern matched ALL files, tried to process maps without bval/bvec → failures
   - ✅ **Fix implemented**: Added scanner-processed map filtering in `run_simple_pipeline.py`
   - ✅ **Subject analysis completed**:
     - STUDY-2990202: No DTI data exists (exclude)
     - STUDY-4960101: 3 incompatible acquisitions, size mismatch (exclude)
     - STUDY-2350101, 3280201, 3580101, 3840101: Have recoverable raw DWI data (4 subjects)
   - ✅ **Config issues fixed**: Added required `paths.logs` to config.yaml

3. **Preprocessing Workflow Issues Documented**:
   - 📋 Created comprehensive issue tracking: `docs/issues/PREPROCESSING_ISSUES_2025-11-24.md`
   - **High Priority Issues Identified**:
     - Eddy-without-TOPUP: Missing acqparams.txt generation for single-direction acquisitions
     - Error handling: Silent failures (return code 0 even when preprocessing fails)
   - **Medium Priority Enhancements**:
     - Orientation validation for multi-acquisition merging
     - Dimension validation to detect incompatible matrix sizes
   - **Testing plan** created for systematic resolution

**Analysis Modules Implemented**:
- `neurovrai/analysis/func/reho.py` - Regional Homogeneity analysis
- `neurovrai/analysis/func/falff.py` - ALFF/fALFF analysis
- `neurovrai/analysis/func/resting_workflow.py` - Integrated workflow
- Command-line interface with configurable parameters

**Files Modified**:
- `run_simple_pipeline.py` - Scanner-processed map filtering (lines 118-137)
- `config.yaml` - Added `paths.logs` requirement
- `neurovrai/analysis/func/__init__.py` - Updated exports
- `README.md`, `CLAUDE.md` - Updated documentation

**Impact**:
- Resting-state fMRI analysis now production-ready
- 4 of 6 missing TBSS subjects identified as recoverable (pending workflow fixes)
- TBSS currently: 17/23 subjects → potential 21/23 after fixes
- Clear roadmap for making preprocessing fully flexible (single/multi-shell, with/without TOPUP)

**Next Steps**:
1. Implement eddy-without-TOPUP functionality (`create_eddy_files_single_direction()`)
2. Fix error handling to propagate failures properly
3. Add orientation and dimension validation
4. Batch process recoverable subjects

---

## 📝 Updates (2025-11-16 PM)

### Bug Fixes & BEDPOSTX Integration ✅
**Goal**: Fix file-finding issues and make BEDPOSTX standard preprocessing

**Completed**:
1. **Critical Bug Fixes**:
   - ✅ Fixed anatomical QC path bug (removed double `/anat/anat/` hierarchy)
   - ✅ Fixed DWI QC parameter mismatch (`metrics_dir` → `dti_dir`)
   - ✅ Fixed simple pipeline file finding (now uses `rglob` for subdirectories)

2. **BEDPOSTX Now Enabled by Default**:
   - ✅ BEDPOSTX is now standard DWI preprocessing (fiber orientation estimation)
   - ✅ Config setting: `diffusion.bedpostx.enabled: true` (default)
   - ✅ GPU-accelerated: 20-60 min vs 4-8 hours CPU
   - ✅ Required for future connectomics module

3. **Tractography Migration Plan**:
   - Current `tractography.py` marked for migration to `neurovrai.connectome`
   - BEDPOSTX stays in preprocessing (orientation estimation)
   - Tractography moves to connectome (streamline generation with anatomical constraints)

**Files Modified**:
- `neurovrai/preprocess/workflows/t1w_preprocess.py` - Fixed QC path (line 672)
- `neurovrai/preprocess/workflows/dwi_preprocess.py` - Fixed QC param, added BEDPOSTX config reading
- `run_simple_pipeline.py` - Fixed file finding with rglob
- `create_config.py` - Added BEDPOSTX config section
- `configs/config.yaml` - Added BEDPOSTX defaults

**Impact**: All modalities (anat, DWI, functional, ASL) can now find files correctly and BEDPOSTX runs by default.

---

## 📝 Updates (2025-11-16 AM)

### Configuration System Enhancement ✅
**Goal**: Make all critical processing parameters configurable via `config.yaml`

**Completed**:
1. **Config File Location**: Config now lives in study root (`{study_root}/config.yaml`)
   - Auto-created by `create_config.py --study-root /path/to/study`
   - Each study has its own config co-located with data
   - No more confusion about which config goes with which study

2. **Configurable Parameters** (Option 1 Implementation):
   - ✅ **BET fractional intensity** - Now configurable per modality:
     - `anatomical.bet.frac` (default: 0.5)
     - `diffusion.bet.frac` (default: 0.3)
     - `functional.bet.frac` (default: 0.3)
     - `asl.bet.frac` (default: 0.3)
   - ✅ **N4 bias correction** - 4 parameters now configurable:
     - `anatomical.bias_correction.n_iterations` (default: [50, 50, 30, 20])
     - `anatomical.bias_correction.shrink_factor` (default: 3)
     - `anatomical.bias_correction.convergence_threshold` (default: 0.001)
     - `anatomical.bias_correction.bspline_fitting_distance` (default: 300)
   - ✅ **Atropos segmentation** - 6 parameters now configurable:
     - `anatomical.atropos.number_of_tissue_classes` (default: 3)
     - `anatomical.atropos.initialization` (default: 'KMeans')
     - `anatomical.atropos.n_iterations` (default: 5)
     - `anatomical.atropos.convergence_threshold` (default: 0.001)
     - `anatomical.atropos.mrf_smoothing_factor` (default: 0.1)
     - `anatomical.atropos.mrf_radius` (default: [1, 1, 1])
   - ✅ **TEDANA PCA components** - Already configurable:
     - `functional.tedana.tedpca` (updated default: 225 for 450 volumes)

3. **Future Enhancements Documented**: `docs/FUTURE_ENHANCEMENTS.md`
   - Tractography parameters (n_samples, n_steps, step_length, curvature_threshold)
   - AMICO model parameters (NODDI, SANDI, ActiveAx diffusivities)

4. **Project Cleanup**:
   - ✅ Old runner scripts and documentation removed (2026-09-22; see git history)
   - ✅ Clean root directory with only current production files
   - ✅ Updated README with current Quick Start and project structure

**Files Modified**:
- `config.yaml` - Added all new config sections
- `create_config.py` - Auto-populates all defaults
- `neurovrai/preprocess/workflows/dwi_preprocess.py` - Reads BET frac from config
- `neurovrai/preprocess/workflows/func_preprocess.py` - Reads BET frac from config (2 locations)
- `neurovrai/preprocess/workflows/asl_preprocess.py` - Reads BET frac from config
- `neurovrai/preprocess/workflows/t1w_preprocess.py` - Reads N4 and Atropos params from config
- `README.md` - Updated Quick Start, added Project Structure
- `QUICKSTART.md` - Updated with config location
- `docs/FUTURE_ENHANCEMENTS.md` - Created for Option 2 parameters

**Total Hardcoded Values Fixed**: 13 → Now configurable
**Legacy Scripts Archived**: 5 (run_preprocessing.py, run_full_pipeline.py, etc.)
**Old Docs Archived**: 3 (CONFIG_SETUP.md, CONFIG_SUMMARY.md, SIMPLE_PIPELINE_GUIDE.md)

---

## ✅ Completed & Production-Ready

### 1. Core Infrastructure (100%)
- ✅ **Study Initialization** - Automated setup with DICOM/BIDS discovery
- ✅ Config-driven architecture (YAML with validation)
- ✅ DICOM converter with automatic parameter extraction
- ✅ Transform Registry for spatial transformations
- ✅ Standardized directory hierarchy
- ✅ Continuous streaming pipeline (start preprocessing as data converts)
- ✅ BIDS compatibility

### 2. Anatomical Preprocessing (100%)
**Status**: Fully functional, validated on real-world multi-modal datasets

**Features**:
- N4 bias correction (ANTs)
- Brain extraction (FSL BET)
- Tissue segmentation (ANTs Atropos - faster than FAST)
- Registration to MNI152 (FLIRT/FNIRT)
- Tissue masks for ACompCor (CSF, GM, WM)

**QC**:
- ✅ Skull stripping QC
- ✅ Segmentation QC
- ✅ Registration QC

**Outputs**: `/derivatives/{subject}/anat/`

### 3. DWI Preprocessing (100%)
**Status**: Fully functional with advanced models, validated on multi-shell datasets

**Features**:
- TOPUP susceptibility distortion correction
- GPU-accelerated eddy correction
- DTI fitting (FA, MD, AD, RD)
- **Advanced Models**:
  - DKI (Diffusion Kurtosis Imaging): MK, AK, RK, KFA
  - NODDI (Neurite Orientation): FICVF, ODI, FISO
- Spatial normalization to FMRIB58_FA template
- GPU-accelerated probabilistic tractography
- Atlas-based ROI extraction (Harvard-Oxford, JHU)

**Shell Detection**:
- Auto-detects single-shell vs multi-shell
- Clear labeling: "Multi-shell data: True/False"
- Auto-skips advanced models for single-shell

**QC** (fully integrated as of 2025-11-15):
- ✅ TOPUP QC (field maps, convergence metrics)
- ✅ Motion QC (eddy parameters, FD, outliers)
- ✅ DTI QC (FA/MD/AD/RD distributions, histograms)
- ✅ Automated QC reports in standardized location

**Outputs**: `/derivatives/{subject}/dwi/`, QC: `/qc/{subject}/dwi/`

### 4. ASL Preprocessing (100%)
**Status**: Fully functional, validated on pCASL datasets with M0 calibration

**Features**:
- Automated DICOM parameter extraction (τ, PLD)
- Motion correction (MCFLIRT)
- Label-control separation
- CBF quantification (Alsop et al., 2015)
- M0 calibration with white matter reference
- Partial volume correction
- Registration to anatomical space
- Tissue-specific CBF statistics

**QC**:
- ✅ Motion metrics
- ✅ CBF distributions
- ✅ tSNR analysis
- ✅ HTML QC report

**Outputs**: `/derivatives/{subject}/asl/`

### 5. Functional Preprocessing (100%)
**Status**: Fully functional, validated on multi-echo and single-echo datasets

**Features**:
- Multi-echo TEDANA denoising (automatic for ≥2 echoes)
- Single-echo support with optional ICA-AROMA
- Motion correction (MCFLIRT)
- ACompCor nuisance regression using anatomical tissue masks
- Bandpass temporal filtering
- Spatial smoothing
- Registration to anatomical space (BBR or correlation ratio)
- Optional spatial normalization to MNI152

**Echo Detection**:
- Auto-detects single vs multi-echo data
- Clear labeling: "Input data: 3 echoes" or "Single-echo data"
- Conditional workflow routing (TEDANA vs ICA-AROMA)

**Recent Fixes (2025-11-15)**:
- ✅ Fixed variable naming bugs in ACompCor section
- ✅ Fixed QC output directory (now uses study-level qc/)
- ✅ Completed full multi-echo pipeline test successfully

**QC**:
- ✅ Motion metrics (FD, outliers)
- ✅ DVARS analysis
- ✅ Carpet plots
- ✅ tSNR analysis and histograms
- ✅ HTML QC report

**Outputs**: `/derivatives/{subject}/func/`, QC: `/qc/{subject}/func/`

### 6. Advanced Diffusion Models - AMICO (100%)
**Status**: Fully functional, production-ready alternative to DIPY-based fitting

**Features**:
- **AMICO-accelerated NODDI**: 100x faster than DIPY (30 sec vs 20-25 min)
- **SANDI**: Soma And Neurite Density Imaging
- **ActiveAx**: Axon diameter distribution modeling
- Convex optimization for 100-1000x speedup
- Same accuracy as traditional fitting methods
- Automatic gradient timing extraction/estimation

**Performance**:
- NODDI: ~30 seconds (vs 20-25 min DIPY)
- SANDI: 3-6 minutes
- ActiveAx: 3-6 minutes

**Outputs**: `/derivatives/{subject}/dwi/amico_{model}/`

**Documentation**: `docs/amico/AMICO_FINDINGS.md`

---

## 📋 Planned Features

### 1. FreeSurfer Integration (✅ PRODUCTION READY)

**Current Status**: FreeSurfer integration is **production ready** for structural and functional connectivity.

**What's Implemented** (verified 2026-01-08):
- ✅ Detection of existing FreeSurfer outputs (`freesurfer_utils.py`)
- ✅ ROI extraction from aparc+aseg (Desikan-Killiany 85 ROIs, Destrieux 165 ROIs)
- ✅ Config integration (`freesurfer.enabled`, `freesurfer.subjects_dir`)
- ✅ Optimized FS→T1w transform using `mri_vol2vol` (~5 sec vs 5 min with FLIRT)
- ✅ T1w→DWI transform with smart registration (DWI→T1w + inverse)
- ✅ Transform chain composition (FS→T1w→DWI)
- ✅ QC validation with correlation/NMI metrics (`freesurfer_qc.py`)
- ✅ Visual alignment overlays (tri-planar PNG generation)
- ✅ Ventricle avoidance mask from FreeSurfer labels (4, 5, 14, 15, 43, 44, 72)
- ✅ White matter mask from FreeSurfer (labels 2, 41, 77, 251-255)
- ✅ GMWMI seeding for anatomically precise tractography
- ✅ GM termination mask for ACT-style constraints
- ✅ Subcortical waypoint masks (thalamus, basal ganglia)
- ✅ Structural connectivity with `probtrackx2_gpu`
- ✅ Functional connectivity with FreeSurfer atlases (`atlas_func_transform.py`)

**Not Implemented** (lower priority):
- ⏳ Cortical thickness analysis
- ⏳ Surface-based rendering

**Key Files**:
- `neurovrai/preprocess/utils/freesurfer_utils.py` - Detection, ROI extraction
- `neurovrai/preprocess/utils/freesurfer_transforms.py` - Transform pipeline
- `neurovrai/preprocess/qc/freesurfer_qc.py` - QC validation
- `neurovrai/connectome/structural_connectivity.py` - SC with anatomical constraints

### 2. Enhanced QC (Priority: Medium)
**Scope**:
- ✅ Complete functional QC integration (motion, DVARS, tSNR, carpet plots)
- ⏳ Group-level QC reports across subjects
- ⏳ Interactive HTML dashboards with plotly/bokeh
- ⏳ Automated outlier detection with statistical thresholds

### 3. Additional Analysis Features (Priority: Low-Medium)

**Preprocessing:**
- ⏳ **T1w/T2w Myelin Mapping**: Modernize existing myelin/myelin_workflow.py to current architecture
  - Review legacy implementation
  - Update to functional workflow pattern
  - Integrate with config system
  - Add to production workflows

**DWI Analysis:**
- ⏳ **MNI-Space Tractography**: Group-level probabilistic tractography for cross-subject comparisons
  - Warp subject ROIs to MNI space
  - Run probtrackx2 in standard space
  - Generate group connectivity matrices
- ⏳ **TBSS Pipeline**: Tract-Based Spatial Statistics for FA group analysis
  - Modernize legacy TBSS utilities
  - Implement skeleton projection
  - Voxelwise statistical testing
  - Multi-subject group comparisons

**Functional Analysis:**
- ⏳ **MELODIC (Group ICA)**: Identify consistent brain networks across subjects
  - Temporal concatenation of preprocessed BOLD data
  - Group-level independent component analysis
  - Component spatial maps and time courses
- ⏳ **ReHo (Regional Homogeneity)**: Local connectivity analysis
  - Kendall's coefficient of concordance within clusters
  - Voxelwise or ROI-based ReHo maps
  - Group-level statistical comparisons
- ⏳ **fALFF (Fractional ALFF)**: Frequency-domain analysis
  - Ratio of low-frequency power to total power
  - ALFF and fALFF maps
  - Group-level comparisons

**Anatomical Analysis:**
- ⏳ **VBM (Voxel-Based Morphometry)**: Structural group analysis
  - Choose between FSL (fslvbm) or ANTs (more accurate)
  - Tissue segmentation and normalization
  - Statistical comparison of GM/WM concentration
  - Multi-subject group studies

**ASL Analysis:**
- ⏳ **Group-Level ASL Analysis**: To be determined based on research needs
  - Common approaches: CBF group comparisons, test-retest reliability, arterial transit time analysis
  - CBF as connectivity regressor
  - Perfusion territory mapping

**Note**: These features are not currently prioritized as the core preprocessing workflows (anatomical, DWI, functional, ASL) are production-ready and meet current research needs.

---

## 🐛 Known Issues & Limitations

### Resolved (2025-11-14)
- ✅ DICOM converter only copied first multi-echo file → **FIXED**
- ✅ Functional workflow input node selection bug → **FIXED**

### Active
- None currently identified

### Limitations
- **Functional TOPUP**: Not applicable - no reversed phase-encoding field maps in current protocol
  - All scans (fMRI_CORRECTION + RESTING) have same phase encoding (COL)
  - Would require protocol update to add AP/PA field map pair

---

## 📊 Validation Status

### Datasets Tested
- **Multi-modal validation dataset**:
  - ✅ Anatomical (T1w)
  - ✅ DWI multi-shell (3 shells: b=1000, 2000, 3000 s/mm²)
  - ✅ ASL (pCASL with M0 calibration scan)
  - ✅ Functional (multi-echo resting-state, 3 echoes)

### Metrics Generated

**Anatomical**:
- Brain masks, tissue segmentations
- MNI-registered T1w

**DWI** (22+ total metrics):
- DTI: FA, MD, AD, RD (4)
- DKI: MK, AK, RK, KFA (4)
- NODDI (DIPY): FICVF, ODI, FISO (3)
- NODDI (AMICO): FICVF, ODI, FISO, DIR (4)
- Optional AMICO models: SANDI (5 metrics), ActiveAx (4 metrics)
- Normalized versions: 12 DTI/DKI/NODDI metrics in FMRIB58_FA space

**Functional**:
- Denoised BOLD time series (TEDANA for multi-echo)
- Motion parameters and confounds
- DVARS and framewise displacement
- tSNR maps and carpet plots
- Component classification (TEDANA)

**ASL**:
- CBF maps (native, calibrated)
- Tissue-specific CBF values
- tSNR maps

---

## 📁 Repository Structure

```
neurovrai/
├── neurovrai/               # Main package
│   ├── config.py            # Configuration loading
│   ├── study_initialization.py  # Study setup & data discovery
│   ├── preprocess/          # Preprocessing workflows
│   │   ├── workflows/       # anat, dwi, func, asl workflows
│   │   ├── utils/           # Helper utilities
│   │   └── qc/              # Quality control modules
│   ├── analysis/            # Group statistics (VBM, TBSS, func)
│   └── connectome/          # Connectivity analysis
├── scripts/
│   ├── init_study.py        # Study initialization CLI
│   ├── analysis/            # Analysis runners
│   └── batch/               # Batch processing
├── docs/                    # Documentation
├── run_simple_pipeline.py   # Main preprocessing runner
├── verify_environment.py    # Environment checker
├── CLAUDE.md                # AI assistant guidelines
├── PROJECT_STATUS.md        # This file
└── README.md                # User documentation
```

---

## 🎓 Documentation

### User Documentation
- `README.md` - Installation, usage, configuration
- `docs/configuration.md` - Config file reference
- `docs/workflows.md` - Workflow details
- `CLAUDE.md` and `README.md` - Output organization

### Developer Documentation
- `CLAUDE.md` - AI assistant context
- `docs/DWI_PROCESSING_GUIDE.md` - DWI pipeline details

### Testing & Validation

---

## 🚀 Next Steps

### Immediate (Next Session)
1. Run complete pipeline on additional subjects for validation
2. Generate batch processing examples
3. Create group-level QC summary reports
4. Performance profiling and optimization

### Short-term
1. Implement batch processing utilities for multi-subject workflows
2. Add automated outlier detection across subjects
3. Create interactive QC dashboards
4. Enhanced error handling and logging

### Medium-term
1. FreeSurfer integration
2. Enhanced QC dashboards
3. Additional atlas support for tractography
4. Batch processing utilities

### Long-term
1. AMICO integration (if needed)
2. Docker/Singularity containerization
3. HPC cluster support
4. Web-based QC interface

---

## 📝 Recent Activity Log

### 2025-11-15
- **✅ MILESTONE**: All modalities now production-ready (Anatomical, DWI, Functional, ASL)
- **Completed**: Functional preprocessing pipeline (100%)
  - Multi-echo TEDANA denoising fully operational
  - Single-echo ICA-AROMA auto-detection working
  - Complete functional QC module (motion, DVARS, tSNR, carpet plots)
- **Moved to Production**: AMICO advanced diffusion models
  - 100x speedup for NODDI (30 sec vs 20-25 min)
  - Added SANDI and ActiveAx models
- **Fixed**: Variable naming bugs in func_preprocess.py
- **Fixed**: Functional QC directory structure (study-level qc/)
- **Integrated**: Complete DWI QC (TOPUP, Motion, DTI modules)
- **Standardized**: QC directory structure across all modalities
- **Cleaned**: Repository organization (test scripts → archive, docs → organized structure)
- **Updated**: Documentation and examples to reflect production status

### 2025-11-14 (Session - Part 2)
- **Fixed**: TEDANA NumPy compatibility - upgraded TEDANA 23.0.2 → 25.1.0
- **Fixed**: NumPy version constraint in pyproject.toml (relaxed to >=1.24,<3.0)
- **Fixed**: DWI work directory hierarchy bug (now correctly uses `work/{subject}/dwi_preprocess/`)
- **Updated**: CLAUDE.md - moved ASL and AMICO from "Planned" to "Production-Ready"
- **Updated**: README.md with latest status and bug fixes
- **Running**: Functional preprocessing with TEDANA 25.1.0 (in progress)

### 2025-11-14 (Earlier)
- **Fixed**: Multi-echo DICOM conversion bug
- **Fixed**: Functional workflow input node selection
- **Verified**: Single-shell DWI support with proper labeling
- **Verified**: Single-echo fMRI support with proper labeling
- **Investigated**: fMRI TOPUP (not applicable - no reversed PE)

### 2025-11-13 (Overnight Run)
- **Completed**: Anatomical preprocessing on validation dataset
- **Completed**: DWI preprocessing with advanced models + normalization
- **Completed**: ASL preprocessing with M0 calibration
- **Implemented**: DWI → FMRIB58_FA normalization
- **Implemented**: Functional → MNI152 normalization framework
- **Fixed**: Functional preprocessing bugs (completed successfully after fixes)

### 2025-11-12
- **Completed**: DWI TOPUP validation
- **Completed**: DKI/NODDI implementation
- **Enhanced**: Functional QC module

---

## 💡 Notes

- **GPU Required**: For optimal performance (eddy_cuda, probtrackx2_gpu)
- **FSL Required**: Version 6.0+ for all workflows
- **Python**: 3.13+ (managed via uv)
- **Tested Platform**: Linux (Ubuntu-based)

---

## 🎊 Production Status Summary

**Pipeline is production-ready for ALL modalities: anatomical, DWI (with advanced models), functional (multi/single-echo), ASL preprocessing, AND connectivity analysis.**

**Key Achievements:**
- ✅ **Complete multi-modal coverage**: T1w, DWI, fMRI, ASL
- ✅ **Advanced diffusion models**: DTI, DKI, NODDI (DIPY + AMICO), SANDI, ActiveAx
- ✅ **Multi-echo support**: TEDANA 25.1.0 for optimal fMRI denoising
- ✅ **Comprehensive QC**: Automated quality control for all modalities
- ✅ **Spatial normalization**: MNI152 (anatomical/functional), FMRIB58_FA (DWI)
- ✅ **GPU acceleration**: CUDA support for eddy, BEDPOSTX, probtrackx2, probtrackx2_gpu
- ✅ **Config-driven**: YAML-based configuration for reproducible workflows
- ✅ **BIDS-compatible**: Standardized directory structure and metadata
- ✅ **Connectivity Analysis**:
  - Functional: correlation-based FC with multiple atlases
  - Structural: probtrackx2 tractography with FreeSurfer anatomical constraints
  - Graph metrics: degree, clustering, efficiency, betweenness
  - Network-Based Statistic (NBS) for group comparisons
- ✅ **FreeSurfer Integration**: GMWMI seeding, ACT-style constraints, ventricle avoidance

**All workflows validated on real-world multi-modal datasets with comprehensive QC reports.**
