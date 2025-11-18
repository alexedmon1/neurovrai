# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goals & Current Status

### Overall Objectives
1. **Production-ready MRI preprocessing pipeline** for anatomical (T1w), diffusion (DWI), and resting-state fMRI data
2. **Config-driven architecture** using YAML for all processing parameters (documented in README.md)
3. **Standardized directory hierarchy** across all workflows: `{study_root}/derivatives/{subject}/{modality}/`
4. **Quality control framework** with automated QC for all modalities
5. **Performance optimization** - Fast, reliable processing with GPU acceleration where available

### Current Implementation Status

**✅ Completed & Production-Ready**:
- **Anatomical Preprocessing**: T1w workflow with N4 bias correction, BET skull stripping, FLIRT/FNIRT registration, tissue segmentation (ANTs Atropos)
- **DWI Preprocessing**: Multi-shell/single-shell with optional TOPUP distortion correction, GPU eddy, DTI fitting, spatial normalization to FMRIB58_FA
- **Advanced Diffusion Models**:
  - DKI (DIPY-based): MK, AK, RK, KFA metrics
  - NODDI (DIPY or AMICO): FICVF, ODI, FISO - AMICO is 100x faster (30 sec vs 20-25 min)
  - AMICO also supports SANDI and ActiveAx models
- **Tractography**: GPU-accelerated probtrackx2 with atlas-based ROI extraction (Harvard-Oxford, JHU)
- **ASL Preprocessing**: Motion correction, CBF quantification with kinetic modeling, M0 calibration, partial volume correction, automated DICOM parameter extraction
- **QC Framework**: Comprehensive quality control for all modalities
  - **DWI**: TOPUP, motion (eddy), DTI metrics, skull stripping
  - **Anatomical**: Skull stripping, tissue segmentation, registration
  - **Functional**: Motion (MCFLIRT), tSNR, DVARS, skull stripping
  - **ASL**: Motion, CBF distributions, tSNR, skull stripping
- **Configuration System**: YAML-based config with variable substitution (`neurovrai/preprocess/config.py`)
- **Directory Standardization**: All workflows use `{outdir}/{subject}/{modality}/` pattern

**🔄 In Progress (95% Complete)**:
- **Functional Preprocessing**: Multi-echo (TEDANA) and single-echo (ICA-AROMA) support
  - Recent fixes (2025-11-14): Multi-echo DICOM conversion bug, workflow input node selection
  - Currently testing complete pipeline on IRC805-0580101
  - All features implemented: TEDANA, ICA-AROMA auto-detection, ACompCor, bandpass, smoothing, registration to MNI152

**✅ Completed Recent Sessions (2025-11-13 to 2025-11-14)**:
- **Bug Fixes (2025-11-14)**:
  - Fixed DWI work directory hierarchy (now correctly uses `work/{subject}/dwi_preprocess/`)
  - Fixed TEDANA NumPy 2.0 compatibility (upgraded 23.0.2 → 25.1.0)
  - Fixed multi-echo DICOM conversion and workflow routing
  - Relaxed dependency version constraints in pyproject.toml
- **Spatial Normalization Implementation**:
  - **DWI → FMRIB58_FA**: Complete normalization pipeline implemented and tested
    - Created `neurovrai/preprocess/utils/dwi_normalization.py`
    - Integrated into `dwi_preprocess.py` workflow
    - Generates forward warp (group analyses) + inverse warp (tractography ROIs)
    - Tested on IRC805-0580101: Successfully normalized 12 metrics (FA + DTI + DKI + NODDI)
  - **Functional → MNI152**: Transform reuse strategy implemented
    - Created `neurovrai/preprocess/utils/func_normalization.py`
    - Integrated into `func_preprocess.py` workflow
    - Reuses BBR transform from ACompCor (zero redundant computation)
    - Reuses anatomical→MNI152 transforms from anatomical preprocessing
    - Concatenates transforms via `convertwarp` for single-step normalization
  - **BBR Performance Fix**: Switched from BBR to correlation ratio (20x speedup: 54s vs 10+ min)
  - **Transform Management**: All transforms saved to standardized `transforms/` directories
- **TEDANA Multi-Echo Denoising** (IRC805-0580101):
  - Motion correction: MCFLIRT + applyxfm4D (~40 minutes)
  - TEDANA with 225 PCA components (~1h 18min)
  - ICA converged after 10 attempts (seed 51)
  - Component selection: 208 accepted, 17 rejected
  - 82.55% variance explained
  - All outputs generated: denoised BOLD, component maps, confounds, HTML report

**✅ Completed Previous Sessions**:
- **2025-11-12**: DWI preprocessing with TOPUP, DKI/NODDI metrics, functional QC module enhancements
- **2025-11-11**: ASL preprocessing implementation with M0 calibration and PVC
- **2025-11-10**: AMICO integration (NODDI/SANDI/ActiveAx models with 100x speedup)
- **2025-11-09**: Anatomical preprocessing validation, tissue segmentation QC

**⚠️ FreeSurfer Integration Status - NOT Production Ready**:
- **Current Status**: Detection and extraction hooks only (as of 2025-11-14)
- **Implemented**: FreeSurfer output detection, ROI extraction from aparc+aseg, config integration
- **Missing (CRITICAL)**:
  - ❌ Anatomical→DWI transform pipeline (ROIs would be in wrong space!)
  - ❌ FreeSurfer native space handling and validation
  - ❌ Transform quality control
  - ❌ Validation that FreeSurfer T1 matches preprocessing T1
- **DO NOT enable** `freesurfer.enabled = true` until transform pipeline is complete
- **Estimated work**: 2-3 full development sessions
- **See**: `PROJECT_STATUS.md` lines 128-163 for detailed status

### Key Design Decisions
- **Bias correction**: ANTs N4 (~2.5 min) before segmentation
- **Segmentation**: ANTs Atropos (faster than FSL FAST which was hanging)
- **DWI**: Merge-first approach with TOPUP before eddy
- **fMRI denoising**: TEDANA for multi-echo (not ICA-AROMA - redundant)
- **Work directory structure**: `{study_root}/work/{subject}/` (Nipype adds workflow name automatically)
- **Nipype DataSink hierarchy**: DataSink creates subdirectories based on `container` parameter. When `base_directory` is already `{study_root}/derivatives/{subject}/anat/`, setting `container='anat'` creates redundant `/anat/anat/` hierarchy. **Solution**: Set `container=''` (empty string) to use base_directory as-is.

## Project Overview

This repository contains Python-based MRI preprocessing pipelines built with Nipype for neuroimaging analysis. The project processes multiple MRI modalities (anatomical T1w, diffusion DWI, resting-state fMRI, arterial spin labeling ASL) from DICOM to analysis-ready formats, with FSL, ANTs, and FreeSurfer as the primary neuroimaging tools.

## Development Environment

**Python Version**: 3.13+
**Package Manager**: uv (see `pyproject.toml` and `uv.lock`)
**Virtual Environment**: `.venv/` (managed by uv)

### Key Dependencies
- **nipype** (1.10.0+): Workflow engine wrapping FSL/FreeSurfer interfaces
- **nibabel** (5.3.2+): NIfTI file I/O
- **pydicom** (3.0.1+): DICOM reading/parsing
- **FSL**: Required system dependency (referenced via `$FSLDIR` environment variable)
- **FreeSurfer**: Required for cortical reconstruction workflows
- **tedana** (23.0.2+): Multi-echo fMRI denoising
- **pandas**, **numpy**, **scipy**: Data analysis and numerical operations

### Installation
```bash
# Install dependencies with uv
uv sync

# Activate virtual environment (optional - see "Using uv" below)
source .venv/bin/activate
```

### Using uv (IMPORTANT)

**All Python execution and dependency management should be handled via `uv`.**

This ensures consistent dependency resolution and avoids environment issues.

#### Running Python Scripts
```bash
# ✅ CORRECT - Use uv run
uv run python script.py
uv run python -m pytest tests/

# ❌ INCORRECT - Don't manually activate and run
source .venv/bin/activate
python script.py
```

#### Managing Dependencies
```bash
# ✅ CORRECT - Use uv pip
uv pip install package_name
uv pip install 'package_name<version'
uv pip list
uv pip show package_name

# ❌ INCORRECT - Don't use pip directly
pip install package_name
```

#### Why This Matters
- **Dependency Resolution**: uv ensures all dependencies are compatible with `pyproject.toml` and `uv.lock`
- **Consistency**: Avoids "works on my machine" issues from version mismatches
- **Lock File**: uv automatically updates `uv.lock` when dependencies change
- **Common Issue**: Running scripts without uv may use wrong Python version or miss dependencies

#### Dependency Version Conflicts
**Example**: TEDANA 23.0.2 requires `nilearn<0.11` due to API changes in nilearn 0.12+

When encountering import errors:
1. Check versions: `uv pip list | grep package_name`
2. Install compatible version: `uv pip install 'package_name<version'`
3. Verify import works: `uv run python -c "import package_name"`
4. Document in `pyproject.toml` if it's a permanent constraint

## Configuration

The pipeline uses a YAML-based configuration system as documented in `README.md`.

### Configuration File Format

Create a `config.yaml` file in the project root:

```yaml
# Project paths
project_dir: /mnt/bytopia/IRC805         # Study root directory
rawdata_dir: ${project_dir}/subjects      # Raw data (IRC805 uses 'subjects' not 'rawdata')
derivatives_dir: ${project_dir}/derivatives
work_dir: ${project_dir}/work
transforms_dir: ${project_dir}/transforms

# Execution settings
execution:
  plugin: MultiProc
  n_procs: 6

# Template files
templates:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
  mni152_t1_1mm: /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz

# Anatomical preprocessing
anatomical:
  bet:
    frac: 0.5
    reduce_bias: true
    robust: true
  fast:
    bias_iters: 1
    bias_lowpass: 10
  registration_method: fsl  # or 'ants'
  run_qc: true

# Diffusion preprocessing
diffusion:
  denoise_method: dwidenoise
  topup:
    readout_time: 0.05
  eddy_config:
    flm: linear
    slm: linear
    use_cuda: true
  run_qc: true

# Functional preprocessing
functional:
  tr: 1.029
  te: [10.0, 30.0, 50.0]
  highpass: 0.001
  lowpass: 0.08
  fwhm: 6
  tedana:
    enabled: true
    tedpca: kundu
    tree: kundu
  aroma:
    enabled: false
  acompcor:
    enabled: true
    num_components: 5
    variance_threshold: 0.5
  run_qc: true
```

### Dual Config Format Support

The pipeline supports **two configuration formats** for maximum flexibility:

#### Format 1: Preprocessing Parameters (Current - Recommended)
This is the format used in `config.yaml`. It directly specifies paths and processing parameters.

```yaml
project_dir: /mnt/bytopia/IRC805
rawdata_dir: ${project_dir}/subjects
derivatives_dir: ${project_dir}/derivatives
work_dir: ${project_dir}/work

anatomical:
  bet:
    frac: 0.5
diffusion:
  topup:
    readout_time: 0.05
functional:
  tr: 1.029
```

**Use this format when:**
- You manually organize your data
- You know the sequence names and don't need auto-detection
- You want simple, direct configuration (recommended for most users)

#### Format 2: Study Config (Auto-generated)
This format is generated by `neurovrai/preprocess/config_generator.py` from DICOM files.

```yaml
study:
  name: IRC805 Study
  code: IRC805
  base_dir: /mnt/bytopia/IRC805
paths:
  sourcedata: ${study.base_dir}/sourcedata
  rawdata: ${study.base_dir}/rawdata
  derivatives: ${study.base_dir}/derivatives
sequence_mappings:
  t1w: ['WIP_3D_T1_TFE_SAG_CS3']
  dwi: ['DTI_2shell_b1000_b2000_MB4']
```

**Use this format when:**
- You want to auto-detect sequences from DICOM directories
- You need to map scanner-specific sequence names to modalities
- You're setting up a new study from scratch

#### Automatic Format Detection and Normalization

The config loader (`neurovrai/preprocess/config.py`) automatically:
1. Detects which format you're using
2. Normalizes study config format to preprocessing params format
3. Ensures consistency across sessions

This means **both formats work seamlessly** with all workflows.

### Variable Substitution

The config loader (`neurovrai/preprocess/config.py`) supports:
- **Environment variables**: `${ENV_VAR}`
- **Config references**: `${project_dir}` references other config values

Example:
```yaml
project_dir: /mnt/bytopia/IRC805
derivatives_dir: ${project_dir}/derivatives  # Expands to /mnt/bytopia/IRC805/derivatives
```

### Using Configuration

**In Python scripts**:
```python
from neurovrai.preprocess.config import load_config
from pathlib import Path

# Load config
config = load_config(Path('config.yaml'))

# Access values
project_dir = Path(config['project_dir'])
n_procs = config['execution']['n_procs']
```

**With production runner**:
```bash
# Uses config.yaml in current directory by default
python run_preprocessing.py --subject IRC805-0580101 --modality anat

# Or specify custom config
python run_preprocessing.py --subject IRC805-0580101 --modality anat --config /path/to/custom.yaml

# Override project_dir from command line
python run_preprocessing.py --subject IRC805-0580101 --modality anat --study-root /different/path
```

## Directory Structure

All preprocessing workflows use a standardized directory hierarchy:

```
{study_root}/                          # e.g., /mnt/bytopia/development/IRC805/
├── dicoms/                            # Raw DICOM files
├── nifti/                             # Converted NIfTI files
├── derivatives/                       # Preprocessed outputs
│   ├── anat_preproc/{subject}/
│   ├── dwi_topup/{subject}/
│   ├── func_preproc/{subject}/
│   ├── advanced_diffusion/{subject}/
│   └── tractography/{subject}/
├── work/                              # Temporary Nipype files (can be deleted)
│   └── {subject}/{workflow}/
└── dwi_params/                        # Acquisition parameter files
```

**Key principles:**
- All workflows accept `output_dir` as the **study root** (not derivatives directory)
- `work_dir` is optional and defaults to `{study_root}/work/{subject}/{workflow}/`
- Outputs are organized as `{study_root}/derivatives/{workflow}/{subject}/`
- See `docs/DIRECTORY_STRUCTURE.md` for detailed documentation

## Architecture

The codebase is organized by MRI modality, with each module containing class-based preprocessing workflows:

### Module Structure

- **`dicom/`**: DICOM to NIfTI conversion using dcm2niix
  - `bids.py`: Converts DICOM folders to modality-organized NIfTI structure
  - Uses pydicom SeriesDescription to route files to correct modalities

- **`anat/`**: T1w/T2w anatomical preprocessing
  - `anat-preproc.py`: T1w workflow (reorient → FAST bias correction → BET → FLIRT/FNIRT to MNI152)
  - `freesurfer.py`: Wrapper for FreeSurfer `recon-all` with T1w+T2w inputs
  - VBM (Voxel-Based Morphometry) utilities for group analysis

- **`dwi/`** (✅ Production-Ready): Diffusion-weighted imaging (DTI/DWI/DKI/NODDI)
  - **Modern Workflows** (`neurovrai/preprocess/workflows/`):
    - `dwi_preprocess.py`: Multi-shell/single-shell preprocessing (✅ VALIDATED)
      - Auto-detects single vs multi-shell data
      - Optional TOPUP distortion correction (auto-enabled when reverse PE images available)
      - GPU-accelerated eddy correction (eddy_cuda)
      - DTI fitting with standard metrics (FA, MD, AD, RD)
      - Spatial normalization to FMRIB58_FA template
      - Optional BEDPOSTX for probabilistic tractography modeling
    - `advanced_diffusion.py`: Advanced diffusion models (✅ VALIDATED)
      - **DKI** (Diffusion Kurtosis Imaging): MK, AK, RK, KFA metrics (DIPY)
      - **NODDI** (Neurite Orientation): FICVF, ODI, FISO (DIPY or AMICO)
      - Auto-skips for single-shell data
      - Requires multi-shell data (≥2 non-zero b-values)
    - `amico_models.py`: AMICO-accelerated microstructure models (✅ VALIDATED)
      - **NODDI**: 100x faster than DIPY (30 sec vs 20-25 min)
      - **SANDI**: Soma and neurite density imaging
      - **ActiveAx**: Axon diameter distribution modeling
      - Uses convex optimization for 100-1000x speedup
    - `tractography.py`: Probabilistic tractography with atlas-based ROIs (✅ VALIDATED)
      - GPU-accelerated probtrackx2_gpu (10-50x faster)
      - Atlas-based ROI extraction (Harvard-Oxford, JHU)
      - Optional FreeSurfer ROI support (hooks only - NOT production ready)
      - Seed-to-target connectivity analysis
      - Automatic connectivity matrix generation
  - **Utilities** (`neurovrai/preprocess/utils/`):
    - `topup_helper.py`: Generate acqparams.txt and index.txt files (✅ VALIDATED)
    - `atlas_rois.py`: Atlas-based ROI extraction for tractography (✅ VALIDATED)
    - `dwi_normalization.py`: Spatial normalization to FMRIB58_FA (✅ VALIDATED)
    - `gradient_timing.py`: Extract/estimate gradient timing for AMICO SANDI/ActiveAx
  - **Legacy Workflows** (`archive/dwi/`):
    - `dti-preprocess.py`: Original multi-shell preprocessing
    - `dti_singleShell_preprocess.py`: Single-shell variant
    - TBSS utilities for FA analysis
    - `dti_tract_extraction.py`: Original tractography implementation

- **`func/`** (🔄 95% Complete): Resting-state fMRI preprocessing
  - **Modern Workflows** (`neurovrai/preprocess/workflows/`):
    - `func_preprocess.py`: Multi-echo and single-echo fMRI preprocessing
      - Auto-detects single vs multi-echo data
      - **Multi-echo**: TEDANA denoising (optimal for multi-echo)
      - **Single-echo**: ICA-AROMA motion artifact removal (auto-enabled)
      - Motion correction (MCFLIRT)
      - ACompCor nuisance regression using anatomical tissue masks
      - Bandpass temporal filtering
      - Spatial smoothing
      - Registration to anatomical space (BBR or correlation ratio)
      - Optional spatial normalization to MNI152
  - **Utilities** (`neurovrai/preprocess/utils/`):
    - `func_normalization.py`: Transform reuse for efficient MNI normalization
    - `acompcor_helper.py`: Tissue mask registration and component extraction
  - **Legacy Workflows** (`archive/rest/`):
    - `rest-preproc-dev.py`: Single-echo workflow
    - `rest_workflow.py`: Original multi-echo workflow
    - Dual regression and connectivity matrix generation utilities

- **`asl/`** (✅ Production-Ready): Arterial Spin Labeling preprocessing
  - **Modern Workflows** (`neurovrai/preprocess/workflows/`):
    - `asl_preprocess.py`: Complete ASL preprocessing pipeline (✅ VALIDATED)
      - Automated DICOM parameter extraction (τ, PLD)
      - Motion correction (MCFLIRT)
      - Label-control separation and subtraction
      - CBF quantification with standard kinetic model (Alsop et al., 2015)
      - M0 calibration with white matter reference (corrects for estimation bias)
      - Partial volume correction (PVC) for tissue-specific CBF
      - Registration to anatomical space
      - Optional spatial normalization to MNI152
      - Tissue-specific CBF statistics (GM, WM, CSF)
  - **Utilities** (`neurovrai/preprocess/utils/`):
    - `asl_cbf.py`: CBF quantification and calibration functions
    - `dicom_asl_params.py`: Extract ASL parameters from DICOM headers
  - **QC Modules** (`neurovrai/preprocess/qc/`):
    - `asl_qc.py`: Motion QC, CBF distributions, tSNR analysis, skull stripping QC, HTML reports

- **`myelin/`**: T1w/T2w ratio myelin mapping
  - `myelin_workflow.py`: Computes T1w/T2w ratio as myelin proxy (coregister to MNI → masked division)

- **`analysis/`**: Post-processing statistical analysis
  - `cluster_analysis.py`: Aggregates FSL randomise cluster results across analyses

### Workflow Pattern (UPDATED)

**Modern workflows** (`neurovrai/preprocess/workflows/`) use functional pattern with standardized directory structure:

```python
from pathlib import Path
from neurovrai.preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

# Study root directory
study_root = Path('/mnt/bytopia/development/IRC805')

results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject='IRC805-0580101',
    dwi_files=[...],
    bval_files=[...],
    bvec_files=[...],
    rev_phase_files=[...],
    output_dir=study_root,  # Pass study root
    work_dir=None  # Optional, defaults to {study_root}/work/{subject}/{workflow}/
)

# Outputs saved to: {study_root}/derivatives/dwi_topup/{subject}/
```

**Updated workflows:**
- ✅ `anat_preprocess.py` - Anatomical T1w/T2w preprocessing (Production-Ready)
- ✅ `dwi_preprocess.py` - DWI with optional TOPUP distortion correction (Production-Ready)
- ✅ `func_preprocess.py` - Functional/resting-state fMRI (95% complete, testing)
- ✅ `asl_preprocess.py` - Arterial Spin Labeling perfusion imaging (Production-Ready)
- ✅ `advanced_diffusion.py` - DKI and NODDI (called from dwi_preprocess, Production-Ready)
- ✅ `amico_models.py` - AMICO-accelerated NODDI/SANDI/ActiveAx (Production-Ready)
- ✅ `tractography.py` - Probabilistic tractography (called from dwi_preprocess, Production-Ready)

**Legacy workflows** use class-based pattern:

```python
class ModuleName:
    def __init__(self, subject, folder, parameters, outdir):
        # Initialize file paths using glob patterns
        # Set working directory with os.chdir()

    def preproc_method(self):
        # Create Nipype nodes for FSL/FreeSurfer interfaces
        # Build Workflow with node connections
        # Execute with wf.run('MultiProc', plugin_args={'n_procs': N})
```

**Key Characteristics**:
- Classes change directory (`os.chdir()`) to subject folders during initialization
- File discovery uses `glob()` with pattern matching on specific sequence names
- Workflows output to subject-level directories specified in `outdir`
- Parallel execution via Nipype's MultiProc plugin
- GPU acceleration enabled where available (CUDA for eddy, BEDPOSTX)

## Common Commands

### Running Workflows

Each module is designed to be run as a standalone script with subject iteration:

```bash
# Anatomical preprocessing
python anat/anat-preproc.py

# DTI multi-shell preprocessing
python dwi/dti-preprocess.py

# Resting-state preprocessing
python rest/rest-preproc-dev.py

# FreeSurfer cortical reconstruction
python anat/freesurfer.py

# DICOM to NIfTI conversion
python dicom/bids.py
```

Scripts typically include example usage blocks at the bottom that:
1. Glob subject folders from a study directory (e.g., `/mnt/bytopia/IRC805/subjects/*/nifti`)
2. Parse paths to extract subject IDs
3. Instantiate processing classes
4. Execute workflows with error tracking

### FSL Commands

The codebase relies heavily on FSL tools via Nipype. Common FSL operations:

- **Structural**: `fsl.Reorient2Std`, `fsl.FAST`, `fsl.BET`, `fsl.FLIRT`, `fsl.FNIRT`
- **Diffusion**:
  - `fsl.TOPUP`: Susceptibility distortion correction (NEW)
  - `fsl.ApplyTOPUP`: Apply TOPUP correction to images (NEW)
  - `fsl.Eddy` (CUDA): Motion and eddy current correction with TOPUP integration
  - `fsl.DTIFit`: Diffusion tensor fitting
  - `fsl.BEDPOSTX5` (GPU): Probabilistic fiber orientation modeling
  - `probtrackx2_gpu`: GPU-accelerated probabilistic tractography (NEW)
- **Functional**: `fsl.MCFLIRT`, `fsl.ICA_AROMA`, `fsl.GLM`
- **Utilities**: `fsl.Merge`, `fsl.ExtractROI`, `fsl.ApplyWarp`, `fsl.ApplyMask`

Ensure `$FSLDIR` environment variable is set (typically `/usr/local/fsl`).

## Important Context

### File Path Assumptions
- Scripts assume network storage mount points (e.g., `/mnt/bytopia/IRC805/`)
- Subject data organized as: `<study_root>/subjects/<subject_id>/nifti/<modality>/`
- Output directories typically at subject root: `<study_root>/subjects/<subject_id>/`

### Hardcoded Sequence Names
File discovery uses hardcoded sequence name patterns specific to the scanner protocol:
- T1w: `'WIP_3D_T1_TFE_SAG_CS3'`
- DTI shells: `'DTI_2shell_b1000_b2000_MB4'`, `'DTI_1shell_b3000_MB4'`
- Multi-echo rest: Variations of `'RESTING ME3 MB3 SENSE3'`

When working with new data, check `dicom/bids.py:support.set_modality()` for the full list of recognized sequence names.

### GPU Acceleration
Several workflows require CUDA-enabled GPU:
- `fsl.Eddy`: Set `use_cuda=True`
- `fsl.BEDPOSTX5`: Set `use_gpu=True`

Load CUDA module before running: `module load cuda/9.1` (or appropriate version)

### Workflow Execution
- All workflows use `wf.run('MultiProc', plugin_args={'n_procs': N})` for parallel execution
- `n_procs` typically ranges from 2-10 depending on module
- Some scripts use `joblib.Parallel` for subject-level parallelization (e.g., `myelin_workflow.py`)
- Workflows generate `workflow_graph.dot` visualization files

### MNI152 Templates
Standard space registrations use FSL's MNI152 templates:
- T1w: `$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz` (2mm) or `MNI152_T1_1mm_brain.nii.gz` (1mm)
- Masks: `MNI152_T1_2mm_brain_mask.nii.gz`

### Parameter Files (AUTOMATED)
TOPUP and eddy correction require acquisition parameter files:
- `acqparams.txt`: Phase encoding parameters for each unique acquisition
- `index.txt`: Maps each DWI volume to corresponding line in acqparams.txt

**Automated Generation** (VALIDATED):
```python
from neurovrai.preprocess.utils.topup_helper import create_topup_files_for_multishell

acqparams, index = create_topup_files_for_multishell(
    dwi_files=[Path('b1000.nii.gz'), Path('b2000.nii.gz')],
    pe_direction='AP',  # or 'PA', 'LR', 'RL'
    readout_time=0.05,  # seconds (check protocol)
    output_dir=Path('/study/dwi_params')
)
```

## Testing

No formal test suite exists. Manual verification typically involves:
1. Checking Nipype workflow graph PNG outputs
2. Visual inspection of preprocessed NIfTI files in FSLeyes/AFNI
3. Reviewing text logs in Nipype working directories

## Code Style Notes

**Legacy Code** (`archive/`):
- Heavy use of `glob()` for file discovery with pattern matching
- Frequent `os.chdir()` calls (being refactored to use absolute paths)
- Subject iteration patterns with status file tracking to resume failed runs
- Path parsing via string `.split('/')` (fragile to path structure changes)
- Class-based workflow patterns

**Modern Code** (`neurovrai/preprocess/`):
- Function-based workflow creation (more flexible)
- Uses `pathlib.Path` for cross-platform compatibility
- Configuration-driven execution (YAML configs)
- Modular utilities for reusable components
- Type hints and comprehensive docstrings
- GPU acceleration as default where available

When modifying workflows:
- **New code**: Use function-based patterns from `neurovrai/preprocess/workflows/`
- **Legacy code**: Preserve class-based structure for consistency
- Maintain the pattern of glob-based file discovery with sequence name matching
- Keep subject iteration and status tracking patterns
- Ensure `output_type='NIFTI_GZ'` for FSL nodes
- Add `wf.write_graph()` calls for workflow visualization

## Production-Ready Modern Workflows

### DWI Processing Pipeline (✅ Production-Ready)

**Location**: `neurovrai/preprocess/workflows/dwi_preprocess.py`

**Main Function**: `run_dwi_multishell_topup_preprocessing()`

**Features**:
- ✅ Automatic bval/bvec/nifti merging
- ✅ Auto-detects single-shell vs multi-shell data
- ✅ Optional TOPUP distortion correction (auto-enabled when reverse PE images available)
- ✅ GPU-accelerated eddy correction
- ✅ DTI fitting with standard metrics
- ✅ Spatial normalization to FMRIB58_FA template
- ✅ Optional BEDPOSTX for probabilistic tractography
- ✅ Tested and validated on IRC805-0580101 (multi-shell)

**Usage**:
```python
from neurovrai.preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject='IRC805-0580101',
    dwi_files=[Path('b1000_b2000.nii.gz'), Path('b3000.nii.gz')],
    bval_files=[Path('b1000_b2000.bval'), Path('b3000.bval')],
    bvec_files=[Path('b1000_b2000.bvec'), Path('b3000.bvec')],
    rev_phase_files=[Path('SE_EPI_PA1.nii.gz'), Path('SE_EPI_PA2.nii.gz')],
    output_dir=Path('/derivatives'),
    work_dir=Path('/work'),
    run_bedpostx=False
)
```

### Advanced Diffusion Models (✅ Production-Ready)

**Location**: `neurovrai/preprocess/workflows/advanced_diffusion.py`

**Functions**:
- `fit_dki_model()`: Diffusion Kurtosis Imaging (DIPY-based)
- `fit_noddi_model()`: NODDI tissue modeling (DIPY-based)
- `run_advanced_diffusion_models()`: Run DKI, NODDI, and optionally AMICO models

**Requirements**: Multi-shell data with ≥2 non-zero b-values (auto-skips for single-shell)

**Usage**:
```python
from neurovrai.preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models

results = run_advanced_diffusion_models(
    dwi_file=Path('dwi_eddy_corrected.nii.gz'),
    bval_file=Path('dwi.bval'),
    bvec_file=Path('dwi_rotated.bvec'),  # Use eddy-rotated bvecs
    mask_file=Path('dwi_mask.nii.gz'),
    output_dir=Path('/derivatives/advanced_models'),
    fit_dki=True,
    fit_noddi=True
)

# Access outputs
print(f"Mean Kurtosis: {results['dki']['mk']}")
print(f"Neurite Density: {results['noddi']['ficvf']}")
```

### Probabilistic Tractography (✅ Production-Ready)

**Location**: `neurovrai/preprocess/workflows/tractography.py`

**Main Function**: `run_atlas_based_tractography()`

**Features**:
- GPU-accelerated probtrackx2_gpu (10-50x faster than CPU)
- Automatic atlas warping to DWI space
- Harvard-Oxford and JHU atlases supported
- Optional FreeSurfer ROI support (hooks only - transform pipeline not production ready)
- Connectivity matrix generation
- Tested and validated on IRC805-0580101

**Usage**:
```python
from neurovrai.preprocess.workflows.tractography import run_atlas_based_tractography

results = run_atlas_based_tractography(
    config=config,
    subject='sub-001',
    bedpostx_dir=Path('bedpostx.bedpostX'),
    dwi_reference=Path('FA.nii.gz'),
    output_dir=Path('/derivatives/tractography'),
    seed_regions=['hippocampus_l', 'hippocampus_r'],
    target_regions=['thalamus_l', 'thalamus_r'],
    atlas='HarvardOxford-subcortical',
    n_samples=5000,
    use_gpu=True
)

# Access connectivity results
connectivity = results['connectivity']
```

### AMICO-Accelerated Microstructure Models (✅ Production-Ready)

**Location**: `neurovrai/preprocess/workflows/amico_models.py`

**Main Functions**:
- `fit_noddi_amico()`: NODDI with 100x speedup (30 sec vs 20-25 min DIPY)
- `fit_sandi_amico()`: SANDI (Soma And Neurite Density Imaging)
- `fit_activeax_amico()`: ActiveAx (Axon diameter distribution)

**Features**:
- Convex optimization for 100-1000x speedup over traditional fitting
- Same outputs as DIPY implementations, validated for accuracy
- Requires multi-shell data (≥2 non-zero b-values)
- SANDI and ActiveAx require gradient timing parameters (auto-extracted or estimated)

**Usage**:
```python
from neurovrai.preprocess.workflows.amico_models import fit_noddi_amico

results = fit_noddi_amico(
    dwi_file=Path('dwi_eddy_corrected.nii.gz'),
    bval_file=Path('dwi.bval'),
    bvec_file=Path('dwi_rotated.bvec'),
    mask_file=Path('dwi_mask.nii.gz'),
    output_dir=Path('/derivatives/noddi_amico')
)

# Runtime: ~30 seconds (vs 20-25 min with DIPY)
# Outputs: ficvf, odi, fiso, dir
```

**Performance Comparison**:
- NODDI: 100x faster (30 sec vs 20-25 min)
- SANDI: 3-6 min (no DIPY equivalent)
- ActiveAx: 3-6 min (no DIPY equivalent)

**Documentation**: See `docs/archive/AMICO_INTEGRATION_COMPLETE.md`

### ASL Preprocessing (✅ Production-Ready)

**Location**: `neurovrai/preprocess/workflows/asl_preprocess.py`

**Main Function**: `run_asl_preprocessing()`

**Features**:
- Automated DICOM parameter extraction (labeling duration τ, post-labeling delay PLD)
- Motion correction with MCFLIRT
- Label-control separation and quantification
- CBF quantification using standard kinetic model (Alsop et al., 2015)
- M0 calibration with white matter reference (corrects for estimation bias)
- Partial volume correction (PVC) for improved tissue-specific CBF accuracy
- Registration to anatomical space
- Optional spatial normalization to MNI152
- Tissue-specific CBF statistics (GM, WM, CSF)
- Comprehensive QC: motion metrics, CBF distributions, tSNR analysis, skull stripping QC

**Usage**:
```python
from neurovrai.preprocess.workflows.asl_preprocess import run_asl_preprocessing

results = run_asl_preprocessing(
    config=config,
    subject='sub-001',
    asl_file=Path('asl.nii.gz'),
    output_dir=Path('/study_root'),
    t1w_brain=Path('T1w_brain.nii.gz'),
    gm_mask=Path('gm_mask.nii.gz'),
    wm_mask=Path('wm_mask.nii.gz'),
    dicom_dir=Path('dicom/asl/'),  # For auto parameter extraction
    normalize_to_mni=True
)

# Outputs: CBF maps, M0 maps, tissue-specific statistics, QC report
```

**Tested on**: IRC805-0580101 (pCASL with M0 calibration)

## Quality Control Framework

### Skull Stripping QC (✅ Production-Ready)

Comprehensive skull stripping quality control is available for all modalities. The QC modules assess brain extraction quality and flag potential issues.

#### DWI Skull Strip QC

**Location**: `neurovrai/preprocess/qc/dwi/skull_strip_qc.py`

**Class**: `DWISkullStripQualityControl`

**Features**:
- Brain mask coverage statistics (volume, voxel count, bounding box)
- Intensity quality metrics (contrast ratio, variance)
- Automated quality flags (LOW_CONTRAST, HIGH_VARIANCE, SMALL/LARGE_BRAIN_VOLUME)
- Visual overlay plots (red contours on b0 images)
- JSON metrics export for batch analysis

**Usage**:
```python
from neurovrai.preprocess.qc.dwi.skull_strip_qc import DWISkullStripQualityControl

qc = DWISkullStripQualityControl(
    subject="IRC805-0580101",
    dwi_dir=Path("/study/derivatives/IRC805-0580101/dwi"),
    qc_dir=Path("/study/qc/dwi/IRC805-0580101/skull_strip")
)

# Auto-detects b0 and mask files
results = qc.run_qc()

# Check quality
if not results['quality']['quality_pass']:
    print(f"Quality flags: {results['quality']['quality_flags']}")
```

#### Functional Skull Strip QC

**Location**: `neurovrai/preprocess/qc/func_qc.py`

**Function**: `compute_skull_strip_qc()`

**Usage**:
```python
from neurovrai.preprocess.qc.func_qc import compute_skull_strip_qc

results = compute_skull_strip_qc(
    func_mean_file=Path("mean_func.nii.gz"),
    mask_file=Path("brain_mask.nii.gz"),
    output_dir=Path("/study/qc/func/subject/skull_strip"),
    subject="IRC805-0580101"
)

print(f"Brain volume: {results['brain_volume_cm3']:.2f} cm³")
print(f"Quality pass: {results['quality_pass']}")
```

#### ASL Skull Strip QC

**Location**: `neurovrai/preprocess/qc/asl_qc.py`

**Function**: `compute_asl_skull_strip_qc()`

**Usage**:
```python
from neurovrai.preprocess.qc.asl_qc import compute_asl_skull_strip_qc

results = compute_asl_skull_strip_qc(
    asl_mean_file=Path("M0.nii.gz"),
    mask_file=Path("brain_mask.nii.gz"),
    output_dir=Path("/study/qc/asl/subject/skull_strip"),
    subject="IRC805-0580101"
)
```

#### Anatomical Skull Strip QC

**Location**: `neurovrai/preprocess/qc/anat/skull_strip_qc.py`

**Class**: `SkullStripQualityControl`

**Usage**:
```python
from neurovrai.preprocess.qc.anat.skull_strip_qc import SkullStripQualityControl

qc = SkullStripQualityControl(
    subject="IRC805-0580101",
    anat_dir=Path("/study/derivatives/IRC805-0580101/anat"),
    qc_dir=Path("/study/qc/anat/IRC805-0580101/skull_strip")
)

results = qc.run_qc()
```

#### QC Outputs

All skull strip QC modules generate:
- **Overlay plots**: `brain_mask_overlay.png` or `skull_strip_overlay.png`
- **Metrics JSON**: Quantitative metrics for automated analysis
- **Quality flags**:
  - `LOW_CONTRAST`: Poor brain/non-brain separation (contrast ratio < 1.5-2.0)
  - `HIGH_VARIANCE`: Inconsistent intensities within mask
  - `SMALL_BRAIN_VOLUME`: Possible over-stripping (< 500 cm³)
  - `LARGE_BRAIN_VOLUME`: Possible under-stripping (> 2500 cm³)

**Documentation**: See `docs/skull_strip_qc_usage.md` for detailed usage examples and troubleshooting.

## TODO: Future Enhancements

### FreeSurfer Integration (⚠️ NOT Production Ready)
**Current Status**: Detection and extraction hooks only - transform pipeline missing

**Critical Missing Components**:
- [ ] Anatomical→DWI transform pipeline (required for tractography ROIs)
- [ ] FreeSurfer native space handling and validation
- [ ] Transform quality control and accuracy validation
- [ ] Validation that FreeSurfer T1 matches preprocessing T1

**Estimated Development Time**: 2-3 full sessions

**DO NOT enable** `freesurfer.enabled = true` until complete

See `PROJECT_STATUS.md` lines 128-163 for detailed status.

### Additional Features
- [ ] Enhanced group-level QC reports with interactive dashboards
- [ ] Automated outlier detection across subjects
- [ ] BIDS compliance improvements (currently BIDS-compatible but not formally validated)
- [ ] Containerization (Docker/Singularity) for portable deployment
- [ ] HPC cluster integration for batch processing
- [ ] Web-based QC interface for quality review

## Project Organization

### File Structure

The project follows a clean, organized structure:

```
human-mri-preprocess/
├── README.md                   # Main project documentation
├── CLAUDE.md                   # This file - AI assistant guidelines
├── run_preprocessing.py        # Production preprocessing runner
├── logs/                       # All log files (gitignored)
├── docs/                       # All documentation
│   ├── README.md               # Documentation navigation guide
│   ├── implementation/         # Technical implementation details
│   ├── status/                 # Progress tracking documents
│   ├── amico/                  # AMICO-specific documentation
│   └── archive/                # Outdated/superseded documentation
├── archive/
│   └── tests/                  # Archived test scripts
└── neurovrai/                  # Production package (renamed from mri_preprocess)
    └── preprocess/             # Preprocessing modules
        ├── workflows/          # Validated preprocessing workflows
        ├── utils/              # Helper functions and utilities
        └── qc/                 # Quality control modules
            ├── anat/           # Anatomical QC (skull strip, segmentation, registration)
            ├── dwi/            # DWI QC (motion, TOPUP, DTI, skull strip)
            ├── func_qc.py      # Functional QC (motion, tSNR, skull strip)
            └── asl_qc.py       # ASL QC (motion, CBF, tSNR, skull strip)
```

### Organization Guidelines

**Documentation**:
- **Keep in root**: Only `README.md` and `CLAUDE.md`
- **docs/**: All technical documentation
  - User guides: `cli.md`, `configuration.md`, `workflows.md`
  - Implementation: `docs/implementation/` for technical details
  - Status: `docs/status/` for progress tracking
  - Archive: `docs/archive/` for outdated docs

**Code**:
- **Keep in root**: Only `run_preprocessing.py` (production runner)
- **neurovrai/preprocess/**: All production code
  - `workflows/`: Validated preprocessing workflows
  - `utils/`: Reusable helper functions
  - `qc/`: Quality control modules
    - `anat/`: Anatomical QC modules
    - `dwi/`: DWI QC modules
    - `func_qc.py`: Functional QC functions
    - `asl_qc.py`: ASL QC functions
- **archive/tests/**: Test scripts (not in production use)

**Logs**:
- **All logs go to** `logs/` directory
- Scripts should create logs with: `logging.FileHandler('logs/script_name.log')`
- The `logs/` directory is gitignored

**Output Data** (not in repository):

All workflows MUST use the standardized output directory hierarchy:

```
{outdir}/{subject}/{modality}/
```

**Directory Variables:**
- `outdir`: Base output directory for all derivatives (e.g., `/mnt/bytopia/IRC805/derivatives`)
- `subject`: Subject identifier (e.g., `IRC805-0580101`)
- `modality`: Data type (`anat`, `dwi`, `func`)

**Complete Study Hierarchy:**
```
{study_root}/                                  # e.g., /mnt/bytopia/IRC805/
├── subjects/{subject}/nifti/{modality}/       # Raw NIfTI files
├── derivatives/                               # ALL processed outputs (outdir)
│   └── {subject}/                             # One directory per subject
│       ├── anat/                              # Anatomical preprocessing
│       │   ├── brain.nii.gz
│       │   ├── brain_mask.nii.gz
│       │   ├── bias_corrected.nii.gz
│       │   ├── segmentation/                  # Tissue probability maps
│       │   │   ├── pve_0.nii.gz              # CSF
│       │   │   ├── pve_1.nii.gz              # GM
│       │   │   └── pve_2.nii.gz              # WM
│       │   └── transforms/                    # Spatial transforms
│       ├── dwi/                               # DWI preprocessing
│       │   ├── eddy_corrected.nii.gz
│       │   ├── dti/                           # DTI metrics
│       │   ├── dki/                           # DKI metrics (if run)
│       │   └── noddi/                         # NODDI metrics (if run)
│       └── func/                              # Functional preprocessing
│           ├── preprocessed_bold.nii.gz
│           └── qc/                            # Modality-specific QC
├── work/{subject}/{workflow}/                 # Temporary Nipype files
└── qc/{subject}/                              # Study-level QC reports
    ├── anat/
    ├── dwi/
    └── func/
```

**Implementation Rules:**
1. **All workflows** receive `output_dir` parameter = path to `{study_root}/derivatives`
2. **Workflow creates**: `{output_dir}/{subject}/{modality}/` directory
3. **No intermediate folders** like "mri-preprocess" or "sub-" prefixes
4. **Work directory**: `{study_root}/work/{subject}/{workflow_name}/`
5. **QC reports**: `{study_root}/qc/{subject}/{modality}/`

**Example Usage:**
```python
# In all workflows
def run_workflow(subject: str, output_dir: Path, ...):
    # output_dir = /mnt/bytopia/IRC805/derivatives
    derivatives_dir = output_dir / subject / 'anat'  # or 'dwi' or 'func'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    # Work directory
    work_dir = output_dir.parent / 'work' / subject / 'anat_preproc'
    work_dir.mkdir(parents=True, exist_ok=True)
```

### Cleanup Guidelines

When adding new files:
1. **Documentation**: Place in appropriate `docs/` subdirectory
2. **Logs**: Ensure scripts output to `logs/` directory
3. **Test scripts**: Place in `archive/tests/` or delete after validation
4. **Status docs**: Use `docs/status/` for progress tracking, move to `docs/archive/` when complete

When finishing a feature:
1. Archive temporary test scripts to `archive/tests/`
2. Move implementation notes to `docs/implementation/`
3. Move status docs to `docs/archive/` when complete
4. Update main `README.md` if user-facing features changed
