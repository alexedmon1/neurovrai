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

**✅ Completed & Validated**:
- **Anatomical Preprocessing**: T1w workflow with N4 bias correction, BET skull stripping, FLIRT/FNIRT registration
- **Tissue Segmentation**: Using ANTs Atropos (faster than FSL FAST which was taking 15+ minutes)
- **DWI Preprocessing**: Multi-shell with TOPUP distortion correction, GPU eddy, DTI fitting
- **Advanced Diffusion Models**: DKI and NODDI (DIPY-based)
- **Tractography**: GPU-accelerated probtrackx2 with atlas-based ROI extraction
- **QC Framework**: Comprehensive quality control for DWI (TOPUP, motion, DTI) and anatomical (skull stripping, segmentation, registration)
- **Configuration System**: YAML-based config with variable substitution (`mri_preprocess/config.py`)
- **Directory Standardization**: All workflows use `{outdir}/{subject}/{modality}/` pattern

**🔄 In Progress**:
- **Functional Preprocessing (RUNNING OVERNIGHT)**: Multi-echo resting-state fMRI with TEDANA 25.1.0
  - Modified TEDANA to use 225 PCA components (down from 435) for improved ICA convergence
  - Log: `logs/func_OVERNIGHT.log`
  - **ACTION FOR NEXT SESSION**: Check status of overnight run and verify outputs
  - Expected completion: ~50-55 minutes (MCFLIRT + applyxfm4D + TEDANA with 225 components)

**✅ Completed This Session (2025-11-12)**:
- Anatomical preprocessing validated (IRC805-0580101)
- DWI preprocessing with TOPUP completed (IRC805-0580101)
- DKI metrics: 9 files generated (MK, AK, RK, KFA, FA, MD, AD, RD, tensor)
- NODDI metrics: 4 files generated (FICVF, ODI, FISO, fiber direction)
- Enhanced functional QC module with DVARS and carpet plot capabilities
- Integrated QC module into resting-state workflow

**📋 Planned**:
- FreeSurfer integration for all workflows
- AMICO integration for better NODDI fitting
- Spatial normalization for DWI and functional modalities (see `docs/PLANNED_ANALYSES.md`)

### Key Design Decisions
- **Bias correction**: ANTs N4 (~2.5 min) before segmentation
- **Segmentation**: ANTs Atropos (faster than FSL FAST which was hanging)
- **DWI**: Merge-first approach with TOPUP before eddy
- **fMRI denoising**: TEDANA for multi-echo (not ICA-AROMA - redundant)
- **Work directory structure**: `{study_root}/work/{subject}/` (Nipype adds workflow name automatically)
- **Nipype DataSink hierarchy**: DataSink creates subdirectories based on `container` parameter. When `base_directory` is already `{study_root}/derivatives/{subject}/anat/`, setting `container='anat'` creates redundant `/anat/anat/` hierarchy. **Solution**: Set `container=''` (empty string) to use base_directory as-is.

## Project Overview

This repository contains Python-based MRI preprocessing pipelines built with Nipype for neuroimaging analysis. The project processes multiple MRI modalities (anatomical, diffusion, resting-state fMRI) from DICOM to analysis-ready formats, with FSL and FreeSurfer as the primary neuroimaging tools.

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

# Activate virtual environment
source .venv/bin/activate
```

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
This format is generated by `mri_preprocess/config_generator.py` from DICOM files.

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

The config loader (`mri_preprocess/config.py`) automatically:
1. Detects which format you're using
2. Normalizes study config format to preprocessing params format
3. Ensures consistency across sessions

This means **both formats work seamlessly** with all workflows.

### Variable Substitution

The config loader (`mri_preprocess/config.py`) supports:
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
from mri_preprocess.config import load_config
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

- **`dwi/`** (UPDATED - Modern Implementation): Diffusion-weighted imaging (DTI/DWI/DKI/NODDI)
  - **Modern Workflows** (`mri_preprocess/workflows/`):
    - `dwi_preprocess.py`: Multi-shell preprocessing with TOPUP distortion correction
      - Merge-first approach (shells merged BEFORE correction)
      - TOPUP for susceptibility distortion correction
      - GPU-accelerated eddy correction (eddy_cuda)
      - DTI fitting with standard metrics (FA, MD, AD, RD)
      - Optional BEDPOSTX for probabilistic tractography modeling
    - `advanced_diffusion.py`: Advanced diffusion models (VALIDATED)
      - **DKI** (Diffusion Kurtosis Imaging): MK, AK, RK, KFA metrics
      - **NODDI** (Neurite Orientation Dispersion and Density): ODI, FICVF, FISO
      - DIPY-based implementations
      - Requires multi-shell data (≥2 non-zero b-values)
    - `tractography.py`: Probabilistic tractography with atlas-based ROIs (VALIDATED)
      - GPU-accelerated probtrackx2_gpu (10-50x faster)
      - Atlas-based ROI extraction (Harvard-Oxford, JHU)
      - Seed-to-target connectivity analysis
      - Automatic connectivity matrix generation
  - **Utilities** (`mri_preprocess/utils/`):
    - `topup_helper.py`: Generate acqparams.txt and index.txt files (VALIDATED)
    - `atlas_rois.py`: Atlas-based ROI extraction for tractography (VALIDATED)
  - **Legacy Workflows** (`archive/dwi/`):
    - `dti-preprocess.py`: Original multi-shell preprocessing
    - `dti_singleShell_preprocess.py`: Single-shell variant
    - TBSS utilities for FA analysis
    - `dti_tract_extraction.py`: Original tractography implementation

- **`rest/`**: Resting-state fMRI preprocessing
  - `rest-preproc-dev.py`: Single-echo workflow (structural coregistration → MCFLIRT → ICA-AROMA → bandpass filtering → ACompCor nuisance regression)
  - `rest_workflow.py`: Multi-echo workflow (preprocessing before TEDANA)
  - Dual regression and connectivity matrix generation utilities

- **`myelin/`**: T1w/T2w ratio myelin mapping
  - `myelin_workflow.py`: Computes T1w/T2w ratio as myelin proxy (coregister to MNI → masked division)

- **`analysis/`**: Post-processing statistical analysis
  - `cluster_analysis.py`: Aggregates FSL randomise cluster results across analyses

### Workflow Pattern (UPDATED)

**Modern workflows** (`mri_preprocess/workflows/`) use functional pattern with standardized directory structure:

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
    output_dir=study_root,  # Pass study root
    work_dir=None  # Optional, defaults to {study_root}/work/{subject}/{workflow}/
)

# Outputs saved to: {study_root}/derivatives/dwi_topup/{subject}/
```

**Updated workflows:**
- ✅ `anat_preprocess.py` - Anatomical T1w/T2w preprocessing
- ✅ `dwi_preprocess.py` - DWI with TOPUP distortion correction
- ✅ `func_preprocess.py` - Functional/resting-state fMRI
- ✅ `advanced_diffusion.py` - DKI and NODDI (called from dwi_preprocess)
- ✅ `tractography.py` - Probabilistic tractography (called from dwi_preprocess)

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
from mri_preprocess.utils.topup_helper import create_topup_files_for_multishell

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

**Modern Code** (`mri_preprocess/`):
- Function-based workflow creation (more flexible)
- Uses `pathlib.Path` for cross-platform compatibility
- Configuration-driven execution (YAML configs)
- Modular utilities for reusable components
- Type hints and comprehensive docstrings
- GPU acceleration as default where available

When modifying workflows:
- **New code**: Use function-based patterns from `mri_preprocess/workflows/`
- **Legacy code**: Preserve class-based structure for consistency
- Maintain the pattern of glob-based file discovery with sequence name matching
- Keep subject iteration and status tracking patterns
- Ensure `output_type='NIFTI_GZ'` for FSL nodes
- Add `wf.write_graph()` calls for workflow visualization

## Validated Modern Workflows

### DWI Processing Pipeline (STATUS: TESTING)

**Location**: `mri_preprocess/workflows/dwi_preprocess.py`

**Main Function**: `run_dwi_multishell_topup_preprocessing()`

**Features**:
- ✅ Automatic bval/bvec/nifti merging
- ✅ TOPUP distortion correction
- ✅ GPU-accelerated eddy correction
- ✅ DTI fitting
- ✅ Optional BEDPOSTX
- ⏳ Testing on IRC805 dataset

**Usage**:
```python
from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

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

### Advanced Diffusion Models (STATUS: VALIDATED)

**Location**: `mri_preprocess/workflows/advanced_diffusion.py`

**Functions**:
- `fit_dki_model()`: Diffusion Kurtosis Imaging
- `fit_noddi_model()`: NODDI tissue modeling
- `run_advanced_diffusion_models()`: Run both DKI and NODDI

**Requirements**: Multi-shell data with ≥2 non-zero b-values

**Usage**:
```python
from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models

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

### Probabilistic Tractography (STATUS: VALIDATED)

**Location**: `mri_preprocess/workflows/tractography.py`

**Main Function**: `run_atlas_based_tractography()`

**Features**:
- GPU-accelerated probtrackx2_gpu
- Automatic atlas warping to DWI space
- Harvard-Oxford and JHU atlases supported
- Connectivity matrix generation

**Usage**:
```python
from mri_preprocess.workflows.tractography import run_atlas_based_tractography

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

## TODO: Future Enhancements

### FreeSurfer Integration
- [ ] Add FreeSurfer toggle for all workflows
- [ ] Implement `use_freesurfer` parameter
- [ ] Add `subjects_dir` configuration
- [ ] Warp FreeSurfer segmentations to DWI/fMRI space
- [ ] Extract FreeSurfer-based ROIs for tractography

**Planned API**:
```python
# Future implementation
results = run_atlas_based_tractography(
    ...
    use_freesurfer=True,
    subjects_dir='/freesurfer/SUBJECTS_DIR',
    fs_atlas='aparc+aseg'  # Use FreeSurfer parcellation
)
```

### Additional Features
- [ ] AMICO integration for better NODDI fitting
- [ ] Automated QC report generation
- [ ] BIDS compliance improvements
- [ ] Containerization (Docker/Singularity)

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
└── mri_preprocess/             # Production code
    ├── workflows/              # Validated preprocessing workflows
    ├── utils/                  # Helper functions and utilities
    └── qc/                     # Quality control modules
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
- **mri_preprocess/**: All production code
  - `workflows/`: Validated preprocessing workflows
  - `utils/`: Reusable helper functions
  - `qc/`: Quality control modules
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
