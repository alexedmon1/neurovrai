# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**neurovrai** is a production-ready MRI preprocessing and analysis platform built on FSL, ANTs, and Nipype. It processes anatomical T1w/T2w, diffusion DWI, functional fMRI, and ASL perfusion data from DICOM to publication-ready results.

**Python Version**: 3.13+ | **Package Manager**: uv | **Virtual Environment**: `.venv/`

## Essential Commands

```bash
# Install dependencies
uv sync

# Run preprocessing (primary entry point)
uv run python run_simple_pipeline.py --subject sub-001 --dicom-dir /path/to/dicom --config config.yaml

# Run with parallel modalities (ThreadPoolExecutor, NOT ProcessPoolExecutor)
uv run python run_simple_pipeline.py --subject sub-001 --parallel-modalities

# Run specific modality
uv run python run_simple_pipeline.py --subject sub-001 --skip-dwi --skip-func --skip-asl  # Only T1w

# Run tests (note: no formal pytest suite, tests are in archive/tests/)
uv run python verify_environment.py

# Generate config file
uv run python create_config.py --study-root /path/to/study
```

**IMPORTANT**: Always use `uv run python` instead of activating the venv manually. This ensures consistent dependency resolution.

## Architecture

```
neurovrai/
├── preprocess/           # Subject-level preprocessing
│   ├── workflows/        # Main workflow files (t1w, dwi, func, asl)
│   ├── utils/            # Helper functions
│   └── qc/               # Quality control modules
├── analysis/             # Group-level statistics
│   ├── anat/             # VBM, T1-T2 ratio, WMH
│   ├── func/             # ReHo, fALFF, MELODIC
│   ├── tbss/             # Tract-based spatial statistics
│   └── stats/            # Randomise, GLM, cluster reports
└── connectome/           # Connectivity analysis
    ├── functional_connectivity.py
    ├── structural_connectivity.py
    └── graph_metrics.py
```

### Workflow Entry Points

| Modality | Workflow Function | Location |
|----------|-------------------|----------|
| T1w | `run_t1w_preprocessing()` | `preprocess/workflows/t1w_preprocess.py` |
| T2w | `run_t2w_preprocessing()` | `preprocess/workflows/t2w_preprocess.py` |
| DWI | `run_dwi_multishell_topup_preprocessing()` | `preprocess/workflows/dwi_preprocess.py` |
| Functional | `run_functional_preprocessing()` | `preprocess/workflows/func_preprocess.py` |
| ASL | `run_asl_preprocessing()` | `preprocess/workflows/asl_preprocess.py` |

## Configuration System

### Three-Level Loading
```python
from neurovrai.config import load_config
config = load_config(Path('config.yaml'))
```

1. Load study config from provided path
2. Load defaults from `default.yaml` (in study dir or package dir)
3. Merge with study overrides (study config takes precedence)

### Variable Substitution
- `${ENV_VAR}` - Environment variables
- `${config.key.subkey}` - Config references

### Two Format Support (Auto-Converts)
- **Format 1** (config_generator.py): `study`, `paths`, `sequence_mappings`
- **Format 2** (current): `project_dir`, `rawdata_dir`, derivatives config

## Directory Structure Conventions

All workflows use standardized output hierarchy:

```
{study_root}/
├── subjects/{subject}/nifti/{modality}/  # Raw NIfTI files
├── derivatives/{subject}/{modality}/      # Preprocessed outputs
├── transforms/{subject}/                  # Centralized transform registry
├── work/{subject}/{workflow}/             # Nipype work files
├── analysis/                              # Group-level results
└── qc/{subject}/{modality}/               # QC reports
```

**Critical Rules**:
- Workflows receive `output_dir` = `{study_root}/derivatives`
- Work directory uses function name: `work/{subject}/dwi_preprocess/`
- **NO intermediate folders** like "mri-preprocess" or "sub-" prefixes
- ❌ WRONG: `derivatives/sub-001/mri-preprocess/anat/`
- ✅ CORRECT: `derivatives/sub-001/anat/`

## Transform Registry System

### Centralized Storage Pattern
- Location: `{study_root}/transforms/{subject}/`
- Metadata: `transforms.json` (creation date, source/target spaces)
- Check existing before computing (avoids recomputation)

### Naming Convention (MANDATORY)
- Pattern: `{source}-{target}-{type}.{ext}`
- Examples:
  - `func-t1w-affine.mat`
  - `t1w-mni-warp.nii.gz`
  - `fa-fmrib58-composite.h5`

### Transform Utilities
```python
from neurovrai.utils.transforms import save_transform, find_transform
```

## Data Auto-Detection Patterns

### Functional Data Selection
- **Multi-echo detection**: `_e1`, `_e2`, `_e3` filename patterns
- **Run selection**: Latest COMPLETE run with consecutive echoes
- **Processing choice**:
  - Multi-echo (≥2): TEDANA (ICA-AROMA skipped)
  - Single-echo: ICA-AROMA (or skip if not installed)

### DWI Data Filtering
**Scanner-processed maps to exclude** (they block processing):
- ADC maps
- dWIP (diffusion weighted isotropic projection)
- facWIP (fractional anisotropy weighted isotropic projection)
- isoWIP (isotropic weighted imaging projection)

### Shell Detection
- Groups by unique bval values
- Single-shell: DTI only
- Multi-shell: DTI + DKI + NODDI

## Critical Implementation Patterns

### Execution Model
**Parallel modalities** use `ThreadPoolExecutor`, NOT `ProcessPoolExecutor`:
- Workflows use multiprocessing internally
- Must be thread-safe at workflow level
- Example: `run_simple_pipeline.py --parallel-modalities`

### Nipype DataSink Anti-Pattern
**MAJOR GOTCHA**: If `base_directory` already contains target path, use `container=''`:
```python
# WRONG: Creates /derivatives/sub-001/anat/anat/
datasink = Node(DataSink(base_directory='/derivatives/sub-001/anat/', container='anat'))

# CORRECT: Creates /derivatives/sub-001/anat/
datasink = Node(DataSink(base_directory='/derivatives/sub-001/anat/', container=''))
```

### Atropos Tissue Standardization
**Atropos + K-means produces arbitrary tissue ordering** - MUST standardize:
```python
from neurovrai.preprocess.utils.standardize_atropos_tissues import standardize_atropos_tissues
standardize_atropos_tissues(derivatives_dir)  # Called after workflow
```
- Identifies tissues by intensity: CSF (lowest) → GM (middle) → WM (highest)
- Creates standardized: `csf.nii.gz`, `gm.nii.gz`, `wm.nii.gz`

### ACompCor Implementation
- Tissue masks from **anatomical** segmentation, NOT functional FAST
- Registers anatomical masks to functional space
- Extracts principal components (PCA)

## FreeSurfer Integration

**✅ PRODUCTION READY** for structural and functional connectivity.

### Key Optimizations
- **FS→T1w transform**: `mri_vol2vol --regheader` (~5 sec) vs FLIRT (~5 min)
- **T1w→DWI strategy**: Register DWI→T1w (low-res to high-res), then invert
- **Transform chain**: FS→T1w→DWI via composition
- **probtrackx2_gpu syntax**: Use `--option=value` (NOT `--option value`)

### Anatomical Constraints
- **Ventricle avoidance**: FS labels `[4, 5, 14, 15, 43, 44, 72]`
- **White matter masks**: FS labels `[2, 41, 77, 251-255]`
- **GMWMI seeding**: Gray-white matter interface from FreeSurfer

### Key Files
- `neurovrai/preprocess/utils/freesurfer_utils.py` - Detection, ROI extraction
- `neurovrai/preprocess/utils/freesurfer_transforms.py` - Transform pipeline
- `neurovrai/preprocess/qc/freesurfer_qc.py` - QC validation
- `neurovrai/connectome/structural_connectivity.py` - SC with constraints

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bias correction | ANTs N4 | ~2.5 min, more robust than FSL |
| Segmentation | ANTs Atropos | FSL FAST was hanging |
| DWI distortion | TOPUP per-shell | Apply before merging shells |
| Multi-echo fMRI | TEDANA | ICA-AROMA redundant for multi-echo |
| NODDI fitting | AMICO | 100x faster than DIPY (30s vs 25min) |
| BET fractional intensity | 0.5 (anat), 0.3 (others) | Modality-specific defaults |

## Common Gotchas

1. **Spatial Transforms**: NEVER resample atlases - transform data TO atlas space
   - ❌ WRONG: Resample MNI atlas to native functional space
   - ✅ CORRECT: Normalize 4D BOLD to MNI space, use MNI atlas directly

2. **TOPUP Strategy**: Apply per-shell BEFORE merging (critical for multi-shell)

3. **DICOM Parameter Extraction**: ASL τ/PLD auto-extracted from headers

4. **FSL Environment**: Ensure `$FSLDIR` is set, use `output_type='NIFTI_GZ'` for all FSL nodes

5. **Work Directory Naming**: Defaults to function name (e.g., `dwi_preprocess` from `run_dwi_preprocessing`)

## Running Analysis Pipelines

```bash
# VBM group analysis
uv run python scripts/analysis/run_vbm_group_analysis.py \
    --study-root /path/to/study --method randomise --tissue GM

# TBSS statistics
uv run python -m neurovrai.analysis.tbss.run_tbss_stats \
    --tbss-dir /path/to/tbss --design-dir /path/to/designs

# Functional connectivity (batch)
uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root /path/to/study --atlases harvardoxford_cort juelich

# Structural connectivity
uv run python -m neurovrai.connectome.run_structural_connectivity \
    --subject sub-001 --derivatives-dir /path/to/derivatives --atlas schaefer_200
```

## Code Style

**Modern code** (`neurovrai/preprocess/`):
- Function-based workflows (not class-based)
- `pathlib.Path` for all paths
- Type hints and comprehensive docstrings
- Config-driven execution
- Module-level loggers throughout

**When creating new workflows**:
- Use function-based patterns from existing workflows
- Add `wf.write_graph()` for workflow visualization
- Include QC integration
- Save transforms to centralized location
- Handle Atropos tissue standardization if using segmentation

## Testing & Development

- **Tests location**: `archive/tests/` (ad-hoc validation scripts)
- **No formal pytest suite** currently implemented
- **Environment check**: `uv run python verify_environment.py`
- **Extensive logging**: Module-level loggers everywhere for debugging

## Session Management

**IMPORTANT**: Always update `PROJECT_STATUS.md`:
- After completing significant features or bug fixes
- When changing production-ready status of any module
- Before ending a session (summarize what was accomplished)
- Update the "Last Updated" date at the top of the file

The project status file tracks:
- Latest updates with dates
- Production-ready status of all modules
- Known issues and limitations
- Planned features and their priority