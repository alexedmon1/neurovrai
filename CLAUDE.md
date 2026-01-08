# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**neurovrai** is a production-ready MRI preprocessing and analysis platform built on FSL, ANTs, and Nipype. It processes anatomical T1w, diffusion DWI, functional fMRI, and ASL perfusion data from DICOM to publication-ready results.

**Python Version**: 3.13+ | **Package Manager**: uv | **Virtual Environment**: `.venv/`

## Essential Commands

```bash
# Install dependencies
uv sync

# Run preprocessing (primary entry point)
uv run python run_simple_pipeline.py --subject sub-001 --dicom-dir /path/to/dicom --config config.yaml

# Run specific modality
uv run python run_preprocessing.py --subject SUBJECT_ID --modality [anat|dwi|func|asl]

# Run tests
uv run python -m pytest tests/

# Verify environment
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

### Config Loading

```python
from neurovrai.preprocess.config import load_config
config = load_config(Path('config.yaml'))
```

The config loader supports variable substitution: `${project_dir}` references other config values.

## Directory Structure Conventions

All workflows use standardized output hierarchy:

```
{study_root}/
├── subjects/{subject}/nifti/{modality}/  # Raw NIfTI files
├── derivatives/{subject}/{modality}/      # Preprocessed outputs
├── transforms/{subject}/                  # Spatial transforms (centralized)
├── work/{subject}/{workflow}/             # Temporary Nipype files
├── analysis/                              # Group-level results
└── qc/{subject}/{modality}/               # QC reports
```

**Key Rules**:
- All workflows receive `output_dir` = `{study_root}/derivatives`
- Work directory defaults to `{study_root}/work/{subject}/{workflow}/`
- No intermediate folders like "mri-preprocess" or "sub-" prefixes

## Critical Guidelines

### Spatial Transforms

**NEVER resample atlases** - always co-register properly:
- Transform data TO atlas space, then apply atlas directly
- ❌ WRONG: Resample MNI atlas to native functional space
- ✅ CORRECT: Normalize 4D BOLD to MNI space, use MNI atlas directly

**Transform Naming Convention** (MANDATORY):
- Pattern: `{source}-{target}-{type}.{ext}`
- Examples: `func-t1w-affine.mat`, `t1w-mni-warp.nii.gz`, `fa-fmrib58-composite.h5`
- Storage: `{study_root}/transforms/{subject}/`

**Transform Utilities**:
```python
from neurovrai.utils.transforms import save_transform, find_transform
```

### FSL Requirements

Ensure `$FSLDIR` is set. Common interfaces via Nipype:
- Structural: `fsl.BET`, `fsl.FLIRT`, `fsl.FNIRT`
- Diffusion: `fsl.TOPUP`, `fsl.Eddy`, `fsl.DTIFit`
- Functional: `fsl.MCFLIRT`, `fsl.ICA_AROMA`

### Nipype DataSink

When `base_directory` already includes the target path, set `container=''` to avoid redundant hierarchy:
```python
# WRONG: Creates /derivatives/sub-001/anat/anat/
datasink = Node(DataSink(base_directory='/derivatives/sub-001/anat/', container='anat'))

# CORRECT: Creates /derivatives/sub-001/anat/
datasink = Node(DataSink(base_directory='/derivatives/sub-001/anat/', container=''))
```

## Code Style

**Modern code** (`neurovrai/preprocess/`):
- Function-based workflows
- `pathlib.Path` for all paths
- Type hints and comprehensive docstrings
- Config-driven execution
- Always set `output_type='NIFTI_GZ'` for FSL nodes

**When creating new workflows**:
- Use function-based patterns from existing workflows
- Add `wf.write_graph()` for workflow visualization
- Include QC integration
- Save transforms to centralized location

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Bias correction | ANTs N4 | ~2.5 min, more robust than FSL |
| Segmentation | ANTs Atropos | FSL FAST was hanging |
| DWI distortion | TOPUP + eddy | Merge-first approach |
| Multi-echo fMRI | TEDANA | ICA-AROMA is redundant for multi-echo |
| NODDI fitting | AMICO | 100x faster than DIPY (30s vs 25min) |

## FreeSurfer Integration

**✅ PRODUCTION READY** for structural and functional connectivity.

**Implemented**:
- Detection of existing FreeSurfer outputs (`freesurfer_utils.py`)
- ROI extraction from aparc+aseg (Desikan-Killiany 85 ROIs, Destrieux 165 ROIs)
- Optimized FS→T1w transform using `mri_vol2vol` (~5 sec vs 5 min with FLIRT)
- T1w→DWI transform with smart registration (DWI→T1w + inverse)
- Transform chain composition (FS→T1w→DWI)
- QC validation with correlation/NMI metrics and visual overlays (`freesurfer_qc.py`)
- Anatomical constraints for tractography (ventricle avoidance, WM waypoints, GMWMI seeding)
- Structural connectivity with `probtrackx2_gpu` (`structural_connectivity.py`)
- Functional connectivity with FreeSurfer atlases (`atlas_func_transform.py`)

**Not implemented** (lower priority):
- Cortical thickness analysis
- Surface-based rendering

**Key files**:
- `neurovrai/preprocess/utils/freesurfer_utils.py` - Detection, ROI extraction
- `neurovrai/preprocess/utils/freesurfer_transforms.py` - Transform pipeline
- `neurovrai/preprocess/qc/freesurfer_qc.py` - QC validation
- `neurovrai/connectome/structural_connectivity.py` - SC with anatomical constraints

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

## Project File Organization

- **Root**: Only `README.md`, `CLAUDE.md`, `PROJECT_STATUS.md`, `run_*.py` scripts
- **docs/**: All technical documentation
- **logs/**: All log files (gitignored)
- **archive/**: Legacy code and test scripts (don't extend)

When modifying code:
- Place new utilities in `neurovrai/preprocess/utils/` or `neurovrai/utils/`
- Place QC modules in appropriate `neurovrai/preprocess/qc/{modality}/`
- Scripts go in `scripts/` with appropriate subdirectory

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
