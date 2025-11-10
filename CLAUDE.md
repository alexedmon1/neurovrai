# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

- **`dwi/`**: Diffusion-weighted imaging (DTI/DWI)
  - `dti-preprocess.py`: Multi-shell DTI preprocessing (merge shells → eddy correction → DTIFit → BEDPOSTX)
  - `dti_singleShell_preprocess.py`: Single-shell variant with structural coregistration
  - TBSS (Tract-Based Spatial Statistics) utilities for FA analysis
  - `dti_tract_extraction.py`: Probabilistic tractography (requires BEDPOSTX output)

- **`rest/`**: Resting-state fMRI preprocessing
  - `rest-preproc-dev.py`: Single-echo workflow (structural coregistration → MCFLIRT → ICA-AROMA → bandpass filtering → ACompCor nuisance regression)
  - `rest_workflow.py`: Multi-echo workflow (preprocessing before TEDANA)
  - Dual regression and connectivity matrix generation utilities

- **`myelin/`**: T1w/T2w ratio myelin mapping
  - `myelin_workflow.py`: Computes T1w/T2w ratio as myelin proxy (coregister to MNI → masked division)

- **`analysis/`**: Post-processing statistical analysis
  - `cluster_analysis.py`: Aggregates FSL randomise cluster results across analyses

### Workflow Pattern

All preprocessing modules follow a consistent class-based pattern:

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
- **Diffusion**: `fsl.Eddy` (CUDA), `fsl.DTIFit`, `fsl.BEDPOSTX5` (GPU)
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

### Parameter Files
DTI eddy correction requires acquisition parameter files (not in repo):
- `acqp.txt`: Acquisition parameters (phase encoding direction, readout time)
- `index.txt`: Volume-to-acquisition mapping

These must be created manually based on scanner protocol.

## Testing

No formal test suite exists. Manual verification typically involves:
1. Checking Nipype workflow graph PNG outputs
2. Visual inspection of preprocessed NIfTI files in FSLeyes/AFNI
3. Reviewing text logs in Nipype working directories

## Code Style Notes

- Heavy use of `glob()` for file discovery with pattern matching
- Frequent `os.chdir()` calls (consider refactoring to use absolute paths)
- Subject iteration patterns with status file tracking to resume failed runs
- Path parsing via string `.split('/')` (fragile to path structure changes)
- Inconsistent path construction (mix of string concatenation and `os.path.join()`)

When modifying workflows:
- Preserve the class-based structure for consistency
- Maintain the pattern of glob-based file discovery with sequence name matching
- Keep subject iteration and status tracking patterns
- Ensure `output_type='NIFTI_GZ'` for FSL nodes
- Add `wf.write_graph()` calls for workflow visualization
