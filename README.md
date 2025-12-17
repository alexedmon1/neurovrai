# neurovrai

**Production-ready MRI preprocessing, group analysis, and network neuroscience.**

[![Python](https://img.shields.io/badge/python-3.13%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**neurovrai** is a complete neuroimaging analysis platform built on FSL, ANTs, and Nipype. It provides end-to-end workflows from raw DICOM to publication-ready results.

```
neurovrai/
├── preprocess/    # Subject-level preprocessing (anat, dwi, func, asl)
├── analysis/      # Group-level statistics (VBM, TBSS, ReHo/fALFF, MELODIC)
└── connectome/    # Connectivity matrices, graph metrics, NBS
```

## Features

- **Multi-Modal Preprocessing**: Anatomical T1w, diffusion DWI, functional fMRI, ASL perfusion
- **GPU Acceleration**: CUDA support for eddy and BEDPOSTX (10-50x speedup)
- **Advanced Diffusion Models**: DKI, NODDI with AMICO acceleration (100x faster)
- **Group Statistics**: VBM, TBSS, resting-state metrics with FSL randomise or nilearn GLM
- **Network Analysis**: Functional and structural connectivity, graph theory, NBS
- **Config-Driven**: YAML configuration for reproducible workflows
- **Quality Control**: Automated QC with HTML reports for all modalities

---

## Installation

### Prerequisites

```bash
# Required system dependencies
# - FSL 6.0+ (neuroimaging tools)
# - ANTs (bias correction, segmentation)
# - dcm2niix (DICOM conversion)
# - CUDA (optional, for GPU acceleration)

# Verify FSL
echo $FSLDIR
```

### Install neurovrai

```bash
# Clone repository
git clone https://github.com/alexedmon1/neurovrai.git
cd neurovrai

# Install with uv (recommended)
uv sync

# Verify installation
uv run python verify_environment.py
```

### Install ICA-AROMA (for single-echo fMRI)

```bash
mkdir -p ~/bin && cd ~/bin
git clone https://github.com/maartenmennes/ICA-AROMA.git
chmod +x ~/bin/ICA-AROMA/ICA_AROMA.py

# Update shebang to use neurovrai Python
cd /path/to/neurovrai
VENV_PYTHON=$(pwd)/.venv/bin/python
sed -i "1s|.*|#!$VENV_PYTHON|" ~/bin/ICA-AROMA/ICA_AROMA.py

# Add to PATH
echo 'export PATH="$HOME/bin/ICA-AROMA:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install dependency
uv pip install future
```

---

## Quick Start

### 1. Create Configuration

```bash
uv run python create_config.py --study-root /mnt/data/my_study
```

This creates a `config.yaml` with all processing parameters.

### 2. Run Preprocessing

```bash
# Single subject - all modalities
uv run python run_simple_pipeline.py \
    --subject sub-001 \
    --dicom-dir /mnt/data/my_study/dicom/sub-001 \
    --config config.yaml

# Skip specific modalities
uv run python run_simple_pipeline.py \
    --subject sub-001 \
    --dicom-dir /path/to/dicom \
    --config config.yaml \
    --skip-func --skip-asl
```

### 3. Run Group Analysis

```bash
# VBM analysis
uv run python scripts/analysis/run_vbm_group_analysis.py \
    --study-root /mnt/data/my_study \
    --method randomise \
    --tissue GM \
    --n-permutations 5000

# TBSS analysis
uv run python -m neurovrai.analysis.tbss.run_tbss_stats \
    --tbss-dir /mnt/data/my_study/analysis/tbss \
    --design-dir /mnt/data/my_study/data/designs/tbss \
    --n-permutations 5000
```

### 4. Compute Connectivity

```bash
# Functional connectivity (batch)
uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root /mnt/data/my_study \
    --atlases harvardoxford_cort juelich \
    --output-dir /mnt/data/my_study/analysis/connectivity

# Structural connectivity
uv run python -m neurovrai.connectome.run_structural_connectivity \
    --subject sub-001 \
    --derivatives-dir /mnt/data/my_study/derivatives \
    --atlas schaefer_200 \
    --config config.yaml
```

---

## Preprocessing Pipelines

### Anatomical (T1w)

**Pipeline**: N4 bias correction → BET skull stripping → Atropos segmentation → FLIRT/FNIRT registration

```python
from neurovrai.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing

results = run_anatomical_preprocessing(
    config=config,
    subject='sub-001',
    t1w_file=Path('T1w.nii.gz'),
    output_dir=Path('/derivatives')
)
```

**Outputs**: `brain.nii.gz`, `brain_mask.nii.gz`, `segmentation/pve_*.nii.gz`, `transforms/`

### Diffusion (DWI)

**Pipeline**: TOPUP distortion correction → GPU eddy → DTI fitting → DKI/NODDI (multi-shell) → FMRIB58 normalization

```python
from neurovrai.preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject='sub-001',
    dwi_files=[Path('dwi.nii.gz')],
    bval_files=[Path('dwi.bval')],
    bvec_files=[Path('dwi.bvec')],
    rev_phase_files=[Path('dwi_PA.nii.gz')],  # Optional TOPUP
    output_dir=Path('/derivatives'),
    run_advanced_models=True  # Enable DKI/NODDI
)
```

**Outputs**: `eddy_corrected.nii.gz`, `dti/FA.nii.gz`, `dti/MD.nii.gz`, `dki/MK.nii.gz`, `noddi/ficvf.nii.gz`

**Single-Shell Support** (In Progress): A `run_dwi_singleshell_preprocessing()` function is available for acquisitions without reverse phase-encoding data (no TOPUP). Single-shell data produces DTI metrics only (FA, MD, AD, RD). Testing and pipeline integration pending.

### Functional (fMRI)

**Pipeline**: MCFLIRT motion correction → TEDANA (multi-echo) or ICA-AROMA (single-echo) → ACompCor → Bandpass filtering → Smoothing

```python
from neurovrai.preprocess.workflows.func_preprocess import run_functional_preprocessing

results = run_functional_preprocessing(
    config=config,
    subject='sub-001',
    func_files=[Path('bold.nii.gz')],  # Single or multi-echo
    output_dir=Path('/derivatives'),
    t1w_brain=Path('derivatives/sub-001/anat/brain.nii.gz'),
    normalize_to_mni=True
)
```

**Outputs**: `preprocessed_bold.nii.gz`, `brain_mask.nii.gz`, `tsnr.nii.gz`, `qc/motion_qc.html`

### ASL (Perfusion)

**Pipeline**: MCFLIRT → Label-control separation → CBF quantification → M0 calibration → Partial volume correction

```python
from neurovrai.preprocess.workflows.asl_preprocess import run_asl_preprocessing

results = run_asl_preprocessing(
    config=config,
    subject='sub-001',
    asl_file=Path('asl.nii.gz'),
    output_dir=Path('/derivatives'),
    t1w_brain=Path('brain.nii.gz'),
    gm_mask=Path('pve_1.nii.gz'),
    wm_mask=Path('pve_2.nii.gz'),
    normalize_to_mni=True
)
```

**Outputs**: `cbf.nii.gz`, `cbf_pvc.nii.gz`, `m0.nii.gz`, `cbf_stats.json`

---

## Group Analysis

### VBM (Voxel-Based Morphometry)

Analyze structural brain differences using tissue probability maps.

```python
from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data, run_vbm_analysis

# Prepare data
prepare_vbm_data(
    subjects=['sub-001', 'sub-002'],
    derivatives_dir=Path('/derivatives'),
    output_dir=Path('/analysis/vbm'),
    tissue_type='GM',
    smoothing_fwhm=4.0
)

# Run analysis (FSL randomise or nilearn GLM)
run_vbm_analysis(
    vbm_dir=Path('/analysis/vbm/GM'),
    participants_file=Path('participants.csv'),
    formula='age + sex + group',
    method='randomise',
    n_permutations=5000
)
```

### TBSS (Tract-Based Spatial Statistics)

White matter analysis using diffusion tensor metrics.

```python
from neurovrai.analysis.tbss.prepare_tbss import prepare_tbss_data
from neurovrai.analysis.tbss.run_tbss_stats import run_tbss_statistics

prepare_tbss_data(
    derivatives_dir=Path('/derivatives'),
    output_dir=Path('/analysis/tbss'),
    subjects=['sub-001', 'sub-002']
)

run_tbss_statistics(
    tbss_dir=Path('/analysis/tbss'),
    participants_file=Path('participants.csv'),
    n_permutations=5000
)
```

### Resting-State (ReHo/fALFF)

Compute regional homogeneity and low-frequency fluctuation amplitude.

```python
from neurovrai.analysis.func.resting_workflow import run_resting_state_analysis

results = run_resting_state_analysis(
    func_file=Path('preprocessed_bold.nii.gz'),
    mask_file=Path('brain_mask.nii.gz'),
    output_dir=Path('/derivatives/sub-001/func'),
    compute_reho=True,
    compute_falff=True
)
```

### MELODIC (Group ICA)

Group-level independent component analysis.

```python
from neurovrai.analysis.func.melodic import run_melodic_group_analysis

run_melodic_group_analysis(
    derivatives_dir=Path('/derivatives'),
    output_dir=Path('/analysis/melodic'),
    subjects=['sub-001', 'sub-002'],
    n_components=20
)
```

---

## Connectome Analysis

### Functional Connectivity

Compute correlation matrices from fMRI timeseries using native-space analysis.

```python
from neurovrai.connectome import extract_roi_timeseries, compute_functional_connectivity

# Extract timeseries
timeseries, roi_names = extract_roi_timeseries(
    data_file=Path('preprocessed_bold.nii.gz'),
    atlas=Path('atlas.nii.gz')
)

# Compute connectivity
fc_results = compute_functional_connectivity(
    timeseries=timeseries,
    roi_names=roi_names,
    method='pearson',
    fisher_z=True
)
```

**Available Atlases**: Harvard-Oxford (cortical/subcortical), Juelich, Talairach, Schaefer, Desikan-Killiany

### Structural Connectivity

Tractography-based connectivity using FSL probtrackx2 with GPU acceleration.

```python
from neurovrai.connectome.structural_connectivity import compute_structural_connectivity

sc_results = compute_structural_connectivity(
    bedpostx_dir=Path('dwi.bedpostX'),
    atlas_file=Path('atlas_in_dwi.nii.gz'),
    output_dir=Path('/connectome'),
    n_samples=5000,
    avoid_ventricles=True,
    batch_mode=True,   # Process one ROI at a time (GPU-friendly)
    use_gpu=True       # Use probtrackx2_gpu for acceleration
)
```

**Batch Mode**: When `batch_mode=True`, tractography processes one seed ROI at a time instead of all ROIs simultaneously. This dramatically reduces memory usage (~3GB vs ~18GB+ for network mode) and enables reliable GPU acceleration. Recommended for large atlases (>50 regions).

### Graph Metrics

Network topology analysis.

```python
from neurovrai.connectome import compute_node_metrics, compute_global_metrics, identify_hubs

node_metrics = compute_node_metrics(matrix=fc_matrix, threshold=0.3)
global_metrics = compute_global_metrics(matrix=fc_matrix, threshold=0.3)
hubs = identify_hubs(node_metrics, method='betweenness', percentile=90)
```

### Network-Based Statistic

Permutation-based group comparison.

```python
from neurovrai.connectome import compute_network_based_statistic

nbs_results = compute_network_based_statistic(
    group1_matrices,
    group2_matrices,
    threshold=3.0,
    n_permutations=5000
)
```

---

## Configuration

Create `config.yaml` for study-specific parameters:

```yaml
project_dir: /mnt/data/my_study
derivatives_dir: ${project_dir}/derivatives
work_dir: ${project_dir}/work

execution:
  plugin: MultiProc
  n_procs: 8

templates:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz

anatomical:
  bet:
    frac: 0.5
    robust: true
  run_qc: true

diffusion:
  topup:
    readout_time: 0.05
  eddy_config:
    use_cuda: true
  run_qc: true

functional:
  tr: 2.0
  highpass: 0.001
  lowpass: 0.08
  fwhm: 6
  tedana:
    enabled: false  # true for multi-echo
  acompcor:
    enabled: true
  run_qc: true
```

---

## Project Structure

```
neurovrai/
├── README.md                   # This file
├── CLAUDE.md                   # AI assistant guidelines
├── PROJECT_STATUS.md           # Current implementation status
├── config.yaml                 # Example configuration
├── create_config.py            # Config generator
├── run_simple_pipeline.py      # Main pipeline runner
├── verify_environment.py       # Installation verifier
│
├── neurovrai/                  # Main package
│   ├── preprocess/             # Preprocessing workflows
│   │   ├── workflows/          # anat, dwi, func, asl
│   │   ├── utils/              # Helper functions
│   │   └── qc/                 # Quality control
│   ├── analysis/               # Group statistics
│   │   ├── anat/               # VBM
│   │   ├── tbss/               # TBSS
│   │   ├── func/               # ReHo, fALFF, MELODIC
│   │   └── stats/              # Randomise, GLM, cluster reports
│   └── connectome/             # Connectivity analysis
│       ├── functional_connectivity.py
│       ├── structural_connectivity.py
│       ├── graph_metrics.py
│       └── batch_*.py          # Batch processing
│
├── scripts/                    # Utility scripts
│   ├── analysis/               # Group analysis runners
│   ├── batch/                  # Batch processing
│   └── monitoring/             # Progress monitoring
│
├── examples/                   # Usage examples
├── docs/                       # Documentation
└── archive/                    # Legacy code
```

---

## Output Structure

All outputs follow a standardized hierarchy:

```
{study_root}/
├── derivatives/                # Preprocessed outputs
│   └── {subject}/
│       ├── anat/               # Anatomical
│       ├── dwi/                # Diffusion
│       ├── func/               # Functional
│       └── asl/                # Perfusion
├── analysis/                   # Group analysis
│   ├── vbm/
│   ├── tbss/
│   └── connectivity/
├── work/                       # Temporary Nipype files
└── qc/                         # Quality control reports
```

---

## Quality Control

All preprocessing generates automated QC reports in HTML format:

| Modality | QC Metrics |
|----------|------------|
| **Anatomical** | Skull stripping, tissue segmentation, MNI registration |
| **Diffusion** | Motion (eddy), TOPUP correction, DTI metrics, skull stripping |
| **Functional** | Motion (MCFLIRT), tSNR, DVARS, skull stripping |
| **ASL** | Motion, CBF distributions, tSNR, skull stripping |

Reports saved to: `derivatives/{subject}/{modality}/qc/`

---

## Performance

| Process | CPU | GPU | Speedup |
|---------|-----|-----|---------|
| Eddy correction | ~45 min | ~4 min | 10x |
| NODDI (DIPY) | 20-25 min | - | - |
| NODDI (AMICO) | 30 sec | - | 100x |
| BEDPOSTX | 12-24 hr | 1-4 hr | 10x |

**Recommended**: 8+ cores, 32+ GB RAM, NVIDIA GPU with CUDA

---

## Citation

If you use neurovrai, please cite the underlying tools:

- **FSL**: Jenkinson et al. (2012). FSL. NeuroImage, 62(2):782-790.
- **TEDANA**: DuPre et al. (2021). TE-dependent analysis of multi-echo fMRI. JOSS, 6(66):3669.
- **NBS**: Zalesky et al. (2010). Network-based statistic. NeuroImage, 53(4):1197-1207.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: https://github.com/alexedmon1/neurovrai/issues
- **Documentation**: See `docs/` directory and module READMEs
