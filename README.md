# neurovrai

**Complete neuroimaging analysis platform: preprocessing, group statistics, and network neuroscience.**

[![Version](https://img.shields.io/badge/version-2.0.0--alpha-blue.svg)](https://github.com/alexedmon1/neurovrai)
[![Python](https://img.shields.io/badge/python-3.13%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**neurovrai** (French: "true neuro") provides end-to-end neuroimaging analysis from raw DICOM to publication-ready results across three integrated modules:

```
neurovrai/
├── preprocess/    ✅ Subject-level preprocessing (anat, dwi, func, asl)
├── analysis/      ✅ Group-level statistics (VBM, TBSS, ReHo/fALFF, MELODIC)
└── connectome/    ✅ Connectivity & network neuroscience
```

**Key Features**:
- 🚀 **Multi-Modal**: Anatomical, diffusion, functional, ASL preprocessing
- ⚡ **GPU Accelerated**: CUDA support for eddy, BEDPOSTX (10-50x speedup)
- 🧠 **Advanced Models**: DKI, NODDI with AMICO acceleration (100x faster)
- 📊 **Group Statistics**: VBM, TBSS, resting-state metrics with FSL randomise or nilearn GLM
- 🌐 **Network Analysis**: Connectivity matrices, graph theory, NBS
- 🎯 **Config-Driven**: YAML configuration for reproducible workflows
- 🔍 **Quality Control**: Automated QC with HTML reports for all modalities

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Preprocessing](#preprocessing)
  - [Anatomical](#anatomical-preprocessing)
  - [Diffusion](#diffusion-preprocessing)
  - [Functional](#functional-preprocessing)
  - [ASL](#asl-preprocessing)
- [Group Analysis](#group-analysis)
  - [VBM (Voxel-Based Morphometry)](#vbm-voxel-based-morphometry)
  - [TBSS (Tract-Based Spatial Statistics)](#tbss-tract-based-spatial-statistics)
  - [Resting-State fMRI](#resting-state-fmri-analysis)
  - [MELODIC (Group ICA)](#melodic-group-ica)
- [Connectome Analysis](#connectome-analysis)
  - [ROI Extraction](#roi-extraction)
  - [Functional Connectivity](#functional-connectivity)
  - [Graph Theory Metrics](#graph-theory-metrics)
  - [Network-Based Statistic](#network-based-statistic)
- [Configuration](#configuration)
- [Examples](#examples)

---

## Installation

### Prerequisites

**Required system dependencies**:
- FSL 6.0+ (neuroimaging tools)
- Python 3.13+
- CUDA toolkit (optional, for GPU acceleration)

**Install FSL**:
```bash
# Follow official FSL installation: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
# Ensure $FSLDIR is set in your environment
```

**Install ICA-AROMA** (required for single-echo fMRI preprocessing):
```bash
# Clone ICA-AROMA to ~/bin
mkdir -p ~/bin
cd ~/bin
git clone https://github.com/maartenmennes/ICA-AROMA.git

# Make executable
chmod +x ~/bin/ICA-AROMA/ICA_AROMA.py

# Update shebang to use neurovrai venv Python
cd /path/to/neurovrai  # Replace with your neurovrai path
VENV_PYTHON=$(pwd)/.venv/bin/python
sed -i "1s|.*|#!$VENV_PYTHON|" ~/bin/ICA-AROMA/ICA_AROMA.py

# Add to PATH (if not already added)
echo 'export PATH="$HOME/bin/ICA-AROMA:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install required Python package
uv pip install future

# Verify installation
ICA_AROMA.py -h
```

### Install neurovrai

```bash
# Clone repository
git clone https://github.com/alexedmon1/neurovrai.git
cd neurovrai

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Verify Installation

```bash
# Check FSL
echo $FSLDIR
fslinfo --version

# Check Python environment
uv run python -c "import neurovrai; print('neurovrai ready!')"
```

---

## Quick Start

### 1. Create Configuration File

```yaml
# config.yaml
project_dir: /mnt/data/my_study
rawdata_dir: ${project_dir}/rawdata
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
  registration_method: fsl
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
    enabled: false  # Set true for multi-echo
  acompcor:
    enabled: true
  run_qc: true
```

### 2. Run Preprocessing

**CLI Method**:
```bash
# Anatomical preprocessing
uv run python run_preprocessing.py \
    --subject sub-001 \
    --modality anat \
    --config config.yaml

# Diffusion preprocessing
uv run python run_preprocessing.py \
    --subject sub-001 \
    --modality dwi \
    --config config.yaml
```

**Python API**:
```python
from pathlib import Path
from neurovrai.preprocess.config import load_config
from neurovrai.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing

config = load_config(Path('config.yaml'))

run_anatomical_preprocessing(
    config=config,
    subject='sub-001',
    t1w_file=Path('/data/sub-001/anat/T1w.nii.gz'),
    output_dir=Path(config['derivatives_dir'])
)
```

### 3. Run Group Analysis

```python
from neurovrai.analysis.anat.vbm_workflow import run_vbm_analysis

run_vbm_analysis(
    vbm_dir='/data/derivatives/vbm/GM',
    participants_file='/data/participants.csv',
    formula='age + sex + group',
    contrasts=['age_positive', 'group_patients_vs_controls'],
    n_permutations=5000
)
```

### 4. Connectivity Analysis

```python
from neurovrai.connectome import (
    extract_roi_timeseries,
    compute_functional_connectivity,
    compute_network_based_statistic
)

# Extract timeseries
timeseries, roi_names = extract_roi_timeseries(
    data_file='preprocessed_bold.nii.gz',
    atlas='schaefer_400.nii.gz'
)

# Compute connectivity
fc_results = compute_functional_connectivity(
    timeseries=timeseries,
    roi_names=roi_names,
    method='pearson',
    fisher_z=True
)

# Group comparison with NBS
nbs_results = compute_network_based_statistic(
    group1_matrices,
    group2_matrices,
    threshold=3.0,
    n_permutations=5000
)
```

---

## Preprocessing

### Anatomical Preprocessing

Process T1-weighted (and optionally T2-weighted) structural images.

**Pipeline**:
1. N4 bias field correction (ANTs)
2. Brain extraction (FSL BET)
3. Tissue segmentation (ANTs Atropos)
4. Registration to MNI152 (FSL FLIRT/FNIRT)
5. Quality control reports

**CLI Usage**:
```bash
uv run python run_preprocessing.py \
    --subject sub-001 \
    --modality anat \
    --config config.yaml
```

**Python API**:
```python
from neurovrai.preprocess.workflows.anat_preprocess import run_anatomical_preprocessing

results = run_anatomical_preprocessing(
    config=config,
    subject='sub-001',
    t1w_file=Path('/data/sub-001/anat/T1w.nii.gz'),
    t2w_file=Path('/data/sub-001/anat/T2w.nii.gz'),  # Optional
    output_dir=Path('/data/derivatives')
)
```

**Outputs**:
```
derivatives/sub-001/anat/
├── brain.nii.gz                    # Skull-stripped brain
├── brain_mask.nii.gz               # Brain mask
├── bias_corrected.nii.gz           # N4 corrected
├── segmentation/
│   ├── pve_0.nii.gz               # CSF probability
│   ├── pve_1.nii.gz               # GM probability
│   └── pve_2.nii.gz               # WM probability
├── transforms/
│   ├── anat2mni_warp.nii.gz       # Nonlinear warp
│   └── mni2anat_warp.nii.gz       # Inverse warp
└── qc/
    └── skull_strip_qc.html         # QC report
```

---

### Diffusion Preprocessing

Process diffusion-weighted imaging (DWI) data with optional TOPUP distortion correction.

**Pipeline**:
1. Denoising (dwidenoise)
2. TOPUP distortion correction (optional, auto-detected)
3. GPU-accelerated eddy current correction
4. DTI fitting (FA, MD, AD, RD)
5. Advanced models: DKI, NODDI (multi-shell only)
6. AMICO acceleration (NODDI/SANDI/ActiveAx)
7. Spatial normalization to FMRIB58_FA
8. Quality control reports

**CLI Usage**:
```bash
uv run python run_preprocessing.py \
    --subject sub-001 \
    --modality dwi \
    --config config.yaml
```

**Python API**:
```python
from neurovrai.preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject='sub-001',
    dwi_files=[Path('b1000.nii.gz'), Path('b2000.nii.gz')],
    bval_files=[Path('b1000.bval'), Path('b2000.bval')],
    bvec_files=[Path('b1000.bvec'), Path('b2000.bvec')],
    rev_phase_files=[Path('SE_EPI_PA.nii.gz')],  # Optional
    output_dir=Path('/data/derivatives'),
    run_advanced_models=True  # Enable DKI/NODDI
)
```

**Advanced Models**:
```python
from neurovrai.preprocess.workflows.amico_models import fit_noddi_amico

# AMICO-accelerated NODDI (100x faster than DIPY)
noddi_results = fit_noddi_amico(
    dwi_file=Path('dwi_eddy.nii.gz'),
    bval_file=Path('dwi.bval'),
    bvec_file=Path('dwi_rotated.bvec'),
    mask_file=Path('dwi_mask.nii.gz'),
    output_dir=Path('noddi_output')
)
# Runtime: ~30 seconds (vs 20-25 min with DIPY)
```

**Outputs**:
```
derivatives/sub-001/dwi/
├── eddy_corrected.nii.gz           # Preprocessed DWI
├── dwi_mask.nii.gz                 # Brain mask
├── dti/
│   ├── FA.nii.gz                   # Fractional anisotropy
│   ├── MD.nii.gz                   # Mean diffusivity
│   ├── AD.nii.gz                   # Axial diffusivity
│   └── RD.nii.gz                   # Radial diffusivity
├── dki/                            # DKI metrics (multi-shell only)
│   ├── MK.nii.gz                   # Mean kurtosis
│   ├── AK.nii.gz                   # Axial kurtosis
│   └── RK.nii.gz                   # Radial kurtosis
├── noddi/                          # NODDI metrics (multi-shell only)
│   ├── ficvf.nii.gz               # Neurite density
│   ├── odi.nii.gz                 # Orientation dispersion
│   └── fiso.nii.gz                # Isotropic fraction
└── qc/
    ├── motion_qc.html              # Motion QC
    └── dti_qc.html                 # DTI metrics QC
```

---

### Functional Preprocessing

Process resting-state or task-based fMRI data with multi-echo (TEDANA) or single-echo (ICA-AROMA) support.

**Pipeline**:
1. Motion correction (MCFLIRT)
2. Multi-echo: TEDANA denoising OR Single-echo: ICA-AROMA
3. ACompCor nuisance regression
4. Bandpass filtering (0.001-0.08 Hz default)
5. Spatial smoothing
6. Registration to anatomical space
7. Optional MNI normalization
8. Quality control reports

**CLI Usage**:
```bash
uv run python run_preprocessing.py \
    --subject sub-001 \
    --modality func \
    --config config.yaml
```

**Python API**:
```python
from neurovrai.preprocess.workflows.func_preprocess import run_functional_preprocessing

results = run_functional_preprocessing(
    config=config,
    subject='sub-001',
    func_files=[Path('echo1.nii.gz'), Path('echo2.nii.gz')],  # Multi-echo
    # OR
    # func_files=[Path('bold.nii.gz')],  # Single-echo
    output_dir=Path('/data/derivatives'),
    t1w_brain=Path('derivatives/sub-001/anat/brain.nii.gz'),
    normalize_to_mni=True
)
```

**Outputs**:
```
derivatives/sub-001/func/
├── preprocessed_bold.nii.gz        # Final preprocessed fMRI
├── brain_mask.nii.gz               # Functional brain mask
├── mean_bold.nii.gz                # Temporal mean
├── tsnr.nii.gz                     # Temporal SNR map
├── tedana/                         # Multi-echo outputs
│   └── desc-optcom_bold.nii.gz    # Optimal combination
└── qc/
    ├── motion_qc.html              # Motion parameters
    └── tsnr_qc.html                # tSNR QC
```

---

### ASL Preprocessing

Process arterial spin labeling (ASL) perfusion imaging.

**Pipeline**:
1. Motion correction (MCFLIRT)
2. Label-control separation
3. CBF quantification with kinetic modeling
4. M0 calibration with WM reference
5. Partial volume correction (PVC)
6. Registration to anatomical space
7. Optional MNI normalization
8. Quality control reports

**CLI Usage**:
```bash
uv run python run_preprocessing.py \
    --subject sub-001 \
    --modality asl \
    --config config.yaml
```

**Python API**:
```python
from neurovrai.preprocess.workflows.asl_preprocess import run_asl_preprocessing

results = run_asl_preprocessing(
    config=config,
    subject='sub-001',
    asl_file=Path('asl.nii.gz'),
    output_dir=Path('/data/derivatives'),
    t1w_brain=Path('derivatives/sub-001/anat/brain.nii.gz'),
    gm_mask=Path('derivatives/sub-001/anat/segmentation/pve_1.nii.gz'),
    wm_mask=Path('derivatives/sub-001/anat/segmentation/pve_2.nii.gz'),
    dicom_dir=Path('rawdata/sub-001/asl'),  # For auto parameter extraction
    normalize_to_mni=True
)
```

**Outputs**:
```
derivatives/sub-001/asl/
├── cbf.nii.gz                      # Cerebral blood flow map
├── cbf_pvc.nii.gz                  # PVC-corrected CBF
├── m0.nii.gz                       # M0 calibration image
├── cbf_stats.json                  # Tissue-specific CBF
└── qc/
    ├── motion_qc.html              # Motion QC
    └── cbf_qc.html                 # CBF distributions
```

---

## Group Analysis

### VBM (Voxel-Based Morphometry)

Analyze structural brain differences at the voxel level.

**Workflow**:
1. Prepare VBM data (normalize & smooth tissue maps)
2. Create design matrix with neuroaider
3. Run statistical analysis:
   - **FSL randomise**: Nonparametric permutation testing with TFCE correction (recommended for robustness)
   - **nilearn GLM**: Parametric second-level GLM with FDR/Bonferroni correction (faster, pure Python)
4. Generate cluster reports with anatomical localization

**CLI Usage**:
```bash
# Step 1: Prepare VBM data
uv run python -c "
from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data
from pathlib import Path

prepare_vbm_data(
    subjects=['sub-001', 'sub-002', 'sub-003'],
    derivatives_dir=Path('/data/derivatives'),
    output_dir=Path('/data/analysis/vbm'),
    tissue_type='GM',
    smoothing_fwhm=4.0
)
"

# Step 2: Run group analysis
# Option A: FSL randomise (nonparametric, TFCE correction)
uv run python run_vbm_group_analysis.py \
    --study-root /data \
    --method randomise \
    --tissue GM \
    --n-permutations 5000

# Option B: nilearn GLM (parametric, FDR/Bonferroni correction)
uv run python run_vbm_group_analysis.py \
    --study-root /data \
    --method glm \
    --tissue GM \
    --z-threshold 2.3

# Option C: Both methods for comparison
uv run python run_vbm_group_analysis.py \
    --study-root /data \
    --method both \
    --tissue GM
```

**Python API**:
```python
from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data, run_vbm_analysis
from pathlib import Path

# Prepare data
prepare_vbm_data(
    subjects=['sub-001', 'sub-002', 'sub-003'],
    derivatives_dir=Path('/data/derivatives'),
    output_dir=Path('/data/analysis/vbm'),
    tissue_type='GM',
    smoothing_fwhm=4.0
)

# Run analysis with FSL randomise
run_vbm_analysis(
    vbm_dir=Path('/data/analysis/vbm/GM'),
    participants_file=Path('/data/participants.csv'),
    formula='age + sex + group',
    contrasts={'age_positive': [0, 1, 0, 0]},
    method='randomise',
    n_permutations=5000
)

# Or with nilearn GLM (faster, parametric)
run_vbm_analysis(
    vbm_dir=Path('/data/analysis/vbm/GM'),
    participants_file=Path('/data/participants.csv'),
    formula='age + sex + group',
    contrasts={'age_positive': [0, 1, 0, 0]},
    method='glm',
    z_threshold=2.3
)
```

**Outputs**:
```
analysis/vbm/GM/
├── smoothed/
│   └── sub-*_GM_smooth.nii.gz      # Smoothed tissue maps
├── randomise_output/               # FSL randomise results (if method='randomise')
│   ├── randomise_tfce_corrp_tstat1.nii.gz  # Corrected p-values (TFCE)
│   └── randomise_tstat1.nii.gz    # T-statistics
├── glm_output/                     # nilearn GLM results (if method='glm')
│   ├── contrast_name_z_map.nii.gz # Z-statistic maps
│   ├── contrast_name_t_map.nii.gz # T-statistic maps
│   ├── contrast_name_effect.nii.gz # Effect size maps
│   ├── fdr_corrected/              # FDR correction results
│   └── bonferroni_corrected/       # Bonferroni correction results
└── cluster_reports/
    └── age_positive_report.html    # Cluster report with atlas
```

---

### TBSS (Tract-Based Spatial Statistics)

White matter analysis using diffusion tensor metrics.

**Workflow**:
1. Prepare TBSS data (FA images)
2. Run FSL TBSS pipeline (registration, skeletonization)
3. Create design matrix
4. Run randomise with TFCE correction

**Python API**:
```python
from neurovrai.analysis.tbss.prepare_tbss import prepare_tbss_data
from neurovrai.analysis.tbss.run_tbss_stats import run_tbss_statistics
from pathlib import Path

# Step 1: Prepare TBSS data
prepare_tbss_data(
    derivatives_dir=Path('/data/derivatives'),
    output_dir=Path('/data/analysis/tbss'),
    subjects=['sub-001', 'sub-002', 'sub-003']
)

# Step 2: Run statistics
run_tbss_statistics(
    tbss_dir=Path('/data/analysis/tbss'),
    participants_file=Path('/data/participants.csv'),
    formula='age + sex + group',
    contrasts=['age_positive', 'group_patients_vs_controls'],
    n_permutations=5000
)
```

**Outputs**:
```
analysis/tbss/
├── FA/                             # FA images
├── stats/
│   ├── all_FA_skeletonised.nii.gz  # Projected FA values
│   └── mean_FA_skeleton.nii.gz     # Mean skeleton
└── randomise_output/
    └── randomise_tfce_corrp_tstat1.nii.gz
```

---

### Resting-State fMRI Analysis

Compute regional homogeneity (ReHo) and fractional amplitude of low-frequency fluctuations (fALFF).

**Pipeline**:
1. Compute ReHo (Kendall's coefficient)
2. Compute fALFF (0.01-0.08 Hz power)
3. Normalize to MNI space
4. Run group statistics (FSL randomise or nilearn GLM)
5. Generate cluster reports

**Python API**:
```python
from neurovrai.analysis.func.resting_workflow import run_resting_state_analysis
from neurovrai.preprocess.utils.func_normalization import normalize_func_metrics
from neurovrai.analysis.func.run_func_group_analysis import run_func_group_analysis
from pathlib import Path

# Step 1: Compute metrics
results = run_resting_state_analysis(
    func_file=Path('/data/derivatives/sub-001/func/preprocessed.nii.gz'),
    mask_file=Path('/data/derivatives/sub-001/func/brain_mask.nii.gz'),
    output_dir=Path('/data/derivatives/sub-001/func'),
    compute_reho=True,
    compute_falff=True
)

# Step 2: Normalize to MNI
normalize_func_metrics(
    derivatives_dir=Path('/data/derivatives'),
    metric='reho',
    mni_template=Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz')
)

# Step 3: Group analysis
run_func_group_analysis(
    metric='reho',
    derivatives_dir=Path('/data/derivatives'),
    analysis_dir=Path('/data/analysis'),
    participants_file=Path('/data/participants.tsv'),
    study_name='my_study',
    n_permutations=5000
)
```

**Outputs**:
```
derivatives/sub-001/func/
├── reho.nii.gz                     # ReHo map
├── reho_z.nii.gz                   # Z-scored ReHo
├── falff.nii.gz                    # fALFF map
└── falff_z.nii.gz                  # Z-scored fALFF

analysis/func/reho/my_study/
├── randomise_output/
│   └── randomise_tfce_corrp_tstat1.nii.gz
└── cluster_reports/
    └── age_positive_report.html
```

---

### MELODIC (Group ICA)

Group-level independent component analysis.

**Python API**:
```python
from neurovrai.analysis.func.melodic import run_melodic_group_analysis
from pathlib import Path

run_melodic_group_analysis(
    derivatives_dir=Path('/data/derivatives'),
    output_dir=Path('/data/analysis/melodic'),
    subjects=['sub-001', 'sub-002', 'sub-003'],
    expected_tr=2.0,
    n_components=20,  # Or None for automatic estimation
    tr_tolerance=0.01
)
```

**Outputs**:
```
analysis/melodic/
├── melodic_IC.nii.gz               # Independent components
├── melodic_mix                     # Time courses
└── report.html                     # HTML report
```

---

## Connectome Analysis

### ROI Extraction

Extract regional data from atlas parcellations (modality-agnostic).

**Python API**:
```python
from neurovrai.connectome import extract_roi_timeseries, extract_roi_values
from pathlib import Path

# Extract functional timeseries
timeseries, roi_names = extract_roi_timeseries(
    data_file=Path('preprocessed_bold.nii.gz'),
    atlas=Path('schaefer_400.nii.gz'),
    mask_file=Path('brain_mask.nii.gz'),
    min_voxels=10
)

# Extract structural values (FA, MD, etc.)
roi_values, voxel_counts = extract_roi_values(
    data_file=Path('FA.nii.gz'),
    atlas=Path('JHU-ICBM-labels-2mm.nii.gz'),
    statistic='mean'
)
```

**CLI Usage**:
```bash
uv run python -m neurovrai.connectome.run_functional_connectivity \
    --func-file preprocessed_bold.nii.gz \
    --atlas schaefer_400.nii.gz \
    --output-dir fc_output/
```

---

### Functional Connectivity

Compute correlation matrices from fMRI timeseries using native-space analysis.

**Batch Processing (Recommended)**:
```bash
# Process all subjects with multiple atlases
uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root /mnt/bytopia/IRC805 \
    --atlases all \
    --output-dir /mnt/bytopia/IRC805/analysis/func/connectivity

# Process specific subjects/atlases
uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root /mnt/bytopia/IRC805 \
    --subjects sub-001 sub-002 \
    --atlases harvardoxford_cort juelich \
    --method pearson
```

**Available Atlases**:
- `harvardoxford_cort`: Harvard-Oxford Cortical (48 regions)
- `harvardoxford_sub`: Harvard-Oxford Subcortical (21 regions)
- `juelich`: Juelich Histological Atlas
- `talairach`: Talairach Atlas
- `cerebellum_mniflirt`: Cerebellum MNI FLIRT

**Python API**:
```python
from neurovrai.connectome import compute_functional_connectivity
from pathlib import Path

fc_results = compute_functional_connectivity(
    timeseries=timeseries,
    roi_names=roi_names,
    method='pearson',  # or 'spearman', 'partial'
    fisher_z=True,
    threshold=0.3,
    output_dir=Path('fc_output')
)

# Access results
fc_matrix = fc_results['fc_matrix']  # Fisher z-transformed if fisher_z=True
correlation = fc_results['correlation_matrix']  # Raw correlation
```

**Outputs**:
```
connectivity/
├── {subject}/
│   ├── {atlas}/
│   │   ├── fc_matrix.npy                   # Connectivity matrix (numpy)
│   │   ├── fc_matrix.csv                   # Connectivity matrix (CSV)
│   │   ├── fc_roi_names.txt                # ROI labels
│   │   ├── fc_summary.json                 # Analysis parameters
│   │   ├── atlas_{atlas}_resampled.nii.gz  # Atlas in functional space
│   │   └── analysis_metadata.json          # Processing metadata
├── logs/                                    # Processing logs
└── batch_processing_summary.json            # Batch summary
```

**Note**: This pipeline uses **native-space functional connectivity** - atlases are resampled to match each subject's functional data dimensions rather than normalizing functional data to MNI. This approach is simpler, faster, and preserves functional resolution. See `neurovrai/connectome/README.md` for details.

---

### Graph Theory Metrics

Compute network topology metrics.

**Python API**:
```python
from neurovrai.connectome import (
    compute_node_metrics,
    compute_global_metrics,
    identify_hubs
)

# Node-level metrics
node_metrics = compute_node_metrics(
    matrix=fc_matrix,
    threshold=0.3,
    weighted=True,
    roi_names=roi_names
)

print(f"Mean degree: {node_metrics['degree'].mean():.2f}")
print(f"Mean clustering: {node_metrics['clustering_coefficient'].mean():.3f}")

# Identify hub nodes
hubs = identify_hubs(
    node_metrics,
    method='betweenness',  # or 'degree'
    percentile=90
)

# Global network metrics
global_metrics = compute_global_metrics(
    matrix=fc_matrix,
    threshold=0.3
)

print(f"Global efficiency: {global_metrics['global_efficiency']:.4f}")
print(f"Path length: {global_metrics['characteristic_path_length']:.4f}")
print(f"Transitivity: {global_metrics['transitivity']:.4f}")
```

---

### Network-Based Statistic

Permutation-based network-level inference for group comparisons.

**Python API**:
```python
from neurovrai.connectome import compute_network_based_statistic
from pathlib import Path

nbs_results = compute_network_based_statistic(
    group1_matrices,  # Shape: (n_subjects, n_rois, n_rois)
    group2_matrices,
    threshold=3.0,  # t-statistic threshold
    n_permutations=5000,
    alpha=0.05,
    output_dir=Path('nbs_output')
)

print(f"Significant components: {nbs_results['n_significant']}")
for i, (size, pval) in enumerate(zip(
    nbs_results['component_sizes'],
    nbs_results['component_pvals']
)):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"Component {i+1}: {size} edges, p={pval:.4f} {sig}")
```

**Outputs**:
```
nbs_output/
├── nbs_t_matrix.npy                # T-statistics
├── nbs_null_distribution.npy       # Null distribution
└── nbs_components.json             # Component info
```

---

## Configuration

### Config File Format

Create a `config.yaml` file with study-specific parameters:

```yaml
# Study paths
project_dir: /mnt/data/my_study
rawdata_dir: ${project_dir}/rawdata
derivatives_dir: ${project_dir}/derivatives
work_dir: ${project_dir}/work

# Execution
execution:
  plugin: MultiProc
  n_procs: 8

# MNI templates
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
    readout_time: 0.05  # Check your protocol
  eddy_config:
    flm: linear
    slm: linear
    use_cuda: true  # Requires CUDA
  run_qc: true

# Functional preprocessing
functional:
  tr: 2.0
  te: [10.0, 30.0, 50.0]  # Multi-echo TEs
  highpass: 0.001  # Hz
  lowpass: 0.08    # Hz
  fwhm: 6          # mm smoothing
  tedana:
    enabled: false  # Set true for multi-echo
    tedpca: kundu
    tree: kundu
  aroma:
    enabled: false  # Auto-enabled for single-echo
  acompcor:
    enabled: true
    num_components: 5
  run_qc: true

# ASL preprocessing
asl:
  labeling_duration: 1.8  # seconds (auto-detected from DICOM if available)
  post_labeling_delay: 2.0  # seconds
  run_qc: true
```

### Load Configuration

```python
from neurovrai.preprocess.config import load_config
from pathlib import Path

config = load_config(Path('config.yaml'))

# Access values
project_dir = Path(config['project_dir'])
n_procs = config['execution']['n_procs']
```

---

## Examples

### Complete Workflows

**1. Full preprocessing workflow**:
```python
from pathlib import Path
from neurovrai.preprocess.config import load_config
from neurovrai.preprocess.workflows import (
    run_anatomical_preprocessing,
    run_dwi_multishell_topup_preprocessing,
    run_functional_preprocessing,
    run_asl_preprocessing
)

config = load_config(Path('config.yaml'))
subject = 'sub-001'
derivatives = Path(config['derivatives_dir'])

# Anatomical
anat_results = run_anatomical_preprocessing(
    config=config,
    subject=subject,
    t1w_file=Path(f'rawdata/{subject}/anat/T1w.nii.gz'),
    output_dir=derivatives
)

# Diffusion
dwi_results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject=subject,
    dwi_files=[Path(f'rawdata/{subject}/dwi/dwi.nii.gz')],
    bval_files=[Path(f'rawdata/{subject}/dwi/dwi.bval')],
    bvec_files=[Path(f'rawdata/{subject}/dwi/dwi.bvec')],
    rev_phase_files=[Path(f'rawdata/{subject}/dwi/dwi_PA.nii.gz')],
    output_dir=derivatives
)

# Functional
func_results = run_functional_preprocessing(
    config=config,
    subject=subject,
    func_files=[Path(f'rawdata/{subject}/func/bold.nii.gz')],
    output_dir=derivatives,
    t1w_brain=derivatives / subject / 'anat' / 'brain.nii.gz'
)
```

**2. Complete connectome analysis**:

See `examples/connectome_complete_workflow.py` for a full demonstration including:
- ROI extraction
- Functional connectivity
- Group analysis
- Graph theory metrics
- Network-Based Statistic
- Visualizations

```bash
uv run python examples/connectome_complete_workflow.py
```

---

## Performance

### GPU Acceleration

**Eddy correction** (CUDA required):
- CPU: ~45 minutes per subject
- GPU: ~4 minutes per subject (10x speedup)

**AMICO models**:
- NODDI: 30 seconds (vs 20-25 min DIPY, 100x speedup)
- SANDI: 3-6 minutes
- ActiveAx: 3-6 minutes

### Recommended Hardware

- **CPU**: 8+ cores for parallel processing
- **RAM**: 32+ GB for large datasets
- **GPU**: NVIDIA CUDA-compatible for eddy/BEDPOSTX
- **Storage**: Fast SSD for work directory

---

## Quality Control

All preprocessing workflows generate automated QC reports:

**Anatomical QC**:
- Skull stripping quality
- Tissue segmentation
- Registration to MNI

**Diffusion QC**:
- Motion parameters (eddy)
- TOPUP correction quality
- DTI metrics distributions
- Skull stripping quality

**Functional QC**:
- Motion parameters (MCFLIRT)
- Temporal SNR (tSNR)
- DVARS
- Skull stripping quality

**ASL QC**:
- Motion parameters
- CBF distributions
- Temporal SNR
- Skull stripping quality

QC reports are saved as HTML files in `derivatives/{subject}/{modality}/qc/`.

---

## Citation

If you use neurovrai in your research, please cite:

**Network-Based Statistic**:
- Zalesky A, Fornito A, Bullmore ET (2010). Network-based statistic: identifying differences in brain networks. NeuroImage, 53(4):1197-1207.

**Graph Theory**:
- Rubinov M, Sporns O (2010). Complex network measures of brain connectivity: uses and interpretations. NeuroImage, 52(3):1059-1069.

**TEDANA**:
- DuPre E, et al. (2021). TE-dependent analysis of multi-echo fMRI with tedana. JOSS, 6(66):3669.

**FSL**:
- Jenkinson M, et al. (2012). FSL. NeuroImage, 62(2):782-790.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: https://github.com/alexedmon1/neurovrai/issues
- **Documentation**: See module-specific READMEs:
  - `neurovrai/connectome/README.md`
  - `neurovrai/analysis/README.md` (coming soon)

---

## Acknowledgments

Built with:
- **FSL** - FMRIB Software Library
- **ANTs** - Advanced Normalization Tools
- **Nipype** - Neuroimaging workflow engine
- **TEDANA** - Multi-echo fMRI analysis
- **DIPY** - Diffusion imaging in Python
- **AMICO** - Accelerated microstructure imaging
- **NetworkX** - Graph theory analysis
