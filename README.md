# neurovrai

**Comprehensive MRI preprocessing, analysis, and connectivity package for neuroimaging research.**

From raw DICOM to group statistics and network neuroscience - a complete, production-ready pipeline for multi-modal MRI data analysis.

[![Version](https://img.shields.io/badge/version-2.0.0--alpha-blue.svg)](https://github.com/alexedmon1/neurovrai)
[![Python](https://img.shields.io/badge/python-3.13%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

**neurovrai** (French: "true neuro") is an integrated neuroimaging analysis package with three main modules:

```
neurovrai/
├── preprocess/    ✅ Production-Ready - Subject-level preprocessing
├── analysis/      🔄 Planned (Phase 3) - Group-level statistics
└── connectome/    🔄 Planned (Phase 4) - Connectivity & networks
```

### neurovrai.preprocess - **Production-Ready** ✅

Complete preprocessing workflows for all major MRI modalities:
- **Anatomical** (T1w/T2w): N4 bias correction, skull stripping, tissue segmentation, MNI registration
- **Diffusion** (DWI): TOPUP, eddy, DTI/DKI/NODDI, BEDPOSTX, spatial normalization
- **Functional** (rs-fMRI): TEDANA (multi-echo), ICA-AROMA (single-echo), ACompCor, bandpass filtering
- **ASL** (perfusion): M0 calibration, CBF quantification, partial volume correction

### neurovrai.analysis - In Development (Phase 3)

Group-level statistical analyses:
- **TBSS** (Tract-Based Spatial Statistics) - **✅ Data preparation implemented**
  - Subject discovery & validation
  - FSL TBSS pipeline integration (steps 1-4)
  - Skeleton projection & QC
  - Tested on real data (IRC805: 17 subjects)
- **Statistical infrastructure** - **✅ Design matrix generation implemented**
  - GLM design matrix creation with patsy formulas
  - FSL format output (.mat, .con files)
  - Contrast specification
- **Functional Connectivity Metrics** - **✅ Implemented**
  - ReHo (Regional Homogeneity) with Kendall's coefficient of concordance
  - ALFF/fALFF (Amplitude of Low-Frequency Fluctuations)
  - Z-score normalization for cross-subject comparison
  - Integrated workflow with automated QC logging
  - Tested on IRC805 data (450 timepoints, ~7 min ReHo + 22 sec fALFF)
- VBM (Voxel-Based Morphometry) - Planned
- MELODIC (Group ICA) - Planned
- Group CBF analysis - Planned

### neurovrai.connectome - Planned (Phase 4)

Connectivity and network neuroscience:
- Structural connectivity (probabilistic tractography)
- Functional connectivity matrices
- Graph theory metrics
- Network visualization
- Multi-modal integration (SC-FC coupling)

## Key Features

### Architecture
- **🎯 Three-Part Design**: Preprocessing → Analysis → Connectivity
- **📦 Single Package**: Integrated modules sharing configuration and data formats
- **⚙️ Config-Driven**: YAML configuration for all parameters
- **🔄 Transform Reuse**: Centralized spatial transformation management
- **📊 Comprehensive QC**: Automated quality control for all modalities

### Preprocessing (Production-Ready)
- **🚀 Multi-Modal**: Anat, DWI, functional, ASL in one pipeline
- **⚡ GPU Accelerated**: CUDA support for eddy, BEDPOSTX (10-50x speedup)
- **🧠 Advanced Models**: DKI, NODDI (DIPY + AMICO 100x acceleration)
- **🎭 Multi-Echo**: TEDANA 25.1.0 with automatic component classification
- **🔍 Quality Control**: Comprehensive automated QC with HTML reports
- **📁 BIDS-Compatible**: Follows neuroimaging data standards

### Performance
- **AMICO Acceleration**: NODDI in 30 seconds (vs 20-25 min DIPY)
- **GPU Processing**: 10-50x speedup for diffusion workflows
- **Parallel Execution**: Multi-modal processing for maximum throughput
- **Transform Reuse**: Zero redundant registration computation

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/alexedmon1/neurovrai.git
cd neurovrai

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Prerequisites

- **Python**: 3.13+ (developed with 3.13)
- **FSL**: 6.0+ (required for preprocessing)
- **ANTs**: 2.3+ (optional for advanced registration)
- **dcm2niix**: For DICOM conversion
- **CUDA**: Optional, for GPU acceleration

### Basic Usage

```bash
# 1. Create configuration
python create_config.py --study-root /path/to/study

# 2. Run single subject (all modalities)
uv run python run_simple_pipeline.py \
    --subject sub-001 \
    --nifti-dir /path/to/study/bids/sub-001 \
    --config /path/to/study/config.yaml

# 3. Or run specific modality
uv run python run_simple_pipeline.py \
    --subject sub-001 \
    --nifti-dir /path/to/study/bids/sub-001 \
    --config /path/to/study/config.yaml \
    --skip-dwi --skip-asl  # Only anatomical and functional

# 4. Batch processing
uv run python run_batch_simple.py --config /path/to/study/config.yaml
```

### Python API

```python
from neurovrai.config import load_config
from neurovrai.preprocess.workflows import (
    run_anat_preprocessing,
    run_dwi_multishell_topup_preprocessing,
    run_func_preprocessing,
    run_asl_preprocessing
)

# Load configuration
config = load_config('config.yaml')

# Run anatomical preprocessing
results = run_anat_preprocessing(
    config=config,
    subject='sub-001',
    t1w_file='/path/to/T1w.nii.gz',
    output_dir='/path/to/study/derivatives'
)

# Results contain all output file paths
print(results['brain'])          # Brain-extracted T1w
print(results['brain_mask'])     # Brain mask
print(results['mni_warp'])       # Warp to MNI space
```

## Directory Structure

neurovrai uses a standardized directory hierarchy:

```
study_root/
├── raw/
│   ├── dicom/              # Raw DICOM files
│   └── bids/               # Converted NIfTI (BIDS format)
│       └── sub-001/
│           ├── anat/
│           ├── dwi/
│           ├── func/
│           └── asl/
├── derivatives/            # Preprocessed outputs
│   └── sub-001/
│       ├── anat/           # Brain masks, segmentation, MNI registration
│       ├── dwi/            # Eddy-corrected, DTI/DKI/NODDI metrics
│       ├── func/           # Denoised BOLD, preprocessed time series
│       └── asl/            # CBF maps, tissue-specific perfusion
├── work/                   # Temporary processing files
│   └── sub-001/
├── qc/                     # Quality control reports
│   └── sub-001/
│       ├── anat/
│       ├── dwi/
│       ├── func/
│       └── asl/
├── transforms/             # Spatial transformation registry
│   └── sub-001/
└── config.yaml             # Study configuration
```

## Preprocessing Workflows

### Anatomical (T1w/T2w)

**Pipeline:**
1. N4 bias field correction (ANTs)
2. Brain extraction (FSL BET)
3. Tissue segmentation (ANTs Atropos - faster than FSL FAST)
4. Registration to MNI152 (FSL FLIRT + FNIRT)
5. Quality control (skull stripping, segmentation, registration)

**Outputs:**
- Brain-extracted images
- Brain masks
- Tissue probability maps (CSF, GM, WM)
- MNI-space registered images
- Spatial transformations

**Time:** 15-30 minutes

### Diffusion (DWI)

**Pipeline:**
1. Optional TOPUP distortion correction (auto-enabled with reverse PE data)
2. GPU-accelerated eddy current/motion correction
3. DTI fitting (FA, MD, AD, RD)
4. Optional BEDPOSTX fiber orientation estimation (for future tractography)
5. Advanced models (auto-enabled for multi-shell):
   - **DKI** (DIPY): MK, AK, RK, KFA metrics
   - **NODDI** (DIPY or AMICO): FICVF, ODI, FISO
   - **AMICO Models**: SANDI, ActiveAx (100x faster)
6. Spatial normalization to FMRIB58_FA template
7. Comprehensive QC (TOPUP, motion, DTI metrics)

**Outputs:**
- Eddy-corrected DWI
- DTI metric maps
- DKI/NODDI metric maps (multi-shell only)
- BEDPOSTX fiber orientations (optional)
- Normalized metrics in MNI space
- Forward/inverse warps

**Time:** 45-90 minutes (30 min with AMICO)

### Functional (rs-fMRI)

**Pipeline:**
1. **Multi-echo path:**
   - Auto-detection of echo count
   - TEDANA denoising (optimal for multi-echo)
   - Motion correction per echo
   - Optimally combined signal
2. **Single-echo path:**
   - Motion correction (MCFLIRT)
   - ICA-AROMA artifact removal (auto-enabled)
3. **Common steps:**
   - ACompCor nuisance regression (CSF/WM components)
   - Bandpass temporal filtering
   - Spatial smoothing
   - Registration to anatomical/MNI space
4. Comprehensive QC (motion, DVARS, tSNR, carpet plots)

**Outputs:**
- Preprocessed BOLD time series
- Motion parameters
- Nuisance regressors
- tSNR maps
- QC reports (HTML)

**Time:** 20-40 min (single-echo), 2-4 hours (multi-echo with TEDANA)

### ASL (Perfusion)

**Pipeline:**
1. Automated DICOM parameter extraction (labeling duration τ, PLD)
2. Motion correction
3. Label-control separation
4. M0 calibration with white matter reference
5. CBF quantification (standard kinetic model, Alsop et al. 2015)
6. Partial volume correction (tissue-specific CBF)
7. Registration to anatomical space
8. Comprehensive QC (motion, CBF, tSNR)

**Outputs:**
- CBF maps
- M0 maps
- Tissue-specific CBF statistics
- QC metrics and plots

**Time:** 15-30 minutes

## Quality Control

neurovrai includes comprehensive automated QC for all modalities:

### Anatomical QC
- **Skull Stripping**: Brain mask overlays, volume statistics, over/under-stripping detection
- **Segmentation**: Tissue volume distributions, probability maps, GM/WM/CSF ratio validation
- **Registration**: MNI overlay visualizations, checkerboard comparisons, spatial correlation metrics

**Outputs:** PNG visualizations, JSON metrics, pass/fail flags

### Diffusion QC
- **TOPUP**: Field map visualizations, convergence plots, distortion correction metrics
- **Motion**: Framewise displacement plots, outlier detection, motion parameter time series
- **DTI**: FA/MD/AD/RD histograms, metric distributions, white matter statistics
- **Advanced Models**: DKI/NODDI metric distributions, fitting quality metrics

**Outputs:** Comprehensive plots, distribution statistics, outlier identification

### Functional QC
- **Motion**: Translation/rotation parameters, framewise displacement, outlier volumes
- **Signal Quality**: DVARS time series, temporal SNR maps, signal variance
- **Denoising**: TEDANA component classification, variance explained, acceptance rates
- **Confounds**: ACompCor components, nuisance regressor validation
- **Visualization**: Carpet plots, motion correlation, tSNR overlays

**Outputs:** HTML reports, interactive plots, comprehensive metrics

### ASL QC
- **Motion**: CBF sensitivity to motion, temporal stability
- **Perfusion**: CBF distributions, tissue-specific values, physiological range validation
- **Signal**: tSNR maps, M0 calibration quality, label-control SNR

**Outputs:** CBF overlays, distribution plots, tissue-specific statistics

### QC Directory Structure

```
qc/sub-001/
├── anat/
│   ├── skull_strip/
│   │   ├── brain_mask_overlay.png
│   │   └── metrics.json
│   ├── segmentation/
│   │   ├── tissue_volumes.png
│   │   ├── tissue_overlays.png
│   │   └── metrics.json
│   ├── registration/
│   │   ├── registration_overlay.png
│   │   ├── registration_checkerboard.png
│   │   └── metrics.json
│   └── combined_qc_results.json
├── dwi/
│   ├── topup/
│   │   ├── field_map.png
│   │   ├── convergence.png
│   │   └── metrics.json
│   ├── motion/
│   │   ├── framewise_displacement.png
│   │   ├── motion_params.png
│   │   └── metrics.json
│   ├── dti/
│   │   ├── fa_histogram.png
│   │   ├── md_histogram.png
│   │   └── metrics.json
│   └── combined_qc_results.json
├── func/
│   ├── motion_qc.html
│   ├── tsnr_map.png
│   ├── carpet_plot.png
│   ├── dvars.png
│   └── metrics.json
└── asl/
    ├── cbf_overlay.png
    ├── cbf_distribution.png
    ├── tsnr_map.png
    └── metrics.json
```

All QC outputs include:
- **Visualizations**: PNG/HTML for quick review
- **Metrics**: JSON files for quantitative analysis
- **Pass/Fail Flags**: Automated quality assessment

## Configuration

neurovrai uses a single YAML configuration file for all modules:

```yaml
# Project paths
project_dir: /path/to/study
rawdata_dir: ${project_dir}/raw/bids
derivatives_dir: ${project_dir}/derivatives
work_dir: ${project_dir}/work
qc_dir: ${project_dir}/qc
transforms_dir: ${project_dir}/transforms

# Execution
execution:
  plugin: MultiProc
  n_procs: 6

# Templates
templates:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
  fmrib58_fa: /usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz

# ============================================================
# PREPROCESSING (neurovrai.preprocess)
# ============================================================

# Anatomical
anatomical:
  bet:
    frac: 0.5
    reduce_bias: true
    robust: true
  segmentation:
    n_iterations: 5
    mrf_weight: 0.1
  registration_method: fsl  # or 'ants'
  run_qc: true

# Diffusion
diffusion:
  denoise_method: dwidenoise
  topup:
    readout_time: 0.05      # Adjust for your acquisition
  eddy:
    use_cuda: true          # GPU acceleration
  bedpostx:
    enabled: true           # Fiber orientation for future tractography
    use_gpu: true
  advanced_models:
    fit_dki: true           # Auto-disabled for single-shell
    fit_noddi: true
    use_amico: true         # 100x faster NODDI
  normalize_to_mni: true
  run_qc: true

# Functional
functional:
  tr: 1.029                 # Repetition time (seconds)
  te: [10.0, 30.0, 50.0]   # Echo times (ms) - for multi-echo
  highpass: 0.001           # Hz
  lowpass: 0.08
  fwhm: 6                   # Smoothing (mm)
  tedana:
    enabled: true           # Auto for multi-echo
    tedpca: 0.95           # Variance threshold
    tree: kundu
  aroma:
    enabled: auto           # Auto-enabled for single-echo
  acompcor:
    enabled: true
    num_components: 5
    variance_threshold: 0.5
  normalize_to_mni: true
  run_qc: true

# ASL
asl:
  labeling_duration: 1.8    # τ (seconds) - auto-extracted from DICOM
  post_labeling_delay: 2.0  # PLD (seconds)
  lambda_blood: 0.9
  t1_blood: 1.65
  alpha: 0.85
  wm_cbf_reference: 25.0
  apply_pvc: true           # Partial volume correction
  normalize_to_mni: true
  run_qc: true

# FreeSurfer (EXPERIMENTAL - not production ready)
freesurfer:
  enabled: false            # Do not enable until transform pipeline complete
  subjects_dir: ${project_dir}/freesurfer

# ============================================================
# ANALYSIS (neurovrai.analysis) - Placeholder for Phase 3
# ============================================================
# Configuration sections will be added in Phase 3

# ============================================================
# CONNECTOME (neurovrai.connectome) - Placeholder for Phase 4
# ============================================================
# Configuration sections will be added in Phase 4
```

## Project Status

### ✅ Production-Ready (neurovrai.preprocess)

All preprocessing modalities are complete and validated:

| Modality | Status | Key Features |
|----------|--------|--------------|
| **Anatomical** | ✅ Complete | N4, BET, Atropos, FNIRT, comprehensive QC |
| **Diffusion** | ✅ Complete | TOPUP, eddy_cuda, DTI/DKI/NODDI, BEDPOSTX, MNI normalization |
| **Functional** | ✅ Complete | TEDANA (multi-echo), ICA-AROMA (single-echo), ACompCor, MNI normalization |
| **ASL** | ✅ Complete | M0 calibration, PVC, CBF quantification, auto DICOM params |
| **QC Framework** | ✅ Complete | Automated QC for all modalities with HTML reports |

**Recent Milestones:**
- 2025-11-17: Fixed functional run selection for scanner retries
- 2025-11-17: Enabled ACompCor in functional pipeline
- 2025-11-17: Package restructured to neurovrai with three-module architecture
- 2025-11-16: Removed tractography from preprocessing (will be reimplemented in neurovrai.connectome)
- 2025-11-15: All modalities production-ready
- 2025-11-14: TEDANA 25.1.0, spatial normalization, bug fixes
- 2025-11-13: ASL preprocessing with M0 calibration and PVC
- 2025-11-12: DKI/NODDI validation, functional QC enhancements
- 2025-11-11: AMICO integration (100x speedup)
- 2025-11-10: Multi-echo TEDANA integration

### 🔄 In Development

| Module | Status | Timeline |
|--------|--------|----------|
| **neurovrai.analysis** | Planned | Phase 3 (4-6 weeks) |
| **neurovrai.connectome** | Planned | Phase 4 (6-8 weeks) |

### ⚠️ Experimental (Not Production Ready)

| Feature | Status | Issue |
|---------|--------|-------|
| **FreeSurfer Integration** | Hooks only | Transform pipeline incomplete |

**See `docs/NEUROVRAI_ARCHITECTURE.md` for detailed roadmap and implementation plan.**

## Processing Time Estimates

Typical times on modern workstation with GPU:

| Modality | Time | Configuration |
|----------|------|---------------|
| Anatomical | 15-30 min | N4, BET, Atropos, FNIRT |
| DWI (basic) | 30-60 min | TOPUP, eddy_cuda, DTI |
| DWI (full) | 45-90 min | + DKI/NODDI (DIPY) |
| DWI (AMICO) | 30-45 min | NODDI in 30 sec (not 25 min) |
| Functional (single) | 20-40 min | Motion, ICA-AROMA, ACompCor |
| Functional (multi) | 2-4 hours | + TEDANA (1-2 hours) |
| ASL | 15-30 min | Motion, CBF, M0 calibration |

**Optimization tips:**
- Enable GPU acceleration for eddy and BEDPOSTX
- Use AMICO for NODDI (100x speedup)
- Run modalities in parallel after anatomical completes
- Use `--parallel-modalities` flag for maximum throughput

## Advanced Usage

### Transform Registry

Efficient spatial transformation reuse across workflows:

```python
from neurovrai.utils.transforms import create_transform_registry

# Create registry
registry = create_transform_registry(config, subject='sub-001')

# Anatomical workflow saves transforms
registry.save_nonlinear_transform(
    warp_file='T1w_to_MNI_warp.nii.gz',
    affine_file='T1w_to_MNI.mat',
    source_space='T1w',
    target_space='MNI152',
    subject='sub-001'
)

# DWI workflow retrieves transforms (zero redundant computation)
warp, affine = registry.get_nonlinear_transform('T1w', 'MNI152')
```

### Batch Processing

```bash
# Sequential processing
for subject in sub-001 sub-002 sub-003; do
  uv run python run_simple_pipeline.py \
    --subject ${subject} \
    --nifti-dir /study/bids/${subject} \
    --config config.yaml
done

# Parallel processing with GNU Parallel
cat subjects.txt | parallel -j 4 \
  uv run python run_simple_pipeline.py \
    --subject {} \
    --nifti-dir /study/bids/{} \
    --config config.yaml
```

### Custom Workflows

```python
from nipype import Workflow, Node
from nipype.interfaces import fsl
from neurovrai.config import load_config
from neurovrai.utils.workflow import setup_logging, get_execution_config

# Load config
config = load_config('config.yaml')

# Create custom workflow
wf = Workflow(name='custom_analysis')
wf.base_dir = config['work_dir']

# Add processing nodes
bet = Node(fsl.BET(frac=0.5, mask=True), name='brain_extraction')
# ... add more nodes

# Execute
wf.run(**get_execution_config(config))
```

## Troubleshooting

### Import Errors

```bash
# Ensure neurovrai is installed
uv run python -c "import neurovrai; print(neurovrai.__version__)"
# Should output: 2.0.0-alpha

# If import fails, reinstall
uv sync
```

### FSL Not Found

```bash
# Check FSL installation
echo $FSLDIR
# Should output: /usr/local/fsl

# Source FSL configuration
source ${FSLDIR}/etc/fslconf/fsl.sh
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
nvidia-smi

# Verify GPU support in FSL
eddy_cuda --help
```

### Memory Issues

Reduce parallel processes in `config.yaml`:

```yaml
execution:
  n_procs: 2  # Reduce from 6 to 2
```

### TEDANA Convergence Issues

If TEDANA ICA fails to converge, adjust PCA threshold:

```yaml
functional:
  tedana:
    tedpca: 225  # Use fixed component count instead of variance threshold
```

## Documentation

- **`README.md`** (this file): Overview and quick start
- **`docs/NEUROVRAI_ARCHITECTURE.md`**: Three-part architecture and roadmap
- **`docs/workflows.md`**: Detailed workflow documentation
- **`docs/DWI_PROCESSING_GUIDE.md`**: DWI-specific guide
- **`PROJECT_STATUS.md`**: Detailed implementation status
- **`CLAUDE.md`**: Development guidelines (for AI assistants)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use neurovrai in your research, please cite:

```bibtex
@software{neurovrai,
  title={neurovrai: Comprehensive MRI Preprocessing and Analysis Package},
  author={Edmond, Alexandre},
  year={2025},
  version={2.0.0-alpha},
  url={https://github.com/alexedmon1/neurovrai}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [Nipype](https://nipype.readthedocs.io/) workflow engine
- Uses [FSL](https://fsl.fmrib.ox.ac.uk/) for neuroimaging processing
- Uses [ANTs](http://stnava.github.io/ANTs/) for advanced registration
- [DIPY](https://dipy.org/) for advanced diffusion models
- [AMICO](https://github.com/daducci/AMICO) for accelerated microstructure modeling
- [TEDANA](https://tedana.readthedocs.io/) for multi-echo fMRI denoising
- Inspired by [fMRIPrep](https://fmriprep.org/) and [QSIPrep](https://qsiprep.readthedocs.io/)

## Support

- **GitHub Issues**: https://github.com/alexedmon1/neurovrai/issues
- **Documentation**: https://github.com/alexedmon1/neurovrai

---

**neurovrai** - *True neuroimaging for the modern age* 🧠
