# Human MRI Preprocessing Pipeline

A production-ready, config-driven MRI preprocessing pipeline for multiple neuroimaging modalities: anatomical (T1w), diffusion-weighted imaging (DWI), resting-state fMRI, and arterial spin labeling (ASL). Built with Nipype for workflow orchestration, supporting both FSL and ANTs for neuroimaging processing.

## Features

### Core Features
- **Complete End-to-End Pipeline**: From DICOM to analysis-ready outputs
- **DICOM to NIfTI Conversion**: Automatic modality detection and parameter extraction from scanner files
- **Continuous Pipeline Architecture**: Streaming workflow execution - preprocessing starts as soon as data is converted
- **Config-Driven Architecture**: YAML-based configuration for all processing parameters with validation
- **BIDS-Compatible**: Follows Brain Imaging Data Structure conventions
- **Transform Registry**: Centralized management of spatial transformations for efficient reuse
- **Modular Workflows**: Separate pipelines for anatomical, diffusion, functional, and ASL preprocessing
- **Standardized Output**: Consistent directory hierarchy across all workflows
- **Dual Execution Modes**: Batch processing or continuous streaming pipeline
- **Production-Ready**: Tested and validated with real-world data

### Anatomical Preprocessing
- N4 bias field correction with ANTs
- Brain extraction with FSL BET
- Tissue segmentation with ANTs Atropos (faster than FSL FAST)
- Registration to MNI152 with FSL FNIRT
- Comprehensive quality control

### Diffusion Preprocessing
- **Optional TOPUP distortion correction** - Auto-enabled when reverse phase-encoding images available
- GPU-accelerated eddy current correction
- DTI fitting with standard metrics (FA, MD, AD, RD)
- Advanced models: DKI and NODDI (auto-enabled for multi-shell data)
- Spatial normalization to FMRIB58_FA template
- Probabilistic tractography with atlas-based ROI extraction
- **FreeSurfer integration hooks** (experimental - transform pipeline not yet implemented)
- Comprehensive QC (TOPUP, motion, DTI metrics)

### Functional Preprocessing
- Multi-echo TEDANA denoising
- Motion correction with MCFLIRT
- ICA-AROMA (optional)
- CompCor nuisance regression
- Bandpass temporal filtering
- Registration to anatomical space

### ASL (Arterial Spin Labeling) Preprocessing
- **Automated DICOM Parameter Extraction**: Auto-extracts acquisition parameters (τ, PLD) from scanner DICOM files
- **M0 Calibration**: White matter reference calibration to correct for M0 estimation bias
- **Partial Volume Correction**: Linear regression method for improved tissue-specific CBF accuracy
- Motion correction with MCFLIRT
- Label-control separation and quantification
- CBF quantification with standard kinetic model (Alsop et al., 2015)
- Tissue-specific CBF statistics
- Registration to anatomical space
- Comprehensive quality control with motion, CBF, and tSNR metrics

### Advanced Features
- **GPU Acceleration**: CUDA support for FSL eddy and BEDPOSTX
- **Flexible Registration**: Support for both FSL (FLIRT/FNIRT) and ANTs
- **Quality Control Framework**: Automated QC for all modalities with detailed reports

## Prerequisites

### System Requirements

- Python 3.10+ (developed with Python 3.13)
- FSL 6.0+ (for anatomical and diffusion preprocessing)
- ANTs 2.3+ (optional, for advanced registration)
- dcm2niix (for DICOM to NIfTI conversion)

### FSL Installation

```bash

# Install FSL from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
# Set FSLDIR environment variable
export FSLDIR=/usr/local/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh
```

### ANTs Installation (Optional)

```bash

# Install ANTs from https://github.com/ANTsX/ANTs
# Add to PATH
export ANTSPATH=/usr/local/bin/
export PATH=${ANTSPATH}:$PATH
```

## Installation

### Using uv (Recommended)

```bash

# Clone the repository
git clone https://github.com/yourusername/human-mri-preprocess.git
cd human-mri-preprocess

# Install dependencies with uv
uv sync

# Activate the environment
source .venv/bin/activate
```

### Using pip

```bash

# Clone the repository
git clone https://github.com/yourusername/human-mri-preprocess.git
cd human-mri-preprocess

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## How to Run the Pipeline

The pipeline provides multiple ways to process MRI data, from simple single-modality runs to complete multi-modal preprocessing.

### Prerequisites

Before running any preprocessing:

1. **Install dependencies** (see Installation section above)
2. **Activate environment**: `source .venv/bin/activate` (if not using `uv run`)
3. **Create config file**: `config.yaml` with your study paths (see Configuration section)
4. **Organize data**: DICOM files or BIDS-organized NIfTI files

### Running Preprocessing for a Subject

#### Option 1: Individual Modality Preprocessing (Recommended)

Run preprocessing one modality at a time for maximum control:

```bash
# IMPORTANT: Use uv run for dependency management
uv run python run_preprocessing.py --subject IRC805-0580101 --modality anat
uv run python run_preprocessing.py --subject IRC805-0580101 --modality dwi
uv run python run_preprocessing.py --subject IRC805-0580101 --modality func
uv run python run_preprocessing.py --subject IRC805-0580101 --modality asl
```

**Workflow order**:
1. **Anatomical first**: Required for functional/ASL registration
2. **DWI**: Independent, can run in parallel with anatomical
3. **Functional**: Requires anatomical outputs for registration
4. **ASL**: Requires anatomical outputs for registration

**Example: Complete subject preprocessing**

```bash
# Step 1: Run anatomical (foundation for other modalities)
uv run python run_preprocessing.py --subject sub-001 --modality anat

# Step 2: Run DWI (can run while functional processes)
uv run python run_preprocessing.py --subject sub-001 --modality dwi &

# Step 3: Run functional (needs anatomical complete)
uv run python run_preprocessing.py --subject sub-001 --modality func &

# Step 4: Run ASL (needs anatomical complete)
uv run python run_preprocessing.py --subject sub-001 --modality asl &

# Wait for all background jobs
wait
```

#### Option 2: All Modalities at Once

Process all available modalities for a subject:

```bash
# Run all modalities (anatomical → others in parallel)
uv run python run_preprocessing.py --subject sub-001 --modality all
```

**What happens**:
1. Anatomical preprocessing runs first
2. After anatomical completes, DWI/functional/ASL run in parallel
3. Optimal resource utilization for multi-modal data

#### Option 3: Full Pipeline from DICOM

Convert DICOM and preprocess in one command:

```bash
# Complete pipeline from raw DICOM
uv run python run_full_pipeline.py \
    --subject sub-001 \
    --dicom-dir /path/to/dicom/sub-001 \
    --config config.yaml
```

**Process**:
1. Converts all DICOM files to NIfTI (with metadata extraction)
2. Detects available modalities automatically
3. Runs preprocessing workflows in optimal order
4. Generates QC reports

#### Option 4: Continuous Streaming Pipeline (Advanced)

For large datasets, start preprocessing as DICOM conversion completes:

```bash
# Streaming pipeline - preprocesses as data converts
uv run python run_continuous_pipeline.py \
    --subject sub-001 \
    --dicom-dir /path/to/dicom/sub-001 \
    --config config.yaml
```

**Benefits**:
- Maximum efficiency - no waiting for full conversion
- Optimal for large studies with many subjects
- Anatomical starts as soon as T1w is converted
- Other modalities start as their files become available

### Command-Line Options

All runner scripts support these options:

```bash
--subject SUBJECT_ID       # Required: Subject identifier
--modality {anat,dwi,func,asl,all}  # Preprocessing modality
--config CONFIG_FILE       # Optional: config.yaml path (default: ./config.yaml)
--study-root STUDY_ROOT    # Optional: Override project_dir from config
--skip-qc                  # Optional: Skip quality control generation
```

### Understanding Data Flow

```
Raw Data → DICOM Conversion → Preprocessing → Quality Control → Analysis-Ready Data
```

**Directory structure during processing**:

```
/mnt/bytopia/IRC805/                    # Study root
├── dicoms/IRC805-0580101/              # Raw DICOM files
├── bids/IRC805-0580101/                # Converted NIfTI + JSON
│   ├── anat/
│   ├── dwi/
│   ├── func/
│   └── asl/
├── derivatives/IRC805-0580101/         # Preprocessed outputs
│   ├── anat/                           # Brain masks, segmentations, MNI-registered T1w
│   ├── dwi/                            # Eddy-corrected DWI, DTI/DKI/NODDI metrics
│   ├── func/                           # Denoised BOLD, preprocessed time series
│   └── asl/                            # CBF maps, tissue-specific perfusion
├── work/IRC805-0580101/                # Temporary Nipype files (can delete after)
├── qc/IRC805-0580101/                  # Quality control reports
│   ├── anat/
│   ├── dwi/
│   ├── func/
│   └── asl/
└── logs/                               # Processing logs
```

### DICOM to NIfTI Conversion

If starting from DICOM files, the pipeline automatically:

- **Detects modalities** from SeriesDescription DICOM tags
- **Extracts parameters**: TR, TE, bvals, bvecs, ASL timing
- **Organizes files**: BIDS-like structure with JSON sidecars
- **Validates completeness**: Ensures all required files present

**Supported modalities**:
- **Anatomical**: T1w, T2w (3D structural scans)
- **Diffusion**: Multi-shell DWI with optional reverse phase-encoding for TOPUP
- **Functional**: Single-echo or multi-echo resting-state fMRI
- **ASL**: pCASL with automatic parameter extraction from DICOM headers

### Processing Time Estimates

Typical processing times on a modern workstation (GPU-enabled):

| Modality | Time Range | Notes |
|----------|------------|-------|
| Anatomical | 15-30 min | N4 bias correction, segmentation, registration |
| DWI | 30-90 min | Depends on shells, TOPUP enabled, advanced models |
| Functional (single-echo) | 20-40 min | Motion correction, smoothing, registration |
| Functional (multi-echo) | 2-4 hours | Includes TEDANA denoising (~1-2 hours) |
| ASL | 15-30 min | Motion correction, CBF quantification |

**Parallelization**: Run DWI, functional, and ASL simultaneously after anatomical completes to maximize throughput.

## Quick Start

### 1. Prepare Your Data

Organize your data in BIDS format:

```
project/
 rawdata/
    sub-001/
        anat/
           sub-001_T1w.nii.gz
           sub-001_T1w.json
        dwi/
            sub-001_dwi.nii.gz
            sub-001_dwi.bval
            sub-001_dwi.bvec
            sub-001_dwi.json
 derivatives/  (created by pipeline)
 work/         (temporary working directory)
```

### 2. Create Configuration File

Create a YAML configuration file (e.g., `config.yaml`):

```yaml

# Project paths
project_dir: /path/to/your/project
rawdata_dir: ${project_dir}/rawdata
derivatives_dir: ${project_dir}/derivatives
work_dir: ${project_dir}/work
transforms_dir: ${project_dir}/transforms

# Execution settings
execution:
  plugin: MultiProc
  n_procs: 4

# Template files
templates:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
  mni152_t1_1mm: /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz

# Anatomical preprocessing
anatomical:
  bet_frac: 0.5
  registration_method: fsl  # or 'ants'

# Diffusion preprocessing
diffusion:
  denoise_method: dwidenoise
  topup:
    enabled: auto  # 'auto', true, or false - auto-detects reverse PE availability
    readout_time: 0.05
  eddy_config:
    flm: linear
    slm: linear
    use_cuda: true
  advanced_models:
    enabled: auto  # Auto-enables for multi-shell data
    fit_dki: true
    fit_noddi: true

# FreeSurfer integration (EXPERIMENTAL - not production ready)
freesurfer:
  enabled: false  # Do not enable until transform pipeline implemented
  subjects_dir: ${project_dir}/freesurfer
```

### 3. Run Preprocessing

```bash
# Run anatomical preprocessing (required first)
uv run python run_preprocessing.py --subject sub-001 --modality anat

# Run additional modalities (see "How to Run the Pipeline" section for details)
uv run python run_preprocessing.py --subject sub-001 --modality dwi
uv run python run_preprocessing.py --subject sub-001 --modality func
```

### 4. Check Outputs

```bash
# View preprocessed outputs
ls derivatives/sub-001/anat/    # Brain masks, segmentations, registrations
ls derivatives/sub-001/dwi/     # Eddy-corrected DWI, DTI/DKI/NODDI metrics
ls derivatives/sub-001/func/    # Denoised BOLD, preprocessed time series

# View quality control reports
ls qc/sub-001/anat/            # Anatomical QC plots
ls qc/sub-001/dwi/             # DWI QC plots
ls qc/sub-001/func/            # Functional QC plots
```

**For detailed usage instructions, processing options, and workflow details, see the "How to Run the Pipeline" section above.**

## Pipeline Workflows

### Anatomical Preprocessing

**Current Validated Workflow:**
1. **Reorientation**: Standardize image orientation (fslreorient2std)
2. **Skull Stripping**: Brain extraction (FSL BET)
3. **Registration**: Linear (FLIRT) and nonlinear (FNIRT) to MNI152
4. **Transform Storage**: Save transforms to TransformRegistry for reuse

**Optional/Future Enhancements:**
- **Bias Correction**: ANTs Light N4 (validated, 2.5 min on high-res data)
- **Tissue Segmentation**: CSF/GM/WM segmentation (ANTs Atropos or FSL FAST - to be configured for resting-state fMRI workflows)

**Outputs**:
- `sub-XXX_T1w_brain.nii.gz`: Skull-stripped T1w
- `sub-XXX_T1w_brain_mask.nii.gz`: Brain mask
- `sub-XXX_T1w_to_MNI152.mat`: Affine transform
- `sub-XXX_T1w_to_MNI152_warp.nii.gz`: Nonlinear warp field
- Transforms saved to TransformRegistry for cross-workflow reuse

### Diffusion Preprocessing

1. **DICOM Conversion**: Convert DICOM to NIfTI (if needed)
2. **Denoising**: Marchenko-Pastur PCA denoising (dwidenoise)
3. **Gibbs Unringing**: Remove Gibbs ringing artifacts
4. **Eddy Current Correction**: FSL eddy with motion correction
5. **Registration**: Register to anatomical space (reuses T1w→MNI transforms)
6. **Tensor Fitting**: DTI model fitting

**Outputs**:
- `sub-XXX_dwi_preprocessed.nii.gz`: Preprocessed DWI
- `sub-XXX_dwi_FA.nii.gz`: Fractional anisotropy map
- `sub-XXX_dwi_MD.nii.gz`: Mean diffusivity map
- `sub-XXX_dwi_to_MNI152.nii.gz`: DWI warped to MNI152

## Quality Control (QC)

The pipeline includes comprehensive quality control modules for both DWI and anatomical preprocessing.

### DWI QC

Automated QC for diffusion preprocessing:

```python
from mri_preprocess.qc.dwi import TOPUPQualityControl, MotionQualityControl, DTIQualityControl

# TOPUP QC: Field map analysis
topup_qc = TOPUPQualityControl(
    subject='sub-001',
    work_dir=Path('derivatives/dwi_topup/sub-001'),
    qc_dir=Path('qc/dwi/sub-001/topup')
)
results = topup_qc.run_qc()

# Motion QC: Framewise displacement
motion_qc = MotionQualityControl(
    subject='sub-001',
    work_dir=Path('derivatives/dwi_topup/sub-001'),
    qc_dir=Path('qc/dwi/sub-001/motion')
)
results = motion_qc.run_qc()

# DTI QC: FA/MD distributions
dti_qc = DTIQualityControl(
    subject='sub-001',
    dti_dir=Path('derivatives/dwi_topup/sub-001/dti'),
    qc_dir=Path('qc/dwi/sub-001/dti')
)
results = dti_qc.run_qc(metrics=['FA', 'MD'])
```

**QC Outputs** (stored in `{study_root}/qc/dwi/{subject}/`):
- TOPUP convergence plots and field map statistics
- Motion parameter plots with outlier detection
- FA/MD histograms and distribution statistics
- JSON metrics files for all QC measures

### Anatomical QC

```python
from mri_preprocess.qc.anat import SkullStripQualityControl

# Skull stripping QC
skull_qc = SkullStripQualityControl(
    subject='sub-001',
    anat_dir=Path('derivatives/anat_preproc/sub-001/anat'),
    qc_dir=Path('qc/anat/sub-001/skull_strip')
)
results = skull_qc.run_qc()
```

**QC Outputs** (stored in `{study_root}/qc/anat/{subject}/`):
- Brain mask overlay visualizations
- Brain volume statistics
- Quality assessment metrics (contrast ratio, over/under-stripping detection)

For complete QC documentation, see `docs/DWI_QC_SPECIFICATION.md`.

## Configuration Options

### Execution Settings

```yaml
execution:
  plugin: MultiProc      # Linear, MultiProc, or PBS
  n_procs: 4            # Number of parallel processes
```

### Registration Methods

```yaml
anatomical:
  registration_method: fsl    # 'fsl' or 'ants'

  # FSL options
  flirt_dof: 12              # Degrees of freedom (6, 9, or 12)
  flirt_cost: corratio       # Cost function

  # ANTs options (if using ANTs)
  ants_metric: MI            # Mutual Information or Cross-Correlation
  ants_convergence: [1000, 500, 250, 100]
```

### Diffusion Processing

```yaml
diffusion:
  denoise_method: dwidenoise  # 'dwidenoise' or 'none'
  gibbs_unring: true

  eddy_config:
    flm: linear              # First-level model
    slm: linear              # Second-level model
    niter: 5
    fwhm: 0
```

## Advanced Usage

### Using the Transform Registry

The Transform Registry enables efficient reuse of spatial transformations:

```python
from mri_preprocess.utils.transforms import create_transform_registry

# Create registry
registry = create_transform_registry(config, 'sub-001')

# Save transforms (done automatically by anatomical workflow)
registry.save_nonlinear_transform(
    warp_file=Path('warp.nii.gz'),
    affine_file=Path('affine.mat'),
    source_space='T1w',
    target_space='MNI152',
    subject='sub-001'
)

# Retrieve transforms (used by diffusion workflow)
warp, affine = registry.get_nonlinear_transform('T1w', 'MNI152')
```

### Batch Processing

```bash

# Process multiple subjects sequentially
for subject in sub-001 sub-002 sub-003; do
  # Anatomical must run first
  mri-preprocess run anatomical --config config.yaml --subject ${subject}
  # Then diffusion can reuse the anatomical transforms
  mri-preprocess run diffusion --config config.yaml --subject ${subject}
done

# Or use GNU Parallel for parallel execution
cat subjects.txt | parallel -j 4 \
  mri-preprocess run anatomical --config config.yaml --subject {}
cat subjects.txt | parallel -j 4 \
  mri-preprocess run diffusion --config config.yaml --subject {}
```

### Custom Workflows

```python
from nipype import Workflow, Node
from nipype.interfaces import fsl
from mri_preprocess.config import load_config

config = load_config(Path('config.yaml'))

# Create custom workflow
wf = Workflow(name='custom_processing')
wf.base_dir = config['work_dir']

# Add nodes
bet = Node(fsl.BET(frac=0.5, mask=True), name='bet')

# ... add more nodes

wf.run()
```

## Performance Benchmarks

Based on testing with 512x512x400 T1w data (Intel 4-core CPU):

| Step | Time | Notes |
|------|------|-------|
| Reorientation (fslreorient2std) | ~1s | Standard orientation |
| Skull Stripping (BET) | ~74s | Robust extraction |
| Bias Correction (Light N4) | ~150s (2.5min) | Optional, validated on minimal bias fields |
| Linear Registration (FLIRT) | ~225s (3.7min) | 12 DOF, corratio |
| Nonlinear Registration (FNIRT) | ~442s (7.4min) | High-quality warping |
| **Total Core Workflow** | ~11min | Reorient + BET + FLIRT + FNIRT |
| **With Light N4** | ~14min | Add 2.5 min for bias correction |

**Note:** ANTs registration (~15-20 min) available as alternative for research requiring maximal accuracy. TransformRegistry enables efficient reuse of transforms across diffusion and fMRI workflows.

## Troubleshooting

### FSL Not Found

```bash

# Ensure FSLDIR is set
echo $FSLDIR

# Should output: /usr/local/fsl

# Source FSL configuration
source ${FSLDIR}/etc/fslconf/fsl.sh
```

### Memory Issues

Reduce parallel processes in your config file:

```yaml
execution:
  plugin: MultiProc
  n_procs: 2  # Reduce from 4 to 2
```

### FAST Hanging on High-Resolution Data

Use ANTs N4BiasFieldCorrection instead in your config file:

```yaml
anatomical:
  bias_correction_method: ants  # Instead of 'fsl'
```

## Project Structure

```
human-mri-preprocess/
 mri_preprocess/
    config.py              # Configuration loading and validation
    cli.py                 # Command-line interface
    utils/
       transforms.py      # TransformRegistry
       workflow.py        # Workflow utilities
       bids.py           # BIDS helpers
    workflows/
        anat_preprocess.py # Anatomical workflow
        dwi_preprocess.py  # Diffusion workflow
 configs/                   # Example configurations
 TESTING_RESULTS.md        # Validation results
 README.md                 # This file
```

## Production Status

**Current Status (2025-11-14)**: Production-ready for anatomical, DWI, and ASL preprocessing. Functional preprocessing in final testing.

### ✅ Production Ready
- **Anatomical**: T1w preprocessing with N4, BET, segmentation, MNI registration
- **DWI**: Multi-shell preprocessing with optional TOPUP, eddy, DTI/DKI/NODDI, tractography
- **ASL**: pCASL preprocessing with M0 calibration and partial volume correction

### 🔄 In Testing
- **Functional**: Multi-echo TEDANA preprocessing (bug fixes complete, validation in progress)

### ⚠️ Experimental (Not Production Ready)
- **FreeSurfer Integration**: Detection and extraction hooks implemented, but transform pipeline (anatomical→DWI) not yet complete. Do not enable until spatial transformation workflow is validated.

For detailed project status and implementation notes, see `PROJECT_STATUS.md`.

## Recent Updates (2025-11-14)

- **Optional TOPUP**: DWI preprocessing now gracefully handles missing reverse phase-encoding images
- **Multi-echo Bug Fixes**: Fixed DICOM converter and workflow routing for multi-echo fMRI
- **FreeSurfer Hooks**: Added detection utilities for existing FreeSurfer outputs (experimental)
- **Spatial Normalization**: Implemented DWI→FMRIB58_FA and functional→MNI152 normalization

## Testing

```bash

# Run tests
pytest tests/

# Run with coverage
pytest --cov=mri_preprocess tests/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{human_mri_preprocess,
  title={Human MRI Preprocessing Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/human-mri-preprocess}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Nipype](https://nipype.readthedocs.io/)
- Uses [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) for neuroimaging processing
- Optional [ANTs](http://stnava.github.io/ANTs/) support for advanced registration
- Inspired by [fMRIPrep](https://fmriprep.org/) and [QSIPrep](https://qsiprep.readthedocs.io/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/human-mri-preprocess/issues
- Documentation: https://github.com/yourusername/human-mri-preprocess/wiki
