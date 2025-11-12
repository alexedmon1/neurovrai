# Human MRI Preprocessing Pipeline

A production-ready, config-driven MRI preprocessing pipeline for anatomical (T1w) and diffusion-weighted imaging (DWI) data. Built with Nipype for workflow orchestration, supporting both FSL and ANTs for neuroimaging processing.

## Features

- **Config-Driven Architecture**: YAML-based configuration for all processing parameters
- **BIDS-Compatible**: Follows Brain Imaging Data Structure conventions
- **Transform Registry**: Centralized management of spatial transformations for efficient reuse
- **Modular Workflows**: Separate anatomical and diffusion preprocessing pipelines
- **TOPUP Distortion Correction**: Advanced susceptibility distortion correction for DWI
- **GPU Acceleration**: CUDA support for FSL eddy correction
- **Standardized Output**: Consistent directory hierarchy across all workflows
- **Quality Control Framework**: Comprehensive QC for DWI (TOPUP, motion, DTI) and anatomical (skull stripping) preprocessing
- **Flexible Registration**: Support for both FSL (FLIRT/FNIRT) and ANTs registration
- **CLI Interface**: Command-line tools for batch processing
- **Production-Ready**: Tested and validated with multi-shell DWI data (see `docs/DWI_TOPUP_TEST_RESULTS.md`)

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
  eddy_config:
    flm: linear
    slm: linear
```

### 3. Run Anatomical Preprocessing

```bash

# Using the CLI
mri-preprocess run anatomical \
  --config config.yaml \
  --subject sub-001

# Or using Python
python -c "
from pathlib import Path
from mri_preprocess.config import load_config
from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing

config = load_config(Path('config.yaml'))
results = run_anat_preprocessing(
    config=config,
    subject='sub-001',
    t1w_file=Path('rawdata/sub-001/anat/sub-001_T1w.nii.gz'),
    output_dir=Path('derivatives'),
    work_dir=Path('work')
)
"
```

### 4. Run Diffusion Preprocessing

```bash

# The diffusion workflow automatically reuses anatomical transforms
mri-preprocess run diffusion \
  --config config.yaml \
  --subject sub-001

# Optional: Enable BEDPOSTX for fiber distributions (slow, requires GPU)
mri-preprocess run diffusion \
  --config config.yaml \
  --subject sub-001 \
  --bedpostx
```

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
