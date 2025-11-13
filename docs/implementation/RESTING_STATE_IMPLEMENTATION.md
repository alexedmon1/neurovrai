# Resting State fMRI Preprocessing - Implementation Summary

## Overview

Successfully implemented a complete resting state fMRI preprocessing pipeline with multi-echo TEDANA denoising and comprehensive quality control.

## Implementation Status: ✅ COMPLETE

### Completed Components

#### 1. Core Preprocessing Pipeline (`mri_preprocess/workflows/func_preprocess.py`)

**Main Function**: `run_func_preprocessing()`

**Pipeline Steps**:
1. **TEDANA Multi-Echo Denoising** (for multi-echo data)
   - Automatic detection of multi-echo vs single-echo data
   - Brain mask creation from first echo
   - Optimal combination of echoes
   - Thermal noise removal
   - BOLD signal identification
   - HTML report generation

2. **Motion Correction** (MCFLIRT)
   - Least squares cost function
   - 6 DOF rigid body registration
   - Motion parameter extraction (rotations + translations)
   - RMS plots for QC

3. **Brain Extraction** (BET)
   - Functional-optimized parameters (frac=0.3)
   - Brain mask generation

4. **Motion Artifact Removal** (ICA-AROMA)
   - Independent component analysis
   - Automatic identification of motion components
   - Both aggressive and non-aggressive denoising

5. **Temporal Filtering** (Bandpass)
   - Default: 0.001 - 0.08 Hz (resting state band)
   - Configurable highpass/lowpass frequencies

6. **Spatial Smoothing**
   - Default: 6mm FWHM
   - Gaussian kernel
   - Improves SNR

7. **Quality Control** (NEW)
   - Motion QC metrics
   - Temporal SNR calculation
   - HTML report generation

#### 2. Quality Control Module (`mri_preprocess/qc/func_qc.py`)

**Function**: `compute_motion_qc()`
- Framewise displacement (FD) calculation
- Rotations converted to mm displacement (50mm head radius)
- Outlier volume detection (default threshold: 0.5mm)
- Mean/max rotation and translation metrics
- Motion parameter plots
- FD time series plot with threshold

**Function**: `compute_tsnr()`
- Temporal SNR = mean / std over time
- Uses FSL fslmaths for computation
- Brain-masked metrics
- tSNR histogram generation
- Mean, median, std, min, max statistics

**Function**: `generate_func_qc_report()`
- Comprehensive HTML report
- Embedded CSS styling
- Color-coded quality indicators:
  - **Motion**: Good (<0.2mm), Acceptable (0.2-0.5mm), Poor (>0.5mm)
  - **Outliers**: Good (<5%), Acceptable (5-20%), Poor (>20%)
  - **tSNR**: Excellent (>100), Good (50-100), Poor (<50)
- Embedded plots (motion, FD, tSNR histogram)
- Link to TEDANA report if available

#### 3. Test Script (`test_rest_preprocessing.py`)

**Features**:
- Tests on IRC805-0580101 multi-echo data
- Verified echo times: 10, 30, 50 ms
- Complete configuration with all parameters
- QC summary in output
- Next steps guidance

## File Structure

```
mri_preprocess/
├── workflows/
│   └── func_preprocess.py          # Main preprocessing workflow
├── qc/
│   └── func_qc.py                  # Quality control functions
└── utils/
    └── workflow.py                  # Shared workflow utilities

test_rest_preprocessing.py           # Test script
RESTING_STATE_PLAN.md               # Planning document
RESTING_STATE_IMPLEMENTATION.md     # This file
```

## Configuration Parameters

```python
config = {
    'tr': 1.029,                    # Repetition time (seconds)
    'te': [10.0, 30.0, 50.0],      # Echo times (milliseconds)
    'highpass': 0.001,              # Highpass filter (Hz)
    'lowpass': 0.08,                # Lowpass filter (Hz)
    'fwhm': 6,                      # Smoothing kernel FWHM (mm)
    'n_procs': 6,                   # Parallel processes
    'tedana': {
        'enabled': True,            # Enable TEDANA
        'tedpca': 'kundu',         # PCA method
        'tree': 'kundu'            # Decision tree
    },
    'aroma': {
        'enabled': True,            # Enable ICA-AROMA
        'denoise_type': 'both'     # Aggressive + non-aggressive
    },
    'acompcor': {
        'enabled': False,           # Requires anatomical masks
        'num_components': 6
    },
    'run_qc': True,                # Run quality control (default)
    'fd_threshold': 0.5            # FD threshold in mm
}
```

## Output Files

The pipeline generates the following outputs in `{study_root}/derivatives/func_preproc/{subject}/`:

### Preprocessing Outputs
- `tedana/tedana_desc-optcom_bold.nii.gz` - Optimally combined echoes
- `tedana/tedana_desc-denoised_bold.nii.gz` - Denoised data
- `tedana/tedana_report.html` - TEDANA analysis report
- `func_preproc/{subject}_preprocessed.nii.gz` - Final preprocessed data
- `motion_correction/{subject}_mcf.par` - Motion parameters
- `workflow_graph.png` - Pipeline visualization

### Quality Control Outputs
- `qc/motion_params.png` - Motion parameter time series
- `qc/framewise_displacement.png` - FD time series
- `qc/tsnr.nii.gz` - tSNR map
- `qc/tsnr_histogram.png` - tSNR distribution
- `qc/{subject}_func_qc_report.html` - **Main QC report**

## Usage Example

```python
from pathlib import Path
from mri_preprocess.workflows.func_preprocess import run_func_preprocessing

# Configuration
config = {
    'tr': 1.029,
    'te': [10.0, 30.0, 50.0],
    'highpass': 0.001,
    'lowpass': 0.08,
    'fwhm': 6,
    'n_procs': 6,
    'tedana': {'enabled': True, 'tedpca': 'kundu', 'tree': 'kundu'},
    'aroma': {'enabled': True, 'denoise_type': 'both'},
    'run_qc': True
}

# Paths
study_root = Path('/mnt/bytopia/IRC805')
subject = 'IRC805-0580101'
rest_dir = study_root / f'subjects/{subject}/nifti/rest'

# Multi-echo files
func_files = [
    rest_dir / f'501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e1.nii.gz',
    rest_dir / f'501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e2.nii.gz',
    rest_dir / f'501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e3.nii.gz'
]

# Run preprocessing
results = run_func_preprocessing(
    config=config,
    subject=subject,
    func_file=func_files,
    output_dir=study_root
)

# Access results
print(f"Preprocessed: {results['preprocessed']}")
print(f"QC Report: {results['qc_report']}")
print(f"Mean FD: {results['motion_qc']['mean_fd']:.3f} mm")
print(f"Mean tSNR: {results['tsnr_qc']['mean_tsnr']:.2f}")
```

## Quality Control Metrics

### Motion Metrics
- **Mean FD**: Average framewise displacement across all volumes
- **Max FD**: Maximum framewise displacement
- **Outlier Volumes**: Number and percentage of volumes exceeding threshold
- **Mean Rotation**: Average rotation in degrees
- **Mean Translation**: Average translation in mm

### tSNR Metrics
- **Mean tSNR**: Average temporal SNR in brain
- **Median tSNR**: Median temporal SNR
- **tSNR Distribution**: Histogram showing voxel-wise tSNR values

## Quality Thresholds

### Framewise Displacement (FD)
- **Good**: <0.2 mm
- **Acceptable**: 0.2-0.5 mm
- **Poor**: >0.5 mm

### Outlier Volumes
- **Good**: <5%
- **Acceptable**: 5-20%
- **Poor**: >20%

### Temporal SNR (tSNR)
- **Excellent**: >100
- **Good**: 50-100
- **Poor**: <50

## Dataset Information

### IRC805 Multi-Echo Resting State
- **Scanner**: Philips Ingenia Elition X 3T
- **Sequence**: Multi-echo gradient echo EPI
- **Multi-band**: MB3 (factor 3)
- **SENSE**: Factor 3
- **Echo times**: 10, 30, 50 ms
- **TR**: 1.029 seconds
- **Volumes**: 450 timepoints
- **Voxel size**: 1.875 × 1.875 × 3.0 mm
- **Matrix**: 128 × 128 × 42
- **Total scan time**: ~7.7 minutes

## Expected Runtime

### Per Subject (IRC805-0580101)
- TEDANA: ~5-10 minutes
- Motion correction: ~5 minutes
- ICA-AROMA: ~15-20 minutes
- Bandpass filtering: ~2 minutes
- Spatial smoothing: ~2 minutes
- Quality control: ~2 minutes
- **Total**: ~30-45 minutes

### For Full IRC805 Cohort (~100 subjects)
- Serial: ~50-75 hours
- Parallel (10 subjects): ~5-7.5 hours

## Next Steps

### Immediate
1. ✅ **COMPLETED**: TEDANA implementation
2. ✅ **COMPLETED**: Motion correction
3. ✅ **COMPLETED**: ICA-AROMA
4. ✅ **COMPLETED**: Temporal filtering
5. ✅ **COMPLETED**: Spatial smoothing
6. ✅ **COMPLETED**: Quality control module
7. ✅ **COMPLETED**: HTML QC report
8. ✅ **COMPLETED**: Integration with main workflow

### Future Enhancements
1. **ACompCor Nuisance Regression**
   - Requires CSF/WM masks from anatomical preprocessing
   - Extract tissue-based nuisance signals
   - Regress out physiological noise

2. **Registration to MNI Space**
   - Coregister functional to T1w (BBR)
   - Apply T1w→MNI transform
   - Warp preprocessed data to standard space

3. **Additional QC Metrics**
   - Carpet plots (voxel intensity over time)
   - DVARS (temporal derivative of RMS variance)
   - Registration quality metrics
   - Tissue contrast metrics

4. **Connectivity Analysis**
   - Seed-based correlation analysis
   - ICA-based network extraction
   - Dual regression with templates
   - Connectivity matrix generation

5. **Group-Level QC**
   - Aggregate QC metrics across subjects
   - Identify outliers in cohort
   - Group mean tSNR maps
   - Motion distribution plots

## Testing

To test the complete pipeline:

```bash
# Activate environment
source .venv/bin/activate

# Run test script
python test_rest_preprocessing.py

# Check outputs
firefox /mnt/bytopia/IRC805/derivatives/func_preproc/IRC805-0580101/qc/IRC805-0580101_func_qc_report.html
```

## Dependencies

### Python Packages (from pyproject.toml)
- nipype ≥1.10.0
- nibabel ≥5.3.2
- tedana ≥23.0.2
- numpy
- pandas
- matplotlib
- scipy

### System Dependencies
- FSL (MCFLIRT, BET, ICA-AROMA, fslmaths)
- AFNI (Bandpass)

## References

### TEDANA
- Kundu et al. (2012). "Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI." NeuroImage.
- TEDANA documentation: https://tedana.readthedocs.io/

### ICA-AROMA
- Pruim et al. (2015). "ICA-AROMA: A robust ICA-based strategy for removing motion artifacts from fMRI data." NeuroImage.

### Framewise Displacement
- Power et al. (2012). "Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion." NeuroImage.

### Temporal SNR
- Murphy et al. (2007). "The impact of global signal regression on resting state correlations: Are anti-correlated networks introduced?" NeuroImage.

## Acknowledgments

Implemented based on best practices from:
- fMRIPrep preprocessing pipeline
- CONN functional connectivity toolbox
- Human Connectome Project (HCP) minimal preprocessing pipelines
- TEDANA development team

---

**Implementation Date**: 2025-11-12
**Status**: Production Ready
**Tested On**: IRC805-0580101 (multi-echo resting state)
