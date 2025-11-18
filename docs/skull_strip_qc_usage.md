# Skull Stripping Quality Control (QC) Usage Guide

This guide demonstrates how to use the skull stripping QC modules for DWI, functional, and ASL data.

## Overview

Skull stripping QC modules assess the quality of brain extraction (BET) by:
- **Brain mask coverage statistics**: Volume, voxel count, bounding box
- **Over/under-stripping detection**: Contrast ratios, intensity variance, volume thresholds
- **Visual slice overlays**: Red contour overlays on axial and sagittal slices
- **Quality flags**: Automated warnings for common issues (LOW_CONTRAST, HIGH_VARIANCE, SMALL/LARGE_BRAIN_VOLUME)

## DWI Skull Stripping QC

### Class-Based Interface

```python
from pathlib import Path
from neurovrai.preprocess.qc.dwi.skull_strip_qc import DWISkullStripQualityControl

# Initialize QC
subject = "IRC805-0580101"
dwi_dir = Path("/mnt/bytopia/IRC805/derivatives/IRC805-0580101/dwi")
qc_dir = Path("/mnt/bytopia/IRC805/qc/dwi/IRC805-0580101/skull_strip")

qc = DWISkullStripQualityControl(
    subject=subject,
    dwi_dir=dwi_dir,
    qc_dir=qc_dir
)

# Run QC (auto-detects b0 and mask files)
results = qc.run_qc()

# Or specify files explicitly
results = qc.run_qc(
    b0_file=dwi_dir / "b0_mean.nii.gz",
    mask_file=dwi_dir / "brain_mask.nii.gz"
)

# Access results
print(f"Brain volume: {results['mask_stats']['brain_volume_cm3']:.2f} cm³")
print(f"Quality flags: {results['quality']['quality_flags']}")
print(f"Overlay saved: {results['outputs']['mask_overlay']}")
```

### Expected Outputs

```
{study_root}/qc/dwi/{subject}/skull_strip/
├── brain_mask_overlay.png          # Visual overlay plot
└── metrics/
    └── skull_strip.json             # Quantitative metrics
```

## Functional Skull Stripping QC

### Function-Based Interface

```python
from pathlib import Path
from neurovrai.preprocess.qc.func_qc import compute_skull_strip_qc

# Specify input files
func_mean_file = Path("/mnt/bytopia/IRC805/derivatives/IRC805-0580101/func/mean_func.nii.gz")
mask_file = Path("/mnt/bytopia/IRC805/derivatives/IRC805-0580101/func/brain_mask.nii.gz")
output_dir = Path("/mnt/bytopia/IRC805/qc/func/IRC805-0580101/skull_strip")

# Run QC
results = compute_skull_strip_qc(
    func_mean_file=func_mean_file,
    mask_file=mask_file,
    output_dir=output_dir,
    subject="IRC805-0580101"
)

# Access results
print(f"Brain volume: {results['brain_volume_cm3']:.2f} cm³")
print(f"Contrast ratio: {results['contrast_ratio']:.2f}")
print(f"Quality pass: {results['quality_pass']}")
print(f"Flags: {results['quality_flags']}")
```

### Expected Outputs

```
{study_root}/qc/func/{subject}/skull_strip/
├── skull_strip_overlay.png         # Visual overlay plot
└── skull_strip_metrics.json        # Quantitative metrics
```

## ASL Skull Stripping QC

### Function-Based Interface

```python
from pathlib import Path
from neurovrai.preprocess.qc.asl_qc import compute_asl_skull_strip_qc

# Specify input files (use M0 or mean control image)
asl_mean_file = Path("/mnt/bytopia/IRC805/derivatives/IRC805-0580101/asl/M0.nii.gz")
mask_file = Path("/mnt/bytopia/IRC805/derivatives/IRC805-0580101/asl/brain_mask.nii.gz")
output_dir = Path("/mnt/bytopia/IRC805/qc/asl/IRC805-0580101/skull_strip")

# Run QC
results = compute_asl_skull_strip_qc(
    asl_mean_file=asl_mean_file,
    mask_file=mask_file,
    output_dir=output_dir,
    subject="IRC805-0580101"
)

# Access results
print(f"Brain volume: {results['brain_volume_cm3']:.2f} cm³")
print(f"Quality flags: {results['quality_flags']}")
```

### Expected Outputs

```
{study_root}/qc/asl/{subject}/skull_strip/
├── skull_strip_overlay.png         # Visual overlay plot
└── skull_strip_metrics.json        # Quantitative metrics
```

## QC Metrics Explained

### Brain Volume Statistics

- **n_voxels**: Number of brain mask voxels
- **brain_volume_mm3**: Brain volume in mm³
- **brain_volume_cm3**: Brain volume in cm³ (typical range: 800-1800 cm³)
- **voxel_size_mm**: Voxel dimensions [x, y, z] in mm
- **bbox**: Bounding box coordinates (x_min, x_max, y_min, y_max, z_min, z_max)
- **bbox_size**: Bounding box dimensions

### Intensity Quality Metrics

- **brain_mean_intensity**: Mean intensity within brain mask
- **brain_std_intensity**: Standard deviation within brain mask
- **outside_mean_intensity**: Mean intensity outside brain mask
- **outside_std_intensity**: Standard deviation outside brain mask
- **contrast_ratio**: brain_mean / outside_mean (higher = better separation)

### Quality Flags

- **LOW_CONTRAST**: Contrast ratio < 1.5 (DWI/func/ASL) or < 2.0 (anatomical)
  - *Indicates*: Poor brain/non-brain separation, possible under-stripping

- **HIGH_VARIANCE**: brain_std / brain_mean > 0.6 (DWI/func/ASL) or > 0.5 (anatomical)
  - *Indicates*: Inconsistent intensities within mask, possible artifacts

- **SMALL_BRAIN_VOLUME**: brain_volume_cm3 < 500 cm³
  - *Indicates*: Possible over-stripping (cortex removed)

- **LARGE_BRAIN_VOLUME**: brain_volume_cm3 > 2500 cm³
  - *Indicates*: Possible under-stripping (skull/dura included)

### Quality Pass Criteria

`quality_pass = True` if and only if no quality flags are raised.

## Integration with Preprocessing Workflows

### DWI Workflow Integration

```python
from neurovrai.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing
from neurovrai.preprocess.qc.dwi.skull_strip_qc import DWISkullStripQualityControl

# Run preprocessing
results = run_dwi_multishell_topup_preprocessing(
    config=config,
    subject="IRC805-0580101",
    dwi_files=dwi_files,
    bval_files=bval_files,
    bvec_files=bvec_files,
    output_dir=study_root
)

# Run skull strip QC
dwi_dir = study_root / "derivatives" / "IRC805-0580101" / "dwi"
qc_dir = study_root / "qc" / "dwi" / "IRC805-0580101" / "skull_strip"

qc = DWISkullStripQualityControl(
    subject="IRC805-0580101",
    dwi_dir=dwi_dir,
    qc_dir=qc_dir
)

qc_results = qc.run_qc(
    b0_file=results.get('b0_mean'),
    mask_file=results.get('brain_mask')
)

# Check quality
if not qc_results['quality']['quality_pass']:
    print(f"WARNING: Skull strip QC failed with flags: {qc_results['quality']['quality_flags']}")
```

### Functional Workflow Integration

```python
from neurovrai.workflows.func_preprocess import run_func_preprocessing
from neurovrai.preprocess.qc.func_qc import compute_skull_strip_qc, compute_motion_qc, compute_tsnr

# Run preprocessing
results = run_func_preprocessing(
    config=config,
    subject="IRC805-0580101",
    func_files=func_files,
    output_dir=study_root
)

# Run comprehensive QC
qc_dir = study_root / "qc" / "func" / "IRC805-0580101"

# Motion QC
motion_qc = compute_motion_qc(
    motion_file=results['motion_params'],
    tr=config['functional']['tr'],
    output_dir=qc_dir / "motion"
)

# tSNR QC
tsnr_qc = compute_tsnr(
    func_file=results['preprocessed_bold'],
    mask_file=results['brain_mask'],
    output_dir=qc_dir / "tsnr"
)

# Skull strip QC
skull_qc = compute_skull_strip_qc(
    func_mean_file=results['mean_func'],
    mask_file=results['brain_mask'],
    output_dir=qc_dir / "skull_strip",
    subject="IRC805-0580101"
)

# Aggregate QC results
if not skull_qc['quality_pass']:
    print(f"WARNING: Skull strip quality issues: {skull_qc['quality_flags']}")
```

## Batch Processing Example

```python
from pathlib import Path
from neurovrai.preprocess.qc.dwi.skull_strip_qc import DWISkullStripQualityControl
import pandas as pd

study_root = Path("/mnt/bytopia/IRC805")
subjects = ["IRC805-0580101", "IRC805-0580102", "IRC805-0580103"]

qc_summary = []

for subject in subjects:
    dwi_dir = study_root / "derivatives" / subject / "dwi"
    qc_dir = study_root / "qc" / "dwi" / subject / "skull_strip"

    qc = DWISkullStripQualityControl(subject=subject, dwi_dir=dwi_dir, qc_dir=qc_dir)
    results = qc.run_qc()

    # Extract key metrics
    qc_summary.append({
        'subject': subject,
        'brain_volume_cm3': results['mask_stats']['brain_volume_cm3'],
        'n_voxels': results['mask_stats']['n_voxels'],
        'contrast_ratio': results['quality']['contrast_ratio'],
        'quality_pass': results['quality']['quality_pass'],
        'quality_flags': ', '.join(results['quality']['quality_flags'])
    })

# Save summary
df = pd.DataFrame(qc_summary)
df.to_csv(study_root / "qc" / "dwi_skull_strip_summary.csv", index=False)
print(df)
```

## Troubleshooting

### Common Issues

**Issue**: `LOW_CONTRAST` flag raised
- **Cause**: Poor separation between brain and non-brain tissue
- **Solution**: Consider adjusting BET parameters (frac, reduce_bias) in config.yaml

**Issue**: `SMALL_BRAIN_VOLUME` flag raised
- **Cause**: BET removed too much brain tissue (over-stripping)
- **Solution**: Decrease BET frac parameter (e.g., 0.5 → 0.3)

**Issue**: `LARGE_BRAIN_VOLUME` flag raised
- **Cause**: BET included skull/dura (under-stripping)
- **Solution**: Increase BET frac parameter (e.g., 0.5 → 0.6 or 0.7)

**Issue**: `HIGH_VARIANCE` flag raised
- **Cause**: Inconsistent intensities, possible motion artifacts or noise
- **Solution**: Check preprocessing quality, consider stronger denoising

### Manual Review

Always visually inspect the overlay plots:
- Red contours should follow brain boundary closely
- Look for missing cortex (over-stripping)
- Look for included skull/eyes/neck (under-stripping)

## References

- FSL BET documentation: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET
- Recommended BET parameters vary by modality:
  - **T1w anatomical**: frac=0.5, reduce_bias=True
  - **DWI (b0)**: frac=0.3-0.4 (lower due to lower SNR)
  - **Functional**: frac=0.4-0.5
  - **ASL**: frac=0.3-0.4 (lower due to lower SNR)
