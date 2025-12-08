# Functional Connectivity (Connectome) Module

Compute functional connectivity matrices (connectomes) from preprocessed resting-state fMRI data.

## Analysis Approach: Native-Space Connectomes

**This pipeline uses native-space functional connectivity analysis.**

### How It Works

1. **Functional data stays in native space** - Uses preprocessed BOLD data as-is (no additional normalization)
2. **Atlases are resampled to functional space** - MNI atlases warped to match each subject's functional dimensions
3. **Timeseries extraction** - Mean signal extracted from each atlas ROI in native space
4. **Connectivity computation** - Pearson/Spearman correlation between all ROI pairs

### No BBR Transforms Needed!

Unlike MNI-space approaches, native-space connectivity:
- **Does NOT use** functional→anatomical (BBR) transforms
- **Does NOT use** anatomical→MNI transforms
- **Does NOT normalize** functional data to MNI

Instead, atlases are simply **resampled** (like zooming/shrinking an image) to match the functional data's native dimensions using nearest-neighbor interpolation.

### Why This Approach?

**Advantages:**
- **Simple and robust** - No transform compatibility issues (FSL vs ANTs)
- **Preserves functional resolution** - No interpolation of 4D timeseries
- **Fast** - No normalization step (~30-60 sec per subject-atlas)
- **Scientifically valid** - Standard approach in rs-fMRI connectivity research

**Trade-off:**
- Each subject has slightly different atlas resampling
- This is a minor, acceptable source of variability in connectivity studies

---

## Batch Functional Connectivity

### Quick Start

Process all subjects with all atlases:

```bash
uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root /mnt/bytopia/IRC805 \
    --atlases all \
    --output-dir /mnt/bytopia/IRC805/analysis/func/connectivity
```

### Command-Line Options

```bash
uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root /path/to/study \              # Study root directory
    --output-dir /path/to/output \             # Output directory (default: {study-root}/analysis/func/connectivity)
    --atlases harvardoxford_cort juelich \     # Specific atlases (default: all)
    --subjects sub-001 sub-002 \               # Specific subjects (default: all with preprocessed data)
    --method pearson \                          # Correlation method: pearson or spearman (default: pearson)
    --no-fisher-z \                            # Disable Fisher z-transformation
    --verbose                                  # Enable verbose logging
```

### Available Atlases

The pipeline includes 5 FSL atlases in MNI 2mm space:

| Atlas Name | Description | Regions |
|------------|-------------|---------|
| `harvardoxford_cort` | Harvard-Oxford Cortical | 48 regions |
| `harvardoxford_sub` | Harvard-Oxford Subcortical | 21 regions |
| `juelich` | Juelich Histological Atlas | Variable |
| `talairach` | Talairach Atlas | Variable |
| `cerebellum_mniflirt` | Cerebellum MNI FLIRT | Variable |

### Output Structure

```
{output-dir}/
├── {subject}/
│   ├── {atlas}/
│   │   ├── fc_matrix.npy                   # Connectivity matrix (NumPy)
│   │   ├── fc_matrix.csv                   # Connectivity matrix (CSV)
│   │   ├── fc_roi_names.txt                # ROI labels
│   │   ├── fc_summary.json                 # Summary statistics
│   │   ├── atlas_{atlas}_resampled.nii.gz  # Atlas resampled to functional space
│   │   └── analysis_metadata.json          # Processing metadata
├── logs/
│   └── batch_fc_YYYYMMDD_HHMMSS.log        # Processing log
└── batch_processing_summary.json            # Batch summary
```

### Example Outputs

**fc_matrix.npy**: Fisher z-transformed connectivity matrix (NumPy array)
```python
import numpy as np
fc_matrix = np.load('fc_matrix.npy')  # Shape: (n_rois, n_rois)
```

**fc_matrix.csv**: Same matrix in CSV format for easy viewing/analysis

**fc_roi_names.txt**: List of ROI names (one per line)

**fc_summary.json**: Summary statistics
```json
{
  "n_rois": 37,
  "n_edges_total": 666,
  "n_edges_nonzero": 666,
  "mean_connectivity": 0.1843,
  "std_connectivity": 0.3125,
  "min_connectivity": -0.4178,
  "max_connectivity": 1.3639
}
```

**analysis_metadata.json**: Complete processing metadata
```json
{
  "subject": "IRC805-0580101",
  "atlas": "harvardoxford_cort",
  "atlas_description": "Harvard-Oxford Cortical (48 regions)",
  "n_rois": 37,
  "n_timepoints": 450,
  "method": "pearson",
  "fisher_z": true,
  "analysis_date": "2025-12-07T20:21:15"
}
```

---

## Python API

### Single Subject Analysis

```python
from pathlib import Path
from neurovrai.connectome.batch_functional_connectivity import (
    process_subject_atlas,
    ATLAS_DEFINITIONS
)

result = process_subject_atlas(
    subject='sub-001',
    files={
        'func': Path('preprocessed_bold.nii.gz'),
        'mask': Path('brain_mask.nii.gz'),
        'brain': Path('brain.nii.gz')
    },
    atlas_name='harvardoxford_cort',
    atlas_config=ATLAS_DEFINITIONS['harvardoxford_cort'],
    output_dir=Path('output'),
    method='pearson',
    fisher_z=True
)

if result['status'] == 'success':
    print(f"Mean connectivity: {result['mean_connectivity']:.4f}")
```

### Extract ROI Timeseries

```python
from neurovrai.connectome.roi_extraction import extract_roi_timeseries, load_atlas

# Load atlas
atlas = load_atlas(
    atlas_file=Path('schaefer_400.nii.gz'),
    labels_file=None  # Optional labels
)

# Extract timeseries
timeseries, roi_names = extract_roi_timeseries(
    data_file=Path('preprocessed_bold.nii.gz'),
    atlas=atlas,
    mask_file=Path('brain_mask.nii.gz'),
    min_voxels=10,
    statistic='mean'  # or 'median', 'pca'
)

print(f"Timeseries shape: {timeseries.shape}")  # (n_timepoints, n_rois)
```

### Compute Connectivity Matrix

```python
from neurovrai.connectome.functional_connectivity import compute_functional_connectivity

fc_results = compute_functional_connectivity(
    timeseries=timeseries,
    roi_names=roi_names,
    method='pearson',  # or 'spearman'
    fisher_z=True,
    partial=False,
    threshold=None,  # Optional threshold
    output_dir=Path('output'),
    output_prefix='fc'
)

# Access results
fc_matrix = fc_results['fc_matrix']  # Fisher z-transformed
raw_corr = fc_results['correlation_matrix']  # Raw correlation
summary = fc_results['summary']
```

---

## Quality Control

### ROI Coverage

Some ROIs may be skipped if they have insufficient voxels after resampling:

```
WARNING - ROI 8.0 has only 0 voxels (< 10), skipping
WARNING - ROI 15.0 has only 4 voxels (< 10), skipping
```

This is expected when atlases are resampled to lower-resolution functional data. The pipeline requires at least 10 voxels per ROI to ensure reliable signal extraction.

### Validation Checks

1. **Spatial dimensions**: Automatic resampling if atlas and functional data don't match
2. **Valid ROIs**: Minimum voxel count per ROI (default: 10)
3. **Timeseries quality**: Warns if extracted timeseries have constant values
4. **Matrix symmetry**: Connectivity matrices are symmetric (correlation is symmetric)

---

## Technical Details

### Atlas Resampling

Atlases are resampled using `nilearn.image.resample_to_img` with:
- **Interpolation**: Nearest-neighbor (preserves discrete labels)
- **Target space**: Functional data's native space
- **Affine alignment**: Based on NIfTI headers

### Connectivity Methods

**Pearson Correlation**:
- Linear correlation between ROI timeseries
- Fast and standard for functional connectivity
- Fisher z-transformation applied by default for group statistics

**Spearman Correlation**:
- Rank-based correlation (non-parametric)
- Robust to outliers and non-linear relationships

### Fisher Z-Transformation

Applied by default to normalize correlation distributions:
```
z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
```

This transformation makes correlation values more suitable for parametric statistics.

---

## Performance

Typical processing times (per subject-atlas pair):
- **Atlas resampling**: ~20 seconds
- **Timeseries extraction**: ~2 seconds (450 timepoints, 37 ROIs)
- **Connectivity computation**: <1 second
- **Total**: ~30-60 seconds per subject-atlas

Batch processing 17 subjects × 5 atlases (85 analyses): ~45-75 minutes

---

## References

- **Native-space connectivity**: Power et al. (2014) NeuroImage
- **Fisher z-transformation**: Fisher (1915) Biometrika
- **Graph theory**: Rubinov & Sporns (2010) NeuroImage
