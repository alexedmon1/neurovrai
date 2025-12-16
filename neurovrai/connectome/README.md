# Connectome Module

Compute connectivity matrices from neuroimaging data:
- **Functional Connectivity**: Correlation-based matrices from resting-state fMRI
- **Structural Connectivity**: Tractography-based matrices from diffusion MRI

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

## Structural Connectivity

Compute tractography-based structural connectivity matrices from diffusion MRI using FSL's probtrackx2.

### Quick Start

```bash
# Single subject
uv run python -m neurovrai.connectome.run_structural_connectivity \
    --subject IRC805-0580101 \
    --derivatives-dir /mnt/bytopia/IRC805/derivatives \
    --atlas schaefer_200 \
    --config /mnt/bytopia/IRC805/config.yaml \
    --output-dir /mnt/bytopia/IRC805/connectome/structural

# Batch processing
uv run python -m neurovrai.connectome.batch_structural_connectivity \
    --study-root /mnt/bytopia/IRC805 \
    --atlases schaefer_200 schaefer_400 \
    --config /mnt/bytopia/IRC805/config.yaml \
    --output-dir /mnt/bytopia/IRC805/connectome/structural
```

### Pipeline Steps

1. **BEDPOSTX**: Fiber orientation modeling (GPU-accelerated if available)
2. **Atlas Transformation**: Transform atlas from MNI/FreeSurfer to DWI native space
3. **Anatomical Constraints**: Create avoidance/waypoint masks from tissue segmentation
4. **Probtrackx2**: Probabilistic tractography in network mode
5. **Matrix Construction**: Build normalized connectivity matrix
6. **Graph Metrics**: Compute network topology metrics (optional)

### Anatomical Constraints

The pipeline supports multiple anatomical constraints for biologically plausible tractography:

| Constraint | Description | Source |
|------------|-------------|--------|
| `avoid_ventricles` | Exclude CSF/ventricles from tractography | FreeSurfer (labels 4,5,14,15,43,44,72) |
| `use_wm_mask` | Constrain tractography to white matter | FreeSurfer or FSL FAST |
| `terminate_at_gm` | Stop streamlines at gray matter boundary | FreeSurfer |
| `use_gmwmi_seeding` | Seed from gray-white matter interface | FreeSurfer surfaces or volume |
| `use_subcortical_waypoints` | Use thalamus/basal ganglia as waypoints | FreeSurfer |

### Configuration

Add to your `config.yaml`:

```yaml
structural_connectivity:
  tractography:
    use_gpu: true              # Use probtrackx2_gpu (5-10x faster)
    n_samples: 5000            # Samples per seed voxel
    step_length: 0.5           # Step length in mm
    curvature_threshold: 0.2   # Curvature threshold (0-1)
    loop_check: true           # Discard looping streamlines

  anatomical_constraints:
    avoid_ventricles: true     # Exclude CSF from tractography (recommended)
    use_wm_mask: true          # Constrain to white matter (ACT-style)
    terminate_at_gm: false     # Stop at gray matter
    wm_source: auto            # 'freesurfer', 'fsl', or 'auto'

  freesurfer_options:
    use_gmwmi_seeding: true    # Seed from gray-white interface (more anatomically precise)
    gmwmi_method: surface      # 'surface' (lh/rh.white) or 'volume' (aparc+aseg)
    use_subcortical_waypoints: false
    subcortical_structures:
      - Left-Thalamus
      - Right-Thalamus

  atlas:
    default: schaefer_200
    available:
      - schaefer_100
      - schaefer_200
      - schaefer_400
      - desikan_killiany       # Requires FreeSurfer

  output:
    normalize: true            # Normalize by waytotal
    threshold: null            # Optional threshold (0-1)
    compute_graph_metrics: true
    save_streamlines: false

  run_qc: true
```

### Available Atlases

| Atlas | Space | Regions | Requirements |
|-------|-------|---------|--------------|
| `schaefer_100` | MNI152 | 100 cortical | None |
| `schaefer_200` | MNI152 | 200 cortical | None |
| `schaefer_400` | MNI152 | 400 cortical | None |
| `desikan_killiany` | FreeSurfer | 68 cortical + subcortical | FreeSurfer recon-all |

### Output Structure

```
connectome/structural/{subject}/
├── sc_matrix.npy                # Connectivity matrix (NumPy)
├── sc_matrix.csv                # Connectivity matrix (CSV)
├── sc_roi_names.txt             # ROI labels
├── sc_summary.json              # Summary statistics
├── graph_metrics.json           # Network topology metrics
├── analysis_metadata.json       # Processing metadata
├── atlas/
│   └── {atlas}_dwi.nii.gz       # Atlas in DWI space
└── probtrackx_output/
    ├── fdt_network_matrix       # Raw tractography output
    └── waytotal                 # Normalization factors
```

### Python API

```python
from neurovrai.connectome.structural_connectivity import compute_structural_connectivity
from neurovrai.connectome.graph_metrics import compute_graph_metrics
from pathlib import Path

# Compute structural connectivity
sc_results = compute_structural_connectivity(
    bedpostx_dir=Path('derivatives/sub-001/dwi.bedpostX'),
    atlas_file=Path('schaefer_200_dwi.nii.gz'),
    output_dir=Path('connectome/structural/sub-001'),
    n_samples=5000,
    avoid_ventricles=True,
    config=config  # Uses tractography settings from config
)

# Access results
sc_matrix = sc_results['connectivity_matrix']
roi_names = sc_results['roi_names']
metadata = sc_results['metadata']

print(f"ROIs: {metadata['n_rois']}")
print(f"Connections: {metadata['n_connections']}")
print(f"Density: {metadata['connection_density']:.3f}")

# Compute graph metrics
graph_metrics = compute_graph_metrics(
    connectivity_matrix=sc_matrix,
    roi_names=roi_names
)

print(f"Global efficiency: {graph_metrics['global']['global_efficiency']:.4f}")
print(f"Clustering: {graph_metrics['global']['clustering_coefficient']:.4f}")
```

### Performance

**BEDPOSTX** (fiber orientation modeling):
- CPU: 4-8 hours per subject
- GPU: 20-60 minutes per subject (bedpostx_gpu)

**Probtrackx2** (tractography):
- CPU: 2-6 hours per subject (5000 samples, 200 ROIs)
- GPU: 20-60 minutes per subject (probtrackx2_gpu)

### Requirements

- **FSL 6.0+**: BEDPOSTX, probtrackx2 (probtrackx2_gpu optional)
- **CUDA** (optional): For GPU acceleration
- **FreeSurfer** (optional): For Desikan-Killiany atlas and GMWMI seeding

---

## References

- **Native-space connectivity**: Power et al. (2014) NeuroImage
- **Fisher z-transformation**: Fisher (1915) Biometrika
- **Graph theory**: Rubinov & Sporns (2010) NeuroImage
- **Probabilistic tractography**: Behrens et al. (2007) NeuroImage
- **ACT tractography**: Smith et al. (2012) NeuroImage
