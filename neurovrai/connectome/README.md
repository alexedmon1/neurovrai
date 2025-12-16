# neurovrai.connectome

**Connectivity analysis and network neuroscience.**

## Overview

This module provides tools for building and analyzing brain connectomes:

| Component | Description | Status |
|-----------|-------------|--------|
| **ROI Extraction** | Timeseries/values from atlas parcellations | Production |
| **Functional Connectivity** | Correlation-based FC matrices | Production |
| **Structural Connectivity** | Tractography-based SC matrices | Production |
| **Graph Metrics** | Network topology analysis | Production |
| **Group Analysis** | NBS, group comparisons | Production |
| **Visualization** | Matrices, connectograms | Production |

## Module Structure

```
connectome/
├── roi_extraction.py              # Atlas-based ROI extraction
├── functional_connectivity.py     # FC matrix computation
├── structural_connectivity.py     # SC with probtrackx2
├── graph_metrics.py               # Network topology
├── group_analysis.py              # NBS, group statistics
├── visualization.py               # Matrix/network plots
│
├── atlas_func_transform.py        # Atlas→functional space
├── atlas_dwi_transform.py         # Atlas→DWI space
├── atlas_labels.py                # Atlas label management
│
├── batch_functional_connectivity.py   # FC batch processing
├── batch_structural_connectivity.py   # SC batch processing
├── batch_graph_metrics.py             # Graph metrics batch
├── batch_group_statistics.py          # Group stats batch
├── batch_visualization.py             # Visualization batch
│
├── run_functional_connectivity.py     # FC CLI
└── run_structural_connectivity.py     # SC CLI
```

---

## Quick Start

### Functional Connectivity (Batch)

```bash
uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root /mnt/data/my_study \
    --atlases harvardoxford_cort schaefer_200 \
    --output-dir /mnt/data/my_study/analysis/connectivity
```

### Structural Connectivity (Single Subject)

```bash
uv run python -m neurovrai.connectome.run_structural_connectivity \
    --subject sub-001 \
    --derivatives-dir /mnt/data/my_study/derivatives \
    --atlas schaefer_200 \
    --config config.yaml
```

---

## Functional Connectivity

### Native-Space Analysis

This pipeline uses **native-space functional connectivity**:
- Functional data stays in native space (no MNI normalization)
- Atlases are resampled to match functional dimensions
- No BBR or anatomical transforms required

**Advantages**: Simple, fast (~30-60 sec/subject-atlas), preserves functional resolution

### Python API

```python
from neurovrai.connectome import (
    extract_roi_timeseries,
    compute_functional_connectivity
)
from pathlib import Path

# Extract timeseries
timeseries, roi_names = extract_roi_timeseries(
    data_file=Path('preprocessed_bold.nii.gz'),
    atlas=Path('schaefer_400.nii.gz'),
    mask_file=Path('brain_mask.nii.gz'),
    min_voxels=10
)

# Compute connectivity
fc_results = compute_functional_connectivity(
    timeseries=timeseries,
    roi_names=roi_names,
    method='pearson',      # or 'spearman'
    fisher_z=True,
    output_dir=Path('fc_output')
)

# Access results
fc_matrix = fc_results['fc_matrix']
summary = fc_results['summary']
```

### Available Atlases

**MNI Space** (resampled to functional):

| Atlas | Regions | Description |
|-------|---------|-------------|
| `harvardoxford_cort` | 48 | Harvard-Oxford Cortical |
| `harvardoxford_sub` | 21 | Harvard-Oxford Subcortical |
| `juelich` | Variable | Juelich Histological |
| `schaefer_100/200/400` | 100/200/400 | Schaefer Parcellation |

**FreeSurfer** (require `--fs-subjects-dir`):

| Atlas | Regions | Description |
|-------|---------|-------------|
| `desikan_killiany` | ~85 | Desikan-Killiany cortical + subcortical |
| `destrieux` | ~165 | Destrieux cortical + subcortical |

### Outputs

```
connectivity/{subject}/{atlas}/
├── fc_matrix.npy              # Connectivity matrix (NumPy)
├── fc_matrix.csv              # Connectivity matrix (CSV)
├── fc_roi_names.txt           # ROI labels
├── fc_summary.json            # Summary statistics
├── atlas_resampled.nii.gz     # Atlas in functional space
└── analysis_metadata.json     # Processing metadata
```

---

## Structural Connectivity

### Pipeline

1. **BEDPOSTX**: Fiber orientation modeling (GPU-accelerated)
2. **Atlas Transform**: MNI/FreeSurfer → DWI native space
3. **Anatomical Constraints**: Ventricle avoidance, WM masks
4. **Probtrackx2**: Network-mode tractography
5. **Matrix Construction**: Waytotal-normalized connectivity

### Python API

```python
from neurovrai.connectome import compute_structural_connectivity
from pathlib import Path

sc_results = compute_structural_connectivity(
    bedpostx_dir=Path('dwi.bedpostX'),
    atlas_file=Path('schaefer_200_dwi.nii.gz'),
    output_dir=Path('sc_output'),
    n_samples=5000,
    avoid_ventricles=True,
    config=config
)

sc_matrix = sc_results['connectivity_matrix']
roi_names = sc_results['roi_names']
```

### Configuration

```yaml
structural_connectivity:
  tractography:
    use_gpu: false  # Default: CPU for stability. Set true for speed with small atlases (<50 ROIs)
    n_samples: 5000
    step_length: 0.5
    curvature_threshold: 0.2

  anatomical_constraints:
    avoid_ventricles: true
    use_wm_mask: true
    terminate_at_gm: false

  freesurfer_options:
    use_gmwmi_seeding: true
    gmwmi_method: surface
```

### Outputs

```
connectome/structural/{subject}/
├── sc_matrix.npy              # Connectivity matrix
├── sc_matrix.csv              # CSV format
├── sc_roi_names.txt           # ROI labels
├── sc_summary.json            # Summary statistics
├── graph_metrics.json         # Network metrics
├── analysis_metadata.json     # Processing metadata
└── probtrackx_output/
    └── fdt_network_matrix     # Raw tractography
```

---

## Graph Metrics

Network topology analysis using NetworkX.

### Python API

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

# Global metrics
global_metrics = compute_global_metrics(
    matrix=fc_matrix,
    threshold=0.3
)

print(f"Global efficiency: {global_metrics['global_efficiency']:.4f}")
print(f"Path length: {global_metrics['characteristic_path_length']:.4f}")
print(f"Transitivity: {global_metrics['transitivity']:.4f}")

# Identify hubs
hubs = identify_hubs(
    node_metrics,
    method='betweenness',  # or 'degree'
    percentile=90
)
```

### Available Metrics

**Node-level**:
- Degree (weighted/unweighted)
- Clustering coefficient
- Betweenness centrality
- Eigenvector centrality
- Local efficiency
- Participation coefficient

**Global**:
- Global efficiency
- Characteristic path length
- Clustering coefficient
- Transitivity
- Modularity
- Small-worldness

---

## Group Analysis

### Network-Based Statistic (NBS)

Permutation-based network-level inference for group comparisons.

```python
from neurovrai.connectome import compute_network_based_statistic

nbs_results = compute_network_based_statistic(
    group1_matrices,  # Shape: (n_subjects, n_rois, n_rois)
    group2_matrices,
    threshold=3.0,     # t-statistic threshold
    n_permutations=5000,
    alpha=0.05
)

print(f"Significant components: {nbs_results['n_significant']}")
for size, pval in zip(nbs_results['component_sizes'], nbs_results['component_pvals']):
    print(f"  {size} edges, p={pval:.4f}")
```

### Group Statistics

```python
from neurovrai.connectome import (
    load_connectivity_matrices,
    average_connectivity_matrices,
    compute_group_difference
)

# Load matrices for all subjects
matrices, subjects = load_connectivity_matrices(
    connectivity_dir=Path('connectivity'),
    atlas='schaefer_200'
)

# Group average
group_mean = average_connectivity_matrices(matrices)

# Group difference
t_matrix, p_matrix = compute_group_difference(
    group1_matrices=patients,
    group2_matrices=controls,
    method='ttest'
)
```

---

## Visualization

```python
from neurovrai.connectome import (
    plot_connectivity_matrix,
    plot_circular_connectogram,
    plot_connectivity_comparison
)

# Matrix heatmap
plot_connectivity_matrix(
    matrix=fc_matrix,
    roi_names=roi_names,
    title='Functional Connectivity',
    output_file='fc_matrix.png'
)

# Circular connectogram
plot_circular_connectogram(
    matrix=fc_matrix,
    roi_names=roi_names,
    threshold=0.3,
    output_file='connectogram.png'
)

# Group comparison
plot_connectivity_comparison(
    matrix1=patients_mean,
    matrix2=controls_mean,
    roi_names=roi_names,
    title='Patients vs Controls',
    output_file='comparison.png'
)
```

---

## Batch Processing

### Functional Connectivity

```bash
uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root /mnt/data/study \
    --atlases all \                    # or specific atlases
    --subjects sub-001 sub-002 \       # or all subjects
    --method pearson \
    --output-dir /mnt/data/study/analysis/connectivity
```

### Structural Connectivity

```bash
uv run python -m neurovrai.connectome.batch_structural_connectivity \
    --study-root /mnt/data/study \
    --atlases schaefer_200 schaefer_400 \
    --config config.yaml \
    --output-dir /mnt/data/study/connectome/structural
```

### Graph Metrics

```bash
uv run python -m neurovrai.connectome.batch_graph_metrics \
    --connectivity-dir /mnt/data/study/analysis/connectivity \
    --atlas schaefer_200 \
    --threshold 0.3 \
    --output-dir /mnt/data/study/analysis/graph_metrics
```

---

## Performance

| Process | Time per Subject |
|---------|------------------|
| FC (per atlas) | 30-60 sec |
| BEDPOSTX (CPU) | 4-8 hours |
| BEDPOSTX (GPU) | 20-60 min |
| Probtrackx2 (CPU) | 2-6 hours |
| Probtrackx2 (GPU) | 20-60 min |
| Graph metrics | <1 sec |

---

## Public API

```python
from neurovrai.connectome import (
    # ROI extraction
    extract_roi_timeseries,
    extract_roi_values,
    load_atlas,

    # Functional connectivity
    compute_functional_connectivity,
    compute_correlation_matrix,
    compute_partial_correlation_matrix,
    fisher_z_transform,
    inverse_fisher_z_transform,
    threshold_matrix,
    compute_seed_connectivity,

    # Structural connectivity
    compute_structural_connectivity,
    run_bedpostx,
    validate_bedpostx_outputs,
    prepare_atlas_for_probtrackx,
    run_probtrackx2_network,
    construct_connectivity_matrix,

    # Visualization
    plot_connectivity_matrix,
    plot_circular_connectogram,
    plot_connectivity_comparison,

    # Group analysis
    load_connectivity_matrices,
    average_connectivity_matrices,
    compute_group_difference,
    compute_network_based_statistic,
    filter_subjects_by_demographics,

    # Graph metrics
    compute_node_metrics,
    compute_global_metrics,
    identify_hubs,
    threshold_and_binarize,
    matrix_to_graph,
)
```

---

## References

- **Native-space connectivity**: Power et al. (2014) NeuroImage
- **Fisher z-transformation**: Fisher (1915) Biometrika
- **Graph theory**: Rubinov & Sporns (2010) NeuroImage
- **NBS**: Zalesky et al. (2010) NeuroImage
- **Probabilistic tractography**: Behrens et al. (2007) NeuroImage
- **ACT tractography**: Smith et al. (2012) NeuroImage
