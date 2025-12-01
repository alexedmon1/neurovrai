# Connectome Analysis Module

Brain connectome construction and analysis from neuroimaging data.

## Architecture

The connectome module follows a **three-pathway design** with comprehensive group analysis and visualization capabilities:

### 1. ROI Extraction (`roi_extraction.py`)
**Modality-agnostic** extraction of regional data from atlas parcellations.

**Features**:
- Load discrete (3D) or probabilistic (4D) atlases
- Extract timeseries from 4D functional data
- Extract statistics from 3D volumes (FA, MD, GM density, etc.)
- Support for weighted extraction (probabilistic atlases)
- Atlas resampling to match data space

**Usage**:
```python
from neurovrai.connectome import extract_roi_timeseries, extract_roi_values

# Extract functional timeseries
timeseries, roi_names = extract_roi_timeseries(
    data_file='preprocessed_bold.nii.gz',
    atlas='schaefer_400.nii.gz',
    mask_file='brain_mask.nii.gz'
)

# Extract anatomical values
roi_values, voxel_counts = extract_roi_values(
    data_file='FA.nii.gz',
    atlas='JHU-ICBM-labels-2mm.nii.gz',
    statistic='mean'
)
```

### 2. Functional Connectivity (`functional_connectivity.py`)
Compute functional connectivity matrices from fMRI timeseries.

**Features**:
- Pearson and Spearman correlation
- Partial correlation (precision-based)
- Fisher z-transformation for statistical analysis
- Matrix thresholding and sparsification
- Seed-based connectivity
- Comprehensive output formats (numpy, CSV, JSON)

**Usage**:
```python
from neurovrai.connectome import compute_functional_connectivity

fc_results = compute_functional_connectivity(
    timeseries=timeseries,
    roi_names=roi_names,
    method='pearson',
    fisher_z=True,
    output_dir='fc_output/'
)
```

### 3. Structural Connectivity (`structural_connectivity.py`)
**[Future implementation]** - Tractography-based structural connectivity using probtrackx2.

## Group-Level Analysis

### Group Averaging (`group_analysis.py`)
Average connectivity matrices across subjects with consistency filtering.

**Features**:
- Mean and standard error computation
- Consistency thresholding (keep edges present in X% of subjects)
- Subject filtering by demographics
- Comprehensive statistical outputs

**Usage**:
```python
from neurovrai.connectome import average_connectivity_matrices

results = average_connectivity_matrices(
    matrices,  # Shape: (n_subjects, n_rois, n_rois)
    consistency_threshold=0.7,  # Keep edges in 70%+ of subjects
    output_dir='group_average/'
)
```

### Group Differences
Statistical comparison between groups with FDR correction.

**Usage**:
```python
from neurovrai.connectome import compute_group_difference

diff_results = compute_group_difference(
    group1_matrices,
    group2_matrices,
    group1_name="Controls",
    group2_name="Patients",
    paired=False,
    alpha=0.05,
    output_dir='group_diff/'
)
```

### Network-Based Statistic (NBS)
Permutation-based family-wise error correction for network-level inference.

**Features**:
- Connected component detection
- Permutation testing (1000-5000+ permutations)
- Component-wise p-values
- Automated output saving (matrices, distributions, JSON)

**Usage**:
```python
from neurovrai.connectome import compute_network_based_statistic

nbs_results = compute_network_based_statistic(
    group1_matrices,
    group2_matrices,
    threshold=3.0,  # t-statistic threshold
    n_permutations=5000,
    alpha=0.05,
    output_dir='nbs_output/'
)

print(f"Significant components: {nbs_results['n_significant']}")
for i, (size, pval) in enumerate(zip(
    nbs_results['component_sizes'],
    nbs_results['component_pvals']
)):
    print(f"Component {i+1}: {size} edges, p={pval:.4f}")
```

## Graph Theory Metrics

### Node-Level Metrics (`graph_metrics.py`)
Compute network metrics for individual nodes.

**Features**:
- Degree and strength
- Clustering coefficient
- Betweenness centrality
- Hub identification

**Usage**:
```python
from neurovrai.connectome import compute_node_metrics, identify_hubs

node_metrics = compute_node_metrics(
    matrix,
    threshold=0.3,
    weighted=True,
    roi_names=roi_names
)

# Identify hub nodes
hubs = identify_hubs(
    node_metrics,
    method='betweenness',  # or 'degree'
    percentile=90
)
```

### Global Network Metrics
Compute network-wide properties.

**Features**:
- Global efficiency
- Characteristic path length
- Transitivity (global clustering)
- Small-world analysis

**Usage**:
```python
from neurovrai.connectome import compute_global_metrics

global_metrics = compute_global_metrics(
    matrix,
    threshold=0.3
)

print(f"Global efficiency: {global_metrics['global_efficiency']:.4f}")
print(f"Path length: {global_metrics['characteristic_path_length']:.4f}")
print(f"Transitivity: {global_metrics['transitivity']:.4f}")
```

## Visualization

### Connectivity Matrices
Heatmap visualization with optional hierarchical clustering.

**Usage**:
```python
from neurovrai.connectome import plot_connectivity_matrix

plot_connectivity_matrix(
    matrix,
    roi_names,
    output_file='connectivity_heatmap.png',
    title='Functional Connectivity',
    cluster=True,  # Apply hierarchical clustering
    cmap='RdBu_r'
)
```

### Circular Connectograms
Network visualization with nodes arranged in a circle.

**Usage**:
```python
from neurovrai.connectome import plot_circular_connectogram

plot_circular_connectogram(
    matrix,
    roi_names,
    output_file='connectogram.png',
    threshold=0.5,  # Only show strong connections
    title='Brain Network'
)
```

### Group Comparisons
Side-by-side visualization of two connectivity matrices.

**Usage**:
```python
from neurovrai.connectome import plot_connectivity_comparison

plot_connectivity_comparison(
    group1_mean,
    group2_mean,
    roi_names,
    output_file='group_comparison.png',
    title1='Controls',
    title2='Patients'
)
```

## Command-Line Interface

```bash
# Basic functional connectivity analysis
python -m neurovrai.connectome.run_functional_connectivity \
    --func-file preprocessed_bold.nii.gz \
    --atlas schaefer_400.nii.gz \
    --output-dir /analysis/fc/subject-001/

# With all options
python -m neurovrai.connectome.run_functional_connectivity \
    --func-file preprocessed_bold.nii.gz \
    --atlas schaefer_400.nii.gz \
    --labels schaefer_400_labels.txt \
    --mask brain_mask.nii.gz \
    --method pearson \
    --fisher-z \
    --threshold 0.3 \
    --output-dir /analysis/fc/subject-001/
```

## Complete Workflow Example

See `examples/connectome_complete_workflow.py` for an end-to-end demonstration including:
1. ROI timeseries extraction
2. Functional connectivity matrix computation
3. Group-level analysis
4. Graph theory metrics
5. Network-Based Statistic
6. Comprehensive visualizations

```bash
uv run python examples/connectome_complete_workflow.py
```

## Status

- ✅ ROI extraction: **Production-ready**
- ✅ Functional connectivity: **Production-ready**
- ✅ Group analysis: **Production-ready**
- ✅ Graph theory metrics: **Production-ready**
- ✅ Network-Based Statistic: **Production-ready**
- ✅ Visualization: **Production-ready**
- ⏳ Structural connectivity: **Planned for future development**

## References

**Network-Based Statistic**:
- Zalesky A, Fornito A, Bullmore ET (2010). Network-based statistic: identifying differences in brain networks. NeuroImage, 53(4):1197-1207.

**Graph Theory Metrics**:
- Rubinov M, Sporns O (2010). Complex network measures of brain connectivity: uses and interpretations. NeuroImage, 52(3):1059-1069.
