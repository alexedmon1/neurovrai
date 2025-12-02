"""
Connectome Analysis Module

This module provides tools for building and analyzing brain connectomes from
neuroimaging data, including both functional and structural connectivity.

Three-Pathway Architecture
--------------------------
1. ROI Extraction (roi_extraction.py)
   - Modality-agnostic ROI extraction from atlas parcellations
   - Support for functional, anatomical, and diffusion data
   - Atlas registration and label management

2. Functional Connectivity (functional_connectivity.py)
   - Timeseries extraction from fMRI data
   - ROI-to-ROI correlation matrices
   - Fisher z-transformation for statistical analysis

3. Structural Connectivity (structural_connectivity.py)
   - Probabilistic tractography with probtrackx2
   - BEDPOSTX integration for fiber orientation modeling
   - ROI-to-ROI tractography-based connectivity matrices
   - Network construction from tractography results

Usage Examples
--------------
# ROI extraction from functional data
timeseries = extract_roi_timeseries(
    func_file='preprocessed_bold.nii.gz',
    atlas_file='schaefer_400.nii.gz',
    mask_file='brain_mask.nii.gz'
)

# Functional connectivity matrix
fc_matrix = compute_functional_connectivity(
    timeseries=timeseries,
    method='pearson',
    fisher_z=True
)

# Structural connectivity from probtrackx2
sc_results = compute_structural_connectivity(
    bedpostx_dir='derivatives/sub-001/dwi.bedpostX',
    atlas_file='parcellations/schaefer_400_dwi.nii.gz',
    output_dir='connectome/structural/sub-001',
    n_samples=5000
)
sc_matrix = sc_results['connectivity_matrix']
"""

from neurovrai.connectome.roi_extraction import (
    extract_roi_timeseries,
    extract_roi_values,
    load_atlas,
)

from neurovrai.connectome.functional_connectivity import (
    compute_functional_connectivity,
    compute_correlation_matrix,
    compute_partial_correlation_matrix,
    fisher_z_transform,
    inverse_fisher_z_transform,
    threshold_matrix,
    compute_seed_connectivity,
)

from neurovrai.connectome.structural_connectivity import (
    compute_structural_connectivity,
    run_bedpostx,
    validate_bedpostx_outputs,
    prepare_atlas_for_probtrackx,
    run_probtrackx2_network,
    construct_connectivity_matrix,
    StructuralConnectivityError,
)

from neurovrai.connectome.visualization import (
    plot_connectivity_matrix,
    plot_circular_connectogram,
    plot_connectivity_comparison,
)

from neurovrai.connectome.group_analysis import (
    load_connectivity_matrices,
    average_connectivity_matrices,
    compute_group_difference,
    compute_network_based_statistic,
    filter_subjects_by_demographics,
)

from neurovrai.connectome.graph_metrics import (
    compute_node_metrics,
    compute_global_metrics,
    identify_hubs,
    threshold_and_binarize,
    matrix_to_graph,
)

__all__ = [
    # ROI extraction
    'extract_roi_timeseries',
    'extract_roi_values',
    'load_atlas',

    # Functional connectivity
    'compute_functional_connectivity',
    'compute_correlation_matrix',
    'compute_partial_correlation_matrix',
    'fisher_z_transform',
    'inverse_fisher_z_transform',
    'threshold_matrix',
    'compute_seed_connectivity',

    # Structural connectivity
    'compute_structural_connectivity',
    'run_bedpostx',
    'validate_bedpostx_outputs',
    'prepare_atlas_for_probtrackx',
    'run_probtrackx2_network',
    'construct_connectivity_matrix',
    'StructuralConnectivityError',

    # Visualization
    'plot_connectivity_matrix',
    'plot_circular_connectogram',
    'plot_connectivity_comparison',

    # Group analysis
    'load_connectivity_matrices',
    'average_connectivity_matrices',
    'compute_group_difference',
    'compute_network_based_statistic',
    'filter_subjects_by_demographics',

    # Graph theory metrics
    'compute_node_metrics',
    'compute_global_metrics',
    'identify_hubs',
    'threshold_and_binarize',
    'matrix_to_graph',
]

__version__ = '0.1.0'
