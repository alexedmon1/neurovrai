#!/usr/bin/env python3
"""
Complete Connectome Analysis Workflow Example

This script demonstrates the complete workflow for brain connectivity analysis,
including:
1. ROI extraction from functional MRI data
2. Functional connectivity matrix computation
3. Group-level analysis
4. Graph theory metrics
5. Network-Based Statistic (NBS)
6. Visualization

Usage:
    uv run python examples/connectome_complete_workflow.py

Note: This example uses synthetic data for demonstration purposes.
Replace with real preprocessed fMRI data for actual analysis.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurovrai.connectome import (
    # ROI extraction
    extract_roi_timeseries,
    load_atlas,

    # Functional connectivity
    compute_functional_connectivity,
    fisher_z_transform,

    # Group analysis
    average_connectivity_matrices,
    compute_group_difference,
    compute_network_based_statistic,

    # Graph theory metrics
    compute_node_metrics,
    compute_global_metrics,
    identify_hubs,

    # Visualization
    plot_connectivity_matrix,
    plot_circular_connectogram,
    plot_connectivity_comparison,
)


def create_synthetic_fmri_data(n_volumes=200, n_rois=50, tr=2.0):
    """
    Create synthetic fMRI data for demonstration

    In practice, use preprocessed fMRI data
    """
    print(f"\nGenerating synthetic fMRI data:")
    print(f"  Volumes: {n_volumes}")
    print(f"  ROIs: {n_rois}")
    print(f"  TR: {tr}s")

    # Create random timeseries with some correlation structure
    base_signal = np.random.randn(n_volumes)
    timeseries = np.zeros((n_volumes, n_rois))

    for i in range(n_rois):
        # Mix base signal with independent noise
        mixing = 0.3 + np.random.rand() * 0.4
        timeseries[:, i] = mixing * base_signal + (1 - mixing) * np.random.randn(n_volumes)

    # Add some ROI-specific structure
    for i in range(0, n_rois, 5):
        # Create mini-networks
        group_signal = np.random.randn(n_volumes)
        for j in range(i, min(i+5, n_rois)):
            timeseries[:, j] += 0.5 * group_signal

    roi_names = [f"ROI_{i:02d}" for i in range(n_rois)]

    return timeseries, roi_names


def demonstrate_functional_connectivity(timeseries, roi_names, output_dir):
    """Demonstrate functional connectivity analysis"""
    print("\n" + "=" * 80)
    print("STEP 1: Functional Connectivity Analysis")
    print("=" * 80)

    fc_dir = output_dir / "functional_connectivity"
    fc_dir.mkdir(exist_ok=True)

    # Compute connectivity matrix
    print("\nComputing functional connectivity matrix...")
    fc_results = compute_functional_connectivity(
        timeseries=timeseries,
        roi_names=roi_names,
        method='pearson',
        fisher_z=True,
        output_dir=fc_dir
    )

    print(f"  Matrix shape: {fc_results['fc_matrix'].shape}")
    print(f"  Mean correlation: {fc_results['fc_matrix'].mean():.3f}")
    print(f"  Connection density: {np.sum(np.abs(fc_results['fc_matrix']) > 0.3) / fc_results['fc_matrix'].size:.3f}")

    # Visualize
    print("\nGenerating visualizations...")
    plot_connectivity_matrix(
        fc_results['fc_matrix'],
        roi_names,
        fc_dir / "connectivity_heatmap.png",
        title="Functional Connectivity Matrix",
        cluster=True
    )

    plot_circular_connectogram(
        fc_results['fc_matrix'],
        roi_names,
        fc_dir / "connectogram.png",
        threshold=0.5,
        title="Functional Connectogram (r > 0.5)"
    )

    print(f"  Saved visualizations to: {fc_dir}")

    return fc_results['fc_matrix']


def demonstrate_group_analysis(n_subjects=20, n_rois=30, output_dir=None):
    """Demonstrate group-level connectivity analysis"""
    print("\n" + "=" * 80)
    print("STEP 2: Group-Level Analysis")
    print("=" * 80)

    group_dir = output_dir / "group_analysis"
    group_dir.mkdir(exist_ok=True)

    # Create two groups with synthetic data
    print(f"\nCreating synthetic data for {n_subjects} subjects per group...")

    group1_matrices = []
    group2_matrices = []

    for i in range(n_subjects):
        # Group 1: Normal connectivity
        ts1, _ = create_synthetic_fmri_data(n_volumes=200, n_rois=n_rois)
        fc1 = np.corrcoef(ts1.T)
        np.fill_diagonal(fc1, 0)
        fc1_z = fisher_z_transform(fc1)
        group1_matrices.append(fc1_z)

        # Group 2: Enhanced connectivity in specific regions
        ts2, _ = create_synthetic_fmri_data(n_volumes=200, n_rois=n_rois)
        fc2 = np.corrcoef(ts2.T)
        # Boost connectivity in subnetwork (ROIs 10-15)
        fc2[10:16, 10:16] += 0.3
        fc2 = (fc2 + fc2.T) / 2
        np.fill_diagonal(fc2, 0)
        fc2_z = fisher_z_transform(fc2)
        group2_matrices.append(fc2_z)

    group1_matrices = np.stack(group1_matrices)
    group2_matrices = np.stack(group2_matrices)

    # Average connectivity
    print("\nComputing group averages...")
    avg_results = average_connectivity_matrices(
        group1_matrices,
        consistency_threshold=0.7,
        output_dir=group_dir / "group1_average"
    )

    print(f"  Group 1 mean connectivity: {avg_results['mean_matrix'].mean():.3f}")
    print(f"  Consistent edges: {avg_results['n_consistent_edges']}")

    # Group difference
    print("\nComputing group differences...")
    diff_results = compute_group_difference(
        group1_matrices,
        group2_matrices,
        group1_name="Controls",
        group2_name="Patients",
        output_dir=group_dir / "group_difference"
    )

    print(f"  Significant edges (FDR corrected): {diff_results['n_significant']}")

    # Visualize comparison
    roi_names = [f"ROI_{i:02d}" for i in range(n_rois)]
    plot_connectivity_comparison(
        diff_results['group1_mean'],
        diff_results['group2_mean'],
        roi_names,
        group_dir / "group_comparison.png",
        "Controls", "Patients"
    )

    print(f"  Saved group analysis results to: {group_dir}")

    return group1_matrices, group2_matrices, roi_names


def demonstrate_graph_metrics(fc_matrix, roi_names, output_dir):
    """Demonstrate graph theory metrics"""
    print("\n" + "=" * 80)
    print("STEP 3: Graph Theory Metrics")
    print("=" * 80)

    graph_dir = output_dir / "graph_metrics"
    graph_dir.mkdir(exist_ok=True)

    # Compute node metrics
    print("\nComputing node-level metrics...")
    node_metrics = compute_node_metrics(
        fc_matrix,
        threshold=0.3,
        weighted=True,
        roi_names=roi_names
    )

    print(f"  Mean degree: {node_metrics['degree'].mean():.2f}")
    print(f"  Mean strength: {node_metrics['strength'].mean():.2f}")
    print(f"  Mean clustering coefficient: {node_metrics['clustering_coefficient'].mean():.3f}")
    print(f"  Mean betweenness centrality: {node_metrics['betweenness_centrality'].mean():.3f}")

    # Identify hubs
    hubs_degree = identify_hubs(node_metrics, method='degree', percentile=90)
    hubs_betweenness = identify_hubs(node_metrics, method='betweenness', percentile=90)

    print(f"\nHub nodes (90th percentile):")
    print(f"  Degree-based hubs: {np.sum(hubs_degree)}")
    print(f"  Betweenness-based hubs: {np.sum(hubs_betweenness)}")

    # Compute global metrics
    print("\nComputing global network metrics...")
    global_metrics = compute_global_metrics(fc_matrix, threshold=0.3)

    print(f"  Global efficiency: {global_metrics['global_efficiency']:.4f}")
    print(f"  Characteristic path length: {global_metrics['characteristic_path_length']:.4f}")
    print(f"  Transitivity: {global_metrics['transitivity']:.4f}")

    # Small-world analysis
    print(f"\nSmall-world properties:")
    print(f"  High clustering + short path length = small-world network")
    print(f"  (Compare to random networks for formal analysis)")

    print(f"\n  Saved graph metrics to: {graph_dir}")

    return node_metrics, global_metrics


def demonstrate_nbs(group1_matrices, group2_matrices, roi_names, output_dir):
    """Demonstrate Network-Based Statistic"""
    print("\n" + "=" * 80)
    print("STEP 4: Network-Based Statistic (NBS)")
    print("=" * 80)

    nbs_dir = output_dir / "nbs"
    nbs_dir.mkdir(exist_ok=True)

    print("\nRunning NBS analysis...")
    print("  Note: Using 100 permutations for speed (use 5000+ for publication)")

    nbs_results = compute_network_based_statistic(
        group1_matrices,
        group2_matrices,
        threshold=2.5,
        n_permutations=100,  # Use more for real analysis
        alpha=0.05,
        output_dir=nbs_dir
    )

    print(f"\nNBS Results:")
    print(f"  Components found: {nbs_results['n_components']}")
    print(f"  Significant components: {nbs_results['n_significant']}")

    if nbs_results['n_components'] > 0:
        for i, (size, pval) in enumerate(zip(
            nbs_results['component_sizes'],
            nbs_results['component_pvals']
        )):
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            print(f"  Component {i+1}: {size} edges, p={pval:.4f} {sig}")

    # Visualize significant components
    if nbs_results['n_significant'] > 0:
        sig_matrix = nbs_results['test_statistic'] * nbs_results['sig_edges']
        plot_connectivity_matrix(
            sig_matrix,
            roi_names,
            nbs_dir / "nbs_significant_edges.png",
            title="NBS Significant Network Differences"
        )
        print(f"\n  Saved NBS results to: {nbs_dir}")

    return nbs_results


def main():
    print("=" * 80)
    print("COMPLETE CONNECTOME ANALYSIS WORKFLOW")
    print("=" * 80)
    print("\nThis example demonstrates:")
    print("  1. Functional connectivity matrix computation")
    print("  2. Group-level connectivity analysis")
    print("  3. Graph theory network metrics")
    print("  4. Network-Based Statistic (NBS)")
    print("  5. Comprehensive visualizations")

    # Create output directory
    output_dir = Path(tempfile.mkdtemp(prefix="connectome_workflow_"))
    print(f"\nOutput directory: {output_dir}")

    # Step 1: Functional Connectivity
    timeseries, roi_names = create_synthetic_fmri_data(n_volumes=200, n_rois=50)
    fc_matrix = demonstrate_functional_connectivity(timeseries, roi_names, output_dir)

    # Step 2: Group Analysis
    group1_matrices, group2_matrices, group_roi_names = demonstrate_group_analysis(
        n_subjects=15,
        n_rois=30,
        output_dir=output_dir
    )

    # Step 3: Graph Metrics
    node_metrics, global_metrics = demonstrate_graph_metrics(
        fc_matrix,
        roi_names,
        output_dir
    )

    # Step 4: Network-Based Statistic
    nbs_results = demonstrate_nbs(
        group1_matrices,
        group2_matrices,
        group_roi_names,
        output_dir
    )

    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated outputs:")
    print("  functional_connectivity/")
    print("    - connectivity_heatmap.png")
    print("    - connectogram.png")
    print("    - fc_matrix.npy")
    print("  group_analysis/")
    print("    - group_comparison.png")
    print("    - group_difference/")
    print("  graph_metrics/")
    print("  nbs/")
    print("    - nbs_significant_edges.png")
    print("    - nbs_components.json")

    print("\nNext steps for real data:")
    print("  1. Replace synthetic data with preprocessed fMRI")
    print("  2. Use actual brain atlas parcellation")
    print("  3. Increase NBS permutations to 5000+")
    print("  4. Compare graph metrics to random networks")
    print("  5. Validate findings with clinical/behavioral data")

    return 0


if __name__ == '__main__':
    sys.exit(main())
