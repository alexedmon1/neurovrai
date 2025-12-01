#!/usr/bin/env python3
"""
Graph Metrics and Network-Based Statistic Test Script

Tests graph theory metrics and NBS functionality.

Usage:
    uv run python archive/tests/test_graph_metrics_nbs.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurovrai.connectome import (
    compute_node_metrics,
    compute_global_metrics,
    identify_hubs,
    threshold_and_binarize,
    compute_network_based_statistic,
    plot_connectivity_matrix,
)


def create_synthetic_network(n_nodes=50, density=0.2, mean_weight=0.5, std_weight=0.2):
    """Create synthetic connectivity matrix"""
    matrix = np.random.randn(n_nodes, n_nodes) * std_weight + mean_weight
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)

    # Apply density threshold
    n_edges_target = int(density * n_nodes * (n_nodes - 1) / 2)
    flat_abs = np.abs(matrix[np.triu_indices(n_nodes, k=1)])
    threshold_val = np.sort(flat_abs)[-n_edges_target]
    matrix[np.abs(matrix) < threshold_val] = 0

    return matrix


def test_thresholding(output_dir):
    """Test matrix thresholding and binarization"""
    print("\n" + "=" * 80)
    print("TEST: Matrix Thresholding")
    print("=" * 80)
    try:
        n_nodes = 20
        matrix = np.random.randn(n_nodes, n_nodes) * 0.3 + 0.5
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)

        print(f"Original matrix: {matrix.shape}")
        print(f"  Non-zero edges: {np.sum(matrix != 0) / 2}")

        # Test absolute threshold
        thresh_matrix = threshold_and_binarize(matrix, threshold=0.5, binarize=True)
        print(f"\nThreshold=0.5, binarized:")
        print(f"  Non-zero edges: {np.sum(thresh_matrix != 0) / 2}")

        # Test density threshold
        density_matrix = threshold_and_binarize(matrix, density=0.1, binarize=False)
        print(f"\nDensity=0.1, weighted:")
        print(f"  Non-zero edges: {np.sum(density_matrix != 0) / 2}")
        print(f"  Density: {np.sum(density_matrix != 0) / matrix.size:.3f}")

        print("  OK Thresholding tests passed")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_metrics(output_dir):
    """Test node-level metrics"""
    print("\n" + "=" * 80)
    print("TEST: Node-Level Metrics")
    print("=" * 80)
    try:
        n_nodes = 30
        matrix = create_synthetic_network(n_nodes, density=0.15, mean_weight=0.6)

        print(f"Network: {n_nodes} nodes")
        print(f"  Edges: {np.sum(matrix != 0) / 2}")

        # Binary metrics
        print("\nBinary metrics:")
        binary_metrics = compute_node_metrics(matrix, threshold=0.3, weighted=False)
        print(f"  Mean degree: {binary_metrics['degree'].mean():.2f}")
        print(f"  Mean clustering: {binary_metrics['clustering_coefficient'].mean():.3f}")
        print(f"  Mean betweenness: {binary_metrics['betweenness_centrality'].mean():.3f}")

        # Weighted metrics
        print("\nWeighted metrics:")
        weighted_metrics = compute_node_metrics(matrix, threshold=0.3, weighted=True)
        print(f"  Mean strength: {weighted_metrics['strength'].mean():.2f}")
        print(f"  Mean clustering: {weighted_metrics['clustering_coefficient'].mean():.3f}")

        # Hub identification
        hubs_degree = identify_hubs(binary_metrics, method='degree', percentile=90)
        hubs_between = identify_hubs(binary_metrics, method='betweenness', percentile=90)
        print(f"\nHub nodes (90th percentile):")
        print(f"  Degree hubs: {np.sum(hubs_degree)}")
        print(f"  Betweenness hubs: {np.sum(hubs_between)}")

        # Visualize degree distribution
        viz_dir = output_dir / "node_metrics"
        viz_dir.mkdir(exist_ok=True)

        # Create visualization of metrics
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].bar(range(n_nodes), binary_metrics['degree'])
        axes[0, 0].set_title('Node Degree')
        axes[0, 0].set_xlabel('Node')
        axes[0, 0].set_ylabel('Degree')

        axes[0, 1].bar(range(n_nodes), weighted_metrics['strength'])
        axes[0, 1].set_title('Node Strength')
        axes[0, 1].set_xlabel('Node')
        axes[0, 1].set_ylabel('Strength')

        axes[1, 0].bar(range(n_nodes), binary_metrics['clustering_coefficient'])
        axes[1, 0].set_title('Clustering Coefficient')
        axes[1, 0].set_xlabel('Node')
        axes[1, 0].set_ylabel('Clustering')

        axes[1, 1].bar(range(n_nodes), binary_metrics['betweenness_centrality'])
        axes[1, 1].set_title('Betweenness Centrality')
        axes[1, 1].set_xlabel('Node')
        axes[1, 1].set_ylabel('Betweenness')

        plt.tight_layout()
        plt.savefig(viz_dir / "node_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved node metrics plot: {viz_dir / 'node_metrics.png'}")

        print("  OK Node metrics tests passed")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_global_metrics(output_dir):
    """Test global network metrics"""
    print("\n" + "=" * 80)
    print("TEST: Global Network Metrics")
    print("=" * 80)
    try:
        n_nodes = 40
        matrix = create_synthetic_network(n_nodes, density=0.12, mean_weight=0.6)

        print(f"Network: {n_nodes} nodes")

        # Compute global metrics
        global_metrics = compute_global_metrics(matrix, threshold=0.3)

        print(f"\nGlobal metrics:")
        print(f"  Global efficiency: {global_metrics['global_efficiency']:.4f}")
        print(f"  Characteristic path length: {global_metrics['characteristic_path_length']:.4f}")
        print(f"  Transitivity: {global_metrics['transitivity']:.4f}")

        # Small-world metrics (compare to random network)
        print(f"\nComparing to random network...")

        # Create random network with same density
        n_edges = int(np.sum(matrix != 0) / 2)
        random_matrix = np.zeros((n_nodes, n_nodes))
        edges = np.random.choice(n_nodes * (n_nodes - 1) // 2, n_edges, replace=False)
        triu_indices = np.triu_indices(n_nodes, k=1)
        for edge_idx in edges:
            i, j = triu_indices[0][edge_idx], triu_indices[1][edge_idx]
            weight = np.random.randn() * 0.2 + 0.6
            random_matrix[i, j] = weight
            random_matrix[j, i] = weight

        random_metrics = compute_global_metrics(random_matrix, threshold=0.3)

        print(f"Random network:")
        print(f"  Global efficiency: {random_metrics['global_efficiency']:.4f}")
        print(f"  Characteristic path length: {random_metrics['characteristic_path_length']:.4f}")
        print(f"  Transitivity: {random_metrics['transitivity']:.4f}")

        # Compute small-world metrics
        gamma = global_metrics['transitivity'] / random_metrics['transitivity']
        lambda_val = global_metrics['characteristic_path_length'] / random_metrics['characteristic_path_length']
        sigma = gamma / lambda_val if lambda_val > 0 else 0

        print(f"\nSmall-world metrics:")
        print(f"  Gamma (C/C_rand): {gamma:.3f}")
        print(f"  Lambda (L/L_rand): {lambda_val:.3f}")
        print(f"  Sigma (gamma/lambda): {sigma:.3f}")
        print(f"  Small-world: {'YES' if sigma > 1 else 'NO'} (sigma > 1)")

        print("  OK Global metrics tests passed")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nbs(output_dir):
    """Test Network-Based Statistic"""
    print("\n" + "=" * 80)
    print("TEST: Network-Based Statistic (NBS)")
    print("=" * 80)
    try:
        n_subjects = 15
        n_nodes = 25

        print(f"Creating synthetic data: {n_subjects} subjects per group, {n_nodes} nodes")

        # Group 1: Normal connectivity
        group1 = []
        for i in range(n_subjects):
            matrix = create_synthetic_network(n_nodes, density=0.15, mean_weight=0.5, std_weight=0.15)
            group1.append(matrix)
        group1 = np.stack(group1)

        # Group 2: Enhanced connectivity in specific region (nodes 10-15)
        group2 = []
        for i in range(n_subjects):
            matrix = create_synthetic_network(n_nodes, density=0.15, mean_weight=0.5, std_weight=0.15)
            # Boost connectivity in specific subnetwork
            matrix[10:16, 10:16] += 0.4
            matrix = (matrix + matrix.T) / 2
            np.fill_diagonal(matrix, 0)
            group2.append(matrix)
        group2 = np.stack(group2)

        print(f"\nGroup 1 mean connectivity: {group1.mean():.3f}")
        print(f"Group 2 mean connectivity: {group2.mean():.3f}")
        print(f"Enhanced subnetwork (10-15) boost: +0.4")

        # Run NBS with fewer permutations for speed
        print(f"\nRunning NBS (1000 permutations for speed)...")
        nbs_dir = output_dir / "nbs"
        nbs_results = compute_network_based_statistic(
            group1, group2,
            threshold=2.5,
            n_permutations=1000,
            alpha=0.05,
            output_dir=nbs_dir
        )

        print(f"\nNBS Results:")
        print(f"  Components found: {nbs_results['n_components']}")
        if nbs_results['n_components'] > 0:
            for i, (size, pval) in enumerate(zip(
                nbs_results['component_sizes'],
                nbs_results['component_pvals']
            )):
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                print(f"  Component {i+1}: {size} edges, p={pval:.4f} {sig}")

        # Visualize group means and difference
        viz_dir = output_dir / "nbs_viz"
        viz_dir.mkdir(exist_ok=True)

        roi_names = [f"Node_{i:02d}" for i in range(n_nodes)]

        group1_mean = group1.mean(axis=0)
        group2_mean = group2.mean(axis=0)

        from neurovrai.connectome import plot_connectivity_comparison
        plot_connectivity_comparison(
            group1_mean, group2_mean,
            roi_names, viz_dir / "group_comparison.png",
            "Group 1", "Group 2"
        )
        print(f"\n  Saved group comparison: {viz_dir / 'group_comparison.png'}")

        # Visualize significant component if found
        if nbs_results['n_components'] > 0:
            sig_matrix = nbs_results['test_statistic'] * nbs_results['sig_edges']
            plot_connectivity_matrix(
                sig_matrix, roi_names,
                viz_dir / "nbs_significant.png",
                title=f"NBS Significant Edges (p<{nbs_results['alpha']})"
            )
            print(f"  Saved NBS results: {viz_dir / 'nbs_significant.png'}")

        print("  OK NBS tests passed")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("GRAPH METRICS AND NBS TEST SUITE")
    print("=" * 80)

    test_dir = Path(tempfile.mkdtemp(prefix="graph_nbs_test_"))
    print(f"\nTest directory: {test_dir}")

    results = []
    results.append(("Thresholding", test_thresholding(test_dir)))
    results.append(("Node Metrics", test_node_metrics(test_dir)))
    results.append(("Global Metrics", test_global_metrics(test_dir)))
    results.append(("Network-Based Statistic", test_nbs(test_dir)))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "OK PASSED" if passed else "FAILED"
        print(f"{name:40s} {status}")

    if all(p for _, p in results):
        print("\n*** ALL TESTS PASSED ***")
        print(f"\nOutputs: {test_dir}")
        return 0
    else:
        print("\n*** SOME TESTS FAILED ***")
        return 1


if __name__ == '__main__':
    sys.exit(main())
