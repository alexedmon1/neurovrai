#!/usr/bin/env python3
"""
Advanced Connectome Module Test Script

Tests visualization and group-level connectivity analysis features.

Usage:
    uv run python archive/tests/test_connectome_advanced.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurovrai.connectome import (
    plot_connectivity_matrix,
    plot_circular_connectogram,
    plot_connectivity_comparison,
    average_connectivity_matrices,
    compute_group_difference,
)


def create_synthetic_matrices(n_subjects, n_rois, mean=0.3, std=0.2):
    """Create synthetic connectivity matrices"""
    print(f"\nCreating {n_subjects} synthetic matrices ({n_rois} ROIs)...")
    matrices = []
    for i in range(n_subjects):
        matrix = np.random.randn(n_rois, n_rois) * std + mean
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        matrices.append(matrix)
    matrices = np.stack(matrices)
    roi_names = [f"ROI_{i:02d}" for i in range(n_rois)]
    print(f"  Created shape: {matrices.shape}")
    return matrices, roi_names


def test_visualization(output_dir):
    """Test visualization functions"""
    print("\n" + "=" * 80)
    print("TEST: Visualization")
    print("=" * 80)
    try:
        n_rois = 20
        matrix = np.random.randn(n_rois, n_rois) * 0.3 + 0.5
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        roi_names = [f"Region_{i:02d}" for i in range(n_rois)]

        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        print("Plotting heatmap...")
        plot_connectivity_matrix(matrix, roi_names, viz_dir / "heatmap.png")
        print("  OK Heatmap created")

        print("Plotting connectogram...")
        plot_circular_connectogram(matrix, roi_names, viz_dir / "connectogram.png", threshold=0.5)
        print("  OK Connectogram created")

        print("OK VISUALIZATION TESTS PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_group_averaging(output_dir):
    """Test group averaging"""
    print("\n" + "=" * 80)
    print("TEST: Group Averaging")
    print("=" * 80)
    try:
        matrices, roi_names = create_synthetic_matrices(10, 15, 0.4, 0.2)
        group_dir = output_dir / "group_average"

        results = average_connectivity_matrices(matrices, output_dir=group_dir)
        print(f"  Mean matrix shape: {results['mean_matrix'].shape}")
        print(f"  Subjects: {results['n_subjects']}")

        results_cons = average_connectivity_matrices(
            matrices, consistency_threshold=0.7,
            output_dir=output_dir / "group_consistent"
        )
        print(f"  Consistent edges: {results_cons['n_consistent_edges']}")

        plot_connectivity_matrix(
            results['mean_matrix'], roi_names,
            group_dir / "mean_matrix.png",
            title=f"Group Average (N={results['n_subjects']})"
        )

        print("OK GROUP AVERAGING TESTS PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_group_difference(output_dir):
    """Test group difference analysis"""
    print("\n" + "=" * 80)
    print("TEST: Group Difference")
    print("=" * 80)
    try:
        group1, roi_names = create_synthetic_matrices(12, 15, 0.3, 0.15)
        group2, _ = create_synthetic_matrices(12, 15, 0.5, 0.15)

        diff_dir = output_dir / "group_difference"
        results = compute_group_difference(
            group1, group2,
            "Controls", "Patients",
            output_dir=diff_dir
        )

        print(f"  Significant edges: {results['n_significant']}")

        plot_connectivity_comparison(
            results['group1_mean'], results['group2_mean'],
            roi_names, diff_dir / "comparison.png",
            "Controls", "Patients"
        )

        print("OK GROUP DIFFERENCE TESTS PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("ADVANCED CONNECTOME TEST SUITE")
    print("=" * 80)

    test_dir = Path(tempfile.mkdtemp(prefix="connectome_advanced_"))
    print(f"\nTest directory: {test_dir}")

    results = []
    results.append(("Visualization", test_visualization(test_dir)))
    results.append(("Group Averaging", test_group_averaging(test_dir)))
    results.append(("Group Difference", test_group_difference(test_dir)))

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
