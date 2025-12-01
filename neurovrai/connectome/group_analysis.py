#!/usr/bin/env python3
"""
Group-Level Connectivity Analysis

Tools for analyzing connectivity across multiple subjects including group
averaging, statistical testing, and consistency thresholding.

Key Features:
- Average connectivity matrices across subjects
- Compute group statistics (mean, std, se)
- Consistency thresholding (edges present in X% of subjects)
- Two-sample t-tests for group differences
- Integration with demographic data
- Comprehensive reporting

Usage:
    from neurovrai.connectome.group_analysis import (
        average_connectivity_matrices,
        compute_group_difference
    )

    # Average matrices across subjects
    group_result = average_connectivity_matrices(
        matrices=subject_matrices,
        subject_ids=subject_list
    )

    # Compare two groups
    comparison = compute_group_difference(
        group1_matrices=control_matrices,
        group2_matrices=patient_matrices
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def load_connectivity_matrices(
    matrix_files: List[Path],
    subject_ids: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load multiple connectivity matrices from files

    Args:
        matrix_files: List of paths to .npy files
        subject_ids: Optional list of subject IDs (inferred from filenames if None)

    Returns:
        Tuple of:
            - matrices: Array of shape (n_subjects, n_rois, n_rois)
            - subject_ids: List of subject IDs

    Raises:
        ValueError: If matrices have inconsistent shapes
    """
    logger.info(f"Loading {len(matrix_files)} connectivity matrices...")

    matrices = []
    loaded_subjects = []

    for i, matrix_file in enumerate(matrix_files):
        matrix_file = Path(matrix_file)

        if not matrix_file.exists():
            logger.warning(f"File not found: {matrix_file}, skipping")
            continue

        # Load matrix
        matrix = np.load(matrix_file)
        matrices.append(matrix)

        # Get subject ID
        if subject_ids is not None:
            loaded_subjects.append(subject_ids[i])
        else:
            # Infer from filename
            loaded_subjects.append(matrix_file.stem)

    if not matrices:
        raise ValueError("No matrices loaded successfully")

    # Check shapes
    shapes = [m.shape for m in matrices]
    if len(set(shapes)) > 1:
        raise ValueError(f"Inconsistent matrix shapes: {set(shapes)}")

    # Stack into 3D array
    matrices_array = np.stack(matrices)

    logger.info(f"  Loaded {len(matrices)} matrices")
    logger.info(f"  Shape: {matrices_array.shape}")

    return matrices_array, loaded_subjects


def average_connectivity_matrices(
    matrices: np.ndarray,
    subject_ids: Optional[List[str]] = None,
    consistency_threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
    output_prefix: str = 'group'
) -> Dict:
    """
    Average connectivity matrices across subjects

    Args:
        matrices: Array of shape (n_subjects, n_rois, n_rois)
        subject_ids: Optional list of subject IDs
        consistency_threshold: Optional threshold (0-1) for edge consistency
                              (e.g., 0.7 = edge present in 70% of subjects)
        output_dir: Optional output directory
        output_prefix: Prefix for output files

    Returns:
        Dictionary containing:
            - mean_matrix: Mean connectivity matrix
            - std_matrix: Standard deviation matrix
            - se_matrix: Standard error matrix
            - consistency_matrix: Binary matrix of edge consistency (if threshold provided)
            - n_subjects: Number of subjects
            - subject_ids: List of subject IDs
    """
    n_subjects, n_rois, _ = matrices.shape

    if subject_ids is None:
        subject_ids = [f"sub-{i:03d}" for i in range(n_subjects)]

    logger.info("=" * 80)
    logger.info("GROUP CONNECTIVITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Number of subjects: {n_subjects}")
    logger.info(f"Matrix size: {n_rois} x {n_rois}")

    # Compute statistics
    logger.info("\nComputing group statistics...")
    mean_matrix = np.mean(matrices, axis=0)
    std_matrix = np.std(matrices, axis=0)
    se_matrix = std_matrix / np.sqrt(n_subjects)

    logger.info(f"  Mean connectivity: {mean_matrix[np.triu_indices(n_rois, k=1)].mean():.4f}")
    logger.info(f"  Std connectivity: {std_matrix[np.triu_indices(n_rois, k=1)].mean():.4f}")

    results = {
        'mean_matrix': mean_matrix,
        'std_matrix': std_matrix,
        'se_matrix': se_matrix,
        'n_subjects': n_subjects,
        'subject_ids': subject_ids
    }

    # Consistency thresholding if requested
    if consistency_threshold is not None:
        logger.info(f"\nApplying consistency threshold: {consistency_threshold}")

        # Count non-zero edges per connection
        nonzero_count = np.sum(matrices != 0, axis=0)
        consistency_matrix = (nonzero_count / n_subjects) >= consistency_threshold

        # Apply to mean matrix
        consistent_mean = mean_matrix.copy()
        consistent_mean[~consistency_matrix] = 0

        n_edges_total = np.sum(np.triu_indices(n_rois, k=1))
        n_edges_consistent = np.sum(consistency_matrix[np.triu_indices(n_rois, k=1)])

        logger.info(f"  Consistent edges: {n_edges_consistent} / {n_edges_total}")
        logger.info(f"  Retention rate: {n_edges_consistent/n_edges_total*100:.1f}%")

        results['consistency_matrix'] = consistency_matrix
        results['consistent_mean_matrix'] = consistent_mean
        results['consistency_threshold'] = consistency_threshold
        results['n_consistent_edges'] = int(n_edges_consistent)

    # Save outputs if directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving outputs to: {output_dir}")

        # Save matrices
        np.save(output_dir / f"{output_prefix}_mean_matrix.npy", mean_matrix)
        np.save(output_dir / f"{output_prefix}_std_matrix.npy", std_matrix)
        np.save(output_dir / f"{output_prefix}_se_matrix.npy", se_matrix)

        if consistency_threshold is not None:
            np.save(output_dir / f"{output_prefix}_consistency_matrix.npy", consistency_matrix)
            np.save(output_dir / f"{output_prefix}_consistent_mean_matrix.npy", consistent_mean)

        # Save summary
        summary = {
            'n_subjects': n_subjects,
            'n_rois': n_rois,
            'subject_ids': subject_ids,
            'mean_connectivity': float(mean_matrix[np.triu_indices(n_rois, k=1)].mean()),
            'std_connectivity': float(std_matrix[np.triu_indices(n_rois, k=1)].mean()),
        }

        if consistency_threshold is not None:
            summary['consistency_threshold'] = consistency_threshold
            summary['n_consistent_edges'] = int(n_edges_consistent)
            summary['consistency_rate'] = float(n_edges_consistent / n_edges_total)

        summary_file = output_dir / f"{output_prefix}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"   Mean matrix: {output_prefix}_mean_matrix.npy")
        logger.info(f"   Std matrix: {output_prefix}_std_matrix.npy")
        logger.info(f"   SE matrix: {output_prefix}_se_matrix.npy")
        logger.info(f"   Summary: {output_prefix}_summary.json")

        results['output_dir'] = str(output_dir)

    logger.info("=" * 80)

    return results


def compute_group_difference(
    group1_matrices: np.ndarray,
    group2_matrices: np.ndarray,
    group1_name: str = 'Group 1',
    group2_name: str = 'Group 2',
    paired: bool = False,
    alpha: float = 0.05,
    output_dir: Optional[Path] = None,
    output_prefix: str = 'group_diff'
) -> Dict:
    """
    Compute statistical difference between two groups

    Args:
        group1_matrices: Array of shape (n_subjects1, n_rois, n_rois)
        group2_matrices: Array of shape (n_subjects2, n_rois, n_rois)
        group1_name: Name for group 1
        group2_name: Name for group 2
        paired: Whether samples are paired (e.g., pre/post)
        alpha: Significance level
        output_dir: Optional output directory
        output_prefix: Prefix for output files

    Returns:
        Dictionary containing:
            - group1_mean: Mean matrix for group 1
            - group2_mean: Mean matrix for group 2
            - difference_matrix: Difference (group1 - group2)
            - t_matrix: T-statistic matrix
            - p_matrix: P-value matrix (uncorrected)
            - p_matrix_fdr: FDR-corrected p-values
            - significant_edges: Binary matrix of significant edges
            - n_significant: Number of significant edges
    """
    n_subjects1 = group1_matrices.shape[0]
    n_subjects2 = group2_matrices.shape[0]
    n_rois = group1_matrices.shape[1]

    logger.info("=" * 80)
    logger.info("GROUP DIFFERENCE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"{group1_name}: {n_subjects1} subjects")
    logger.info(f"{group2_name}: {n_subjects2} subjects")
    logger.info(f"Test type: {'Paired' if paired else 'Independent'} t-test")
    logger.info(f"Significance level: {alpha}")

    # Compute group means
    group1_mean = np.mean(group1_matrices, axis=0)
    group2_mean = np.mean(group2_matrices, axis=0)
    difference_matrix = group1_mean - group2_mean

    logger.info(f"\n{group1_name} mean connectivity: {group1_mean[np.triu_indices(n_rois, k=1)].mean():.4f}")
    logger.info(f"{group2_name} mean connectivity: {group2_mean[np.triu_indices(n_rois, k=1)].mean():.4f}")
    logger.info(f"Mean difference: {difference_matrix[np.triu_indices(n_rois, k=1)].mean():.4f}")

    # Perform t-tests for each connection
    logger.info("\nComputing edge-wise t-tests...")

    t_matrix = np.zeros((n_rois, n_rois))
    p_matrix = np.ones((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            group1_values = group1_matrices[:, i, j]
            group2_values = group2_matrices[:, i, j]

            if paired:
                if n_subjects1 != n_subjects2:
                    raise ValueError("Paired test requires equal number of subjects")
                t_stat, p_val = stats.ttest_rel(group1_values, group2_values)
            else:
                t_stat, p_val = stats.ttest_ind(group1_values, group2_values)

            t_matrix[i, j] = t_stat
            t_matrix[j, i] = t_stat
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

    # FDR correction
    logger.info("Applying FDR correction...")
    upper_triangle_indices = np.triu_indices(n_rois, k=1)
    p_values_flat = p_matrix[upper_triangle_indices]

    # Benjamini-Hochberg procedure
    from statsmodels.stats.multitest import multipletests
    reject, p_values_corrected, _, _ = multipletests(p_values_flat, alpha=alpha, method='fdr_bh')

    # Reconstruct corrected p-value matrix
    p_matrix_fdr = np.ones((n_rois, n_rois))
    p_matrix_fdr[upper_triangle_indices] = p_values_corrected
    p_matrix_fdr = p_matrix_fdr + p_matrix_fdr.T  # Make symmetric

    # Significant edges
    significant_edges = p_matrix_fdr < alpha
    n_significant = np.sum(significant_edges[upper_triangle_indices])

    logger.info(f"\nResults:")
    logger.info(f"  Significant edges (FDR-corrected): {n_significant}")
    logger.info(f"  Proportion significant: {n_significant / len(p_values_flat) * 100:.2f}%")

    if n_significant > 0:
        significant_t = t_matrix[significant_edges]
        logger.info(f"  T-statistics range: [{significant_t.min():.3f}, {significant_t.max():.3f}]")

    results = {
        'group1_mean': group1_mean,
        'group2_mean': group2_mean,
        'difference_matrix': difference_matrix,
        't_matrix': t_matrix,
        'p_matrix': p_matrix,
        'p_matrix_fdr': p_matrix_fdr,
        'significant_edges': significant_edges,
        'n_significant': int(n_significant),
        'n_subjects_group1': n_subjects1,
        'n_subjects_group2': n_subjects2,
        'alpha': alpha
    }

    # Save outputs if directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving outputs to: {output_dir}")

        # Save matrices
        np.save(output_dir / f"{output_prefix}_group1_mean.npy", group1_mean)
        np.save(output_dir / f"{output_prefix}_group2_mean.npy", group2_mean)
        np.save(output_dir / f"{output_prefix}_difference.npy", difference_matrix)
        np.save(output_dir / f"{output_prefix}_t_matrix.npy", t_matrix)
        np.save(output_dir / f"{output_prefix}_p_matrix.npy", p_matrix)
        np.save(output_dir / f"{output_prefix}_p_matrix_fdr.npy", p_matrix_fdr)
        np.save(output_dir / f"{output_prefix}_significant_edges.npy", significant_edges)

        # Save summary
        summary = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'n_subjects_group1': n_subjects1,
            'n_subjects_group2': n_subjects2,
            'paired': paired,
            'alpha': alpha,
            'n_rois': n_rois,
            'n_significant_edges': int(n_significant),
            'proportion_significant': float(n_significant / len(p_values_flat)),
            'group1_mean_connectivity': float(group1_mean[upper_triangle_indices].mean()),
            'group2_mean_connectivity': float(group2_mean[upper_triangle_indices].mean()),
            'mean_difference': float(difference_matrix[upper_triangle_indices].mean())
        }

        summary_file = output_dir / f"{output_prefix}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"   Saved all matrices and summary")

        results['output_dir'] = str(output_dir)

    logger.info("=" * 80)

    return results


def filter_subjects_by_demographics(
    matrix_files: List[Path],
    participants_file: Path,
    filters: Optional[Dict[str, any]] = None
) -> Tuple[List[Path], pd.DataFrame]:
    """
    Filter subjects based on demographic criteria

    Args:
        matrix_files: List of connectivity matrix files
        participants_file: Path to participants CSV
        filters: Dictionary of column:value pairs to filter on

    Returns:
        Tuple of (filtered_matrix_files, filtered_demographics)

    Example:
        filters = {'group': 'patient', 'age': lambda x: x > 30}
    """
    logger.info("Filtering subjects by demographics...")

    # Load participants data
    participants = pd.read_csv(participants_file)

    # Apply filters
    if filters:
        for column, criterion in filters.items():
            if callable(criterion):
                participants = participants[participants[column].apply(criterion)]
            else:
                participants = participants[participants[column] == criterion]

    logger.info(f"  Filtered to {len(participants)} subjects")

    # Match matrix files to filtered subjects
    filtered_files = []
    for matrix_file in matrix_files:
        subject_id = matrix_file.stem.split('_')[0]  # Assumes format: sub-001_fc_matrix.npy
        if subject_id in participants['subject_id'].values:
            filtered_files.append(matrix_file)

    logger.info(f"  Found {len(filtered_files)} matching matrix files")

    return filtered_files, participants


def compute_network_based_statistic(
    group1_matrices: np.ndarray,
    group2_matrices: np.ndarray,
    threshold: float = 3.0,
    n_permutations: int = 5000,
    alpha: float = 0.05,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Compute Network-Based Statistic (NBS)

    The NBS identifies connected components of edges that show significant
    group differences, controlling for multiple comparisons via permutation testing.

    Args:
        group1_matrices: Array of shape (n_subjects1, n_rois, n_rois)
        group2_matrices: Array of shape (n_subjects2, n_rois, n_rois)
        threshold: T-statistic threshold for edge selection
        n_permutations: Number of permutations
        alpha: Significance level
        output_dir: Optional output directory

    Returns:
        Dictionary containing:
            - t_matrix: T-statistic matrix
            - components: List of significant connected components
            - component_sizes: Size of each component
            - component_pvalues: P-value for each component
            - max_component_null: Null distribution of max component sizes

    Reference:
        Zalesky et al. (2010). Network-based statistic: Identifying
        differences in brain networks. NeuroImage, 53(4), 1197-1207.
    """
    import networkx as nx

    n_subjects1, n_rois, _ = group1_matrices.shape
    n_subjects2 = group2_matrices.shape[0]

    logger.info("=" * 80)
    logger.info("NETWORK-BASED STATISTIC (NBS)")
    logger.info("=" * 80)
    logger.info(f"Group 1: {n_subjects1} subjects")
    logger.info(f"Group 2: {n_subjects2} subjects")
    logger.info(f"T-threshold: {threshold}")
    logger.info(f"Permutations: {n_permutations}")

    # Step 1: Compute t-statistics for each edge
    logger.info("\nStep 1: Computing edge-wise t-statistics...")
    t_matrix = np.zeros((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            group1_values = group1_matrices[:, i, j]
            group2_values = group2_matrices[:, i, j]
            t_stat, _ = stats.ttest_ind(group1_values, group2_values)
            t_matrix[i, j] = t_stat
            t_matrix[j, i] = t_stat

    # Step 2: Threshold t-matrix and find connected components
    logger.info(f"\nStep 2: Finding connected components (t > {threshold})...")

    def find_components(t_mat, thresh):
        """Find connected components in thresholded network"""
        adj_matrix = (np.abs(t_mat) > thresh).astype(int)
        np.fill_diagonal(adj_matrix, 0)

        G = nx.from_numpy_array(adj_matrix)
        components = list(nx.connected_components(G))

        # Filter out single-node components
        components = [c for c in components if len(c) > 1]

        # Compute component sizes (number of edges)
        sizes = []
        for comp in components:
            nodes = list(comp)
            subgraph = adj_matrix[np.ix_(nodes, nodes)]
            n_edges = np.sum(subgraph) // 2
            sizes.append(n_edges)

        return components, sizes

    observed_components, observed_sizes = find_components(t_matrix, threshold)

    if not observed_components:
        logger.info("  No connected components found above threshold")
        return {
            't_matrix': t_matrix,
            'components': [],
            'component_sizes': [],
            'component_pvalues': [],
            'max_component_null': [],
            'threshold': threshold,
            'n_permutations': n_permutations
        }

    max_observed_size = max(observed_sizes) if observed_sizes else 0
    logger.info(f"  Found {len(observed_components)} components")
    logger.info(f"  Largest component: {max_observed_size} edges")

    # Step 3: Permutation testing
    logger.info(f"\nStep 3: Running {n_permutations} permutations...")

    # Combine groups for permutation
    all_matrices = np.concatenate([group1_matrices, group2_matrices], axis=0)
    n_total = n_subjects1 + n_subjects2

    null_max_sizes = []

    for perm in range(n_permutations):
        if (perm + 1) % 1000 == 0:
            logger.info(f"  Permutation {perm + 1}/{n_permutations}")

        # Permute group labels
        perm_indices = np.random.permutation(n_total)
        perm_group1 = all_matrices[perm_indices[:n_subjects1]]
        perm_group2 = all_matrices[perm_indices[n_subjects1:]]

        # Compute t-statistics
        t_perm = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                t_stat, _ = stats.ttest_ind(perm_group1[:, i, j], perm_group2[:, i, j])
                t_perm[i, j] = t_stat
                t_perm[j, i] = t_stat

        # Find max component size
        _, perm_sizes = find_components(t_perm, threshold)
        max_size = max(perm_sizes) if perm_sizes else 0
        null_max_sizes.append(max_size)

    null_max_sizes = np.array(null_max_sizes)

    # Step 4: Compute p-values
    logger.info("\nStep 4: Computing component p-values...")

    component_pvalues = []
    for size in observed_sizes:
        p_value = np.sum(null_max_sizes >= size) / n_permutations
        component_pvalues.append(p_value)

    # Report results
    logger.info("\nResults:")
    n_significant = sum(p < alpha for p in component_pvalues)
    logger.info(f"  Significant components: {n_significant}/{len(observed_components)}")

    for i, (comp, size, p_val) in enumerate(zip(observed_components, observed_sizes, component_pvalues)):
        sig_marker = "***" if p_val < alpha else ""
        logger.info(f"  Component {i+1}: {len(comp)} nodes, {size} edges, p={p_val:.4f} {sig_marker}")

    results = {
        't_matrix': t_matrix,
        'components': [list(c) for c in observed_components],
        'component_sizes': observed_sizes,
        'component_pvalues': component_pvalues,
        'max_component_null': null_max_sizes,
        'threshold': threshold,
        'n_permutations': n_permutations,
        'alpha': alpha,
        'n_significant': n_significant
    }

    # Save outputs if directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving outputs to: {output_dir}")

        np.save(output_dir / "nbs_t_matrix.npy", t_matrix)
        np.save(output_dir / "nbs_null_distribution.npy", null_max_sizes)

        # Save component info (convert numpy types to native Python types for JSON)
        component_info = {
            'n_components': len(results['components']),
            'components': results['components'],  # Already lists from line 592
            'component_sizes': [int(s) for s in results['component_sizes']],
            'component_pvalues': [float(p) for p in results['component_pvalues']],
            'threshold': float(threshold),
            'n_permutations': int(n_permutations),
            'alpha': float(alpha),
            'n_significant': int(n_significant)
        }

        with open(output_dir / "nbs_components.json", 'w') as f:
            json.dump(component_info, f, indent=2)

        logger.info("  ✓ Saved NBS results")

        results['output_dir'] = str(output_dir)

    logger.info("=" * 80)

    return results
