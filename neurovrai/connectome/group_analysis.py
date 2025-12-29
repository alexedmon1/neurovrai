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


def load_connectivity_with_rois(
    connectivity_dir: Path,
    matrix_pattern: str = '*_matrix.npy',
    roi_pattern: str = '*_roi_names.txt',
    subject_pattern: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Load connectivity matrices along with their ROI names.

    This function handles varying matrix sizes across subjects by loading
    each matrix with its associated ROI names, enabling ROI intersection
    analysis.

    Args:
        connectivity_dir: Base directory containing subject subdirectories
        matrix_pattern: Glob pattern for matrix files (default: '*_matrix.npy')
        roi_pattern: Glob pattern for ROI name files (default: '*_roi_names.txt')
        subject_pattern: Optional pattern to filter subjects (e.g., 'IRC805-*')

    Returns:
        Dictionary mapping subject_id -> {
            'matrix': np.ndarray,
            'rois': List[str],
            'matrix_file': Path,
            'roi_file': Path
        }

    Example:
        data = load_connectivity_with_rois(
            connectivity_dir=Path('/study/connectome/functional'),
            matrix_pattern='fc_matrix.npy',
            roi_pattern='fc_roi_names.txt'
        )
    """
    connectivity_dir = Path(connectivity_dir)
    logger.info(f"Loading connectivity data from: {connectivity_dir}")

    data = {}

    # Find subject directories
    if subject_pattern:
        subject_dirs = sorted(connectivity_dir.glob(subject_pattern))
    else:
        subject_dirs = sorted([d for d in connectivity_dir.iterdir() if d.is_dir()])

    for subj_dir in subject_dirs:
        subject = subj_dir.name

        # Find matrix file
        matrix_files = list(subj_dir.glob(f'**/{matrix_pattern}'))
        if not matrix_files:
            logger.debug(f"  {subject}: No matrix file found")
            continue
        matrix_file = matrix_files[0]

        # Find ROI names file
        roi_files = list(subj_dir.glob(f'**/{roi_pattern}'))
        if not roi_files:
            logger.debug(f"  {subject}: No ROI names file found")
            continue
        roi_file = roi_files[0]

        # Load matrix
        matrix = np.load(matrix_file)

        # Load ROI names
        with open(roi_file) as f:
            rois = [line.strip() for line in f if line.strip()]

        # Verify dimensions match
        if matrix.shape[0] != len(rois):
            logger.warning(
                f"  {subject}: Matrix size ({matrix.shape[0]}) != ROI count ({len(rois)}), "
                f"using min({matrix.shape[0]}, {len(rois)})"
            )
            n = min(matrix.shape[0], len(rois))
            matrix = matrix[:n, :n]
            rois = rois[:n]

        data[subject] = {
            'matrix': matrix,
            'rois': rois,
            'matrix_file': matrix_file,
            'roi_file': roi_file
        }

    logger.info(f"  Loaded {len(data)} subjects")

    # Show size distribution
    sizes = {}
    for subj, d in data.items():
        size = d['matrix'].shape[0]
        sizes[size] = sizes.get(size, 0) + 1
    logger.info(f"  Matrix sizes: {sizes}")

    return data


def compute_roi_intersection(
    data: Dict[str, Dict],
    min_subjects: Optional[int] = None,
    min_rois_per_subject: int = 1
) -> List[str]:
    """
    Compute the intersection of ROIs present across all subjects.

    Args:
        data: Dictionary from load_connectivity_with_rois()
        min_subjects: If specified, include ROIs present in at least this many subjects
                     (default: None = require all subjects)
        min_rois_per_subject: Minimum number of ROIs a subject must have to be included
                             (default: 1, excludes subjects with empty ROI lists)

    Returns:
        Sorted list of ROI names present in all (or min_subjects) subjects

    Example:
        common_rois = compute_roi_intersection(data)
        # Returns: ['ROI_001', 'ROI_002', ...]
    """
    if not data:
        return []

    # Filter out subjects with too few ROIs
    valid_data = {k: v for k, v in data.items() if len(v['rois']) >= min_rois_per_subject}

    if len(valid_data) < len(data):
        excluded = len(data) - len(valid_data)
        logger.info(f"  Excluded {excluded} subjects with < {min_rois_per_subject} ROIs")

    if not valid_data:
        logger.warning("No subjects have sufficient ROIs")
        return []

    # Get ROI sets for each valid subject
    roi_sets = [set(d['rois']) for d in valid_data.values()]

    if min_subjects is None or min_subjects >= len(valid_data):
        # Strict intersection - ROI must be in all subjects
        common_rois = set.intersection(*roi_sets)
    else:
        # Count ROI occurrences
        from collections import Counter
        all_rois = []
        for roi_set in roi_sets:
            all_rois.extend(roi_set)
        roi_counts = Counter(all_rois)

        # Keep ROIs present in at least min_subjects
        common_rois = {roi for roi, count in roi_counts.items() if count >= min_subjects}

    common_rois = sorted(common_rois)

    logger.info(f"ROI intersection: {len(common_rois)} common ROIs across {len(valid_data)} subjects")

    return common_rois


def extract_common_roi_submatrices(
    data: Dict[str, Dict],
    common_rois: List[str],
    output_dir: Optional[Path] = None
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extract submatrices containing only the common ROIs.

    This function re-indexes each subject's matrix to include only ROIs
    present in the common_rois list, enabling group comparisons across
    subjects with originally different ROI sets.

    Args:
        data: Dictionary from load_connectivity_with_rois()
        common_rois: List of common ROI names from compute_roi_intersection()
        output_dir: Optional directory to save extracted matrices and ROI list

    Returns:
        Tuple of:
            - Dictionary mapping subject_id -> extracted matrix (n_common x n_common)
            - List of common ROI names (same as input for consistency)

    Example:
        matrices, roi_names = extract_common_roi_submatrices(data, common_rois)
        # Now all matrices are the same size and aligned to common_rois
    """
    extracted = {}
    skipped = []

    for subject, subj_data in data.items():
        roi_list = subj_data['rois']
        matrix = subj_data['matrix']

        # Get indices of common ROIs in this subject's ROI list
        indices = []
        valid = True

        for roi in common_rois:
            if roi in roi_list:
                idx = roi_list.index(roi)
                if idx < matrix.shape[0]:
                    indices.append(idx)
                else:
                    logger.warning(f"  {subject}: ROI '{roi}' index {idx} out of bounds")
                    valid = False
                    break
            else:
                logger.debug(f"  {subject}: ROI '{roi}' not found")
                valid = False
                break

        if valid and len(indices) == len(common_rois):
            # Extract submatrix
            submatrix = matrix[np.ix_(indices, indices)]
            extracted[subject] = submatrix
        else:
            skipped.append(subject)

    logger.info(f"Extracted {len(extracted)} subjects with {len(common_rois)} common ROIs")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} subjects: {skipped}")

    # Save outputs if directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save common ROI list
        roi_file = output_dir / 'common_rois.txt'
        with open(roi_file, 'w') as f:
            f.write('\n'.join(common_rois))
        logger.info(f"  Saved common ROI list: {roi_file}")

        # Save extraction summary
        summary = {
            'n_common_rois': len(common_rois),
            'n_subjects_extracted': len(extracted),
            'n_subjects_skipped': len(skipped),
            'subjects_extracted': list(extracted.keys()),
            'subjects_skipped': skipped
        }
        summary_file = output_dir / 'extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Saved extraction summary: {summary_file}")

    return extracted, common_rois


def load_and_align_connectivity(
    connectivity_dir: Path,
    matrix_pattern: str = '*_matrix.npy',
    roi_pattern: str = '*_roi_names.txt',
    subject_pattern: Optional[str] = None,
    min_subjects: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load connectivity matrices and align them to common ROIs.

    This is a convenience function that combines load_connectivity_with_rois,
    compute_roi_intersection, and extract_common_roi_submatrices.

    Args:
        connectivity_dir: Base directory containing subject subdirectories
        matrix_pattern: Glob pattern for matrix files
        roi_pattern: Glob pattern for ROI name files
        subject_pattern: Optional pattern to filter subjects
        min_subjects: Minimum subjects for ROI inclusion (default: all)
        output_dir: Optional directory to save outputs

    Returns:
        Tuple of:
            - matrices: Array of shape (n_subjects, n_rois, n_rois)
            - subject_ids: List of subject IDs
            - roi_names: List of common ROI names

    Example:
        matrices, subjects, rois = load_and_align_connectivity(
            connectivity_dir=Path('/study/connectome/functional'),
            matrix_pattern='fc_matrix.npy',
            roi_pattern='fc_roi_names.txt'
        )
    """
    # Load all data with ROIs
    data = load_connectivity_with_rois(
        connectivity_dir=connectivity_dir,
        matrix_pattern=matrix_pattern,
        roi_pattern=roi_pattern,
        subject_pattern=subject_pattern
    )

    if not data:
        raise ValueError("No connectivity data loaded")

    # Compute ROI intersection
    common_rois = compute_roi_intersection(data, min_subjects=min_subjects)

    if not common_rois:
        raise ValueError("No common ROIs found across subjects")

    # Extract aligned submatrices
    extracted, roi_names = extract_common_roi_submatrices(
        data, common_rois, output_dir=output_dir
    )

    if not extracted:
        raise ValueError("No subjects successfully extracted")

    # Stack into 3D array
    subject_ids = sorted(extracted.keys())
    matrices = np.stack([extracted[s] for s in subject_ids])

    logger.info(f"Final aligned data: {matrices.shape}")

    return matrices, subject_ids, roi_names


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


# =============================================================================
# Convenience Functions for FC and SC Analysis
# =============================================================================

def normalize_roi_name(roi_name: str) -> str:
    """
    Normalize ROI name to standardized 4-digit format: ROI_XXXX

    Handles various input formats:
    - ROI_2, ROI_02, ROI_002 -> ROI_0002
    - ROI_1234 -> ROI_1234

    Args:
        roi_name: ROI name in any format

    Returns:
        Normalized ROI name in ROI_XXXX format
    """
    if roi_name.startswith('ROI_'):
        label = roi_name[4:]
        try:
            num = int(label)
            return f'ROI_{num:04d}'
        except ValueError:
            return roi_name
    return roi_name


def load_fc_matrices_aligned(
    fc_dir: Path,
    atlas: str = 'desikan_killiany',
    subject_pattern: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load and align functional connectivity matrices.

    Convenience wrapper for load_and_align_connectivity with FC-specific defaults.

    FC directory structure: {fc_dir}/{subject}/{atlas}/fc_matrix.npy

    Args:
        fc_dir: Path to functional connectome directory (e.g., {study_root}/connectome/functional)
        atlas: Atlas name (default: 'desikan_killiany')
        subject_pattern: Optional glob pattern for subjects
        output_dir: Optional output directory for aligned matrices

    Returns:
        Tuple of (matrices, subject_ids, roi_names)
        - matrices: np.ndarray of shape (n_subjects, n_rois, n_rois)
        - subject_ids: List of subject identifiers
        - roi_names: List of ROI names (intersection across subjects)
    """
    fc_dir = Path(fc_dir)

    # FC uses structure: {fc_dir}/{subject}/{atlas}/
    # Load connectivity with ROIs
    data = load_connectivity_with_rois(
        connectivity_dir=fc_dir,
        matrix_pattern=f'{atlas}/fc_matrix.npy',
        roi_pattern=f'{atlas}/fc_roi_names.txt',
        subject_pattern=subject_pattern
    )

    if not data:
        raise ValueError(f"No FC data found in {fc_dir}")

    # Normalize ROI names to standardized format
    for subj in data:
        data[subj]['rois'] = [normalize_roi_name(r) for r in data[subj]['rois']]

    # Compute ROI intersection
    common_rois = compute_roi_intersection(data)

    if not common_rois:
        raise ValueError("No common ROIs found across subjects")

    # Extract aligned submatrices
    extracted, roi_names = extract_common_roi_submatrices(data, common_rois, output_dir=output_dir)

    if not extracted:
        raise ValueError("No subjects successfully extracted")

    # Stack into 3D array
    subject_ids = sorted(extracted.keys())
    matrices = np.stack([extracted[s] for s in subject_ids])

    logger.info(f"Final aligned FC data: {matrices.shape}")

    return matrices, subject_ids, roi_names


def load_sc_matrices_aligned(
    sc_dir: Path,
    atlas: str = 'desikan_killiany',
    subject_pattern: Optional[str] = None,
    output_dir: Optional[Path] = None,
    use_rebuilt: bool = True
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load and align structural connectivity matrices.

    Convenience wrapper for load_and_align_connectivity with SC-specific defaults.
    By default, uses rebuilt matrices (sc_matrix_rebuilt.npy) which are properly
    aggregated from probtrackx outputs.

    SC directory structure: {sc_dir}/{atlas}/{subject}/sc_matrix.npy

    Args:
        sc_dir: Path to structural connectome directory (e.g., {study_root}/connectome/structural)
        atlas: Atlas name (default: 'desikan_killiany')
        subject_pattern: Optional glob pattern for subjects
        output_dir: Optional output directory for aligned matrices
        use_rebuilt: If True, use rebuilt matrices from probtrackx (default: True)

    Returns:
        Tuple of (matrices, subject_ids, roi_names)
        - matrices: np.ndarray of shape (n_subjects, n_rois, n_rois)
        - subject_ids: List of subject identifiers
        - roi_names: List of ROI names (intersection across subjects)
    """
    # SC uses structure: {sc_dir}/{atlas}/{subject}/
    atlas_dir = Path(sc_dir) / atlas if atlas else Path(sc_dir)

    if use_rebuilt:
        matrix_pattern = 'sc_matrix_rebuilt.npy'
        roi_pattern = 'sc_roi_names_rebuilt.txt'
    else:
        matrix_pattern = 'sc_matrix.npy'
        roi_pattern = 'sc_roi_names.txt'

    # Load connectivity with ROIs
    data = load_connectivity_with_rois(
        connectivity_dir=atlas_dir,
        matrix_pattern=matrix_pattern,
        roi_pattern=roi_pattern,
        subject_pattern=subject_pattern
    )

    if not data:
        raise ValueError(f"No SC data found in {atlas_dir}")

    # Normalize ROI names to standardized format
    for subj in data:
        data[subj]['rois'] = [normalize_roi_name(r) for r in data[subj]['rois']]

    # Compute ROI intersection
    common_rois = compute_roi_intersection(data)

    if not common_rois:
        raise ValueError("No common ROIs found across subjects")

    # Extract aligned submatrices
    extracted, roi_names = extract_common_roi_submatrices(data, common_rois, output_dir=output_dir)

    if not extracted:
        raise ValueError("No subjects successfully extracted")

    # Stack into 3D array
    subject_ids = sorted(extracted.keys())
    matrices = np.stack([extracted[s] for s in subject_ids])

    logger.info(f"Final aligned SC data: {matrices.shape}")

    return matrices, subject_ids, roi_names
