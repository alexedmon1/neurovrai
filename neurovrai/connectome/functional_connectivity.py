#!/usr/bin/env python3
"""
Functional Connectivity Module

Compute functional connectivity matrices from fMRI timeseries data.

Key Features:
- ROI-to-ROI correlation matrices (Pearson, partial correlation)
- Fisher z-transformation for statistical analysis
- Seed-based connectivity maps
- Support for multiple correlation methods
- Graph construction and thresholding

Usage:
    # Extract timeseries (from roi_extraction module)
    timeseries, roi_names = extract_roi_timeseries(
        data_file='preprocessed_bold.nii.gz',
        atlas='schaefer_400.nii.gz'
    )

    # Compute functional connectivity matrix
    fc_matrix = compute_functional_connectivity(
        timeseries=timeseries,
        method='pearson',
        fisher_z=True
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    timeseries: np.ndarray,
    method: str = 'pearson'
) -> np.ndarray:
    """
    Compute correlation matrix from timeseries data

    Args:
        timeseries: Array of shape (n_timepoints, n_rois)
        method: Correlation method ('pearson', 'spearman')

    Returns:
        Correlation matrix of shape (n_rois, n_rois)

    Raises:
        ValueError: If method is unknown or timeseries invalid
    """
    if timeseries.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")

    n_timepoints, n_rois = timeseries.shape

    logger.info(f"Computing {method} correlation matrix...")
    logger.info(f"  Timeseries shape: {timeseries.shape}")

    if method == 'pearson':
        # Use numpy's corrcoef for efficiency
        corr_matrix = np.corrcoef(timeseries.T)
    elif method == 'spearman':
        # Compute Spearman rank correlation
        corr_matrix = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(i, n_rois):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    rho, _ = spearmanr(timeseries[:, i], timeseries[:, j])
                    corr_matrix[i, j] = rho
                    corr_matrix[j, i] = rho
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    logger.info(f"  Correlation matrix shape: {corr_matrix.shape}")

    return corr_matrix


def compute_partial_correlation_matrix(
    timeseries: np.ndarray
) -> np.ndarray:
    """
    Compute partial correlation matrix using precision matrix

    Partial correlation removes the effect of all other ROIs when computing
    the correlation between two ROIs.

    Args:
        timeseries: Array of shape (n_timepoints, n_rois)

    Returns:
        Partial correlation matrix of shape (n_rois, n_rois)

    Note:
        Uses the negative normalized precision matrix. Requires n_timepoints > n_rois.
    """
    if timeseries.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")

    n_timepoints, n_rois = timeseries.shape

    if n_timepoints <= n_rois:
        raise ValueError(
            f"Partial correlation requires n_timepoints ({n_timepoints}) > "
            f"n_rois ({n_rois})"
        )

    logger.info("Computing partial correlation matrix...")
    logger.info(f"  Timeseries shape: {timeseries.shape}")

    # Compute covariance matrix
    cov_matrix = np.cov(timeseries.T)

    # Compute precision matrix (inverse covariance)
    precision_matrix = np.linalg.inv(cov_matrix)

    # Convert precision to partial correlation
    # pcorr_ij = -prec_ij / sqrt(prec_ii * prec_jj)
    diag = np.sqrt(np.diag(precision_matrix))
    partial_corr = -precision_matrix / np.outer(diag, diag)

    # Set diagonal to 1
    np.fill_diagonal(partial_corr, 1.0)

    logger.info(f"  Partial correlation matrix shape: {partial_corr.shape}")

    return partial_corr


def fisher_z_transform(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Apply Fisher z-transformation to correlation matrix

    Fisher z-transform stabilizes variance and makes correlations more
    normally distributed, improving statistical analysis.

    z = 0.5 * ln((1 + r) / (1 - r))

    Args:
        correlation_matrix: Correlation matrix (n_rois, n_rois)

    Returns:
        Z-transformed matrix (n_rois, n_rois)

    Note:
        Diagonal (self-correlations = 1.0) will be infinite, handled as NaN
    """
    # Clip correlations to valid range to avoid numerical issues
    r = np.clip(correlation_matrix, -0.9999, 0.9999)

    # Apply Fisher z-transform
    z = 0.5 * np.log((1 + r) / (1 - r))

    # Set diagonal to 0 (self-connections not meaningful)
    np.fill_diagonal(z, 0.0)

    logger.info("Applied Fisher z-transformation")

    return z


def inverse_fisher_z_transform(z_matrix: np.ndarray) -> np.ndarray:
    """
    Apply inverse Fisher z-transformation

    r = (exp(2z) - 1) / (exp(2z) + 1)

    Args:
        z_matrix: Z-transformed matrix (n_rois, n_rois)

    Returns:
        Correlation matrix (n_rois, n_rois)
    """
    exp_2z = np.exp(2 * z_matrix)
    r = (exp_2z - 1) / (exp_2z + 1)

    # Restore diagonal
    np.fill_diagonal(r, 1.0)

    logger.info("Applied inverse Fisher z-transformation")

    return r


def threshold_matrix(
    matrix: np.ndarray,
    threshold: float,
    absolute: bool = True,
    binarize: bool = False
) -> np.ndarray:
    """
    Threshold connectivity matrix

    Args:
        matrix: Connectivity matrix (n_rois, n_rois)
        threshold: Threshold value
        absolute: If True, threshold absolute values (ignores sign)
        binarize: If True, return binary matrix (1 for above threshold, 0 below)

    Returns:
        Thresholded matrix
    """
    thresholded = matrix.copy()

    if absolute:
        mask = np.abs(thresholded) < threshold
    else:
        mask = thresholded < threshold

    thresholded[mask] = 0

    if binarize:
        thresholded = (np.abs(thresholded) >= threshold).astype(float)

    # Restore diagonal
    np.fill_diagonal(thresholded, 1.0 if not binarize else 1.0)

    logger.info(f"Thresholded matrix at {threshold} ({'absolute' if absolute else 'raw'})")
    logger.info(f"  Retained {np.sum(~mask) / 2} edges (upper triangle)")

    return thresholded


def compute_functional_connectivity(
    timeseries: np.ndarray,
    roi_names: Optional[List[str]] = None,
    method: str = 'pearson',
    fisher_z: bool = True,
    partial: bool = False,
    threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
    output_prefix: str = 'fc'
) -> Dict:
    """
    Compute functional connectivity matrix from timeseries

    Main function that orchestrates connectivity computation with various options.

    Args:
        timeseries: Array of shape (n_timepoints, n_rois)
        roi_names: Optional list of ROI names for labeling
        method: Correlation method ('pearson', 'spearman')
        fisher_z: Apply Fisher z-transformation
        partial: Compute partial correlation instead of Pearson/Spearman
        threshold: Optional threshold for sparsifying matrix
        output_dir: Optional directory to save outputs
        output_prefix: Prefix for output files

    Returns:
        Dictionary containing:
            - connectivity_matrix: Connectivity matrix (n_rois, n_rois)
            - roi_names: List of ROI names
            - method: Method used
            - fisher_z: Whether Fisher z-transform was applied
            - threshold: Threshold value if applied
            - output_files: Dict of saved file paths (if output_dir provided)

    Example:
        results = compute_functional_connectivity(
            timeseries=ts,
            roi_names=names,
            method='pearson',
            fisher_z=True,
            output_dir=Path('/analysis/fc/')
        )
    """
    if timeseries.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")

    n_timepoints, n_rois = timeseries.shape

    logger.info("=" * 80)
    logger.info("FUNCTIONAL CONNECTIVITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Timeseries shape: {timeseries.shape}")
    logger.info(f"Method: {'partial correlation' if partial else method}")
    logger.info(f"Fisher z-transform: {fisher_z}")
    logger.info(f"Threshold: {threshold if threshold else 'None'}")

    # Generate ROI names if not provided
    if roi_names is None:
        roi_names = [f"ROI_{i:03d}" for i in range(n_rois)]

    if len(roi_names) != n_rois:
        raise ValueError(
            f"Number of ROI names ({len(roi_names)}) doesn't match "
            f"number of ROIs ({n_rois})"
        )

    # Compute connectivity matrix
    if partial:
        connectivity_matrix = compute_partial_correlation_matrix(timeseries)
        method_used = 'partial_correlation'
    else:
        connectivity_matrix = compute_correlation_matrix(timeseries, method=method)
        method_used = method

    # Apply Fisher z-transform if requested
    if fisher_z:
        connectivity_matrix = fisher_z_transform(connectivity_matrix)

    # Apply threshold if requested
    if threshold is not None:
        connectivity_matrix = threshold_matrix(
            connectivity_matrix,
            threshold=threshold,
            absolute=True,
            binarize=False
        )

    # Compute summary statistics
    # Get upper triangle (excluding diagonal)
    upper_triangle = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
    nonzero_edges = upper_triangle[upper_triangle != 0]

    summary = {
        'n_rois': n_rois,
        'n_edges_total': len(upper_triangle),
        'n_edges_nonzero': len(nonzero_edges),
        'mean_connectivity': float(np.mean(nonzero_edges)) if len(nonzero_edges) > 0 else 0.0,
        'std_connectivity': float(np.std(nonzero_edges)) if len(nonzero_edges) > 0 else 0.0,
        'min_connectivity': float(np.min(nonzero_edges)) if len(nonzero_edges) > 0 else 0.0,
        'max_connectivity': float(np.max(nonzero_edges)) if len(nonzero_edges) > 0 else 0.0,
    }

    logger.info("\nSummary Statistics:")
    logger.info(f"  ROIs: {summary['n_rois']}")
    logger.info(f"  Edges (non-zero): {summary['n_edges_nonzero']} / {summary['n_edges_total']}")
    logger.info(f"  Mean connectivity: {summary['mean_connectivity']:.4f}")
    logger.info(f"  Std connectivity: {summary['std_connectivity']:.4f}")
    logger.info(f"  Range: [{summary['min_connectivity']:.4f}, {summary['max_connectivity']:.4f}]")

    # Save outputs if directory provided
    output_files = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving outputs to: {output_dir}")

        # Save connectivity matrix as numpy array
        matrix_file = output_dir / f"{output_prefix}_matrix.npy"
        np.save(matrix_file, connectivity_matrix)
        output_files['matrix_npy'] = str(matrix_file)
        logger.info(f"  Matrix (numpy): {matrix_file.name}")

        # Save as CSV with ROI labels
        df = pd.DataFrame(
            connectivity_matrix,
            index=roi_names,
            columns=roi_names
        )
        csv_file = output_dir / f"{output_prefix}_matrix.csv"
        df.to_csv(csv_file)
        output_files['matrix_csv'] = str(csv_file)
        logger.info(f"  Matrix (CSV): {csv_file.name}")

        # Save ROI names
        names_file = output_dir / f"{output_prefix}_roi_names.txt"
        with open(names_file, 'w') as f:
            for name in roi_names:
                f.write(f"{name}\n")
        output_files['roi_names'] = str(names_file)
        logger.info(f"  ROI names: {names_file.name}")

        # Save summary statistics
        summary_file = output_dir / f"{output_prefix}_summary.json"
        summary_full = {
            'method': method_used,
            'fisher_z': fisher_z,
            'threshold': threshold,
            'n_timepoints': n_timepoints,
            **summary
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_full, f, indent=2)
        output_files['summary'] = str(summary_file)
        logger.info(f"  Summary: {summary_file.name}")

    logger.info("=" * 80)
    logger.info("FUNCTIONAL CONNECTIVITY ANALYSIS COMPLETE")
    logger.info("=" * 80)

    return {
        'connectivity_matrix': connectivity_matrix,
        'roi_names': roi_names,
        'method': method_used,
        'fisher_z': fisher_z,
        'threshold': threshold,
        'summary': summary,
        'output_files': output_files
    }


def compute_seed_connectivity(
    timeseries: np.ndarray,
    seed_timeseries: np.ndarray,
    method: str = 'pearson',
    fisher_z: bool = True
) -> np.ndarray:
    """
    Compute seed-based connectivity

    Correlate a seed timeseries with all ROI timeseries.

    Args:
        timeseries: Array of shape (n_timepoints, n_rois)
        seed_timeseries: Seed timeseries of shape (n_timepoints,)
        method: Correlation method ('pearson', 'spearman')
        fisher_z: Apply Fisher z-transformation

    Returns:
        Seed connectivity vector of shape (n_rois,)
    """
    if timeseries.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")

    if seed_timeseries.ndim != 1:
        raise ValueError(f"Expected 1D seed timeseries, got shape {seed_timeseries.shape}")

    n_timepoints, n_rois = timeseries.shape

    if len(seed_timeseries) != n_timepoints:
        raise ValueError(
            f"Seed timeseries length ({len(seed_timeseries)}) doesn't match "
            f"data timepoints ({n_timepoints})"
        )

    logger.info("Computing seed-based connectivity...")
    logger.info(f"  Timeseries shape: {timeseries.shape}")
    logger.info(f"  Method: {method}")

    # Compute correlation with seed
    seed_connectivity = np.zeros(n_rois)
    for i in range(n_rois):
        if method == 'pearson':
            r, _ = pearsonr(seed_timeseries, timeseries[:, i])
        elif method == 'spearman':
            r, _ = spearmanr(seed_timeseries, timeseries[:, i])
        else:
            raise ValueError(f"Unknown method: {method}")

        seed_connectivity[i] = r

    # Apply Fisher z-transform if requested
    if fisher_z:
        seed_connectivity = 0.5 * np.log((1 + seed_connectivity) / (1 - seed_connectivity))

    logger.info(f"  Seed connectivity range: [{np.min(seed_connectivity):.4f}, {np.max(seed_connectivity):.4f}]")

    return seed_connectivity
