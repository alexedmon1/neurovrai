#!/usr/bin/env python3
"""
Regional Homogeneity (ReHo) Analysis for Resting-State fMRI

ReHo measures the similarity/synchronization of time series in a local region
using Kendall's coefficient of concordance (KCC). Higher ReHo values indicate
more synchronized activity in that region.

References:
- Zang et al. (2004). Regional homogeneity approach to fMRI data analysis.
  NeuroImage, 22(1), 394-400.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy import stats
import logging


def compute_kendall_concordance(timeseries: np.ndarray) -> float:
    """
    Compute Kendall's coefficient of concordance (KCC) for multiple time series

    Args:
        timeseries: Array of shape (n_voxels, n_timepoints)
                   Each row is a time series from one voxel

    Returns:
        KCC value between 0 and 1
        - 1 = perfect agreement
        - 0 = no agreement

    Formula:
        W = 12 * sum(Ri^2) / (K^2 * (N^3 - N))
        where:
            K = number of raters (voxels)
            N = number of items (timepoints)
            Ri = sum of ranks for timepoint i
    """
    n_voxels, n_timepoints = timeseries.shape

    if n_voxels < 2:
        return 0.0

    # Rank each voxel's time series
    # ranks shape: (n_voxels, n_timepoints)
    ranks = np.array([stats.rankdata(ts) for ts in timeseries])

    # Sum ranks across voxels for each timepoint
    rank_sums = np.sum(ranks, axis=0)

    # Mean of rank sums
    mean_rank_sum = np.mean(rank_sums)

    # Sum of squared deviations
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)

    # Kendall's W
    W = (12 * ss) / (n_voxels ** 2 * (n_timepoints ** 3 - n_timepoints))

    return float(W)


def get_neighborhood_timeseries(data_4d: np.ndarray,
                                x: int, y: int, z: int,
                                neighborhood: int = 27,
                                mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract time series from voxel neighborhood

    Args:
        data_4d: 4D functional data (x, y, z, time)
        x, y, z: Center voxel coordinates
        neighborhood: Number of neighbors (7, 19, or 27)
                     7  = face neighbors (1 voxel away)
                     19 = face + edge neighbors
                     27 = full 3x3x3 cube (default)
        mask: Optional brain mask

    Returns:
        Array of shape (n_neighbors, n_timepoints)
    """
    nx, ny, nz, nt = data_4d.shape

    # Define neighborhood offsets
    if neighborhood == 7:
        # 6 face neighbors + center
        offsets = [
            (0, 0, 0),   # center
            (-1, 0, 0), (1, 0, 0),   # x neighbors
            (0, -1, 0), (0, 1, 0),   # y neighbors
            (0, 0, -1), (0, 0, 1)    # z neighbors
        ]
    elif neighborhood == 19:
        # 18 face + edge neighbors + center
        offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if abs(dx) + abs(dy) + abs(dz) <= 2:
                        offsets.append((dx, dy, dz))
    else:  # 27
        # Full 3x3x3 cube
        offsets = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
        ]

    # Extract time series from valid neighbors
    timeseries_list = []
    for dx, dy, dz in offsets:
        nx_coord = x + dx
        ny_coord = y + dy
        nz_coord = z + dz

        # Check bounds
        if (0 <= nx_coord < nx and
            0 <= ny_coord < ny and
            0 <= nz_coord < nz):

            # Check mask if provided
            if mask is not None and not mask[nx_coord, ny_coord, nz_coord]:
                continue

            # Extract time series
            ts = data_4d[nx_coord, ny_coord, nz_coord, :]

            # Skip if all zeros or constant
            if np.std(ts) > 0:
                timeseries_list.append(ts)

    if len(timeseries_list) < 2:
        return np.array([])

    return np.array(timeseries_list)


def compute_reho_map(func_file: Path,
                     mask_file: Optional[Path] = None,
                     neighborhood: int = 27,
                     output_file: Optional[Path] = None) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Compute ReHo map for whole brain

    Args:
        func_file: Path to preprocessed 4D functional image
                  Should be detrended, filtered, and nuisance-regressed
        mask_file: Optional brain mask
        neighborhood: Neighborhood size (7, 19, or 27)
        output_file: Optional path to save ReHo map

    Returns:
        Tuple of (reho_data, reho_img)
        - reho_data: 3D array of ReHo values
        - reho_img: NIfTI image object
    """
    logging.info("=" * 80)
    logging.info("Computing ReHo (Regional Homogeneity)")
    logging.info("=" * 80)
    logging.info(f"Input: {func_file}")
    logging.info(f"Neighborhood: {neighborhood} voxels")

    # Load functional data
    func_img = nib.load(func_file)
    func_data = func_img.get_fdata()

    if func_data.ndim != 4:
        raise ValueError(f"Expected 4D functional data, got {func_data.ndim}D")

    nx, ny, nz, nt = func_data.shape
    logging.info(f"Dimensions: {nx} x {ny} x {nz} x {nt} timepoints")

    # Load mask
    if mask_file is not None:
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata().astype(bool)
        logging.info(f"Using brain mask: {mask_file}")
        logging.info(f"  Brain voxels: {np.sum(mask_data)}")
    else:
        # Create mask from non-zero voxels
        mask_data = np.any(func_data != 0, axis=-1)
        logging.info("No mask provided, using non-zero voxels")
        logging.info(f"  Brain voxels: {np.sum(mask_data)}")

    # Initialize ReHo map
    reho_data = np.zeros((nx, ny, nz), dtype=np.float32)

    # Compute ReHo for each voxel
    brain_voxels = np.where(mask_data)
    n_voxels = len(brain_voxels[0])

    logging.info(f"Computing ReHo for {n_voxels} voxels...")

    for i, (x, y, z) in enumerate(zip(*brain_voxels)):
        if (i + 1) % 10000 == 0:
            logging.info(f"  Progress: {i+1}/{n_voxels} voxels ({100*(i+1)/n_voxels:.1f}%)")

        # Get neighborhood time series
        neighborhood_ts = get_neighborhood_timeseries(
            func_data, x, y, z,
            neighborhood=neighborhood,
            mask=mask_data
        )

        if len(neighborhood_ts) < 2:
            continue

        # Compute KCC
        kcc = compute_kendall_concordance(neighborhood_ts)
        reho_data[x, y, z] = kcc

    logging.info("  ✓ ReHo computation complete")

    # Create output image
    reho_img = nib.Nifti1Image(reho_data, func_img.affine, func_img.header)

    # Save if requested
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        nib.save(reho_img, output_file)
        logging.info(f"  Saved: {output_file}")

    # Compute statistics
    brain_reho = reho_data[mask_data]
    logging.info("\nReHo Statistics:")
    logging.info(f"  Mean: {np.mean(brain_reho):.4f}")
    logging.info(f"  Std:  {np.std(brain_reho):.4f}")
    logging.info(f"  Min:  {np.min(brain_reho):.4f}")
    logging.info(f"  Max:  {np.max(brain_reho):.4f}")
    logging.info("=" * 80)

    return reho_data, reho_img


def compute_reho_zscore(reho_file: Path,
                       mask_file: Optional[Path] = None,
                       output_file: Optional[Path] = None) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Standardize ReHo map to z-scores

    Z-score normalization makes ReHo values comparable across subjects

    Args:
        reho_file: Path to ReHo map
        mask_file: Optional brain mask
        output_file: Optional path to save z-scored map

    Returns:
        Tuple of (zscore_data, zscore_img)
    """
    logging.info("Standardizing ReHo to z-scores...")

    # Load ReHo map
    reho_img = nib.load(reho_file)
    reho_data = reho_img.get_fdata()

    # Load mask
    if mask_file is not None:
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata().astype(bool)
    else:
        mask_data = reho_data > 0

    # Compute mean and std within brain
    brain_reho = reho_data[mask_data]
    mean_reho = np.mean(brain_reho)
    std_reho = np.std(brain_reho)

    # Z-score
    zscore_data = np.zeros_like(reho_data)
    zscore_data[mask_data] = (brain_reho - mean_reho) / std_reho

    # Create output image
    zscore_img = nib.Nifti1Image(zscore_data, reho_img.affine, reho_img.header)

    # Save if requested
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        nib.save(zscore_img, output_file)
        logging.info(f"  Saved z-scored ReHo: {output_file}")

    return zscore_data, zscore_img


if __name__ == '__main__':
    import argparse

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Compute Regional Homogeneity (ReHo) for resting-state fMRI"
    )
    parser.add_argument(
        '--func',
        type=Path,
        required=True,
        help='Preprocessed 4D functional image'
    )
    parser.add_argument(
        '--mask',
        type=Path,
        help='Brain mask'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output ReHo map'
    )
    parser.add_argument(
        '--neighborhood',
        type=int,
        default=27,
        choices=[7, 19, 27],
        help='Neighborhood size (default: 27)'
    )
    parser.add_argument(
        '--zscore',
        action='store_true',
        help='Also save z-scored ReHo map'
    )

    args = parser.parse_args()

    # Compute ReHo
    reho_data, reho_img = compute_reho_map(
        func_file=args.func,
        mask_file=args.mask,
        neighborhood=args.neighborhood,
        output_file=args.output
    )

    # Compute z-scored version
    if args.zscore:
        zscore_output = args.output.parent / f"{args.output.stem}_zscore.nii.gz"
        compute_reho_zscore(
            reho_file=args.output,
            mask_file=args.mask,
            output_file=zscore_output
        )
