#!/usr/bin/env python3
"""
Quality Control (QC) utilities for functional MRI preprocessing.

This module provides functions for:
1. Motion assessment and visualization
2. Temporal SNR (tSNR) calculation
3. Registration quality checks
4. Skull stripping quality assessment
5. Automated QC report generation
6. Outlier detection (framewise displacement, DVARS)

Usage:
    from neurovrai.preprocess.qc.func_qc import compute_motion_qc, compute_tsnr, compute_skull_strip_qc, generate_func_qc_report
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_motion_qc(
    motion_file: Path,
    tr: float,
    output_dir: Path,
    fd_threshold: float = 0.5,
    dvars_threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Compute motion QC metrics and create visualizations.

    Parameters
    ----------
    motion_file : Path
        FSL motion parameters file (.par) from MCFLIRT
        Format: 6 columns (3 rotations in radians, 3 translations in mm)
    tr : float
        Repetition time in seconds
    output_dir : Path
        Output directory for QC files
    fd_threshold : float
        Framewise displacement threshold in mm (default: 0.5mm)
    dvars_threshold : float
        DVARS threshold (default: 1.5)

    Returns
    -------
    dict
        Motion QC metrics:
        - mean_fd: Mean framewise displacement
        - max_fd: Maximum framewise displacement
        - n_outliers_fd: Number of volumes exceeding FD threshold
        - mean_rotation: Mean absolute rotation (degrees)
        - mean_translation: Mean absolute translation (mm)
        - motion_plot: Path to motion plot
        - fd_plot: Path to FD plot
    """
    logger.info("=" * 70)
    logger.info("Motion QC Assessment")
    logger.info("=" * 70)
    logger.info(f"Motion file: {motion_file}")
    logger.info(f"TR: {tr}s")
    logger.info(f"FD threshold: {fd_threshold}mm")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load motion parameters
    # FSL format: rot_x, rot_y, rot_z (radians), trans_x, trans_y, trans_z (mm)
    motion_params = np.loadtxt(motion_file)
    n_volumes = motion_params.shape[0]

    # Convert rotations from radians to degrees
    rotations_deg = np.rad2deg(motion_params[:, :3])
    translations_mm = motion_params[:, 3:]

    # Calculate absolute displacements
    abs_rot = np.abs(rotations_deg)
    abs_trans = np.abs(translations_mm)

    # Calculate framewise displacement (FD)
    # FD = sum of absolute derivatives of 6 motion parameters
    # Rotations converted to mm assuming 50mm radius sphere
    fd = compute_framewise_displacement(motion_params)

    # Identify outlier volumes
    outliers_fd = fd > fd_threshold
    n_outliers = np.sum(outliers_fd)

    # Calculate summary metrics
    metrics = {
        'n_volumes': n_volumes,
        'mean_fd': np.mean(fd),
        'median_fd': np.median(fd),
        'max_fd': np.max(fd),
        'std_fd': np.std(fd),
        'n_outliers_fd': n_outliers,
        'percent_outliers': (n_outliers / n_volumes) * 100,
        'mean_rotation': np.mean(abs_rot),
        'max_rotation': np.max(abs_rot),
        'mean_translation': np.mean(abs_trans),
        'max_translation': np.max(abs_trans)
    }

    logger.info("Motion Summary:")
    logger.info(f"  Total volumes: {n_volumes}")
    logger.info(f"  Mean FD: {metrics['mean_fd']:.3f} mm")
    logger.info(f"  Max FD: {metrics['max_fd']:.3f} mm")
    logger.info(f"  Outlier volumes (FD>{fd_threshold}mm): {n_outliers} ({metrics['percent_outliers']:.1f}%)")
    logger.info(f"  Mean rotation: {metrics['mean_rotation']:.3f}°")
    logger.info(f"  Mean translation: {metrics['mean_translation']:.3f} mm")
    logger.info("")

    # Create motion plots
    logger.info("Creating motion visualization plots...")

    # Plot 1: Motion parameters over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Rotations
    timepoints = np.arange(n_volumes) * tr
    axes[0].plot(timepoints, rotations_deg[:, 0], 'r-', label='Roll (X)', linewidth=1)
    axes[0].plot(timepoints, rotations_deg[:, 1], 'g-', label='Pitch (Y)', linewidth=1)
    axes[0].plot(timepoints, rotations_deg[:, 2], 'b-', label='Yaw (Z)', linewidth=1)
    axes[0].set_ylabel('Rotation (degrees)', fontsize=12)
    axes[0].set_title('Head Motion Parameters', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # Translations
    axes[1].plot(timepoints, translations_mm[:, 0], 'r-', label='X', linewidth=1)
    axes[1].plot(timepoints, translations_mm[:, 1], 'g-', label='Y', linewidth=1)
    axes[1].plot(timepoints, translations_mm[:, 2], 'b-', label='Z', linewidth=1)
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Translation (mm)', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    motion_plot = output_dir / 'motion_parameters.png'
    plt.savefig(motion_plot, dpi=150, bbox_inches='tight')
    plt.close()
    metrics['motion_plot'] = motion_plot
    logger.info(f"  Saved: {motion_plot}")

    # Plot 2: Framewise Displacement
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timepoints, fd, 'k-', linewidth=1.5, label='FD')
    ax.axhline(y=fd_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({fd_threshold}mm)')
    ax.fill_between(timepoints, 0, fd, where=outliers_fd, color='red', alpha=0.3, label='Outliers')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Framewise Displacement (mm)', fontsize=12)
    ax.set_title(f'Framewise Displacement (Mean={metrics["mean_fd"]:.3f}mm, Outliers={n_outliers})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fd_plot = output_dir / 'framewise_displacement.png'
    plt.savefig(fd_plot, dpi=150, bbox_inches='tight')
    plt.close()
    metrics['fd_plot'] = fd_plot
    logger.info(f"  Saved: {fd_plot}")

    # Save motion metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = output_dir / 'motion_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    metrics['metrics_csv'] = metrics_csv
    logger.info(f"  Saved: {metrics_csv}")
    logger.info("")

    return metrics


def compute_framewise_displacement(motion_params: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """
    Compute framewise displacement from motion parameters.

    Parameters
    ----------
    motion_params : ndarray
        Motion parameters (N x 6): [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        Rotations in radians, translations in mm
    radius : float
        Head radius in mm for converting rotations to displacements (default: 50mm)

    Returns
    -------
    ndarray
        Framewise displacement for each volume (length N)
    """
    # Calculate derivatives (differences between consecutive timepoints)
    derivatives = np.diff(motion_params, axis=0)

    # Convert rotations (radians) to mm using arc length: s = r * theta
    rot_disp = radius * np.abs(derivatives[:, :3])
    trans_disp = np.abs(derivatives[:, 3:])

    # FD = sum of absolute displacements
    fd = np.sum(rot_disp, axis=1) + np.sum(trans_disp, axis=1)

    # Prepend 0 for first volume (no derivative)
    fd = np.insert(fd, 0, 0)

    return fd


def compute_tsnr(
    func_file: Path,
    mask_file: Optional[Path],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Compute temporal Signal-to-Noise Ratio (tSNR).

    tSNR = mean(signal) / std(signal) over time

    Parameters
    ----------
    func_file : Path
        Functional 4D image file
    mask_file : Path, optional
        Brain mask file
    output_dir : Path
        Output directory for tSNR maps

    Returns
    -------
    dict
        tSNR metrics:
        - mean_tsnr: Mean tSNR in brain
        - median_tsnr: Median tSNR in brain
        - tsnr_map: Path to tSNR map
        - tsnr_histogram: Path to histogram plot
    """
    logger.info("=" * 70)
    logger.info("tSNR Calculation")
    logger.info("=" * 70)
    logger.info(f"Functional file: {func_file}")
    logger.info(f"Mask file: {mask_file}")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute tSNR using FSL
    tsnr_file = output_dir / 'tsnr.nii.gz'
    mean_file = output_dir / 'mean.nii.gz'
    std_file = output_dir / 'std.nii.gz'

    # Temporal mean
    cmd_mean = ['fslmaths', str(func_file), '-Tmean', str(mean_file)]
    subprocess.run(cmd_mean, check=True, capture_output=True)

    # Temporal std
    cmd_std = ['fslmaths', str(func_file), '-Tstd', str(std_file)]
    subprocess.run(cmd_std, check=True, capture_output=True)

    # tSNR = mean / std
    cmd_tsnr = ['fslmaths', str(mean_file), '-div', str(std_file), str(tsnr_file)]
    subprocess.run(cmd_tsnr, check=True, capture_output=True)

    logger.info(f"  Computed tSNR map: {tsnr_file}")

    # Load tSNR map
    tsnr_img = nib.load(tsnr_file)
    tsnr_data = tsnr_img.get_fdata()

    # Apply mask if provided
    if mask_file and mask_file.exists():
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata() > 0
        tsnr_masked = tsnr_data[mask_data]
    else:
        # Use non-zero voxels as mask
        tsnr_masked = tsnr_data[tsnr_data > 0]

    # Calculate metrics
    metrics = {
        'mean_tsnr': np.mean(tsnr_masked),
        'median_tsnr': np.median(tsnr_masked),
        'std_tsnr': np.std(tsnr_masked),
        'min_tsnr': np.min(tsnr_masked),
        'max_tsnr': np.max(tsnr_masked),
        'tsnr_map': tsnr_file,
        'mean_map': mean_file,
        'std_map': std_file
    }

    logger.info("tSNR Summary:")
    logger.info(f"  Mean tSNR: {metrics['mean_tsnr']:.2f}")
    logger.info(f"  Median tSNR: {metrics['median_tsnr']:.2f}")
    logger.info(f"  Range: {metrics['min_tsnr']:.2f} - {metrics['max_tsnr']:.2f}")
    logger.info("")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tsnr_masked, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(metrics['mean_tsnr'], color='r', linestyle='--', linewidth=2, label=f"Mean={metrics['mean_tsnr']:.2f}")
    ax.axvline(metrics['median_tsnr'], color='g', linestyle='--', linewidth=2, label=f"Median={metrics['median_tsnr']:.2f}")
    ax.set_xlabel('tSNR', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Temporal SNR Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_plot = output_dir / 'tsnr_histogram.png'
    plt.savefig(hist_plot, dpi=150, bbox_inches='tight')
    plt.close()
    metrics['tsnr_histogram'] = hist_plot
    logger.info(f"  Saved histogram: {hist_plot}")
    logger.info("")

    return metrics


def compute_dvars(
    func_file: Path,
    mask_file: Optional[Path],
    output_dir: Path,
    dvars_threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Compute DVARS (spatial standard deviation of temporal derivative).

    DVARS quantifies how much the brain intensity changes from one volume to the next.
    High DVARS values indicate artifacts or sudden intensity changes.

    Parameters
    ----------
    func_file : Path
        Functional 4D image file
    mask_file : Path, optional
        Brain mask file
    output_dir : Path
        Output directory for DVARS plots
    dvars_threshold : float
        DVARS threshold for outlier detection (default: 1.5 standard deviations)

    Returns
    -------
    dict
        DVARS metrics:
        - dvars: DVARS time series
        - mean_dvars: Mean DVARS
        - std_dvars: Standard deviation of DVARS
        - n_outliers_dvars: Number of volumes exceeding threshold
        - dvars_plot: Path to DVARS plot
    """
    logger.info("=" * 70)
    logger.info("DVARS Computation")
    logger.info("=" * 70)
    logger.info(f"Functional file: {func_file}")
    logger.info(f"Mask file: {mask_file}")
    logger.info(f"DVARS threshold: {dvars_threshold} SD")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load functional data
    func_img = nib.load(func_file)
    func_data = func_img.get_fdata()
    n_volumes = func_data.shape[3]

    # Load mask
    if mask_file and mask_file.exists():
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata() > 0
    else:
        # Create mask from mean signal
        mean_img = np.mean(func_data, axis=3)
        mask_data = mean_img > (0.1 * np.max(mean_img))

    n_voxels = np.sum(mask_data)
    logger.info(f"  Total volumes: {n_volumes}")
    logger.info(f"  Masked voxels: {n_voxels}")

    # Compute DVARS
    # DVARS = sqrt(mean(diff^2)) where diff is the temporal derivative
    dvars = np.zeros(n_volumes)

    for t in range(1, n_volumes):
        # Temporal derivative
        diff = func_data[:, :, :, t] - func_data[:, :, :, t-1]
        # Masked squared differences
        diff_squared = diff[mask_data] ** 2
        # DVARS = sqrt(mean(diff^2))
        dvars[t] = np.sqrt(np.mean(diff_squared))

    # Standardize DVARS (divide by median for robustness)
    dvars_robust = dvars / np.median(dvars[dvars > 0])

    # Identify outliers (DVARS > threshold * std)
    outliers = dvars_robust > dvars_threshold
    n_outliers = np.sum(outliers)

    # Calculate metrics
    metrics = {
        'dvars': dvars,
        'dvars_robust': dvars_robust,
        'mean_dvars': np.mean(dvars[1:]),  # Skip first volume (always 0)
        'median_dvars': np.median(dvars[1:]),
        'std_dvars': np.std(dvars[1:]),
        'max_dvars': np.max(dvars),
        'n_outliers_dvars': n_outliers,
        'percent_outliers_dvars': (n_outliers / n_volumes) * 100
    }

    logger.info("DVARS Summary:")
    logger.info(f"  Mean DVARS: {metrics['mean_dvars']:.2f}")
    logger.info(f"  Median DVARS: {metrics['median_dvars']:.2f}")
    logger.info(f"  Max DVARS: {metrics['max_dvars']:.2f}")
    logger.info(f"  Outlier volumes: {n_outliers} ({metrics['percent_outliers_dvars']:.1f}%)")
    logger.info("")

    # Create DVARS plot
    logger.info("Creating DVARS visualization...")
    fig, ax = plt.subplots(figsize=(12, 5))

    timepoints = np.arange(n_volumes)
    ax.plot(timepoints, dvars_robust, 'k-', linewidth=1.5, label='Standardized DVARS')
    ax.axhline(y=dvars_threshold, color='r', linestyle='--', linewidth=2,
               label=f'Threshold ({dvars_threshold} SD)')
    ax.fill_between(timepoints, 0, dvars_robust, where=outliers,
                     color='red', alpha=0.3, label='Outliers')
    ax.set_xlabel('Volume', fontsize=12)
    ax.set_ylabel('Standardized DVARS', fontsize=12)
    ax.set_title(f'DVARS (Mean={metrics["mean_dvars"]:.2f}, Outliers={n_outliers})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    dvars_plot = output_dir / 'dvars.png'
    plt.savefig(dvars_plot, dpi=150, bbox_inches='tight')
    plt.close()
    metrics['dvars_plot'] = dvars_plot
    logger.info(f"  Saved: {dvars_plot}")

    # Save DVARS to CSV
    dvars_df = pd.DataFrame({
        'volume': timepoints,
        'dvars': dvars,
        'dvars_standardized': dvars_robust,
        'outlier': outliers
    })
    dvars_csv = output_dir / 'dvars.csv'
    dvars_df.to_csv(dvars_csv, index=False)
    metrics['dvars_csv'] = dvars_csv
    logger.info(f"  Saved: {dvars_csv}")
    logger.info("")

    return metrics


def create_carpet_plot(
    func_file: Path,
    mask_file: Optional[Path],
    motion_file: Optional[Path],
    output_dir: Path,
    tr: float = 1.0
) -> Dict[str, Any]:
    """
    Create carpet plot (voxel intensity time series visualization).

    A carpet plot displays voxel intensities over time, organized by tissue type.
    Useful for identifying global signal fluctuations and artifacts.

    Parameters
    ----------
    func_file : Path
        Functional 4D image file
    mask_file : Path, optional
        Brain mask file
    motion_file : Path, optional
        Motion parameters file (.par) for overlaying FD
    output_dir : Path
        Output directory for carpet plot
    tr : float
        Repetition time in seconds

    Returns
    -------
    dict
        Carpet plot info:
        - carpet_plot: Path to carpet plot image
    """
    logger.info("=" * 70)
    logger.info("Carpet Plot Generation")
    logger.info("=" * 70)
    logger.info(f"Functional file: {func_file}")
    logger.info(f"Mask file: {mask_file}")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load functional data
    func_img = nib.load(func_file)
    func_data = func_img.get_fdata()
    n_volumes = func_data.shape[3]

    # Load mask
    if mask_file and mask_file.exists():
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata() > 0
    else:
        # Create mask from mean signal
        mean_img = np.mean(func_data, axis=3)
        mask_data = mean_img > (0.1 * np.max(mean_img))

    # Extract voxel time series within mask
    voxel_timeseries = func_data[mask_data, :]  # Shape: (n_voxels, n_volumes)

    # Downsample voxels for visualization (max 5000 voxels)
    max_voxels = 5000
    if voxel_timeseries.shape[0] > max_voxels:
        # Random sampling
        indices = np.random.choice(voxel_timeseries.shape[0], max_voxels, replace=False)
        voxel_timeseries = voxel_timeseries[indices, :]

    # Z-score normalize each voxel's time series
    voxel_timeseries_norm = (voxel_timeseries - np.mean(voxel_timeseries, axis=1, keepdims=True)) / \
                             (np.std(voxel_timeseries, axis=1, keepdims=True) + 1e-10)

    # Sort voxels by mean intensity for better visualization
    mean_intensity = np.mean(voxel_timeseries, axis=1)
    sort_idx = np.argsort(mean_intensity)
    voxel_timeseries_sorted = voxel_timeseries_norm[sort_idx, :]

    logger.info(f"  Displaying {voxel_timeseries_sorted.shape[0]} voxels")
    logger.info(f"  Time series length: {n_volumes} volumes")

    # Create carpet plot
    logger.info("Creating carpet plot visualization...")
    fig = plt.figure(figsize=(15, 8))

    # Main carpet plot
    ax_carpet = plt.subplot2grid((5, 1), (1, 0), rowspan=4)

    timepoints = np.arange(n_volumes) * tr
    im = ax_carpet.imshow(voxel_timeseries_sorted, aspect='auto', cmap='gray',
                           extent=[0, timepoints[-1], 0, voxel_timeseries_sorted.shape[0]],
                           interpolation='nearest', vmin=-3, vmax=3)
    ax_carpet.set_xlabel('Time (seconds)', fontsize=12)
    ax_carpet.set_ylabel('Voxels (sorted by intensity)', fontsize=12)
    ax_carpet.set_title('Carpet Plot: Voxel Intensity Time Series', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_carpet, orientation='vertical', pad=0.01)
    cbar.set_label('Z-scored Intensity', fontsize=10)

    # Optional: Overlay framewise displacement on top
    ax_fd = plt.subplot2grid((5, 1), (0, 0), sharex=ax_carpet)
    if motion_file and motion_file.exists():
        motion_params = np.loadtxt(motion_file)
        fd = compute_framewise_displacement(motion_params)
        ax_fd.plot(timepoints, fd, 'k-', linewidth=1.5)
        ax_fd.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax_fd.set_ylabel('FD (mm)', fontsize=10)
        ax_fd.set_title('Framewise Displacement', fontsize=12)
        ax_fd.grid(True, alpha=0.3)
        ax_fd.set_xlim([0, timepoints[-1]])
    else:
        ax_fd.text(0.5, 0.5, 'No motion data available',
                   ha='center', va='center', transform=ax_fd.transAxes, fontsize=12)
        ax_fd.set_yticks([])

    plt.setp(ax_fd.get_xticklabels(), visible=False)

    plt.tight_layout()
    carpet_plot = output_dir / 'carpet_plot.png'
    plt.savefig(carpet_plot, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {carpet_plot}")
    logger.info("")

    metrics = {
        'carpet_plot': carpet_plot,
        'n_voxels_displayed': voxel_timeseries_sorted.shape[0],
        'n_volumes': n_volumes
    }

    return metrics


def compute_skull_strip_qc(
    func_mean_file: Path,
    mask_file: Path,
    output_dir: Path,
    subject: str = "unknown"
) -> Dict[str, Any]:
    """
    Compute skull stripping quality metrics for functional data.

    Parameters
    ----------
    func_mean_file : Path
        Mean functional image (temporal mean of 4D functional data)
    mask_file : Path
        Brain mask from BET
    output_dir : Path
        Output directory for QC files
    subject : str
        Subject identifier (for labeling)

    Returns
    -------
    dict
        Skull stripping QC metrics:
        - brain_volume_mm3: Brain volume in mm³
        - brain_volume_cm3: Brain volume in cm³
        - n_voxels: Number of brain voxels
        - contrast_ratio: Ratio of brain to non-brain mean intensity
        - quality_flags: List of quality warnings
        - quality_pass: Boolean indicating if quality checks passed
        - mask_overlay: Path to overlay visualization
        - metrics_json: Path to saved metrics JSON
    """
    logger.info("=" * 70)
    logger.info("Functional Skull Strip QC Assessment")
    logger.info("=" * 70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Mean func file: {func_mean_file}")
    logger.info(f"Mask file: {mask_file}")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    func_img = nib.load(func_mean_file)
    func_data = func_img.get_fdata()
    voxel_size = func_img.header.get_zooms()[:3]

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata() > 0

    # Calculate mask statistics
    n_voxels = int(np.sum(mask_data))
    voxel_volume_mm3 = np.prod(voxel_size)
    brain_volume_mm3 = n_voxels * voxel_volume_mm3
    brain_volume_cm3 = brain_volume_mm3 / 1000.0

    # Calculate bounding box
    coords = np.where(mask_data)
    bbox = {
        'x_min': int(np.min(coords[0])),
        'x_max': int(np.max(coords[0])),
        'y_min': int(np.min(coords[1])),
        'y_max': int(np.max(coords[1])),
        'z_min': int(np.min(coords[2])),
        'z_max': int(np.max(coords[2]))
    }

    bbox_size = {
        'x': bbox['x_max'] - bbox['x_min'],
        'y': bbox['y_max'] - bbox['y_min'],
        'z': bbox['z_max'] - bbox['z_min']
    }

    # Calculate intensity statistics
    brain_intensities = func_data[mask_data > 0]
    outside_intensities = func_data[mask_data == 0]

    # Remove zero values (background)
    brain_intensities = brain_intensities[brain_intensities > 0]
    outside_intensities = outside_intensities[outside_intensities > 0]

    brain_mean = np.mean(brain_intensities)
    brain_std = np.std(brain_intensities)
    outside_mean = np.mean(outside_intensities) if len(outside_intensities) > 0 else 0
    outside_std = np.std(outside_intensities) if len(outside_intensities) > 0 else 0

    # Contrast between brain and non-brain
    contrast_ratio = brain_mean / outside_mean if outside_mean > 0 else np.inf

    # Assess quality - Functional has lower contrast than anatomical
    quality_flags = []

    if contrast_ratio < 1.5:  # Lower threshold for functional
        quality_flags.append('LOW_CONTRAST')
        logger.warning("Low contrast between brain and non-brain regions")

    if brain_std / brain_mean > 0.6:  # Functional can have more variance
        quality_flags.append('HIGH_VARIANCE')
        logger.warning("High intensity variance within brain mask")

    # Check brain volume is reasonable (typical range: 800-1800 cm³)
    if brain_volume_cm3 < 500:
        quality_flags.append('SMALL_BRAIN_VOLUME')
        logger.warning(f"Unusually small brain volume: {brain_volume_cm3:.1f} cm³")
    elif brain_volume_cm3 > 2500:
        quality_flags.append('LARGE_BRAIN_VOLUME')
        logger.warning(f"Unusually large brain volume: {brain_volume_cm3:.1f} cm³")

    metrics = {
        'subject': subject,
        'modality': 'func',
        'n_voxels': n_voxels,
        'brain_volume_mm3': float(brain_volume_mm3),
        'brain_volume_cm3': float(brain_volume_cm3),
        'voxel_size_mm': list(voxel_size),
        'bbox': bbox,
        'bbox_size': bbox_size,
        'brain_mean_intensity': float(brain_mean),
        'brain_std_intensity': float(brain_std),
        'outside_mean_intensity': float(outside_mean),
        'outside_std_intensity': float(outside_std),
        'contrast_ratio': float(contrast_ratio),
        'quality_flags': quality_flags,
        'quality_pass': len(quality_flags) == 0
    }

    logger.info("Functional Skull Strip QC Summary:")
    logger.info(f"  Brain volume: {brain_volume_cm3:.2f} cm³")
    logger.info(f"  N voxels: {n_voxels}")
    logger.info(f"  Contrast ratio: {contrast_ratio:.2f}")
    logger.info(f"  Quality flags: {quality_flags if quality_flags else 'PASS'}")
    logger.info("")

    # Create overlay plot
    logger.info("Creating mask overlay visualization...")
    center_x = func_data.shape[0] // 2
    center_y = func_data.shape[1] // 2
    center_z = func_data.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Sagittal slices
    for i, offset in enumerate([-5, 0, 5]):
        ax = axes[0, i]
        slice_idx = center_x + offset
        ax.imshow(func_data[slice_idx, :, :].T, cmap='gray', origin='lower')
        ax.contour(mask_data[slice_idx, :, :].T, colors='red', linewidths=1)
        ax.set_title(f'Sagittal (x={slice_idx})', fontsize=10)
        ax.axis('off')

    # Axial slices
    for i, offset in enumerate([-5, 0, 5]):
        ax = axes[1, i]
        slice_idx = center_z + offset
        ax.imshow(func_data[:, :, slice_idx].T, cmap='gray', origin='lower')
        ax.contour(mask_data[:, :, slice_idx].T, colors='red', linewidths=1)
        ax.set_title(f'Axial (z={slice_idx})', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Functional Brain Mask Overlay - {subject}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    overlay_plot = output_dir / 'skull_strip_overlay.png'
    plt.savefig(overlay_plot, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved overlay: {overlay_plot}")

    # Save metrics to JSON
    metrics_json = output_dir / 'skull_strip_metrics.json'
    import json
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  Saved metrics: {metrics_json}")
    logger.info("")

    metrics['mask_overlay'] = str(overlay_plot)
    metrics['metrics_json'] = str(metrics_json)

    return metrics


def generate_func_qc_report(
    subject: str,
    motion_metrics: Dict[str, Any],
    tsnr_metrics: Optional[Dict[str, Any]],
    dvars_metrics: Optional[Dict[str, Any]],
    carpet_metrics: Optional[Dict[str, Any]],
    tedana_report: Optional[Path],
    output_file: Path
) -> Path:
    """
    Generate HTML QC report for functional preprocessing.

    Parameters
    ----------
    subject : str
        Subject identifier
    motion_metrics : dict
        Motion QC metrics from compute_motion_qc()
    tsnr_metrics : dict, optional
        tSNR metrics from compute_tsnr()
    dvars_metrics : dict, optional
        DVARS metrics from compute_dvars()
    carpet_metrics : dict, optional
        Carpet plot metrics from create_carpet_plot()
    tedana_report : Path, optional
        Path to TEDANA HTML report
    output_file : Path
        Output HTML file path

    Returns
    -------
    Path
        Path to generated HTML report
    """
    logger.info("=" * 70)
    logger.info("Generating QC Report")
    logger.info("=" * 70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Output: {output_file}")
    logger.info("")

    # HTML template
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Functional MRI QC Report - {subject}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-size: 1.5em;
            color: #2c3e50;
            margin-top: 5px;
        }}
        .status-good {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .status-bad {{
            color: #e74c3c;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .button {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            display: inline-block;
            margin: 10px 5px;
        }}
        .button:hover {{
            background-color: #2980b9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Functional MRI Quality Control Report</h1>
        <p><strong>Subject:</strong> {subject}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Motion Assessment</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Mean FD</div>
                <div class="metric-value">{motion_metrics['mean_fd']:.3f} mm</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max FD</div>
                <div class="metric-value">{motion_metrics['max_fd']:.3f} mm</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Outlier Volumes</div>
                <div class="metric-value">{motion_metrics['n_outliers_fd']} ({motion_metrics['percent_outliers']:.1f}%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Rotation</div>
                <div class="metric-value">{motion_metrics['mean_rotation']:.3f}°</div>
            </div>
        </div>

        <p><strong>Motion Quality: </strong>
        <span class="{'status-good' if motion_metrics['mean_fd'] < 0.2 else 'status-warning' if motion_metrics['mean_fd'] < 0.5 else 'status-bad'}">
            {'GOOD' if motion_metrics['mean_fd'] < 0.2 else 'ACCEPTABLE' if motion_metrics['mean_fd'] < 0.5 else 'POOR'}
        </span>
        </p>

        <img src="{motion_metrics['motion_plot'].name}" alt="Motion Parameters">
        <img src="{motion_metrics['fd_plot'].name}" alt="Framewise Displacement">
"""

    # Add tSNR section if available
    if tsnr_metrics:
        html += f"""
        <h2>Temporal SNR</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Mean tSNR</div>
                <div class="metric-value">{tsnr_metrics['mean_tsnr']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Median tSNR</div>
                <div class="metric-value">{tsnr_metrics['median_tsnr']:.2f}</div>
            </div>
        </div>

        <p><strong>tSNR Quality: </strong>
        <span class="{'status-good' if tsnr_metrics['mean_tsnr'] > 100 else 'status-warning' if tsnr_metrics['mean_tsnr'] > 50 else 'status-bad'}">
            {'EXCELLENT' if tsnr_metrics['mean_tsnr'] > 100 else 'GOOD' if tsnr_metrics['mean_tsnr'] > 50 else 'POOR'}
        </span>
        </p>

        <img src="{tsnr_metrics['tsnr_histogram'].name}" alt="tSNR Histogram">
"""

    # Add DVARS section if available
    if dvars_metrics:
        html += f"""
        <h2>DVARS (Artifact Detection)</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Mean DVARS</div>
                <div class="metric-value">{dvars_metrics['mean_dvars']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Median DVARS</div>
                <div class="metric-value">{dvars_metrics['median_dvars']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">DVARS Outliers</div>
                <div class="metric-value">{dvars_metrics['n_outliers_dvars']} ({dvars_metrics['percent_outliers_dvars']:.1f}%)</div>
            </div>
        </div>

        <p><strong>DVARS Quality: </strong>
        <span class="{'status-good' if dvars_metrics['percent_outliers_dvars'] < 5 else 'status-warning' if dvars_metrics['percent_outliers_dvars'] < 15 else 'status-bad'}">
            {'GOOD' if dvars_metrics['percent_outliers_dvars'] < 5 else 'ACCEPTABLE' if dvars_metrics['percent_outliers_dvars'] < 15 else 'POOR'}
        </span>
        </p>

        <img src="{dvars_metrics['dvars_plot'].name}" alt="DVARS Time Series">
"""

    # Add carpet plot if available
    if carpet_metrics:
        html += f"""
        <h2>Carpet Plot (Voxel Intensity Visualization)</h2>
        <p>The carpet plot shows voxel intensities over time, organized by signal intensity.
        This visualization helps identify global signal fluctuations, artifacts, and motion-related intensity changes.</p>

        <img src="{carpet_metrics['carpet_plot'].name}" alt="Carpet Plot">
"""

    # Add TEDANA link if available
    if tedana_report and tedana_report.exists():
        html += f"""
        <h2>TEDANA Multi-Echo Denoising</h2>
        <p>Multi-echo denoising was performed using TEDANA.</p>
        <a href="{tedana_report}" class="button">View TEDANA Report</a>
"""

    # Close HTML
    html += """
        <h2>Summary</h2>
        <p>This QC report summarizes the quality metrics from functional MRI preprocessing. Review the plots and metrics above to assess data quality.</p>

        <h3>Quality Guidelines:</h3>
        <ul>
            <li><strong>Mean FD:</strong> <span class="status-good">&lt;0.2mm = Good</span>, <span class="status-warning">0.2-0.5mm = Acceptable</span>, <span class="status-bad">&gt;0.5mm = Poor</span></li>
            <li><strong>Outlier Volumes:</strong> <span class="status-good">&lt;5% = Good</span>, <span class="status-warning">5-20% = Acceptable</span>, <span class="status-bad">&gt;20% = Poor</span></li>
            <li><strong>Mean tSNR:</strong> <span class="status-good">&gt;100 = Excellent</span>, <span class="status-warning">50-100 = Good</span>, <span class="status-bad">&lt;50 = Poor</span></li>
            <li><strong>DVARS Outliers:</strong> <span class="status-good">&lt;5% = Good</span>, <span class="status-warning">5-15% = Acceptable</span>, <span class="status-bad">&gt;15% = Poor</span></li>
        </ul>

        <h3>Additional Notes:</h3>
        <ul>
            <li><strong>DVARS</strong> measures the spatial standard deviation of temporal derivative. High DVARS values indicate sudden intensity changes due to artifacts or motion.</li>
            <li><strong>Carpet plots</strong> provide a comprehensive view of signal across all voxels over time. Look for global signal fluctuations, respiratory artifacts, or motion-related patterns.</li>
        </ul>
    </div>
</body>
</html>
"""

    # Write HTML file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(html)

    logger.info(f"QC report generated: {output_file}")
    logger.info("")

    return output_file
