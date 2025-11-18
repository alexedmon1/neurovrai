#!/usr/bin/env python3
"""
Quality Control (QC) utilities for ASL (Arterial Spin Labeling) preprocessing.

This module provides functions for:
1. Motion assessment from MCFLIRT parameters
2. CBF distribution analysis and physiological validation
3. Tissue-specific CBF metrics (GM/WM/CSF)
4. Temporal SNR (tSNR) calculation for perfusion signal
5. Skull stripping quality assessment
6. Automated HTML QC report generation

Usage:
    from neurovrai.preprocess.qc.asl_qc import compute_asl_motion_qc, compute_cbf_qc, compute_asl_skull_strip_qc, generate_asl_qc_report
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


def compute_asl_motion_qc(
    motion_file: Path,
    output_dir: Path,
    fd_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute motion QC metrics for ASL data.

    Parameters
    ----------
    motion_file : Path
        FSL motion parameters file (.par) from MCFLIRT
        Format: 6 columns (3 rotations in radians, 3 translations in mm)
    output_dir : Path
        Output directory for QC files
    fd_threshold : float
        Framewise displacement threshold in mm (default: 0.5mm)

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
    logger.info("ASL Motion QC Assessment")
    logger.info("=" * 70)
    logger.info(f"Motion file: {motion_file}")
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
    volume_indices = np.arange(n_volumes)
    axes[0].plot(volume_indices, rotations_deg[:, 0], 'r-', label='Roll (X)', linewidth=1)
    axes[0].plot(volume_indices, rotations_deg[:, 1], 'g-', label='Pitch (Y)', linewidth=1)
    axes[0].plot(volume_indices, rotations_deg[:, 2], 'b-', label='Yaw (Z)', linewidth=1)
    axes[0].set_ylabel('Rotation (degrees)', fontsize=12)
    axes[0].set_title('ASL Head Motion Parameters', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # Translations
    axes[1].plot(volume_indices, translations_mm[:, 0], 'r-', label='X', linewidth=1)
    axes[1].plot(volume_indices, translations_mm[:, 1], 'g-', label='Y', linewidth=1)
    axes[1].plot(volume_indices, translations_mm[:, 2], 'b-', label='Z', linewidth=1)
    axes[1].set_xlabel('Volume', fontsize=12)
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
    ax.plot(volume_indices, fd, 'k-', linewidth=1.5, label='FD')
    ax.axhline(y=fd_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({fd_threshold}mm)')
    ax.fill_between(volume_indices, 0, fd, where=outliers_fd, color='red', alpha=0.3, label='Outliers')
    ax.set_xlabel('Volume', fontsize=12)
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


def compute_cbf_qc(
    cbf_file: Path,
    mask_file: Path,
    output_dir: Path,
    gm_mask: Optional[Path] = None,
    wm_mask: Optional[Path] = None,
    csf_mask: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute CBF quality control metrics.

    Parameters
    ----------
    cbf_file : Path
        CBF map (ml/100g/min)
    mask_file : Path
        Brain mask file
    output_dir : Path
        Output directory for QC files
    gm_mask : Path, optional
        Gray matter mask
    wm_mask : Path, optional
        White matter mask
    csf_mask : Path, optional
        CSF mask

    Returns
    -------
    dict
        CBF QC metrics:
        - mean_cbf: Mean CBF in brain
        - median_cbf: Median CBF in brain
        - std_cbf: Standard deviation of CBF
        - gm_cbf: Gray matter CBF (if mask provided)
        - wm_cbf: White matter CBF (if mask provided)
        - cbf_histogram: Path to histogram plot
        - tissue_cbf_plot: Path to tissue-specific plot (if masks provided)
    """
    logger.info("=" * 70)
    logger.info("CBF Quality Control")
    logger.info("=" * 70)
    logger.info(f"CBF file: {cbf_file}")
    logger.info(f"Mask file: {mask_file}")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CBF map
    cbf_img = nib.load(cbf_file)
    cbf_data = cbf_img.get_fdata()

    # Load brain mask
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata() > 0

    # Extract CBF within brain mask
    cbf_brain = cbf_data[mask_data]

    # Calculate whole-brain metrics
    metrics = {
        'mean_cbf': np.mean(cbf_brain),
        'median_cbf': np.median(cbf_brain),
        'std_cbf': np.std(cbf_brain),
        'min_cbf': np.min(cbf_brain),
        'max_cbf': np.max(cbf_brain),
        'n_voxels': len(cbf_brain)
    }

    logger.info("Whole-Brain CBF Summary:")
    logger.info(f"  Mean: {metrics['mean_cbf']:.2f} ml/100g/min")
    logger.info(f"  Median: {metrics['median_cbf']:.2f} ml/100g/min")
    logger.info(f"  Std: {metrics['std_cbf']:.2f} ml/100g/min")
    logger.info(f"  Range: [{metrics['min_cbf']:.2f}, {metrics['max_cbf']:.2f}]")
    logger.info(f"  Brain voxels: {metrics['n_voxels']}")
    logger.info("")

    # Tissue-specific CBF (if masks provided)
    tissue_metrics = {}
    if gm_mask and gm_mask.exists() and wm_mask and wm_mask.exists():
        logger.info("Computing tissue-specific CBF...")

        # Load tissue masks
        gm_data = nib.load(gm_mask).get_fdata() > 0.5
        wm_data = nib.load(wm_mask).get_fdata() > 0.5

        # Gray matter CBF
        gm_cbf = cbf_data[gm_data]
        tissue_metrics['gm'] = {
            'mean': np.mean(gm_cbf),
            'median': np.median(gm_cbf),
            'std': np.std(gm_cbf),
            'n_voxels': len(gm_cbf)
        }

        # White matter CBF
        wm_cbf = cbf_data[wm_data]
        tissue_metrics['wm'] = {
            'mean': np.mean(wm_cbf),
            'median': np.median(wm_cbf),
            'std': np.std(wm_cbf),
            'n_voxels': len(wm_cbf)
        }

        # CSF CBF (if available)
        if csf_mask and csf_mask.exists():
            csf_data = nib.load(csf_mask).get_fdata() > 0.5
            csf_cbf = cbf_data[csf_data]
            tissue_metrics['csf'] = {
                'mean': np.mean(csf_cbf),
                'median': np.median(csf_cbf),
                'std': np.std(csf_cbf),
                'n_voxels': len(csf_cbf)
            }

        logger.info("Tissue-Specific CBF:")
        logger.info(f"  Gray Matter: {tissue_metrics['gm']['mean']:.2f} ± {tissue_metrics['gm']['std']:.2f} ml/100g/min")
        logger.info(f"  White Matter: {tissue_metrics['wm']['mean']:.2f} ± {tissue_metrics['wm']['std']:.2f} ml/100g/min")
        if 'csf' in tissue_metrics:
            logger.info(f"  CSF: {tissue_metrics['csf']['mean']:.2f} ± {tissue_metrics['csf']['std']:.2f} ml/100g/min")
        logger.info("")

        # Check against expected physiological ranges
        gm_expected = (40, 60)  # ml/100g/min
        wm_expected = (20, 30)

        gm_status = "NORMAL" if gm_expected[0] <= tissue_metrics['gm']['mean'] <= gm_expected[1] else "ABNORMAL"
        wm_status = "NORMAL" if wm_expected[0] <= tissue_metrics['wm']['mean'] <= wm_expected[1] else "ABNORMAL"

        logger.info("Physiological Validation:")
        logger.info(f"  GM CBF: {gm_status} (expected: {gm_expected[0]}-{gm_expected[1]} ml/100g/min)")
        logger.info(f"  WM CBF: {wm_status} (expected: {wm_expected[0]}-{wm_expected[1]} ml/100g/min)")
        logger.info("")

        metrics['tissue_metrics'] = tissue_metrics
        metrics['gm_status'] = gm_status
        metrics['wm_status'] = wm_status

    # Create visualizations
    logger.info("Creating CBF visualization plots...")

    # Plot 1: CBF histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Clip extreme values for better visualization
    cbf_clipped = np.clip(cbf_brain, 0, 200)

    ax.hist(cbf_clipped, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(metrics['mean_cbf'], color='r', linestyle='--', linewidth=2,
               label=f"Mean={metrics['mean_cbf']:.2f}")
    ax.axvline(metrics['median_cbf'], color='g', linestyle='--', linewidth=2,
               label=f"Median={metrics['median_cbf']:.2f}")

    # Add expected ranges
    ax.axvspan(40, 60, alpha=0.2, color='green', label='Expected GM range')
    ax.axvspan(20, 30, alpha=0.2, color='orange', label='Expected WM range')

    ax.set_xlabel('CBF (ml/100g/min)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Cerebral Blood Flow Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_plot = output_dir / 'cbf_histogram.png'
    plt.savefig(hist_plot, dpi=150, bbox_inches='tight')
    plt.close()
    metrics['cbf_histogram'] = hist_plot
    logger.info(f"  Saved: {hist_plot}")

    # Plot 2: Tissue-specific CBF (if available)
    if tissue_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        tissues = list(tissue_metrics.keys())
        means = [tissue_metrics[t]['mean'] for t in tissues]
        stds = [tissue_metrics[t]['std'] for t in tissues]

        colors = {'gm': 'steelblue', 'wm': 'lightcoral', 'csf': 'lightgreen'}
        tissue_colors = [colors[t] for t in tissues]

        bars = ax.bar(tissues, means, yerr=stds, capsize=10,
                      color=tissue_colors, edgecolor='black', linewidth=1.5, alpha=0.7)

        # Add expected ranges as horizontal lines
        ax.axhline(y=50, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Expected GM (40-60)')
        ax.axhline(y=25, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Expected WM (20-30)')

        ax.set_ylabel('CBF (ml/100g/min)', fontsize=12)
        ax.set_title('Tissue-Specific Cerebral Blood Flow', fontsize=14, fontweight='bold')
        ax.set_xticklabels([t.upper() for t in tissues], fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                   f'{mean:.1f}±{std:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        tissue_plot = output_dir / 'tissue_cbf.png'
        plt.savefig(tissue_plot, dpi=150, bbox_inches='tight')
        plt.close()
        metrics['tissue_cbf_plot'] = tissue_plot
        logger.info(f"  Saved: {tissue_plot}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'mean_cbf': metrics['mean_cbf'],
        'median_cbf': metrics['median_cbf'],
        'std_cbf': metrics['std_cbf'],
        'min_cbf': metrics['min_cbf'],
        'max_cbf': metrics['max_cbf'],
        'n_voxels': metrics['n_voxels']
    }])

    if tissue_metrics:
        for tissue, vals in tissue_metrics.items():
            metrics_df[f'{tissue}_mean'] = vals['mean']
            metrics_df[f'{tissue}_std'] = vals['std']

    metrics_csv = output_dir / 'cbf_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    metrics['cbf_metrics_csv'] = metrics_csv
    logger.info(f"  Saved: {metrics_csv}")
    logger.info("")

    return metrics


def compute_perfusion_tsnr(
    perfusion_file: Path,
    mask_file: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Compute temporal SNR for perfusion-weighted signal.

    Note: For ASL, this calculates tSNR on the 4D perfusion-weighted images
    (label-control difference images), not the final CBF map.

    Parameters
    ----------
    perfusion_file : Path
        4D perfusion-weighted image (ΔM time series)
    mask_file : Path
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
    """
    logger.info("=" * 70)
    logger.info("Perfusion tSNR Calculation")
    logger.info("=" * 70)
    logger.info(f"Perfusion file: {perfusion_file}")
    logger.info(f"Mask file: {mask_file}")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute tSNR using FSL
    tsnr_file = output_dir / 'perfusion_tsnr.nii.gz'
    mean_file = output_dir / 'perfusion_mean.nii.gz'
    std_file = output_dir / 'perfusion_std.nii.gz'

    # Temporal mean
    cmd_mean = ['fslmaths', str(perfusion_file), '-Tmean', str(mean_file)]
    subprocess.run(cmd_mean, check=True, capture_output=True)

    # Temporal std
    cmd_std = ['fslmaths', str(perfusion_file), '-Tstd', str(std_file)]
    subprocess.run(cmd_std, check=True, capture_output=True)

    # tSNR = mean / std
    cmd_tsnr = ['fslmaths', str(mean_file), '-div', str(std_file), str(tsnr_file)]
    subprocess.run(cmd_tsnr, check=True, capture_output=True)

    logger.info(f"  Computed tSNR map: {tsnr_file}")

    # Load tSNR map and mask
    tsnr_img = nib.load(tsnr_file)
    tsnr_data = tsnr_img.get_fdata()

    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata() > 0

    tsnr_masked = tsnr_data[mask_data]

    # Calculate metrics
    metrics = {
        'mean_tsnr': np.mean(tsnr_masked),
        'median_tsnr': np.median(tsnr_masked),
        'std_tsnr': np.std(tsnr_masked),
        'min_tsnr': np.min(tsnr_masked),
        'max_tsnr': np.max(tsnr_masked),
        'tsnr_map': tsnr_file
    }

    logger.info("Perfusion tSNR Summary:")
    logger.info(f"  Mean tSNR: {metrics['mean_tsnr']:.2f}")
    logger.info(f"  Median tSNR: {metrics['median_tsnr']:.2f}")
    logger.info(f"  Range: {metrics['min_tsnr']:.2f} - {metrics['max_tsnr']:.2f}")
    logger.info("")

    return metrics


def compute_asl_skull_strip_qc(
    asl_mean_file: Path,
    mask_file: Path,
    output_dir: Path,
    subject: str = "unknown"
) -> Dict[str, Any]:
    """
    Compute skull stripping quality metrics for ASL data.

    Parameters
    ----------
    asl_mean_file : Path
        Mean ASL image (temporal mean of control or M0 images)
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
    logger.info("ASL Skull Strip QC Assessment")
    logger.info("=" * 70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Mean ASL file: {asl_mean_file}")
    logger.info(f"Mask file: {mask_file}")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    asl_img = nib.load(asl_mean_file)
    asl_data = asl_img.get_fdata()
    voxel_size = asl_img.header.get_zooms()[:3]

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
    brain_intensities = asl_data[mask_data > 0]
    outside_intensities = asl_data[mask_data == 0]

    # Remove zero values (background)
    brain_intensities = brain_intensities[brain_intensities > 0]
    outside_intensities = outside_intensities[outside_intensities > 0]

    brain_mean = np.mean(brain_intensities)
    brain_std = np.std(brain_intensities)
    outside_mean = np.mean(outside_intensities) if len(outside_intensities) > 0 else 0
    outside_std = np.std(outside_intensities) if len(outside_intensities) > 0 else 0

    # Contrast between brain and non-brain
    contrast_ratio = brain_mean / outside_mean if outside_mean > 0 else np.inf

    # Assess quality - ASL has lower contrast than anatomical
    quality_flags = []

    if contrast_ratio < 1.5:  # Lower threshold for ASL
        quality_flags.append('LOW_CONTRAST')
        logger.warning("Low contrast between brain and non-brain regions")

    if brain_std / brain_mean > 0.6:  # ASL can have more variance
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
        'modality': 'asl',
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

    logger.info("ASL Skull Strip QC Summary:")
    logger.info(f"  Brain volume: {brain_volume_cm3:.2f} cm³")
    logger.info(f"  N voxels: {n_voxels}")
    logger.info(f"  Contrast ratio: {contrast_ratio:.2f}")
    logger.info(f"  Quality flags: {quality_flags if quality_flags else 'PASS'}")
    logger.info("")

    # Create overlay plot
    logger.info("Creating mask overlay visualization...")
    center_x = asl_data.shape[0] // 2
    center_y = asl_data.shape[1] // 2
    center_z = asl_data.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Sagittal slices
    for i, offset in enumerate([-5, 0, 5]):
        ax = axes[0, i]
        slice_idx = center_x + offset
        ax.imshow(asl_data[slice_idx, :, :].T, cmap='gray', origin='lower')
        ax.contour(mask_data[slice_idx, :, :].T, colors='red', linewidths=1)
        ax.set_title(f'Sagittal (x={slice_idx})', fontsize=10)
        ax.axis('off')

    # Axial slices
    for i, offset in enumerate([-5, 0, 5]):
        ax = axes[1, i]
        slice_idx = center_z + offset
        ax.imshow(asl_data[:, :, slice_idx].T, cmap='gray', origin='lower')
        ax.contour(mask_data[:, :, slice_idx].T, colors='red', linewidths=1)
        ax.set_title(f'Axial (z={slice_idx})', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'ASL Brain Mask Overlay - {subject}', fontsize=14, fontweight='bold')
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


def generate_asl_qc_report(
    subject: str,
    motion_metrics: Dict[str, Any],
    cbf_metrics: Dict[str, Any],
    tsnr_metrics: Optional[Dict[str, Any]],
    output_file: Path
) -> Path:
    """
    Generate HTML QC report for ASL preprocessing.

    Parameters
    ----------
    subject : str
        Subject identifier
    motion_metrics : dict
        Motion QC metrics from compute_asl_motion_qc()
    cbf_metrics : dict
        CBF QC metrics from compute_cbf_qc()
    tsnr_metrics : dict, optional
        tSNR metrics from compute_perfusion_tsnr()
    output_file : Path
        Output HTML file path

    Returns
    -------
    Path
        Path to generated HTML report
    """
    logger.info("=" * 70)
    logger.info("Generating ASL QC Report")
    logger.info("=" * 70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Output: {output_file}")
    logger.info("")

    # HTML template
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ASL QC Report - {subject}</title>
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
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #e74c3c;
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
            border-left: 4px solid #e74c3c;
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
            background-color: #e74c3c;
            color: white;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ASL (Arterial Spin Labeling) Quality Control Report</h1>
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

        <h2>Cerebral Blood Flow (CBF) Assessment</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Mean CBF</div>
                <div class="metric-value">{cbf_metrics['mean_cbf']:.2f} ml/100g/min</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Median CBF</div>
                <div class="metric-value">{cbf_metrics['median_cbf']:.2f} ml/100g/min</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Std CBF</div>
                <div class="metric-value">{cbf_metrics['std_cbf']:.2f} ml/100g/min</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Brain Voxels</div>
                <div class="metric-value">{cbf_metrics['n_voxels']}</div>
            </div>
        </div>

        <img src="{cbf_metrics['cbf_histogram'].name}" alt="CBF Histogram">
"""

    # Add tissue-specific CBF section if available
    if 'tissue_metrics' in cbf_metrics:
        tissue = cbf_metrics['tissue_metrics']
        gm_status = cbf_metrics.get('gm_status', 'UNKNOWN')
        wm_status = cbf_metrics.get('wm_status', 'UNKNOWN')

        gm_class = 'status-good' if gm_status == 'NORMAL' else 'status-warning'
        wm_class = 'status-good' if wm_status == 'NORMAL' else 'status-warning'

        html += f"""
        <h2>Tissue-Specific CBF</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Gray Matter CBF</div>
                <div class="metric-value">{tissue['gm']['mean']:.2f} ± {tissue['gm']['std']:.2f}</div>
                <p class="{gm_class}">{gm_status}</p>
            </div>
            <div class="metric-card">
                <div class="metric-label">White Matter CBF</div>
                <div class="metric-value">{tissue['wm']['mean']:.2f} ± {tissue['wm']['std']:.2f}</div>
                <p class="{wm_class}">{wm_status}</p>
            </div>
"""

        if 'csf' in tissue:
            html += f"""
            <div class="metric-card">
                <div class="metric-label">CSF CBF</div>
                <div class="metric-value">{tissue['csf']['mean']:.2f} ± {tissue['csf']['std']:.2f}</div>
            </div>
"""

        html += """
        </div>

        <div class="info-box">
            <strong>Expected Physiological Ranges:</strong>
            <ul>
                <li>Gray Matter: 40-60 ml/100g/min</li>
                <li>White Matter: 20-30 ml/100g/min</li>
                <li>CSF: ~0 ml/100g/min (no perfusion)</li>
            </ul>
        </div>

        <img src="{tissue_cbf_plot}" alt="Tissue-Specific CBF">
""".format(tissue_cbf_plot=cbf_metrics['tissue_cbf_plot'].name)

    # Add tSNR section if available
    if tsnr_metrics:
        html += f"""
        <h2>Perfusion Temporal SNR</h2>
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

        <div class="info-box">
            <strong>Note:</strong> tSNR is calculated on the perfusion-weighted signal (ΔM),
            not the raw ASL images. Lower tSNR values are expected compared to BOLD fMRI due
            to the small perfusion signal (~1-2% of total signal).
        </div>
"""

    # Close HTML
    html += """
        <h2>Summary</h2>
        <p>This QC report summarizes quality metrics from ASL preprocessing.
        Review the plots and metrics above to assess CBF data quality.</p>

        <h3>Quality Assessment Guidelines:</h3>
        <ul>
            <li><strong>Motion:</strong> ASL is sensitive to motion. Mean FD &lt;0.2mm is ideal, &lt;0.5mm is acceptable.</li>
            <li><strong>CBF Values:</strong> Check tissue-specific CBF against expected physiological ranges.</li>
            <li><strong>High/Low CBF:</strong> Values outside expected ranges may indicate:
                <ul>
                    <li>Incorrect acquisition parameters (labeling duration, PLD)</li>
                    <li>Poor labeling efficiency</li>
                    <li>Physiological variation (age, disease)</li>
                </ul>
            </li>
        </ul>

        <h3>Additional Notes:</h3>
        <ul>
            <li><strong>ASL Signal:</strong> Perfusion-weighted signal is only 1-2% of total signal,
            making ASL sensitive to noise and motion.</li>
            <li><strong>Acquisition Parameters:</strong> Verify scanner-specific parameters
            (labeling duration, PLD, labeling efficiency) from DICOM headers for accurate CBF quantification.</li>
            <li><strong>Partial Volume Effects:</strong> Low spatial resolution can lead to
            partial volume averaging, affecting CBF values at tissue boundaries.</li>
        </ul>
    </div>
</body>
</html>
"""

    # Write HTML file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(html)

    logger.info(f"ASL QC report generated: {output_file}")
    logger.info("")

    return output_file
