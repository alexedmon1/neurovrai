#!/usr/bin/env python3
"""
Motion Quality Control Module for DWI.

Extracts and analyzes eddy motion parameters:
- Translation (x, y, z)
- Rotation (x, y, z)
- Framewise displacement (FD)
- Outlier detection
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


class MotionQualityControl:
    """
    Quality control for DWI motion correction (eddy).

    Analyzes eddy motion parameters and detects outliers.
    """

    def __init__(self, subject: str, work_dir: Path, qc_dir: Path):
        """
        Initialize Motion QC.

        Parameters
        ----------
        subject : str
            Subject identifier
        work_dir : Path
            Working directory containing eddy outputs
        qc_dir : Path
            QC output directory (e.g., {study_root}/qc/dwi/{subject}/motion/)
        """
        self.subject = subject
        self.work_dir = Path(work_dir)
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Motion QC for {subject}")
        logger.info(f"  Work dir: {self.work_dir}")
        logger.info(f"  QC dir: {self.qc_dir}")

    def load_motion_parameters(
        self,
        eddy_params_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load eddy motion parameters.

        Parameters
        ----------
        eddy_params_file : Path, optional
            Path to eddy parameters file (.eddy_parameters).
            If None, will search in work_dir.

        Returns
        -------
        pd.DataFrame
            Motion parameters with columns:
            [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
        """
        if eddy_params_file is None:
            # Look for eddy parameters file
            param_candidates = list(self.work_dir.glob('*eddy_corrected.eddy_parameters'))
            if param_candidates:
                eddy_params_file = param_candidates[0]
            else:
                logger.warning("No eddy parameters file found")
                return pd.DataFrame()

        if not eddy_params_file.exists():
            logger.warning(f"Eddy parameters file not found: {eddy_params_file}")
            return pd.DataFrame()

        logger.info(f"Loading motion parameters: {eddy_params_file}")

        # Load parameters (6 columns: 3 translations + 3 rotations)
        params = np.loadtxt(eddy_params_file)

        # Create DataFrame
        df = pd.DataFrame(params, columns=[
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z'
        ])

        logger.info(f"Loaded motion parameters for {len(df)} volumes")
        return df

    def calculate_framewise_displacement(
        self,
        motion_df: pd.DataFrame,
        radius: float = 50.0
    ) -> np.ndarray:
        """
        Calculate framewise displacement (FD).

        FD = sum(abs(delta_trans)) + sum(abs(delta_rot * radius))

        Parameters
        ----------
        motion_df : pd.DataFrame
            Motion parameters DataFrame
        radius : float, optional
            Head radius in mm for converting rotation to displacement.
            Default is 50 mm (typical brain radius).

        Returns
        -------
        np.ndarray
            Framewise displacement for each volume (first volume = 0)
        """
        if motion_df.empty:
            return np.array([])

        # Calculate differences
        trans_cols = ['trans_x', 'trans_y', 'trans_z']
        rot_cols = ['rot_x', 'rot_y', 'rot_z']

        delta_trans = motion_df[trans_cols].diff().fillna(0)
        delta_rot = motion_df[rot_cols].diff().fillna(0)

        # Convert rotation (radians) to displacement (mm)
        delta_rot_mm = delta_rot * radius

        # Calculate FD
        fd = delta_trans.abs().sum(axis=1) + delta_rot_mm.abs().sum(axis=1)

        return fd.values

    def detect_outliers(
        self,
        fd: np.ndarray,
        threshold: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Detect motion outliers based on FD threshold.

        Parameters
        ----------
        fd : np.ndarray
            Framewise displacement values
        threshold : float, optional
            FD threshold in mm (default: 1.0 mm)

        Returns
        -------
        outlier_indices : np.ndarray
            Indices of volumes exceeding threshold
        n_outliers : int
            Number of outlier volumes
        """
        if len(fd) == 0:
            return np.array([]), 0

        outlier_indices = np.where(fd > threshold)[0]
        n_outliers = len(outlier_indices)

        logger.info(f"Detected {n_outliers} outlier volumes (FD > {threshold} mm)")

        return outlier_indices, n_outliers

    def calculate_motion_statistics(
        self,
        motion_df: pd.DataFrame,
        fd: np.ndarray
    ) -> Dict:
        """
        Calculate summary statistics for motion parameters.

        Parameters
        ----------
        motion_df : pd.DataFrame
            Motion parameters
        fd : np.ndarray
            Framewise displacement

        Returns
        -------
        dict
            Motion statistics
        """
        if motion_df.empty or len(fd) == 0:
            return {}

        stats = {
            'subject': self.subject,
            'n_volumes': len(motion_df),
            'fd_mean': float(np.mean(fd)),
            'fd_max': float(np.max(fd)),
            'fd_std': float(np.std(fd)),
            'fd_median': float(np.median(fd)),
            'fd_percentile_95': float(np.percentile(fd, 95)),
            'translation_rms': float(np.sqrt(np.mean(
                motion_df[['trans_x', 'trans_y', 'trans_z']].values**2
            ))),
            'rotation_rms': float(np.sqrt(np.mean(
                motion_df[['rot_x', 'rot_y', 'rot_z']].values**2
            ))),
            'max_translation': float(np.max(np.abs(
                motion_df[['trans_x', 'trans_y', 'trans_z']].values
            ))),
            'max_rotation': float(np.max(np.abs(
                motion_df[['rot_x', 'rot_y', 'rot_z']].values
            )))
        }

        logger.info(f"Motion statistics:")
        logger.info(f"  FD mean: {stats['fd_mean']:.3f} mm")
        logger.info(f"  FD max: {stats['fd_max']:.3f} mm")
        logger.info(f"  Translation RMS: {stats['translation_rms']:.3f} mm")
        logger.info(f"  Rotation RMS: {stats['rotation_rms']:.4f} rad")

        return stats

    def plot_motion_parameters(
        self,
        motion_df: pd.DataFrame,
        fd: np.ndarray,
        outlier_indices: np.ndarray,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Plot motion parameters and framewise displacement.

        Parameters
        ----------
        motion_df : pd.DataFrame
            Motion parameters
        fd : np.ndarray
            Framewise displacement
        outlier_indices : np.ndarray
            Indices of outlier volumes
        output_file : Path, optional
            Output file path. If None, will save to qc_dir

        Returns
        -------
        Path
            Path to saved plot
        """
        if output_file is None:
            output_file = self.qc_dir / 'motion_parameters.png'

        if motion_df.empty:
            logger.warning("No motion data to plot")
            return None

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        volumes = np.arange(len(motion_df))

        # Plot translations
        ax = axes[0]
        ax.plot(volumes, motion_df['trans_x'], 'r-', label='X', linewidth=1)
        ax.plot(volumes, motion_df['trans_y'], 'g-', label='Y', linewidth=1)
        ax.plot(volumes, motion_df['trans_z'], 'b-', label='Z', linewidth=1)
        ax.set_xlabel('Volume', fontsize=10)
        ax.set_ylabel('Translation (mm)', fontsize=10)
        ax.set_title(f'Motion Parameters - {self.subject}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot rotations
        ax = axes[1]
        ax.plot(volumes, motion_df['rot_x'], 'r-', label='X', linewidth=1)
        ax.plot(volumes, motion_df['rot_y'], 'g-', label='Y', linewidth=1)
        ax.plot(volumes, motion_df['rot_z'], 'b-', label='Z', linewidth=1)
        ax.set_xlabel('Volume', fontsize=10)
        ax.set_ylabel('Rotation (rad)', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot framewise displacement
        ax = axes[2]
        ax.plot(volumes, fd, 'k-', linewidth=1.5, label='FD')
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Threshold (1 mm)')

        # Mark outliers
        if len(outlier_indices) > 0:
            ax.scatter(outlier_indices, fd[outlier_indices],
                      color='red', s=50, zorder=5, label=f'Outliers (n={len(outlier_indices)})')

        ax.set_xlabel('Volume', fontsize=10)
        ax.set_ylabel('Framewise Displacement (mm)', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add stats text
        stats_text = (
            f"Mean FD: {np.mean(fd):.3f} mm\n"
            f"Max FD: {np.max(fd):.3f} mm\n"
            f"Outliers: {len(outlier_indices)}/{len(fd)}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved motion plot: {output_file}")
        return output_file

    def save_metrics_json(
        self,
        motion_stats: Dict,
        outlier_info: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Save motion metrics to JSON file.

        Parameters
        ----------
        motion_stats : dict
            Motion statistics
        outlier_info : dict
            Outlier information
        output_file : Path, optional
            Output JSON file. If None, will save to qc_dir

        Returns
        -------
        Path
            Path to saved JSON file
        """
        if output_file is None:
            output_file = self.qc_dir.parent / 'metrics' / 'motion.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Combine metrics
        combined_metrics = {
            **motion_stats,
            'outliers': outlier_info
        }

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(combined_metrics, f, indent=2)

        logger.info(f"Saved motion metrics: {output_file}")
        return output_file

    def run_qc(
        self,
        eddy_params_file: Optional[Path] = None,
        fd_threshold: float = 1.0
    ) -> Dict:
        """
        Run complete motion QC analysis.

        Parameters
        ----------
        eddy_params_file : Path, optional
            Path to eddy parameters file
        fd_threshold : float, optional
            FD threshold for outlier detection (default: 1.0 mm)

        Returns
        -------
        dict
            Combined QC metrics and output paths
        """
        logger.info(f"Running Motion QC for {self.subject}")

        # Load motion parameters
        motion_df = self.load_motion_parameters(eddy_params_file)

        if motion_df.empty:
            logger.warning("No motion parameters found, skipping Motion QC")
            return {
                'subject': self.subject,
                'motion_stats': {},
                'outliers': {},
                'outputs': {}
            }

        # Calculate framewise displacement
        fd = self.calculate_framewise_displacement(motion_df)

        # Detect outliers
        outlier_indices, n_outliers = self.detect_outliers(fd, fd_threshold)

        # Calculate statistics
        motion_stats = self.calculate_motion_statistics(motion_df, fd)

        # Create outlier info
        outlier_info = {
            'n_outliers': int(n_outliers),
            'outlier_indices': outlier_indices.tolist(),
            'threshold_mm': fd_threshold,
            'percent_outliers': float(n_outliers / len(fd) * 100) if len(fd) > 0 else 0.0
        }

        # Plot motion parameters
        motion_plot = self.plot_motion_parameters(
            motion_df, fd, outlier_indices
        )

        # Save metrics to JSON
        metrics_file = self.save_metrics_json(motion_stats, outlier_info)

        results = {
            'subject': self.subject,
            'motion_stats': motion_stats,
            'outliers': outlier_info,
            'outputs': {
                'motion_plot': str(motion_plot) if motion_plot else None,
                'metrics_json': str(metrics_file)
            }
        }

        logger.info("Motion QC completed")
        return results
