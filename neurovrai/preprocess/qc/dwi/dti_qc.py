#!/usr/bin/env python3
"""
DTI Quality Control Module.

Analyzes DTI metrics (FA, MD, L1, L2, L3) for quality control:
- Distribution statistics
- Histogram visualization
- Outlier detection
- Regional analysis (white matter, gray matter)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


class DTIQualityControl:
    """
    Quality control for DTI metrics.

    Analyzes FA, MD, and eigenvalue maps for quality assessment.
    """

    def __init__(self, subject: str, dti_dir: Path, qc_dir: Path):
        """
        Initialize DTI QC.

        Parameters
        ----------
        subject : str
            Subject identifier
        dti_dir : Path
            Directory containing DTI outputs (FA, MD, eigenvalues)
        qc_dir : Path
            QC output directory (e.g., {study_root}/qc/dwi/{subject}/dti/)
        """
        self.subject = subject
        self.dti_dir = Path(dti_dir)
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized DTI QC for {subject}")
        logger.info(f"  DTI dir: {self.dti_dir}")
        logger.info(f"  QC dir: {self.qc_dir}")

    def load_dti_map(self, metric: str) -> Tuple[Optional[np.ndarray], Optional[nib.Nifti1Image]]:
        """
        Load a DTI metric map.

        Parameters
        ----------
        metric : str
            DTI metric name: 'FA', 'MD', 'L1', 'L2', 'L3', 'AD', 'RD'
            - AD (Axial Diffusivity) = L1
            - RD (Radial Diffusivity) = (L2 + L3) / 2

        Returns
        -------
        data : np.ndarray or None
            DTI metric data
        img : nib.Nifti1Image or None
            NIfTI image object
        """
        # Special handling for derived metrics
        if metric == 'AD':
            # AD (Axial Diffusivity) is the same as L1
            logger.info(f"Loading AD (using L1)...")
            return self.load_dti_map('L1')

        elif metric == 'RD':
            # RD (Radial Diffusivity) = (L2 + L3) / 2
            logger.info(f"Loading RD (calculating from L2 and L3)...")
            l2_data, l2_img = self.load_dti_map('L2')
            l3_data, l3_img = self.load_dti_map('L3')

            if l2_data is None or l3_data is None:
                logger.warning("Cannot calculate RD: L2 or L3 missing")
                return None, None

            # Calculate RD as average of L2 and L3
            rd_data = (l2_data + l3_data) / 2.0
            logger.info(f"Calculated RD from L2 and L3")

            # Return data with L2 image header (same geometry)
            return rd_data, l2_img

        # Common DTIFit output patterns
        patterns = [
            f'*dtifit__{metric}.nii.gz',
            f'*dtifit_{metric}.nii.gz',
            f'*_{metric}.nii.gz',
            f'{metric}.nii.gz'
        ]

        for pattern in patterns:
            candidates = list(self.dti_dir.glob(pattern))
            if candidates:
                dti_file = candidates[0]
                logger.info(f"Loading {metric} map: {dti_file.name}")
                img = nib.load(dti_file)
                data = img.get_fdata()
                return data, img

        logger.warning(f"No {metric} map found in {self.dti_dir}")
        return None, None

    def calculate_dti_statistics(
        self,
        data: np.ndarray,
        metric_name: str,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate statistics for a DTI metric.

        Parameters
        ----------
        data : np.ndarray
            DTI metric data
        metric_name : str
            Name of the metric (e.g., 'FA', 'MD')
        mask : np.ndarray, optional
            Brain mask (if None, uses data > 0)

        Returns
        -------
        dict
            Statistics including mean, median, std, percentiles
        """
        if mask is None:
            mask = data > 0

        masked_data = data[mask]

        if len(masked_data) == 0:
            logger.warning(f"No valid data for {metric_name}")
            return {}

        stats = {
            'metric': metric_name,
            'n_voxels': int(np.sum(mask)),
            'mean': float(np.mean(masked_data)),
            'median': float(np.median(masked_data)),
            'std': float(np.std(masked_data)),
            'min': float(np.min(masked_data)),
            'max': float(np.max(masked_data)),
            'percentile_25': float(np.percentile(masked_data, 25)),
            'percentile_75': float(np.percentile(masked_data, 75)),
            'percentile_95': float(np.percentile(masked_data, 95))
        }

        logger.info(f"{metric_name} statistics:")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  Std: {stats['std']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        return stats

    def detect_dti_outliers(
        self,
        data: np.ndarray,
        metric_name: str,
        expected_range: Tuple[float, float]
    ) -> Dict:
        """
        Detect outlier voxels based on expected physiological ranges.

        Parameters
        ----------
        data : np.ndarray
            DTI metric data
        metric_name : str
            Metric name
        expected_range : tuple
            Expected (min, max) values for the metric

        Returns
        -------
        dict
            Outlier information
        """
        mask = data > 0
        masked_data = data[mask]

        min_val, max_val = expected_range

        below_min = np.sum(masked_data < min_val)
        above_max = np.sum(masked_data > max_val)
        total_voxels = len(masked_data)

        outlier_info = {
            'metric': metric_name,
            'expected_range': expected_range,
            'n_below_min': int(below_min),
            'n_above_max': int(above_max),
            'percent_outliers': float((below_min + above_max) / total_voxels * 100) if total_voxels > 0 else 0.0
        }

        if outlier_info['percent_outliers'] > 5.0:
            logger.warning(
                f"{metric_name} has {outlier_info['percent_outliers']:.1f}% "
                f"outliers outside [{min_val}, {max_val}]"
            )

        return outlier_info

    def plot_dti_histogram(
        self,
        data: np.ndarray,
        metric_name: str,
        stats: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Plot histogram of DTI metric values.

        Parameters
        ----------
        data : np.ndarray
            DTI metric data
        metric_name : str
            Metric name
        stats : dict
            Statistics dictionary
        output_file : Path, optional
            Output file path

        Returns
        -------
        Path
            Path to saved plot
        """
        if output_file is None:
            output_file = self.qc_dir / f'{metric_name.lower()}_histogram.png'

        mask = data > 0
        masked_data = data[mask]

        if len(masked_data) == 0:
            logger.warning(f"No data to plot for {metric_name}")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        n_bins = 100
        ax.hist(masked_data, bins=n_bins, color='steelblue', alpha=0.7, edgecolor='black')

        # Add mean and median lines
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.4f}")
        ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.4f}")

        ax.set_xlabel(f'{metric_name} Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{metric_name} Distribution - {self.subject}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add stats text box
        stats_text = (
            f"Mean: {stats['mean']:.4f}\n"
            f"Median: {stats['median']:.4f}\n"
            f"Std: {stats['std']:.4f}\n"
            f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
            f"N voxels: {stats['n_voxels']}"
        )
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved {metric_name} histogram: {output_file}")
        return output_file

    def save_metrics_json(
        self,
        all_stats: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Save DTI metrics to JSON file.

        Parameters
        ----------
        all_stats : dict
            Combined DTI statistics
        output_file : Path, optional
            Output JSON file

        Returns
        -------
        Path
            Path to saved JSON file
        """
        if output_file is None:
            output_file = self.qc_dir.parent / 'metrics' / 'dti.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

        logger.info(f"Saved DTI metrics: {output_file}")
        return output_file

    def run_qc(
        self,
        metrics: Optional[List[str]] = None,
        mask_file: Optional[Path] = None
    ) -> Dict:
        """
        Run complete DTI QC analysis.

        Parameters
        ----------
        metrics : list of str, optional
            List of metrics to analyze. Default: ['FA', 'MD']
        mask_file : Path, optional
            Path to brain mask file

        Returns
        -------
        dict
            Combined QC results and output paths
        """
        logger.info(f"Running DTI QC for {self.subject}")

        if metrics is None:
            metrics = ['FA', 'MD']

        # Load mask if provided
        brain_mask = None
        if mask_file and mask_file.exists():
            logger.info(f"Loading brain mask: {mask_file}")
            mask_img = nib.load(mask_file)
            brain_mask = mask_img.get_fdata() > 0

        # Expected physiological ranges
        expected_ranges = {
            'FA': (0.0, 1.0),   # FA is bounded [0, 1]
            'MD': (0.0, 0.003), # MD in mm²/s, typical brain range
            'L1': (0.0, 0.003), # Primary eigenvector
            'L2': (0.0, 0.003), # Secondary eigenvector
            'L3': (0.0, 0.003), # Tertiary eigenvector
            'AD': (0.0, 0.003), # Axial diffusivity (= L1)
            'RD': (0.0, 0.003)  # Radial diffusivity (= (L2 + L3) / 2)
        }

        all_stats = {'subject': self.subject, 'metrics': {}}
        outputs = {}

        for metric in metrics:
            logger.info(f"Analyzing {metric}...")

            # Load DTI map
            data, img = self.load_dti_map(metric)

            if data is None:
                logger.warning(f"Skipping {metric} (not found)")
                continue

            # Calculate statistics
            stats = self.calculate_dti_statistics(data, metric, brain_mask)

            if not stats:
                continue

            # Detect outliers
            if metric in expected_ranges:
                outlier_info = self.detect_dti_outliers(data, metric, expected_ranges[metric])
                stats['outliers'] = outlier_info

            # Plot histogram
            hist_plot = self.plot_dti_histogram(data, metric, stats)
            if hist_plot:
                outputs[f'{metric.lower()}_histogram'] = str(hist_plot)

            all_stats['metrics'][metric] = stats

        # Save metrics to JSON
        if all_stats['metrics']:
            metrics_file = self.save_metrics_json(all_stats)
            outputs['metrics_json'] = str(metrics_file)

        results = {
            'subject': self.subject,
            'dti_stats': all_stats,
            'outputs': outputs
        }

        logger.info("DTI QC completed")
        return results
