#!/usr/bin/env python3
"""
TOPUP Quality Control Module.

Extracts and analyzes TOPUP convergence metrics, field maps,
and distortion correction effectiveness.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


class TOPUPQualityControl:
    """
    Quality control for TOPUP distortion correction.

    Analyzes TOPUP convergence, field maps, and correction effectiveness.
    """

    def __init__(self, subject: str, work_dir: Path, qc_dir: Path):
        """
        Initialize TOPUP QC.

        Parameters
        ----------
        subject : str
            Subject identifier
        work_dir : Path
            Working directory containing TOPUP outputs
        qc_dir : Path
            QC output directory (e.g., {study_root}/qc/dwi/{subject}/topup/)
        """
        self.subject = subject
        self.work_dir = Path(work_dir)
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized TOPUP QC for {subject}")
        logger.info(f"  Work dir: {self.work_dir}")
        logger.info(f"  QC dir: {self.qc_dir}")

    def extract_convergence_metrics(
        self,
        topup_log: Optional[Path] = None
    ) -> Dict:
        """
        Extract TOPUP convergence metrics from log file or stdout.

        Parameters
        ----------
        topup_log : Path, optional
            Path to TOPUP log file. If None, will look in work_dir

        Returns
        -------
        dict
            Convergence metrics including iterations, SSD values, etc.
        """
        # Find log file if not provided
        if topup_log is None:
            # Look for common log file patterns
            log_candidates = list(self.work_dir.glob('*.topup_log'))
            if not log_candidates:
                log_candidates = list(self.work_dir.glob('topup*.log'))
            if log_candidates:
                topup_log = log_candidates[0]
            else:
                logger.warning("No TOPUP log file found, will look for stdout")

        metrics = {
            'subject': self.subject,
            'converged': False,
            'iterations': 0,
            'final_ssd': None,
            'initial_ssd': None,
            'improvement_percent': None,
            'ssd_by_iteration': [],
            'convergence_rate': 'unknown'
        }

        # Parse log file
        if topup_log and topup_log.exists():
            logger.info(f"Parsing TOPUP log: {topup_log}")
            with open(topup_log, 'r') as f:
                log_content = f.read()

            # Extract SSD values from log
            # Pattern: "Iteration X: SSD = Y"
            pattern = r'Iteration\s+(\d+).*?SSD\s*=\s*([\d.e+-]+)'
            matches = re.findall(pattern, log_content, re.IGNORECASE)

            if matches:
                for iteration, ssd in matches:
                    ssd_value = float(ssd)
                    metrics['ssd_by_iteration'].append(ssd_value)

                metrics['iterations'] = len(matches)
                metrics['initial_ssd'] = metrics['ssd_by_iteration'][0]
                metrics['final_ssd'] = metrics['ssd_by_iteration'][-1]

                # Calculate improvement
                if metrics['initial_ssd'] > 0:
                    metrics['improvement_percent'] = (
                        (metrics['initial_ssd'] - metrics['final_ssd']) /
                        metrics['initial_ssd'] * 100
                    )

                # Determine convergence
                # Converged if: final < initial and iterations < 25
                metrics['converged'] = (
                    metrics['final_ssd'] < metrics['initial_ssd'] and
                    metrics['iterations'] < 25
                )

                # Assess convergence rate
                if metrics['iterations'] <= 10:
                    metrics['convergence_rate'] = 'fast'
                elif metrics['iterations'] <= 15:
                    metrics['convergence_rate'] = 'normal'
                else:
                    metrics['convergence_rate'] = 'slow'

                logger.info(f"TOPUP convergence: {metrics['iterations']} iterations")
                logger.info(f"  Initial SSD: {metrics['initial_ssd']:.2f}")
                logger.info(f"  Final SSD: {metrics['final_ssd']:.2f}")
                logger.info(f"  Improvement: {metrics['improvement_percent']:.1f}%")

        return metrics

    def plot_convergence(
        self,
        metrics: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Plot TOPUP convergence curve.

        Parameters
        ----------
        metrics : dict
            Convergence metrics from extract_convergence_metrics()
        output_file : Path, optional
            Output file path. If None, will save to qc_dir

        Returns
        -------
        Path
            Path to saved plot
        """
        if output_file is None:
            output_file = self.qc_dir / 'convergence_plot.png'

        ssd_values = metrics['ssd_by_iteration']
        if not ssd_values:
            logger.warning("No SSD values to plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = list(range(1, len(ssd_values) + 1))
        ax.plot(iterations, ssd_values, 'b-o', linewidth=2, markersize=6)

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Sum of Squared Differences (SSD)', fontsize=12)
        ax.set_title(f'TOPUP Convergence - {self.subject}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add final SSD as text
        ax.text(
            0.95, 0.95,
            f"Final SSD: {metrics['final_ssd']:.2f}\n"
            f"Iterations: {metrics['iterations']}\n"
            f"Improvement: {metrics['improvement_percent']:.1f}%",
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved convergence plot: {output_file}")
        return output_file

    def analyze_field_map(
        self,
        fieldcoef_file: Optional[Path] = None
    ) -> Dict:
        """
        Analyze TOPUP field coefficient map.

        Parameters
        ----------
        fieldcoef_file : Path, optional
            Path to field coefficient file. If None, will look in work_dir

        Returns
        -------
        dict
            Field map statistics (mean/max displacement, etc.)
        """
        if fieldcoef_file is None:
            # Look for fieldcoef file
            fieldcoef_candidates = list(self.work_dir.glob('*fieldcoef*.nii.gz'))
            if fieldcoef_candidates:
                fieldcoef_file = fieldcoef_candidates[0]
            else:
                logger.warning("No field coefficient file found")
                return {}

        if not fieldcoef_file.exists():
            logger.warning(f"Field coefficient file not found: {fieldcoef_file}")
            return {}

        logger.info(f"Analyzing field map: {fieldcoef_file}")

        # Load field map
        field_img = nib.load(fieldcoef_file)
        field_data = field_img.get_fdata()

        # Calculate displacement statistics
        # Field coefficients are typically in Hz, convert to mm
        # Assumes 3T scanner (gamma = 42.58 MHz/T)
        # displacement_mm = field_Hz / (gamma * B0)
        # For simplicity, we'll report raw field values

        metrics = {
            'subject': self.subject,
            'field_statistics': {
                'mean_field_hz': float(np.mean(np.abs(field_data))),
                'max_field_hz': float(np.max(np.abs(field_data))),
                'std_field_hz': float(np.std(field_data)),
                'percentile_95_hz': float(np.percentile(np.abs(field_data), 95))
            }
        }

        logger.info(f"Field map stats:")
        logger.info(f"  Mean: {metrics['field_statistics']['mean_field_hz']:.2f} Hz")
        logger.info(f"  Max: {metrics['field_statistics']['max_field_hz']:.2f} Hz")

        return metrics

    def save_metrics_json(
        self,
        convergence_metrics: Dict,
        field_metrics: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Save combined TOPUP metrics to JSON file.

        Parameters
        ----------
        convergence_metrics : dict
            Convergence metrics
        field_metrics : dict
            Field map statistics
        output_file : Path, optional
            Output JSON file. If None, will save to qc_dir

        Returns
        -------
        Path
            Path to saved JSON file
        """
        if output_file is None:
            output_file = self.qc_dir.parent / 'metrics' / 'topup_convergence.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Combine metrics
        combined_metrics = {
            **convergence_metrics,
            **field_metrics
        }

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(combined_metrics, f, indent=2)

        logger.info(f"Saved TOPUP metrics: {output_file}")
        return output_file

    def run_qc(
        self,
        topup_log: Optional[Path] = None,
        fieldcoef_file: Optional[Path] = None
    ) -> Dict:
        """
        Run complete TOPUP QC analysis.

        Parameters
        ----------
        topup_log : Path, optional
            Path to TOPUP log file
        fieldcoef_file : Path, optional
            Path to field coefficient file

        Returns
        -------
        dict
            Combined QC metrics and output paths
        """
        logger.info(f"Running TOPUP QC for {self.subject}")

        # Extract convergence metrics
        convergence_metrics = self.extract_convergence_metrics(topup_log)

        # Plot convergence
        convergence_plot = self.plot_convergence(convergence_metrics)

        # Analyze field map
        field_metrics = self.analyze_field_map(fieldcoef_file)

        # Save metrics to JSON
        metrics_file = self.save_metrics_json(convergence_metrics, field_metrics)

        results = {
            'subject': self.subject,
            'convergence_metrics': convergence_metrics,
            'field_metrics': field_metrics,
            'outputs': {
                'convergence_plot': str(convergence_plot) if convergence_plot else None,
                'metrics_json': str(metrics_file)
            }
        }

        logger.info("TOPUP QC completed")
        return results
