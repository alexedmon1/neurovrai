#!/usr/bin/env python3
"""
Tissue Segmentation Quality Control Module.

Analyzes FAST tissue segmentation quality:
- GM/WM/CSF volume statistics
- Tissue probability distributions
- Tissue ratio validation
- Segmentation visualization
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


class SegmentationQualityControl:
    """
    Quality control for tissue segmentation (FAST).

    Analyzes GM, WM, and CSF segmentation quality.
    """

    def __init__(self, subject: str, anat_dir: Path, qc_dir: Path):
        """
        Initialize Segmentation QC.

        Parameters
        ----------
        subject : str
            Subject identifier
        anat_dir : Path
            Directory containing FAST segmentation outputs
        qc_dir : Path
            QC output directory (e.g., {study_root}/qc/anat/{subject}/segmentation/)
        """
        self.subject = subject
        self.anat_dir = Path(anat_dir)
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Segmentation QC for {subject}")
        logger.info(f"  Anat dir: {self.anat_dir}")
        logger.info(f"  QC dir: {self.qc_dir}")

    def load_tissue_maps(
        self,
        csf_file: Optional[Path] = None,
        gm_file: Optional[Path] = None,
        wm_file: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load tissue probability maps (FAST outputs).

        FAST outputs are typically named:
        - *_pve_0.nii.gz (CSF)
        - *_pve_1.nii.gz (GM)
        - *_pve_2.nii.gz (WM)

        Parameters
        ----------
        csf_file : Path, optional
            CSF probability map
        gm_file : Path, optional
            GM probability map
        wm_file : Path, optional
            WM probability map

        Returns
        -------
        csf_data : np.ndarray
            CSF probability map
        gm_data : np.ndarray
            GM probability map
        wm_data : np.ndarray
            WM probability map
        """
        # Auto-detect tissue maps if not provided
        if csf_file is None:
            patterns = ['*_pve_0.nii.gz', '*csf*.nii.gz', '*CSF*.nii.gz']
            for pattern in patterns:
                candidates = list(self.anat_dir.glob(pattern))
                if candidates:
                    csf_file = candidates[0]
                    break

        if gm_file is None:
            patterns = ['*_pve_1.nii.gz', '*gm*.nii.gz', '*GM*.nii.gz']
            for pattern in patterns:
                candidates = list(self.anat_dir.glob(pattern))
                if candidates:
                    gm_file = candidates[0]
                    break

        if wm_file is None:
            patterns = ['*_pve_2.nii.gz', '*wm*.nii.gz', '*WM*.nii.gz']
            for pattern in patterns:
                candidates = list(self.anat_dir.glob(pattern))
                if candidates:
                    wm_file = candidates[0]
                    break

        # Load tissue maps
        csf_data = None
        gm_data = None
        wm_data = None

        if csf_file and csf_file.exists():
            logger.info(f"Loading CSF: {csf_file.name}")
            csf_data = nib.load(csf_file).get_fdata()

        if gm_file and gm_file.exists():
            logger.info(f"Loading GM: {gm_file.name}")
            gm_data = nib.load(gm_file).get_fdata()

        if wm_file and wm_file.exists():
            logger.info(f"Loading WM: {wm_file.name}")
            wm_data = nib.load(wm_file).get_fdata()

        return csf_data, gm_data, wm_data

    def calculate_tissue_volumes(
        self,
        csf_data: np.ndarray,
        gm_data: np.ndarray,
        wm_data: np.ndarray,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        threshold: float = 0.5
    ) -> Dict:
        """
        Calculate tissue volumes from probability maps.

        Parameters
        ----------
        csf_data : np.ndarray
            CSF probability map
        gm_data : np.ndarray
            GM probability map
        wm_data : np.ndarray
            WM probability map
        voxel_size : tuple
            Voxel dimensions (mm)
        threshold : float
            Probability threshold for tissue assignment (default: 0.5)

        Returns
        -------
        dict
            Tissue volume statistics
        """
        voxel_volume_mm3 = np.prod(voxel_size)

        # Calculate volumes based on probability threshold
        csf_voxels = int(np.sum(csf_data > threshold)) if csf_data is not None else 0
        gm_voxels = int(np.sum(gm_data > threshold)) if gm_data is not None else 0
        wm_voxels = int(np.sum(wm_data > threshold)) if wm_data is not None else 0

        csf_volume_cm3 = (csf_voxels * voxel_volume_mm3) / 1000.0
        gm_volume_cm3 = (gm_voxels * voxel_volume_mm3) / 1000.0
        wm_volume_cm3 = (wm_voxels * voxel_volume_mm3) / 1000.0

        total_volume_cm3 = csf_volume_cm3 + gm_volume_cm3 + wm_volume_cm3

        # Calculate tissue fractions
        csf_fraction = csf_volume_cm3 / total_volume_cm3 if total_volume_cm3 > 0 else 0
        gm_fraction = gm_volume_cm3 / total_volume_cm3 if total_volume_cm3 > 0 else 0
        wm_fraction = wm_volume_cm3 / total_volume_cm3 if total_volume_cm3 > 0 else 0

        volumes = {
            'subject': self.subject,
            'threshold': threshold,
            'voxel_size_mm': list(voxel_size),
            'csf': {
                'voxels': csf_voxels,
                'volume_cm3': float(csf_volume_cm3),
                'fraction': float(csf_fraction)
            },
            'gm': {
                'voxels': gm_voxels,
                'volume_cm3': float(gm_volume_cm3),
                'fraction': float(gm_fraction)
            },
            'wm': {
                'voxels': wm_voxels,
                'volume_cm3': float(wm_volume_cm3),
                'fraction': float(wm_fraction)
            },
            'total_volume_cm3': float(total_volume_cm3)
        }

        logger.info(f"Tissue volumes (threshold={threshold}):")
        logger.info(f"  CSF: {csf_volume_cm3:.2f} cm³ ({csf_fraction*100:.1f}%)")
        logger.info(f"  GM:  {gm_volume_cm3:.2f} cm³ ({gm_fraction*100:.1f}%)")
        logger.info(f"  WM:  {wm_volume_cm3:.2f} cm³ ({wm_fraction*100:.1f}%)")
        logger.info(f"  Total: {total_volume_cm3:.2f} cm³")

        return volumes

    def validate_tissue_ratios(
        self,
        volumes: Dict
    ) -> Dict:
        """
        Validate tissue ratios against expected physiological ranges.

        Typical adult brain composition:
        - GM: 40-50%
        - WM: 35-45%
        - CSF: 10-25%

        Parameters
        ----------
        volumes : dict
            Tissue volume statistics

        Returns
        -------
        dict
            Validation results
        """
        gm_fraction = volumes['gm']['fraction']
        wm_fraction = volumes['wm']['fraction']
        csf_fraction = volumes['csf']['fraction']

        quality_flags = []

        # Check GM fraction
        if gm_fraction < 0.30 or gm_fraction > 0.60:
            quality_flags.append('GM_FRACTION_ABNORMAL')
            logger.warning(f"GM fraction outside expected range: {gm_fraction*100:.1f}%")

        # Check WM fraction
        if wm_fraction < 0.25 or wm_fraction > 0.55:
            quality_flags.append('WM_FRACTION_ABNORMAL')
            logger.warning(f"WM fraction outside expected range: {wm_fraction*100:.1f}%")

        # Check CSF fraction
        if csf_fraction < 0.05 or csf_fraction > 0.35:
            quality_flags.append('CSF_FRACTION_ABNORMAL')
            logger.warning(f"CSF fraction outside expected range: {csf_fraction*100:.1f}%")

        # Check GM/WM ratio (typically 0.8 - 1.5)
        gm_wm_ratio = gm_fraction / wm_fraction if wm_fraction > 0 else 0
        if gm_wm_ratio < 0.6 or gm_wm_ratio > 2.0:
            quality_flags.append('GM_WM_RATIO_ABNORMAL')
            logger.warning(f"GM/WM ratio abnormal: {gm_wm_ratio:.2f}")

        validation = {
            'gm_wm_ratio': float(gm_wm_ratio),
            'quality_flags': quality_flags,
            'quality_pass': len(quality_flags) == 0,
            'expected_ranges': {
                'gm_fraction': [0.30, 0.60],
                'wm_fraction': [0.25, 0.55],
                'csf_fraction': [0.05, 0.35],
                'gm_wm_ratio': [0.6, 2.0]
            }
        }

        if validation['quality_pass']:
            logger.info("Tissue ratios within expected ranges")
        else:
            logger.warning(f"Quality flags: {quality_flags}")

        return validation

    def plot_tissue_volumes(
        self,
        volumes: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Plot tissue volume bar chart.

        Parameters
        ----------
        volumes : dict
            Tissue volume statistics
        output_file : Path, optional
            Output file path

        Returns
        -------
        Path
            Path to saved plot
        """
        if output_file is None:
            output_file = self.qc_dir / 'tissue_volumes.png'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Absolute volumes
        tissues = ['CSF', 'GM', 'WM']
        volume_values = [
            volumes['csf']['volume_cm3'],
            volumes['gm']['volume_cm3'],
            volumes['wm']['volume_cm3']
        ]
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

        ax1.bar(tissues, volume_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Volume (cm³)', fontsize=12)
        ax1.set_title('Tissue Volumes', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for i, v in enumerate(volume_values):
            ax1.text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold')

        # Tissue fractions (pie chart)
        fractions = [
            volumes['csf']['fraction'],
            volumes['gm']['fraction'],
            volumes['wm']['fraction']
        ]

        ax2.pie(fractions, labels=tissues, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Tissue Fractions', fontsize=14, fontweight='bold')

        plt.suptitle(f'Tissue Segmentation - {self.subject}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved tissue volume plot: {output_file}")
        return output_file

    def save_metrics_json(
        self,
        volumes: Dict,
        validation: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Save segmentation metrics to JSON.

        Parameters
        ----------
        volumes : dict
            Tissue volume statistics
        validation : dict
            Validation results
        output_file : Path, optional
            Output JSON file

        Returns
        -------
        Path
            Path to saved JSON
        """
        if output_file is None:
            output_file = self.qc_dir.parent / 'metrics' / 'segmentation.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)

        combined_metrics = {
            **volumes,
            'validation': validation
        }

        with open(output_file, 'w') as f:
            json.dump(combined_metrics, f, indent=2)

        logger.info(f"Saved segmentation metrics: {output_file}")
        return output_file

    def run_qc(
        self,
        csf_file: Optional[Path] = None,
        gm_file: Optional[Path] = None,
        wm_file: Optional[Path] = None,
        threshold: float = 0.5
    ) -> Dict:
        """
        Run complete segmentation QC.

        Parameters
        ----------
        csf_file : Path, optional
            CSF probability map
        gm_file : Path, optional
            GM probability map
        wm_file : Path, optional
            WM probability map
        threshold : float, optional
            Probability threshold (default: 0.5)

        Returns
        -------
        dict
            Combined QC results and output paths
        """
        logger.info(f"Running Segmentation QC for {self.subject}")

        # Load tissue maps
        csf_data, gm_data, wm_data = self.load_tissue_maps(csf_file, gm_file, wm_file)

        if csf_data is None and gm_data is None and wm_data is None:
            logger.warning("No tissue maps found, skipping Segmentation QC")
            return {
                'subject': self.subject,
                'volumes': {},
                'validation': {},
                'outputs': {}
            }

        # Calculate tissue volumes
        volumes = self.calculate_tissue_volumes(
            csf_data, gm_data, wm_data,
            voxel_size=(1.0, 1.0, 1.0),  # Can be extracted from NIfTI header
            threshold=threshold
        )

        # Validate tissue ratios
        validation = self.validate_tissue_ratios(volumes)

        # Plot tissue volumes
        volume_plot = self.plot_tissue_volumes(volumes)

        # Save metrics
        metrics_file = self.save_metrics_json(volumes, validation)

        results = {
            'subject': self.subject,
            'volumes': volumes,
            'validation': validation,
            'outputs': {
                'volume_plot': str(volume_plot) if volume_plot else None,
                'metrics_json': str(metrics_file)
            }
        }

        logger.info("Segmentation QC completed")
        return results
