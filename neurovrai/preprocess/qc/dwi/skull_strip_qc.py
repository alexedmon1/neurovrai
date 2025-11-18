#!/usr/bin/env python3
"""
Skull Stripping Quality Control Module for DWI.

Evaluates BET skull stripping quality on DWI data:
- Brain mask coverage statistics
- Over/under-stripping detection
- Visual slice overlays on b0 images
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


class DWISkullStripQualityControl:
    """
    Quality control for DWI skull stripping (BET on mean b0).

    Analyzes brain extraction quality and mask statistics.
    """

    def __init__(self, subject: str, dwi_dir: Path, qc_dir: Path):
        """
        Initialize DWI Skull Strip QC.

        Parameters
        ----------
        subject : str
            Subject identifier
        dwi_dir : Path
            Directory containing DWI outputs (b0, mask)
        qc_dir : Path
            QC output directory (e.g., {study_root}/qc/dwi/{subject}/skull_strip/)
        """
        self.subject = subject
        self.dwi_dir = Path(dwi_dir)
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized DWI Skull Strip QC for {subject}")
        logger.info(f"  DWI dir: {self.dwi_dir}")
        logger.info(f"  QC dir: {self.qc_dir}")

    def load_images(
        self,
        b0_file: Optional[Path] = None,
        mask_file: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load b0 and mask images.

        Parameters
        ----------
        b0_file : Path, optional
            Mean b0 image or DWI reference
        mask_file : Path, optional
            Brain mask

        Returns
        -------
        b0_data : np.ndarray
            b0 image data
        mask_data : np.ndarray
            Brain mask data
        """
        # Auto-detect files if not provided
        if b0_file is None:
            patterns = ['*b0_mean.nii.gz', '*b0.nii.gz', '*nodif.nii.gz']
            for pattern in patterns:
                candidates = list(self.dwi_dir.glob(pattern))
                if candidates:
                    b0_file = candidates[0]
                    break

        if mask_file is None:
            patterns = ['*brain_mask.nii.gz', '*mask.nii.gz', '*nodif_brain_mask.nii.gz']
            for pattern in patterns:
                candidates = list(self.dwi_dir.glob(pattern))
                if candidates:
                    mask_file = candidates[0]
                    break

        # Load images
        b0_data = None
        mask_data = None

        if b0_file and b0_file.exists():
            logger.info(f"Loading b0: {b0_file.name}")
            b0_data = nib.load(b0_file).get_fdata()

        if mask_file and mask_file.exists():
            logger.info(f"Loading mask: {mask_file.name}")
            mask_data = nib.load(mask_file).get_fdata() > 0

        return b0_data, mask_data

    def calculate_mask_statistics(
        self,
        mask_data: np.ndarray,
        voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    ) -> Dict:
        """
        Calculate brain mask statistics.

        Parameters
        ----------
        mask_data : np.ndarray
            Brain mask
        voxel_size : tuple
            Voxel dimensions (mm) - DWI typically 2mm isotropic

        Returns
        -------
        dict
            Mask statistics
        """
        if mask_data is None:
            return {}

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

        stats = {
            'subject': self.subject,
            'modality': 'dwi',
            'n_voxels': n_voxels,
            'brain_volume_mm3': float(brain_volume_mm3),
            'brain_volume_cm3': float(brain_volume_cm3),
            'voxel_size_mm': list(voxel_size),
            'bbox': bbox,
            'bbox_size': bbox_size
        }

        logger.info(f"DWI Mask statistics:")
        logger.info(f"  Brain volume: {brain_volume_cm3:.2f} cm³")
        logger.info(f"  N voxels: {n_voxels}")
        logger.info(f"  Bounding box: {bbox_size}")

        return stats

    def check_stripping_quality(
        self,
        b0_data: np.ndarray,
        mask_data: np.ndarray
    ) -> Dict:
        """
        Check for over-stripping or under-stripping.

        Parameters
        ----------
        b0_data : np.ndarray
            b0 image data
        mask_data : np.ndarray
            Brain mask

        Returns
        -------
        dict
            Quality assessment
        """
        if b0_data is None or mask_data is None:
            return {}

        # Calculate intensity statistics inside and outside mask
        brain_intensities = b0_data[mask_data > 0]
        outside_intensities = b0_data[mask_data == 0]

        # Remove zero values (background)
        brain_intensities = brain_intensities[brain_intensities > 0]
        outside_intensities = outside_intensities[outside_intensities > 0]

        brain_mean = np.mean(brain_intensities)
        brain_std = np.std(brain_intensities)
        outside_mean = np.mean(outside_intensities) if len(outside_intensities) > 0 else 0
        outside_std = np.std(outside_intensities) if len(outside_intensities) > 0 else 0

        # Contrast between brain and non-brain
        contrast_ratio = brain_mean / outside_mean if outside_mean > 0 else np.inf

        # Assess quality - DWI has lower contrast than T1w
        quality_flags = []

        if contrast_ratio < 1.5:  # Lower threshold for DWI
            quality_flags.append('LOW_CONTRAST')
            logger.warning("Low contrast between brain and non-brain regions")

        if brain_std / brain_mean > 0.6:  # DWI can have more variance
            quality_flags.append('HIGH_VARIANCE')
            logger.warning("High intensity variance within brain mask")

        # Check brain volume is reasonable (typical range: 800-1800 cm³)
        brain_volume_cm3 = np.sum(mask_data) * np.prod([2.0, 2.0, 2.0]) / 1000.0
        if brain_volume_cm3 < 500:
            quality_flags.append('SMALL_BRAIN_VOLUME')
            logger.warning(f"Unusually small brain volume: {brain_volume_cm3:.1f} cm³")
        elif brain_volume_cm3 > 2500:
            quality_flags.append('LARGE_BRAIN_VOLUME')
            logger.warning(f"Unusually large brain volume: {brain_volume_cm3:.1f} cm³")

        quality = {
            'brain_mean_intensity': float(brain_mean),
            'brain_std_intensity': float(brain_std),
            'outside_mean_intensity': float(outside_mean),
            'outside_std_intensity': float(outside_std),
            'contrast_ratio': float(contrast_ratio),
            'quality_flags': quality_flags,
            'quality_pass': len(quality_flags) == 0
        }

        logger.info(f"DWI Stripping quality:")
        logger.info(f"  Contrast ratio: {contrast_ratio:.2f}")
        logger.info(f"  Quality flags: {quality_flags if quality_flags else 'PASS'}")

        return quality

    def plot_mask_overlay(
        self,
        b0_data: np.ndarray,
        mask_data: np.ndarray,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Plot brain mask overlay on b0 image.

        Parameters
        ----------
        b0_data : np.ndarray
            b0 image data
        mask_data : np.ndarray
            Brain mask
        output_file : Path, optional
            Output file path

        Returns
        -------
        Path
            Path to saved plot
        """
        if output_file is None:
            output_file = self.qc_dir / 'brain_mask_overlay.png'

        if b0_data is None or mask_data is None:
            logger.warning("Cannot create overlay plot without b0 and mask")
            return None

        # Get central slices
        center_x = b0_data.shape[0] // 2
        center_y = b0_data.shape[1] // 2
        center_z = b0_data.shape[2] // 2

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Sagittal slices
        for i, offset in enumerate([-5, 0, 5]):
            ax = axes[0, i]
            slice_idx = center_x + offset
            ax.imshow(b0_data[slice_idx, :, :].T, cmap='gray', origin='lower')
            ax.contour(mask_data[slice_idx, :, :].T, colors='red', linewidths=1)
            ax.set_title(f'Sagittal (x={slice_idx})', fontsize=10)
            ax.axis('off')

        # Axial slices
        for i, offset in enumerate([-5, 0, 5]):
            ax = axes[1, i]
            slice_idx = center_z + offset
            ax.imshow(b0_data[:, :, slice_idx].T, cmap='gray', origin='lower')
            ax.contour(mask_data[:, :, slice_idx].T, colors='red', linewidths=1)
            ax.set_title(f'Axial (z={slice_idx})', fontsize=10)
            ax.axis('off')

        plt.suptitle(f'DWI Brain Mask Overlay - {self.subject}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved DWI mask overlay: {output_file}")
        return output_file

    def save_metrics_json(
        self,
        mask_stats: Dict,
        quality_check: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Save skull stripping metrics to JSON.

        Parameters
        ----------
        mask_stats : dict
            Mask statistics
        quality_check : dict
            Quality assessment
        output_file : Path, optional
            Output JSON file

        Returns
        -------
        Path
            Path to saved JSON
        """
        if output_file is None:
            output_file = self.qc_dir.parent / 'metrics' / 'skull_strip.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)

        combined_metrics = {
            **mask_stats,
            'quality': quality_check
        }

        with open(output_file, 'w') as f:
            json.dump(combined_metrics, f, indent=2)

        logger.info(f"Saved DWI skull strip metrics: {output_file}")
        return output_file

    def run_qc(
        self,
        b0_file: Optional[Path] = None,
        mask_file: Optional[Path] = None
    ) -> Dict:
        """
        Run complete DWI skull stripping QC.

        Parameters
        ----------
        b0_file : Path, optional
            Mean b0 image
        mask_file : Path, optional
            Brain mask

        Returns
        -------
        dict
            Combined QC results and output paths
        """
        logger.info(f"Running DWI Skull Strip QC for {self.subject}")

        # Load images
        b0_data, mask_data = self.load_images(b0_file, mask_file)

        if mask_data is None:
            logger.warning("No brain mask found, skipping DWI Skull Strip QC")
            return {
                'subject': self.subject,
                'modality': 'dwi',
                'mask_stats': {},
                'quality': {},
                'outputs': {}
            }

        # Calculate mask statistics (DWI typically 2mm isotropic)
        mask_stats = self.calculate_mask_statistics(mask_data, voxel_size=(2.0, 2.0, 2.0))

        # Check stripping quality
        quality_check = self.check_stripping_quality(b0_data, mask_data) if b0_data is not None else {}

        # Plot mask overlay
        overlay_plot = self.plot_mask_overlay(b0_data, mask_data) if b0_data is not None else None

        # Save metrics
        metrics_file = self.save_metrics_json(mask_stats, quality_check)

        results = {
            'subject': self.subject,
            'modality': 'dwi',
            'mask_stats': mask_stats,
            'quality': quality_check,
            'outputs': {
                'mask_overlay': str(overlay_plot) if overlay_plot else None,
                'metrics_json': str(metrics_file)
            }
        }

        logger.info("DWI Skull Strip QC completed")
        return results
