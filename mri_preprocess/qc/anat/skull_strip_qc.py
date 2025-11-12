#!/usr/bin/env python3
"""
Skull Stripping Quality Control Module.

Evaluates BET skull stripping quality:
- Brain mask coverage statistics
- Over/under-stripping detection
- Visual slice overlays
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


class SkullStripQualityControl:
    """
    Quality control for skull stripping (BET).

    Analyzes brain extraction quality and mask statistics.
    """

    def __init__(self, subject: str, anat_dir: Path, qc_dir: Path):
        """
        Initialize Skull Strip QC.

        Parameters
        ----------
        subject : str
            Subject identifier
        anat_dir : Path
            Directory containing anatomical outputs (brain, mask)
        qc_dir : Path
            QC output directory (e.g., {study_root}/qc/anat/{subject}/skull_strip/)
        """
        self.subject = subject
        self.anat_dir = Path(anat_dir)
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Skull Strip QC for {subject}")
        logger.info(f"  Anat dir: {self.anat_dir}")
        logger.info(f"  QC dir: {self.qc_dir}")

    def load_images(
        self,
        t1w_file: Optional[Path] = None,
        brain_file: Optional[Path] = None,
        mask_file: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load T1w, brain, and mask images.

        Parameters
        ----------
        t1w_file : Path, optional
            Original T1w image
        brain_file : Path, optional
            Brain-extracted image
        mask_file : Path, optional
            Brain mask

        Returns
        -------
        t1w_data : np.ndarray
            T1w image data
        brain_data : np.ndarray
            Brain-extracted data
        mask_data : np.ndarray
            Brain mask data
        """
        # Auto-detect files if not provided
        if t1w_file is None:
            patterns = ['*T1w.nii.gz', '*t1w.nii.gz', '*T1*.nii.gz']
            for pattern in patterns:
                candidates = list(self.anat_dir.glob(pattern))
                if candidates:
                    t1w_file = candidates[0]
                    break

        if brain_file is None:
            patterns = ['*brain.nii.gz', '*bet.nii.gz']
            for pattern in patterns:
                candidates = list(self.anat_dir.glob(pattern))
                if candidates:
                    brain_file = candidates[0]
                    break

        if mask_file is None:
            patterns = ['*brain_mask.nii.gz', '*mask.nii.gz']
            for pattern in patterns:
                candidates = list(self.anat_dir.glob(pattern))
                if candidates:
                    mask_file = candidates[0]
                    break

        # Load images
        t1w_data = None
        brain_data = None
        mask_data = None

        if t1w_file and t1w_file.exists():
            logger.info(f"Loading T1w: {t1w_file.name}")
            t1w_data = nib.load(t1w_file).get_fdata()

        if brain_file and brain_file.exists():
            logger.info(f"Loading brain: {brain_file.name}")
            brain_data = nib.load(brain_file).get_fdata()

        if mask_file and mask_file.exists():
            logger.info(f"Loading mask: {mask_file.name}")
            mask_data = nib.load(mask_file).get_fdata() > 0

        return t1w_data, brain_data, mask_data

    def calculate_mask_statistics(
        self,
        mask_data: np.ndarray,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict:
        """
        Calculate brain mask statistics.

        Parameters
        ----------
        mask_data : np.ndarray
            Brain mask
        voxel_size : tuple
            Voxel dimensions (mm)

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
            'n_voxels': n_voxels,
            'brain_volume_mm3': float(brain_volume_mm3),
            'brain_volume_cm3': float(brain_volume_cm3),
            'voxel_size_mm': list(voxel_size),
            'bbox': bbox,
            'bbox_size': bbox_size
        }

        logger.info(f"Mask statistics:")
        logger.info(f"  Brain volume: {brain_volume_cm3:.2f} cm³")
        logger.info(f"  N voxels: {n_voxels}")
        logger.info(f"  Bounding box: {bbox_size}")

        return stats

    def check_stripping_quality(
        self,
        t1w_data: np.ndarray,
        mask_data: np.ndarray
    ) -> Dict:
        """
        Check for over-stripping or under-stripping.

        Parameters
        ----------
        t1w_data : np.ndarray
            Original T1w data
        mask_data : np.ndarray
            Brain mask

        Returns
        -------
        dict
            Quality assessment
        """
        if t1w_data is None or mask_data is None:
            return {}

        # Calculate intensity statistics inside and outside mask
        brain_intensities = t1w_data[mask_data > 0]
        outside_intensities = t1w_data[mask_data == 0]

        # Remove zero values (background)
        brain_intensities = brain_intensities[brain_intensities > 0]
        outside_intensities = outside_intensities[outside_intensities > 0]

        brain_mean = np.mean(brain_intensities)
        brain_std = np.std(brain_intensities)
        outside_mean = np.mean(outside_intensities) if len(outside_intensities) > 0 else 0
        outside_std = np.std(outside_intensities) if len(outside_intensities) > 0 else 0

        # Contrast between brain and non-brain
        contrast_ratio = brain_mean / outside_mean if outside_mean > 0 else np.inf

        # Assess quality
        quality_flags = []

        if contrast_ratio < 2.0:
            quality_flags.append('LOW_CONTRAST')
            logger.warning("Low contrast between brain and non-brain regions")

        if brain_std / brain_mean > 0.5:
            quality_flags.append('HIGH_VARIANCE')
            logger.warning("High intensity variance within brain mask")

        quality = {
            'brain_mean_intensity': float(brain_mean),
            'brain_std_intensity': float(brain_std),
            'outside_mean_intensity': float(outside_mean),
            'outside_std_intensity': float(outside_std),
            'contrast_ratio': float(contrast_ratio),
            'quality_flags': quality_flags,
            'quality_pass': len(quality_flags) == 0
        }

        logger.info(f"Stripping quality:")
        logger.info(f"  Contrast ratio: {contrast_ratio:.2f}")
        logger.info(f"  Quality flags: {quality_flags if quality_flags else 'PASS'}")

        return quality

    def plot_mask_overlay(
        self,
        t1w_data: np.ndarray,
        mask_data: np.ndarray,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Plot brain mask overlay on T1w image.

        Parameters
        ----------
        t1w_data : np.ndarray
            T1w image data
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

        if t1w_data is None or mask_data is None:
            logger.warning("Cannot create overlay plot without T1w and mask")
            return None

        # Get central slices
        center_x = t1w_data.shape[0] // 2
        center_y = t1w_data.shape[1] // 2
        center_z = t1w_data.shape[2] // 2

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Sagittal slices
        for i, offset in enumerate([-10, 0, 10]):
            ax = axes[0, i]
            slice_idx = center_x + offset
            ax.imshow(t1w_data[slice_idx, :, :].T, cmap='gray', origin='lower')
            ax.contour(mask_data[slice_idx, :, :].T, colors='red', linewidths=1)
            ax.set_title(f'Sagittal (x={slice_idx})', fontsize=10)
            ax.axis('off')

        # Axial slices
        for i, offset in enumerate([-10, 0, 10]):
            ax = axes[1, i]
            slice_idx = center_z + offset
            ax.imshow(t1w_data[:, :, slice_idx].T, cmap='gray', origin='lower')
            ax.contour(mask_data[:, :, slice_idx].T, colors='red', linewidths=1)
            ax.set_title(f'Axial (z={slice_idx})', fontsize=10)
            ax.axis('off')

        plt.suptitle(f'Brain Mask Overlay - {self.subject}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved mask overlay: {output_file}")
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

        logger.info(f"Saved skull strip metrics: {output_file}")
        return output_file

    def run_qc(
        self,
        t1w_file: Optional[Path] = None,
        brain_file: Optional[Path] = None,
        mask_file: Optional[Path] = None
    ) -> Dict:
        """
        Run complete skull stripping QC.

        Parameters
        ----------
        t1w_file : Path, optional
            Original T1w image
        brain_file : Path, optional
            Brain-extracted image
        mask_file : Path, optional
            Brain mask

        Returns
        -------
        dict
            Combined QC results and output paths
        """
        logger.info(f"Running Skull Strip QC for {self.subject}")

        # Load images
        t1w_data, brain_data, mask_data = self.load_images(t1w_file, brain_file, mask_file)

        if mask_data is None:
            logger.warning("No brain mask found, skipping Skull Strip QC")
            return {
                'subject': self.subject,
                'mask_stats': {},
                'quality': {},
                'outputs': {}
            }

        # Calculate mask statistics
        # Assume 1mm isotropic voxels (can be extracted from NIfTI header)
        mask_stats = self.calculate_mask_statistics(mask_data, voxel_size=(1.0, 1.0, 1.0))

        # Check stripping quality
        quality_check = self.check_stripping_quality(t1w_data, mask_data) if t1w_data is not None else {}

        # Plot mask overlay
        overlay_plot = self.plot_mask_overlay(t1w_data, mask_data) if t1w_data is not None else None

        # Save metrics
        metrics_file = self.save_metrics_json(mask_stats, quality_check)

        results = {
            'subject': self.subject,
            'mask_stats': mask_stats,
            'quality': quality_check,
            'outputs': {
                'mask_overlay': str(overlay_plot) if overlay_plot else None,
                'metrics_json': str(metrics_file)
            }
        }

        logger.info("Skull Strip QC completed")
        return results
