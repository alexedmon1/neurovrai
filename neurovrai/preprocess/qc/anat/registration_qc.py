#!/usr/bin/env python3
"""
Registration Quality Control Module.

Evaluates FLIRT/FNIRT registration to MNI152:
- Alignment visual check with edge overlays
- Correlation with template
- Dice coefficient for brain masks
- Registration accuracy metrics
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import ndimage


logger = logging.getLogger(__name__)


class RegistrationQualityControl:
    """
    Quality control for registration to MNI152 (FLIRT/FNIRT).

    Analyzes registration accuracy through visual and quantitative metrics.
    """

    def __init__(self, subject: str, anat_dir: Path, qc_dir: Path):
        """
        Initialize Registration QC.

        Parameters
        ----------
        subject : str
            Subject identifier
        anat_dir : Path
            Directory containing registered outputs
        qc_dir : Path
            QC output directory (e.g., {study_root}/qc/anat/{subject}/registration/)
        """
        self.subject = subject
        self.anat_dir = Path(anat_dir)
        self.qc_dir = Path(qc_dir)
        self.qc_dir.mkdir(parents=True, exist_ok=True)

        # Get FSLDIR for template access
        self.fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')

        logger.info(f"Initialized Registration QC for {subject}")
        logger.info(f"  Anat dir: {self.anat_dir}")
        logger.info(f"  QC dir: {self.qc_dir}")
        logger.info(f"  FSLDIR: {self.fsldir}")

    def load_images(
        self,
        registered_file: Optional[Path] = None,
        template_file: Optional[Path] = None,
        registered_mask: Optional[Path] = None,
        template_mask: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load registered image and MNI152 template.

        Parameters
        ----------
        registered_file : Path, optional
            Registered brain image in MNI152 space
        template_file : Path, optional
            MNI152 template (defaults to FSL standard)
        registered_mask : Path, optional
            Brain mask for registered image
        template_mask : Path, optional
            Brain mask for template

        Returns
        -------
        reg_data : np.ndarray
            Registered image data
        template_data : np.ndarray
            Template image data
        reg_mask : np.ndarray
            Registered brain mask
        template_mask : np.ndarray
            Template brain mask
        """
        # Auto-detect registered file if not provided
        if registered_file is None:
            patterns = [
                '*MNI152*.nii.gz',
                '*mni152*.nii.gz',
                '*2mm.nii.gz',
                '*std*.nii.gz',
                '*warped*.nii.gz'
            ]
            for pattern in patterns:
                candidates = list(self.anat_dir.glob(pattern))
                # Filter out masks
                candidates = [c for c in candidates if 'mask' not in c.name.lower()]
                if candidates:
                    registered_file = candidates[0]
                    break

        # Use FSL MNI152 template if not provided
        if template_file is None:
            # Try 2mm brain template first
            template_file = Path(self.fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm_brain.nii.gz'
            if not template_file.exists():
                # Try 1mm
                template_file = Path(self.fsldir) / 'data' / 'standard' / 'MNI152_T1_1mm_brain.nii.gz'

        # Auto-detect masks
        if registered_mask is None:
            patterns = ['*MNI152*mask*.nii.gz', '*mni152*mask*.nii.gz', '*std*mask*.nii.gz']
            for pattern in patterns:
                candidates = list(self.anat_dir.glob(pattern))
                if candidates:
                    registered_mask = candidates[0]
                    break

        if template_mask is None:
            # Use FSL template mask
            template_mask = Path(self.fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm_brain_mask.nii.gz'
            if not template_mask.exists():
                template_mask = Path(self.fsldir) / 'data' / 'standard' / 'MNI152_T1_1mm_brain_mask.nii.gz'

        # Load images
        reg_data = None
        template_data = None
        reg_mask_data = None
        template_mask_data = None

        if registered_file and registered_file.exists():
            logger.info(f"Loading registered: {registered_file.name}")
            reg_data = nib.load(registered_file).get_fdata()
        else:
            logger.warning(f"Registered file not found: {registered_file}")

        if template_file and template_file.exists():
            logger.info(f"Loading template: {template_file.name}")
            template_data = nib.load(template_file).get_fdata()
        else:
            logger.warning(f"Template file not found: {template_file}")

        if registered_mask and registered_mask.exists():
            logger.info(f"Loading registered mask: {registered_mask.name}")
            reg_mask_data = nib.load(registered_mask).get_fdata() > 0

        if template_mask and template_mask.exists():
            logger.info(f"Loading template mask: {template_mask.name}")
            template_mask_data = nib.load(template_mask).get_fdata() > 0

        return reg_data, template_data, reg_mask_data, template_mask_data

    def calculate_alignment_metrics(
        self,
        reg_data: np.ndarray,
        template_data: np.ndarray,
        reg_mask: Optional[np.ndarray] = None,
        template_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate quantitative alignment metrics.

        Parameters
        ----------
        reg_data : np.ndarray
            Registered image
        template_data : np.ndarray
            Template image
        reg_mask : np.ndarray, optional
            Registered brain mask
        template_mask : np.ndarray, optional
            Template brain mask

        Returns
        -------
        dict
            Alignment metrics
        """
        if reg_data is None or template_data is None:
            return {}

        metrics = {'subject': self.subject}

        # Ensure same shape
        if reg_data.shape != template_data.shape:
            logger.warning(f"Shape mismatch: reg={reg_data.shape}, template={template_data.shape}")
            metrics['shape_mismatch'] = True
            return metrics

        # Create brain mask if not provided
        if reg_mask is None:
            reg_mask = reg_data > 0
        if template_mask is None:
            template_mask = template_data > 0

        # Combined mask (intersection)
        combined_mask = reg_mask & template_mask

        if not np.any(combined_mask):
            logger.warning("No overlapping brain regions found")
            metrics['no_overlap'] = True
            return metrics

        # Extract masked intensities
        reg_intensities = reg_data[combined_mask]
        template_intensities = template_data[combined_mask]

        # Normalize intensities to [0, 1]
        reg_norm = (reg_intensities - np.min(reg_intensities)) / (np.max(reg_intensities) - np.min(reg_intensities) + 1e-10)
        template_norm = (template_intensities - np.min(template_intensities)) / (np.max(template_intensities) - np.min(template_intensities) + 1e-10)

        # Pearson correlation
        correlation = np.corrcoef(reg_norm, template_norm)[0, 1]

        # Normalized cross-correlation
        ncc = np.mean((reg_norm - np.mean(reg_norm)) * (template_norm - np.mean(template_norm))) / \
              (np.std(reg_norm) * np.std(template_norm) + 1e-10)

        # Mean absolute difference
        mad = np.mean(np.abs(reg_norm - template_norm))

        # Root mean square error
        rmse = np.sqrt(np.mean((reg_norm - template_norm) ** 2))

        metrics['correlation'] = float(correlation)
        metrics['ncc'] = float(ncc)
        metrics['mad'] = float(mad)
        metrics['rmse'] = float(rmse)

        logger.info(f"Alignment metrics:")
        logger.info(f"  Correlation: {correlation:.4f}")
        logger.info(f"  NCC: {ncc:.4f}")
        logger.info(f"  MAD: {mad:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")

        # Dice coefficient for masks
        if reg_mask is not None and template_mask is not None:
            intersection = np.sum(reg_mask & template_mask)
            union = np.sum(reg_mask) + np.sum(template_mask)
            dice = 2.0 * intersection / union if union > 0 else 0.0

            metrics['dice_coefficient'] = float(dice)
            logger.info(f"  Dice coefficient: {dice:.4f}")

        # Quality assessment
        quality_flags = []

        if correlation < 0.85:
            quality_flags.append('LOW_CORRELATION')
            logger.warning(f"Low correlation: {correlation:.4f}")

        if 'dice_coefficient' in metrics and metrics['dice_coefficient'] < 0.85:
            quality_flags.append('LOW_DICE')
            logger.warning(f"Low Dice coefficient: {metrics['dice_coefficient']:.4f}")

        if mad > 0.15:
            quality_flags.append('HIGH_MAD')
            logger.warning(f"High MAD: {mad:.4f}")

        metrics['quality_flags'] = quality_flags
        metrics['quality_pass'] = len(quality_flags) == 0

        if metrics['quality_pass']:
            logger.info("Registration quality: PASS")
        else:
            logger.warning(f"Registration quality: FAIL ({quality_flags})")

        return metrics

    def detect_edges(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Detect edges in image using Sobel operator.

        Parameters
        ----------
        image : np.ndarray
            Input image
        sigma : float
            Gaussian smoothing sigma

        Returns
        -------
        np.ndarray
            Edge map
        """
        # Smooth image
        smoothed = ndimage.gaussian_filter(image, sigma=sigma)

        # Sobel edge detection
        sx = ndimage.sobel(smoothed, axis=0)
        sy = ndimage.sobel(smoothed, axis=1)
        sz = ndimage.sobel(smoothed, axis=2)

        # Edge magnitude
        edges = np.sqrt(sx**2 + sy**2 + sz**2)

        return edges

    def plot_registration_overlay(
        self,
        reg_data: np.ndarray,
        template_data: np.ndarray,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Plot registration overlay with edge contours.

        Parameters
        ----------
        reg_data : np.ndarray
            Registered image
        template_data : np.ndarray
            Template image
        output_file : Path, optional
            Output file path

        Returns
        -------
        Path
            Path to saved plot
        """
        if output_file is None:
            output_file = self.qc_dir / 'registration_overlay.png'

        if reg_data is None or template_data is None:
            logger.warning("Cannot create overlay without images")
            return None

        # Detect edges
        logger.info("Detecting edges for overlay...")
        reg_edges = self.detect_edges(reg_data, sigma=1.0)
        template_edges = self.detect_edges(template_data, sigma=1.0)

        # Normalize for display
        reg_norm = (reg_data - np.min(reg_data)) / (np.max(reg_data) - np.min(reg_data) + 1e-10)
        template_norm = (template_data - np.min(template_data)) / (np.max(template_data) - np.min(template_data) + 1e-10)

        # Get central slices
        center_x = reg_data.shape[0] // 2
        center_y = reg_data.shape[1] // 2
        center_z = reg_data.shape[2] // 2

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        # Sagittal slices
        for i, offset in enumerate([-15, 0, 15]):
            ax = axes[0, i]
            slice_idx = center_x + offset

            # Show template as background
            ax.imshow(template_norm[slice_idx, :, :].T, cmap='gray', origin='lower', alpha=0.7)

            # Overlay registered edges in red
            reg_edges_slice = reg_edges[slice_idx, :, :].T
            threshold = np.percentile(reg_edges_slice[reg_edges_slice > 0], 90)
            ax.contour(reg_edges_slice, levels=[threshold], colors='red', linewidths=1.5, alpha=0.8)

            # Overlay template edges in green
            template_edges_slice = template_edges[slice_idx, :, :].T
            threshold = np.percentile(template_edges_slice[template_edges_slice > 0], 90)
            ax.contour(template_edges_slice, levels=[threshold], colors='green', linewidths=1.5, alpha=0.8)

            ax.set_title(f'Sagittal (x={slice_idx})', fontsize=11, fontweight='bold')
            ax.axis('off')

        # Coronal slices
        for i, offset in enumerate([-15, 0, 15]):
            ax = axes[1, i]
            slice_idx = center_y + offset

            ax.imshow(template_norm[:, slice_idx, :].T, cmap='gray', origin='lower', alpha=0.7)

            reg_edges_slice = reg_edges[:, slice_idx, :].T
            threshold = np.percentile(reg_edges_slice[reg_edges_slice > 0], 90)
            ax.contour(reg_edges_slice, levels=[threshold], colors='red', linewidths=1.5, alpha=0.8)

            template_edges_slice = template_edges[:, slice_idx, :].T
            threshold = np.percentile(template_edges_slice[template_edges_slice > 0], 90)
            ax.contour(template_edges_slice, levels=[threshold], colors='green', linewidths=1.5, alpha=0.8)

            ax.set_title(f'Coronal (y={slice_idx})', fontsize=11, fontweight='bold')
            ax.axis('off')

        # Axial slices
        for i, offset in enumerate([-15, 0, 15]):
            ax = axes[2, i]
            slice_idx = center_z + offset

            ax.imshow(template_norm[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.7)

            reg_edges_slice = reg_edges[:, :, slice_idx].T
            threshold = np.percentile(reg_edges_slice[reg_edges_slice > 0], 90)
            ax.contour(reg_edges_slice, levels=[threshold], colors='red', linewidths=1.5, alpha=0.8)

            template_edges_slice = template_edges[:, :, slice_idx].T
            threshold = np.percentile(template_edges_slice[template_edges_slice > 0], 90)
            ax.contour(template_edges_slice, levels=[threshold], colors='green', linewidths=1.5, alpha=0.8)

            ax.set_title(f'Axial (z={slice_idx})', fontsize=11, fontweight='bold')
            ax.axis('off')

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Registered'),
            Line2D([0], [0], color='green', lw=2, label='Template (MNI152)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12, frameon=True)

        plt.suptitle(f'Registration to MNI152 - {self.subject}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved registration overlay: {output_file}")
        return output_file

    def plot_checkerboard(
        self,
        reg_data: np.ndarray,
        template_data: np.ndarray,
        output_file: Optional[Path] = None,
        n_squares: int = 8
    ) -> Path:
        """
        Plot checkerboard overlay for registration assessment.

        Parameters
        ----------
        reg_data : np.ndarray
            Registered image
        template_data : np.ndarray
            Template image
        output_file : Path, optional
            Output file path
        n_squares : int
            Number of checkerboard squares per dimension

        Returns
        -------
        Path
            Path to saved plot
        """
        if output_file is None:
            output_file = self.qc_dir / 'registration_checkerboard.png'

        if reg_data is None or template_data is None:
            logger.warning("Cannot create checkerboard without images")
            return None

        # Normalize
        reg_norm = (reg_data - np.min(reg_data)) / (np.max(reg_data) - np.min(reg_data) + 1e-10)
        template_norm = (template_data - np.min(template_data)) / (np.max(template_data) - np.min(template_data) + 1e-10)

        # Create checkerboard mask
        shape = reg_data.shape
        mask = np.zeros(shape, dtype=bool)

        square_size = [s // n_squares for s in shape]

        for i in range(n_squares):
            for j in range(n_squares):
                for k in range(n_squares):
                    if (i + j + k) % 2 == 0:
                        x_start = i * square_size[0]
                        x_end = min((i + 1) * square_size[0], shape[0])
                        y_start = j * square_size[1]
                        y_end = min((j + 1) * square_size[1], shape[1])
                        z_start = k * square_size[2]
                        z_end = min((k + 1) * square_size[2], shape[2])

                        mask[x_start:x_end, y_start:y_end, z_start:z_end] = True

        # Create checkerboard composite
        checkerboard = np.where(mask, reg_norm, template_norm)

        # Get central slices
        center_x = shape[0] // 2
        center_y = shape[1] // 2
        center_z = shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Sagittal
        axes[0].imshow(checkerboard[center_x, :, :].T, cmap='gray', origin='lower')
        axes[0].set_title(f'Sagittal (x={center_x})', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Coronal
        axes[1].imshow(checkerboard[:, center_y, :].T, cmap='gray', origin='lower')
        axes[1].set_title(f'Coronal (y={center_y})', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Axial
        axes[2].imshow(checkerboard[:, :, center_z].T, cmap='gray', origin='lower')
        axes[2].set_title(f'Axial (z={center_z})', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle(f'Checkerboard Overlay - {self.subject}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved checkerboard: {output_file}")
        return output_file

    def save_metrics_json(
        self,
        metrics: Dict,
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Save registration metrics to JSON.

        Parameters
        ----------
        metrics : dict
            Registration metrics
        output_file : Path, optional
            Output JSON file

        Returns
        -------
        Path
            Path to saved JSON
        """
        if output_file is None:
            output_file = self.qc_dir.parent / 'metrics' / 'registration.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved registration metrics: {output_file}")
        return output_file

    def run_qc(
        self,
        registered_file: Optional[Path] = None,
        template_file: Optional[Path] = None,
        registered_mask: Optional[Path] = None,
        template_mask: Optional[Path] = None
    ) -> Dict:
        """
        Run complete registration QC.

        Parameters
        ----------
        registered_file : Path, optional
            Registered brain image in MNI152 space
        template_file : Path, optional
            MNI152 template
        registered_mask : Path, optional
            Registered brain mask
        template_mask : Path, optional
            Template brain mask

        Returns
        -------
        dict
            Combined QC results and output paths
        """
        logger.info(f"Running Registration QC for {self.subject}")

        # Load images
        reg_data, template_data, reg_mask, template_mask = self.load_images(
            registered_file, template_file, registered_mask, template_mask
        )

        if reg_data is None or template_data is None:
            logger.warning("Missing registered or template image, skipping Registration QC")
            return {
                'subject': self.subject,
                'metrics': {},
                'outputs': {}
            }

        # Calculate alignment metrics
        metrics = self.calculate_alignment_metrics(reg_data, template_data, reg_mask, template_mask)

        # Plot registration overlay
        overlay_plot = self.plot_registration_overlay(reg_data, template_data)

        # Plot checkerboard
        checkerboard_plot = self.plot_checkerboard(reg_data, template_data)

        # Save metrics
        metrics_file = self.save_metrics_json(metrics)

        results = {
            'subject': self.subject,
            'metrics': metrics,
            'outputs': {
                'registration_overlay': str(overlay_plot) if overlay_plot else None,
                'checkerboard': str(checkerboard_plot) if checkerboard_plot else None,
                'metrics_json': str(metrics_file)
            }
        }

        logger.info("Registration QC completed")
        return results
