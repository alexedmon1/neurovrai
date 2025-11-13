#!/usr/bin/env python3
"""
Test script for Registration QC module.

Tests registration QC with synthetic registered brain data.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import nibabel as nib
from scipy import ndimage

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.qc.anat.registration_qc import RegistrationQualityControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_brain(shape=(91, 109, 91), offset=(0, 0, 0), rotation_deg=0):
    """
    Create synthetic brain image.

    Parameters
    ----------
    shape : tuple
        Image dimensions (MNI152 2mm is 91x109x91)
    offset : tuple
        Translation offset in voxels
    rotation_deg : float
        Rotation angle in degrees (around z-axis)

    Returns
    -------
    np.ndarray
        Synthetic brain image
    """
    image = np.zeros(shape)

    # Define brain center with offset
    center = np.array(shape) // 2 + np.array(offset)

    # Create ellipsoid brain (realistic proportions)
    radii = np.array([30, 35, 30])  # x, y, z radii

    # Rotation matrix (around z-axis)
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # Apply rotation
                dx = x - center[0]
                dy = y - center[1]
                dz = z - center[2]

                # Rotate in xy-plane
                dx_rot = cos_t * dx - sin_t * dy
                dy_rot = sin_t * dx + cos_t * dy

                # Ellipsoid distance
                dist_sq = (dx_rot / radii[0])**2 + (dy_rot / radii[1])**2 + (dz / radii[2])**2

                if dist_sq <= 1.0:
                    # Brain tissue with structure
                    tissue_intensity = 1000

                    # Add WM core (brighter)
                    if dist_sq < 0.4:
                        tissue_intensity = 1200

                    # Add GM cortex (intermediate)
                    elif dist_sq > 0.7:
                        tissue_intensity = 900

                    # Add noise and texture
                    noise = np.random.randn() * 50
                    image[x, y, z] = tissue_intensity + noise

    # Smooth slightly
    image = ndimage.gaussian_filter(image, sigma=0.8)

    return image


def create_synthetic_mask(brain_data):
    """
    Create brain mask from brain image.

    Parameters
    ----------
    brain_data : np.ndarray
        Brain image

    Returns
    -------
    np.ndarray
        Binary brain mask
    """
    mask = brain_data > 100

    # Clean up mask
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask, iterations=1)
    mask = ndimage.binary_dilation(mask, iterations=1)

    return mask.astype(np.uint8)


def main():
    """Test Registration QC with synthetic data."""

    subject = 'TEST-SYNTHETIC-REG'

    # Create temporary directory for test
    test_dir = Path('/tmp/registration_qc_test')
    anat_dir = test_dir / 'anat'
    qc_dir = test_dir / 'qc'
    anat_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("Testing Registration QC Module (Synthetic Data)")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Anat directory: {anat_dir}")
    logger.info(f"QC output: {qc_dir}")
    logger.info("")

    # Create synthetic MNI152 template (perfect registration target)
    logger.info("Creating synthetic MNI152 template...")
    shape = (91, 109, 91)  # MNI152 2mm dimensions
    template_data = create_synthetic_brain(shape=shape, offset=(0, 0, 0), rotation_deg=0)
    template_mask = create_synthetic_mask(template_data)

    # Create synthetic registered brain with slight misalignment
    logger.info("Creating synthetic registered brain (with slight misalignment)...")
    # Add small offset (2 voxels) and rotation (5 degrees) to simulate imperfect registration
    registered_data = create_synthetic_brain(shape=shape, offset=(2, 1, -1), rotation_deg=5)
    registered_mask = create_synthetic_mask(registered_data)

    # Save as NIfTI with MNI152 affine
    logger.info("Saving NIfTI files...")

    # MNI152 2mm affine
    affine = np.array([
        [-2.,   0.,   0.,  90.],
        [ 0.,   2.,   0., -126.],
        [ 0.,   0.,   2.,  -72.],
        [ 0.,   0.,   0.,   1.]
    ])

    # Save template
    template_img = nib.Nifti1Image(template_data, affine)
    template_file = anat_dir / 'MNI152_T1_2mm_brain_synthetic.nii.gz'
    nib.save(template_img, template_file)

    template_mask_img = nib.Nifti1Image(template_mask, affine)
    template_mask_file = anat_dir / 'MNI152_T1_2mm_brain_mask_synthetic.nii.gz'
    nib.save(template_mask_img, template_mask_file)

    # Save registered brain
    registered_img = nib.Nifti1Image(registered_data, affine)
    registered_file = anat_dir / 'T1w_MNI152_2mm.nii.gz'
    nib.save(registered_img, registered_file)

    registered_mask_img = nib.Nifti1Image(registered_mask, affine)
    registered_mask_file = anat_dir / 'T1w_MNI152_2mm_mask.nii.gz'
    nib.save(registered_mask_img, registered_mask_file)

    logger.info(f"  Template: {template_file.name}")
    logger.info(f"  Template mask: {template_mask_file.name}")
    logger.info(f"  Registered: {registered_file.name}")
    logger.info(f"  Registered mask: {registered_mask_file.name}")

    # Create Registration QC instance
    logger.info("")
    logger.info("="*70)
    logger.info("Initializing Registration QC")
    logger.info("="*70)

    reg_qc = RegistrationQualityControl(
        subject=subject,
        anat_dir=anat_dir,
        qc_dir=qc_dir / 'registration'
    )

    # Run QC
    logger.info("")
    logger.info("="*70)
    logger.info("Running Registration QC Analysis")
    logger.info("="*70)

    try:
        results = reg_qc.run_qc(
            registered_file=registered_file,
            template_file=template_file,
            registered_mask=registered_mask_file,
            template_mask=template_mask_file
        )

        logger.info("")
        logger.info("="*70)
        logger.info("Registration QC Results")
        logger.info("="*70)

        # Display alignment metrics
        metrics = results['metrics']
        if metrics:
            logger.info("Alignment Metrics:")
            if 'correlation' in metrics:
                logger.info(f"  Pearson correlation: {metrics['correlation']:.4f}")
            if 'ncc' in metrics:
                logger.info(f"  Normalized CC: {metrics['ncc']:.4f}")
            if 'dice_coefficient' in metrics:
                logger.info(f"  Dice coefficient: {metrics['dice_coefficient']:.4f}")
            if 'mad' in metrics:
                logger.info(f"  Mean absolute diff: {metrics['mad']:.4f}")
            if 'rmse' in metrics:
                logger.info(f"  RMSE: {metrics['rmse']:.4f}")

            logger.info("")
            logger.info("Quality Assessment:")
            logger.info(f"  Quality pass: {metrics.get('quality_pass', False)}")
            if metrics.get('quality_flags'):
                logger.info(f"  Flags: {metrics['quality_flags']}")
            else:
                logger.info(f"  Flags: None (registration quality is good)")

        # Display output files
        logger.info("")
        logger.info("Output Files:")
        outputs = results['outputs']
        for key, path in outputs.items():
            if path:
                logger.info(f"  {key}: {path}")

        # Verify outputs exist
        logger.info("")
        logger.info("Verifying outputs...")
        all_exist = True
        for key, path in outputs.items():
            if path:
                path_obj = Path(path)
                exists = path_obj.exists()
                status = "✓" if exists else "✗"
                logger.info(f"  {status} {path_obj.name}")
                if not exists:
                    all_exist = False

        if all_exist:
            logger.info("")
            logger.info("="*70)
            logger.info("SUCCESS! All QC outputs generated")
            logger.info("="*70)
            logger.info(f"QC directory: {qc_dir}")
            logger.info("")
            logger.info("To view the registration overlay:")
            logger.info(f"  open {results['outputs']['registration_overlay']}")
            logger.info("")
            logger.info("To view the checkerboard:")
            logger.info(f"  open {results['outputs']['checkerboard']}")
            logger.info("")
            logger.info("Expected behavior:")
            logger.info("  - Registration should show slight misalignment (intentional)")
            logger.info("  - Red edges (registered) should be offset from green edges (template)")
            logger.info("  - Correlation should be good but not perfect (~0.85-0.95)")
            return 0
        else:
            logger.error("")
            logger.error("Some outputs were not generated")
            return 1

    except Exception as e:
        logger.error(f"ERROR during QC: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
