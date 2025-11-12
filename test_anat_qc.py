#!/usr/bin/env python3
"""
Test script for Anatomical QC module (Skull Strip).

Tests the skull stripping QC framework with synthetic data.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import nibabel as nib

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.qc.anat.skull_strip_qc import SkullStripQualityControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_t1w_and_mask(output_dir: Path):
    """
    Create synthetic T1w image and brain mask for testing.

    Returns
    -------
    t1w_file : Path
        Path to synthetic T1w image
    mask_file : Path
        Path to synthetic brain mask
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic 3D T1w image (64x64x64)
    shape = (64, 64, 64)

    # Create T1w with brain tissue
    t1w_data = np.zeros(shape)

    # Add brain sphere (center region)
    center = np.array(shape) // 2
    radius = 20

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist < radius:
                    # Brain tissue intensity ~1000
                    t1w_data[x, y, z] = 1000 + np.random.randn() * 100
                elif dist < radius + 5:
                    # Skull intensity ~500
                    t1w_data[x, y, z] = 500 + np.random.randn() * 50
                else:
                    # Background noise
                    t1w_data[x, y, z] = np.random.randn() * 10

    # Create brain mask (slightly smaller than actual brain to simulate BET)
    mask_data = np.zeros(shape, dtype=np.uint8)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if dist < radius - 2:
                    mask_data[x, y, z] = 1

    # Save as NIfTI
    affine = np.eye(4)

    t1w_img = nib.Nifti1Image(t1w_data, affine)
    t1w_file = output_dir / 'T1w.nii.gz'
    nib.save(t1w_img, t1w_file)

    mask_img = nib.Nifti1Image(mask_data, affine)
    mask_file = output_dir / 'T1w_brain_mask.nii.gz'
    nib.save(mask_img, mask_file)

    brain_data = t1w_data * mask_data
    brain_img = nib.Nifti1Image(brain_data, affine)
    brain_file = output_dir / 'T1w_brain.nii.gz'
    nib.save(brain_img, brain_file)

    logger.info(f"Created synthetic T1w: {t1w_file}")
    logger.info(f"Created synthetic mask: {mask_file}")
    logger.info(f"Created synthetic brain: {brain_file}")

    return t1w_file, brain_file, mask_file


def main():
    """Test Skull Strip QC with synthetic data."""

    subject = 'TEST-SYNTHETIC-ANAT'

    # Create temporary directory for test
    test_dir = Path('/tmp/anat_qc_test')
    anat_dir = test_dir / 'anat'
    qc_dir = test_dir / 'qc'

    logger.info("="*70)
    logger.info("Testing Skull Strip QC Module (Synthetic Data)")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Anat directory: {anat_dir}")
    logger.info(f"QC output: {qc_dir}")
    logger.info("")

    # Create synthetic data
    logger.info("Creating synthetic T1w and brain mask...")
    t1w_file, brain_file, mask_file = create_synthetic_t1w_and_mask(anat_dir)

    # Create Skull Strip QC instance
    logger.info("")
    logger.info("="*70)
    logger.info("Initializing Skull Strip QC")
    logger.info("="*70)

    skull_qc = SkullStripQualityControl(
        subject=subject,
        anat_dir=anat_dir,
        qc_dir=qc_dir / 'skull_strip'
    )

    # Run QC
    logger.info("")
    logger.info("="*70)
    logger.info("Running Skull Strip QC Analysis")
    logger.info("="*70)

    try:
        results = skull_qc.run_qc(
            t1w_file=t1w_file,
            brain_file=brain_file,
            mask_file=mask_file
        )

        logger.info("")
        logger.info("="*70)
        logger.info("Skull Strip QC Results")
        logger.info("="*70)

        # Display mask statistics
        mask_stats = results['mask_stats']
        logger.info("Mask Statistics:")
        logger.info(f"  N voxels: {mask_stats['n_voxels']}")
        logger.info(f"  Brain volume: {mask_stats['brain_volume_cm3']:.2f} cm³")
        logger.info(f"  Bounding box size: {mask_stats['bbox_size']}")

        # Display quality check
        if 'quality' in results and results['quality']:
            quality = results['quality']
            logger.info("")
            logger.info("Quality Assessment:")
            logger.info(f"  Brain mean intensity: {quality['brain_mean_intensity']:.2f}")
            logger.info(f"  Outside mean intensity: {quality['outside_mean_intensity']:.2f}")
            logger.info(f"  Contrast ratio: {quality['contrast_ratio']:.2f}")
            logger.info(f"  Quality pass: {quality['quality_pass']}")
            if quality['quality_flags']:
                logger.info(f"  Flags: {quality['quality_flags']}")

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
            logger.info("To view the mask overlay:")
            logger.info(f"  open {results['outputs']['mask_overlay']}")
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
