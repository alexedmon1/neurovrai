#!/usr/bin/env python3
"""
Test script for Segmentation QC module.

Tests tissue segmentation QC with synthetic data.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import nibabel as nib

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.qc.anat.segmentation_qc import SegmentationQualityControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_tissue_maps(output_dir: Path):
    """
    Create synthetic tissue probability maps for testing.

    Simulates realistic GM/WM/CSF distributions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic 3D volume (64x64x64)
    shape = (64, 64, 64)

    # Initialize probability maps
    csf_map = np.zeros(shape)
    gm_map = np.zeros(shape)
    wm_map = np.zeros(shape)

    center = np.array(shape) // 2

    # Create realistic tissue distribution
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

                # CSF in ventricles and around edges
                if dist < 8:  # Central ventricles
                    csf_map[x, y, z] = 0.8 + np.random.rand() * 0.2
                elif dist > 22:  # Peripheral CSF
                    csf_map[x, y, z] = 0.6 + np.random.rand() * 0.3

                # GM in cortex
                elif 18 < dist <= 22:
                    gm_map[x, y, z] = 0.7 + np.random.rand() * 0.3

                # WM in center
                elif 8 <= dist <= 18:
                    wm_map[x, y, z] = 0.75 + np.random.rand() * 0.25

    # Normalize to ensure probabilities sum to ~1.0
    total = csf_map + gm_map + wm_map
    mask = total > 0
    csf_map[mask] /= total[mask]
    gm_map[mask] /= total[mask]
    wm_map[mask] /= total[mask]

    # Save as NIfTI
    affine = np.eye(4)

    csf_img = nib.Nifti1Image(csf_map, affine)
    csf_file = output_dir / 'T1w_pve_0.nii.gz'  # FAST naming convention
    nib.save(csf_img, csf_file)

    gm_img = nib.Nifti1Image(gm_map, affine)
    gm_file = output_dir / 'T1w_pve_1.nii.gz'
    nib.save(gm_img, gm_file)

    wm_img = nib.Nifti1Image(wm_map, affine)
    wm_file = output_dir / 'T1w_pve_2.nii.gz'
    nib.save(wm_img, wm_file)

    logger.info(f"Created synthetic tissue maps:")
    logger.info(f"  CSF: {csf_file}")
    logger.info(f"  GM:  {gm_file}")
    logger.info(f"  WM:  {wm_file}")

    return csf_file, gm_file, wm_file


def main():
    """Test Segmentation QC with synthetic data."""

    subject = 'TEST-SYNTHETIC-SEG'

    # Create temporary directory for test
    test_dir = Path('/tmp/segmentation_qc_test')
    anat_dir = test_dir / 'anat'
    qc_dir = test_dir / 'qc'

    logger.info("="*70)
    logger.info("Testing Segmentation QC Module (Synthetic Data)")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Anat directory: {anat_dir}")
    logger.info(f"QC output: {qc_dir}")
    logger.info("")

    # Create synthetic data
    logger.info("Creating synthetic tissue segmentation maps...")
    csf_file, gm_file, wm_file = create_synthetic_tissue_maps(anat_dir)

    # Create Segmentation QC instance
    logger.info("")
    logger.info("="*70)
    logger.info("Initializing Segmentation QC")
    logger.info("="*70)

    seg_qc = SegmentationQualityControl(
        subject=subject,
        anat_dir=anat_dir,
        qc_dir=qc_dir / 'segmentation'
    )

    # Run QC
    logger.info("")
    logger.info("="*70)
    logger.info("Running Segmentation QC Analysis")
    logger.info("="*70)

    try:
        results = seg_qc.run_qc(
            csf_file=csf_file,
            gm_file=gm_file,
            wm_file=wm_file,
            threshold=0.5
        )

        logger.info("")
        logger.info("="*70)
        logger.info("Segmentation QC Results")
        logger.info("="*70)

        # Display tissue volumes
        volumes = results['volumes']
        logger.info("Tissue Volumes:")
        logger.info(f"  CSF: {volumes['csf']['volume_cm3']:.2f} cm³ ({volumes['csf']['fraction']*100:.1f}%)")
        logger.info(f"  GM:  {volumes['gm']['volume_cm3']:.2f} cm³ ({volumes['gm']['fraction']*100:.1f}%)")
        logger.info(f"  WM:  {volumes['wm']['volume_cm3']:.2f} cm³ ({volumes['wm']['fraction']*100:.1f}%)")
        logger.info(f"  Total: {volumes['total_volume_cm3']:.2f} cm³")

        # Display validation
        validation = results['validation']
        logger.info("")
        logger.info("Quality Validation:")
        logger.info(f"  GM/WM ratio: {validation['gm_wm_ratio']:.2f}")
        logger.info(f"  Quality pass: {validation['quality_pass']}")
        if validation['quality_flags']:
            logger.info(f"  Flags: {validation['quality_flags']}")
        else:
            logger.info(f"  Flags: None (all within expected ranges)")

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
            logger.info("To view the tissue volume plot:")
            logger.info(f"  open {results['outputs']['volume_plot']}")
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
