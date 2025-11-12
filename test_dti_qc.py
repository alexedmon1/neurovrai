#!/usr/bin/env python3
"""
Test script for DTI QC module.

Tests the DTI QC framework on IRC805-0580101 data.
"""

import sys
from pathlib import Path
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.qc.dwi.dti_qc import DTIQualityControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test DTI QC on IRC805-0580101."""

    subject = 'IRC805-0580101'

    # Paths
    study_root = Path('/mnt/bytopia/development/IRC805')
    dti_dir = study_root / 'derivatives' / 'dwi_topup' / subject / 'dti'
    qc_root = study_root / 'qc' / 'dwi' / subject

    logger.info("="*70)
    logger.info("Testing DTI QC Module")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"DTI directory: {dti_dir}")
    logger.info(f"QC output: {qc_root}")
    logger.info("")

    # Check if DTI directory exists
    if not dti_dir.exists():
        logger.error(f"DTI directory not found: {dti_dir}")
        logger.error("Please run test_dwi_topup.py first to generate DTI outputs")
        return 1

    # Check for DTI outputs
    logger.info("Searching for DTI outputs...")
    fa_files = list(dti_dir.glob('*FA*.nii.gz'))
    md_files = list(dti_dir.glob('*MD*.nii.gz'))

    if not fa_files:
        logger.error("No FA map found")
        return 1

    if not md_files:
        logger.error("No MD map found")
        return 1

    logger.info(f"Found FA map: {fa_files[0].name}")
    logger.info(f"Found MD map: {md_files[0].name}")

    # Look for brain mask
    mask_dir = study_root / 'derivatives' / 'dwi_topup' / subject / 'mask'
    mask_files = list(mask_dir.glob('*brain_mask*.nii.gz')) if mask_dir.exists() else []
    mask_file = mask_files[0] if mask_files else None

    if mask_file:
        logger.info(f"Found brain mask: {mask_file.name}")

    # Create DTI QC instance
    logger.info("")
    logger.info("="*70)
    logger.info("Initializing DTI QC")
    logger.info("="*70)

    dti_qc = DTIQualityControl(
        subject=subject,
        dti_dir=dti_dir,
        qc_dir=qc_root / 'dti'
    )

    # Run QC
    logger.info("")
    logger.info("="*70)
    logger.info("Running DTI QC Analysis")
    logger.info("="*70)

    try:
        results = dti_qc.run_qc(
            metrics=['FA', 'MD'],
            mask_file=mask_file
        )

        logger.info("")
        logger.info("="*70)
        logger.info("DTI QC Results")
        logger.info("="*70)

        # Display DTI statistics
        dti_stats = results['dti_stats']
        for metric, stats in dti_stats['metrics'].items():
            logger.info(f"\n{metric} Statistics:")
            logger.info(f"  N voxels: {stats['n_voxels']}")
            logger.info(f"  Mean: {stats['mean']:.4f}")
            logger.info(f"  Median: {stats['median']:.4f}")
            logger.info(f"  Std: {stats['std']:.4f}")
            logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            logger.info(f"  25th percentile: {stats['percentile_25']:.4f}")
            logger.info(f"  75th percentile: {stats['percentile_75']:.4f}")
            logger.info(f"  95th percentile: {stats['percentile_95']:.4f}")

            if 'outliers' in stats:
                outliers = stats['outliers']
                logger.info(f"  Outliers: {outliers['percent_outliers']:.2f}%")
                logger.info(f"    Below min: {outliers['n_below_min']}")
                logger.info(f"    Above max: {outliers['n_above_max']}")

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
            logger.info(f"QC directory: {qc_root}")
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
