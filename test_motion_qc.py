#!/usr/bin/env python3
"""
Test script for Motion QC module.

Tests the Motion QC framework on IRC805-0580101 data.
"""

import sys
from pathlib import Path
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.qc.dwi.motion_qc import MotionQualityControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test Motion QC on IRC805-0580101."""

    subject = 'IRC805-0580101'

    # Paths
    study_root = Path('/mnt/bytopia/development/IRC805')
    work_dir = study_root / 'work' / subject / 'dwi_topup'
    qc_root = study_root / 'qc' / 'dwi' / subject

    logger.info("="*70)
    logger.info("Testing Motion QC Module")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Work directory: {work_dir}")
    logger.info(f"QC output: {qc_root}")
    logger.info("")

    # Check if work directory exists
    if not work_dir.exists():
        logger.error(f"Work directory not found: {work_dir}")
        logger.error("Please run test_dwi_topup.py first to generate eddy outputs")
        return 1

    # Find eddy outputs
    logger.info("Searching for eddy outputs...")
    eddy_params_files = list(work_dir.glob('*eddy_corrected.eddy_parameters'))

    if not eddy_params_files:
        logger.error("No eddy parameters file found")
        logger.error("Expected: eddy_corrected.eddy_parameters")
        return 1

    eddy_params_file = eddy_params_files[0]
    logger.info(f"Found eddy parameters: {eddy_params_file.name}")

    # Create Motion QC instance
    logger.info("")
    logger.info("="*70)
    logger.info("Initializing Motion QC")
    logger.info("="*70)

    motion_qc = MotionQualityControl(
        subject=subject,
        work_dir=work_dir,
        qc_dir=qc_root / 'motion'
    )

    # Run QC
    logger.info("")
    logger.info("="*70)
    logger.info("Running Motion QC Analysis")
    logger.info("="*70)

    try:
        results = motion_qc.run_qc(
            eddy_params_file=eddy_params_file,
            fd_threshold=1.0
        )

        logger.info("")
        logger.info("="*70)
        logger.info("Motion QC Results")
        logger.info("="*70)

        # Display motion statistics
        motion_stats = results['motion_stats']
        logger.info("Motion Statistics:")
        logger.info(f"  Volumes: {motion_stats['n_volumes']}")
        logger.info(f"  Mean FD: {motion_stats['fd_mean']:.3f} mm")
        logger.info(f"  Max FD: {motion_stats['fd_max']:.3f} mm")
        logger.info(f"  Median FD: {motion_stats['fd_median']:.3f} mm")
        logger.info(f"  Translation RMS: {motion_stats['translation_rms']:.3f} mm")
        logger.info(f"  Rotation RMS: {motion_stats['rotation_rms']:.4f} rad")
        logger.info(f"  Max translation: {motion_stats['max_translation']:.3f} mm")
        logger.info(f"  Max rotation: {motion_stats['max_rotation']:.4f} rad")

        # Display outlier information
        logger.info("")
        logger.info("Outlier Detection:")
        outliers = results['outliers']
        logger.info(f"  Threshold: {outliers['threshold_mm']} mm")
        logger.info(f"  Outliers: {outliers['n_outliers']} / {motion_stats['n_volumes']}")
        logger.info(f"  Percentage: {outliers['percent_outliers']:.1f}%")

        if outliers['n_outliers'] > 0:
            logger.info(f"  Outlier volumes: {outliers['outlier_indices'][:10]}...")  # Show first 10

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
