#!/usr/bin/env python3
"""
Synthetic test for Motion QC module.

Creates synthetic motion parameters to test the Motion QC framework.
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.qc.dwi.motion_qc import MotionQualityControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_motion_params(n_volumes=60, output_file=None):
    """
    Create synthetic motion parameters for testing.

    Simulates realistic motion with some outliers.
    """
    np.random.seed(42)

    # Generate baseline motion (small random walk)
    trans = np.cumsum(np.random.randn(n_volumes, 3) * 0.1, axis=0)
    rot = np.cumsum(np.random.randn(n_volumes, 3) * 0.01, axis=0)

    # Add some outlier volumes (sudden movements)
    outlier_volumes = [15, 32, 48]
    for vol in outlier_volumes:
        if vol < n_volumes:
            trans[vol, :] += np.random.randn(3) * 2.0  # Large translation
            rot[vol, :] += np.random.randn(3) * 0.05  # Large rotation

    # Combine
    params = np.hstack([trans, rot])

    if output_file:
        np.savetxt(output_file, params, fmt='%.6f')
        logger.info(f"Created synthetic motion params: {output_file}")
        logger.info(f"  Volumes: {n_volumes}")
        logger.info(f"  Outliers: {len(outlier_volumes)} at volumes {outlier_volumes}")

    return params


def main():
    """Test Motion QC with synthetic data."""

    subject = 'TEST-SYNTHETIC'

    # Create temporary directory for test
    test_dir = Path('/tmp/motion_qc_test')
    test_dir.mkdir(parents=True, exist_ok=True)

    qc_dir = test_dir / 'qc'

    logger.info("="*70)
    logger.info("Testing Motion QC Module (Synthetic Data)")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Test directory: {test_dir}")
    logger.info(f"QC output: {qc_dir}")
    logger.info("")

    # Create synthetic motion parameters
    logger.info("Creating synthetic motion parameters...")
    synthetic_params_file = test_dir / 'eddy_corrected.eddy_parameters'
    create_synthetic_motion_params(n_volumes=60, output_file=synthetic_params_file)

    # Create Motion QC instance
    logger.info("")
    logger.info("="*70)
    logger.info("Initializing Motion QC")
    logger.info("="*70)

    motion_qc = MotionQualityControl(
        subject=subject,
        work_dir=test_dir,
        qc_dir=qc_dir / 'motion'
    )

    # Run QC
    logger.info("")
    logger.info("="*70)
    logger.info("Running Motion QC Analysis")
    logger.info("="*70)

    try:
        results = motion_qc.run_qc(
            eddy_params_file=synthetic_params_file,
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
            logger.info(f"  Outlier volumes: {outliers['outlier_indices']}")

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
            logger.info("To view the motion plot:")
            logger.info(f"  open {results['outputs']['motion_plot']}")
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
