#!/usr/bin/env python3
"""
Test script for TOPUP QC module.

Tests the TOPUP quality control framework on IRC805-0580101 data.
"""

import sys
from pathlib import Path
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.qc.dwi.topup_qc import TOPUPQualityControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test TOPUP QC on IRC805-0580101."""

    subject = 'IRC805-0580101'

    # Paths
    study_root = Path('/mnt/bytopia/development/IRC805')
    work_dir = study_root / 'work' / subject / 'dwi_topup'
    qc_root = study_root / 'qc' / 'dwi' / subject

    logger.info("="*70)
    logger.info("Testing TOPUP QC Module")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Work directory: {work_dir}")
    logger.info(f"QC output: {qc_root}")
    logger.info("")

    # Check if work directory exists
    if not work_dir.exists():
        logger.error(f"Work directory not found: {work_dir}")
        logger.error("Please run test_dwi_topup.py first to generate TOPUP outputs")
        return 1

    # Find TOPUP outputs
    logger.info("Searching for TOPUP outputs...")
    fieldcoef_files = list(work_dir.glob('*fieldcoef*.nii.gz'))

    if not fieldcoef_files:
        logger.error("No TOPUP fieldcoef file found")
        logger.error("Expected: topup_results_fieldcoef.nii.gz")
        return 1

    fieldcoef_file = fieldcoef_files[0]
    logger.info(f"Found field coefficient: {fieldcoef_file.name}")

    # Create TOPUP QC instance
    logger.info("")
    logger.info("="*70)
    logger.info("Initializing TOPUP QC")
    logger.info("="*70)

    topup_qc = TOPUPQualityControl(
        subject=subject,
        work_dir=work_dir,
        qc_dir=qc_root / 'topup'
    )

    # Run QC
    logger.info("")
    logger.info("="*70)
    logger.info("Running TOPUP QC Analysis")
    logger.info("="*70)

    try:
        results = topup_qc.run_qc(
            topup_log=None,  # Will auto-detect
            fieldcoef_file=fieldcoef_file
        )

        logger.info("")
        logger.info("="*70)
        logger.info("TOPUP QC Results")
        logger.info("="*70)

        # Display convergence metrics
        conv_metrics = results['convergence_metrics']
        logger.info("Convergence:")
        logger.info(f"  Converged: {conv_metrics['converged']}")
        logger.info(f"  Iterations: {conv_metrics['iterations']}")
        logger.info(f"  Initial SSD: {conv_metrics['initial_ssd']}")
        logger.info(f"  Final SSD: {conv_metrics['final_ssd']}")
        if conv_metrics['improvement_percent'] is not None:
            logger.info(f"  Improvement: {conv_metrics['improvement_percent']:.1f}%")
        else:
            logger.info(f"  Improvement: N/A (no convergence data)")
        logger.info(f"  Rate: {conv_metrics['convergence_rate']}")

        # Display field metrics
        if 'field_statistics' in results['field_metrics']:
            logger.info("")
            logger.info("Field Map Statistics:")
            field_stats = results['field_metrics']['field_statistics']
            logger.info(f"  Mean field: {field_stats['mean_field_hz']:.2f} Hz")
            logger.info(f"  Max field: {field_stats['max_field_hz']:.2f} Hz")
            logger.info(f"  Std field: {field_stats['std_field_hz']:.2f} Hz")

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
