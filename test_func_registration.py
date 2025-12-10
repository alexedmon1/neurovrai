#!/usr/bin/env python3
"""
Test script for new ANTs-based functional registration pipeline.

This tests the refactored func_preprocess.py workflow that:
1. Performs motion correction BEFORE any filtering
2. Computes ANTs-based fMRI → T1w registration on raw motion-corrected data
3. Concatenates with T1w → MNI transforms
4. Creates inverse MNI → fMRI transform for atlas transformation
"""

import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from neurovrai.preprocess.workflows.func_preprocess import run_func_preprocessing
from neurovrai.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test functional registration pipeline on IRC805-0580101."""

    # Subject and paths
    subject = 'IRC805-0580101'
    study_root = Path('/mnt/bytopia/IRC805')

    # Configuration - use study config
    config_file = study_root / 'config.yaml'
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        logger.error("Please ensure config.yaml exists in study root")
        return 1

    config = load_config(config_file)
    logger.info(f"Loaded config from: {config_file}")

    # Multi-echo functional data (use resting-state scan, not calibration)
    func_dir = study_root / 'bids' / subject / 'func'
    echo_files = sorted(func_dir.glob('501_*RESTING*_e?.nii.gz'))

    if not echo_files:
        logger.error(f"No functional data found in {func_dir}")
        return 1

    logger.info(f"Found {len(echo_files)} echo files:")
    for ef in echo_files:
        logger.info(f"  {ef.name}")

    # Anatomical derivatives (for tissue masks and transforms)
    anat_derivatives = study_root / 'derivatives' / subject / 'anat'

    if not anat_derivatives.exists():
        logger.error(f"Anatomical derivatives not found: {anat_derivatives}")
        logger.error("Please run anatomical preprocessing first")
        return 1

    # Output directories
    output_dir = study_root
    work_dir = study_root / 'work' / subject / 'func_preproc_test'

    # Clean work directory for fresh test
    if work_dir.exists():
        logger.info(f"Cleaning work directory: {work_dir}")
        import shutil
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    logger.info("=" * 70)
    logger.info("Starting functional preprocessing with ANTs registration")
    logger.info("=" * 70)
    logger.info("")

    try:
        results = run_func_preprocessing(
            config=config,
            subject=subject,
            func_file=echo_files,  # Multi-echo
            output_dir=output_dir,
            work_dir=work_dir,
            anat_derivatives=anat_derivatives
        )

        logger.info("=" * 70)
        logger.info("Preprocessing Complete!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Key outputs:")
        logger.info(f"  Preprocessed: {results.get('preprocessed')}")
        logger.info(f"  func→T1w transform: {results.get('func_to_t1w_transform')}")
        logger.info(f"  func→MNI transform: {results.get('func_to_mni_transform')}")
        logger.info(f"  MNI→func transform: {results.get('mni_to_func_transform')}")
        logger.info("")

        # Check that registration transforms exist
        if 'func_to_t1w_transform' in results:
            logger.info("✓ ANTs registration pipeline completed successfully")
            return 0
        else:
            logger.error("✗ ANTs registration transforms not found in results")
            return 1

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
