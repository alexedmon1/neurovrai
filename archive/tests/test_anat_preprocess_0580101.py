#!/usr/bin/env python3
"""
Test anatomical preprocessing for IRC805-0580101.

This will generate the tissue segmentations needed for functional ACompCor.

Usage:
    python test_anat_preprocess_0580101.py
"""

import logging
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_anat_preprocess.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run anatomical preprocessing for IRC805-0580101."""
    logger.info("=" * 70)
    logger.info("Testing Anatomical Preprocessing - IRC805-0580101")
    logger.info("=" * 70)
    logger.info("")

    # Paths
    study_root = Path('/mnt/bytopia/IRC805')
    subject = 'IRC805-0580101'

    # Configuration
    config = {
        'paths': {
            'logs': str(study_root / 'logs')
        },
        'templates': {
            'mni152_t1_2mm': str(Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'))
        },
        'bet': {
            'frac': 0.5,
            'reduce_bias': True,
            'robust': True
        },
        'fast': {
            'bias_iters': 4,
            'bias_lowpass': 10
        },
        'execution': {
            'plugin': 'MultiProc',
            'plugin_args': {'n_procs': 6}
        },
        'n_procs': 6,
        'run_qc': True
    }

    # Anatomical files
    anat_dir = study_root / f'subjects/{subject}/nifti/anat'
    t1w_file = anat_dir / '201_IRC805-0580101_WIP_3D_T1_TFE_SAG_CS3.nii.gz'

    # Verify input files exist
    logger.info("Verifying input files...")
    if t1w_file.exists():
        size_mb = t1w_file.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ T1w: {t1w_file.name} ({size_mb:.1f} MB)")
    else:
        logger.error(f"  ✗ T1w: {t1w_file} NOT FOUND")
        return 1
    logger.info("")

    try:
        # Run preprocessing
        logger.info("Starting anatomical preprocessing...")
        logger.info("This will generate:")
        logger.info("  - Brain-extracted T1w")
        logger.info("  - Bias-corrected T1w")
        logger.info("  - Tissue segmentation (CSF, GM, WM)")
        logger.info("  - QC metrics and visualizations")
        logger.info("")

        results = run_anat_preprocessing(
            config=config,
            subject=subject,
            t1w_file=t1w_file,
            output_dir=study_root,
            run_qc=config['run_qc']
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("ANATOMICAL PREPROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Output files:")

        # Calculate derivatives directory path
        derivatives_dir = study_root / 'derivatives' / 'anat_preproc' / subject
        logger.info(f"  Derivatives dir: {derivatives_dir}")

        if results.get('brain'):
            logger.info(f"  Brain: {results['brain']}")
        if results.get('brain_mask'):
            logger.info(f"  Brain mask: {results['brain_mask']}")
        if results.get('bias_corrected'):
            logger.info(f"  Bias corrected: {results['bias_corrected']}")
        logger.info("")

        logger.info("Tissue Segmentations (for ACompCor):")
        if results.get('csf_prob'):
            logger.info(f"  CSF: {results['csf_prob']}")
        if results.get('gm_prob'):
            logger.info(f"  GM:  {results['gm_prob']}")
        if results.get('wm_prob'):
            logger.info(f"  WM:  {results['wm_prob']}")
        logger.info("")

        logger.info("Next steps:")
        logger.info("  1. Run functional preprocessing with ACompCor enabled")
        logger.info("  2. ACompCor will use these tissue segmentations automatically")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
