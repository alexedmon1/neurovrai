#!/usr/bin/env python3
"""
Test script for resting state fMRI preprocessing with TEDANA.

This script tests the full preprocessing pipeline on IRC805-0580101:
1. Multi-echo motion correction (MCFLIRT on middle echo)
2. Apply motion transforms to all echoes
3. TEDANA multi-echo denoising (removes thermal noise)
4. Bandpass filtering (0.001-0.08 Hz)
5. Spatial smoothing (6mm FWHM)
6. Quality control (motion metrics, tSNR, HTML report)

Note: ICA-AROMA is disabled as it's redundant with TEDANA for multi-echo data.

Key improvement: Motion correction is applied BEFORE TEDANA to ensure perfect
echo alignment, following best practices from fMRIPrep and TEDANA documentation.

Usage:
    python test_rest_preprocessing.py
"""

import logging
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.workflows.func_preprocess import run_func_preprocessing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_rest_preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run resting state preprocessing test."""
    logger.info("=" * 70)
    logger.info("Testing Resting State fMRI Preprocessing")
    logger.info("=" * 70)
    logger.info("")

    # Configuration
    config = {
        'tr': 1.029,  # seconds
        'te': [10.0, 30.0, 50.0],  # Echo times in milliseconds
        'highpass': 0.001,  # Hz
        'lowpass': 0.08,    # Hz (standard resting state band)
        'fwhm': 6,          # Smoothing kernel FWHM (mm)
        'n_procs': 6,       # Number of parallel processes
        'tedana': {
            'enabled': True,
            'tedpca': 'kundu',
            'tree': 'kundu'
        },
        'aroma': {
            'enabled': False,  # Disabled: redundant with TEDANA for multi-echo
            'denoise_type': 'both'
        },
        'acompcor': {
            'enabled': False,  # Will enable after anatomical preprocessing
            'num_components': 6
        }
    }

    # Paths
    study_root = Path('/mnt/bytopia/IRC805')
    subject = 'IRC805-0580101'

    # Multi-echo functional files
    rest_dir = study_root / f'subjects/{subject}/nifti/rest'
    func_files = [
        rest_dir / f'501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e1.nii.gz',
        rest_dir / f'501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e2.nii.gz',
        rest_dir / f'501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e3.nii.gz'
    ]

    # Verify input files exist
    logger.info("Verifying input files...")
    for i, func_file in enumerate(func_files, 1):
        if func_file.exists():
            size_mb = func_file.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ Echo {i}: {func_file.name} ({size_mb:.1f} MB)")
        else:
            logger.error(f"  ✗ Echo {i}: {func_file} NOT FOUND")
            return 1
    logger.info("")

    # Anatomical tissue masks (optional for now - will add after anat preprocessing)
    # anat_dir = study_root / f'derivatives/anat_preproc/{subject}'
    # csf_mask = anat_dir / 'fast_seg_0.nii.gz'  # CSF
    # wm_mask = anat_dir / 'fast_seg_2.nii.gz'   # White matter

    try:
        # Run preprocessing
        logger.info("Starting preprocessing pipeline...")
        logger.info("")

        results = run_func_preprocessing(
            config=config,
            subject=subject,
            func_file=func_files,  # List for multi-echo
            output_dir=study_root,  # Study root
            # csf_mask=csf_mask,  # Add after anatomical preprocessing
            # wm_mask=wm_mask,    # Add after anatomical preprocessing
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  Derivatives dir: {results['derivatives_dir']}")
        logger.info(f"  Working dir: {results['work_dir']}")
        if 'preprocessed' in results and results['preprocessed']:
            logger.info(f"  Preprocessed: {results['preprocessed']}")
        if 'motion_params' in results and results['motion_params']:
            logger.info(f"  Motion params: {results['motion_params']}")
        if 'tedana_optcom' in results:
            logger.info(f"  TEDANA optcom: {results['tedana_optcom']}")
        if 'tedana_report' in results:
            logger.info(f"  TEDANA report: {results['tedana_report']}")
        if 'qc_report' in results:
            logger.info(f"  QC report: {results['qc_report']}")
        logger.info("")

        logger.info("Quality Control Summary:")
        if 'motion_qc' in results:
            motion_qc = results['motion_qc']
            logger.info(f"  Mean FD: {motion_qc['mean_fd']:.3f} mm")
            logger.info(f"  Max FD: {motion_qc['max_fd']:.3f} mm")
            logger.info(f"  Outlier volumes: {motion_qc['n_outliers_fd']} ({motion_qc['percent_outliers']:.1f}%)")
        if 'tsnr_qc' in results:
            tsnr_qc = results['tsnr_qc']
            logger.info(f"  Mean tSNR: {tsnr_qc['mean_tsnr']:.2f}")
            logger.info(f"  Median tSNR: {tsnr_qc['median_tsnr']:.2f}")
        logger.info("")

        logger.info("Next steps:")
        logger.info("  1. Review QC report:")
        logger.info(f"     firefox {results.get('qc_report', 'N/A')}")
        logger.info("  2. Check TEDANA report (HTML):")
        logger.info(f"     firefox {results.get('tedana_report', 'N/A')}")
        logger.info("  3. Visualize preprocessed data in fsleyes")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
