#!/usr/bin/env python3
"""
Test advanced DWI processing (DKI and NODDI) for IRC805-0580101.

This script runs DKI and NODDI models on the eddy-corrected DWI data.
"""

from pathlib import Path
import logging
from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/advanced_dwi.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run advanced DWI models."""
    subject = 'IRC805-0580101'
    study_root = Path('/mnt/bytopia/IRC805')

    # Input files from DWI preprocessing
    dwi_dir = study_root / 'derivatives' / subject / 'dwi'
    work_dir = study_root / 'work' / subject

    dwi_file = dwi_dir / 'eddy_corrected' / 'eddy_corrected.nii.gz'
    bval_file = work_dir / 'dwi_merged.bval'

    # Use eddy-rotated bvecs if available, otherwise use merged bvecs
    rotated_bvec = dwi_dir / 'rotated_bvec' / 'eddy_corrected.eddy_rotated_bvecs'
    if rotated_bvec.exists():
        bvec_file = rotated_bvec
        logger.info(f"Using eddy-rotated bvecs: {bvec_file}")
    else:
        bvec_file = work_dir / 'dwi_merged.bvec'
        logger.info(f"Using merged bvecs: {bvec_file}")

    mask_file = dwi_dir / 'mask' / 'dwi_merged_roi_brain_mask.nii.gz'

    # Output directory for advanced models
    output_dir = dwi_dir  # Will create dki/ and noddi/ subdirectories

    logger.info("=" * 70)
    logger.info(f"ADVANCED DWI MODELS: {subject}")
    logger.info("=" * 70)
    logger.info(f"DWI file: {dwi_file}")
    logger.info(f"bval file: {bval_file}")
    logger.info(f"bvec file: {bvec_file}")
    logger.info(f"Mask file: {mask_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Check all files exist
    for f in [dwi_file, bval_file, bvec_file, mask_file]:
        if not f.exists():
            logger.error(f"Required file not found: {f}")
            return False

    try:
        # Run advanced diffusion models
        logger.info("Running DKI and NODDI models...")
        results = run_advanced_diffusion_models(
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            mask_file=mask_file,
            output_dir=output_dir,
            fit_dki=True,
            fit_noddi=True
        )

        logger.info("=" * 70)
        logger.info("ADVANCED DWI MODELS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("DKI outputs:")
        for key, path in results.get('dki', {}).items():
            logger.info(f"  {key}: {path}")
        logger.info("")
        logger.info("NODDI outputs:")
        for key, path in results.get('noddi', {}).items():
            logger.info(f"  {key}: {path}")

        return True

    except Exception as e:
        logger.error(f"Advanced DWI models failed: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
