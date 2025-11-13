#!/usr/bin/env python3
"""
Test ActiveAx (CylinderZeppelinBall) model with isExvivo=False fix.
"""

import logging
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from mri_preprocess.workflows.amico_models import fit_activeax_amico

def main():
    logger.info("="*70)
    logger.info("Testing ActiveAx with isExvivo=False fix")
    logger.info("="*70)

    subject = 'IRC805-0580101'

    # Input files
    study_root = Path('/mnt/bytopia/development/IRC805')
    dwi_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'eddy_corrected' / 'eddy_corrected.nii.gz'
    bval_file = study_root / 'work' / subject / 'dwi_topup' / 'dwi_merged.bval'
    bvec_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'rotated_bvec' / 'eddy_corrected.eddy_rotated_bvecs'
    mask_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'mask' / 'dwi_merged_roi_brain_mask.nii.gz'

    # Output directory
    output_dir = Path(f'/mnt/bytopia/development/IRC805/derivatives/dwi_topup/{subject}/advanced_models_amico')

    logger.info(f"Subject: {subject}")
    logger.info("")

    logger.info("Checking input files...")
    for name, path in [('DWI', dwi_file), ('bval', bval_file),
                       ('bvec', bvec_file), ('mask', mask_file)]:
        if path.exists():
            logger.info(f"  ✓ {name}: {path.name}")
        else:
            logger.error(f"  ✗ {name}: NOT FOUND - {path}")
            return 1

    logger.info("")
    logger.info("Running ActiveAx (CylinderZeppelinBall)...")
    logger.info("")

    try:
        results = fit_activeax_amico(
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            mask_file=mask_file,
            output_dir=output_dir,
            n_threads=12
        )

        logger.info("")
        logger.info("="*70)
        logger.info("ActiveAx COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("")
        logger.info("Output files:")
        for key, path in results.items():
            if path and path.exists():
                logger.info(f"  ✓ {key}: {path}")
            else:
                logger.warning(f"  ✗ {key}: NOT FOUND")

        return 0

    except Exception as e:
        logger.error(f"ActiveAx fitting failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
