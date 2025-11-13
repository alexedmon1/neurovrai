#!/usr/bin/env python3
"""
Test ONLY AMICO models (skip DKI since it takes 20+ minutes and we already have results).

This tests:
- NODDI (AMICO) - ~2-5 minutes
- SANDI (AMICO) - ~3-6 minutes
- ActiveAx (AMICO) - ~3-6 minutes

Expected total runtime: ~10-15 minutes
"""

import sys
from pathlib import Path
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test AMICO models (skip DKI) on IRC805 data."""

    # Subject and paths
    subject = 'IRC805-0580101'
    study_root = Path('/mnt/bytopia/development/IRC805')

    # Input files (from DWI preprocessing)
    dwi_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'eddy_corrected' / 'eddy_corrected.nii.gz'
    bval_file = study_root / 'work' / subject / 'dwi_topup' / 'dwi_merged.bval'
    bvec_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'rotated_bvec' / 'eddy_corrected.eddy_rotated_bvecs'
    mask_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'mask' / 'dwi_merged_roi_brain_mask.nii.gz'

    # Output directory
    output_dir = study_root / 'derivatives' / 'dwi_topup' / subject / 'advanced_models_amico'

    logger.info("="*70)
    logger.info("Testing AMICO Models (NODDI, SANDI, ActiveAx)")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info("")
    logger.info("Skipping DKI (already completed with DIPY)")
    logger.info("")

    # Check files exist
    logger.info("Checking input files...")
    files_ok = True
    for name, fpath in [('DWI', dwi_file), ('bval', bval_file), ('bvec', bvec_file), ('mask', mask_file)]:
        if fpath.exists():
            logger.info(f"  ✓ {name}: {fpath.name}")
        else:
            logger.error(f"  ✗ {name}: NOT FOUND - {fpath}")
            files_ok = False

    if not files_ok:
        logger.error("Missing required files. Exiting.")
        return 1

    logger.info("")
    logger.info("Models to fit:")
    logger.info("  1. NODDI (AMICO) - ~2-5 minutes")
    logger.info("  2. SANDI (AMICO) - ~3-6 minutes")
    logger.info("  3. ActiveAx/CylinderZeppelinBall (AMICO) - ~3-6 minutes")
    logger.info("")
    logger.info("Total expected runtime: ~8-15 minutes")
    logger.info("")

    try:
        results = run_advanced_diffusion_models(
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            mask_file=mask_file,
            output_dir=output_dir,
            fit_dki=False,        # Skip DKI (already done)
            fit_noddi=True,       # AMICO
            fit_sandi=True,       # AMICO
            fit_activeax=True,    # AMICO - CylinderZeppelinBall
            use_amico=True
        )

        logger.info("")
        logger.info("="*70)
        logger.info("AMICO Models Completed Successfully!")
        logger.info("="*70)
        logger.info("")

        # Display NODDI results
        if 'noddi' in results and results['noddi']:
            logger.info("NODDI (Neurite Orientation Dispersion) Outputs:")
            for metric, path in results['noddi'].items():
                if path and Path(path).exists():
                    logger.info(f"  ✓ {metric}: {Path(path).name}")
                else:
                    logger.warning(f"  ✗ {metric}: NOT GENERATED")
            logger.info("")

        # Display SANDI results
        if 'sandi' in results and results['sandi']:
            logger.info("SANDI (Soma And Neurite Density) Outputs:")
            for metric, path in results['sandi'].items():
                if path and Path(path).exists():
                    logger.info(f"  ✓ {metric}: {Path(path).name}")
                else:
                    logger.warning(f"  ✗ {metric}: NOT GENERATED")
            logger.info("")

        # Display ActiveAx results
        if 'activeax' in results and results['activeax']:
            logger.info("ActiveAx (Axon Diameter Distribution) Outputs:")
            for metric, path in results['activeax'].items():
                if path and Path(path).exists():
                    logger.info(f"  ✓ {metric}: {Path(path).name}")
                else:
                    logger.warning(f"  ✗ {metric}: NOT GENERATED")
            logger.info("")

        logger.info(f"Output directory: {output_dir}")
        logger.info("")
        logger.info("For detailed metric descriptions, see:")
        logger.info("  AMICO_MODELS_DOCUMENTATION.md")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"AMICO modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
