#!/usr/bin/env python3
"""
Test advanced DWI analyses (DKI and NODDI) on real IRC805 data.

This script tests the advanced diffusion models on preprocessed multi-shell DWI data.
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
    """Test advanced DWI models on IRC805 data."""

    # Subject and paths
    subject = 'IRC805-0580101'
    study_root = Path('/mnt/bytopia/development/IRC805')

    # Input files (from DWI preprocessing)
    dwi_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'eddy_corrected' / 'eddy_corrected.nii.gz'
    bval_file = study_root / 'work' / subject / 'dwi_topup' / 'dwi_merged.bval'
    bvec_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'rotated_bvec' / 'eddy_corrected.eddy_rotated_bvecs'
    mask_file = study_root / 'derivatives' / 'dwi_topup' / subject / 'mask' / 'dwi_merged_roi_brain_mask.nii.gz'

    # Output directory
    output_dir = study_root / 'derivatives' / 'dwi_topup' / subject / 'advanced_models'

    logger.info("="*70)
    logger.info("Testing Advanced DWI Models (DKI & NODDI)")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
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
    logger.info("B-value information:")
    logger.info("  Expected: b=0, 1000, 2000, 3000 s/mm²")
    logger.info("  Total volumes: ~220")
    logger.info("  Multi-shell: ✓ (required for DKI/NODDI)")
    logger.info("")

    # Run advanced models
    logger.info("="*70)
    logger.info("Running Advanced Diffusion Models")
    logger.info("="*70)
    logger.info("")

    try:
        logger.info("This will take several minutes...")
        logger.info("  - DKI fitting: ~5-10 minutes")
        logger.info("  - NODDI fitting: ~10-15 minutes")
        logger.info("")

        results = run_advanced_diffusion_models(
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            mask_file=mask_file,
            output_dir=output_dir,
            fit_dki=True,
            fit_noddi=True
        )

        logger.info("")
        logger.info("="*70)
        logger.info("Advanced DWI Models Completed Successfully!")
        logger.info("="*70)
        logger.info("")

        # Display DKI results
        if 'dki' in results and results['dki']:
            logger.info("DKI (Diffusion Kurtosis Imaging) Outputs:")
            for metric, path in results['dki'].items():
                if path and Path(path).exists():
                    logger.info(f"  ✓ {metric}: {Path(path).name}")
                else:
                    logger.warning(f"  ✗ {metric}: NOT GENERATED")

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
        logger.info(f"Output directory: {output_dir}")
        logger.info("")
        logger.info("Expected DKI metrics:")
        logger.info("  - MK (Mean Kurtosis): Overall kurtosis measure")
        logger.info("  - AK (Axial Kurtosis): Kurtosis along principal diffusion direction")
        logger.info("  - RK (Radial Kurtosis): Kurtosis perpendicular to principal direction")
        logger.info("  - KFA (Kurtosis Fractional Anisotropy): Directional kurtosis")
        logger.info("")
        logger.info("Expected NODDI metrics:")
        logger.info("  - ODI (Orientation Dispersion Index): Neurite dispersion (0-1)")
        logger.info("  - FICVF (Intracellular Volume Fraction): Neurite density (0-1)")
        logger.info("  - FISO (Isotropic Volume Fraction): Free water (0-1)")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Advanced diffusion modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
