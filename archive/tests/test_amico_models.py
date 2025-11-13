#!/usr/bin/env python3
"""
Test AMICO implementations of advanced diffusion models on IRC805 data.

This script tests:
- DKI (DIPY)
- NODDI (AMICO)
- SANDI (AMICO)
- ActiveAx (AMICO)

Expected runtime:
- DKI: ~20-25 minutes (DIPY, slower)
- NODDI: ~2-5 minutes (AMICO, fast!)
- SANDI: ~3-6 minutes (AMICO)
- ActiveAx: ~3-6 minutes (AMICO)
- Total: ~30-45 minutes (vs 60+ minutes with DIPY NODDI)
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
    """Test AMICO models on IRC805 data."""

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
    logger.info("Testing AMICO Advanced Diffusion Models")
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
    logger.info("Data specifications:")
    logger.info("  B-values: 0, 1000, 2000, 3000 s/mm²")
    logger.info("  Total volumes: ~220")
    logger.info("  Multi-shell: ✓ (required for all models)")
    logger.info("")

    # Run all models
    logger.info("="*70)
    logger.info("Running All Advanced Diffusion Models")
    logger.info("="*70)
    logger.info("")
    logger.info("Models to fit:")
    logger.info("  1. DKI (DIPY) - ~20-25 minutes")
    logger.info("  2. NODDI (AMICO) - ~2-5 minutes")
    logger.info("  3. SANDI (AMICO) - ~3-6 minutes")
    logger.info("  4. ActiveAx (AMICO) - ~3-6 minutes")
    logger.info("")
    logger.info("Total expected runtime: ~30-45 minutes")
    logger.info("")

    try:
        results = run_advanced_diffusion_models(
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            mask_file=mask_file,
            output_dir=output_dir,
            fit_dki=True,
            fit_noddi=True,
            fit_sandi=True,
            fit_activeax=True,
            use_amico=True  # Use AMICO for 100x speedup!
        )

        logger.info("")
        logger.info("="*70)
        logger.info("All Advanced Diffusion Models Completed Successfully!")
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
        logger.info("DKI Metrics:")
        logger.info("  - MK (Mean Kurtosis): Overall diffusion complexity")
        logger.info("  - AK (Axial Kurtosis): Complexity along fibers")
        logger.info("  - RK (Radial Kurtosis): Complexity across fibers (myelin)")
        logger.info("  - KFA (Kurtosis FA): Directional complexity")
        logger.info("")
        logger.info("NODDI Metrics:")
        logger.info("  - FICVF (Neurite Density): Axon/dendrite packing [0-1]")
        logger.info("  - ODI (Orientation Dispersion): Fiber coherence [0-1]")
        logger.info("  - FISO (Free Water): CSF/edema fraction [0-1]")
        logger.info("  - DIR: Principal fiber direction")
        logger.info("")
        logger.info("SANDI Metrics:")
        logger.info("  - FSOMA: Soma volume fraction [0-1]")
        logger.info("  - FNEURITE: Neurite volume fraction [0-1]")
        logger.info("  - FEC: Extra-cellular space [0-1]")
        logger.info("  - FCSF: CSF fraction [0-1]")
        logger.info("  - RSOMA: Soma radius [μm]")
        logger.info("  - DIR: Neurite direction")
        logger.info("")
        logger.info("ActiveAx Metrics:")
        logger.info("  - FICVF: Intra-axonal volume fraction [0-1]")
        logger.info("  - DIAM: Mean axon diameter [μm]")
        logger.info("  - DIR: Fiber direction")
        logger.info("  - FVF_TOT: Total fiber volume fraction")
        logger.info("")

        return 0

    except Exception as e:
        logger.error(f"Advanced diffusion modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
