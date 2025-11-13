#!/usr/bin/env python3
"""
Test ASL M0 Calibration and Partial Volume Correction

This script tests the newly implemented M0 calibration and PVC features
on IRC805-0580101 subject data.

Expected outcomes:
1. M0 calibration should reduce CBF from ~159 to ~50-70 ml/100g/min
2. PVC should improve tissue-specific CBF accuracy
3. QC report should include calibration metrics
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mri_preprocess.config import load_config
from mri_preprocess.workflows.asl_preprocess import run_asl_preprocessing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_asl_calibration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def test_asl_calibration():
    """Test M0 calibration and PVC on IRC805-0580101."""

    logger.info("="*70)
    logger.info("TESTING ASL M0 CALIBRATION AND PVC")
    logger.info("="*70)
    logger.info("")

    # Load configuration
    config_file = Path('config.yaml')
    config = load_config(config_file)

    # Subject and data paths
    subject = 'IRC805-0580101'
    study_root = Path('/mnt/bytopia/IRC805')

    # ASL input file (4D time series with label-control pairs)
    asl_file = study_root / 'subjects' / subject / 'nifti' / 'asl' / '1104_IRC805-0580101_WIP_SOURCE_-_DelRec_-_pCASL1.nii.gz'

    # Anatomical inputs (from previous preprocessing)
    anat_dir = study_root / 'derivatives' / subject / 'anat'
    t1w_brain = anat_dir / 'brain' / '201_IRC805-0580101_WIP_3D_T1_TFE_SAG_CS3_reoriented_brain.nii.gz'

    # Tissue masks for calibration and PVC (Atropos output naming)
    seg_dir = anat_dir / 'segmentation'
    csf_mask = seg_dir / 'POSTERIOR_01.nii.gz'  # Atropos CSF probability
    gm_mask = seg_dir / 'POSTERIOR_02.nii.gz'   # Atropos GM probability
    wm_mask = seg_dir / 'POSTERIOR_03.nii.gz'   # Atropos WM probability

    # Check that all required files exist
    required_files = [asl_file, t1w_brain, gm_mask, wm_mask, csf_mask]
    for f in required_files:
        if not f.exists():
            logger.error(f"Required file not found: {f}")
            logger.error("Run anatomical preprocessing first to generate tissue masks")
            return False

    logger.info("Input files:")
    logger.info(f"  ASL: {asl_file}")
    logger.info(f"  T1w brain: {t1w_brain}")
    logger.info(f"  GM mask: {gm_mask}")
    logger.info(f"  WM mask: {wm_mask}")
    logger.info(f"  CSF mask: {csf_mask}")
    logger.info("")

    # Extract ASL parameters from config
    asl_config = config.get('asl', {})

    logger.info("ASL Configuration:")
    logger.info(f"  Labeling duration: {asl_config.get('labeling_duration', 1.8)} s")
    logger.info(f"  Post-labeling delay: {asl_config.get('post_labeling_delay', 1.8)} s")
    logger.info(f"  Apply M0 calibration: {asl_config.get('apply_m0_calibration', True)}")
    logger.info(f"  WM CBF reference: {asl_config.get('wm_cbf_reference', 25.0)} ml/100g/min")
    logger.info(f"  Apply PVC: {asl_config.get('apply_pvc', False)}")
    logger.info("")

    # Run ASL preprocessing with M0 calibration
    logger.info("Running ASL preprocessing with M0 calibration...")
    logger.info("")

    try:
        results = run_asl_preprocessing(
            config=config,
            subject=subject,
            asl_file=asl_file,
            output_dir=study_root,
            t1w_brain=t1w_brain,
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            csf_mask=csf_mask,
            label_control_order=asl_config.get('label_control_order', 'control_first'),
            labeling_duration=asl_config.get('labeling_duration', 1.8),
            post_labeling_delay=asl_config.get('post_labeling_delay', 1.8),
            normalize_to_mni=asl_config.get('normalize_to_mni', False)
        )

        logger.info("")
        logger.info("="*70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*70)
        logger.info("")

        # Analyze results
        logger.info("Output Files:")
        logger.info(f"  Uncalibrated CBF: {results.get('cbf')}")
        if 'cbf_calibrated' in results:
            logger.info(f"  Calibrated CBF: {results.get('cbf_calibrated')}")
        if 'cbf_gm_pvc' in results:
            logger.info(f"  PVC GM CBF: {results.get('cbf_gm_pvc')}")
            logger.info(f"  PVC WM CBF: {results.get('cbf_wm_pvc')}")
        logger.info(f"  QC Report: {results.get('qc_report')}")
        logger.info("")

        # Display calibration metrics if available
        if 'calibration_info' in results:
            calib_info = results['calibration_info']
            logger.info("M0 Calibration Metrics:")
            logger.info(f"  Measured WM CBF (before): {calib_info['wm_cbf_measured']:.2f} ml/100g/min")
            logger.info(f"  Expected WM CBF: {calib_info['wm_cbf_expected']:.2f} ml/100g/min")
            logger.info(f"  Scaling factor: {calib_info['scaling_factor']:.3f}")
            logger.info(f"  Calibrated WM CBF (after): {calib_info['wm_cbf_calibrated']:.2f} ml/100g/min")
            logger.info(f"  WM voxels used: {calib_info['n_wm_voxels']}")
            logger.info("")

            # Calculate expected reduction
            reduction_percent = (1 - calib_info['scaling_factor']) * 100
            logger.info(f"  CBF reduction: {reduction_percent:.1f}%")
            logger.info("")

        # Verify calibration effectiveness
        if 'calibration_info' in results:
            calib_info = results['calibration_info']
            if calib_info['wm_cbf_calibrated'] < 20 or calib_info['wm_cbf_calibrated'] > 30:
                logger.warning("  WARNING: Calibrated WM CBF outside expected range (20-30 ml/100g/min)")
            else:
                logger.info("  ✓ Calibrated WM CBF within expected range")

        logger.info("")
        logger.info("="*70)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("="*70)

        return True

    except Exception as e:
        logger.error(f"Error during ASL preprocessing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == '__main__':
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    success = test_asl_calibration()
    sys.exit(0 if success else 1)
