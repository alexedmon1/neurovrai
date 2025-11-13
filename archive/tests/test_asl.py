#!/usr/bin/env python3
"""
Test script for ASL preprocessing workflow.

Tests basic ASL processing on IRC805-0580101:
1. Motion correction
2. Label-control subtraction
3. CBF quantification
4. Brain extraction
"""

import logging
from pathlib import Path
from mri_preprocess.config import load_config
from mri_preprocess.workflows.asl_preprocess import run_asl_preprocessing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/asl_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    # Load configuration
    config = load_config(Path('config.yaml'))

    # Subject and paths
    subject = 'IRC805-0580101'
    study_root = Path('/mnt/bytopia/IRC805')

    # ASL source file (4D time series with label-control pairs)
    asl_file = study_root / 'subjects' / subject / 'nifti' / 'asl' / '1104_IRC805-0580101_WIP_SOURCE_-_DelRec_-_pCASL1.nii.gz'

    # Anatomical files (for registration)
    anat_dir = study_root / 'derivatives' / subject / 'anat'
    t1w_brain = anat_dir / f'{subject}_brain.nii.gz'

    # Check if files exist
    if not asl_file.exists():
        logger.error(f"ASL file not found: {asl_file}")
        return

    logger.info("="*70)
    logger.info("Testing ASL Preprocessing Workflow")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"ASL file: {asl_file}")
    logger.info(f"T1w brain: {t1w_brain if t1w_brain.exists() else 'Not found'}")
    logger.info("")

    # Run ASL preprocessing
    results = run_asl_preprocessing(
        config=config.get('asl', {}),
        subject=subject,
        asl_file=asl_file,
        output_dir=study_root,
        t1w_brain=t1w_brain if t1w_brain.exists() else None,
        label_control_order='control_first',
        labeling_duration=1.8,
        post_labeling_delay=1.8,
        normalize_to_mni=False
    )

    logger.info("")
    logger.info("="*70)
    logger.info("ASL Preprocessing Test Complete")
    logger.info("="*70)
    logger.info("")
    logger.info("Outputs:")
    for key, value in results.items():
        if isinstance(value, Path):
            logger.info(f"  {key}: {value}")

    # Verify CBF values are in physiological range
    if 'cbf' in results:
        import nibabel as nib
        import numpy as np

        cbf_img = nib.load(results['cbf'])
        cbf_data = cbf_img.get_fdata()

        if 'brain_mask' in results:
            mask = nib.load(results['brain_mask']).get_fdata()
            cbf_masked = cbf_data[mask > 0]

            logger.info("")
            logger.info("CBF Statistics (within brain mask):")
            logger.info(f"  Mean: {np.mean(cbf_masked):.2f} ml/100g/min")
            logger.info(f"  Median: {np.median(cbf_masked):.2f} ml/100g/min")
            logger.info(f"  Std: {np.std(cbf_masked):.2f} ml/100g/min")
            logger.info(f"  Range: [{np.min(cbf_masked):.2f}, {np.max(cbf_masked):.2f}]")
            logger.info("")
            logger.info("Expected ranges:")
            logger.info("  Gray matter: 40-60 ml/100g/min")
            logger.info("  White matter: 20-30 ml/100g/min")


if __name__ == '__main__':
    main()
