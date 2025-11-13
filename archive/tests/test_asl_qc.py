#!/usr/bin/env python3
"""
Test script for ASL preprocessing workflow with QC.

Tests the complete ASL preprocessing pipeline including quality control
and tissue-specific CBF analysis on IRC805-0580101 subject.
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
        logging.FileHandler('logs/asl_qc_test.log'),
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

    # Anatomical files (for registration and tissue masks)
    anat_dir = study_root / 'derivatives' / subject / 'anat'
    t1w_brain = anat_dir / f'{subject}_brain.nii.gz'

    # Tissue segmentation masks (for tissue-specific CBF)
    seg_dir = anat_dir / 'segmentation'
    gm_mask = seg_dir / f'{subject}_pve_1.nii.gz'  # Gray matter
    wm_mask = seg_dir / f'{subject}_pve_2.nii.gz'  # White matter
    csf_mask = seg_dir / f'{subject}_pve_0.nii.gz'  # CSF

    # Check if files exist
    if not asl_file.exists():
        logger.error(f"ASL file not found: {asl_file}")
        return

    logger.info("="*70)
    logger.info("Testing ASL Preprocessing Workflow with QC")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"ASL file: {asl_file}")
    logger.info(f"T1w brain: {t1w_brain if t1w_brain.exists() else 'Not found'}")
    logger.info(f"GM mask: {gm_mask if gm_mask.exists() else 'Not found'}")
    logger.info(f"WM mask: {wm_mask if wm_mask.exists() else 'Not found'}")
    logger.info(f"CSF mask: {csf_mask if csf_mask.exists() else 'Not found'}")
    logger.info("")

    # Run ASL preprocessing with QC enabled
    results = run_asl_preprocessing(
        config=config.get('asl', {}),
        subject=subject,
        asl_file=asl_file,
        output_dir=study_root,
        t1w_brain=t1w_brain if t1w_brain.exists() else None,
        gm_mask=gm_mask if gm_mask.exists() else None,
        wm_mask=wm_mask if wm_mask.exists() else None,
        csf_mask=csf_mask if csf_mask.exists() else None,
        label_control_order='control_first',
        labeling_duration=1.8,
        post_labeling_delay=1.8,
        normalize_to_mni=False
    )

    logger.info("")
    logger.info("="*70)
    logger.info("ASL Preprocessing with QC Test Complete")
    logger.info("="*70)
    logger.info("")
    logger.info("Outputs:")
    for key, value in results.items():
        if isinstance(value, Path):
            logger.info(f"  {key}: {value}")

    # Highlight QC outputs
    if 'qc_report' in results:
        logger.info("")
        logger.info("QC Report Generated:")
        logger.info(f"  HTML Report: {results['qc_report']}")
        logger.info(f"  QC Directory: {results['qc_dir']}")
        logger.info("")
        logger.info("To view the report:")
        logger.info(f"  Open: {results['qc_report']}")


if __name__ == '__main__':
    main()
