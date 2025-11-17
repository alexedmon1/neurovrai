#!/usr/bin/env python3
"""
Simple MRI Preprocessing Pipeline

A straightforward, human-readable pipeline that processes MRI data sequentially.

Usage:
    python run_simple_pipeline.py --subject IRC805-0580101 --dicom-dir /path/to/dicom --config config.yaml

Steps:
    1. Convert DICOM to NIfTI (if needed)
    2. Anatomical preprocessing (required first)
    3. DWI preprocessing (optional)
    4. Functional preprocessing (optional)
    5. ASL preprocessing (optional)
"""

import argparse
import logging
import sys
from pathlib import Path

# Import configuration and workflow functions
from mri_preprocess.config import load_config
from mri_preprocess.utils.dicom_converter import convert_subject_dicoms
from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing
from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing
from mri_preprocess.workflows.func_preprocess import run_func_preprocessing
from mri_preprocess.workflows.asl_preprocess import run_asl_preprocessing


# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simple_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def convert_dicom_to_nifti(subject, dicom_dir, output_dir):
    """Convert DICOM files to NIfTI format."""
    logger.info("="*70)
    logger.info("STEP 1: Converting DICOM to NIfTI")
    logger.info("="*70)

    try:
        convert_subject_dicoms(
            subject=subject,
            dicom_dir=dicom_dir,
            output_dir=output_dir / 'bids'
        )
        logger.info("✓ DICOM conversion complete\n")
        return output_dir / 'bids' / subject

    except Exception as e:
        logger.error(f"✗ DICOM conversion failed: {e}\n")
        return None


def preprocess_anatomical(subject, config, nifti_dir, derivatives_dir, work_dir):
    """Run anatomical (T1w) preprocessing."""
    logger.info("="*70)
    logger.info("STEP 2: Anatomical Preprocessing")
    logger.info("="*70)

    # Find T1w file
    anat_dir = nifti_dir / 'anat'
    t1w_files = list(anat_dir.glob('*T1*.nii.gz'))

    if not t1w_files:
        logger.error("✗ No T1w file found\n")
        return None

    t1w_file = t1w_files[0]
    logger.info(f"Input: {t1w_file.name}")

    try:
        results = run_anat_preprocessing(
            config=config,
            subject=subject,
            t1w_file=t1w_file,
            output_dir=derivatives_dir,
            work_dir=work_dir
        )
        logger.info("✓ Anatomical preprocessing complete\n")
        return results

    except Exception as e:
        logger.error(f"✗ Anatomical preprocessing failed: {e}\n")
        return None


def preprocess_dwi(subject, config, nifti_dir, derivatives_dir, work_dir):
    """Run diffusion (DWI) preprocessing."""
    logger.info("="*70)
    logger.info("STEP 3: DWI Preprocessing")
    logger.info("="*70)

    # Find DWI files
    dwi_dir = nifti_dir / 'dwi'
    if not dwi_dir.exists():
        logger.info("⊘ No DWI data found - skipping\n")
        return None

    dwi_files = sorted(list(dwi_dir.glob('*DTI*.nii.gz')))
    if not dwi_files:
        logger.info("⊘ No DWI files found - skipping\n")
        return None

    # Get corresponding bval/bvec files
    bval_files = [f.with_suffix('').with_suffix('.bval') for f in dwi_files]
    bvec_files = [f.with_suffix('').with_suffix('.bvec') for f in dwi_files]

    # Find reverse phase encoding files for TOPUP
    rev_phase_files = list(dwi_dir.glob('*SE_EPI*.nii.gz'))

    logger.info(f"Found {len(dwi_files)} DWI files")
    logger.info(f"Found {len(rev_phase_files)} reverse phase files")

    try:
        results = run_dwi_multishell_topup_preprocessing(
            config=config,
            subject=subject,
            dwi_files=dwi_files,
            bval_files=bval_files,
            bvec_files=bvec_files,
            rev_phase_files=rev_phase_files if rev_phase_files else None,
            output_dir=derivatives_dir,
            work_dir=work_dir
        )
        logger.info("✓ DWI preprocessing complete\n")
        return results

    except Exception as e:
        logger.error(f"✗ DWI preprocessing failed: {e}\n")
        return None


def preprocess_functional(subject, config, nifti_dir, derivatives_dir, work_dir):
    """Run functional (resting-state fMRI) preprocessing."""
    logger.info("="*70)
    logger.info("STEP 4: Functional Preprocessing")
    logger.info("="*70)

    # Find functional files
    func_dir = nifti_dir / 'func'
    if not func_dir.exists():
        logger.info("⊘ No functional data found - skipping\n")
        return None

    func_files = list(func_dir.glob('*RESTING*.nii.gz'))
    if not func_files:
        logger.info("⊘ No functional files found - skipping\n")
        return None

    # Check that anatomical preprocessing was done
    anat_dir = derivatives_dir / subject / 'anat'

    # Find brain file (may be in brain/ subdirectory)
    brain_files = list(anat_dir.rglob('*brain.nii.gz'))
    if not brain_files:
        logger.error("✗ Anatomical preprocessing required first\n")
        return None
    t1w_brain = brain_files[0]

    logger.info(f"Found {len(func_files)} functional files")
    is_multi_echo = len(func_files) > 1 or 'ME' in func_files[0].name
    logger.info(f"Multi-echo: {is_multi_echo}")

    try:
        results = run_func_preprocessing(
            config=config,
            subject=subject,
            func_files=func_files,
            output_dir=derivatives_dir,
            t1w_brain=t1w_brain,
            work_dir=work_dir
        )
        logger.info("✓ Functional preprocessing complete\n")
        return results

    except Exception as e:
        logger.error(f"✗ Functional preprocessing failed: {e}\n")
        return None


def preprocess_asl(subject, config, nifti_dir, derivatives_dir, work_dir, dicom_dir):
    """Run ASL (perfusion) preprocessing."""
    logger.info("="*70)
    logger.info("STEP 5: ASL Preprocessing")
    logger.info("="*70)

    # Find ASL files
    asl_dir = nifti_dir / 'asl'
    if not asl_dir.exists():
        logger.info("⊘ No ASL data found - skipping\n")
        return None

    asl_files = list(asl_dir.glob('*pCASL*.nii.gz'))
    if not asl_files:
        logger.info("⊘ No ASL files found - skipping\n")
        return None

    # Use SOURCE file if available
    source_files = [f for f in asl_files if 'SOURCE' in f.name]
    asl_file = source_files[0] if source_files else asl_files[0]

    # Check that anatomical preprocessing was done
    anat_dir = derivatives_dir / subject / 'anat'
    seg_dir = anat_dir / 'segmentation'

    # Find brain file (may be in brain/ subdirectory)
    brain_files = list(anat_dir.rglob('*brain.nii.gz'))
    if not brain_files:
        logger.error("✗ Anatomical preprocessing required first\n")
        return None
    t1w_brain = brain_files[0]

    gm_mask = seg_dir / 'POSTERIOR_02.nii.gz'
    wm_mask = seg_dir / 'POSTERIOR_03.nii.gz'
    csf_mask = seg_dir / 'POSTERIOR_01.nii.gz'

    # Find DICOM directory for parameter extraction
    dicom_asl_dir = None
    if dicom_dir:
        for date_dir in dicom_dir.glob('*'):
            asl_subdirs = list(date_dir.glob('*pCASL*'))
            if asl_subdirs:
                dicom_asl_dir = asl_subdirs[0]
                break

    logger.info(f"Input: {asl_file.name}")
    logger.info(f"DICOM parameters: {'Available' if dicom_asl_dir else 'Using defaults from config'}")

    try:
        results = run_asl_preprocessing(
            config=config,
            subject=subject,
            asl_file=asl_file,
            output_dir=derivatives_dir,
            t1w_brain=t1w_brain,
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            csf_mask=csf_mask,
            dicom_dir=dicom_asl_dir,
            work_dir=work_dir
        )
        logger.info("✓ ASL preprocessing complete\n")
        return results

    except Exception as e:
        logger.error(f"✗ ASL preprocessing failed: {e}\n")
        return None


def main():
    """Main pipeline execution."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Simple MRI preprocessing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--subject', required=True, help='Subject ID (e.g., IRC805-0580101)')
    parser.add_argument('--config', type=Path, default=Path('config.yaml'), help='Config file')
    parser.add_argument('--dicom-dir', type=Path, help='DICOM directory (if converting from DICOM)')
    parser.add_argument('--nifti-dir', type=Path, help='NIfTI directory (if already converted)')
    parser.add_argument('--skip-anat', action='store_true', help='Skip anatomical preprocessing')
    parser.add_argument('--skip-dwi', action='store_true', help='Skip DWI preprocessing')
    parser.add_argument('--skip-func', action='store_true', help='Skip functional preprocessing')
    parser.add_argument('--skip-asl', action='store_true', help='Skip ASL preprocessing')

    args = parser.parse_args()

    # Validate inputs
    if not args.dicom_dir and not args.nifti_dir:
        logger.error("Must provide either --dicom-dir or --nifti-dir")
        sys.exit(1)

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Get study root from config
    study_root = Path(config['project_dir'])
    logger.info(f"Study root: {study_root}")

    # Setup directories
    derivatives_dir = study_root / 'derivatives'
    work_dir = study_root / 'work' / args.subject

    derivatives_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Subject: {args.subject}")
    logger.info(f"Output: {derivatives_dir / args.subject}")
    logger.info("")

    # Step 1: DICOM Conversion (if needed)
    if args.dicom_dir:
        nifti_dir = convert_dicom_to_nifti(args.subject, args.dicom_dir, study_root)
        if not nifti_dir:
            logger.error("Pipeline failed at DICOM conversion")
            sys.exit(1)
    else:
        nifti_dir = args.nifti_dir
        logger.info("Using existing NIfTI files\n")

    # Step 2: Anatomical Preprocessing (required for other modalities)
    if not args.skip_anat:
        anat_results = preprocess_anatomical(args.subject, config, nifti_dir, derivatives_dir, work_dir)
        if not anat_results:
            logger.error("Pipeline failed at anatomical preprocessing")
            sys.exit(1)
    else:
        logger.info("Skipping anatomical preprocessing\n")

    # Step 3: DWI Preprocessing
    if not args.skip_dwi:
        preprocess_dwi(args.subject, config, nifti_dir, derivatives_dir, work_dir)

    # Step 4: Functional Preprocessing
    if not args.skip_func:
        preprocess_functional(args.subject, config, nifti_dir, derivatives_dir, work_dir)

    # Step 5: ASL Preprocessing
    if not args.skip_asl:
        preprocess_asl(args.subject, config, nifti_dir, derivatives_dir, work_dir, args.dicom_dir)

    # Summary
    logger.info("="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Results: {derivatives_dir / args.subject}")
    logger.info(f"Working files: {work_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
