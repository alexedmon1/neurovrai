#!/usr/bin/env python3
"""
Batch Process All IRC805 Subjects

This script processes all subjects from raw DICOM through complete preprocessing:
1. Validates configuration
2. DICOM → NIfTI conversion
3. Anatomical preprocessing (required first)
4. DWI, Functional, ASL preprocessing (parallel)
5. Quality control for all modalities

Progress is tracked and the script can be resumed if interrupted.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f'batch_all_subjects_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import config system
from mri_preprocess.config import load_config

def validate_config(config_file: Path) -> Dict:
    """Validate configuration file."""
    logger.info("="*70)
    logger.info("CONFIGURATION VALIDATION")
    logger.info("="*70)

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)

    try:
        config = load_config(config_file)
        logger.info(f"✓ Configuration loaded from: {config_file}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Validate key paths
    required_keys = ['project_dir', 'execution']
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            sys.exit(1)

    project_dir = Path(config['project_dir'])
    if not project_dir.exists():
        logger.error(f"Project directory does not exist: {project_dir}")
        sys.exit(1)

    logger.info(f"✓ Project directory: {project_dir}")
    logger.info(f"✓ Execution mode: {config['execution']['plugin']}")
    logger.info(f"✓ Parallel processes: {config['execution']['n_procs']}")
    logger.info("")

    return config


def find_subjects(dicom_root: Path) -> List[str]:
    """Find all subjects with DICOM data."""
    subjects = []
    for subject_dir in sorted(dicom_root.glob('IRC805-*')):
        if subject_dir.is_dir():
            subjects.append(subject_dir.name)
    return subjects


def load_progress(progress_file: Path) -> Dict:
    """Load progress from JSON file."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress: Dict, progress_file: Path):
    """Save progress to JSON file."""
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def run_dicom_conversion(subject: str, config: Dict, dicom_dir: Path, progress: Dict, progress_file: Path) -> bool:
    """Convert DICOM to NIfTI for a subject."""
    if progress.get(subject, {}).get('dicom_conversion') == 'complete':
        logger.info(f"  DICOM conversion already complete for {subject}")
        return True

    from mri_preprocess.dicom.bids import run_dicom_to_bids

    try:
        logger.info(f"  Converting DICOM → NIfTI...")

        project_dir = Path(config['project_dir'])
        output_dir = project_dir / 'bids' / subject

        run_dicom_to_bids(
            dicom_dir=dicom_dir,
            output_dir=output_dir,
            subject_id=subject
        )

        # Mark as complete
        if subject not in progress:
            progress[subject] = {}
        progress[subject]['dicom_conversion'] = 'complete'
        save_progress(progress, progress_file)

        logger.info(f"  ✓ DICOM conversion complete")
        return True

    except Exception as e:
        logger.error(f"  ✗ DICOM conversion failed: {e}")
        progress[subject]['dicom_conversion'] = f'failed: {str(e)}'
        save_progress(progress, progress_file)
        return False


def run_anatomical(subject: str, config: Dict, progress: Dict, progress_file: Path) -> bool:
    """Run anatomical preprocessing."""
    if progress.get(subject, {}).get('anat') == 'complete':
        logger.info(f"  Anatomical already complete for {subject}")
        return True

    from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing

    project_dir = Path(config['project_dir'])
    bids_dir = project_dir / 'bids' / subject / 'anat'
    t1w_files = list(bids_dir.glob('*T1*.nii.gz'))

    if not t1w_files:
        logger.warning(f"  No T1w files found for {subject}")
        return False

    try:
        logger.info(f"  Running anatomical preprocessing...")

        results = run_anat_preprocessing(
            config=config,
            subject=subject,
            t1w_file=t1w_files[0],
            output_dir=project_dir / 'derivatives',
            work_dir=project_dir / 'work' / subject
        )

        # Mark as complete
        if subject not in progress:
            progress[subject] = {}
        progress[subject]['anat'] = 'complete'
        save_progress(progress, progress_file)

        logger.info(f"  ✓ Anatomical complete")
        return True

    except Exception as e:
        logger.error(f"  ✗ Anatomical failed: {e}")
        progress[subject]['anat'] = f'failed: {str(e)}'
        save_progress(progress, progress_file)
        return False


def run_dwi(subject: str, config: Dict, progress: Dict, progress_file: Path) -> bool:
    """Run DWI preprocessing."""
    if progress.get(subject, {}).get('dwi') == 'complete':
        logger.info(f"  DWI already complete for {subject}")
        return True

    from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

    project_dir = Path(config['project_dir'])
    dwi_dir = project_dir / 'bids' / subject / 'dwi'

    if not dwi_dir.exists():
        logger.info(f"  No DWI data for {subject}")
        progress[subject]['dwi'] = 'no_data'
        save_progress(progress, progress_file)
        return False

    dwi_files = sorted(list(dwi_dir.glob('*DTI*.nii.gz')))
    bval_files = sorted(list(dwi_dir.glob('*DTI*.bval')))
    bvec_files = sorted(list(dwi_dir.glob('*DTI*.bvec')))
    rev_phase_files = sorted(list(dwi_dir.glob('*SE_EPI*.nii.gz')))

    if not dwi_files:
        logger.info(f"  No DWI files found for {subject}")
        progress[subject]['dwi'] = 'no_data'
        save_progress(progress, progress_file)
        return False

    try:
        logger.info(f"  Running DWI preprocessing...")

        results = run_dwi_multishell_topup_preprocessing(
            config=config,
            subject=subject,
            dwi_files=dwi_files,
            bval_files=bval_files,
            bvec_files=bvec_files,
            rev_phase_files=rev_phase_files if rev_phase_files else None,
            output_dir=project_dir / 'derivatives',
            work_dir=project_dir / 'work' / subject
        )

        # Mark as complete
        if subject not in progress:
            progress[subject] = {}
        progress[subject]['dwi'] = 'complete'
        save_progress(progress, progress_file)

        logger.info(f"  ✓ DWI complete")
        return True

    except Exception as e:
        logger.error(f"  ✗ DWI failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        progress[subject]['dwi'] = f'failed: {str(e)}'
        save_progress(progress, progress_file)
        return False


def run_functional(subject: str, config: Dict, progress: Dict, progress_file: Path) -> bool:
    """Run functional preprocessing."""
    # Check dependency
    if progress.get(subject, {}).get('anat') != 'complete':
        logger.warning(f"  Functional requires anatomical to be complete for {subject}")
        return False

    if progress.get(subject, {}).get('func') == 'complete':
        logger.info(f"  Functional already complete for {subject}")
        return True

    from mri_preprocess.workflows.func_preprocess import run_func_preprocessing

    project_dir = Path(config['project_dir'])
    func_dir = project_dir / 'bids' / subject / 'func'

    if not func_dir.exists():
        logger.info(f"  No functional data for {subject}")
        progress[subject]['func'] = 'no_data'
        save_progress(progress, progress_file)
        return False

    func_files = sorted(list(func_dir.glob('*RESTING*.nii.gz')))
    anat_derivatives = project_dir / 'derivatives' / subject / 'anat'

    if not func_files:
        logger.info(f"  No functional files found for {subject}")
        progress[subject]['func'] = 'no_data'
        save_progress(progress, progress_file)
        return False

    try:
        logger.info(f"  Running functional preprocessing...")

        results = run_func_preprocessing(
            config=config,
            subject=subject,
            func_file=func_files,
            output_dir=project_dir / 'derivatives',
            anat_derivatives=anat_derivatives,
            work_dir=project_dir / 'work' / subject
        )

        # Mark as complete
        if subject not in progress:
            progress[subject] = {}
        progress[subject]['func'] = 'complete'
        save_progress(progress, progress_file)

        logger.info(f"  ✓ Functional complete")
        return True

    except Exception as e:
        logger.error(f"  ✗ Functional failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        progress[subject]['func'] = f'failed: {str(e)}'
        save_progress(progress, progress_file)
        return False


def run_asl(subject: str, config: Dict, progress: Dict, progress_file: Path) -> bool:
    """Run ASL preprocessing."""
    # Check dependency
    if progress.get(subject, {}).get('anat') != 'complete':
        logger.warning(f"  ASL requires anatomical to be complete for {subject}")
        return False

    if progress.get(subject, {}).get('asl') == 'complete':
        logger.info(f"  ASL already complete for {subject}")
        return True

    from mri_preprocess.workflows.asl_preprocess import run_asl_preprocessing

    project_dir = Path(config['project_dir'])
    asl_dir = project_dir / 'bids' / subject / 'asl'

    if not asl_dir.exists():
        logger.info(f"  No ASL data for {subject}")
        progress[subject]['asl'] = 'no_data'
        save_progress(progress, progress_file)
        return False

    asl_files = list(asl_dir.glob('*ASL*.nii.gz'))
    anat_derivatives = project_dir / 'derivatives' / subject / 'anat'

    if not asl_files:
        logger.info(f"  No ASL files found for {subject}")
        progress[subject]['asl'] = 'no_data'
        save_progress(progress, progress_file)
        return False

    # Get anatomical outputs
    t1w_brain = anat_derivatives / 'brain' / 'brain.nii.gz'
    gm_mask = list((anat_derivatives / 'segmentation').glob('*POSTERIOR_02.nii.gz'))[0]
    wm_mask = list((anat_derivatives / 'segmentation').glob('*POSTERIOR_03.nii.gz'))[0]
    dicom_dir = project_dir / 'raw' / 'dicom' / subject / 'asl'

    try:
        logger.info(f"  Running ASL preprocessing...")

        results = run_asl_preprocessing(
            config=config,
            subject=subject,
            asl_file=asl_files[0],
            output_dir=project_dir / 'derivatives',
            t1w_brain=t1w_brain,
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            dicom_dir=dicom_dir if dicom_dir.exists() else None,
            normalize_to_mni=config.get('asl', {}).get('normalize_to_mni', False)
        )

        # Mark as complete
        if subject not in progress:
            progress[subject] = {}
        progress[subject]['asl'] = 'complete'
        save_progress(progress, progress_file)

        logger.info(f"  ✓ ASL complete")
        return True

    except Exception as e:
        logger.error(f"  ✗ ASL failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        progress[subject]['asl'] = f'failed: {str(e)}'
        save_progress(progress, progress_file)
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch process all IRC805 subjects')
    parser.add_argument('--config', type=Path, default=Path('config.yaml'),
                       help='Configuration file (default: config.yaml)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run')
    parser.add_argument('--subjects', nargs='+',
                       help='Process specific subjects (default: all)')

    args = parser.parse_args()

    # Validate configuration
    config = validate_config(args.config)
    project_dir = Path(config['project_dir'])

    # Setup progress tracking
    progress_file = log_dir / 'batch_all_subjects_progress.json'
    progress = load_progress(progress_file) if args.resume else {}

    # Find subjects
    dicom_root = project_dir / 'raw' / 'dicom'
    if not dicom_root.exists():
        logger.error(f"DICOM root not found: {dicom_root}")
        sys.exit(1)

    all_subjects = find_subjects(dicom_root)

    if args.subjects:
        subjects = [s for s in all_subjects if s in args.subjects]
    else:
        subjects = all_subjects

    logger.info("="*70)
    logger.info("BATCH PROCESSING - IRC805 STUDY")
    logger.info("="*70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Project root: {project_dir}")
    logger.info(f"Total subjects: {len(subjects)}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Progress file: {progress_file}")
    logger.info("")

    # Process each subject
    for i, subject in enumerate(subjects, 1):
        logger.info("="*70)
        logger.info(f"[{i}/{len(subjects)}] PROCESSING {subject}")
        logger.info("="*70)

        dicom_dir = dicom_root / subject

        # Step 1: DICOM Conversion
        logger.info("Step 1: DICOM → NIfTI Conversion")
        dicom_success = run_dicom_conversion(subject, config, dicom_dir, progress, progress_file)

        if not dicom_success:
            logger.warning(f"Skipping {subject} - DICOM conversion failed")
            continue

        # Step 2: Anatomical (required)
        logger.info("Step 2: Anatomical Preprocessing")
        anat_success = run_anatomical(subject, config, progress, progress_file)

        if not anat_success:
            logger.warning(f"Skipping modalities for {subject} - anatomical failed")
            continue

        # Step 3: DWI (independent)
        logger.info("Step 3: DWI Preprocessing")
        run_dwi(subject, config, progress, progress_file)

        # Step 4: Functional (depends on anatomical)
        logger.info("Step 4: Functional Preprocessing")
        run_functional(subject, config, progress, progress_file)

        # Step 5: ASL (depends on anatomical)
        logger.info("Step 5: ASL Preprocessing")
        run_asl(subject, config, progress, progress_file)

        logger.info("")

    # Final Summary
    logger.info("="*70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*70)

    summary = {
        'dicom_conversion': {'complete': 0, 'failed': 0},
        'anat': {'complete': 0, 'failed': 0, 'skipped': 0},
        'dwi': {'complete': 0, 'failed': 0, 'no_data': 0},
        'func': {'complete': 0, 'failed': 0, 'no_data': 0},
        'asl': {'complete': 0, 'failed': 0, 'no_data': 0}
    }

    for subject, modalities in progress.items():
        for modality in ['dicom_conversion', 'anat', 'dwi', 'func', 'asl']:
            status = modalities.get(modality, 'skipped')
            if status == 'complete':
                summary[modality]['complete'] += 1
            elif status == 'no_data':
                summary[modality]['no_data'] += 1
            elif 'failed' in str(status):
                summary[modality]['failed'] += 1
            else:
                summary[modality]['skipped'] += 1

    logger.info(f"Total subjects processed: {len(subjects)}")
    logger.info("")
    logger.info("DICOM Conversion:")
    logger.info(f"  ✓ Complete: {summary['dicom_conversion']['complete']}")
    logger.info(f"  ✗ Failed: {summary['dicom_conversion']['failed']}")
    logger.info("")
    logger.info("Anatomical:")
    logger.info(f"  ✓ Complete: {summary['anat']['complete']}")
    logger.info(f"  ✗ Failed: {summary['anat']['failed']}")
    logger.info(f"  - Skipped: {summary['anat']['skipped']}")
    logger.info("")
    logger.info("DWI:")
    logger.info(f"  ✓ Complete: {summary['dwi']['complete']}")
    logger.info(f"  ✗ Failed: {summary['dwi']['failed']}")
    logger.info(f"  - No data: {summary['dwi']['no_data']}")
    logger.info("")
    logger.info("Functional:")
    logger.info(f"  ✓ Complete: {summary['func']['complete']}")
    logger.info(f"  ✗ Failed: {summary['func']['failed']}")
    logger.info(f"  - No data: {summary['func']['no_data']}")
    logger.info("")
    logger.info("ASL:")
    logger.info(f"  ✓ Complete: {summary['asl']['complete']}")
    logger.info(f"  ✗ Failed: {summary['asl']['failed']}")
    logger.info(f"  - No data: {summary['asl']['no_data']}")
    logger.info("")
    logger.info(f"Progress saved to: {progress_file}")
    logger.info(f"Full log: {log_file}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
