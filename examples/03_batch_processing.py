#!/usr/bin/env python3
"""
Example 3: Batch Processing Multiple Subjects

This example demonstrates how to process multiple subjects in a study,
with error handling, progress tracking, and resumption capabilities.
"""

import logging
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
from neurovrai.config import load_config

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'batch_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Load configuration
config = load_config(Path('config.yaml'))

# Study paths
study_root = Path('/mnt/bytopia/IRC805')
bids_root = study_root / 'bids'
derivatives_dir = study_root / 'derivatives'
work_root = study_root / 'work'

# Progress tracking
status_file = log_dir / 'batch_progress.json'


# ============================================================================
# Helper Functions
# ============================================================================

def load_progress() -> Dict:
    """Load processing progress from JSON file."""
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress: Dict):
    """Save processing progress to JSON file."""
    with open(status_file, 'w') as f:
        json.dump(progress, f, indent=2)


def find_subjects() -> List[str]:
    """Find all subjects with NIfTI data."""
    subjects = []
    for subject_dir in sorted(bids_root.glob('IRC805-*')):
        if subject_dir.is_dir():
            subjects.append(subject_dir.name)
    return subjects


def process_anatomical(subject: str, progress: Dict) -> bool:
    """Process anatomical data for a subject."""
    if progress.get(subject, {}).get('anat') == 'complete':
        logger.info(f"  Anatomical already complete for {subject}, skipping")
        return True

    from neurovrai.preprocess.workflows.anat_preprocess import run_anat_preprocessing

    nifti_dir = bids_root / subject / 'anat'
    t1w_files = list(nifti_dir.glob('*T1*.nii.gz'))

    if not t1w_files:
        logger.warning(f"  No T1w files found for {subject}")
        return False

    try:
        results = run_anat_preprocessing(
            config=config,
            subject=subject,
            t1w_file=t1w_files[0],
            output_dir=derivatives_dir,
            work_dir=work_root / subject
        )

        # Mark as complete
        if subject not in progress:
            progress[subject] = {}
        progress[subject]['anat'] = 'complete'
        save_progress(progress)

        logger.info(f"  ✓ Anatomical complete for {subject}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Anatomical failed for {subject}: {e}")
        progress[subject]['anat'] = f'failed: {str(e)}'
        save_progress(progress)
        return False


def process_dwi(subject: str, progress: Dict) -> bool:
    """Process DWI data for a subject."""
    if progress.get(subject, {}).get('dwi') == 'complete':
        logger.info(f"  DWI already complete for {subject}, skipping")
        return True

    from neurovrai.preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

    dwi_dir = bids_root / subject / 'dwi'
    if not dwi_dir.exists():
        logger.info(f"  No DWI directory for {subject}")
        return False

    dwi_files = sorted(list(dwi_dir.glob('*DTI*.nii.gz')))
    bval_files = sorted(list(dwi_dir.glob('*DTI*.bval')))
    bvec_files = sorted(list(dwi_dir.glob('*DTI*.bvec')))
    rev_phase_files = sorted(list(dwi_dir.glob('*SE_EPI*.nii.gz')))

    if not dwi_files:
        logger.info(f"  No DWI files found for {subject}")
        return False

    try:
        results = run_dwi_multishell_topup_preprocessing(
            config=config,
            subject=subject,
            dwi_files=dwi_files,
            bval_files=bval_files,
            bvec_files=bvec_files,
            rev_phase_files=rev_phase_files if rev_phase_files else None,
            output_dir=derivatives_dir,
            work_dir=work_root / subject
        )

        # Mark as complete
        if subject not in progress:
            progress[subject] = {}
        progress[subject]['dwi'] = 'complete'
        save_progress(progress)

        logger.info(f"  ✓ DWI complete for {subject}")
        return True

    except Exception as e:
        logger.error(f"  ✗ DWI failed for {subject}: {e}")
        progress[subject]['dwi'] = f'failed: {str(e)}'
        save_progress(progress)
        return False


def process_functional(subject: str, progress: Dict) -> bool:
    """Process functional data for a subject."""
    # Check if anatomical is complete (dependency)
    if progress.get(subject, {}).get('anat') != 'complete':
        logger.warning(f"  Functional requires anatomical to be complete for {subject}")
        return False

    if progress.get(subject, {}).get('func') == 'complete':
        logger.info(f"  Functional already complete for {subject}, skipping")
        return True

    from neurovrai.preprocess.workflows.func_preprocess import run_func_preprocessing

    func_dir = bids_root / subject / 'func'
    if not func_dir.exists():
        logger.info(f"  No functional directory for {subject}")
        return False

    func_files = sorted(list(func_dir.glob('*RESTING*.nii.gz')))
    anat_derivatives = derivatives_dir / subject / 'anat'

    if not func_files:
        logger.info(f"  No functional files found for {subject}")
        return False

    try:
        results = run_func_preprocessing(
            config=config,
            subject=subject,
            func_file=func_files,
            output_dir=derivatives_dir,
            anat_derivatives=anat_derivatives,
            work_dir=work_root / subject
        )

        # Mark as complete
        if subject not in progress:
            progress[subject] = {}
        progress[subject]['func'] = 'complete'
        save_progress(progress)

        logger.info(f"  ✓ Functional complete for {subject}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Functional failed for {subject}: {e}")
        progress[subject]['func'] = f'failed: {str(e)}'
        save_progress(progress)
        return False


# ============================================================================
# Main Batch Processing
# ============================================================================

def main():
    logger.info("="*70)
    logger.info("BATCH PREPROCESSING - IRC805 STUDY")
    logger.info("="*70)
    logger.info(f"Study root: {study_root}")
    logger.info(f"Progress file: {status_file}")
    logger.info("")

    # Find all subjects
    subjects = find_subjects()
    logger.info(f"Found {len(subjects)} subjects")
    logger.info("")

    # Load progress
    progress = load_progress()

    # Process each subject
    for i, subject in enumerate(subjects, 1):
        logger.info(f"[{i}/{len(subjects)}] Processing {subject}")
        logger.info("-" * 70)

        # 1. Anatomical (required)
        anat_success = process_anatomical(subject, progress)

        # 2. DWI (independent)
        dwi_success = process_dwi(subject, progress)

        # 3. Functional (depends on anatomical)
        func_success = process_functional(subject, progress)

        logger.info("")

    # ========================================================================
    # Final Summary
    # ========================================================================

    logger.info("="*70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*70)

    summary = {
        'anat': {'complete': 0, 'failed': 0, 'skipped': 0},
        'dwi': {'complete': 0, 'failed': 0, 'skipped': 0},
        'func': {'complete': 0, 'failed': 0, 'skipped': 0}
    }

    for subject, modalities in progress.items():
        for modality in ['anat', 'dwi', 'func']:
            status = modalities.get(modality, 'skipped')
            if status == 'complete':
                summary[modality]['complete'] += 1
            elif 'failed' in str(status):
                summary[modality]['failed'] += 1
            else:
                summary[modality]['skipped'] += 1

    logger.info(f"Total subjects: {len(subjects)}")
    logger.info("")
    logger.info("Anatomical:")
    logger.info(f"  ✓ Complete: {summary['anat']['complete']}")
    logger.info(f"  ✗ Failed: {summary['anat']['failed']}")
    logger.info(f"  - Skipped: {summary['anat']['skipped']}")
    logger.info("")
    logger.info("DWI:")
    logger.info(f"  ✓ Complete: {summary['dwi']['complete']}")
    logger.info(f"  ✗ Failed: {summary['dwi']['failed']}")
    logger.info(f"  - Skipped: {summary['dwi']['skipped']}")
    logger.info("")
    logger.info("Functional:")
    logger.info(f"  ✓ Complete: {summary['func']['complete']}")
    logger.info(f"  ✗ Failed: {summary['func']['failed']}")
    logger.info(f"  - Skipped: {summary['func']['skipped']}")
    logger.info("")
    logger.info(f"Progress saved to: {status_file}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
