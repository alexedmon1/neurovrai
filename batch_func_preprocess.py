#!/usr/bin/env python3
"""
Batch functional preprocessing for IRC805 subjects.

This script processes all subjects with functional MRI data in the IRC805 study.
It skips subjects that have already been preprocessed.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Setup logging
log_dir = Path('/mnt/bytopia/IRC805/logs')
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / f'batch_func_preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def find_subjects_with_func(bids_dir: Path):
    """Find all subjects with functional data."""
    subjects = []
    for subject_dir in sorted(bids_dir.glob('IRC805-*')):
        if subject_dir.is_dir():
            func_dir = subject_dir / 'func'
            if func_dir.exists() and any(func_dir.glob('*.nii.gz')):
                subjects.append(subject_dir.name)
    return subjects


def check_if_preprocessed(derivatives_dir: Path, subject: str):
    """Check if subject has already been preprocessed."""
    subject_func_dir = derivatives_dir / subject / 'func'

    # Check for key output files (match current pipeline output names)
    preprocessed_file = subject_func_dir / f'{subject}_bold_preprocessed.nii.gz'

    # Also check for any preprocessed file as fallback
    if not preprocessed_file.exists():
        preprocessed_files = list(subject_func_dir.glob('*_bold_preprocessed.nii.gz'))
        if preprocessed_files:
            return True

    if preprocessed_file.exists():
        return True
    return False


def preprocess_subject(subject: str, config_path: Path):
    """
    Preprocess a single subject using run_simple_pipeline.py.

    Parameters
    ----------
    subject : str
        Subject ID
    config_path : Path
        Path to config.yaml

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info(f"Processing subject: {subject}")
    logger.info("=" * 80)

    # Construct command
    # run_simple_pipeline.py requires --nifti-dir and uses skip flags for modality selection
    nifti_dir = Path('/mnt/bytopia/IRC805/bids') / subject

    cmd = [
        'uv', 'run', 'python',
        'run_simple_pipeline.py',
        '--subject', subject,
        '--config', str(config_path),
        '--nifti-dir', str(nifti_dir),
        '--skip-anat',  # Anatomical already preprocessed
        '--skip-dwi',   # Only run functional
        '--skip-asl'    # Only run functional
    ]

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("")

    try:
        # Run preprocessing
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per subject
        )

        logger.info("STDOUT:")
        logger.info(result.stdout)

        if result.stderr:
            logger.warning("STDERR:")
            logger.warning(result.stderr)

        logger.info(f"✅ {subject} completed successfully")
        logger.info("")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"❌ {subject} timed out after 2 hours")
        logger.error("")
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {subject} failed with exit code {e.returncode}")
        logger.error("STDOUT:")
        logger.error(e.stdout)
        logger.error("STDERR:")
        logger.error(e.stderr)
        logger.error("")
        return False

    except Exception as e:
        logger.error(f"❌ {subject} failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("")
        return False


def main():
    """Main batch processing function."""
    logger.info("=" * 80)
    logger.info("IRC805 Batch Functional Preprocessing")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    # Paths
    study_root = Path('/mnt/bytopia/IRC805')
    bids_dir = study_root / 'bids'
    derivatives_dir = study_root / 'derivatives'
    config_path = study_root / 'config.yaml'

    # Verify paths
    if not bids_dir.exists():
        logger.error(f"BIDS directory not found: {bids_dir}")
        return 1

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    # Find subjects
    logger.info("Finding subjects with functional data...")
    subjects = find_subjects_with_func(bids_dir)
    logger.info(f"Found {len(subjects)} subjects with functional data")
    logger.info("")

    # Check preprocessing status
    logger.info("Checking preprocessing status...")
    subjects_to_process = []
    subjects_already_done = []

    for subject in subjects:
        if check_if_preprocessed(derivatives_dir, subject):
            subjects_already_done.append(subject)
            logger.info(f"  ✓ {subject} - Already preprocessed")
        else:
            subjects_to_process.append(subject)
            logger.info(f"  ○ {subject} - Needs preprocessing")

    logger.info("")
    logger.info(f"Already preprocessed: {len(subjects_already_done)}")
    logger.info(f"To be processed: {len(subjects_to_process)}")
    logger.info("")

    if not subjects_to_process:
        logger.info("All subjects already preprocessed!")
        return 0

    # Process subjects
    logger.info("=" * 80)
    logger.info("Starting batch preprocessing...")
    logger.info("=" * 80)
    logger.info("")

    successful = []
    failed = []

    for i, subject in enumerate(subjects_to_process, 1):
        logger.info(f"Progress: {i}/{len(subjects_to_process)}")

        if preprocess_subject(subject, config_path):
            successful.append(subject)
        else:
            failed.append(subject)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Batch Preprocessing Summary")
    logger.info("=" * 80)
    logger.info(f"Total subjects: {len(subjects)}")
    logger.info(f"Already done: {len(subjects_already_done)}")
    logger.info(f"Processed: {len(subjects_to_process)}")
    logger.info(f"  - Successful: {len(successful)}")
    logger.info(f"  - Failed: {len(failed)}")
    logger.info("")

    if successful:
        logger.info("Successful subjects:")
        for subject in successful:
            logger.info(f"  ✅ {subject}")
        logger.info("")

    if failed:
        logger.info("Failed subjects:")
        for subject in failed:
            logger.info(f"  ❌ {subject}")
        logger.info("")

    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    return 0 if not failed else 1


if __name__ == '__main__':
    exit(main())
