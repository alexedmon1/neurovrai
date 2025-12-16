#!/usr/bin/env python3
"""
Migrate BEDPOSTX outputs from work directory to derivatives.

This script finds all bedpostx.bedpostX folders in the work directory and copies them
to the appropriate location in derivatives for use by structural connectivity analysis.

Usage:
    python scripts/migrate_bedpostx_to_derivatives.py --study-root /mnt/bytopia/IRC805
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_bedpostx_in_work(work_dir: Path):
    """
    Find all bedpostx.bedpostX directories in work directory.

    Parameters
    ----------
    work_dir : Path
        Work directory to search

    Returns
    -------
    list
        List of (subject_id, bedpostx_path) tuples
    """
    bedpostx_dirs = []

    # Pattern: work/{subject}/dwi_preprocess/bedpostx/bedpostx.bedpostX
    for bedpostx_path in work_dir.glob('*/dwi_preprocess/bedpostx/bedpostx.bedpostX'):
        # Extract subject from path
        subject = bedpostx_path.parts[-4]  # work/{SUBJECT}/dwi_preprocess/bedpostx/bedpostx.bedpostX
        if subject.startswith('IRC805-'):
            bedpostx_dirs.append((subject, bedpostx_path))

    return bedpostx_dirs


def migrate_bedpostx(
    subject: str,
    bedpostx_work_dir: Path,
    derivatives_dir: Path,
    dry_run: bool = False
) -> bool:
    """
    Migrate BEDPOSTX output to derivatives directory.

    Parameters
    ----------
    subject : str
        Subject ID
    bedpostx_work_dir : Path
        Path to bedpostx.bedpostX in work directory
    derivatives_dir : Path
        Derivatives directory
    dry_run : bool
        If True, only log what would be done without actually copying

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    dest_dir = derivatives_dir / subject / 'dwi' / 'bedpostx'
    bedpostx_input_dir = derivatives_dir / subject / 'dwi' / 'bedpostx_input'

    logger.info(f"  Subject: {subject}")
    logger.info(f"  Source: {bedpostx_work_dir}")
    logger.info(f"  Destination: {dest_dir}")

    # Check if source exists and has expected structure
    if not bedpostx_work_dir.exists():
        logger.error(f"  ✗ Source directory not found")
        return False

    # Check if destination already exists
    if dest_dir.exists():
        logger.info(f"  Destination already exists, will overwrite")
        if not dry_run:
            shutil.rmtree(dest_dir)

    # Copy directory (preserve symlinks)
    if dry_run:
        logger.info(f"  [DRY RUN] Would copy BEDPOSTX output directory")
        if bedpostx_input_dir.exists():
            logger.info(f"  [DRY RUN] Would copy input files from bedpostx_input/")
        return True
    else:
        try:
            # Copy BEDPOSTX output, preserving symlinks (like merged -> .)
            shutil.copytree(bedpostx_work_dir, dest_dir, symlinks=True)
            logger.info(f"  ✓ Copied BEDPOSTX output directory")

            # Copy essential input files from bedpostx_input if it exists
            # For older preprocessing runs, reconstruct from other derivatives locations
            copied = []

            if bedpostx_input_dir.exists():
                # New workflow: copy from bedpostx_input
                input_files = ['nodif_brain_mask.nii.gz', 'bvals', 'bvecs']
                for fname in input_files:
                    src_file = bedpostx_input_dir / fname
                    if src_file.exists():
                        dest_file = dest_dir / fname
                        if not dest_file.exists():
                            shutil.copy2(src_file, dest_file)
                            copied.append(fname)
            else:
                # Legacy workflow: reconstruct from other derivatives directories
                dwi_dir = derivatives_dir / subject / 'dwi'

                # Copy brain mask from mask directory
                mask_files = list((dwi_dir / 'mask').glob('*brain_mask.nii.gz')) if (dwi_dir / 'mask').exists() else []
                if mask_files:
                    dest_mask = dest_dir / 'nodif_brain_mask.nii.gz'
                    if not dest_mask.exists():
                        shutil.copy2(mask_files[0], dest_mask)
                        copied.append('nodif_brain_mask.nii.gz')

                # Copy bvals and bvecs from rotated_bvec or eddy_corrected directory
                for fname in ['bvals', 'bvecs']:
                    src_file = None
                    # Try rotated_bvec first
                    if (dwi_dir / 'rotated_bvec').exists():
                        candidates = list((dwi_dir / 'rotated_bvec').glob(f'*{fname}'))
                        if candidates:
                            src_file = candidates[0]
                    # Try eddy_corrected if not found
                    if src_file is None and (dwi_dir / 'eddy_corrected').exists():
                        candidates = list((dwi_dir / 'eddy_corrected').glob(f'*{fname}'))
                        if candidates:
                            src_file = candidates[0]

                    if src_file:
                        dest_file = dest_dir / fname
                        if not dest_file.exists():
                            shutil.copy2(src_file, dest_file)
                            copied.append(fname)

            if copied:
                logger.info(f"  ✓ Copied input files: {', '.join(copied)}")

            # Verify critical files are present
            critical_files = ['merged', 'nodif_brain_mask.nii.gz']
            missing = []
            for fname in critical_files:
                file_path = dest_dir / fname
                if not file_path.exists() and not (file_path.is_symlink()):
                    missing.append(fname)

            if missing:
                logger.warning(f"  ⚠ Missing critical files: {missing}")
                logger.warning(f"  ⚠ Probtrackx2 may not work correctly")
                return False

            logger.info(f"  ✓ Migration successful")
            return True

        except Exception as e:
            logger.error(f"  ✗ Failed to copy: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate BEDPOSTX outputs from work to derivatives",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Path to study root directory (e.g., /mnt/bytopia/IRC805)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to migrate (default: all subjects with BEDPOSTX)'
    )

    args = parser.parse_args()

    study_root = args.study_root
    work_dir = study_root / 'work'
    derivatives_dir = study_root / 'derivatives'

    if not work_dir.exists():
        logger.error(f"Work directory not found: {work_dir}")
        return 1

    if not derivatives_dir.exists():
        logger.error(f"Derivatives directory not found: {derivatives_dir}")
        return 1

    logger.info("=" * 80)
    logger.info("BEDPOSTX Migration to Derivatives")
    logger.info("=" * 80)
    logger.info(f"Study root: {study_root}")
    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Derivatives directory: {derivatives_dir}")
    if args.dry_run:
        logger.info("Mode: DRY RUN (no files will be copied)")
    logger.info("")

    # Find all BEDPOSTX directories in work
    logger.info("Searching for BEDPOSTX outputs in work directory...")
    bedpostx_dirs = find_bedpostx_in_work(work_dir)

    if not bedpostx_dirs:
        logger.info("No BEDPOSTX outputs found in work directory")
        return 0

    logger.info(f"Found {len(bedpostx_dirs)} BEDPOSTX outputs")
    logger.info("")

    # Filter by subjects if specified
    if args.subjects:
        bedpostx_dirs = [
            (subj, path) for subj, path in bedpostx_dirs
            if subj in args.subjects
        ]
        logger.info(f"Filtered to {len(bedpostx_dirs)} subjects: {args.subjects}")
        logger.info("")

    # Migrate each BEDPOSTX output
    logger.info("=" * 80)
    logger.info("Migrating BEDPOSTX Outputs")
    logger.info("=" * 80)
    logger.info("")

    successful = []
    failed = []

    for i, (subject, bedpostx_path) in enumerate(bedpostx_dirs, 1):
        logger.info(f"[{i}/{len(bedpostx_dirs)}] Processing {subject}")

        success = migrate_bedpostx(
            subject=subject,
            bedpostx_work_dir=bedpostx_path,
            derivatives_dir=derivatives_dir,
            dry_run=args.dry_run
        )

        if success:
            successful.append(subject)
        else:
            failed.append(subject)

        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("Migration Summary")
    logger.info("=" * 80)
    logger.info(f"Total subjects: {len(bedpostx_dirs)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info("")

    if successful:
        logger.info("Successfully migrated:")
        for subject in successful:
            logger.info(f"  ✓ {subject}")
        logger.info("")

    if failed:
        logger.info("Failed:")
        for subject in failed:
            logger.info(f"  ✗ {subject}")
        logger.info("")

    logger.info("=" * 80)

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
