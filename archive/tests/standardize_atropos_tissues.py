#!/usr/bin/env python3
"""
Standardize Atropos Tissue Segmentation Names

Identifies which Atropos POSTERIOR files correspond to which tissues (CSF, GM, WM)
and creates standardized symlinks or copies with consistent names.

Atropos with K-means initialization produces posteriors in arbitrary order.
This script identifies tissues based on mean T1w intensity and creates:
- csf.nii.gz -> POSTERIOR_XX.nii.gz (lowest intensity)
- gm.nii.gz -> POSTERIOR_XX.nii.gz (middle intensity)
- wm.nii.gz -> POSTERIOR_XX.nii.gz (highest intensity)

Usage:
    python standardize_atropos_tissues.py --derivatives-dir /study/derivatives
    python standardize_atropos_tissues.py --derivatives-dir /study/derivatives --copy  # Copy instead of symlink
    python standardize_atropos_tissues.py --derivatives-dir /study/derivatives --subject sub-001  # Single subject
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import shutil

import nibabel as nib
import numpy as np


def identify_atropos_tissues(seg_dir: Path, t1w_brain: Path) -> Optional[Dict[str, Path]]:
    """
    Identify which Atropos POSTERIOR corresponds to which tissue.

    Args:
        seg_dir: Segmentation directory with POSTERIOR_*.nii.gz files
        t1w_brain: T1w brain image for intensity-based identification

    Returns:
        Dictionary mapping tissue names to POSTERIOR files, or None if failed
    """
    try:
        # Load T1w image
        t1w_img = nib.load(t1w_brain)
        t1w_data = t1w_img.get_fdata()

        # Load all posteriors and calculate mean T1w intensity
        posteriors = {}
        for post_file in sorted(seg_dir.glob('POSTERIOR_*.nii.gz')):
            post_img = nib.load(post_file)
            post_data = post_img.get_fdata()

            # Calculate weighted mean intensity
            masked_intensity = t1w_data * post_data
            mean_intensity = np.sum(masked_intensity) / (np.sum(post_data) + 1e-10)

            posteriors[post_file] = mean_intensity

        if len(posteriors) < 3:
            logging.warning(f"Found only {len(posteriors)} posteriors, need 3")
            return None

        # Sort by mean intensity (CSF=lowest, GM=middle, WM=highest)
        sorted_posteriors = sorted(posteriors.items(), key=lambda x: x[1])

        tissue_map = {
            'csf': sorted_posteriors[0][0],  # Lowest intensity
            'gm': sorted_posteriors[1][0],   # Middle intensity
            'wm': sorted_posteriors[2][0]    # Highest intensity
        }

        # Log the identification
        for tissue, post_file in tissue_map.items():
            intensity = posteriors[post_file]
            logging.info(f"    {tissue.upper():3s} -> {post_file.name} (intensity: {intensity:.2f})")

        return tissue_map

    except Exception as e:
        logging.error(f"Failed to identify tissues: {e}")
        return None


def standardize_subject_tissues(
    subject_dir: Path,
    use_symlinks: bool = True,
    overwrite: bool = False
) -> bool:
    """
    Standardize tissue names for a single subject.

    Args:
        subject_dir: Subject directory (contains anat/segmentation/)
        use_symlinks: If True, create symlinks; if False, copy files
        overwrite: If True, overwrite existing standardized files

    Returns:
        True if successful, False otherwise
    """
    seg_dir = subject_dir / 'anat' / 'segmentation'
    if not seg_dir.exists():
        logging.warning(f"Segmentation directory not found: {seg_dir}")
        return False

    # Check if already standardized
    if not overwrite:
        existing = [seg_dir / f'{tissue}.nii.gz' for tissue in ['csf', 'gm', 'wm']]
        if all(f.exists() for f in existing):
            logging.info(f"  Already standardized (use --overwrite to redo)")
            return True

    # Find T1w brain for intensity-based identification
    # Try multiple possible locations
    t1w_brain_candidates = [
        subject_dir / 'anat' / 'brain.nii.gz',
        subject_dir / 'anat' / 'T1w_brain.nii.gz',
    ]

    # Also check for files in brain/ subdirectory (Nipype DataSink structure)
    brain_dir = subject_dir / 'anat' / 'brain'
    if brain_dir.exists():
        brain_files = list(brain_dir.glob('*brain.nii.gz'))
        t1w_brain_candidates.extend(brain_files)

    # Use the first one that exists
    t1w_brain = None
    for candidate in t1w_brain_candidates:
        if candidate and candidate.exists():
            t1w_brain = candidate
            logging.info(f"  Found T1w brain: {t1w_brain.name}")
            break

    if t1w_brain is None:
        logging.warning(f"  T1w brain not found in any expected location")
        return False

    # Identify tissues
    logging.info(f"  Identifying tissues...")
    tissue_map = identify_atropos_tissues(seg_dir, t1w_brain)

    if tissue_map is None:
        return False

    # Create standardized files
    success = True
    for tissue, source_file in tissue_map.items():
        target_file = seg_dir / f'{tissue}.nii.gz'

        # Remove existing file/link if overwriting
        if overwrite and target_file.exists():
            target_file.unlink()

        try:
            if use_symlinks:
                # Create relative symlink
                rel_source = source_file.relative_to(seg_dir)
                target_file.symlink_to(rel_source)
                logging.info(f"  ✓ Created symlink: {target_file.name} -> {source_file.name}")
            else:
                # Copy file
                shutil.copy2(source_file, target_file)
                logging.info(f"  ✓ Copied: {source_file.name} -> {target_file.name}")

        except Exception as e:
            logging.error(f"  ✗ Failed to create {target_file.name}: {e}")
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(
        description='Standardize Atropos tissue segmentation names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all subjects (create symlinks)
  python standardize_atropos_tissues.py --derivatives-dir /study/derivatives

  # Process single subject
  python standardize_atropos_tissues.py --derivatives-dir /study/derivatives --subject sub-001

  # Copy files instead of symlinks
  python standardize_atropos_tissues.py --derivatives-dir /study/derivatives --copy

  # Overwrite existing standardized files
  python standardize_atropos_tissues.py --derivatives-dir /study/derivatives --overwrite
        """
    )

    parser.add_argument(
        '--derivatives-dir',
        type=Path,
        required=True,
        help='Derivatives directory containing subject folders'
    )
    parser.add_argument(
        '--subject',
        type=str,
        help='Process single subject (e.g., sub-001 or IRC805-0580101)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of creating symlinks'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing standardized files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

    derivatives_dir = args.derivatives_dir.resolve()

    if not derivatives_dir.exists():
        logging.error(f"Derivatives directory not found: {derivatives_dir}")
        sys.exit(1)

    logging.info("=" * 80)
    logging.info("STANDARDIZING ATROPOS TISSUE SEGMENTATIONS")
    logging.info("=" * 80)
    logging.info(f"Derivatives: {derivatives_dir}")
    logging.info(f"Method: {'Copy' if args.copy else 'Symlink'}")
    logging.info("")

    # Find subjects to process
    if args.subject:
        subjects = [args.subject]
    else:
        # Find all subject directories
        subjects = sorted([d.name for d in derivatives_dir.iterdir()
                          if d.is_dir() and (d / 'anat' / 'segmentation').exists()])

    if not subjects:
        logging.error("No subjects found with segmentation data")
        sys.exit(1)

    logging.info(f"Found {len(subjects)} subjects to process\n")

    # Process subjects
    successful = 0
    failed = 0
    skipped = 0

    for subject in subjects:
        logging.info(f"Processing: {subject}")
        subject_dir = derivatives_dir / subject

        # Check if Atropos files exist
        seg_dir = subject_dir / 'anat' / 'segmentation'
        posteriors = list(seg_dir.glob('POSTERIOR_*.nii.gz'))

        if not posteriors:
            logging.info(f"  No Atropos posteriors found (may be using FAST)")
            skipped += 1
            continue

        # Standardize
        if standardize_subject_tissues(subject_dir, use_symlinks=not args.copy, overwrite=args.overwrite):
            successful += 1
        else:
            failed += 1

        logging.info("")

    # Summary
    logging.info("=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total subjects: {len(subjects)}")
    logging.info(f"  Successful: {successful}")
    logging.info(f"  Failed: {failed}")
    logging.info(f"  Skipped: {skipped}")
    logging.info("=" * 80)

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
