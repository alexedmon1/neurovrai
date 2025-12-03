#!/usr/bin/env python3
"""
Standardize Atropos Tissue Segmentation Names (Version 2 - Enhanced Error Handling)

Identifies which Atropos POSTERIOR files correspond to which tissues (CSF, GM, WM)
and creates standardized symlinks or copies with consistent names.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import shutil

# Setup basic logging immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

try:
    import nibabel as nib
    import numpy as np
    logger.info("Successfully imported nibabel and numpy")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please install: pip install nibabel numpy")
    sys.exit(1)


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
        logger.info(f"    Loading T1w brain: {t1w_brain}")
        if not t1w_brain.exists():
            logger.error(f"    T1w brain file not found: {t1w_brain}")
            return None

        # Load T1w image
        t1w_img = nib.load(str(t1w_brain))
        t1w_data = t1w_img.get_fdata()
        logger.info(f"    T1w shape: {t1w_data.shape}")

        # Load all posteriors and calculate mean T1w intensity
        posteriors = {}
        post_files = list(seg_dir.glob('POSTERIOR_*.nii.gz'))
        logger.info(f"    Found {len(post_files)} POSTERIOR files")

        for post_file in sorted(post_files):
            logger.info(f"    Loading: {post_file.name}")
            post_img = nib.load(str(post_file))
            post_data = post_img.get_fdata()

            # Calculate weighted mean intensity
            masked_intensity = t1w_data * post_data
            mean_intensity = np.sum(masked_intensity) / (np.sum(post_data) + 1e-10)

            posteriors[post_file] = mean_intensity
            logger.info(f"      Mean intensity: {mean_intensity:.2f}")

        if len(posteriors) < 3:
            logger.warning(f"    Found only {len(posteriors)} posteriors, need 3")
            return None

        # Sort by mean intensity (CSF=lowest, GM=middle, WM=highest)
        sorted_posteriors = sorted(posteriors.items(), key=lambda x: x[1])

        tissue_map = {
            'csf': sorted_posteriors[0][0],  # Lowest intensity
            'gm': sorted_posteriors[1][0],   # Middle intensity
            'wm': sorted_posteriors[2][0]    # Highest intensity
        }

        # Log the identification
        logger.info("    Tissue identification:")
        for tissue, post_file in tissue_map.items():
            intensity = posteriors[post_file]
            logger.info(f"      {tissue.upper():3s} -> {post_file.name} (intensity: {intensity:.2f})")

        return tissue_map

    except Exception as e:
        logger.error(f"    Failed to identify tissues: {e}", exc_info=True)
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
    try:
        seg_dir = subject_dir / 'anat' / 'segmentation'
        logger.info(f"  Segmentation dir: {seg_dir}")

        if not seg_dir.exists():
            logger.warning(f"  Segmentation directory not found: {seg_dir}")
            return False

        # Check if already standardized
        if not overwrite:
            existing = [seg_dir / f'{tissue}.nii.gz' for tissue in ['csf', 'gm', 'wm']]
            if all(f.exists() for f in existing):
                logger.info(f"  Already standardized (use --overwrite to redo)")
                return True

        # Find T1w brain for intensity-based identification
        # Try multiple possible locations
        t1w_brain_candidates = [
            subject_dir / 'anat' / 'brain.nii.gz',
            subject_dir / 'anat' / 'T1w_brain.nii.gz',
            # Also check in brain subdirectory (Nipype DataSink structure)
        ]

        # Also check for files in brain/ subdirectory
        brain_dir = subject_dir / 'anat' / 'brain'
        if brain_dir.exists():
            brain_files = list(brain_dir.glob('*brain.nii.gz'))
            t1w_brain_candidates.extend(brain_files)

        # Use the first one that exists
        t1w_brain = None
        for candidate in t1w_brain_candidates:
            if candidate and candidate.exists():
                t1w_brain = candidate
                logger.info(f"  Found T1w brain: {t1w_brain.name}")
                break

        if t1w_brain is None:
            logger.warning(f"  T1w brain not found in any expected location")
            logger.warning(f"    Checked: anat/brain.nii.gz, anat/T1w_brain.nii.gz, anat/brain/*brain.nii.gz")
            return False

        # Identify tissues
        logger.info(f"  Identifying tissues...")
        tissue_map = identify_atropos_tissues(seg_dir, t1w_brain)

        if tissue_map is None:
            return False

        # Create standardized files
        success = True
        for tissue, source_file in tissue_map.items():
            target_file = seg_dir / f'{tissue}.nii.gz'

            # Remove existing file/link if overwriting
            if overwrite and target_file.exists():
                logger.info(f"  Removing existing: {target_file.name}")
                target_file.unlink()

            try:
                if use_symlinks:
                    # Create relative symlink
                    rel_source = source_file.relative_to(seg_dir)
                    target_file.symlink_to(rel_source)
                    logger.info(f"  ✓ Created symlink: {target_file.name} -> {source_file.name}")
                else:
                    # Copy file
                    shutil.copy2(str(source_file), str(target_file))
                    logger.info(f"  ✓ Copied: {source_file.name} -> {target_file.name}")

            except Exception as e:
                logger.error(f"  ✗ Failed to create {target_file.name}: {e}", exc_info=True)
                success = False

        return success

    except Exception as e:
        logger.error(f"  Exception in standardize_subject_tissues: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    try:
        parser = argparse.ArgumentParser(
            description='Standardize Atropos tissue segmentation names',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process all subjects (create symlinks)
  python standardize_atropos_tissues_v2.py --derivatives-dir /study/derivatives

  # Process single subject
  python standardize_atropos_tissues_v2.py --derivatives-dir /study/derivatives --subject sub-001

  # Copy files instead of symlinks
  python standardize_atropos_tissues_v2.py --derivatives-dir /study/derivatives --copy

  # Overwrite existing standardized files
  python standardize_atropos_tissues_v2.py --derivatives-dir /study/derivatives --overwrite
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

        # Update logging level if verbose
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        derivatives_dir = args.derivatives_dir.resolve()
        logger.info(f"Derivatives directory: {derivatives_dir}")

        if not derivatives_dir.exists():
            logger.error(f"Derivatives directory not found: {derivatives_dir}")
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("STANDARDIZING ATROPOS TISSUE SEGMENTATIONS")
        logger.info("=" * 80)
        logger.info(f"Derivatives: {derivatives_dir}")
        logger.info(f"Method: {'Copy' if args.copy else 'Symlink'}")
        logger.info("")

        # Find subjects to process
        if args.subject:
            subjects = [args.subject]
            logger.info(f"Processing single subject: {args.subject}")
        else:
            # Find all subject directories with segmentation data
            subject_dirs = []
            for d in derivatives_dir.iterdir():
                if d.is_dir():
                    seg_dir = d / 'anat' / 'segmentation'
                    if seg_dir.exists() and list(seg_dir.glob('POSTERIOR_*.nii.gz')):
                        subject_dirs.append(d.name)

            subjects = sorted(subject_dirs)
            logger.info(f"Found {len(subjects)} subjects with Atropos segmentation")

        if not subjects:
            logger.error("No subjects found with Atropos segmentation data")
            sys.exit(1)

        logger.info("")

        # Process subjects
        successful = 0
        failed = 0
        skipped = 0

        for subject in subjects:
            logger.info(f"Processing: {subject}")
            subject_dir = derivatives_dir / subject

            # Check if Atropos files exist
            seg_dir = subject_dir / 'anat' / 'segmentation'
            if not seg_dir.exists():
                logger.info(f"  No segmentation directory")
                skipped += 1
                continue

            posteriors = list(seg_dir.glob('POSTERIOR_*.nii.gz'))

            if not posteriors:
                logger.info(f"  No Atropos posteriors found (may be using FAST)")
                skipped += 1
                continue

            # Standardize
            if standardize_subject_tissues(subject_dir, use_symlinks=not args.copy, overwrite=args.overwrite):
                successful += 1
            else:
                failed += 1

            logger.info("")

        # Summary
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total subjects: {len(subjects)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Skipped: {skipped}")
        logger.info("=" * 80)

        if failed > 0:
            logger.error(f"{failed} subjects failed - check errors above")
            sys.exit(1)
        else:
            logger.info("All subjects processed successfully!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error in main(): {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
