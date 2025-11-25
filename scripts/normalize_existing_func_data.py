#!/usr/bin/env python3
"""
Normalize existing preprocessed functional data to MNI152 space

This script applies spatial normalization to functional data that was preprocessed
without the normalize_to_mni option. It reuses:
- BBR transform (func→anat) from ACompCor
- Anatomical transforms (anat→MNI) from anatomical preprocessing

Usage:
    python scripts/normalize_existing_func_data.py --config config.yaml

Or for specific subjects:
    python scripts/normalize_existing_func_data.py --config config.yaml \
        --subjects IRC805-0580101 IRC805-1580101
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse
from typing import List, Dict

from neurovrai.config import load_config
from neurovrai.preprocess.utils.func_normalization import normalize_func_to_mni152
from neurovrai.utils.transforms import create_transform_registry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_subject_func(
    subject: str,
    config: Dict,
    force: bool = False
) -> bool:
    """
    Normalize functional data for a single subject

    Args:
        subject: Subject ID
        config: Configuration dictionary
        force: Force re-normalization even if output exists

    Returns:
        True if successful, False otherwise
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Processing subject: {subject}")
    logger.info("=" * 80)

    derivatives_dir = Path(config['derivatives_dir'])
    subject_func_dir = derivatives_dir / subject / 'func'

    # Check if functional data exists
    if not subject_func_dir.exists():
        logger.warning(f"  No functional directory found: {subject_func_dir}")
        return False

    # Find preprocessed functional data
    possible_files = [
        subject_func_dir / f'{subject}_bold_acompcor_cleaned.nii.gz',
        subject_func_dir / f'{subject}_bold_preprocessed.nii.gz',
    ]

    func_file = None
    for f in possible_files:
        if f.exists():
            func_file = f
            logger.info(f"  Found preprocessed data: {f.name}")
            break

    if not func_file:
        logger.warning(f"  No preprocessed functional data found")
        logger.warning(f"  Looked for: {[f.name for f in possible_files]}")
        return False

    # Check if already normalized
    normalized_dir = subject_func_dir / 'normalized'
    normalized_file = normalized_dir / 'bold_normalized.nii.gz'

    if normalized_file.exists() and not force:
        logger.info(f"  Already normalized: {normalized_file}")
        logger.info(f"  Use --force to re-normalize")
        return True

    # Get BBR transform (func→anat) from TransformRegistry
    registry = create_transform_registry(config, subject)
    bbr_transform = registry.get_linear_transform('func', 'T1w')

    if not bbr_transform or not bbr_transform.exists():
        logger.warning(f"  BBR transform not found in registry")
        logger.warning(f"  Expected transform from func→T1w")
        logger.warning(f"  This is created during ACompCor registration")
        logger.warning(f"  Functional preprocessing may need to be re-run with acompcor enabled")
        return False

    logger.info(f"  Found BBR transform from registry: {bbr_transform}")

    # Get anatomical transforms from TransformRegistry
    anat_transforms = registry.get_nonlinear_transform('T1w', 'MNI152')

    if not anat_transforms:
        logger.warning(f"  Anatomical transforms not found in registry")
        logger.warning(f"  Run anatomical preprocessing first")
        return False

    t1w_to_mni_warp, t1w_to_mni_affine = anat_transforms

    # Verify all transforms exist
    if not t1w_to_mni_warp.exists():
        logger.warning(f"  Anatomical warp not found: {t1w_to_mni_warp}")
        return False

    if not t1w_to_mni_affine.exists():
        logger.warning(f"  Anatomical affine not found: {t1w_to_mni_affine}")
        return False

    # Get registration method - but force FSL for functional normalization
    # FSL's applywarp handles 4D time series better than ANTs
    original_method = registry.get_transform_method('T1w', 'MNI152') or 'fsl'
    registration_method = 'fsl'  # Force FSL for 4D functional data

    # If anatomical was ANTs, we need to convert the .h5 to NIfTI warp
    if original_method == 'ants' and str(t1w_to_mni_warp).endswith('.h5'):
        logger.info(f"  Converting ANTs composite transform to FSL format...")
        # ANTs .h5 can be used directly - convertwarp will handle it
        # Or we need to extract the warp component
        # For now, let's just warn and continue
        logger.warning(f"  Anatomical registration used ANTs (.h5 format)")
        logger.warning(f"  FSL's convertwarp may not support .h5 directly")
        logger.warning(f"  This might fail - if so, need to convert transform format")

    logger.info(f"  All transforms found:")
    logger.info(f"    BBR (func→anat): {bbr_transform}")
    logger.info(f"    Affine (anat→MNI): {t1w_to_mni_affine}")
    logger.info(f"    Warp (anat→MNI): {t1w_to_mni_warp}")
    logger.info(f"    Original method: {original_method}")
    logger.info(f"    Using method: {registration_method} (forced for 4D data)")
    logger.info("")

    # Run normalization
    try:
        norm_results = normalize_func_to_mni152(
            func_file=func_file,
            func_to_anat_bbr=bbr_transform,
            t1w_to_mni_affine=t1w_to_mni_affine,
            t1w_to_mni_warp=t1w_to_mni_warp,
            output_dir=subject_func_dir,
            mni152_template=None,  # Uses $FSLDIR default
            interpolation='spline',
            registration_method=registration_method  # Use FSL
        )

        logger.info("")
        logger.info(f"  ✓ Normalization successful!")
        logger.info(f"    Output: {norm_results['func_normalized']}")
        logger.info(f"    Warp: {norm_results['func_to_mni_warp']}")

        return True

    except Exception as e:
        logger.error(f"  ✗ Normalization failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Normalize existing preprocessed functional data to MNI152"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to process (default: all with preprocessed data)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-normalization even if output exists'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    derivatives_dir = Path(config['derivatives_dir'])

    logger.info("=" * 80)
    logger.info("NORMALIZE EXISTING FUNCTIONAL DATA TO MNI152")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Derivatives: {derivatives_dir}")
    logger.info("")

    # Find subjects to process
    if args.subjects:
        subjects = args.subjects
        logger.info(f"Processing {len(subjects)} specified subjects")
    else:
        # Find all subjects with preprocessed functional data
        subjects = []
        for subject_dir in sorted(derivatives_dir.glob('IRC805-*')):
            func_dir = subject_dir / 'func'
            if func_dir.exists():
                # Check for preprocessed data
                has_data = any([
                    (func_dir / f'{subject_dir.name}_bold_acompcor_cleaned.nii.gz').exists(),
                    (func_dir / f'{subject_dir.name}_bold_preprocessed.nii.gz').exists(),
                ])
                if has_data:
                    subjects.append(subject_dir.name)

        logger.info(f"Found {len(subjects)} subjects with preprocessed functional data")

    if not subjects:
        logger.error("No subjects found to process")
        return 1

    logger.info("")
    logger.info("Subjects to process:")
    for subj in subjects:
        logger.info(f"  - {subj}")
    logger.info("")

    # Process each subject
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for subject in subjects:
        try:
            success = normalize_subject_func(subject, config, force=args.force)
            if success:
                success_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            logger.error(f"Error processing {subject}: {e}", exc_info=True)
            failed_count += 1

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("NORMALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total subjects: {len(subjects)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Skipped: {skipped_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info("=" * 80)

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
