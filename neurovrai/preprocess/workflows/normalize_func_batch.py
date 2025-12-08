#!/usr/bin/env python3
"""
Batch normalize preprocessed functional data to MNI152 space

Uses existing BBR and anatomical transforms to normalize functional data.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

from neurovrai.preprocess.utils.func_normalization import normalize_func_to_mni152


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"normalize_func_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


def find_subjects(study_root: Path) -> list:
    """Find subjects with preprocessed functional data"""
    derivatives = study_root / "derivatives"
    subjects = []

    for func_file in derivatives.glob("*/func/*_bold_preprocessed.nii.gz"):
        subject = func_file.parts[-3]
        subjects.append(subject)

    return sorted(list(set(subjects)))


def normalize_subject(
    subject: str,
    study_root: Path,
    logger: logging.Logger
) -> dict:
    """Normalize one subject's functional data to MNI"""

    logger.info(f"\n{'='*80}")
    logger.info(f"Subject: {subject}")
    logger.info(f"{'='*80}")

    try:
        # Setup paths
        deriv_dir = study_root / "derivatives" / subject
        func_dir = deriv_dir / "func"
        anat_dir = deriv_dir / "anat"

        # Find preprocessed functional data
        func_file = func_dir / f"{subject}_bold_preprocessed.nii.gz"
        if not func_file.exists():
            raise FileNotFoundError(f"Preprocessed functional data not found: {func_file}")

        logger.info(f"Functional data: {func_file}")

        # Find BBR transform (from ACompCor)
        bbr_mat = func_dir / "transforms" / "func_to_anat_bbr.mat"
        if not bbr_mat.exists():
            # Try alternative location
            bbr_mat = func_dir / "func_to_anat_bbr.mat"
            if not bbr_mat.exists():
                raise FileNotFoundError(
                    f"BBR transform not found. Expected at:\n"
                    f"  {func_dir / 'transforms' / 'func_to_anat_bbr.mat'}\n"
                    f"  or {func_dir / 'func_to_anat_bbr.mat'}"
                )

        logger.info(f"BBR transform: {bbr_mat}")

        # Find anatomical transforms
        anat_to_mni_affine = anat_dir / "transforms" / "T1w_to_MNI152_affine.mat"
        anat_to_mni_warp = anat_dir / "transforms" / "T1w_to_MNI152_warp.nii.gz"

        if not anat_to_mni_affine.exists():
            raise FileNotFoundError(f"Anatomical affine transform not found: {anat_to_mni_affine}")
        if not anat_to_mni_warp.exists():
            raise FileNotFoundError(f"Anatomical warp not found: {anat_to_mni_warp}")

        logger.info(f"Anatomical affine: {anat_to_mni_affine}")
        logger.info(f"Anatomical warp: {anat_to_mni_warp}")

        # Normalize to MNI
        logger.info("\nNormalizing to MNI152...")
        results = normalize_func_to_mni152(
            func_file=func_file,
            func_to_anat_bbr=bbr_mat,
            t1w_to_mni_affine=anat_to_mni_affine,
            t1w_to_mni_warp=anat_to_mni_warp,
            output_dir=func_dir,
            interpolation='spline',
            registration_method='fsl'
        )

        logger.info(f"✓ Success! Normalized data: {results['func_normalized']}")

        return {
            'subject': subject,
            'status': 'success',
            'func_normalized': str(results['func_normalized'])
        }

    except Exception as e:
        logger.error(f"✗ Failed: {str(e)}", exc_info=True)
        return {
            'subject': subject,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Batch normalize preprocessed functional data to MNI152"
    )

    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Study root directory'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to process (default: all)'
    )

    parser.add_argument(
        '--output-log-dir',
        type=Path,
        help='Log directory (default: {study-root}/analysis/func/normalization/logs)'
    )

    args = parser.parse_args()

    # Setup logging
    if args.output_log_dir:
        log_dir = args.output_log_dir
    else:
        log_dir = args.study_root / "analysis" / "func" / "normalization"

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir)

    logger.info("="*80)
    logger.info("BATCH FUNCTIONAL MNI NORMALIZATION")
    logger.info("="*80)
    logger.info(f"Study root: {args.study_root}")

    # Find subjects
    if args.subjects:
        subjects = args.subjects
        logger.info(f"Processing {len(subjects)} specified subjects")
    else:
        subjects = find_subjects(args.study_root)
        logger.info(f"Found {len(subjects)} subjects with preprocessed functional data")

    logger.info(f"Subjects: {', '.join(subjects)}")
    logger.info("="*80)

    # Process subjects
    results = []
    successful = 0
    failed = 0

    for i, subject in enumerate(subjects, 1):
        logger.info(f"\n[{i}/{len(subjects)}] Processing {subject}...")

        result = normalize_subject(subject, args.study_root, logger)
        results.append(result)

        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1

    # Save summary
    summary_file = log_dir / 'normalization_summary.json'
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'study_root': str(args.study_root),
        'total': len(subjects),
        'successful': successful,
        'failed': failed,
        'results': results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("BATCH NORMALIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total subjects: {len(subjects)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(subjects)*100:.1f}%")
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info("="*80)

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
