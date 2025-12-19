#!/usr/bin/env python3
"""
Normalize preprocessed 4D BOLD data to MNI space.

Uses the standardized transforms from {study_root}/transforms/{subject}/func-mni-composite.h5
to transform preprocessed BOLD data to MNI152 2mm space.

Output: {derivatives}/{subject}/func/{subject}_bold_mni.nii.gz
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


def normalize_bold_to_mni(
    bold_file: Path,
    transform_file: Path,
    reference_file: Path,
    output_file: Path,
    n_threads: int = 4
) -> bool:
    """
    Apply composite transform to normalize BOLD to MNI space.

    Uses antsApplyTransforms with appropriate interpolation for 4D data.
    """

    # For 4D data, we need to use -e 3 (time series) or process volume by volume
    # antsApplyTransforms handles 4D data with -e 3
    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-e', '3',  # Time series (4D)
        '-i', str(bold_file),
        '-r', str(reference_file),
        '-t', str(transform_file),
        '-o', str(output_file),
        '-n', 'LanczosWindowedSinc',  # Best for functional data
        '-v', '1'
    ]

    print(f"  Running: {' '.join(cmd[:6])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout for large 4D files
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("  ERROR: Transform timed out (>30 min)")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def process_subject(
    subject: str,
    study_root: Path,
    reference_file: Path,
    force: bool = False
) -> dict:
    """Process one subject's BOLD normalization."""

    result = {
        'subject': subject,
        'status': 'unknown',
        'output': None,
        'error': None
    }

    # Find input files
    bold_file = study_root / 'derivatives' / subject / 'func' / f'{subject}_bold_preprocessed.nii.gz'
    transform_file = study_root / 'transforms' / subject / 'func-mni-composite.h5'
    output_file = study_root / 'derivatives' / subject / 'func' / f'{subject}_bold_mni.nii.gz'

    # Check if output exists
    if output_file.exists() and not force:
        print(f"  {subject}: Output exists, skipping (use --force to reprocess)")
        result['status'] = 'skipped'
        result['output'] = str(output_file)
        return result

    # Check inputs
    if not bold_file.exists():
        print(f"  {subject}: BOLD file not found: {bold_file}")
        result['status'] = 'error'
        result['error'] = 'BOLD file not found'
        return result

    if not transform_file.exists():
        print(f"  {subject}: Transform not found: {transform_file}")
        result['status'] = 'error'
        result['error'] = 'Transform not found'
        return result

    print(f"  {subject}: Normalizing to MNI...")

    # Run normalization
    success = normalize_bold_to_mni(
        bold_file=bold_file,
        transform_file=transform_file,
        reference_file=reference_file,
        output_file=output_file
    )

    if success:
        result['status'] = 'success'
        result['output'] = str(output_file)
        print(f"  {subject}: Done -> {output_file.name}")
    else:
        result['status'] = 'error'
        result['error'] = 'Transform failed'

    return result


def find_subjects_with_bold(study_root: Path) -> list:
    """Find all subjects with preprocessed BOLD data."""
    subjects = []
    derivatives = study_root / 'derivatives'

    for subj_dir in sorted(derivatives.glob('IRC805-*')):
        bold_file = subj_dir / 'func' / f'{subj_dir.name}_bold_preprocessed.nii.gz'
        if bold_file.exists():
            subjects.append(subj_dir.name)

    return subjects


def main():
    parser = argparse.ArgumentParser(
        description='Normalize preprocessed BOLD to MNI space',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help='Specific subjects to process (default: all with BOLD data)'
    )

    parser.add_argument(
        '--reference',
        type=Path,
        default=Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'),
        help='MNI reference image (default: FSL MNI152 2mm)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess even if output exists'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of subjects to process in parallel'
    )

    args = parser.parse_args()

    # Find subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = find_subjects_with_bold(args.study_root)

    print(f"{'='*60}")
    print("BOLD to MNI Normalization")
    print(f"{'='*60}")
    print(f"Study root: {args.study_root}")
    print(f"Reference: {args.reference}")
    print(f"Subjects: {len(subjects)}")
    print(f"{'='*60}")

    # Process subjects
    results = []
    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {subject}")
        result = process_subject(
            subject=subject,
            study_root=args.study_root,
            reference_file=args.reference,
            force=args.force
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    if errors > 0:
        print("\nSubjects with errors:")
        for r in results:
            if r['status'] == 'error':
                print(f"  {r['subject']}: {r['error']}")

    # Save log
    log_file = args.study_root / 'logs' / f'bold_mni_normalization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    log_file.parent.mkdir(exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'study_root': str(args.study_root),
            'reference': str(args.reference),
            'results': results
        }, f, indent=2)

    print(f"\nLog saved: {log_file}")

    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
