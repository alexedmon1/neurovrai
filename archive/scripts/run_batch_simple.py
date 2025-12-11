#!/usr/bin/env python3
"""
Simple Batch Processing Script

Process multiple subjects sequentially using the simple pipeline.

Usage:
    python run_batch_simple.py --config config.yaml

This script will:
1. Find all subjects with DICOM data
2. Process each subject sequentially
3. Track which subjects succeed/fail
4. Continue processing even if some subjects fail
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from neurovrai.config import load_config


def find_subjects(dicom_root):
    """Find all subjects with DICOM directories."""
    subjects = []
    for subject_dir in sorted(dicom_root.glob('IRC805-*')):
        if subject_dir.is_dir():
            subjects.append(subject_dir.name)
    return subjects


def run_subject(subject, dicom_dir, config_file):
    """Run pipeline for a single subject."""
    print("="*70)
    print(f"PROCESSING: {subject}")
    print("="*70)

    cmd = [
        'uv', 'run', 'python', 'run_simple_pipeline.py',
        '--subject', subject,
        '--dicom-dir', str(dicom_dir),
        '--config', str(config_file)
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ SUCCESS: {subject}\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ FAILED: {subject} (exit code {e.returncode})\n")
        return False


def main():
    """Main batch processing."""

    parser = argparse.ArgumentParser(description='Batch process multiple subjects')
    parser.add_argument('--config', type=Path, default=Path('config.yaml'), help='Config file')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to process (optional)')
    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    study_root = Path(config['project_dir'])
    dicom_root = study_root / 'raw' / 'dicom'

    if not dicom_root.exists():
        print(f"ERROR: DICOM directory not found: {dicom_root}")
        sys.exit(1)

    # Find subjects
    if args.subjects:
        subjects = args.subjects
        print(f"Processing {len(subjects)} specified subjects")
    else:
        subjects = find_subjects(dicom_root)
        print(f"Found {len(subjects)} subjects in {dicom_root}")

    print(f"Study root: {study_root}")
    print()

    # Track results
    results = {
        'success': [],
        'failed': []
    }

    start_time = datetime.now()

    # Process each subject
    for i, subject in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {subject}")

        dicom_dir = dicom_root / subject
        if not dicom_dir.exists():
            print(f"✗ DICOM directory not found: {dicom_dir}")
            results['failed'].append(subject)
            continue

        success = run_subject(subject, dicom_dir, args.config)

        if success:
            results['success'].append(subject)
        else:
            results['failed'].append(subject)

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total subjects: {len(subjects)}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Duration: {duration}")
    print()

    if results['failed']:
        print("Failed subjects:")
        for subject in results['failed']:
            print(f"  - {subject}")
        print()

    print("="*70)

    # Exit with error code if any failed
    sys.exit(0 if not results['failed'] else 1)


if __name__ == '__main__':
    main()
