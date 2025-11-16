#!/usr/bin/env python3
"""
Simple batch processor - runs all IRC805 subjects through complete pipeline.

Uses the existing run_full_pipeline.py for each subject.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration
STUDY_ROOT = Path('/mnt/bytopia/IRC805')
DICOM_ROOT = STUDY_ROOT / 'raw' / 'dicom'
CONFIG_FILE = Path('config.yaml')

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = Path('logs') / f'batch_{timestamp}.log'
log_file.parent.mkdir(exist_ok=True)

def log_and_print(message):
    """Log to file and print to console."""
    print(message)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {message}\n")

def find_subjects():
    """Find all subjects with DICOM data."""
    if not DICOM_ROOT.exists():
        log_and_print(f"ERROR: DICOM directory not found: {DICOM_ROOT}")
        sys.exit(1)

    subjects = sorted([d.name for d in DICOM_ROOT.glob('IRC805-*') if d.is_dir()])
    return subjects

def run_subject(subject: str) -> bool:
    """Run complete pipeline for one subject."""
    log_and_print(f"\n{'='*70}")
    log_and_print(f"PROCESSING: {subject}")
    log_and_print(f"{'='*70}")

    dicom_dir = DICOM_ROOT / subject

    # Run the full pipeline
    cmd = [
        'uv', 'run', 'python', 'run_full_pipeline.py',
        '--subject', subject,
        '--dicom-dir', str(dicom_dir),
        '--config', str(CONFIG_FILE)
    ]

    log_and_print(f"Running: {' '.join(cmd)}")
    log_and_print(f"\n--- Pipeline output for {subject} ---")

    try:
        # Run with real-time output streaming (no capture_output)
        # This lets you see detailed progress messages
        result = subprocess.run(
            cmd,
            timeout=36000  # 10 hour timeout per subject
        )

        if result.returncode == 0:
            log_and_print(f"\n✓ SUCCESS: {subject}")
            return True
        else:
            log_and_print(f"\n✗ FAILED: {subject} (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        log_and_print(f"\n✗ TIMEOUT: {subject} (exceeded 10 hours)")
        return False
    except Exception as e:
        log_and_print(f"\n✗ ERROR: {subject} - {e}")
        return False

def main():
    """Main batch processing."""
    log_and_print("="*70)
    log_and_print("IRC805 BATCH PROCESSING")
    log_and_print("="*70)
    log_and_print(f"Study root: {STUDY_ROOT}")
    log_and_print(f"Config: {CONFIG_FILE}")
    log_and_print(f"Log file: {log_file}")
    log_and_print("")

    # Validate config exists
    if not CONFIG_FILE.exists():
        log_and_print(f"ERROR: Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    # Find subjects
    subjects = find_subjects()
    log_and_print(f"Found {len(subjects)} subjects")
    log_and_print("")

    # Process each subject
    results = {}
    for i, subject in enumerate(subjects, 1):
        log_and_print(f"\n[{i}/{len(subjects)}] Starting {subject}...")
        success = run_subject(subject)
        results[subject] = 'success' if success else 'failed'

    # Summary
    log_and_print("\n" + "="*70)
    log_and_print("BATCH PROCESSING COMPLETE")
    log_and_print("="*70)

    success_count = sum(1 for v in results.values() if v == 'success')
    failed_count = len(results) - success_count

    log_and_print(f"Total subjects: {len(subjects)}")
    log_and_print(f"✓ Successful: {success_count}")
    log_and_print(f"✗ Failed: {failed_count}")
    log_and_print("")

    if failed_count > 0:
        log_and_print("Failed subjects:")
        for subject, status in results.items():
            if status == 'failed':
                log_and_print(f"  - {subject}")

    log_and_print(f"\nFull log: {log_file}")
    log_and_print("="*70)

if __name__ == '__main__':
    main()
