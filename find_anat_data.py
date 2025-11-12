#!/usr/bin/env python3
"""
Helper script to find available anatomical preprocessing data.

Searches for preprocessed anatomical data and reports what QC modules can be run.
"""

import sys
from pathlib import Path
import argparse


def find_subjects_with_anat_data(study_root: Path):
    """
    Find all subjects with anatomical preprocessing data.

    Parameters
    ----------
    study_root : Path
        Study root directory

    Returns
    -------
    dict
        Subject -> data availability mapping
    """
    subjects = {}

    # Search standard locations
    search_locations = [
        study_root / 'derivatives' / 'anat_preproc',
        study_root / 'derivatives' / 'anatomical',
        study_root / 'subjects',
    ]

    for location in search_locations:
        if not location.exists():
            continue

        # Find subject directories
        for subj_dir in location.iterdir():
            if not subj_dir.is_dir():
                continue

            subject = subj_dir.name

            # Look for anatomical files
            anat_files = {
                'has_t1w': False,
                'has_brain': False,
                'has_mask': False,
                'has_segmentation': False,
                'has_registered': False,
                'anat_dir': subj_dir
            }

            # Check for files
            all_files = list(subj_dir.rglob('*.nii.gz'))

            for f in all_files:
                fname_lower = f.name.lower()

                if 't1w' in fname_lower or 't1' in fname_lower:
                    if 'brain' not in fname_lower and 'mask' not in fname_lower:
                        anat_files['has_t1w'] = True

                if 'brain' in fname_lower and 'mask' not in fname_lower:
                    anat_files['has_brain'] = True

                if 'mask' in fname_lower:
                    anat_files['has_mask'] = True

                if 'pve' in fname_lower or ('seg' in fname_lower and any(x in fname_lower for x in ['csf', 'gm', 'wm'])):
                    anat_files['has_segmentation'] = True

                if 'mni' in fname_lower or 'std' in fname_lower or 'warped' in fname_lower:
                    anat_files['has_registered'] = True

            # Only add if at least one file found
            if any([anat_files[k] for k in anat_files if k.startswith('has_')]):
                if subject not in subjects:
                    subjects[subject] = anat_files
                else:
                    # Merge with existing
                    for key in anat_files:
                        if key.startswith('has_'):
                            subjects[subject][key] = subjects[subject][key] or anat_files[key]

    return subjects


def print_summary(subjects: dict):
    """Print summary of available data."""
    if not subjects:
        print("No anatomical preprocessing data found.")
        return

    print(f"\nFound {len(subjects)} subject(s) with anatomical data:\n")
    print("=" * 100)
    print(f"{'Subject':<30} {'Skull Strip QC':<20} {'Segmentation QC':<20} {'Registration QC':<20}")
    print("=" * 100)

    for subject, data in sorted(subjects.items()):
        # Determine which QC modules can run
        skull_strip_ok = "✓" if data['has_mask'] else "✗"
        segmentation_ok = "✓" if data['has_segmentation'] else "✗"
        registration_ok = "✓" if data['has_registered'] else "✗"

        print(f"{subject:<30} {skull_strip_ok:<20} {segmentation_ok:<20} {registration_ok:<20}")

    print("=" * 100)
    print("\nLegend:")
    print("  ✓ = QC module can run (required files found)")
    print("  ✗ = QC module cannot run (missing required files)")
    print("\nFile requirements:")
    print("  - Skull Strip QC: brain mask (*mask.nii.gz)")
    print("  - Segmentation QC: tissue segmentations (*pve_*.nii.gz)")
    print("  - Registration QC: registered image (*MNI152*.nii.gz)")
    print()


def main():
    """Find available anatomical data."""
    parser = argparse.ArgumentParser(description='Find available anatomical preprocessing data')
    parser.add_argument('study_root', type=Path, help='Study root directory')
    parser.add_argument('--detail', action='store_true', help='Show detailed file listing')

    args = parser.parse_args()

    study_root = Path(args.study_root)

    if not study_root.exists():
        print(f"Error: Study root does not exist: {study_root}")
        return 1

    print(f"Searching for anatomical data in: {study_root}")

    subjects = find_subjects_with_anat_data(study_root)

    if args.detail and subjects:
        print("\n" + "=" * 100)
        print("DETAILED FILE LISTING")
        print("=" * 100)
        for subject, data in sorted(subjects.items()):
            print(f"\n{subject}:")
            print(f"  Directory: {data['anat_dir']}")
            print(f"  Files:")
            for f in sorted(data['anat_dir'].rglob('*.nii.gz')):
                rel_path = f.relative_to(data['anat_dir'])
                print(f"    - {rel_path}")

    print_summary(subjects)

    # Print example command
    if subjects:
        first_subject = list(subjects.keys())[0]
        print("\nExample QC command:")
        print(f"  python test_anat_qc_complete.py --subject {first_subject} --study-root {study_root}")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
