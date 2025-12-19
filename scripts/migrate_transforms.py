#!/usr/bin/env python3
"""
Migrate and standardize transforms to centralized location.

Naming convention: {source}-{target}-{type}.{ext}
Location: {study_root}/transforms/{subject}/

Examples:
- func-t1w-affine.mat
- t1w-mni-composite.h5
- dwi-fmrib58-warp.nii.gz
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse


# Mapping from old names to new standardized names
TRANSFORM_MAPPINGS = {
    # Central transforms folder
    'ASL_to_T1w_affine.mat': 'asl-t1w-affine.mat',
    'func_to_T1w_affine.mat': 'func-t1w-affine.mat',
    'T1w_to_MNI152_composite.h5': 't1w-mni-composite.h5',

    # Derivatives func registration
    'func_to_mni_Composite.h5': 'func-mni-composite.h5',
    'func_to_t1w0GenericAffine.mat': 'func-t1w-affine.mat',  # May duplicate
    'func_to_t1wInverseWarped.nii.gz': 't1w-func-warped.nii.gz',  # Reference image
    'func_to_t1wWarped.nii.gz': 'func-t1w-warped.nii.gz',  # Reference image

    # Derivatives anat transforms
    'ants_Composite.h5': 't1w-mni-composite.h5',  # May duplicate

    # Derivatives DWI transforms
    'FA_to_FMRIB58_affine.mat': 'dwi-fmrib58-affine.mat',
    'FA_to_FMRIB58_affine.nii.gz': 'dwi-fmrib58-affine-ref.nii.gz',
    'FA_to_FMRIB58_warp.nii.gz': 'dwi-fmrib58-warp.nii.gz',
    'FMRIB58_to_FA_warp.nii.gz': 'fmrib58-dwi-warp.nii.gz',
}


def find_transforms(subject_dir: Path) -> dict:
    """Find all transforms for a subject across all locations."""
    transforms = {}

    # Check central transforms folder
    central = subject_dir
    if central.exists():
        for f in central.glob('*'):
            if f.suffix in ['.mat', '.h5', '.nii.gz', '.lta', '.txt'] or f.name.endswith('.nii.gz'):
                transforms[f.name] = {'path': f, 'location': 'central'}

    # Check derivatives locations
    deriv_base = subject_dir.parent.parent / 'derivatives' / subject_dir.name

    # Func registration
    func_reg = deriv_base / 'func' / 'registration'
    if func_reg.exists():
        for f in func_reg.glob('*'):
            if f.suffix in ['.mat', '.h5', '.txt'] or f.name.endswith('.nii.gz'):
                if f.name not in ['func_mean.nii.gz']:  # Skip non-transform files
                    transforms[f.name] = {'path': f, 'location': 'derivatives/func'}

    # Anat transforms
    anat_xfm = deriv_base / 'anat' / 'transforms'
    if anat_xfm.exists():
        for f in anat_xfm.glob('*'):
            if f.suffix in ['.mat', '.h5', '.txt'] or f.name.endswith('.nii.gz'):
                transforms[f.name] = {'path': f, 'location': 'derivatives/anat'}

    # DWI transforms
    dwi_xfm = deriv_base / 'dwi' / 'transforms'
    if dwi_xfm.exists():
        for f in dwi_xfm.glob('*'):
            if f.suffix in ['.mat', '.h5', '.txt'] or f.name.endswith('.nii.gz'):
                transforms[f.name] = {'path': f, 'location': 'derivatives/dwi'}

    return transforms


def migrate_subject(subject_dir: Path, dry_run: bool = True) -> dict:
    """Migrate transforms for one subject to standardized names."""
    subject = subject_dir.name
    transforms = find_transforms(subject_dir)

    migration_log = {
        'subject': subject,
        'migrated': [],
        'skipped': [],
        'errors': []
    }

    print(f"\n{'='*60}")
    print(f"Subject: {subject}")
    print(f"{'='*60}")

    for old_name, info in transforms.items():
        new_name = TRANSFORM_MAPPINGS.get(old_name)

        if new_name is None:
            # Unknown transform, keep original name but note it
            if old_name != 'transforms.json' and old_name != 'transform_list.txt':
                print(f"  ? {old_name} - no mapping defined")
                migration_log['skipped'].append({
                    'file': old_name,
                    'reason': 'no mapping defined'
                })
            continue

        source_path = info['path']
        target_path = subject_dir / new_name

        if info['location'] == 'central' and source_path.name == new_name:
            # Already correctly named
            print(f"  ✓ {old_name} - already correct")
            continue

        if target_path.exists() and info['location'] != 'central':
            # Target already exists from central folder
            print(f"  ~ {old_name} ({info['location']}) -> skipped (already in central)")
            migration_log['skipped'].append({
                'file': old_name,
                'reason': 'already exists in central'
            })
            continue

        print(f"  → {old_name} ({info['location']}) -> {new_name}")

        if not dry_run:
            try:
                if info['location'] == 'central':
                    # Rename in place
                    source_path.rename(target_path)
                else:
                    # Copy from derivatives to central
                    shutil.copy2(source_path, target_path)

                migration_log['migrated'].append({
                    'old': old_name,
                    'new': new_name,
                    'source_location': info['location']
                })
            except Exception as e:
                print(f"    ERROR: {e}")
                migration_log['errors'].append({
                    'file': old_name,
                    'error': str(e)
                })

    return migration_log


def update_transforms_json(subject_dir: Path, dry_run: bool = True):
    """Update transforms.json with all available transforms."""
    json_path = subject_dir / 'transforms.json'

    # Find all transforms
    transforms = {}
    for f in subject_dir.glob('*'):
        if f.name == 'transforms.json':
            continue
        if f.suffix in ['.mat', '.h5', '.txt', '.lta'] or f.name.endswith('.nii.gz'):
            # Parse source-target-type from filename
            parts = f.stem.replace('.nii', '').split('-')
            if len(parts) >= 3:
                source, target, xfm_type = parts[0], parts[1], '-'.join(parts[2:])
                transforms[f.name] = {
                    'source_space': source,
                    'target_space': target,
                    'type': xfm_type,
                    'file': f.name
                }

    metadata = {
        'subject': subject_dir.name,
        'updated': datetime.now().isoformat(),
        'transforms': transforms
    }

    if not dry_run:
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Updated transforms.json with {len(transforms)} transforms")
    else:
        print(f"  Would update transforms.json with {len(transforms)} transforms")


def main():
    parser = argparse.ArgumentParser(description='Migrate transforms to standardized names')
    parser.add_argument('--study-root', type=Path, required=True, help='Study root directory')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects (default: all)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--execute', action='store_true', help='Actually perform the migration')

    args = parser.parse_args()

    if not args.execute and not args.dry_run:
        print("Use --dry-run to preview or --execute to perform migration")
        return

    transforms_dir = args.study_root / 'transforms'

    if args.subjects:
        subject_dirs = [transforms_dir / s for s in args.subjects]
    else:
        subject_dirs = sorted(transforms_dir.glob('IRC805-*'))

    print(f"Transform Migration Tool")
    print(f"Study root: {args.study_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print(f"Subjects: {len(subject_dirs)}")

    all_logs = []
    for subject_dir in subject_dirs:
        if subject_dir.exists():
            log = migrate_subject(subject_dir, dry_run=args.dry_run)
            update_transforms_json(subject_dir, dry_run=args.dry_run)
            all_logs.append(log)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_migrated = sum(len(log['migrated']) for log in all_logs)
    total_skipped = sum(len(log['skipped']) for log in all_logs)
    total_errors = sum(len(log['errors']) for log in all_logs)
    print(f"Migrated: {total_migrated}")
    print(f"Skipped: {total_skipped}")
    print(f"Errors: {total_errors}")


if __name__ == '__main__':
    main()
