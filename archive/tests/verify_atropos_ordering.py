#!/usr/bin/env python3
"""
Quick script to verify if POSTER IOR_02 is actually GM for all subjects.
"""
from pathlib import Path
import nibabel as nib
import numpy as np

def check_subject_tissues(subject: str, derivatives_dir: Path):
    """Check which POSTERIOR file should be GM for this subject."""
    seg_dir = derivatives_dir / subject / 'anat' / 'segmentation'
    t1w_brain = derivatives_dir / subject / 'anat' / 'brain.nii.gz'

    if not t1w_brain.exists():
        return None

    # Load T1w
    t1w_img = nib.load(t1w_brain)
    t1w_data = t1w_img.get_fdata()

    # Check all posteriors
    posteriors = {}
    for post_file in sorted(seg_dir.glob('POSTERIOR_*.nii.gz')):
        post_img = nib.load(post_file)
        post_data = post_img.get_fdata()

        # Calculate mean intensity
        masked_intensity = t1w_data * post_data
        mean_intensity = np.sum(masked_intensity) / (np.sum(post_data) + 1e-10)
        posteriors[post_file.name] = mean_intensity

    # Sort by intensity
    sorted_posts = sorted(posteriors.items(), key=lambda x: x[1])

    if len(sorted_posts) >= 3:
        return {
            'subject': subject,
            'CSF': sorted_posts[0][0],
            'GM': sorted_posts[1][0],
            'WM': sorted_posts[2][0]
        }
    return None

# Check a few subjects
derivatives_dir = Path('/mnt/bytopia/IRC805/derivatives')
test_subjects = [
    'IRC805-0580101',
    'IRC805-1580101',
    'IRC805-1640101',
    'IRC805-2350101',
    'IRC805-2990202'
]

print("="  * 80)
print("VERIFYING ATROPOS TISSUE ORDERING")
print("=" * 80)
print()

for subject in test_subjects:
    result = check_subject_tissues(subject, derivatives_dir)
    if result:
        print(f"{subject}:")
        print(f"  CSF: {result['CSF']}")
        print(f"  GM:  {result['GM']}")
        print(f"  WM:  {result['WM']}")
        if result['GM'] != 'POSTERIOR_02.nii.gz':
            print(f"  ⚠️  WARNING: GM is NOT POSTERIOR_02!")
        print()
