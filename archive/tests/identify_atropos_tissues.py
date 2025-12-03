#!/usr/bin/env python3
"""
Identify which Atropos POSTERIOR corresponds to which tissue type.

Atropos with K-means initialization produces posteriors in arbitrary order.
This script identifies tissues based on mean intensity in the original T1w image:
- CSF: lowest intensity
- GM: intermediate intensity
- WM: highest intensity
"""

import sys
import nibabel as nib
import numpy as np
from pathlib import Path

def identify_tissue_posteriors(seg_dir: Path, t1w_file: Path):
    """
    Identify which posterior file corresponds to which tissue type.

    Args:
        seg_dir: Segmentation directory with POSTERIOR_*.nii.gz files
        t1w_file: Original T1w image file

    Returns:
        Dictionary mapping tissue names to posterior filenames
    """
    # Load T1w image
    t1w_img = nib.load(t1w_file)
    t1w_data = t1w_img.get_fdata()

    # Load all posteriors and calculate mean T1w intensity
    posteriors = {}
    for post_file in sorted(seg_dir.glob('POSTERIOR_*.nii.gz')):
        post_img = nib.load(post_file)
        post_data = post_img.get_fdata()

        # Calculate weighted mean intensity (posterior probability * T1w intensity)
        masked_intensity = t1w_data * post_data
        mean_intensity = np.sum(masked_intensity) / np.sum(post_data)

        posteriors[post_file.name] = {
            'file': post_file,
            'mean_intensity': mean_intensity
        }

    # Sort by mean intensity
    sorted_posteriors = sorted(posteriors.items(), key=lambda x: x[1]['mean_intensity'])

    # Assign tissue types (CSF=lowest, GM=middle, WM=highest)
    tissue_map = {}
    if len(sorted_posteriors) >= 3:
        tissue_map['CSF'] = sorted_posteriors[0][0]  # Lowest intensity
        tissue_map['GM'] = sorted_posteriors[1][0]   # Middle intensity
        tissue_map['WM'] = sorted_posteriors[2][0]   # Highest intensity

    return tissue_map, posteriors


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python identify_atropos_tissues.py <seg_dir> <t1w_file>")
        sys.exit(1)

    seg_dir = Path(sys.argv[1])
    t1w_file = Path(sys.argv[2])

    tissue_map, posteriors = identify_tissue_posteriors(seg_dir, t1w_file)

    print("Tissue identification based on T1w intensity:")
    print("=" * 60)
    for tissue, post_file in tissue_map.items():
        intensity = posteriors[post_file]['mean_intensity']
        print(f"{tissue:8s} -> {post_file:20s} (intensity: {intensity:.2f})")

    print("\nAll posteriors:")
    for post_file, info in sorted(posteriors.items(), key=lambda x: x[1]['mean_intensity']):
        print(f"  {post_file}: {info['mean_intensity']:.2f}")
