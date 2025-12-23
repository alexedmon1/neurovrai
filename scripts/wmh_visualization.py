#!/usr/bin/env python3
"""
WMH Visualization and FreeSurfer Comparison Script

Generates:
1. Multi-slice overlay visualization of detected WMH
2. Comparison table with FreeSurfer hypointensity values
"""

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_wmh_multiSlice_visualization(
    t2w_mni: Path,
    wmh_mask: Path,
    output_file: Path,
    n_slices: int = 9,
    title: str = "WMH Detection"
):
    """
    Create multi-slice axial visualization of WMH overlay on T2w.

    Parameters
    ----------
    t2w_mni : Path
        T2w image in MNI space
    wmh_mask : Path
        Binary WMH mask
    output_file : Path
        Output PNG file
    n_slices : int
        Number of axial slices to show
    title : str
        Title for the visualization
    """
    t2w_img = nib.load(t2w_mni)
    t2w_data = t2w_img.get_fdata()

    wmh_img = nib.load(wmh_mask)
    wmh_data = wmh_img.get_fdata()

    # Get slice indices with WMH content
    z_indices = np.where(np.any(wmh_data > 0, axis=(0, 1)))[0]

    if len(z_indices) == 0:
        # No WMH detected, show middle slices
        z_dim = t2w_data.shape[2]
        z_indices = np.linspace(z_dim * 0.2, z_dim * 0.8, n_slices, dtype=int)
    else:
        # Sample slices across the range with WMH
        if len(z_indices) > n_slices:
            idx = np.linspace(0, len(z_indices) - 1, n_slices, dtype=int)
            z_indices = z_indices[idx]

    n_slices = len(z_indices)
    n_cols = 3
    n_rows = int(np.ceil(n_slices / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes):
        if i < len(z_indices):
            z = z_indices[i]

            # T2w background
            ax.imshow(np.rot90(t2w_data[:, :, z]), cmap='gray',
                     vmin=np.percentile(t2w_data, 1),
                     vmax=np.percentile(t2w_data, 99))

            # WMH overlay in red
            wmh_slice = np.rot90(wmh_data[:, :, z])
            masked_wmh = np.ma.masked_where(wmh_slice == 0, wmh_slice)
            ax.imshow(masked_wmh, cmap='Reds', alpha=0.7, vmin=0, vmax=1)

            # Count WMH voxels in this slice
            n_wmh = int(np.sum(wmh_data[:, :, z] > 0))
            ax.set_title(f'z={z} ({n_wmh} vox)', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved visualization: {output_file}")


def extract_freesurfer_wmh(aseg_stats_file: Path) -> dict:
    """
    Extract WM-hypointensities values from FreeSurfer aseg.stats.

    Parameters
    ----------
    aseg_stats_file : Path
        Path to aseg.stats file

    Returns
    -------
    dict
        Dictionary with nvoxels and volume_mm3
    """
    with open(aseg_stats_file, 'r') as f:
        for line in f:
            if 'WM-hypointensities' in line and 'non-WM' not in line and 'Left' not in line and 'Right' not in line:
                parts = line.split()
                # Format: Index SegId NVoxels Volume_mm3 StructName ...
                nvoxels = int(parts[2])
                volume_mm3 = float(parts[3])
                return {
                    'fs_nvoxels': nvoxels,
                    'fs_volume_mm3': volume_mm3
                }
    return {'fs_nvoxels': 0, 'fs_volume_mm3': 0}


def compare_wmh_with_freesurfer(
    hyperintensities_dir: Path,
    freesurfer_dir: Path,
    output_file: Path
):
    """
    Compare our WMH detection with FreeSurfer hypointensity values.

    Parameters
    ----------
    hyperintensities_dir : Path
        WMH analysis output directory
    freesurfer_dir : Path
        FreeSurfer subjects directory
    output_file : Path
        Output CSV file
    """
    hyperintensities_dir = Path(hyperintensities_dir)
    freesurfer_dir = Path(freesurfer_dir)

    comparisons = []

    for subj_dir in sorted(hyperintensities_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name in ['group', 'logs']:
            continue

        subject = subj_dir.name

        # Load our WMH metrics
        metrics_file = subj_dir / 'wmh_metrics.json'
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        if 'error' in metrics:
            continue

        our_volume = metrics.get('wmh_summary', {}).get('total_volume_mm3', 0)
        our_n_lesions = metrics.get('wmh_summary', {}).get('n_lesions', 0)
        our_voxels = metrics.get('wmh_summary', {}).get('total_voxels', 0)

        # Load FreeSurfer values
        aseg_stats = freesurfer_dir / subject / 'stats' / 'aseg.stats'
        if aseg_stats.exists():
            fs_values = extract_freesurfer_wmh(aseg_stats)
        else:
            fs_values = {'fs_nvoxels': None, 'fs_volume_mm3': None}

        comparisons.append({
            'subject': subject,
            'our_n_lesions': our_n_lesions,
            'our_voxels': our_voxels,
            'our_volume_mm3': our_volume,
            'fs_nvoxels': fs_values['fs_nvoxels'],
            'fs_volume_mm3': fs_values['fs_volume_mm3'],
            'ratio_our_vs_fs': (our_volume / fs_values['fs_volume_mm3']
                               if fs_values['fs_volume_mm3'] and fs_values['fs_volume_mm3'] > 0
                               else None)
        })

    df = pd.DataFrame(comparisons)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved comparison: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("WMH Detection vs FreeSurfer Comparison")
    print("=" * 80)
    print(df.to_string(index=False))

    if df['fs_volume_mm3'].notna().any():
        our_mean = df['our_volume_mm3'].mean()
        fs_mean = df['fs_volume_mm3'].mean()
        ratio = our_mean / fs_mean if fs_mean > 0 else None

        print("\n" + "-" * 40)
        print(f"Our mean WMH volume: {our_mean:.1f} mm³")
        print(f"FreeSurfer mean WM-hypointensities: {fs_mean:.1f} mm³")
        if ratio:
            print(f"Ratio (Our/FS): {ratio:.2f}x")
        print("-" * 40)

    return df


def main():
    parser = argparse.ArgumentParser(description='WMH Visualization and Comparison')
    subparsers = parser.add_subparsers(dest='command')

    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Create WMH overlay visualization')
    viz_parser.add_argument('--subject', required=True, help='Subject ID')
    viz_parser.add_argument('--hyperintensities-dir', required=True, type=Path)
    viz_parser.add_argument('--output', type=Path, help='Output PNG file')

    # Comparison command
    cmp_parser = subparsers.add_parser('compare', help='Compare with FreeSurfer')
    cmp_parser.add_argument('--hyperintensities-dir', required=True, type=Path)
    cmp_parser.add_argument('--freesurfer-dir', required=True, type=Path)
    cmp_parser.add_argument('--output', type=Path, help='Output CSV file')

    # Batch visualization command
    batch_parser = subparsers.add_parser('batch-visualize', help='Visualize all subjects')
    batch_parser.add_argument('--hyperintensities-dir', required=True, type=Path)

    args = parser.parse_args()

    if args.command == 'visualize':
        subj_dir = args.hyperintensities_dir / args.subject
        t2w_mni = subj_dir / 't2w_mni.nii.gz'
        wmh_mask = subj_dir / 'wmh_mask.nii.gz'
        output = args.output or subj_dir / 'wmh_visualization.png'

        create_wmh_multiSlice_visualization(
            t2w_mni, wmh_mask, output,
            title=f"WMH Detection: {args.subject}"
        )

    elif args.command == 'compare':
        output = args.output or args.hyperintensities_dir / 'group' / 'wmh_vs_freesurfer.csv'
        compare_wmh_with_freesurfer(
            args.hyperintensities_dir,
            args.freesurfer_dir,
            output
        )

    elif args.command == 'batch-visualize':
        for subj_dir in sorted(args.hyperintensities_dir.iterdir()):
            if not subj_dir.is_dir() or subj_dir.name in ['group', 'logs']:
                continue

            t2w_mni = subj_dir / 't2w_mni.nii.gz'
            wmh_mask = subj_dir / 'wmh_mask.nii.gz'
            output = subj_dir / 'wmh_visualization.png'

            if t2w_mni.exists() and wmh_mask.exists():
                create_wmh_multiSlice_visualization(
                    t2w_mni, wmh_mask, output,
                    title=f"WMH Detection: {subj_dir.name}"
                )


if __name__ == '__main__':
    main()
