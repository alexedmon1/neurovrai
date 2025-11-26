#!/usr/bin/env python3
"""
Mask existing ReHo/fALFF maps with brain masks and recompute z-scores
"""

import logging
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import stats


def mask_and_zscore_map(
    input_map: Path,
    mask_file: Path,
    output_map: Path
) -> dict:
    """
    Apply brain mask and recompute z-score normalization

    Args:
        input_map: Input map (e.g., reho.nii.gz or falff.nii.gz)
        mask_file: Brain mask
        output_map: Output masked and z-scored map

    Returns:
        Dict with statistics
    """
    # Load data
    img = nib.load(input_map)
    data = img.get_fdata()

    mask_img = nib.load(mask_file)
    mask = mask_img.get_fdata().astype(bool)

    # Apply mask
    masked_data = np.copy(data)
    masked_data[~mask] = 0

    # Recompute z-scores within brain only
    brain_values = masked_data[mask]
    brain_mean = np.mean(brain_values)
    brain_std = np.std(brain_values)

    # Z-score
    zscore_data = np.zeros_like(masked_data)
    zscore_data[mask] = (brain_values - brain_mean) / brain_std

    # Save
    output_map.parent.mkdir(parents=True, exist_ok=True)
    zscore_img = nib.Nifti1Image(zscore_data, img.affine, img.header)
    nib.save(zscore_img, output_map)

    return {
        'mean': float(brain_mean),
        'std': float(brain_std),
        'n_voxels': int(np.sum(mask)),
        'output': str(output_map)
    }


def process_all_subjects(
    derivatives_dir: Path,
    metric: str
) -> list:
    """
    Process all subjects for a given metric

    Args:
        derivatives_dir: Path to derivatives directory
        metric: 'reho' or 'falff'

    Returns:
        List of processed subjects
    """
    subjects_processed = []

    # Find all subjects with the metric
    for subject_dir in sorted(derivatives_dir.glob('IRC805-*/func')):
        subject_id = subject_dir.parent.name

        # Check if metric exists
        metric_dir = subject_dir / metric
        if not metric_dir.exists():
            logging.warning(f"{subject_id}: No {metric} directory")
            continue

        # Get files
        if metric == 'reho':
            input_map = metric_dir / 'reho.nii.gz'
        else:  # falff
            input_map = metric_dir / 'falff.nii.gz'

        mask_file = subject_dir / 'func_mask.nii.gz'
        output_map = metric_dir / f'{metric}_zscore_masked.nii.gz'

        if not input_map.exists():
            logging.warning(f"{subject_id}: No {metric} map")
            continue

        if not mask_file.exists():
            logging.warning(f"{subject_id}: No brain mask")
            continue

        # Process
        logging.info(f"Processing {subject_id} {metric}...")
        try:
            stats_dict = mask_and_zscore_map(input_map, mask_file, output_map)
            subjects_processed.append({
                'subject': subject_id,
                'metric': metric,
                **stats_dict
            })
            logging.info(f"  ✓ {stats_dict['n_voxels']:,} brain voxels")
        except Exception as e:
            logging.error(f"  ✗ Error: {e}")
            continue

    return subjects_processed


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('logs/mask_func_metrics.log'),
            logging.StreamHandler()
        ]
    )

    derivatives_dir = Path('/mnt/bytopia/IRC805/derivatives')

    logging.info("="*80)
    logging.info("MASKING AND RECOMPUTING REHO/FALFF WITH BRAIN MASKS")
    logging.info("="*80)

    # Process ReHo
    logging.info("\nProcessing ReHo...")
    reho_results = process_all_subjects(derivatives_dir, 'reho')
    logging.info(f"✓ Processed {len(reho_results)} subjects")

    # Process fALFF
    logging.info("\nProcessing fALFF...")
    falff_results = process_all_subjects(derivatives_dir, 'falff')
    logging.info(f"✓ Processed {len(falff_results)} subjects")

    logging.info("\n" + "="*80)
    logging.info("COMPLETE")
    logging.info("="*80)
    logging.info(f"ReHo: {len(reho_results)} subjects")
    logging.info(f"fALFF: {len(falff_results)} subjects")
    logging.info("\nNew masked z-score maps saved as: *_zscore_masked.nii.gz")


if __name__ == '__main__':
    main()
