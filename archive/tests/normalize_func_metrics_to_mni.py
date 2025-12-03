#!/usr/bin/env python3
"""
Normalize ReHo and fALFF maps to MNI152 space using func→anat→MNI transforms
"""

import logging
from pathlib import Path
import subprocess
import nibabel as nib
import numpy as np
import os


def setup_logging(log_file: Path):
    """Configure logging"""
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def normalize_metric_to_mni(
    metric_file: Path,
    func_to_anat_mat: Path,
    anat_to_mni_warp: Path,
    anat_brain: Path,
    mni_template: Path,
    output_file: Path
) -> bool:
    """
    Normalize functional metric to MNI152 space

    Uses two-stage transformation:
    1. func → anat (FLIRT)
    2. anat → MNI (ANTs)

    Args:
        metric_file: ReHo or fALFF map in native functional space
        func_to_anat_mat: FLIRT matrix for func→anat
        anat_to_mni_warp: ANTs composite transform for anat→MNI
        anat_brain: Anatomical brain image (reference for intermediate step)
        mni_template: MNI152 template (final reference)
        output_file: Output normalized metric file

    Returns:
        True if successful
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Apply func→anat transform (FLIRT)
    intermediate_file = output_file.parent / f"{output_file.stem}_in_anat.nii.gz"

    logging.info(f"  Step 1: Transforming to anatomical space...")
    cmd = [
        'flirt',
        '-in', str(metric_file),
        '-ref', str(anat_brain),
        '-applyxfm',
        '-init', str(func_to_anat_mat),
        '-out', str(intermediate_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"  ✗ FLIRT failed: {result.stderr}")
        return False

    # Step 2: Apply anat→MNI transform (ANTs)
    logging.info(f"  Step 2: Transforming to MNI space...")
    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(intermediate_file),
        '-r', str(mni_template),
        '-t', str(anat_to_mni_warp),
        '-o', str(output_file),
        '-n', 'Linear'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"  ✗ ANTs failed: {result.stderr}")
        return False

    # Clean up intermediate file
    intermediate_file.unlink()

    # Verify output dimensions
    img = nib.load(output_file)
    expected_shape = (91, 109, 91)  # MNI152 2mm
    if img.shape != expected_shape:
        logging.warning(f"  ⚠ Output shape {img.shape} != expected {expected_shape}")

    logging.info(f"  ✓ Normalized to MNI space: {img.shape}")
    return True


def mask_and_zscore_mni(
    metric_file: Path,
    mni_mask: Path,
    output_file: Path
) -> dict:
    """
    Apply MNI brain mask and recompute z-scores

    Args:
        metric_file: Normalized metric in MNI space
        mni_mask: MNI brain mask
        output_file: Output masked and z-scored file

    Returns:
        Dict with statistics
    """
    # Load data
    img = nib.load(metric_file)
    data = img.get_fdata()

    mask_img = nib.load(mni_mask)
    mask = mask_img.get_fdata().astype(bool)

    # Apply mask
    masked_data = np.copy(data)
    masked_data[~mask] = 0

    # Recompute z-scores within brain
    brain_values = masked_data[mask]
    brain_mean = np.mean(brain_values)
    brain_std = np.std(brain_values)

    # Z-score
    zscore_data = np.zeros_like(masked_data)
    zscore_data[mask] = (brain_values - brain_mean) / brain_std

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    zscore_img = nib.Nifti1Image(zscore_data, img.affine, img.header)
    nib.save(zscore_img, output_file)

    return {
        'mean': float(brain_mean),
        'std': float(brain_std),
        'n_voxels': int(np.sum(mask))
    }


def process_subject(
    subject_id: str,
    derivatives_dir: Path,
    metric: str,
    mni_template: Path,
    mni_mask: Path
) -> bool:
    """
    Process one subject: normalize ReHo/fALFF to MNI and apply z-scoring

    Args:
        subject_id: Subject identifier
        derivatives_dir: Derivatives directory
        metric: 'reho' or 'falff'
        mni_template: MNI152 template
        mni_mask: MNI brain mask

    Returns:
        True if successful
    """
    subject_dir = derivatives_dir / subject_id
    func_dir = subject_dir / 'func'
    anat_dir = subject_dir / 'anat'

    # Check required files
    metric_dir = func_dir / metric
    if metric == 'reho':
        metric_file = metric_dir / 'reho.nii.gz'
    else:  # falff
        metric_file = metric_dir / 'falff.nii.gz'

    func_to_anat_mat = func_dir / 'acompcor' / 'func_to_t1w.mat'
    anat_to_mni_warp = anat_dir / 'transforms' / 'ants_Composite.h5'

    # Find brain file in subdirectory (name varies)
    brain_files = list((anat_dir / 'brain').glob('*_brain.nii.gz'))
    if not brain_files:
        logging.warning(f"{subject_id}: No brain file found in {anat_dir / 'brain'}")
        return False
    anat_brain = brain_files[0]

    # Check all required files exist
    required_files = [metric_file, func_to_anat_mat, anat_to_mni_warp]
    missing = [f for f in required_files if not f.exists()]

    if missing:
        logging.warning(f"{subject_id}: Missing files: {[str(f) for f in missing]}")
        return False

    logging.info(f"\nProcessing {subject_id} {metric}...")

    # Output file for normalized metric
    output_mni = metric_dir / f'{metric}_mni.nii.gz'

    # Step 1: Normalize to MNI
    success = normalize_metric_to_mni(
        metric_file=metric_file,
        func_to_anat_mat=func_to_anat_mat,
        anat_to_mni_warp=anat_to_mni_warp,
        anat_brain=anat_brain,
        mni_template=mni_template,
        output_file=output_mni
    )

    if not success:
        return False

    # Step 2: Apply MNI mask and z-score
    output_zscore = metric_dir / f'{metric}_mni_zscore_masked.nii.gz'

    logging.info(f"  Step 3: Applying MNI mask and z-scoring...")
    stats = mask_and_zscore_mni(
        metric_file=output_mni,
        mni_mask=mni_mask,
        output_file=output_zscore
    )

    logging.info(f"  ✓ Complete: {stats['n_voxels']:,} brain voxels")

    return True


def main():
    # Setup logging
    log_file = Path('logs/normalize_func_metrics_to_mni.log')
    setup_logging(log_file)

    logging.info("="*80)
    logging.info("NORMALIZING REHO/FALFF TO MNI152 SPACE")
    logging.info("="*80)

    # Paths
    derivatives_dir = Path('/mnt/bytopia/IRC805/derivatives')
    mni_template = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz')
    mni_mask = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')

    # Verify MNI template exists
    if not mni_template.exists():
        logging.error(f"MNI template not found: {mni_template}")
        return 1

    if not mni_mask.exists():
        logging.error(f"MNI mask not found: {mni_mask}")
        return 1

    # Find all subjects
    subjects = sorted([d.name for d in derivatives_dir.glob('IRC805-*') if d.is_dir()])

    logging.info(f"\nFound {len(subjects)} subjects")

    # Process ReHo
    logging.info("\n" + "="*80)
    logging.info("PROCESSING REHO")
    logging.info("="*80)

    reho_success = []
    for subject in subjects:
        if process_subject(subject, derivatives_dir, 'reho', mni_template, mni_mask):
            reho_success.append(subject)

    logging.info(f"\n✓ ReHo: {len(reho_success)}/{len(subjects)} subjects normalized")

    # Process fALFF
    logging.info("\n" + "="*80)
    logging.info("PROCESSING FALFF")
    logging.info("="*80)

    falff_success = []
    for subject in subjects:
        if process_subject(subject, derivatives_dir, 'falff', mni_template, mni_mask):
            falff_success.append(subject)

    logging.info(f"\n✓ fALFF: {len(falff_success)}/{len(subjects)} subjects normalized")

    # Summary
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    logging.info(f"ReHo normalized: {len(reho_success)} subjects")
    logging.info(f"fALFF normalized: {len(falff_success)} subjects")
    logging.info(f"\nNormalized files saved as: *_mni_zscore_masked.nii.gz")

    return 0


if __name__ == '__main__':
    exit(main())
