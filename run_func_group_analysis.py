#!/usr/bin/env python3
"""
Group-level ReHo and fALFF Analysis

Gathers individual subject maps, creates design matrices, and runs FSL randomise
with TFCE for nonparametric statistical testing.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np

# Add neurovrai to path
sys.path.insert(0, str(Path(__file__).parent))

from neurovrai.analysis.stats.randomise_wrapper import run_randomise, summarize_results


def setup_logging(log_file: Path):
    """Configure logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def gather_subject_maps(
    derivatives_dir: Path,
    metric: str,
    output_dir: Path
) -> tuple[Path, list[str]]:
    """
    Gather individual subject maps into a 4D volume

    Args:
        derivatives_dir: Path to derivatives directory
        metric: 'reho' or 'falff'
        output_dir: Output directory for 4D image

    Returns:
        Tuple of (4D image path, list of subject IDs)
    """
    logging.info(f"\nGathering {metric.upper()} maps...")
    logging.info("=" * 80)

    # Find all subject maps (using MNI-normalized MASKED z-scored versions)
    if metric == 'reho':
        map_name = 'reho_mni_zscore_masked.nii.gz'
    else:  # falff
        map_name = 'falff_mni_zscore_masked.nii.gz'

    subject_maps = []
    subject_ids = []

    # Search for subjects
    for subject_dir in sorted(derivatives_dir.glob('IRC805-*/func')):
        subject_id = subject_dir.parent.name
        map_file = subject_dir / metric / map_name

        if map_file.exists():
            subject_maps.append(map_file)
            subject_ids.append(subject_id)
            logging.info(f"  ✓ {subject_id}")
        else:
            logging.warning(f"  ✗ {subject_id} - missing {metric} map")

    if len(subject_maps) == 0:
        raise ValueError(f"No {metric} maps found in {derivatives_dir}")

    logging.info(f"\nFound {len(subject_maps)} subjects with {metric} data")

    # Load first image to get dimensions
    first_img = nib.load(subject_maps[0])
    shape_3d = first_img.shape
    affine = first_img.affine
    header = first_img.header

    logging.info(f"Image dimensions: {shape_3d}")

    # Create 4D array
    data_4d = np.zeros((*shape_3d, len(subject_maps)), dtype=np.float32)

    # Load all subjects
    logging.info(f"\nLoading {len(subject_maps)} volumes...")
    for i, map_file in enumerate(subject_maps):
        img = nib.load(map_file)
        data_4d[..., i] = img.get_fdata()

    # Save 4D image
    output_dir.mkdir(parents=True, exist_ok=True)
    output_4d = output_dir / f'all_{metric}_4D.nii.gz'

    img_4d = nib.Nifti1Image(data_4d, affine, header)
    nib.save(img_4d, output_4d)

    logging.info(f"✓ Saved 4D image: {output_4d}")
    logging.info(f"  Shape: {data_4d.shape}")

    return output_4d, subject_ids


def create_mean_mask(image_4d: Path, output_dir: Path) -> Path:
    """
    Create a mean mask from 4D image (non-zero voxels across subjects)

    Args:
        image_4d: Path to 4D image
        output_dir: Output directory

    Returns:
        Path to mask file
    """
    logging.info("\nCreating group mask...")

    img = nib.load(image_4d)
    data = img.get_fdata()

    # Mask: voxels that are non-zero in at least 50% of subjects
    n_subjects = data.shape[3]
    threshold = n_subjects * 0.5

    nonzero_count = np.sum(data != 0, axis=3)
    mask = (nonzero_count >= threshold).astype(np.uint8)

    n_voxels = np.sum(mask)
    logging.info(f"  Mask contains {n_voxels:,} voxels")

    mask_file = output_dir / 'group_mask.nii.gz'
    mask_img = nib.Nifti1Image(mask, img.affine, img.header)
    nib.save(mask_img, mask_file)

    logging.info(f"✓ Saved mask: {mask_file}")

    return mask_file


def create_design_matrices(
    subject_ids: list[str],
    participants_file: Path,
    output_dir: Path
):
    """
    Create FSL design matrices and contrasts

    Args:
        subject_ids: List of subject IDs in 4D image order
        participants_file: Path to participants.tsv
        output_dir: Output directory
    """
    logging.info("\nCreating design matrices...")
    logging.info("=" * 80)

    # Load participants data
    if participants_file.suffix == '.tsv':
        df = pd.read_csv(participants_file, sep='\t')
    else:
        df = pd.read_csv(participants_file)

    # Match subjects to 4D image order
    df_ordered = pd.DataFrame()
    for subject_id in subject_ids:
        match = df[df['participant_id'] == subject_id]
        if len(match) == 0:
            logging.warning(f"Subject {subject_id} not found in participants file")
            # Use default values
            match = pd.DataFrame({
                'participant_id': [subject_id],
                'age': [30.0],
                'group': [0]
            })
        df_ordered = pd.concat([df_ordered, match.iloc[0:1]], ignore_index=True)

    # Normalize age (z-score) and get group (already 0/1)
    age_norm = (df_ordered['age'] - df_ordered['age'].mean()) / df_ordered['age'].std()
    group = df_ordered['group'].values

    # Create design matrix (FSL format)
    # Columns: intercept, age, group
    n_subjects = len(subject_ids)
    design = np.column_stack([
        np.ones(n_subjects),  # Intercept
        age_norm.values,       # Age (normalized)
        group                  # Group (0/1)
    ])

    # Save design matrix (FSL format)
    design_file = output_dir / 'design.mat'
    with open(design_file, 'w') as f:
        # Header
        f.write('/NumWaves\t3\n')
        f.write(f'/NumPoints\t{n_subjects}\n')
        f.write('/PPheights\t\t1.000000e+00\t1.000000e+00\t1.000000e+00\n\n')
        f.write('/Matrix\n')
        # Data
        for row in design:
            f.write('\t'.join([f'{val:.6e}' for val in row]) + '\n')

    logging.info(f"✓ Design matrix: {design_file}")
    logging.info(f"  Subjects: {n_subjects}")
    logging.info(f"  Regressors: 3 (intercept, age, group)")

    # Create contrast matrix
    # Contrast 1: Age positive
    # Contrast 2: Age negative
    # Contrast 3: Group 1 > Group 0
    # Contrast 4: Group 0 > Group 1
    contrasts = np.array([
        [0, 1, 0],   # Age positive
        [0, -1, 0],  # Age negative
        [0, 0, 1],   # Group 1 > Group 0
        [0, 0, -1],  # Group 0 > Group 1
    ])

    contrast_file = output_dir / 'design.con'
    with open(contrast_file, 'w') as f:
        # Header
        f.write('/ContrastName1\tage_positive\n')
        f.write('/ContrastName2\tage_negative\n')
        f.write('/ContrastName3\tgroup1_gt_group0\n')
        f.write('/ContrastName4\tgroup0_gt_group1\n')
        f.write('/NumWaves\t3\n')
        f.write(f'/NumContrasts\t{len(contrasts)}\n')
        f.write('/PPheights\t\t1.000000e+00\t1.000000e+00\n')
        f.write('/RequiredEffect\t\t1.000\t1.000\n\n')
        f.write('/Matrix\n')
        # Data
        for row in contrasts:
            f.write('\t'.join([f'{val:.6e}' for val in row]) + '\n')

    logging.info(f"✓ Contrast matrix: {contrast_file}")
    logging.info(f"  Contrasts: 4 (age+, age-, group1>group0, group0>group1)")

    return design_file, contrast_file


def run_group_analysis(
    metric: str,
    derivatives_dir: Path,
    analysis_dir: Path,
    participants_file: Path,
    study_name: str = 'mock_study',
    n_permutations: int = 500
):
    """
    Run complete group-level analysis for ReHo or fALFF

    Args:
        metric: 'reho' or 'falff'
        derivatives_dir: Path to derivatives directory
        analysis_dir: Path to analysis directory
        participants_file: Path to participants file
        study_name: Name of analysis study (e.g., 'mock_study', 'age_analysis')
        n_permutations: Number of randomise permutations
    """
    logging.info("\n" + "=" * 80)
    logging.info(f"{metric.upper()} GROUP-LEVEL ANALYSIS - {study_name}")
    logging.info("=" * 80)

    # Setup output directory with study subdirectory
    output_dir = analysis_dir / 'func' / metric / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather 4D data
    image_4d, subject_ids = gather_subject_maps(derivatives_dir, metric, output_dir)

    # 2. Create mask
    mask_file = create_mean_mask(image_4d, output_dir)

    # 3. Create design matrices
    design_mat, contrast_con = create_design_matrices(
        subject_ids, participants_file, output_dir
    )

    # 4. Run randomise
    logging.info("\n" + "=" * 80)
    logging.info("Running FSL Randomise")
    logging.info("=" * 80)

    randomise_dir = output_dir / 'randomise_output'

    result = run_randomise(
        input_file=image_4d,
        design_mat=design_mat,
        contrast_con=contrast_con,
        output_dir=randomise_dir,
        mask=mask_file,
        n_permutations=n_permutations,
        tfce=True,
        seed=42
    )

    # 5. Summarize results
    summary = summarize_results(randomise_dir, threshold=0.95)

    logging.info(f"\n✓ {metric.upper()} analysis complete!")
    logging.info(f"  Results: {randomise_dir}")

    return result, summary


def main():
    """Main execution"""
    # Paths
    study_root = Path('/mnt/bytopia/IRC805')
    derivatives_dir = study_root / 'derivatives'
    analysis_dir = study_root / 'analysis'
    participants_file = analysis_dir / 'participants_synthetic.tsv'

    # Setup logging
    log_file = Path('logs/func_group_analysis.log')
    log_file.parent.mkdir(exist_ok=True)
    setup_logging(log_file)

    logging.info("=" * 80)
    logging.info("FUNCTIONAL CONNECTIVITY GROUP ANALYSIS")
    logging.info("=" * 80)
    logging.info(f"Study root: {study_root}")
    logging.info(f"Derivatives: {derivatives_dir}")
    logging.info(f"Analysis: {analysis_dir}")
    logging.info(f"Participants: {participants_file}")

    try:
        # Run ReHo analysis
        reho_result, reho_summary = run_group_analysis(
            metric='reho',
            derivatives_dir=derivatives_dir,
            analysis_dir=analysis_dir,
            participants_file=participants_file,
            study_name='mock_study',
            n_permutations=500
        )

        # Run fALFF analysis
        falff_result, falff_summary = run_group_analysis(
            metric='falff',
            derivatives_dir=derivatives_dir,
            analysis_dir=analysis_dir,
            participants_file=participants_file,
            study_name='mock_study',
            n_permutations=500
        )

        logging.info("\n" + "=" * 80)
        logging.info("ALL ANALYSES COMPLETE")
        logging.info("=" * 80)
        logging.info(f"ReHo results: {reho_result['output_dir']}")
        logging.info(f"fALFF results: {falff_result['output_dir']}")

    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
