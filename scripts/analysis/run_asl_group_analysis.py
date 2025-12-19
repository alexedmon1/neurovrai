#!/usr/bin/env python3
"""
Group-level ASL (Arterial Spin Labeling) Analysis

Gathers individual subject CBF maps, creates design matrices with neuroaider,
and runs FSL randomise with TFCE for nonparametric statistical testing.

Usage:
    python run_asl_group_analysis.py
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
from neurovrai.analysis.stats.nilearn_glm import run_second_level_glm, summarize_glm_results
from neurovrai.analysis.utils.design_validation import validate_design_alignment


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
    output_dir: Path,
    design_dir: Path
) -> tuple[Path, list[str]]:
    """
    Gather individual subject CBF maps into a 4D volume

    Args:
        derivatives_dir: Path to derivatives directory
        output_dir: Output directory for 4D image
        design_dir: Path to design directory (for subject order)

    Returns:
        Tuple of (4D image path, list of subject IDs)
    """
    logging.info(f"\nGathering ASL CBF maps...")
    logging.info("=" * 80)

    # Load design matrix to get correct subject order
    participants_file = design_dir / 'participants_matched.tsv'
    if not participants_file.exists():
        raise FileNotFoundError(f"Participants file not found: {participants_file}")

    participants_df = pd.read_csv(participants_file, sep='\t')
    design_subjects = participants_df['participant_id'].tolist()

    logging.info(f"Design matrix has {len(design_subjects)} subjects")
    logging.info(f"Will use design matrix subject order for 4D image")

    # Find all subject CBF maps (using MNI-normalized versions)
    # IMPORTANT: Iterate in design matrix order
    subject_maps = []
    subject_ids = []

    # Search for subjects IN DESIGN MATRIX ORDER
    for subject_id in design_subjects:
        subject_dir = derivatives_dir / subject_id / 'asl'
        map_name = f'{subject_id}_cbf_mni.nii.gz'
        map_file = subject_dir / map_name

        if map_file.exists():
            subject_maps.append(map_file)
            subject_ids.append(subject_id)
            logging.info(f"  ✓ {subject_id}")
        else:
            logging.warning(f"  ✗ {subject_id} - missing CBF map")

    if len(subject_maps) == 0:
        raise ValueError(f"No CBF maps found in {derivatives_dir}")

    logging.info(f"\nFound {len(subject_maps)} subjects with ASL data")

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
    output_4d = output_dir / 'all_cbf_4D.nii.gz'

    img_4d = nib.Nifti1Image(data_4d, affine, header)
    nib.save(img_4d, output_4d)

    logging.info(f"✓ Saved 4D image: {output_4d}")
    logging.info(f"  Shape: {data_4d.shape}")

    return output_4d, subject_ids


def create_mean_mask(image_4d: Path, output_dir: Path, brain_mask: Path = None) -> Path:
    """
    Create a group mask from 4D image, constrained to brain region.

    The mask includes voxels that:
    1. Are within the brain (MNI brain mask)
    2. Have valid (non-zero) data in at least 50% of subjects

    Args:
        image_4d: Path to 4D image
        output_dir: Output directory
        brain_mask: Path to brain mask (default: MNI152_T1_2mm_brain_mask)

    Returns:
        Path to mask file
    """
    import os
    logging.info("\nCreating group mask...")

    img = nib.load(image_4d)
    data = img.get_fdata()

    # Load brain mask (CRITICAL: constrains analysis to brain only)
    if brain_mask is None:
        fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')
        brain_mask = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm_brain_mask.nii.gz'

    if not brain_mask.exists():
        raise FileNotFoundError(f"Brain mask not found: {brain_mask}")

    brain_img = nib.load(brain_mask)
    brain_data = brain_img.get_fdata()

    logging.info(f"  Using brain mask: {brain_mask}")
    n_brain_voxels = np.sum(brain_data > 0)
    logging.info(f"  Brain mask voxels: {n_brain_voxels:,}")

    # Create data mask: voxels that are non-zero in at least 50% of subjects
    n_subjects = data.shape[3]
    threshold = n_subjects * 0.5

    nonzero_count = np.sum(data != 0, axis=3)
    data_mask = (nonzero_count >= threshold).astype(np.uint8)

    n_data_voxels = np.sum(data_mask)
    logging.info(f"  Data coverage voxels: {n_data_voxels:,}")

    # CRITICAL: Intersect with brain mask to exclude non-brain voxels
    mask = (data_mask * (brain_data > 0)).astype(np.uint8)

    n_voxels = np.sum(mask)
    logging.info(f"  Final mask voxels (brain ∩ data): {n_voxels:,}")

    # Warn if significant data is outside brain (indicates normalization issue)
    outside_brain = np.sum(data_mask * (brain_data == 0))
    if outside_brain > 1000:
        logging.warning(f"  ⚠ {outside_brain:,} non-zero voxels were outside brain mask (excluded)")

    mask_file = output_dir / 'group_mask.nii.gz'
    mask_img = nib.Nifti1Image(mask, img.affine, img.header)
    nib.save(mask_img, mask_file)

    logging.info(f"✓ Saved mask: {mask_file}")

    return mask_file


def load_design_matrices(
    subject_ids: list[str],
    design_dir: Path,
    output_dir: Path,
    image_4d: Path
):
    """
    Load pre-generated FSL design matrices and validate alignment

    Args:
        subject_ids: List of subject IDs in 4D image order
        design_dir: Directory containing pre-generated design files
        output_dir: Output directory to copy design files
        image_4d: Path to 4D merged image for validation

    Returns:
        Tuple of (design_mat_file, contrast_file, contrast_names)
    """
    import json
    import shutil

    logging.info("\nLoading pre-generated design matrices...")
    logging.info("=" * 80)
    logging.info(f"Design directory: {design_dir}")

    design_dir = Path(design_dir)
    source_design_mat = design_dir / 'design.mat'
    source_design_con = design_dir / 'design.con'
    source_design_summary = design_dir / 'design_summary.json'

    # Validate design files exist
    if not source_design_mat.exists():
        raise FileNotFoundError(f"Design matrix not found: {source_design_mat}")
    if not source_design_con.exists():
        raise FileNotFoundError(f"Contrast file not found: {source_design_con}")

    # Load design summary
    if source_design_summary.exists():
        with open(source_design_summary, 'r') as f:
            design_summary = json.load(f)
        logging.info(f"✓ Design summary loaded:")
        logging.info(f"    Subjects: {design_summary['n_subjects']}")
        logging.info(f"    Predictors: {design_summary['n_predictors']}")
        logging.info(f"    Columns: {design_summary['columns']}")
        logging.info(f"    Contrasts ({design_summary['n_contrasts']}): {design_summary['contrasts']}")
        contrast_names = design_summary['contrasts']
    else:
        logging.warning(f"Design summary not found: {source_design_summary}")
        # Parse from design.con if summary not available
        with open(source_design_con, 'r') as f:
            con_lines = f.readlines()
            n_contrasts = int([l for l in con_lines if '/NumContrasts' in l][0].split()[1])
        contrast_names = [f"contrast_{i+1}" for i in range(n_contrasts)]

    # Validate design alignment with MRI data
    validation_result = validate_design_alignment(
        design_dir=design_dir,
        mri_files=[image_4d],  # 4D merged file
        subject_ids=subject_ids,
        analysis_type="ASL"
    )

    logging.info(f"✓ Design validation passed: {validation_result['n_subjects']} subjects aligned")

    # Copy design files to output directory
    design_mat_file = output_dir / 'design.mat'
    design_con_file = output_dir / 'design.con'
    design_summary_file = output_dir / 'design_summary.json'

    shutil.copy(source_design_mat, design_mat_file)
    shutil.copy(source_design_con, design_con_file)
    if source_design_summary.exists():
        shutil.copy(source_design_summary, design_summary_file)

    logging.info(f"✓ Copied design files to output directory:")
    logging.info(f"    - {design_mat_file}")
    logging.info(f"    - {design_con_file}")
    if source_design_summary.exists():
        logging.info(f"    - {design_summary_file}")

    return design_mat_file, design_con_file, contrast_names


def run_group_analysis(
    derivatives_dir: Path,
    analysis_dir: Path,
    design_dir: Path,
    study_name: str = 'asl_group',
    method: str = 'randomise',
    n_permutations: int = 5000,
    z_threshold: float = 2.3
):
    """
    Run complete group-level analysis for ASL CBF

    Args:
        derivatives_dir: Path to derivatives directory
        analysis_dir: Path to analysis directory
        design_dir: Directory containing pre-generated design matrices
        study_name: Name of analysis study (e.g., 'asl_group', 'age_analysis')
        method: Statistical method ('randomise', 'glm', or 'both')
        n_permutations: Number of randomise permutations
        z_threshold: Z-score threshold for GLM (default: 2.3 ≈ p<0.01)
    """
    # Validate method
    valid_methods = ['randomise', 'glm', 'both']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

    logging.info("\n" + "=" * 80)
    logging.info(f"ASL GROUP-LEVEL ANALYSIS - {study_name}")
    logging.info("=" * 80)

    # Setup output directory with study subdirectory
    output_dir = analysis_dir / 'asl' / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather 4D data (using design matrix subject order)
    image_4d, subject_ids = gather_subject_maps(derivatives_dir, output_dir, design_dir)

    # 2. Create mask
    mask_file = create_mean_mask(image_4d, output_dir)

    # 3. Load pre-generated design matrices
    design_mat, contrast_con, contrast_names = load_design_matrices(
        subject_ids, design_dir, output_dir, image_4d
    )

    # 4. Run statistical analysis (randomise and/or GLM)
    results = {}

    if method in ['randomise', 'both']:
        logging.info("\n" + "=" * 80)
        logging.info("Running FSL Randomise (Nonparametric)")
        logging.info("=" * 80)

        randomise_dir = output_dir / 'randomise_output'

        randomise_result = run_randomise(
            input_file=image_4d,
            design_mat=design_mat,
            contrast_con=contrast_con,
            output_dir=randomise_dir,
            mask=mask_file,
            n_permutations=n_permutations,
            tfce=True,
            seed=42
        )

        randomise_summary = summarize_results(randomise_dir, threshold=0.95)

        results['randomise'] = {
            'result': randomise_result,
            'summary': randomise_summary,
            'output_dir': randomise_dir
        }

        logging.info(f"✓ Randomise completed in {randomise_result['elapsed_time']:.1f} seconds")

    if method in ['glm', 'both']:
        logging.info("\n" + "=" * 80)
        logging.info("Running Nilearn GLM (Parametric)")
        logging.info("=" * 80)

        glm_dir = output_dir / 'glm_output'
        glm_dir.mkdir(parents=True, exist_ok=True)

        # Read FSL design matrix and convert to pandas DataFrame
        with open(design_mat, 'r') as f:
            lines = f.readlines()
            mat_start = [i for i, l in enumerate(lines) if '/Matrix' in l][0] + 1
            design_data = np.loadtxt(lines[mat_start:])

        # Get column names from design summary
        import json
        summary_file = output_dir / 'design_summary.json'
        with open(summary_file) as f:
            summary = json.load(f)
            col_names = summary['columns']

        design_df = pd.DataFrame(design_data, columns=col_names)

        # Read FSL contrasts
        with open(contrast_con, 'r') as f:
            lines = f.readlines()
            mat_start = [i for i, l in enumerate(lines) if '/Matrix' in l][0] + 1
            contrast_data = np.loadtxt(lines[mat_start:])

        if contrast_data.ndim == 1:
            contrast_data = contrast_data.reshape(1, -1)

        contrasts_dict = {name: contrast_data[i] for i, name in enumerate(contrast_names)}

        # Split 4D image into individual subject files
        merged_img = nib.load(str(image_4d))
        merged_data = merged_img.get_fdata()
        n_subjects = merged_data.shape[-1]

        subject_files = []
        for i in range(n_subjects):
            subject_data = merged_data[..., i]
            subject_img = nib.Nifti1Image(subject_data, merged_img.affine, merged_img.header)
            subject_file = glm_dir / f'subject_{i:03d}.nii.gz'
            nib.save(subject_img, str(subject_file))
            subject_files.append(subject_file)

        # Run nilearn GLM
        glm_result = run_second_level_glm(
            input_files=subject_files,
            design_matrix=design_df,
            contrasts=contrasts_dict,
            output_dir=glm_dir,
            mask=mask_file,
            smoothing_fwhm=None,
            n_jobs=-1
        )

        # Summarize with multiple comparison corrections
        glm_summary = summarize_glm_results(
            glm_results=glm_result,
            output_dir=glm_dir,
            alpha=0.05,
            corrections=['fdr', 'bonferroni']
        )

        results['glm'] = {
            'result': glm_result,
            'summary': glm_summary,
            'output_dir': glm_dir
        }

        logging.info(f"✓ GLM completed in {glm_result['elapsed_time']:.1f} seconds")

    # Summary
    logging.info(f"\n✓ ASL CBF analysis complete!")
    if 'randomise' in results:
        logging.info(f"  Randomise results: {results['randomise']['output_dir']}")
    if 'glm' in results:
        logging.info(f"  GLM results: {results['glm']['output_dir']}")

    return results


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run group-level ASL CBF analysis with pre-generated design matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Study root directory'
    )
    parser.add_argument(
        '--design-dir',
        type=Path,
        required=True,
        help='Directory containing pre-generated design matrices (design.mat, design.con, design_summary.json)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='randomise',
        choices=['randomise', 'glm', 'both'],
        help='Statistical method (default: randomise)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=5000,
        help='Number of randomise permutations (default: 5000)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=2.3,
        help='Z-score threshold for GLM (default: 2.3)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='asl_analysis',
        help='Analysis study name (default: asl_analysis)'
    )

    args = parser.parse_args()

    # Derive paths
    derivatives_dir = args.study_root / 'derivatives'
    analysis_dir = args.study_root / 'analysis'

    # Setup logging
    log_file = Path('logs') / 'asl_group_analysis.log'
    log_file.parent.mkdir(exist_ok=True)
    setup_logging(log_file)

    logging.info("=" * 80)
    logging.info("ASL CEREBRAL BLOOD FLOW GROUP ANALYSIS")
    logging.info("=" * 80)
    logging.info(f"Study root: {args.study_root}")
    logging.info(f"Derivatives: {derivatives_dir}")
    logging.info(f"Analysis: {analysis_dir}")
    logging.info(f"Design directory: {args.design_dir}")
    logging.info(f"Method: {args.method}")

    # Validate design files exist
    design_dir = Path(args.design_dir)
    if not (design_dir / 'design.mat').exists():
        logging.error(f"Design matrix not found: {design_dir / 'design.mat'}")
        logging.error("Please run generate_design_matrices.py first to create design files")
        sys.exit(1)

    try:
        # Run ASL CBF analysis
        asl_results = run_group_analysis(
            derivatives_dir=derivatives_dir,
            analysis_dir=analysis_dir,
            design_dir=args.design_dir,
            study_name=args.study_name,
            method=args.method,
            n_permutations=args.n_permutations,
            z_threshold=args.z_threshold
        )

        logging.info("\n" + "=" * 80)
        logging.info("ANALYSIS COMPLETE")
        logging.info("=" * 80)

        if 'randomise' in asl_results:
            logging.info(f"Randomise: {asl_results['randomise']['output_dir']}")
        if 'glm' in asl_results:
            logging.info(f"GLM: {asl_results['glm']['output_dir']}")

    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
