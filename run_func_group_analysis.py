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

from neuroaider import DesignHelper
from neurovrai.analysis.stats.randomise_wrapper import run_randomise, summarize_results
from neurovrai.analysis.stats.nilearn_glm import run_second_level_glm, summarize_glm_results


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
    formula: str,
    output_dir: Path
):
    """
    Create FSL design matrices and contrasts using neuroaider

    Args:
        subject_ids: List of subject IDs in 4D image order
        participants_file: Path to participants.tsv
        formula: Model formula (e.g., 'mriglu+sex+age' for binary group comparison)
        output_dir: Output directory
    """
    logging.info("\nCreating design matrices with neuroaider...")
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
            raise ValueError(f"Subject {subject_id} not found in participants file")
        df_ordered = pd.concat([df_ordered, match.iloc[0:1]], ignore_index=True)

    logging.info(f"✓ Matched {len(df_ordered)} subjects to 4D image order")

    # Parse formula to detect binary groups
    formula_terms = [t.strip() for t in formula.split('+')]
    first_var = formula_terms[0].replace('C(', '').replace(')', '')

    # Detect binary categorical for no-intercept dummy coding
    use_binary_coding = False
    if first_var in df_ordered.columns:
        n_levels = df_ordered[first_var].nunique()
        if n_levels == 2:
            use_binary_coding = True
            logging.info(f"✓ Detected binary group variable '{first_var}' with {n_levels} levels")
            logging.info(f"✓ Using dummy coding WITHOUT intercept for direct group comparison")

    # Initialize DesignHelper
    helper = DesignHelper(
        participants_file=df_ordered,
        subject_column='participant_id',
        add_intercept=not use_binary_coding
    )

    # Add variables from formula
    for term in formula_terms:
        var_name = term.replace('C(', '').replace(')', '').strip()

        if var_name not in df_ordered.columns:
            raise ValueError(f"Variable '{var_name}' not found in participants file")

        # Determine if categorical or continuous
        if pd.api.types.is_numeric_dtype(df_ordered[var_name]):
            n_unique = df_ordered[var_name].nunique()
            if n_unique <= 10 and use_binary_coding and var_name == first_var:
                helper.add_categorical(var_name, coding='dummy')
                logging.info(f"✓ Added categorical: {var_name} (dummy coding, no intercept)")
            elif n_unique <= 10:
                logging.warning(f"Variable '{var_name}' has {n_unique} unique values - treating as continuous")
                helper.add_covariate(var_name, mean_center=True)
            else:
                helper.add_covariate(var_name, mean_center=True)
                logging.info(f"✓ Added covariate: {var_name} (mean-centered)")
        else:
            helper.add_categorical(var_name, coding='effect' if not use_binary_coding else 'dummy')
            logging.info(f"✓ Added categorical: {var_name}")

    # Build design matrix
    design_mat, column_names = helper.build_design_matrix()

    # Auto-generate contrasts for binary groups
    if use_binary_coding:
        logging.info(f"✓ Auto-generating binary group contrasts for '{first_var}'")
        helper.add_binary_group_contrasts(first_var)

    # Add covariate contrasts if present
    for term in formula_terms:
        var_name = term.replace('C(', '').replace(')', '').strip()
        if var_name in column_names and var_name != first_var:
            # Add positive and negative contrasts for continuous covariates
            helper.add_contrast(f"{var_name}_positive", covariate=var_name, direction='+')
            helper.add_contrast(f"{var_name}_negative", covariate=var_name, direction='-')
            logging.info(f"✓ Added contrasts for covariate: {var_name}")

    # Save design files
    design_file = output_dir / 'design.mat'
    contrast_file = output_dir / 'design.con'
    summary_file = output_dir / 'design_summary.json'

    helper.save(
        design_mat_file=design_file,
        design_con_file=contrast_file,
        summary_file=summary_file
    )

    logging.info(f"✓ Design matrix: {design_file}")
    logging.info(f"  Subjects: {len(df_ordered)}")
    logging.info(f"  Predictors: {len(column_names)}")
    logging.info(f"  Columns: {column_names}")

    # Get contrast information
    _, contrast_names = helper.build_contrast_matrix()
    logging.info(f"✓ Contrast matrix: {contrast_file}")
    logging.info(f"  Contrasts ({len(contrast_names)}): {contrast_names}")

    return design_file, contrast_file


def run_group_analysis(
    metric: str,
    derivatives_dir: Path,
    analysis_dir: Path,
    participants_file: Path,
    formula: str,
    study_name: str = 'mock_study',
    method: str = 'randomise',
    n_permutations: int = 500,
    z_threshold: float = 2.3
):
    """
    Run complete group-level analysis for ReHo or fALFF

    Args:
        metric: 'reho' or 'falff'
        derivatives_dir: Path to derivatives directory
        analysis_dir: Path to analysis directory
        participants_file: Path to participants file
        formula: Model formula (e.g., 'mriglu+sex+age' for binary group comparison)
        study_name: Name of analysis study (e.g., 'mock_study', 'age_analysis')
        method: Statistical method ('randomise', 'glm', or 'both')
        n_permutations: Number of randomise permutations
        z_threshold: Z-score threshold for GLM (default: 2.3 ≈ p<0.01)
    """
    # Validate method
    valid_methods = ['randomise', 'glm', 'both']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
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
        subject_ids, participants_file, formula, output_dir
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
        import nibabel as nib
        with open(design_mat, 'r') as f:
            lines = f.readlines()
            # Skip header lines and read matrix
            mat_start = [i for i, l in enumerate(lines) if '/Matrix' in l][0] + 1
            design_data = np.loadtxt(lines[mat_start:])

        design_df = pd.DataFrame(design_data, columns=['intercept', 'age', 'group'])

        # Read FSL contrasts and convert to dictionary
        with open(contrast_con, 'r') as f:
            lines = f.readlines()
            mat_start = [i for i, l in enumerate(lines) if '/Matrix' in l][0] + 1
            contrast_data = np.loadtxt(lines[mat_start:])

        if contrast_data.ndim == 1:
            contrast_data = contrast_data.reshape(1, -1)

        contrast_names = ['age_positive', 'age_negative', 'group1_gt_group0', 'group0_gt_group1']
        contrasts_dict = {name: contrast_data[i] for i, name in enumerate(contrast_names[:len(contrast_data)])}

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
    logging.info(f"\n✓ {metric.upper()} analysis complete!")
    if 'randomise' in results:
        logging.info(f"  Randomise results: {results['randomise']['output_dir']}")
    if 'glm' in results:
        logging.info(f"  GLM results: {results['glm']['output_dir']}")

    return results


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

    # Model specification
    formula = 'mriglu+sex+age'  # Binary group comparison with covariates
    method = 'randomise'  # Change to 'glm' or 'both' as needed

    try:
        # Run ReHo analysis
        reho_results = run_group_analysis(
            metric='reho',
            derivatives_dir=derivatives_dir,
            analysis_dir=analysis_dir,
            participants_file=participants_file,
            formula=formula,
            study_name='mock_study',
            method=method,
            n_permutations=500
        )

        # Run fALFF analysis
        falff_results = run_group_analysis(
            metric='falff',
            derivatives_dir=derivatives_dir,
            analysis_dir=analysis_dir,
            participants_file=participants_file,
            formula=formula,
            study_name='mock_study',
            method=method,
            n_permutations=500
        )

        logging.info("\n" + "=" * 80)
        logging.info("ALL ANALYSES COMPLETE")
        logging.info("=" * 80)

        if 'randomise' in reho_results:
            logging.info(f"ReHo (randomise): {reho_results['randomise']['output_dir']}")
        if 'glm' in reho_results:
            logging.info(f"ReHo (GLM): {reho_results['glm']['output_dir']}")

        if 'randomise' in falff_results:
            logging.info(f"fALFF (randomise): {falff_results['randomise']['output_dir']}")
        if 'glm' in falff_results:
            logging.info(f"fALFF (GLM): {falff_results['glm']['output_dir']}")

    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
