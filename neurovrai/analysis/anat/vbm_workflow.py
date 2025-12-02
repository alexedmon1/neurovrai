#!/usr/bin/env python3
"""
Voxel-Based Morphometry (VBM) Analysis Workflow

Performs group-level statistical analysis of brain structure using tissue probability maps.
Follows FSL VBM-style analysis with spatial normalization, optional modulation, and smoothing.

VBM Workflow:
1. Subject-level preparation:
   - Use tissue segmentation from anatomical preprocessing
   - Normalize tissue maps to MNI152 template space
   - Optional: Modulate by Jacobian determinant to preserve volume
   - Smooth normalized maps (typically 2-8mm FWHM)

2. Group-level analysis:
   - Create design matrix from participant demographics
   - Run FSL randomise for voxel-wise statistics
   - Generate cluster reports with anatomical localization

Usage:
    # Prepare subject data
    python vbm_workflow.py prepare \
        --derivatives-dir /study/derivatives \
        --subjects sub-001 sub-002 ... \
        --output-dir /study/analysis/vbm \
        --tissue GM \
        --smooth-fwhm 4

    # Run group statistics
    python vbm_workflow.py analyze \
        --vbm-dir /study/analysis/vbm \
        --participants /study/participants.tsv \
        --design age+sex \
        --contrasts age_positive,age_negative

Or from Python:
    from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data, run_vbm_analysis

    # Prepare data
    prepare_vbm_data(
        subjects=['sub-001', 'sub-002', ...],
        derivatives_dir=Path('/study/derivatives'),
        output_dir=Path('/study/analysis/vbm'),
        tissue_type='GM',
        smooth_fwhm=4.0
    )

    # Run analysis
    run_vbm_analysis(
        vbm_dir=Path('/study/analysis/vbm'),
        participants_file=Path('/study/participants.tsv'),
        formula='age + sex',
        contrasts={'age_positive': [1, 0, 0]}
    )
"""

import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Union, Tuple
import pandas as pd
import numpy as np

import nibabel as nib
from nipype.interfaces import fsl
from nipype.pipeline import engine as pe
from nipype.interfaces.utility import IdentityInterface, Function

from neurovrai.analysis.stats.design_matrix import create_design_matrix
from neurovrai.analysis.stats.randomise_wrapper import run_randomise
from neurovrai.analysis.stats.glm_wrapper import run_fsl_glm, threshold_zstat, summarize_glm_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_tissue_map(
    tissue_file: Path,
    reference: Path,
    transform_file: Path,
    output_file: Path,
    modulate: bool = False
) -> Path:
    """
    Normalize tissue probability map to standard space

    Args:
        tissue_file: Subject's tissue probability map (e.g., GM)
        reference: Reference template (e.g., MNI152 T1)
        transform_file: Warp field from anatomical preprocessing
        output_file: Output normalized tissue map
        modulate: Whether to modulate by Jacobian (preserves volume)

    Returns:
        Path to normalized tissue map
    """
    logger.info(f"Normalizing {tissue_file.name}...")

    # Detect transform format and use appropriate tool
    if transform_file.suffix == '.h5':
        # Use ANTs for .h5 composite transforms
        from nipype.interfaces import ants

        apply_transform = ants.ApplyTransforms()
        apply_transform.inputs.input_image = str(tissue_file)
        apply_transform.inputs.reference_image = str(reference)
        apply_transform.inputs.transforms = [str(transform_file)]
        apply_transform.inputs.interpolation = 'Linear'
        apply_transform.inputs.output_image = str(output_file)

        result = apply_transform.run()

    else:
        # Use FSL for .nii.gz warp fields
        applywarp = fsl.ApplyWarp()
        applywarp.inputs.in_file = str(tissue_file)
        applywarp.inputs.ref_file = str(reference)
        applywarp.inputs.field_file = str(transform_file)
        applywarp.inputs.out_file = str(output_file)
        applywarp.inputs.output_type = 'NIFTI_GZ'

        if modulate:
            # Modulate by Jacobian determinant to preserve volume
            applywarp.inputs.premat = None  # Use warp field directly
            # Note: FSL's applywarp doesn't have direct modulation
            # For proper modulation, need to compute Jacobian separately
            logger.warning("Modulation requested but not fully implemented - using standard warping")

        result = applywarp.run()

    logger.info(f"  ✓ Created: {output_file}")
    return Path(output_file)


def smooth_tissue_map(
    tissue_file: Path,
    output_file: Path,
    fwhm: float = 4.0
) -> Path:
    """
    Smooth tissue probability map

    Args:
        tissue_file: Input tissue map
        output_file: Output smoothed map
        fwhm: Full-width half-maximum of Gaussian kernel (mm)

    Returns:
        Path to smoothed tissue map
    """
    logger.info(f"Smoothing {tissue_file.name} (FWHM={fwhm}mm)...")

    # Convert FWHM to sigma for fslmaths -s
    # sigma = FWHM / (2 * sqrt(2 * ln(2)))
    import math
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))

    # Use fslmaths for smoothing (more direct control)
    from nipype.interfaces import fsl
    smooth = fsl.ImageMaths()
    smooth.inputs.in_file = str(tissue_file)
    smooth.inputs.op_string = f'-s {sigma:.4f}'
    smooth.inputs.out_file = str(output_file)
    smooth.inputs.output_type = 'NIFTI_GZ'

    result = smooth.run()

    logger.info(f"  ✓ Created: {output_file}")
    return Path(output_file)


def prepare_vbm_data(
    subjects: List[str],
    derivatives_dir: Path,
    output_dir: Path,
    tissue_type: str = 'GM',
    smooth_fwhm: float = 4.0,
    modulate: bool = False,
    reference: Optional[Path] = None,
    mask: Optional[Path] = None
) -> Dict:
    """
    Prepare tissue probability maps for VBM analysis

    Args:
        subjects: List of subject IDs
        derivatives_dir: Base derivatives directory
        output_dir: Output directory for prepared VBM data
        tissue_type: Tissue type to analyze ('GM', 'WM', or 'CSF')
        smooth_fwhm: Smoothing kernel FWHM in mm (0 = no smoothing)
        modulate: Whether to modulate by Jacobian (preserves volume)
        reference: Reference template (default: MNI152 2mm)
        mask: Group mask (will be created if not provided)

    Returns:
        Dictionary with preparation results

    Outputs:
        {output_dir}/
            subjects/
                {subject}_tissue_mni.nii.gz - Normalized tissue maps
                {subject}_tissue_mni_smooth.nii.gz - Smoothed normalized maps
            merged_tissue.nii.gz - 4D merged volume for all subjects
            subject_list.txt - List of subjects included
            group_mask.nii.gz - Group mask
            vbm_info.json - Analysis metadata
    """
    output_dir = Path(output_dir)
    subjects_dir = output_dir / 'subjects'
    subjects_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"vbm_preparation_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("VBM DATA PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Subjects: {len(subjects)}")
    logger.info(f"Tissue type: {tissue_type}")
    logger.info(f"Smoothing: {smooth_fwhm}mm FWHM")
    logger.info(f"Modulation: {modulate}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Map tissue type to segmentation file
    # Support both FSL FAST (pve_*.nii.gz) and ANTs Atropos (POSTERIOR_*.nii.gz) naming
    tissue_map_fast = {
        'GM': 'pve_1.nii.gz',  # Grey matter
        'WM': 'pve_2.nii.gz',  # White matter
        'CSF': 'pve_0.nii.gz'  # CSF
    }

    tissue_map_atropos = {
        'GM': 'POSTERIOR_02.nii.gz',  # Grey matter (tissue class 2)
        'WM': 'POSTERIOR_03.nii.gz',  # White matter (tissue class 3)
        'CSF': 'POSTERIOR_01.nii.gz'  # CSF (tissue class 1)
    }

    if tissue_type not in tissue_map_fast:
        raise ValueError(f"Unknown tissue type: {tissue_type}. Must be one of {list(tissue_map_fast.keys())}")

    # Set defaults
    if reference is None:
        reference = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz')
        logger.info(f"Using default reference: {reference}")

    # Process each subject
    processed_files = []
    missing_subjects = []
    failed_subjects = []

    for subject in subjects:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Processing: {subject}")
        logger.info('=' * 40)

        try:
            # Locate tissue probability map from anatomical preprocessing
            # Try both FSL FAST and ANTs Atropos naming conventions
            tissue_file_fast = derivatives_dir / subject / 'anat' / 'segmentation' / tissue_map_fast[tissue_type]
            tissue_file_atropos = derivatives_dir / subject / 'anat' / 'segmentation' / tissue_map_atropos[tissue_type]

            if tissue_file_fast.exists():
                tissue_file = tissue_file_fast
                logger.info(f"  Found tissue file (FAST): {tissue_file.name}")
            elif tissue_file_atropos.exists():
                tissue_file = tissue_file_atropos
                logger.info(f"  Found tissue file (Atropos): {tissue_file.name}")
            else:
                logger.warning(f"  ✗ Tissue file not found (tried both FAST and Atropos naming)")
                logger.warning(f"    FAST: {tissue_file_fast}")
                logger.warning(f"    Atropos: {tissue_file_atropos}")
                missing_subjects.append(subject)
                continue

            # Locate transform (warp field) from anatomical preprocessing
            # Try different possible locations: derivatives, transforms registry, etc.
            study_root = derivatives_dir.parent if (derivatives_dir.parent / 'transforms').exists() else derivatives_dir
            possible_transforms = [
                derivatives_dir / subject / 'anat' / 'transforms' / 'highres2standard_warp.nii.gz',
                derivatives_dir / subject / 'anat' / 'highres2standard_warp.nii.gz',
                derivatives_dir / subject / 'anat' / 'transforms' / 'ants_Composite.h5',
                study_root / 'transforms' / subject / 'T1w_to_MNI152_composite.h5',
                study_root / 'transforms' / subject / 'T1w_to_MNI152_warp.nii.gz',
            ]

            transform_file = None
            for tf in possible_transforms:
                if tf.exists():
                    transform_file = tf
                    logger.info(f"  Found transform: {tf}")
                    break

            if transform_file is None:
                logger.warning(f"  ✗ Transform not found for {subject}")
                logger.warning(f"    Tried: {[str(tf) for tf in possible_transforms[:3]]}")
                missing_subjects.append(subject)
                continue

            # Normalize tissue map to MNI space
            normalized_file = subjects_dir / f"{subject}_{tissue_type}_mni.nii.gz"
            normalize_tissue_map(
                tissue_file=tissue_file,
                reference=reference,
                transform_file=transform_file,
                output_file=normalized_file,
                modulate=modulate
            )

            # Smooth if requested
            if smooth_fwhm > 0:
                smoothed_file = subjects_dir / f"{subject}_{tissue_type}_mni_smooth.nii.gz"
                smooth_tissue_map(
                    tissue_file=normalized_file,
                    output_file=smoothed_file,
                    fwhm=smooth_fwhm
                )
                final_file = smoothed_file
            else:
                final_file = normalized_file

            processed_files.append(str(final_file))
            logger.info(f"  ✓ {subject} complete")

        except Exception as e:
            logger.error(f"  ✗ Failed to process {subject}: {e}", exc_info=True)
            failed_subjects.append(subject)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Processed: {len(processed_files)}/{len(subjects)}")
    logger.info(f"Missing data: {len(missing_subjects)}")
    logger.info(f"Failed: {len(failed_subjects)}")

    if len(processed_files) < 2:
        raise ValueError(f"Need at least 2 subjects for VBM analysis, only processed {len(processed_files)}")

    # Merge into 4D volume
    logger.info("\nMerging tissue maps...")
    merged_file = output_dir / f"merged_{tissue_type}.nii.gz"

    merge = fsl.Merge()
    merge.inputs.in_files = processed_files
    merge.inputs.dimension = 't'
    merge.inputs.merged_file = str(merged_file)
    merge.inputs.output_type = 'NIFTI_GZ'
    merge.run()

    logger.info(f"✓ Created 4D merged volume: {merged_file}")

    # Create group mask if not provided
    if mask is None:
        logger.info("\nCreating group mask...")
        mask = output_dir / 'group_mask.nii.gz'

        # Create mask by thresholding mean image
        maths = fsl.ImageMaths()
        maths.inputs.in_file = str(merged_file)
        maths.inputs.op_string = '-Tmean -thr 0.1 -bin'
        maths.inputs.out_file = str(mask)
        maths.inputs.output_type = 'NIFTI_GZ'
        maths.run()

        logger.info(f"✓ Created group mask: {mask}")

    # Save subject list
    subject_list_file = output_dir / 'subject_list.txt'
    with open(subject_list_file, 'w') as f:
        for i, pf in enumerate(processed_files):
            subject_id = subjects[i] if i < len(subjects) else f"subject_{i}"
            f.write(f'{subject_id}\t{pf}\n')

    # Save metadata
    vbm_info = {
        'timestamp': timestamp,
        'n_subjects': len(processed_files),
        'tissue_type': tissue_type,
        'smooth_fwhm': smooth_fwhm,
        'modulate': modulate,
        'reference': str(reference),
        'merged_file': str(merged_file),
        'mask_file': str(mask),
        'subject_list': str(subject_list_file),
        'processed_subjects': [Path(f).stem.split('_')[0] for f in processed_files],
        'missing_subjects': missing_subjects,
        'failed_subjects': failed_subjects
    }

    info_file = output_dir / 'vbm_info.json'
    with open(info_file, 'w') as f:
        json.dump(vbm_info, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("VBM DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Merged data: {merged_file}")
    logger.info(f"Group mask: {mask}")
    logger.info(f"Metadata: {info_file}")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 80 + "\n")

    return vbm_info


def run_vbm_analysis(
    vbm_dir: Path,
    participants_file: Path,
    formula: str,
    contrasts: Dict[str, List[float]],
    method: str = 'randomise',
    n_permutations: int = 5000,
    tfce: bool = True,
    cluster_threshold: float = 0.95,
    z_threshold: float = 2.3
) -> Dict:
    """
    Run group-level VBM statistical analysis

    Args:
        vbm_dir: Directory containing prepared VBM data (from prepare_vbm_data)
        participants_file: TSV file with participant demographics
                          Must have 'participant_id' column
        formula: Design formula (e.g., 'age + sex')
        contrasts: Dictionary of contrast names to contrast vectors
                  e.g., {'age_positive': [1, 0, 0], 'age_negative': [-1, 0, 0]}
        method: Statistical method ('randomise', 'glm', or 'both')
        n_permutations: Number of permutations for randomise
        tfce: Use Threshold-Free Cluster Enhancement (randomise only)
        cluster_threshold: Cluster significance threshold (default: 0.95 = p<0.05 for randomise)
        z_threshold: Z-score threshold for GLM (default: 2.3 ≈ p<0.01)

    Returns:
        Dictionary with analysis results

    Outputs:
        {vbm_dir}/stats/
            design.mat, design.con - Design matrix and contrasts
            randomise_output/ - Statistical maps from randomise (if method='randomise' or 'both')
            glm_output/ - Statistical maps from GLM (if method='glm' or 'both')
            cluster_reports/ - Cluster tables and visualizations
    """
    # Validate method
    valid_methods = ['randomise', 'glm', 'both']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
    vbm_dir = Path(vbm_dir)
    stats_dir = vbm_dir / 'stats'
    stats_dir.mkdir(exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = stats_dir / f"vbm_analysis_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("VBM GROUP ANALYSIS")
    logger.info("=" * 80)

    # Load VBM metadata
    info_file = vbm_dir / 'vbm_info.json'
    if not info_file.exists():
        raise FileNotFoundError(f"VBM info file not found: {info_file}. Run prepare_vbm_data first.")

    with open(info_file) as f:
        vbm_info = json.load(f)

    merged_file = Path(vbm_info['merged_file'])
    mask_file = Path(vbm_info['mask_file'])
    processed_subjects = vbm_info['processed_subjects']

    logger.info(f"Input data: {merged_file}")
    logger.info(f"Mask: {mask_file}")
    logger.info(f"Subjects: {len(processed_subjects)}")
    logger.info(f"Formula: {formula}")
    logger.info(f"Contrasts: {list(contrasts.keys())}")
    logger.info("")

    # Load participant demographics
    participants_df = pd.read_csv(participants_file, sep='\t')

    # Filter to processed subjects
    participants_df = participants_df[participants_df['participant_id'].isin(processed_subjects)]
    participants_df = participants_df.set_index('participant_id')
    participants_df = participants_df.loc[processed_subjects]  # Ensure same order

    logger.info(f"Loaded demographics for {len(participants_df)} subjects")

    # Create design matrix
    logger.info("\n" + "=" * 80)
    logger.info("CREATING DESIGN MATRIX")
    logger.info("=" * 80)

    # Call design matrix function with correct parameters
    design_mat, column_names = create_design_matrix(
        df=participants_df,
        formula=formula,
        demean_continuous=True,
        add_intercept=True
    )

    # Save design matrix in FSL vest format
    design_mat_file = stats_dir / 'design.mat'
    with open(design_mat_file, 'w') as f:
        f.write(f"/NumWaves {design_mat.shape[1]}\n")
        f.write(f"/NumPoints {design_mat.shape[0]}\n")
        f.write("/Matrix\n")
        np.savetxt(f, design_mat, fmt='%.6f')
    logger.info(f"Design matrix shape: {design_mat.shape}")
    logger.info(f"Columns: {column_names}")
    logger.info(f"✓ Saved: {design_mat_file}")

    # Create contrasts matrix
    n_predictors = design_mat.shape[1]
    contrast_mat = []
    contrast_names = []

    for contrast_name, contrast_vec in contrasts.items():
        if len(contrast_vec) != n_predictors:
            raise ValueError(
                f"Contrast '{contrast_name}' has {len(contrast_vec)} values "
                f"but design matrix has {n_predictors} predictors"
            )
        contrast_mat.append(contrast_vec)
        contrast_names.append(contrast_name)
        logger.info(f"  Contrast: {contrast_name} = {contrast_vec}")

    contrast_mat = np.array(contrast_mat)

    # Save contrasts in FSL vest format
    design_con_file = stats_dir / 'design.con'
    with open(design_con_file, 'w') as f:
        f.write(f"/NumWaves {contrast_mat.shape[1]}\n")
        f.write(f"/NumContrasts {contrast_mat.shape[0]}\n")
        f.write("/Matrix\n")
        np.savetxt(f, contrast_mat, fmt='%.6f')
    logger.info(f"✓ Saved: {design_con_file} ({len(contrast_names)} contrasts)")

    # Save contrast names for reference
    contrast_names_file = stats_dir / 'contrast_names.txt'
    with open(contrast_names_file, 'w') as f:
        f.write('\n'.join(contrast_names))
    logger.info(f"✓ Saved: {contrast_names_file}")

    # Initialize results
    randomise_result = None
    glm_result = None

    # Run statistical analysis (randomise and/or GLM)
    if method in ['randomise', 'both']:
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING RANDOMISE (Nonparametric)")
        logger.info("=" * 80)

        randomise_dir = stats_dir / 'randomise_output'
        randomise_dir.mkdir(exist_ok=True)

        randomise_result = run_randomise(
            input_file=merged_file,
            design_mat=design_mat_file,
            contrast_con=design_con_file,
            mask=mask_file,
            output_dir=randomise_dir,
            n_permutations=n_permutations,
            tfce=tfce
        )

        logger.info(f"✓ Randomise complete: {randomise_dir}")

    if method in ['glm', 'both']:
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING GLM (Parametric)")
        logger.info("=" * 80)

        glm_dir = stats_dir / 'glm_output'
        glm_dir.mkdir(exist_ok=True)

        glm_result = run_fsl_glm(
            input_file=merged_file,
            design_mat=design_mat_file,
            contrast_con=design_con_file,
            mask=mask_file,
            output_dir=glm_dir
        )

        logger.info(f"✓ GLM complete: {glm_dir}")

        # Threshold GLM results
        threshold_dir = glm_dir / 'thresholded'
        threshold_result = threshold_zstat(
            zstat_file=Path(glm_result['output_files']['zstat']),
            output_dir=threshold_dir,
            z_threshold=z_threshold,
            cluster_threshold=10,
            mask=mask_file
        )

        # Summarize GLM results
        glm_summary = summarize_glm_results(
            output_dir=glm_dir,
            contrast_names=contrast_names,
            z_threshold=z_threshold
        )

    # Generate cluster reports for randomise results (if available)
    cluster_results = {}

    if randomise_result:
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING CLUSTER REPORTS (Randomise)")
        logger.info("=" * 80)

        cluster_reports_dir = stats_dir / 'cluster_reports_randomise'
        cluster_reports_dir.mkdir(exist_ok=True)

        # Import enhanced cluster reporting (with atlas localization)
        from neurovrai.analysis.stats.enhanced_cluster_report import create_enhanced_cluster_report

        for idx, contrast_name in enumerate(contrasts.keys(), start=1):
            logger.info(f"\nProcessing contrast: {contrast_name}")

            # Find corresponding statistical maps
            stat_map = randomise_dir / f"randomise_tstat{idx}.nii.gz"
            corrp_map = randomise_dir / f"randomise_tfce_corrp_tstat{idx}.nii.gz"

            if not stat_map.exists() or not corrp_map.exists():
                logger.warning(f"  Statistical maps not found for {contrast_name}")
                continue

            # Create enhanced cluster report with visualization
            try:
                report = create_enhanced_cluster_report(
                    stat_map=stat_map,
                    corrp_map=corrp_map,
                    threshold=1.0 - cluster_threshold,  # Convert to p-value (e.g., 0.95 -> 0.05)
                    output_dir=cluster_reports_dir,
                    contrast_name=contrast_name,
                    max_clusters=10,  # Top 10 clusters
                    liberal_threshold=0.7,  # p<0.3 for exploratory view
                    background_image=None  # Will use MNI152 T1 template
                )
                cluster_results[contrast_name] = report
                logger.info(f"  ✓ Report: {report.get('report_html', 'N/A')}")

            except Exception as e:
                logger.error(f"  ✗ Failed to create report: {e}", exc_info=True)

    # Save results summary
    results = {
        'timestamp': timestamp,
        'vbm_dir': str(vbm_dir),
        'n_subjects': len(processed_subjects),
        'tissue_type': vbm_info['tissue_type'],
        'formula': formula,
        'contrasts': list(contrasts.keys()),
        'method': method,
        'design_matrix': str(design_mat_file),
        'design_contrasts': str(design_con_file)
    }

    # Add method-specific results
    if randomise_result:
        results.update({
            'n_permutations': n_permutations,
            'tfce': tfce,
            'randomise_dir': str(randomise_dir),
            'cluster_reports': {k: v['report_html'] for k, v in cluster_results.items()} if cluster_results else {}
        })

    if glm_result:
        results.update({
            'z_threshold': z_threshold,
            'glm_dir': str(glm_dir),
            'glm_summary': glm_summary
        })

    results_file = stats_dir / 'vbm_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "=" * 80)
    logger.info("VBM ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results: {results_file}")

    if randomise_result:
        logger.info(f"Randomise output: {randomise_dir}")
        if cluster_results:
            logger.info(f"Cluster reports (randomise): {cluster_reports_dir}")

    if glm_result:
        logger.info(f"GLM output: {glm_dir}")
        logger.info(f"  ✓ {len([c for c in glm_summary['contrasts'] if c['significant']])} / {len(glm_summary['contrasts'])} significant contrasts (z > {z_threshold})")

    logger.info(f"Log: {log_file}")
    logger.info("=" * 80 + "\n")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="VBM analysis workflow"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare VBM data')
    prepare_parser.add_argument(
        '--derivatives-dir',
        type=Path,
        required=True,
        help='Derivatives directory'
    )
    prepare_parser.add_argument(
        '--subjects',
        nargs='+',
        required=True,
        help='List of subject IDs'
    )
    prepare_parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )
    prepare_parser.add_argument(
        '--tissue',
        choices=['GM', 'WM', 'CSF'],
        default='GM',
        help='Tissue type (default: GM)'
    )
    prepare_parser.add_argument(
        '--smooth-fwhm',
        type=float,
        default=4.0,
        help='Smoothing FWHM in mm (default: 4.0, 0=no smoothing)'
    )
    prepare_parser.add_argument(
        '--modulate',
        action='store_true',
        help='Modulate by Jacobian to preserve volume'
    )

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run VBM group analysis')
    analyze_parser.add_argument(
        '--vbm-dir',
        type=Path,
        required=True,
        help='VBM directory (from prepare step)'
    )
    analyze_parser.add_argument(
        '--participants',
        type=Path,
        required=True,
        help='Participants TSV file'
    )
    analyze_parser.add_argument(
        '--design',
        type=str,
        required=True,
        help='Design formula (e.g., "age + sex")'
    )
    analyze_parser.add_argument(
        '--contrasts',
        type=str,
        required=True,
        help='Comma-separated contrast names (e.g., "age_positive,age_negative")'
    )
    analyze_parser.add_argument(
        '--method',
        type=str,
        choices=['randomise', 'glm', 'both'],
        default='randomise',
        help='Statistical method (default: randomise)'
    )
    analyze_parser.add_argument(
        '--n-permutations',
        type=int,
        default=5000,
        help='Number of permutations for randomise (default: 5000)'
    )
    analyze_parser.add_argument(
        '--no-tfce',
        action='store_true',
        help='Disable TFCE correction (randomise only)'
    )
    analyze_parser.add_argument(
        '--z-threshold',
        type=float,
        default=2.3,
        help='Z-score threshold for GLM (default: 2.3, approx p<0.01)'
    )

    args = parser.parse_args()

    if args.command == 'prepare':
        # Prepare VBM data
        results = prepare_vbm_data(
            subjects=args.subjects,
            derivatives_dir=args.derivatives_dir,
            output_dir=args.output_dir,
            tissue_type=args.tissue,
            smooth_fwhm=args.smooth_fwhm,
            modulate=args.modulate
        )

        print("\n" + "=" * 80)
        print("VBM DATA PREPARATION COMPLETE")
        print("=" * 80)
        print(f"Processed subjects: {results['n_subjects']}")
        print(f"Tissue type: {results['tissue_type']}")
        print(f"Output: {results['merged_file']}")
        print("=" * 80)

    elif args.command == 'analyze':
        # Parse contrasts
        contrast_names = [c.strip() for c in args.contrasts.split(',')]

        # Create simple contrasts (will need to be customized based on design matrix)
        # For now, assume first column is intercept, second is main effect
        contrasts = {}
        for i, name in enumerate(contrast_names):
            if 'positive' in name.lower():
                contrasts[name] = [0, 1] + [0] * (len(contrast_names) - 1)
            elif 'negative' in name.lower():
                contrasts[name] = [0, -1] + [0] * (len(contrast_names) - 1)

        # Run VBM analysis
        results = run_vbm_analysis(
            vbm_dir=args.vbm_dir,
            participants_file=args.participants,
            formula=args.design,
            contrasts=contrasts,
            method=args.method,
            n_permutations=args.n_permutations,
            tfce=not args.no_tfce,
            z_threshold=args.z_threshold
        )

        print("\n" + "=" * 80)
        print("VBM ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Subjects: {results['n_subjects']}")
        print(f"Contrasts: {', '.join(results['contrasts'])}")
        print(f"Results: {results['randomise_dir']}")
        print("=" * 80)

    else:
        parser.print_help()
