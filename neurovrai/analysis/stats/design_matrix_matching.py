#!/usr/bin/env python3
"""
Design Matrix Matching Utilities

Functions to match design matrices with available MRI data for different analyses.
Ensures that design matrix rows correspond to the correct subjects/images.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_design_matrix_by_subjects(
    design_mat: np.ndarray,
    column_names: List[str],
    participants_df: pd.DataFrame,
    available_subjects: List[str],
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Filter and reorder design matrix to match available subjects

    This ensures the design matrix rows correspond exactly to the subjects
    that have data for a specific analysis.

    Args:
        design_mat: Full design matrix (n_subjects × n_predictors)
        column_names: Predictor names
        participants_df: DataFrame with participant_id as index and covariates
        available_subjects: List of subject IDs that have data for this analysis
        output_dir: Optional directory to save filtered design files

    Returns:
        Dictionary with:
            - design_mat: Filtered design matrix
            - column_names: Predictor names (unchanged)
            - participants_df: Filtered participants DataFrame
            - matched_subjects: List of subjects in final design (in order)
            - n_subjects_original: Original number of subjects
            - n_subjects_matched: Number of matched subjects
            - excluded_subjects: Subjects that were excluded

    Raises:
        ValueError: If no subjects match or if required subjects missing
    """
    logger.info("Filtering design matrix to match available subjects...")

    # Get original subject list from participants_df
    original_subjects = participants_df.index.tolist()

    # Find intersection of subjects (those in both design and available)
    matched_subjects = [s for s in original_subjects if s in available_subjects]
    excluded_subjects = [s for s in original_subjects if s not in available_subjects]

    logger.info(f"  Original design: {len(original_subjects)} subjects")
    logger.info(f"  Available data: {len(available_subjects)} subjects")
    logger.info(f"  Matched: {len(matched_subjects)} subjects")

    if len(excluded_subjects) > 0:
        logger.warning(f"  Excluded {len(excluded_subjects)} subjects without data:")
        for subj in excluded_subjects:
            logger.warning(f"    - {subj}")

    # Check if any available subjects are missing from design
    missing_from_design = [s for s in available_subjects if s not in original_subjects]
    if len(missing_from_design) > 0:
        logger.warning(f"  {len(missing_from_design)} subjects have data but no demographics:")
        for subj in missing_from_design:
            logger.warning(f"    - {subj}")

    if len(matched_subjects) == 0:
        raise ValueError("No subjects match between design matrix and available data!")

    if len(matched_subjects) < 3:
        raise ValueError(f"Too few matched subjects ({len(matched_subjects)}). Need at least 3 for statistics.")

    # Filter and reorder participants_df
    filtered_df = participants_df.loc[matched_subjects].copy()

    # Filter and reorder design_mat
    # Get indices of matched subjects in original order
    original_subject_indices = {subj: i for i, subj in enumerate(original_subjects)}
    matched_indices = [original_subject_indices[subj] for subj in matched_subjects]

    filtered_design = design_mat[matched_indices, :]

    # Save filtered design if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save design matrix
        design_file = output_dir / 'design_filtered.mat'
        with open(design_file, 'w') as f:
            f.write(f"/NumWaves {filtered_design.shape[1]}\n")
            f.write(f"/NumPoints {filtered_design.shape[0]}\n")
            f.write("/Matrix\n")
            np.savetxt(f, filtered_design, fmt='%.6f')
        logger.info(f"  ✓ Saved: {design_file}")

        # Save subject list
        subject_list_file = output_dir / 'subject_list_filtered.txt'
        with open(subject_list_file, 'w') as f:
            f.write('\n'.join(matched_subjects))
        logger.info(f"  ✓ Saved: {subject_list_file}")

        # Save filtered participants
        participants_file = output_dir / 'participants_filtered.tsv'
        filtered_df.to_csv(participants_file, sep='\t')
        logger.info(f"  ✓ Saved: {participants_file}")

    return {
        'design_mat': filtered_design,
        'column_names': column_names,
        'participants_df': filtered_df,
        'matched_subjects': matched_subjects,
        'n_subjects_original': len(original_subjects),
        'n_subjects_matched': len(matched_subjects),
        'excluded_subjects': excluded_subjects,
        'missing_from_design': missing_from_design
    }


def discover_subjects_for_analysis(
    derivatives_dir: Path,
    analysis_type: str,
    metric: Optional[str] = None
) -> List[str]:
    """
    Automatically discover subjects that have data for a specific analysis

    Args:
        derivatives_dir: Path to derivatives directory
        analysis_type: Type of analysis
            - 'tbss': DTI preprocessing with FA for TBSS
            - 'vbm': Anatomical segmentation for VBM
            - 'dti': DTI metrics (FA, MD, AD, RD)
            - 'dki': Diffusion Kurtosis Imaging metrics
            - 'noddi': NODDI microstructure metrics
            - 'func': Functional metrics (requires metric parameter)
        metric: For functional analysis, specify 'reho' or 'falff'

    Returns:
        List of subject IDs with available data

    Example:
        # DTI for TBSS
        subjects = discover_subjects_for_analysis(
            Path('/study/derivatives'),
            'tbss'
        )

        # NODDI analysis
        subjects = discover_subjects_for_analysis(
            Path('/study/derivatives'),
            'noddi'
        )

        # Functional ReHo
        subjects = discover_subjects_for_analysis(
            Path('/study/derivatives'),
            'func',
            metric='reho'
        )
    """
    logger.info(f"Discovering subjects for {analysis_type}...")

    derivatives_dir = Path(derivatives_dir)
    subjects = []

    if analysis_type == 'tbss':
        # Look for preprocessed DWI data with DTI metrics (for TBSS preparation)
        for subject_dir in sorted(derivatives_dir.glob('*/dwi')):
            # Check for FA file from DTI preprocessing
            fa_file = subject_dir / 'dti' / 'dtifit__FA.nii.gz'
            if not fa_file.exists():
                # Try alternative naming
                fa_file = subject_dir / 'dti' / 'dti_FA.nii.gz'
            if fa_file.exists():
                subject_id = subject_dir.parent.name
                subjects.append(subject_id)

    elif analysis_type == 'vbm':
        # Look for tissue segmentation files (FAST or Atropos)
        # VBM needs tissue probability maps for normalization
        for subject_dir in sorted(derivatives_dir.glob('*/anat')):
            seg_dir = subject_dir / 'segmentation'
            if not seg_dir.exists():
                continue

            # Check for GM probability map (pve_1 from FAST or POSTERIOR_02 from Atropos)
            gm_fast = seg_dir / 'pve_1.nii.gz'
            gm_atropos = seg_dir / 'POSTERIOR_02.nii.gz'

            if gm_fast.exists() or gm_atropos.exists():
                subject_id = subject_dir.parent.name
                subjects.append(subject_id)

    elif analysis_type == 'func':
        if metric not in ['reho', 'falff']:
            raise ValueError("For functional analysis, must specify metric='reho' or 'falff'")

        # Look for ReHo or fALFF maps
        map_name = f'{metric}_mni_zscore_masked.nii.gz'
        for subject_dir in sorted(derivatives_dir.glob('*/func')):
            metric_file = subject_dir / metric / map_name
            if metric_file.exists():
                subject_id = subject_dir.parent.name
                subjects.append(subject_id)

    elif analysis_type == 'dti':
        # Look for DTI metrics (FA, MD, etc.)
        for subject_dir in sorted(derivatives_dir.glob('*/dwi')):
            fa_file = subject_dir / 'dti' / 'dtifit__FA.nii.gz'
            if not fa_file.exists():
                fa_file = subject_dir / 'dti' / 'dti_FA.nii.gz'
            if fa_file.exists():
                subject_id = subject_dir.parent.name
                subjects.append(subject_id)

    elif analysis_type == 'dki':
        # Look for DKI (Diffusion Kurtosis Imaging) metrics
        for subject_dir in sorted(derivatives_dir.glob('*/dwi')):
            dki_dir = subject_dir / 'dki'
            # Check for MK (mean kurtosis) file
            mk_file = dki_dir / 'dki_mk.nii.gz'
            if mk_file.exists():
                subject_id = subject_dir.parent.name
                subjects.append(subject_id)

    elif analysis_type == 'noddi':
        # Look for NODDI metrics (FICVF, ODI, etc.)
        for subject_dir in sorted(derivatives_dir.glob('*/dwi')):
            noddi_dir = subject_dir / 'noddi'
            # Check for FICVF (neurite density) file
            ficvf_file = noddi_dir / 'ficvf.nii.gz'
            if not ficvf_file.exists():
                # Try AMICO naming
                ficvf_file = noddi_dir / 'FIT_ICVF.nii.gz'
            if ficvf_file.exists():
                subject_id = subject_dir.parent.name
                subjects.append(subject_id)

    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    logger.info(f"  Found {len(subjects)} subjects with {analysis_type} data")

    return subjects


def validate_design_data_match(
    design_mat: np.ndarray,
    subject_list: List[str],
    data_file: Optional[Path] = None,
    data_4d_shape: Optional[Tuple] = None
) -> Dict:
    """
    Validate that design matrix matches the data being analyzed

    Args:
        design_mat: Design matrix (n_subjects × n_predictors)
        subject_list: List of subjects in analysis (in order)
        data_file: Optional path to 4D data file to check
        data_4d_shape: Optional shape tuple if data already loaded

    Returns:
        Dictionary with validation results:
            - valid: True if all checks pass
            - errors: List of error messages
            - warnings: List of warning messages
            - n_subjects_design: Number of subjects in design
            - n_subjects_data: Number of subjects in data (if checked)

    Raises:
        ValueError: If critical validation fails
    """
    logger.info("Validating design matrix matches data...")

    errors = []
    warnings = []

    n_subjects_design = design_mat.shape[0]
    n_subjects_list = len(subject_list)

    # Check design matrix matches subject list
    if n_subjects_design != n_subjects_list:
        errors.append(
            f"Design matrix has {n_subjects_design} rows but "
            f"subject list has {n_subjects_list} entries"
        )

    # Check 4D data if provided
    n_subjects_data = None
    if data_file is not None:
        import nibabel as nib
        try:
            img = nib.load(data_file)
            shape = img.shape
            if len(shape) == 4:
                n_subjects_data = shape[3]
            else:
                errors.append(f"Data file is not 4D: {shape}")
        except Exception as e:
            errors.append(f"Failed to load data file: {e}")

    elif data_4d_shape is not None:
        if len(data_4d_shape) == 4:
            n_subjects_data = data_4d_shape[3]
        else:
            errors.append(f"Data shape is not 4D: {data_4d_shape}")

    # Check data matches design
    if n_subjects_data is not None:
        if n_subjects_design != n_subjects_data:
            errors.append(
                f"Design matrix has {n_subjects_design} rows but "
                f"4D data has {n_subjects_data} volumes"
            )

    # Check for duplicate subjects
    if len(subject_list) != len(set(subject_list)):
        duplicates = [s for s in set(subject_list) if subject_list.count(s) > 1]
        errors.append(f"Duplicate subjects in list: {duplicates}")

    # Check minimum sample size
    if n_subjects_design < 3:
        errors.append(f"Too few subjects for statistics: {n_subjects_design}")
    elif n_subjects_design < 10:
        warnings.append(f"Small sample size: {n_subjects_design} subjects")

    # Report results
    if errors:
        logger.error("  ✗ Validation FAILED:")
        for err in errors:
            logger.error(f"    - {err}")
        raise ValueError(f"Design/data validation failed: {errors}")

    if warnings:
        logger.warning("  ⚠ Validation warnings:")
        for warn in warnings:
            logger.warning(f"    - {warn}")

    logger.info(f"  ✓ Validation passed")
    logger.info(f"    Design matrix: {n_subjects_design} subjects")
    if n_subjects_data:
        logger.info(f"    Data file: {n_subjects_data} volumes")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'n_subjects_design': n_subjects_design,
        'n_subjects_data': n_subjects_data
    }


def create_matched_design_for_analysis(
    participants_file: Path,
    derivatives_dir: Path,
    formula: str,
    analysis_type: str,
    output_dir: Path,
    metric: Optional[str] = None,
    demean_continuous: bool = True,
    add_intercept: bool = True
) -> Dict:
    """
    All-in-one function: discover subjects, create design matrix, validate

    This is the recommended high-level function that handles the complete workflow.

    Args:
        participants_file: Path to participants TSV/CSV with demographics
        derivatives_dir: Path to derivatives directory
        formula: Design formula (e.g., 'group + age + sex')
        analysis_type: Type of analysis ('tbss', 'vbm', 'func', 'dwi_metrics')
        output_dir: Output directory for design files
        metric: For functional, specify 'reho' or 'falff'
        demean_continuous: Demean continuous variables
        add_intercept: Add intercept column

    Returns:
        Dictionary with design_mat, column_names, matched_subjects, etc.

    Example:
        from neurovrai.analysis.stats.design_matrix_matching import create_matched_design_for_analysis

        result = create_matched_design_for_analysis(
            participants_file=Path('/study/participants.tsv'),
            derivatives_dir=Path('/study/derivatives'),
            formula='group + age + sex',
            analysis_type='tbss',
            output_dir=Path('/study/analysis/tbss/model')
        )
    """
    from neurovrai.analysis.stats.design_matrix import create_design_matrix

    logger.info("=" * 80)
    logger.info("CREATING MATCHED DESIGN MATRIX FOR ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Analysis type: {analysis_type}")
    if metric:
        logger.info(f"Metric: {metric}")
    logger.info(f"Formula: {formula}")
    logger.info("")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load participants data
    logger.info("1. Loading participants data...")
    if str(participants_file).endswith('.tsv'):
        df = pd.read_csv(participants_file, sep='\t')
    else:
        df = pd.read_csv(participants_file)

    if 'participant_id' not in df.columns:
        raise ValueError("Participants file must have 'participant_id' column")

    df = df.set_index('participant_id')
    logger.info(f"   Loaded {len(df)} participants")

    # Step 2: Discover subjects with data
    logger.info(f"\n2. Discovering subjects with {analysis_type} data...")
    available_subjects = discover_subjects_for_analysis(
        derivatives_dir, analysis_type, metric
    )

    if len(available_subjects) == 0:
        raise ValueError(f"No subjects found with {analysis_type} data")

    # Step 3: Filter participants to matched subjects
    logger.info(f"\n3. Matching participants with available data...")
    matched_subjects = [s for s in available_subjects if s in df.index]

    if len(matched_subjects) == 0:
        raise ValueError("No subjects match between participants file and available data")

    logger.info(f"   Matched: {len(matched_subjects)} subjects")

    df_matched = df.loc[matched_subjects]

    # Step 4: Create design matrix
    logger.info(f"\n4. Creating design matrix...")
    design_mat, column_names = create_design_matrix(
        df=df_matched,
        formula=formula,
        demean_continuous=demean_continuous,
        add_intercept=add_intercept
    )

    logger.info(f"   ✓ Design matrix: {design_mat.shape}")
    logger.info(f"   ✓ Predictors: {column_names}")

    # Step 5: Save design files
    logger.info(f"\n5. Saving design files...")

    # Save design matrix
    design_file = output_dir / 'design.mat'
    with open(design_file, 'w') as f:
        f.write(f"/NumWaves {design_mat.shape[1]}\n")
        f.write(f"/NumPoints {design_mat.shape[0]}\n")
        f.write("/Matrix\n")
        np.savetxt(f, design_mat, fmt='%.6f')
    logger.info(f"   ✓ {design_file}")

    # Save subject list
    subject_list_file = output_dir / 'subject_list.txt'
    with open(subject_list_file, 'w') as f:
        f.write('\n'.join(matched_subjects))
    logger.info(f"   ✓ {subject_list_file}")

    # Save matched participants
    participants_out = output_dir / 'participants_matched.tsv'
    df_matched.to_csv(participants_out, sep='\t')
    logger.info(f"   ✓ {participants_out}")

    logger.info("\n" + "=" * 80)
    logger.info("DESIGN MATRIX CREATION COMPLETE")
    logger.info("=" * 80)

    return {
        'design_mat': design_mat,
        'column_names': column_names,
        'matched_subjects': matched_subjects,
        'participants_df': df_matched,
        'design_file': design_file,
        'subject_list_file': subject_list_file,
        'n_subjects': len(matched_subjects)
    }
