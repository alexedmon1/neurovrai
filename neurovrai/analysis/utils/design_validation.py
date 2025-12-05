"""
Design Matrix Validation Utilities

Ensures that design matrices, participant files, and MRI data are correctly aligned
before running statistical analyses. This is CRITICAL for FSL randomise and GLM analyses
where subject order must match exactly.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import nibabel as nib

logger = logging.getLogger(__name__)


def parse_fsl_design_mat(design_mat_file: Path) -> Tuple[int, int]:
    """
    Parse FSL design.mat file to extract dimensions

    Args:
        design_mat_file: Path to design.mat file

    Returns:
        Tuple of (n_subjects, n_predictors)
    """
    with open(design_mat_file, 'r') as f:
        lines = f.readlines()

    # Extract dimensions from header
    n_waves = int([l for l in lines if '/NumWaves' in l][0].split()[1])
    n_points = int([l for l in lines if '/NumPoints' in l][0].split()[1])

    return n_points, n_waves


def parse_fsl_design_con(design_con_file: Path) -> int:
    """
    Parse FSL design.con file to extract number of contrasts

    Args:
        design_con_file: Path to design.con file

    Returns:
        Number of contrasts
    """
    with open(design_con_file, 'r') as f:
        lines = f.readlines()

    n_contrasts = int([l for l in lines if '/NumContrasts' in l][0].split()[1])
    return n_contrasts


def validate_design_alignment(
    design_dir: Path,
    mri_files: List[Path],
    subject_ids: List[str],
    analysis_type: str = "group analysis"
) -> Dict:
    """
    Validate that design matrix, participants file, and MRI data are correctly aligned

    Args:
        design_dir: Directory containing design.mat, design.con, participants_matched.tsv
        mri_files: List of MRI files in the order they will be analyzed
        subject_ids: List of subject IDs corresponding to mri_files (in same order)
        analysis_type: Type of analysis (for logging)

    Returns:
        Dictionary with validation results

    Raises:
        ValueError: If validation fails
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"VALIDATING DESIGN ALIGNMENT FOR {analysis_type.upper()}")
    logger.info("=" * 80)

    # 1. Validate design files exist
    design_mat_file = design_dir / 'design.mat'
    design_con_file = design_dir / 'design.con'
    design_summary_file = design_dir / 'design_summary.json'
    participants_file = design_dir / 'participants_matched.tsv'

    if not design_mat_file.exists():
        raise FileNotFoundError(f"Design matrix not found: {design_mat_file}")
    if not design_con_file.exists():
        raise FileNotFoundError(f"Contrast file not found: {design_con_file}")
    if not participants_file.exists():
        raise FileNotFoundError(f"Participants file not found: {participants_file}")

    logger.info(f"✓ Design files exist in: {design_dir}")

    # 2. Parse design matrix dimensions
    n_design_subjects, n_predictors = parse_fsl_design_mat(design_mat_file)
    n_contrasts = parse_fsl_design_con(design_con_file)

    logger.info(f"\n✓ Design matrix dimensions:")
    logger.info(f"    Subjects (rows): {n_design_subjects}")
    logger.info(f"    Predictors (columns): {n_predictors}")
    logger.info(f"    Contrasts: {n_contrasts}")

    # 3. Load participants file
    participants_df = pd.read_csv(participants_file, sep='\t')
    n_participants = len(participants_df)

    logger.info(f"\n✓ Participants file:")
    logger.info(f"    Subjects: {n_participants}")
    logger.info(f"    Columns: {list(participants_df.columns)}")

    # 4. Check MRI files
    n_mri_files = len(mri_files)
    logger.info(f"\n✓ MRI data:")
    logger.info(f"    Files: {n_mri_files}")

    # 5. Load design summary if available
    design_summary = None
    if design_summary_file.exists():
        with open(design_summary_file, 'r') as f:
            design_summary = json.load(f)
        logger.info(f"\n✓ Design summary:")
        logger.info(f"    Design matrix columns: {design_summary['columns']}")
        logger.info(f"    Contrasts: {design_summary['contrasts']}")

    # 6. CRITICAL VALIDATION: Check all counts match
    logger.info("\n" + "=" * 80)
    logger.info("ALIGNMENT VALIDATION")
    logger.info("=" * 80)

    errors = []
    warnings = []

    # Check subject counts match
    if n_design_subjects != n_participants:
        errors.append(f"Design matrix has {n_design_subjects} subjects but participants file has {n_participants}")
    else:
        logger.info(f"✓ Subject count matches: {n_design_subjects} subjects")

    if n_design_subjects != n_mri_files:
        errors.append(f"Design matrix has {n_design_subjects} subjects but found {n_mri_files} MRI files")
    else:
        logger.info(f"✓ MRI file count matches: {n_mri_files} files")

    # 7. CRITICAL: Verify subject order matches
    participants_ordered = participants_df['participant_id'].tolist()

    if len(subject_ids) != len(participants_ordered):
        errors.append(f"Subject ID list length ({len(subject_ids)}) doesn't match participants file ({len(participants_ordered)})")
    else:
        # Check if order matches exactly
        order_mismatch = []
        for i, (mri_subj, design_subj) in enumerate(zip(subject_ids, participants_ordered)):
            if mri_subj != design_subj:
                order_mismatch.append(f"  Position {i}: MRI={mri_subj}, Design={design_subj}")

        if order_mismatch:
            errors.append("Subject order mismatch between MRI files and design matrix:\n" + "\n".join(order_mismatch[:5]))
            if len(order_mismatch) > 5:
                errors.append(f"  ... and {len(order_mismatch) - 5} more mismatches")
        else:
            logger.info(f"✓ Subject order matches perfectly:")
            for i, subj in enumerate(subject_ids[:5]):
                logger.info(f"    Position {i+1}: {subj}")
            if len(subject_ids) > 5:
                logger.info(f"    ... and {len(subject_ids) - 5} more subjects")

    # 8. Check for subjects in design but missing MRI data
    missing_mri = set(participants_ordered) - set(subject_ids)
    if missing_mri:
        warnings.append(f"Subjects in design but missing MRI data: {missing_mri}")

    # 9. Check for subjects with MRI data but not in design
    extra_mri = set(subject_ids) - set(participants_ordered)
    if extra_mri:
        warnings.append(f"Subjects with MRI data but not in design: {extra_mri}")

    # 10. Validate 4D image if provided as merged file
    if len(mri_files) == 1 and mri_files[0].name.endswith('_4D.nii.gz'):
        img = nib.load(mri_files[0])
        n_volumes = img.shape[3] if len(img.shape) == 4 else 1

        logger.info(f"\n✓ 4D Image validation:")
        logger.info(f"    Shape: {img.shape}")
        logger.info(f"    Volumes (subjects): {n_volumes}")

        if n_volumes != n_design_subjects:
            errors.append(f"4D image has {n_volumes} volumes but design matrix has {n_design_subjects} subjects")
        else:
            logger.info(f"✓ 4D image volume count matches design: {n_volumes} volumes")

    # 11. Report results
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)

    if warnings:
        logger.warning(f"\n⚠ Warnings ({len(warnings)}):")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    if errors:
        logger.error(f"\n✗ VALIDATION FAILED ({len(errors)} errors):")
        for error in errors:
            logger.error(f"  - {error}")
        logger.error("\nPlease fix alignment issues before running analysis!")
        raise ValueError("Design alignment validation failed - see errors above")

    logger.info("\n✓ VALIDATION PASSED - All checks successful!")
    logger.info("=" * 80 + "\n")

    return {
        'n_subjects': n_design_subjects,
        'n_predictors': n_predictors,
        'n_contrasts': n_contrasts,
        'subject_order': subject_ids,
        'design_summary': design_summary,
        'warnings': warnings
    }


def create_subject_order_report(
    output_file: Path,
    subject_ids: List[str],
    mri_files: List[Path],
    participants_df: pd.DataFrame
):
    """
    Create a detailed report showing subject order for verification

    Args:
        output_file: Output file path for report
        subject_ids: List of subject IDs in analysis order
        mri_files: List of MRI files in analysis order
        participants_df: DataFrame with participant demographics
    """
    with open(output_file, 'w') as f:
        f.write("# Subject Order Verification Report\n\n")
        f.write("This report shows the exact order of subjects in the statistical analysis.\n")
        f.write("The order MUST match between: MRI files, design matrix rows, and randomise input.\n\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total subjects: {len(subject_ids)}\n\n")

        f.write("| Position | Subject ID | MRI File | Demographics |\n")
        f.write("|----------|------------|----------|---------------|\n")

        for i, (subj_id, mri_file) in enumerate(zip(subject_ids, mri_files)):
            # Get demographics for this subject
            demo = participants_df[participants_df['participant_id'] == subj_id].iloc[0]
            demo_str = ", ".join([f"{k}={v}" for k, v in demo.items() if k != 'participant_id'])

            f.write(f"| {i+1:3d} | {subj_id} | {mri_file.name} | {demo_str} |\n")

    logger.info(f"✓ Created subject order report: {output_file}")
