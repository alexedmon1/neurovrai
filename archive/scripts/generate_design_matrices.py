#!/usr/bin/env python3
"""
Generate Design Matrices with Neuroaider

Creates FSL design matrices (design.mat) and contrasts (design.con) using neuroaider
for group-level statistical analysis. This is the FIRST STEP before running any analysis.

Usage:
    # Generate design for binary group comparison
    python generate_design_matrices.py \
        --participants /path/to/participants.tsv \
        --output-dir /path/to/design/output \
        --formula 'mriglu+sex+age'

    # Generate designs for all analysis types
    python generate_design_matrices.py --all
"""

import logging
import sys
import argparse
from pathlib import Path
import pandas as pd
import json

# Add neurovrai to path
sys.path.insert(0, str(Path(__file__).parent))

from neuroaider import DesignHelper
from neurovrai.analysis.utils.modality_subjects import (
    load_participants_for_modality,
    save_subject_list,
    get_modality_info
)


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )


def generate_design(
    participants_file: Path,
    formula: str,
    output_dir: Path,
    analysis_name: str = None
):
    """
    Generate design matrix and contrasts using neuroaider

    Args:
        participants_file: Path to participants TSV file
        formula: Model formula (e.g., 'mriglu+sex+age')
        output_dir: Output directory for design files
        analysis_name: Optional analysis name for logging

    Returns:
        Dictionary with paths to generated files
    """
    logger = logging.getLogger(__name__)

    if analysis_name:
        logger.info("\n" + "=" * 80)
        logger.info(f"GENERATING DESIGN: {analysis_name}")
        logger.info("=" * 80)

    logger.info(f"Participants: {participants_file}")
    logger.info(f"Formula: {formula}")
    logger.info(f"Output: {output_dir}")

    # Load participants data
    if participants_file.suffix == '.tsv':
        df = pd.read_csv(participants_file, sep='\t')
    else:
        df = pd.read_csv(participants_file)

    logger.info(f"  Subjects: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")

    # Parse formula to detect binary groups
    formula_terms = [t.strip() for t in formula.split('+')]
    first_var = formula_terms[0].replace('C(', '').replace(')', '')

    # Detect binary categorical for no-intercept dummy coding
    use_binary_coding = False
    if first_var in df.columns:
        n_levels = df[first_var].nunique()
        if n_levels == 2:
            use_binary_coding = True
            levels = sorted(df[first_var].unique())
            logger.info(f"\n✓ Detected binary group variable '{first_var}'")
            logger.info(f"  Levels: {levels} (n={n_levels})")
            logger.info(f"  Coding: Dummy WITHOUT intercept (direct group comparison)")

    # Initialize DesignHelper
    helper = DesignHelper(
        participants_file=df,
        subject_column='participant_id',
        add_intercept=not use_binary_coding
    )

    # Add variables from formula
    for term in formula_terms:
        var_name = term.replace('C(', '').replace(')', '').strip()

        if var_name not in df.columns:
            raise ValueError(f"Variable '{var_name}' not found in participants file")

        # Determine if categorical or continuous
        if pd.api.types.is_numeric_dtype(df[var_name]):
            n_unique = df[var_name].nunique()
            if n_unique <= 10 and use_binary_coding and var_name == first_var:
                helper.add_categorical(var_name, coding='dummy')
                logger.info(f"✓ Added categorical: {var_name} (dummy coding, no intercept)")
            elif n_unique <= 10:
                logger.warning(f"Variable '{var_name}' has {n_unique} unique values - treating as continuous")
                helper.add_covariate(var_name, mean_center=True)
            else:
                helper.add_covariate(var_name, mean_center=True)
                logger.info(f"✓ Added covariate: {var_name} (mean-centered)")
        else:
            helper.add_categorical(var_name, coding='effect' if not use_binary_coding else 'dummy')
            logger.info(f"✓ Added categorical: {var_name}")

    # Build design matrix
    design_mat, column_names = helper.build_design_matrix()
    logger.info(f"\n✓ Design matrix shape: {design_mat.shape}")
    logger.info(f"  Columns: {column_names}")

    # Auto-generate contrasts for binary groups
    if use_binary_coding:
        logger.info(f"\n✓ Auto-generating binary group contrasts for '{first_var}'")
        helper.add_binary_group_contrasts(first_var)

    # Add covariate contrasts if present
    for term in formula_terms:
        var_name = term.replace('C(', '').replace(')', '').strip()
        if var_name in column_names and var_name != first_var:
            # Add positive and negative contrasts for continuous covariates
            helper.add_contrast(f"{var_name}_positive", covariate=var_name, direction='+')
            helper.add_contrast(f"{var_name}_negative", covariate=var_name, direction='-')
            logger.info(f"✓ Added contrasts for covariate: {var_name}")

    # Get contrast information
    contrast_mat, contrast_names = helper.build_contrast_matrix()
    logger.info(f"\n✓ Contrasts ({len(contrast_names)}):")
    for name, weights in zip(contrast_names, contrast_mat):
        logger.info(f"  {name}: {weights.tolist()}")

    # Save design files
    output_dir.mkdir(parents=True, exist_ok=True)
    design_file = output_dir / 'design.mat'
    contrast_file = output_dir / 'design.con'
    summary_file = output_dir / 'design_summary.json'

    helper.save(
        design_mat_file=design_file,
        design_con_file=contrast_file,
        summary_file=summary_file
    )

    logger.info(f"\n✓ Generated design files:")
    logger.info(f"  Design matrix: {design_file}")
    logger.info(f"  Contrasts: {contrast_file}")
    logger.info(f"  Summary: {summary_file}")

    return {
        'design_mat': design_file,
        'design_con': contrast_file,
        'summary': summary_file,
        'n_subjects': len(df),
        'n_predictors': len(column_names),
        'n_contrasts': len(contrast_names)
    }


def generate_design_for_modality(
    participants_file: Path,
    study_root: Path,
    modality: str,
    formula: str,
    output_dir: Path,
    metric: str = None
):
    """
    Generate design matrix for a specific modality using auto-detected subjects

    This function:
    1. Auto-detects which subjects have MRI data for the modality
    2. Filters participants file to only those subjects
    3. Generates design matrix with correct subject ordering
    4. Saves subject list file for verification

    Args:
        participants_file: Path to master participants file (e.g., gludata.csv)
        study_root: Study root directory
        modality: Modality type ('vbm', 'asl', 'reho', 'falff', 'tbss')
        formula: Model formula (e.g., 'mriglu+sex+age')
        output_dir: Output directory for design files
        metric: For TBSS, specify metric (e.g., 'FA', 'MD')

    Returns:
        Dictionary with paths to generated files
    """
    logger = logging.getLogger(__name__)

    # Get modality info
    info = get_modality_info(modality)
    analysis_name = f"{info['short_name']}"
    if metric:
        analysis_name += f" {metric}"

    logger.info("\n" + "=" * 80)
    logger.info(f"GENERATING DESIGN: {analysis_name}")
    logger.info("=" * 80)
    logger.info(f"Modality: {modality}")
    if metric:
        logger.info(f"Metric: {metric}")
    logger.info(f"Participants file: {participants_file}")
    logger.info(f"Formula: {formula}")
    logger.info(f"Output: {output_dir}")

    # Load participants data filtered to subjects with MRI data
    df, ordered_subjects = load_participants_for_modality(
        participants_file=participants_file,
        study_root=study_root,
        modality=modality,
        metric=metric
    )

    logger.info(f"  Subjects with data: {len(ordered_subjects)}")
    logger.info(f"  Subject IDs: {ordered_subjects[:5]}..." if len(ordered_subjects) > 5 else f"  Subject IDs: {ordered_subjects}")

    # Parse formula to detect binary groups
    formula_terms = [t.strip() for t in formula.split('+')]
    first_var = formula_terms[0].replace('C(', '').replace(')', '')

    # Detect binary categorical for no-intercept dummy coding
    use_binary_coding = False
    if first_var in df.columns:
        n_levels = df[first_var].nunique()
        if n_levels == 2:
            use_binary_coding = True
            levels = sorted(df[first_var].unique())
            logger.info(f"\n✓ Detected binary group variable '{first_var}'")
            logger.info(f"  Levels: {levels} (n={n_levels})")
            logger.info(f"  Coding: Dummy WITHOUT intercept (direct group comparison)")

    # Initialize DesignHelper
    helper = DesignHelper(
        participants_file=df,
        subject_column='participant_id',
        add_intercept=not use_binary_coding
    )

    # Add variables from formula
    for term in formula_terms:
        var_name = term.replace('C(', '').replace(')', '').strip()

        if var_name not in df.columns:
            raise ValueError(f"Variable '{var_name}' not found in participants file")

        # Determine if categorical or continuous
        if pd.api.types.is_numeric_dtype(df[var_name]):
            n_unique = df[var_name].nunique()
            if n_unique <= 10 and use_binary_coding and var_name == first_var:
                helper.add_categorical(var_name, coding='dummy')
                logger.info(f"✓ Added categorical: {var_name} (dummy coding, no intercept)")
            elif n_unique <= 10:
                logger.warning(f"Variable '{var_name}' has {n_unique} unique values - treating as continuous")
                helper.add_covariate(var_name, mean_center=True)
            else:
                helper.add_covariate(var_name, mean_center=True)
                logger.info(f"✓ Added covariate: {var_name} (mean-centered)")
        else:
            helper.add_categorical(var_name, coding='effect' if not use_binary_coding else 'dummy')
            logger.info(f"✓ Added categorical: {var_name}")

    # Build design matrix
    design_mat, column_names = helper.build_design_matrix()
    logger.info(f"\n✓ Design matrix shape: {design_mat.shape}")
    logger.info(f"  Columns: {column_names}")

    # Auto-generate contrasts for binary groups
    if use_binary_coding:
        logger.info(f"\n✓ Auto-generating binary group contrasts for '{first_var}'")
        helper.add_binary_group_contrasts(first_var)

    # Add covariate contrasts if present
    for term in formula_terms:
        var_name = term.replace('C(', '').replace(')', '').strip()
        if var_name in column_names and var_name != first_var:
            # Add positive and negative contrasts for continuous covariates
            helper.add_contrast(f"{var_name}_positive", covariate=var_name, direction='+')
            helper.add_contrast(f"{var_name}_negative", covariate=var_name, direction='-')
            logger.info(f"✓ Added contrasts for covariate: {var_name}")

    # Get contrast information
    contrast_mat, contrast_names = helper.build_contrast_matrix()
    logger.info(f"\n✓ Contrasts ({len(contrast_names)}):")
    for name, weights in zip(contrast_names, contrast_mat):
        logger.info(f"  {name}: {weights.tolist()}")

    # Save design files
    output_dir.mkdir(parents=True, exist_ok=True)
    design_file = output_dir / 'design.mat'
    contrast_file = output_dir / 'design.con'
    summary_file = output_dir / 'design_summary.json'
    subject_list_file = output_dir / 'subject_order.txt'

    helper.save(
        design_mat_file=design_file,
        design_con_file=contrast_file,
        summary_file=summary_file
    )

    # Save subject list with ordering information
    save_subject_list(ordered_subjects, subject_list_file, analysis_name)

    logger.info(f"\n✓ Generated design files:")
    logger.info(f"  Design matrix: {design_file}")
    logger.info(f"  Contrasts: {contrast_file}")
    logger.info(f"  Summary: {summary_file}")
    logger.info(f"  Subject list: {subject_list_file}")

    return {
        'design_mat': design_file,
        'design_con': contrast_file,
        'summary': summary_file,
        'subject_list': subject_list_file,
        'n_subjects': len(ordered_subjects),
        'n_predictors': len(column_names),
        'n_contrasts': len(contrast_names),
        'subjects': ordered_subjects
    }




def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Generate FSL design matrices using modality-aware subject detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate design for VBM (uses subjects with VBM data in derivatives)
  python generate_design_matrices.py \\
      --modality vbm \\
      --participants /mnt/bytopia/IRC805/data/gludata.csv \\
      --study-root /mnt/bytopia/IRC805 \\
      --output-dir /mnt/bytopia/IRC805/data/designs/vbm

  # Generate design for TBSS FA metric
  python generate_design_matrices.py \\
      --modality tbss \\
      --metric FA \\
      --participants /mnt/bytopia/IRC805/data/gludata.csv \\
      --study-root /mnt/bytopia/IRC805 \\
      --output-dir /mnt/bytopia/IRC805/data/designs/tbss

  # Generate design for ReHo functional analysis
  python generate_design_matrices.py \\
      --modality reho \\
      --participants /mnt/bytopia/IRC805/data/gludata.csv \\
      --study-root /mnt/bytopia/IRC805 \\
      --output-dir /mnt/bytopia/IRC805/data/designs/func_reho

Supported modalities:
  - vbm: Voxel-Based Morphometry
  - asl: Arterial Spin Labeling
  - reho: Regional Homogeneity (functional)
  - falff: Fractional ALFF (functional)
  - tbss: Tract-Based Spatial Statistics (requires --metric)
        """
    )

    parser.add_argument(
        '--modality',
        type=str,
        required=True,
        choices=['vbm', 'asl', 'reho', 'falff', 'tbss'],
        help='MRI modality to generate design for'
    )
    parser.add_argument(
        '--metric',
        type=str,
        help='For TBSS modality, specify metric (FA, MD, AD, RD, MK, AK, RK, KFA, FICVF, ODI, FISO)'
    )
    parser.add_argument(
        '--participants',
        type=Path,
        required=True,
        help='Path to master participants file (e.g., gludata.csv)'
    )
    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Study root directory (e.g., /mnt/bytopia/IRC805)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for design files'
    )
    parser.add_argument(
        '--formula',
        type=str,
        default='mriglu+sex+age',
        help='Model formula (default: mriglu+sex+age)'
    )

    args = parser.parse_args()

    setup_logging()

    # Validate TBSS requires metric
    if args.modality == 'tbss' and not args.metric:
        parser.error("--metric required for TBSS modality (e.g., FA, MD, MK)")

    try:
        result = generate_design_for_modality(
            participants_file=args.participants,
            study_root=args.study_root,
            modality=args.modality,
            formula=args.formula,
            output_dir=args.output_dir,
            metric=args.metric
        )

        logging.info("\n" + "=" * 80)
        logging.info("SUCCESS")
        logging.info("=" * 80)
        logging.info(f"Generated design for {args.modality.upper()}")
        logging.info(f"  Subjects: {result['n_subjects']}")
        logging.info(f"  Predictors: {result['n_predictors']}")
        logging.info(f"  Contrasts: {result['n_contrasts']}")
        logging.info(f"  Location: {result['design_mat'].parent}")
        logging.info("")

    except Exception as e:
        logging.error(f"\n✗ Design generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
