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


def generate_all_designs(study_root: Path):
    """
    Generate design matrices for all analysis types in IRC805 study

    Args:
        study_root: Study root directory (e.g., /mnt/bytopia/IRC805)
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 80)
    logger.info("GENERATING ALL DESIGN MATRICES")
    logger.info("=" * 80)
    logger.info(f"Study root: {study_root}\n")

    designs_dir = study_root / 'data' / 'designs'
    formula = 'mriglu+sex+age'

    analysis_types = ['vbm', 'asl', 'func_reho', 'func_falff']

    # Add TBSS designs for all diffusion metrics
    tbss_metrics = ['FA', 'MD', 'AD', 'RD', 'MK', 'AK', 'RK', 'KFA', 'FICVF', 'ODI', 'FISO']
    analysis_types.extend([f'tbss/{metric}' for metric in tbss_metrics])

    results = {}
    for analysis_type in analysis_types:
        # Construct paths
        if analysis_type.startswith('tbss/'):
            metric = analysis_type.split('/')[1]
            design_dir = designs_dir / 'tbss' / metric
            analysis_name = f"TBSS {metric}"
        else:
            design_dir = designs_dir / analysis_type
            analysis_name = analysis_type.upper()

        participants_file = design_dir / 'participants_matched.tsv'

        if not participants_file.exists():
            logger.warning(f"⚠ Skipping {analysis_name} - participants file not found: {participants_file}")
            continue

        try:
            result = generate_design(
                participants_file=participants_file,
                formula=formula,
                output_dir=design_dir,
                analysis_name=analysis_name
            )
            results[analysis_type] = result
        except Exception as e:
            logger.error(f"✗ Failed to generate design for {analysis_name}: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successfully generated {len(results)} design matrices:\n")

    for analysis_type, result in results.items():
        logger.info(f"{analysis_type}:")
        logger.info(f"  Subjects: {result['n_subjects']}")
        logger.info(f"  Predictors: {result['n_predictors']}")
        logger.info(f"  Contrasts: {result['n_contrasts']}")
        logger.info(f"  Location: {result['design_mat'].parent}\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Generate FSL design matrices using neuroaider',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate design for specific analysis
  python generate_design_matrices.py \\
      --participants /mnt/bytopia/IRC805/data/designs/vbm/participants_matched.tsv \\
      --output-dir /mnt/bytopia/IRC805/data/designs/vbm \\
      --formula 'mriglu+sex+age'

  # Generate designs for all analyses
  python generate_design_matrices.py --all --study-root /mnt/bytopia/IRC805
        """
    )

    parser.add_argument(
        '--participants',
        type=Path,
        help='Path to participants TSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for design files'
    )
    parser.add_argument(
        '--formula',
        type=str,
        default='mriglu+sex+age',
        help='Model formula (default: mriglu+sex+age)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate designs for all analysis types'
    )
    parser.add_argument(
        '--study-root',
        type=Path,
        default=Path('/mnt/bytopia/IRC805'),
        help='Study root directory (default: /mnt/bytopia/IRC805)'
    )

    args = parser.parse_args()

    setup_logging()

    try:
        if args.all:
            # Generate all designs
            generate_all_designs(args.study_root)
        else:
            # Generate single design
            if not args.participants or not args.output_dir:
                parser.error("--participants and --output-dir required (or use --all)")

            generate_design(
                participants_file=args.participants,
                formula=args.formula,
                output_dir=args.output_dir
            )

    except Exception as e:
        logging.error(f"\n✗ Design generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
