#!/usr/bin/env python3
"""
VBM Group Analysis Runner

Convenience script for running VBM (Voxel-Based Morphometry) group analysis
with flexible statistical methods (randomise, GLM, or both).
"""

import logging
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add neurovrai to path
sys.path.insert(0, str(Path(__file__).parent))

from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data, run_vbm_analysis


def setup_logging(log_file: Path):
    """Configure logging to file and console"""
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Run VBM group analysis with randomise and/or nilearn GLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with randomise (recommended - TFCE correction)
  python run_vbm_group_analysis.py --study-root /mnt/bytopia/IRC805

  # Run with nilearn GLM (faster, parametric statistics with FDR/Bonferroni correction)
  python run_vbm_group_analysis.py --study-root /mnt/bytopia/IRC805 --method glm

  # Run both methods for comparison
  python run_vbm_group_analysis.py --study-root /mnt/bytopia/IRC805 --method both

  # Custom parameters
  python run_vbm_group_analysis.py \\
    --study-root /mnt/bytopia/IRC805 \\
    --method both \\
    --tissue GM \\
    --smooth-fwhm 6.0 \\
    --n-permutations 10000

NOTE: GLM uses nilearn's SecondLevelModel with FDR and Bonferroni correction.
      Randomise uses FSL's permutation testing with TFCE correction (more robust).
        """
    )

    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Study root directory (e.g., /mnt/bytopia/IRC805)'
    )
    parser.add_argument(
        '--tissue',
        type=str,
        choices=['GM', 'WM', 'CSF'],
        default='GM',
        help='Tissue type to analyze (default: GM)'
    )
    parser.add_argument(
        '--smooth-fwhm',
        type=float,
        default=4.0,
        help='Smoothing FWHM in mm (default: 4.0, 0=no smoothing)'
    )
    parser.add_argument(
        '--modulate',
        action='store_true',
        help='Modulate by Jacobian to preserve volume'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['randomise', 'glm', 'both'],
        default='randomise',
        help='Statistical method (default: randomise)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=5000,
        help='Number of permutations for randomise (default: 5000)'
    )
    parser.add_argument(
        '--no-tfce',
        action='store_true',
        help='Disable TFCE correction (randomise only)'
    )
    parser.add_argument(
        '--z-threshold',
        type=float,
        default=2.3,
        help='Z-score threshold for GLM (default: 2.3, approx p<0.01)'
    )
    parser.add_argument(
        '--cluster-threshold',
        type=float,
        default=0.95,
        help='Cluster threshold for randomise (default: 0.95)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='vbm_analysis',
        help='Analysis name for output directory (default: vbm_analysis)'
    )
    parser.add_argument(
        '--design-dir',
        type=Path,
        required=False,
        help='Directory with pre-generated design matrices (design.mat, design.con). If not specified, uses {study-root}/data/designs/vbm/'
    )

    args = parser.parse_args()

    # Setup paths
    study_root = args.study_root.resolve()
    derivatives_dir = study_root / 'derivatives'
    analysis_dir = study_root / 'analysis'

    # Set design directory (use provided or default)
    if args.design_dir:
        design_dir = args.design_dir.resolve()
    else:
        design_dir = study_root / 'data' / 'designs' / 'vbm'

    # Verify design files exist
    design_mat = design_dir / 'design.mat'
    design_con = design_dir / 'design.con'
    participants_file = design_dir / 'participants_matched.tsv'

    if not design_mat.exists():
        logger.error(f"Design matrix not found: {design_mat}")
        logger.info("\nPlease generate design matrices first:")
        logger.info("  python generate_design_matrices.py --all")
        sys.exit(1)

    if not design_con.exists():
        logger.error(f"Contrast file not found: {design_con}")
        sys.exit(1)

    if not participants_file.exists():
        logger.error(f"Participants file not found: {participants_file}")
        sys.exit(1)

    # Setup logging
    log_file = Path('logs') / f'vbm_group_analysis_{args.tissue.lower()}.log'
    setup_logging(log_file)

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("VBM GROUP ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Study root: {study_root}")
    logger.info(f"Derivatives: {derivatives_dir}")
    logger.info(f"Analysis: {analysis_dir}")
    logger.info(f"Tissue type: {args.tissue}")
    logger.info(f"Statistical method: {args.method}")
    logger.info(f"Participants: {participants_file}")

    # Verify participants file exists
    if not participants_file.exists():
        logger.error(f"Participants file not found: {participants_file}")
        logger.info("\nPlease create a participants file with columns:")
        logger.info("  - participant_id: Subject ID (e.g., IRC805-0580101)")
        logger.info("  - age: Age in years")
        logger.info("  - sex: 0 or 1")
        logger.info("  - group: 0 or 1 (optional)")
        sys.exit(1)

    # Find subjects with anatomical data
    subjects = []
    for subject_dir in sorted(derivatives_dir.glob('IRC805-*/anat')):
        subject_id = subject_dir.parent.name
        seg_dir = subject_dir / 'segmentation'
        if not seg_dir.exists():
            continue

        # Check for GM probability map (FAST or Atropos)
        gm_fast = seg_dir / 'pve_1.nii.gz'
        gm_atropos = seg_dir / 'POSTERIOR_02.nii.gz'

        if gm_fast.exists() or gm_atropos.exists():
            subjects.append(subject_id)

    if not subjects:
        logger.error(f"No subjects with anatomical preprocessing found in {derivatives_dir}")
        logger.info("\nRun anatomical preprocessing first:")
        logger.info("  python run_preprocessing.py --subject SUBJECT_ID --modality anat")
        sys.exit(1)

    logger.info(f"\nFound {len(subjects)} subjects with anatomical data")

    # Filter subjects to only those in the participants file
    participants_df = pd.read_csv(participants_file, sep='\t')
    available_participant_ids = set(participants_df['participant_id'].values)

    subjects_in_design = [s for s in subjects if s in available_participant_ids]
    excluded_subjects = [s for s in subjects if s not in available_participant_ids]

    if excluded_subjects:
        logger.info(f"\nExcluding {len(excluded_subjects)} subjects not in design:")
        for s in excluded_subjects:
            logger.info(f"  - {s}")

    if not subjects_in_design:
        logger.error("No subjects match the participants file!")
        sys.exit(1)

    logger.info(f"\nUsing {len(subjects_in_design)} subjects that match design matrix")

    try:
        # Step 1: Prepare VBM data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: PREPARE VBM DATA")
        logger.info("=" * 80)

        output_dir = analysis_dir / 'anat' / 'vbm' / args.study_name
        output_dir.mkdir(parents=True, exist_ok=True)

        prep_results = prepare_vbm_data(
            subjects=subjects_in_design,
            derivatives_dir=derivatives_dir,
            output_dir=output_dir,
            tissue_type=args.tissue,
            smooth_fwhm=args.smooth_fwhm,
            modulate=args.modulate
        )

        logger.info(f"\n✓ VBM data preparation complete")
        logger.info(f"  Processed: {prep_results['n_subjects']} subjects")
        logger.info(f"  Tissue: {prep_results['tissue_type']}")
        logger.info(f"  Merged 4D: {prep_results['merged_file']}")

        # Step 2: Run statistical analysis
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: STATISTICAL ANALYSIS")
        logger.info("=" * 80)

        # Use pre-generated design matrices from design directory
        # Design matrices created by generate_design_matrices.py with neuroaider
        # Contains: design.mat, design.con, participants_matched.tsv
        # For binary mriglu groups: 6 contrasts auto-generated
        #   - mriglu_positive [0, 0, 1, -1]: Controlled > Uncontrolled
        #   - mriglu_negative [0, 0, -1, 1]: Uncontrolled > Controlled
        #   - sex_positive/negative: Sex effects
        #   - age_positive/negative: Age effects

        analysis_results = run_vbm_analysis(
            vbm_dir=output_dir,
            design_dir=design_dir,
            method=args.method,
            n_permutations=args.n_permutations,
            tfce=not args.no_tfce,
            cluster_threshold=args.cluster_threshold,
            z_threshold=args.z_threshold
        )

        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)

        if args.method in ['randomise', 'both']:
            logger.info(f"Randomise results: {analysis_results.get('randomise_dir', 'N/A')}")

        if args.method in ['glm', 'both']:
            logger.info(f"GLM results: {analysis_results.get('glm_dir', 'N/A')}")

        logger.info(f"Log file: {log_file}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n✗ Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
