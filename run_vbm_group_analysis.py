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
        description='Run VBM group analysis with randomise and/or GLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with randomise (default)
  python run_vbm_group_analysis.py --study-root /mnt/bytopia/IRC805

  # Run with GLM only (faster, for exploratory analysis)
  python run_vbm_group_analysis.py --study-root /mnt/bytopia/IRC805 --method glm

  # Run both methods for comparison
  python run_vbm_group_analysis.py --study-root /mnt/bytopia/IRC805 --method both

  # Custom parameters
  python run_vbm_group_analysis.py \\
    --study-root /mnt/bytopia/IRC805 \\
    --method both \\
    --tissue GM \\
    --smooth-fwhm 6.0 \\
    --n-permutations 10000 \\
    --z-threshold 2.58
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

    args = parser.parse_args()

    # Setup paths
    study_root = args.study_root.resolve()
    derivatives_dir = study_root / 'derivatives'
    analysis_dir = study_root / 'analysis'
    participants_file = analysis_dir / 'participants_synthetic.tsv'

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
        # Check for required files (brain-warped GM map)
        gm_mni = subject_dir / 'segmentation' / 'pve_1_mni.nii.gz'
        if gm_mni.exists():
            subjects.append(subject_id)

    if not subjects:
        logger.error(f"No subjects with anatomical preprocessing found in {derivatives_dir}")
        logger.info("\nRun anatomical preprocessing first:")
        logger.info("  python run_preprocessing.py --subject SUBJECT_ID --modality anat")
        sys.exit(1)

    logger.info(f"\nFound {len(subjects)} subjects with anatomical data")

    try:
        # Step 1: Prepare VBM data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: PREPARE VBM DATA")
        logger.info("=" * 80)

        output_dir = analysis_dir / 'anat' / 'vbm' / args.study_name
        output_dir.mkdir(parents=True, exist_ok=True)

        prep_results = prepare_vbm_data(
            subjects=subjects,
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

        # Define contrasts
        # For a design with: [intercept, age, sex, group]
        # You may need to adjust these based on your actual design matrix
        contrasts = {
            'age_positive': [0, 1, 0, 0],
            'age_negative': [0, -1, 0, 0],
            'group_positive': [0, 0, 0, 1],
            'group_negative': [0, 0, 0, -1]
        }

        analysis_results = run_vbm_analysis(
            vbm_dir=Path(prep_results['output_dir']),
            participants_file=participants_file,
            formula='age + sex + group',
            contrasts=contrasts,
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
