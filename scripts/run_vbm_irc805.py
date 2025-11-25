#!/usr/bin/env python3
"""
Run VBM analysis on IRC805 anatomical data with synthetic demographics

This script:
1. Prepares grey matter probability maps from anatomical preprocessing
2. Creates synthetic participant demographics for testing
3. Runs group-level VBM statistical analysis
4. Generates enhanced cluster reports with tri-planar visualizations

Usage:
    python scripts/run_vbm_irc805.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np

from neurovrai.config import load_config
from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data, run_vbm_analysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_participants(subjects: list, output_file: Path) -> pd.DataFrame:
    """
    Create synthetic participant demographics for VBM testing

    Args:
        subjects: List of subject IDs
        output_file: Where to save the TSV file

    Returns:
        DataFrame with synthetic demographics
    """
    logger.info("Creating synthetic participant demographics...")

    np.random.seed(42)  # Reproducible

    n_subjects = len(subjects)

    # Create synthetic demographics
    data = {
        'participant_id': subjects,
        'age': np.random.normal(45, 15, n_subjects).clip(20, 80),  # Age 20-80, mean 45
        'sex': np.random.choice([0, 1], n_subjects),  # 0=female, 1=male
        'group': np.random.choice([0, 1], n_subjects),  # 0=control, 1=patient
        'education_years': np.random.normal(14, 3, n_subjects).clip(8, 20),  # Education
    }

    df = pd.DataFrame(data)

    # Round numeric columns
    df['age'] = df['age'].round(1)
    df['education_years'] = df['education_years'].round(0).astype(int)

    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep='\t', index=False)

    logger.info(f"Synthetic demographics saved to: {output_file}")
    logger.info(f"\nSummary:")
    logger.info(f"  N subjects: {n_subjects}")
    logger.info(f"  Age: {df['age'].mean():.1f} ± {df['age'].std():.1f} years")
    logger.info(f"  Sex: {(df['sex'] == 0).sum()} female, {(df['sex'] == 1).sum()} male")
    logger.info(f"  Group: {(df['group'] == 0).sum()} controls, {(df['group'] == 1).sum()} patients")
    logger.info("")

    return df


def main():
    # Load config
    config = load_config(Path('config.yaml'))

    derivatives_dir = Path(config['derivatives_dir'])
    analysis_dir = Path(config['analysis_dir'])
    vbm_dir = analysis_dir / 'vbm'

    logger.info("=" * 80)
    logger.info("VBM ANALYSIS - IRC805 GREY MATTER")
    logger.info("=" * 80)
    logger.info(f"Derivatives: {derivatives_dir}")
    logger.info(f"Output: {vbm_dir}")
    logger.info("")

    # Find subjects with anatomical preprocessing
    logger.info("Finding subjects with anatomical preprocessing...")
    subjects = []

    for subject_dir in sorted(derivatives_dir.glob('IRC805-*')):
        subject = subject_dir.name
        anat_dir = subject_dir / 'anat'

        # Check for tissue segmentation (grey matter) - try both naming conventions
        gm_file_fast = anat_dir / 'segmentation' / 'pve_1.nii.gz'  # FSL FAST
        gm_file_atropos = anat_dir / 'segmentation' / 'POSTERIOR_02.nii.gz'  # ANTs Atropos

        has_gm = gm_file_fast.exists() or gm_file_atropos.exists()

        # Check for transform to MNI - try multiple locations
        study_root = derivatives_dir.parent
        possible_transforms = [
            anat_dir / 'transforms' / 'ants_Composite.h5',
            study_root / 'transforms' / subject / 'T1w_to_MNI152_composite.h5',
            study_root / 'transforms' / subject / 'T1w_to_MNI152_warp.nii.gz',
        ]

        has_transform = any(tf.exists() for tf in possible_transforms)

        if has_gm and has_transform:
            subjects.append(subject)
            logger.info(f"  ✓ {subject}")

    logger.info("")
    logger.info(f"Found {len(subjects)} subjects with required data")

    if len(subjects) < 10:
        logger.warning(f"VBM typically requires 10+ subjects per group")
        logger.warning(f"Only found {len(subjects)} subjects - results may be underpowered")

    # Create synthetic participants file
    participants_file = analysis_dir / 'participants_synthetic.tsv'
    participants_df = create_synthetic_participants(subjects, participants_file)

    # =========================================================================
    # Step 1: Prepare VBM Data
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: PREPARING VBM DATA")
    logger.info("=" * 80)
    logger.info("")

    gm_dir = vbm_dir / 'GM'

    try:
        prep_results = prepare_vbm_data(
            subjects=subjects,
            derivatives_dir=derivatives_dir,
            output_dir=gm_dir,
            tissue_type='GM',
            smooth_fwhm=4.0,  # 4mm smoothing (standard for VBM)
            modulate=False    # No modulation (standard VBM)
        )

        logger.info("")
        logger.info(f"✓ VBM data preparation complete")
        logger.info(f"  Processed: {prep_results['n_subjects']} subjects")
        logger.info(f"  Merged data: {prep_results['merged_file']}")
        logger.info(f"  Group mask: {prep_results['mask_file']}")

    except Exception as e:
        logger.error(f"VBM data preparation failed: {e}", exc_info=True)
        return 1

    # =========================================================================
    # Step 2: Run VBM Statistics
    # =========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: RUNNING VBM STATISTICAL ANALYSIS")
    logger.info("=" * 80)
    logger.info("")

    # Define design and contrasts
    # Formula: age + sex + group
    # Design matrix columns: [intercept, age, sex, group]
    formula = 'age + sex + group'

    contrasts = {
        'age_positive': [0, 1, 0, 0],      # Positive age effect (GM decreases with age)
        'age_negative': [0, -1, 0, 0],     # Negative age effect
        'sex_difference': [0, 0, 1, 0],    # Sex differences (M > F)
        'group_difference': [0, 0, 0, 1],  # Group differences (patients > controls)
    }

    logger.info("Design:")
    logger.info(f"  Formula: {formula}")
    logger.info(f"  Contrasts: {list(contrasts.keys())}")
    logger.info("")

    try:
        stats_results = run_vbm_analysis(
            vbm_dir=gm_dir,
            participants_file=participants_file,
            formula=formula,
            contrasts=contrasts,
            n_permutations=5000,  # 5000 permutations for TFCE
            tfce=True,
            cluster_threshold=0.95  # p < 0.05 for conservative clusters
        )

        logger.info("")
        logger.info("✓ VBM statistical analysis complete")
        logger.info(f"  Statistical maps: {stats_results['randomise_dir']}")
        logger.info(f"  Cluster reports: {stats_results['cluster_reports']}")

    except Exception as e:
        logger.error(f"VBM statistical analysis failed: {e}", exc_info=True)
        return 1

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("VBM ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Subjects analyzed: {stats_results['n_subjects']}")
    logger.info(f"Tissue type: Grey Matter")
    logger.info(f"Smoothing: 4mm FWHM")
    logger.info("")
    logger.info("Results:")
    logger.info(f"  VBM directory: {gm_dir}")
    logger.info(f"  Statistical maps: {gm_dir / 'stats' / 'randomise_output'}")
    logger.info(f"  Cluster reports: {gm_dir / 'stats' / 'cluster_reports'}")
    logger.info("")
    logger.info("Cluster reports include:")
    logger.info("  - Top 10 clusters at p<0.05 (conservative)")
    logger.info("  - Additional clusters at p<0.3 (liberal, exploratory)")
    logger.info("  - Tri-planar mosaics (axial, coronal, sagittal)")
    logger.info("  - Anatomical localization (if available)")
    logger.info("  - HTML visualization")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
