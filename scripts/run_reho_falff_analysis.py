#!/usr/bin/env python3
"""
Run ReHo and fALFF group-level analysis with FSL randomise and enhanced cluster reporting.

This script:
1. Computes ReHo and fALFF for all subjects with preprocessed functional data
2. Creates synthetic design matrix with age and group effects
3. Runs FSL randomise with TFCE for both metrics
4. Generates enhanced cluster reports with Harvard-Oxford grey matter atlas
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurovrai.analysis.func.resting_workflow import run_resting_state_analysis
from neurovrai.analysis.stats.enhanced_cluster_report import create_enhanced_cluster_report


def discover_subjects_with_func_data(derivatives_dir: Path):
    """Discover subjects with preprocessed functional data"""
    subjects = []
    for subj_dir in sorted(derivatives_dir.glob("IRC805-*")):
        func_dir = subj_dir / "func"
        if func_dir.exists():
            # Look for preprocessed BOLD file
            bold_files = list(func_dir.glob("*bold_preprocessed.nii.gz"))
            if bold_files:
                subjects.append(subj_dir.name)

    return subjects


def generate_synthetic_demographics(subjects: list, seed: int = 42):
    """Generate synthetic demographics for subjects"""
    np.random.seed(seed)
    n = len(subjects)

    # Generate synthetic data
    age = np.random.uniform(20, 70, n)  # Age 20-70
    group = np.random.choice([0, 1], n)  # Binary group (control vs patient)
    sex = np.random.choice([0, 1], n)  # Binary sex (0=F, 1=M)

    df = pd.DataFrame({
        'subject': subjects,
        'age': age,
        'group': group,
        'sex': sex
    })

    return df


def create_design_matrix_and_contrasts(demographics: pd.DataFrame, output_dir: Path):
    """
    Create FSL design matrix and contrast files

    Design matrix columns:
    1. Intercept
    2. Age (demeaned)
    3. Group (patient - control)
    4. Sex (M - F)
    """
    n = len(demographics)

    # Demean continuous variables
    age_dm = demographics['age'] - demographics['age'].mean()

    # Create design matrix
    design = np.column_stack([
        np.ones(n),  # Intercept
        age_dm,      # Age effect
        demographics['group'],  # Group effect
        demographics['sex']     # Sex effect
    ])

    # Save design matrix in FSL vest format
    design_file = output_dir / 'design.mat'
    with open(design_file, 'w') as f:
        f.write('/NumWaves 4\n')
        f.write(f'/NumPoints {n}\n')
        f.write('/Matrix\n')
        for row in design:
            f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

    # Create contrast matrix
    # Contrasts:
    # 1. Age positive (regions increase with age)
    # 2. Age negative (regions decrease with age)
    # 3. Group difference (patient > control)
    # 4. Sex difference (M > F)
    contrasts = np.array([
        [0, 1, 0, 0],   # Age positive
        [0, -1, 0, 0],  # Age negative
        [0, 0, 1, 0],   # Group difference
        [0, 0, 0, 1]    # Sex difference
    ])

    contrast_file = output_dir / 'design.con'
    with open(contrast_file, 'w') as f:
        f.write('/NumWaves 4\n')
        f.write(f'/NumContrasts {len(contrasts)}\n')
        f.write('/Matrix\n')
        for row in contrasts:
            f.write(' '.join(f'{x:.6f}' for x in row) + '\n')

    # Save demographics with subject list for randomise
    subj_list_file = output_dir / 'subjects.txt'
    demographics[['subject']].to_csv(subj_list_file, index=False, header=False)

    demographics_file = output_dir / 'demographics.csv'
    demographics.to_csv(demographics_file, index=False)

    return design_file, contrast_file, demographics_file


def merge_4d_metric(subjects: list, derivatives_dir: Path, metric: str):
    """Merge individual subject metric maps into 4D file"""
    print(f"\nMerging {metric} maps for {len(subjects)} subjects...")

    metric_files = []
    missing = []

    for subj in subjects:
        metric_file = derivatives_dir / subj / 'func' / f'{subj}_{metric}_z.nii.gz'
        if metric_file.exists():
            metric_files.append(metric_file)
        else:
            print(f"  Warning: Missing {metric} for {subj}")
            missing.append(subj)

    if missing:
        print(f"  Skipping {len(missing)} subjects with missing {metric} data")

    if len(metric_files) < 3:
        raise ValueError(f"Not enough {metric} files found (need at least 3, found {len(metric_files)})")

    # Load all images
    imgs = [nib.load(str(f)) for f in metric_files]

    # Stack into 4D
    data_4d = np.stack([img.get_fdata() for img in imgs], axis=-1)

    # Create 4D image
    img_4d = nib.Nifti1Image(data_4d, imgs[0].affine, imgs[0].header)

    return img_4d, [s for s, f in zip(subjects, metric_files) if f in metric_files]


def run_randomise(input_4d: Path, design_mat: Path, contrast: Path,
                 output_dir: Path, n_perm: int = 5000):
    """Run FSL randomise with TFCE"""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / 'randomise'

    cmd = [
        'randomise',
        '-i', str(input_4d),
        '-o', str(output_base),
        '-d', str(design_mat),
        '-t', str(contrast),
        '-n', str(n_perm),
        '-T',  # TFCE
        '-V'   # Verbose
    ]

    print(f"\nRunning randomise with {n_perm} permutations...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running randomise:")
        print(result.stderr)
        raise RuntimeError("Randomise failed")

    print("Randomise completed successfully")
    return output_base


def generate_cluster_reports(randomise_base: Path, output_dir: Path,
                             contrast_names: list, threshold: float = 0.05):
    """Generate enhanced cluster reports with Harvard-Oxford atlas"""
    print(f"\nGenerating cluster reports (p < {threshold})...")

    reports_dir = output_dir / 'cluster_reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Background image for visualization (MNI152 T1)
    import os
    fsldir = os.getenv('FSLDIR', '/usr/local/fsl')
    bg_image = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm.nii.gz'

    for i, contrast_name in enumerate(contrast_names, 1):
        print(f"\n  Processing contrast {i}: {contrast_name}")

        stat_map = Path(str(randomise_base) + f'_tstat{i}.nii.gz')
        corrp_map = Path(str(randomise_base) + f'_tfce_corrp_tstat{i}.nii.gz')

        if not stat_map.exists() or not corrp_map.exists():
            print(f"    Warning: Missing files for contrast {i}")
            continue

        # Check max corrp value
        corrp_img = nib.load(corrp_map)
        max_corrp = np.max(corrp_img.get_fdata())
        print(f"    Max corrp: {max_corrp:.4f} (min p: {1-max_corrp:.4f})")

        try:
            report_results = create_enhanced_cluster_report(
                stat_map=stat_map,
                corrp_map=corrp_map,
                threshold=threshold,
                output_dir=reports_dir / contrast_name,
                contrast_name=contrast_name,
                max_clusters=10,
                background_image=bg_image,
                atlas_type='harvard-oxford'
            )

            results[contrast_name] = report_results
            print(f"    Found {report_results['n_clusters']} clusters")
            print(f"    Report: {report_results['report_html']}")

        except Exception as e:
            print(f"    Error generating report: {e}")
            continue

    return results


def main():
    """Main analysis pipeline"""
    print("="*80)
    print("ReHo/fALFF Group-Level Analysis with FSL Randomise")
    print("="*80)

    # Configuration
    derivatives_dir = Path('/mnt/bytopia/IRC805/derivatives')
    analysis_dir = Path('/mnt/bytopia/IRC805/analysis/resting_state')
    analysis_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Step 1: Discover subjects with functional data
    print("\n[1/7] Discovering subjects with preprocessed functional data...")
    subjects = discover_subjects_with_func_data(derivatives_dir)
    print(f"Found {len(subjects)} subjects with functional data")

    if len(subjects) < 3:
        print("Error: Need at least 3 subjects for group analysis")
        return

    # Step 2: Compute ReHo and fALFF for all subjects
    print("\n[2/7] Computing ReHo and fALFF metrics...")
    failed_subjects = []

    for i, subject in enumerate(subjects, 1):
        print(f"\n  [{i}/{len(subjects)}] Processing {subject}...")

        func_dir = derivatives_dir / subject / 'func'
        bold_file = next(func_dir.glob("*bold_preprocessed.nii.gz"), None)

        if not bold_file:
            print(f"    Warning: No preprocessed BOLD file found")
            failed_subjects.append(subject)
            continue

        try:
            results = run_resting_state_analysis(
                func_file=bold_file,
                mask_file=None,  # Will auto-detect or compute
                output_dir=func_dir,
                subject_id=subject,
                reho_neighborhood=27,
                falff_low_freq=0.01,
                falff_high_freq=0.08,
                compute_zscore=True
            )

            print(f"    ✓ ReHo: {results['reho']['output']}")
            print(f"    ✓ fALFF: {results['falff']['output']}")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed_subjects.append(subject)
            continue

    # Remove failed subjects
    subjects = [s for s in subjects if s not in failed_subjects]
    print(f"\nSuccessfully processed {len(subjects)} subjects")

    if len(subjects) < 3:
        print("Error: Too few subjects remaining after processing")
        return

    # Step 3: Generate synthetic demographics
    print("\n[3/7] Generating synthetic demographics...")
    demographics = generate_synthetic_demographics(subjects)
    print(f"Generated demographics for {len(demographics)} subjects")
    print(f"  Age range: {demographics['age'].min():.1f} - {demographics['age'].max():.1f}")
    print(f"  Group distribution: {demographics['group'].value_counts().to_dict()}")

    # Step 4: Create design matrix and contrasts
    print("\n[4/7] Creating design matrix and contrasts...")
    design_file, contrast_file, demographics_file = create_design_matrix_and_contrasts(
        demographics, analysis_dir)
    print(f"  Design matrix: {design_file}")
    print(f"  Contrasts: {contrast_file}")
    print(f"  Demographics: {demographics_file}")

    # Step 5: Merge ReHo and fALFF into 4D files
    print("\n[5/7] Merging metric maps into 4D files...")

    reho_4d, reho_subjects = merge_4d_metric(subjects, derivatives_dir, 'reho')
    reho_4d_file = analysis_dir / 'reho_4d.nii.gz'
    nib.save(reho_4d, reho_4d_file)
    print(f"  ReHo 4D: {reho_4d_file} ({len(reho_subjects)} subjects)")

    falff_4d, falff_subjects = merge_4d_metric(subjects, derivatives_dir, 'falff')
    falff_4d_file = analysis_dir / 'falff_4d.nii.gz'
    nib.save(falff_4d, falff_4d_file)
    print(f"  fALFF 4D: {falff_4d_file} ({len(falff_subjects)} subjects)")

    # Verify subject lists match
    if reho_subjects != falff_subjects:
        print("Warning: Subject lists don't match between ReHo and fALFF")
        return

    # Step 6: Run randomise for ReHo
    print("\n[6/7] Running FSL randomise...")

    print("\n  === ReHo Analysis ===")
    reho_output_dir = analysis_dir / 'reho' / 'stats'
    reho_randomise_base = run_randomise(
        reho_4d_file, design_file, contrast_file, reho_output_dir, n_perm=5000)

    print("\n  === fALFF Analysis ===")
    falff_output_dir = analysis_dir / 'falff' / 'stats'
    falff_randomise_base = run_randomise(
        falff_4d_file, design_file, contrast_file, falff_output_dir, n_perm=5000)

    # Step 7: Generate cluster reports
    print("\n[7/7] Generating cluster reports with Harvard-Oxford atlas...")

    contrast_names = ['age_positive', 'age_negative', 'group_difference', 'sex_difference']

    print("\n  === ReHo Cluster Reports ===")
    reho_reports = generate_cluster_reports(
        reho_randomise_base, reho_output_dir, contrast_names, threshold=0.05)

    print("\n  === fALFF Cluster Reports ===")
    falff_reports = generate_cluster_reports(
        falff_randomise_base, falff_output_dir, contrast_names, threshold=0.05)

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAnalysis directory: {analysis_dir}")
    print(f"Number of subjects: {len(reho_subjects)}")
    print(f"\nReHo results:")
    for name, results in reho_reports.items():
        print(f"  {name}: {results['n_clusters']} clusters")
    print(f"\nfALFF results:")
    for name, results in falff_reports.items():
        print(f"  {name}: {results['n_clusters']} clusters")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
