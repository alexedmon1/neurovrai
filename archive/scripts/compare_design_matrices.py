#!/usr/bin/env python3
"""
Compare design matrices generated from gludata.csv vs participants_matched.tsv

This script:
1. Loads gludata.csv (master participants file)
2. For each analysis type, identifies subjects with available data
3. Generates new design matrices using filtered gludata.csv
4. Compares new vs old design matrices
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add neurovrai to path
sys.path.insert(0, str(Path(__file__).parent))

from neuroaider import DesignHelper


def find_subjects_with_data(study_root: Path, analysis_type: str):
    """
    Find subjects that have data for a specific analysis type

    Args:
        study_root: Study root directory
        analysis_type: Type of analysis ('vbm', 'asl', 'func_reho', 'func_falff', 'tbss')

    Returns:
        List of subject IDs with data
    """
    subjects = []

    if analysis_type == 'vbm':
        # Check for VBM smoothed GM files
        vbm_dir = study_root / 'analysis' / 'anat' / 'vbm' / 'subjects'
        if vbm_dir.exists():
            for subj_file in vbm_dir.glob('IRC805-*_GM_mni_smooth.nii.gz'):
                subject_id = subj_file.stem.replace('_GM_mni_smooth', '')
                subjects.append(subject_id)

    elif analysis_type == 'asl':
        # Check for ASL CBF MNI files
        derivatives_dir = study_root / 'derivatives'
        for subj_dir in derivatives_dir.glob('IRC805-*/asl'):
            subject_id = subj_dir.parent.name
            cbf_file = subj_dir / f'{subject_id}_cbf_mni.nii.gz'
            if cbf_file.exists():
                subjects.append(subject_id)

    elif analysis_type == 'func_reho':
        # Check for ReHo MNI files
        derivatives_dir = study_root / 'derivatives'
        for subj_dir in derivatives_dir.glob('IRC805-*/func'):
            subject_id = subj_dir.parent.name
            reho_file = subj_dir / 'reho' / 'reho_mni_zscore_masked.nii.gz'
            if reho_file.exists():
                subjects.append(subject_id)

    elif analysis_type == 'func_falff':
        # Check for fALFF MNI files
        derivatives_dir = study_root / 'derivatives'
        for subj_dir in derivatives_dir.glob('IRC805-*/func'):
            subject_id = subj_dir.parent.name
            falff_file = subj_dir / 'falff' / 'falff_mni_zscore_masked.nii.gz'
            if falff_file.exists():
                subjects.append(subject_id)

    elif analysis_type == 'tbss':
        # Check for TBSS FA files
        tbss_fa_dir = study_root / 'analysis' / 'tbss' / 'FA'
        if tbss_fa_dir.exists():
            for fa_file in tbss_fa_dir.glob('IRC805-*_FA.nii.gz'):
                subject_id = fa_file.stem.replace('_FA', '')
                subjects.append(subject_id)

    return sorted(subjects)


def load_and_filter_gludata(gludata_file: Path, subject_ids: list):
    """
    Load gludata.csv and filter to specific subjects

    Args:
        gludata_file: Path to gludata.csv
        subject_ids: List of subject IDs to keep

    Returns:
        Filtered DataFrame
    """
    # Load gludata
    df = pd.read_csv(gludata_file)

    # Create participant_id column from Subject
    df['participant_id'] = 'IRC805-' + df['Subject'].astype(str)

    # Standardize column names (lowercase)
    # Rename AGE -> age for compatibility
    if 'AGE' in df.columns:
        df['age'] = df['AGE']

    # Filter to subjects with data
    df_filtered = df[df['participant_id'].isin(subject_ids)].copy()

    # Sort by participant_id to match analysis order
    df_filtered = df_filtered.sort_values('participant_id').reset_index(drop=True)

    return df_filtered


def generate_design_from_df(df: pd.DataFrame, formula: str, analysis_name: str):
    """
    Generate design matrix from DataFrame

    Args:
        df: Participants DataFrame
        formula: Model formula
        analysis_name: Name for logging

    Returns:
        Dictionary with design matrix info
    """
    print(f"\n{'='*80}")
    print(f"GENERATING DESIGN: {analysis_name}")
    print(f"{'='*80}")
    print(f"  Subjects: {len(df)}")
    print(f"  Formula: {formula}")

    # Parse formula
    formula_terms = [t.strip() for t in formula.split('+')]
    first_var = formula_terms[0].replace('C(', '').replace(')', '')

    # Detect binary coding
    use_binary_coding = False
    if first_var in df.columns:
        n_levels = df[first_var].nunique()
        if n_levels == 2:
            use_binary_coding = True
            print(f"  ✓ Binary group variable '{first_var}' detected")

    # Initialize DesignHelper
    helper = DesignHelper(
        participants_file=df,
        subject_column='participant_id',
        add_intercept=not use_binary_coding
    )

    # Add variables
    for term in formula_terms:
        var_name = term.replace('C(', '').replace(')', '').strip()

        if pd.api.types.is_numeric_dtype(df[var_name]):
            n_unique = df[var_name].nunique()
            if n_unique <= 10 and use_binary_coding and var_name == first_var:
                helper.add_categorical(var_name, coding='dummy')
            else:
                helper.add_covariate(var_name, mean_center=True)
        else:
            helper.add_categorical(var_name, coding='effect' if not use_binary_coding else 'dummy')

    # Build design matrix
    design_mat, column_names = helper.build_design_matrix()

    # Add contrasts
    if use_binary_coding:
        helper.add_binary_group_contrasts(first_var)

    for term in formula_terms:
        var_name = term.replace('C(', '').replace(')', '').strip()
        if var_name in column_names and var_name != first_var:
            helper.add_contrast(f"{var_name}_positive", covariate=var_name, direction='+')
            helper.add_contrast(f"{var_name}_negative", covariate=var_name, direction='-')

    # Get contrast matrix
    contrast_mat, contrast_names = helper.build_contrast_matrix()

    print(f"  ✓ Design matrix: {design_mat.shape}")
    print(f"  ✓ Columns: {column_names}")
    print(f"  ✓ Contrasts: {contrast_names}")

    return {
        'design_mat': design_mat,
        'column_names': column_names,
        'contrast_mat': contrast_mat,
        'contrast_names': contrast_names,
        'subjects': df['participant_id'].tolist(),
        'helper': helper
    }


def compare_designs(old_dir: Path, new_result: dict, analysis_name: str):
    """
    Compare old and new design matrices

    Args:
        old_dir: Directory with old design files
        new_result: New design result dictionary
        analysis_name: Name for logging

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*80}")
    print(f"COMPARING DESIGNS: {analysis_name}")
    print(f"{'='*80}")

    differences = []

    # Load old design summary
    old_summary_file = old_dir / 'design_summary.json'
    if old_summary_file.exists():
        with open(old_summary_file) as f:
            old_summary = json.load(f)

        print(f"\nOLD design:")
        print(f"  Subjects: {old_summary['n_subjects']}")
        print(f"  Columns: {old_summary['columns']}")
        print(f"  Contrasts: {old_summary['contrasts']}")

        print(f"\nNEW design:")
        print(f"  Subjects: {len(new_result['subjects'])}")
        print(f"  Columns: {new_result['column_names']}")
        print(f"  Contrasts: {new_result['contrast_names']}")

        # Compare subject count
        if old_summary['n_subjects'] != len(new_result['subjects']):
            diff = f"Subject count differs: {old_summary['n_subjects']} -> {len(new_result['subjects'])}"
            differences.append(diff)
            print(f"  ⚠ {diff}")

        # Compare columns
        if old_summary['columns'] != new_result['column_names']:
            diff = f"Columns differ: {old_summary['columns']} -> {new_result['column_names']}"
            differences.append(diff)
            print(f"  ⚠ {diff}")

        # Compare contrasts
        if old_summary['contrasts'] != new_result['contrast_names']:
            diff = f"Contrasts differ: {old_summary['contrasts']} -> {new_result['contrast_names']}"
            differences.append(diff)
            print(f"  ⚠ {diff}")

        # Compare subject order
        if 'subjects' in old_summary:
            if old_summary['subjects'] != new_result['subjects']:
                diff = "Subject order differs"
                differences.append(diff)
                print(f"  ⚠ {diff}")
                print(f"     OLD: {old_summary['subjects'][:5]}...")
                print(f"     NEW: {new_result['subjects'][:5]}...")

    if not differences:
        print(f"\n  ✓ No differences found - designs are identical!")

    return {
        'analysis': analysis_name,
        'differences': differences,
        'identical': len(differences) == 0
    }


def main():
    """Main execution"""
    study_root = Path('/mnt/bytopia/IRC805')
    gludata_file = study_root / 'data' / 'gludata.csv'
    designs_dir = study_root / 'data' / 'designs'
    backup_dir = Path('/mnt/bytopia/IRC805/data/designs_backup_20251205_180011')

    formula = 'mriglu+sex+age'

    # Analysis types to process
    analysis_types = ['vbm', 'asl', 'func_reho', 'func_falff', 'tbss']

    print(f"{'='*80}")
    print(f"DESIGN MATRIX COMPARISON")
    print(f"{'='*80}")
    print(f"Gludata file: {gludata_file}")
    print(f"Backup directory: {backup_dir}")
    print(f"Formula: {formula}")

    comparison_results = []

    for analysis_type in analysis_types:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {analysis_type.upper()}")
        print(f"{'='*80}")

        # Find subjects with data
        subjects = find_subjects_with_data(study_root, analysis_type)
        print(f"  Found {len(subjects)} subjects with {analysis_type} data")

        if len(subjects) == 0:
            print(f"  ⚠ Skipping - no subjects found")
            continue

        # Load and filter gludata
        df = load_and_filter_gludata(gludata_file, subjects)
        print(f"  Filtered gludata to {len(df)} subjects")

        # Generate new design
        new_result = generate_design_from_df(df, formula, analysis_type.upper())

        # Compare with old design
        old_dir = backup_dir / analysis_type
        comparison = compare_designs(old_dir, new_result, analysis_type.upper())
        comparison_results.append(comparison)

        # Save new design to temp location for inspection
        temp_dir = designs_dir / f'{analysis_type}_NEW'
        temp_dir.mkdir(parents=True, exist_ok=True)

        new_result['helper'].save(
            design_mat_file=temp_dir / 'design.mat',
            design_con_file=temp_dir / 'design.con',
            summary_file=temp_dir / 'design_summary.json'
        )

        print(f"\n  ✓ Saved new design to: {temp_dir}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")

    for result in comparison_results:
        if result['identical']:
            print(f"  ✓ {result['analysis']}: IDENTICAL")
        else:
            print(f"  ⚠ {result['analysis']}: DIFFERENCES FOUND")
            for diff in result['differences']:
                print(f"      - {diff}")

    print()


if __name__ == '__main__':
    main()
