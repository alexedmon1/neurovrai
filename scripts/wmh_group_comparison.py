#!/usr/bin/env python3
"""
WMH Group Comparison Script

Compares WMH metrics between groups defined in gludata.csv.
mriglu: 1 = controlled gestational diabetes, 2 = uncontrolled gestational diabetes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_wmh_data(hyperintensities_dir: Path) -> pd.DataFrame:
    """Load WMH metrics for all subjects."""
    data = []

    for subj_dir in sorted(hyperintensities_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name in ['group', 'logs']:
            continue

        metrics_file = subj_dir / 'wmh_metrics.json'
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        if 'error' in metrics:
            continue

        wmh_summary = metrics.get('wmh_summary', {})
        size_dist = metrics.get('size_distribution', {})

        # Extract subject ID without IRC805- prefix
        subject_id = subj_dir.name.replace('IRC805-', '')

        data.append({
            'subject_id': int(subject_id),
            'subject': subj_dir.name,
            'n_lesions': wmh_summary.get('n_lesions', 0),
            'total_volume_mm3': wmh_summary.get('total_volume_mm3', 0),
            'mean_lesion_volume': size_dist.get('mean_volume_mm3', 0),
            'max_lesion_volume': size_dist.get('max_volume_mm3', 0)
        })

    return pd.DataFrame(data)


def load_gludata(gludata_path: Path) -> pd.DataFrame:
    """Load and parse gludata.csv."""
    df = pd.read_csv(gludata_path, encoding='utf-8-sig')

    # Ensure Subject column is integer
    df['Subject'] = df['Subject'].astype(int)

    # Extract relevant columns
    cols_of_interest = ['Subject', 'mriglu', 'AGE', 'BMI', 'sex']
    available_cols = [c for c in cols_of_interest if c in df.columns]

    return df[available_cols]


def compare_groups(wmh_df: pd.DataFrame, glu_df: pd.DataFrame, output_dir: Path):
    """Compare WMH metrics between mriglu groups."""

    # Merge WMH data with group assignments
    merged = wmh_df.merge(glu_df, left_on='subject_id', right_on='Subject', how='inner')

    print("\n" + "=" * 80)
    print("WMH Analysis: Group Comparison (mriglu)")
    print("=" * 80)
    print(f"\nGroup definitions:")
    print("  mriglu=1: Poor glycemic control")
    print("  mriglu=2: Good glycemic control")

    # Count subjects per group
    group_counts = merged.groupby('mriglu').size()
    print(f"\nSample sizes:")
    for grp, count in group_counts.items():
        label = "Poor control" if grp == 1 else "Good control"
        print(f"  mriglu={grp} ({label}): n={count}")

    # Group statistics
    print("\n" + "-" * 60)
    print("Group Statistics:")
    print("-" * 60)

    metrics = ['n_lesions', 'total_volume_mm3', 'mean_lesion_volume', 'max_lesion_volume']
    metric_labels = {
        'n_lesions': 'Number of Lesions',
        'total_volume_mm3': 'Total WMH Volume (mm³)',
        'mean_lesion_volume': 'Mean Lesion Size (mm³)',
        'max_lesion_volume': 'Max Lesion Size (mm³)'
    }

    results = []

    for metric in metrics:
        group1 = merged[merged['mriglu'] == 1][metric]
        group2 = merged[merged['mriglu'] == 2][metric]

        # Descriptive stats
        mean1, std1 = group1.mean(), group1.std()
        mean2, std2 = group2.mean(), group2.std()

        # Statistical test (Mann-Whitney U for non-normal distributions)
        stat, p_mw = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        # Also compute t-test for reference
        t_stat, p_t = stats.ttest_ind(group1, group2)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1)-1)*std1**2 + (len(group2)-1)*std2**2) /
                            (len(group1) + len(group2) - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        print(f"\n{metric_labels[metric]}:")
        print(f"  Poor control (n={len(group1)}):  {mean1:.2f} ± {std1:.2f}")
        print(f"  Good control (n={len(group2)}):  {mean2:.2f} ± {std2:.2f}")
        print(f"  Mann-Whitney U p-value: {p_mw:.4f}")
        print(f"  T-test p-value: {p_t:.4f}")
        print(f"  Cohen's d: {cohens_d:.3f}")

        results.append({
            'metric': metric,
            'metric_label': metric_labels[metric],
            'poor_control_mean': mean1,
            'poor_control_std': std1,
            'poor_control_n': len(group1),
            'good_control_mean': mean2,
            'good_control_std': std2,
            'good_control_n': len(group2),
            'mannwhitney_p': p_mw,
            'ttest_p': p_t,
            'cohens_d': cohens_d
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'wmh_group_comparison.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")

    # Individual subject data
    subject_file = output_dir / 'wmh_by_subject_with_groups.csv'
    merged.to_csv(subject_file, index=False)
    print(f"Subject data saved to: {subject_file}")

    # Interpretation
    print("\n" + "=" * 80)
    print("Interpretation:")
    print("=" * 80)

    sig_findings = [r for r in results if r['mannwhitney_p'] < 0.05]

    if sig_findings:
        print("\nSignificant findings (p < 0.05):")
        for r in sig_findings:
            direction = "higher" if r['poor_control_mean'] > r['good_control_mean'] else "lower"
            print(f"  - {r['metric_label']}: Poor control group has {direction} values")
            print(f"    (p={r['mannwhitney_p']:.4f}, d={r['cohens_d']:.3f})")
    else:
        print("\nNo significant differences found between groups (all p > 0.05)")
        print("\nNote: This analysis is exploratory. Consider:")
        print("  - Sample sizes may be too small for statistical power")
        print("  - WMH detection without FLAIR has limited sensitivity")
        print("  - Age and other covariates may confound the results")

    return results_df, merged


def main():
    parser = argparse.ArgumentParser(description='Compare WMH between mriglu groups')
    parser.add_argument('--hyperintensities-dir', required=True, type=Path,
                       help='WMH analysis output directory')
    parser.add_argument('--gludata', required=True, type=Path,
                       help='Path to gludata.csv')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory (default: hyperintensities_dir/group)')

    args = parser.parse_args()

    output_dir = args.output_dir or (args.hyperintensities_dir / 'group')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    wmh_df = load_wmh_data(args.hyperintensities_dir)
    glu_df = load_gludata(args.gludata)

    logger.info(f"Loaded WMH data for {len(wmh_df)} subjects")
    logger.info(f"Loaded gludata for {len(glu_df)} subjects")

    # Run comparison
    compare_groups(wmh_df, glu_df, output_dir)


if __name__ == '__main__':
    main()
