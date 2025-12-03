#!/usr/bin/env python3
"""
Generate HTML cluster reports for ReHo and fALFF analyses
"""

import sys
from pathlib import Path

# Add neurovrai to path
sys.path.insert(0, str(Path(__file__).parent))

from neurovrai.analysis.stats.enhanced_cluster_report import create_enhanced_cluster_report


def generate_func_reports(
    analysis_dir: Path,
    metric: str,
    threshold: float = 0.5,
    study_name: str = 'mock_study'
):
    """
    Generate HTML reports for all contrasts in a functional analysis

    Args:
        analysis_dir: Path to analysis directory (e.g., analysis/func/reho/mock_study)
        metric: 'reho' or 'falff'
        threshold: Corrp threshold (default: 0.5 for p < 0.5)
        study_name: Name of the study (default: 'mock_study')
    """
    randomise_dir = analysis_dir / 'randomise_output'
    report_dir = analysis_dir / 'cluster_reports'
    report_dir.mkdir(exist_ok=True)

    # Find all corrp files
    corrp_files = sorted(randomise_dir.glob('randomise_tfce_corrp_tstat*.nii.gz'))

    if not corrp_files:
        print(f"No corrected p-value files found in {randomise_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Generating {metric.upper()} Cluster Reports")
    print(f"{'='*80}")
    print(f"Threshold: corrp > {threshold} (p < {1-threshold})")
    print(f"Output: {report_dir}")

    contrast_names = {
        'tstat1': 'age_positive',
        'tstat2': 'age_negative',
        'tstat3': 'group1_gt_group0',
        'tstat4': 'group0_gt_group1'
    }

    for corrp_file in corrp_files:
        # Extract contrast number
        filename = corrp_file.name
        for tstat, name in contrast_names.items():
            if tstat in filename:
                contrast_name = name
                break
        else:
            continue

        print(f"\n  Processing: {contrast_name}")

        # Get corresponding tstat file
        tstat_file = randomise_dir / filename.replace('tfce_corrp_', '')

        if not tstat_file.exists():
            print(f"    ✗ tstat file not found: {tstat_file}")
            continue

        # Generate report
        try:
            # Set analysis type (capitalize metric name: 'reho' -> 'ReHo', 'falff' -> 'fALFF')
            analysis_type_name = 'ReHo' if metric == 'reho' else 'fALFF'

            result = create_enhanced_cluster_report(
                stat_map=tstat_file,
                corrp_map=corrp_file,
                threshold=1-threshold,  # Convert corrp threshold to p-value (0.5 -> 0.5)
                output_dir=report_dir,
                contrast_name=f"{metric}_{contrast_name}",
                liberal_threshold=threshold,
                atlas_type='harvard-oxford',  # Use grey matter atlas for functional data
                study_name=study_name,
                analysis_type=analysis_type_name
            )
            if result and 'html_report' in result:
                print(f"    ✓ Report: {result['html_report']}")
            else:
                print(f"    ✓ Processing complete: {result}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"Reports saved to: {report_dir}")
    print(f"{'='*80}\n")


def main():
    study_root = Path('/mnt/bytopia/IRC805')
    analysis_dir = study_root / 'analysis' / 'func'

    # Generate reports for ReHo (mock_study)
    print("\n" + "="*80)
    print("REHO CLUSTER REPORTS (mock_study)")
    print("="*80)
    generate_func_reports(
        analysis_dir=analysis_dir / 'reho' / 'mock_study',
        metric='reho',
        threshold=0.5,
        study_name='mock_study'
    )

    # Generate reports for fALFF (mock_study)
    print("\n" + "="*80)
    print("FALFF CLUSTER REPORTS (mock_study)")
    print("="*80)
    generate_func_reports(
        analysis_dir=analysis_dir / 'falff' / 'mock_study',
        metric='falff',
        threshold=0.5,
        study_name='mock_study'
    )


if __name__ == '__main__':
    main()
