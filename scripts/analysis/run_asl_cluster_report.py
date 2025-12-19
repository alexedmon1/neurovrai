#!/usr/bin/env python3
"""
ASL Cluster Analysis and Report Generation

Runs cluster analysis on ASL randomise outputs at specified thresholds
and generates HTML reports with anatomical localization.

Usage:
    python run_asl_cluster_report.py --randomise-dir /path/to/randomise_output --threshold 0.30
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurovrai.analysis.stats.enhanced_cluster_report import (
    extract_clusters_with_atlas,
    generate_enhanced_html_report,
    load_harvard_oxford_atlas
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Contrast names from the design
CONTRAST_NAMES = {
    1: 'mriGlu_vs_NoMriGlu',
    2: 'NoMriGlu_vs_mriGlu',
    3: 'Positive_Age_Effect',
    4: 'Negative_Age_Effect',
    5: 'Male_vs_Female',
    6: 'Female_vs_Male'
}


def run_cluster_analysis(
    randomise_dir: Path,
    output_dir: Path,
    threshold: float = 0.30,
    study_name: str = 'IRC805',
    analysis_type: str = 'ASL-CBF'
) -> List[Dict]:
    """
    Run cluster analysis on all contrasts in randomise output directory.

    Args:
        randomise_dir: Directory containing randomise outputs
        output_dir: Directory for cluster reports
        threshold: P-value threshold (default: 0.30 for exploratory)
        study_name: Name of study
        analysis_type: Type of analysis (e.g., 'ASL-CBF')

    Returns:
        List of results dictionaries
    """
    randomise_dir = Path(randomise_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert p-value threshold to corrp threshold
    # p < 0.30 means corrp > 0.70
    corrp_threshold = 1.0 - threshold

    logger.info("=" * 70)
    logger.info(f"ASL CLUSTER ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Randomise directory: {randomise_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"P-value threshold: p < {threshold}")
    logger.info(f"Corrp threshold: corrp > {corrp_threshold}")
    logger.info("=" * 70)

    results = []

    # Find all contrast outputs
    for tstat_idx in range(1, 7):  # 6 contrasts
        tstat_file = randomise_dir / f'randomise_tstat{tstat_idx}.nii.gz'
        corrp_file = randomise_dir / f'randomise_tfce_corrp_tstat{tstat_idx}.nii.gz'

        if not tstat_file.exists() or not corrp_file.exists():
            logger.warning(f"Contrast {tstat_idx}: Missing files, skipping")
            continue

        contrast_name = CONTRAST_NAMES.get(tstat_idx, f'contrast_{tstat_idx}')
        logger.info(f"\n[Contrast {tstat_idx}] {contrast_name}")

        try:
            # Get MNI152 template as background
            import os
            fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')
            mni_template = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm.nii.gz'

            # Extract clusters using Harvard-Oxford atlas (for CBF/grey matter)
            clusters, bg_data = extract_clusters_with_atlas(
                stat_file=tstat_file,
                corrp_file=corrp_file,
                mean_fa_file=mni_template,
                p_thresh=corrp_threshold,
                min_size=5,
                top_n=10,
                atlas_type='harvard-oxford',
                use_fmrib_template=False
            )

            logger.info(f"  Found {len(clusters)} clusters at p < {threshold}")

            if clusters:
                # Generate HTML report
                contrast_output_dir = output_dir / f'contrast_{tstat_idx}_{contrast_name}'

                report_path = generate_enhanced_html_report(
                    clusters=clusters,
                    mean_fa_data=bg_data,
                    contrast_name=contrast_name,
                    output_dir=contrast_output_dir,
                    stat_file=tstat_file,
                    atlas_type='harvard-oxford',
                    analysis_type=analysis_type,
                    study_name=study_name,
                    threshold_p=threshold
                )

                logger.info(f"  Report: {report_path}")

                results.append({
                    'contrast': tstat_idx,
                    'name': contrast_name,
                    'n_clusters': len(clusters),
                    'report_path': str(report_path),
                    'clusters': [
                        {
                            'id': c['cluster_id'],
                            'size': c['size'],
                            'peak_stat': c['peak_stat'],
                            'peak_p': c['peak_p'],
                            'peak_coords': c['peak_coords'],
                            'locations': c['locations']
                        }
                        for c in clusters
                    ]
                })
            else:
                results.append({
                    'contrast': tstat_idx,
                    'name': contrast_name,
                    'n_clusters': 0,
                    'report_path': None,
                    'clusters': []
                })
                logger.info(f"  No significant clusters found")

        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                'contrast': tstat_idx,
                'name': contrast_name,
                'error': str(e)
            })

    return results


def generate_summary_html(
    results: List[Dict],
    output_dir: Path,
    threshold: float,
    study_name: str,
    analysis_type: str
) -> Path:
    """Generate summary HTML report linking all contrast reports."""

    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{analysis_type} Cluster Analysis Summary - {study_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .contrast-card {{
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
        }}
        .contrast-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .contrast-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .cluster-count {{
            background: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .cluster-count.none {{
            background: #95a5a6;
        }}
        .cluster-count.found {{
            background: #27ae60;
        }}
        .view-link {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 15px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }}
        .view-link:hover {{
            background: #2980b9;
        }}
        .cluster-preview {{
            margin-top: 15px;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }}
        .region-tag {{
            display: inline-block;
            background: #ecf0f1;
            padding: 3px 10px;
            margin: 2px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #dee2e6;
            color: #7f8c8d;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 {analysis_type} Cluster Analysis Summary</h1>
        <h3 style="color: #7f8c8d;">Study: {study_name} | Generated: {timestamp}</h3>

        <div class="summary-box">
            <strong>Threshold:</strong> p &lt; {threshold} (TFCE corrected) |
            <strong>Atlas:</strong> Harvard-Oxford Cortical & Subcortical |
            <strong>Minimum cluster size:</strong> 5 voxels
        </div>
"""

    total_clusters = sum(r.get('n_clusters', 0) for r in results)
    html += f"""
        <div class="summary-box" style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);">
            <strong>Total significant clusters found:</strong> {total_clusters} across {len(results)} contrasts
        </div>
"""

    for result in results:
        n_clusters = result.get('n_clusters', 0)
        count_class = 'found' if n_clusters > 0 else 'none'

        html += f"""
        <div class="contrast-card">
            <div class="contrast-header">
                <span class="contrast-name">{result['name']}</span>
                <span class="cluster-count {count_class}">{n_clusters} clusters</span>
            </div>
"""

        if n_clusters > 0 and result.get('report_path'):
            # Get relative path for link
            report_path = Path(result['report_path'])
            rel_path = report_path.relative_to(output_dir)

            html += f"""
            <a href="{rel_path}" class="view-link">📊 View Detailed Report</a>

            <div class="cluster-preview">
                <strong>Top regions:</strong><br>
"""
            # Show top regions from first few clusters
            regions_shown = set()
            for cluster in result.get('clusters', [])[:3]:
                for loc in cluster.get('locations', [])[:2]:
                    region = loc['region']
                    if region not in regions_shown:
                        regions_shown.add(region)
                        html += f'<span class="region-tag">{region}</span>\n'

            html += """
            </div>
"""
        elif 'error' in result:
            html += f"""
            <div style="color: #e74c3c; margin-top: 10px;">
                ⚠️ Error: {result['error']}
            </div>
"""

        html += """
        </div>
"""

    html += f"""
        <div class="footer">
            <p>Generated by neurovrai enhanced cluster analysis pipeline</p>
            <p>Threshold: p &lt; {threshold} | Atlas: Harvard-Oxford</p>
        </div>
    </div>
</body>
</html>
"""

    summary_file = output_dir / 'cluster_analysis_summary.html'
    with open(summary_file, 'w') as f:
        f.write(html)

    return summary_file


def main():
    parser = argparse.ArgumentParser(description='ASL Cluster Analysis')
    parser.add_argument(
        '--randomise-dir',
        type=Path,
        required=True,
        help='Directory containing randomise outputs'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for reports (default: randomise_dir/cluster_reports_pXX)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.30,
        help='P-value threshold (default: 0.30 for exploratory)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='IRC805',
        help='Study name for reports'
    )
    parser.add_argument(
        '--analysis-type',
        type=str,
        default='ASL-CBF',
        help='Analysis type for reports'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        threshold_str = str(args.threshold).replace('.', '')
        args.output_dir = args.randomise_dir / f'cluster_reports_p{threshold_str}'

    # Run cluster analysis
    results = run_cluster_analysis(
        randomise_dir=args.randomise_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        study_name=args.study_name,
        analysis_type=args.analysis_type
    )

    # Generate summary HTML
    summary_path = generate_summary_html(
        results=results,
        output_dir=args.output_dir,
        threshold=args.threshold,
        study_name=args.study_name,
        analysis_type=args.analysis_type
    )

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    total_clusters = 0
    for r in results:
        n = r.get('n_clusters', 0)
        total_clusters += n
        status = f"{n} clusters" if n > 0 else "No clusters"
        logger.info(f"  {r['name']}: {status}")

    logger.info(f"\nTotal clusters: {total_clusters}")
    logger.info(f"Summary report: {summary_path}")

    # Save results JSON
    results_file = args.output_dir / 'cluster_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'threshold': args.threshold,
            'study_name': args.study_name,
            'analysis_type': args.analysis_type,
            'results': results
        }, f, indent=2)

    logger.info(f"Results JSON: {results_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
