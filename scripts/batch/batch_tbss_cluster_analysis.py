#!/usr/bin/env python3
"""
Batch TBSS Cluster Analysis
Generates enhanced cluster reports for all TBSS metrics and contrasts
"""
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from neurovrai.analysis.stats.enhanced_cluster_report import (
    extract_clusters_with_atlas,
    generate_enhanced_html_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tbss_cluster_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TBSS_DIR = Path('/mnt/bytopia/IRC805/analysis/tbss')
OUTPUT_BASE = TBSS_DIR / 'cluster_reports'
P_THRESHOLD = 0.05  # Standard p-value threshold

# Metrics to analyze
METRICS = ['FA', 'MD', 'AD', 'RD', 'MK', 'AK', 'RK', 'KFA', 'FICVF', 'ODI', 'FISO']

# Contrast definitions
# mriglu: 1=controlled, 2=uncontrolled
CONTRASTS = {
    1: 'controlled_gt_uncontrolled',
    2: 'uncontrolled_gt_controlled',
    3: 'sex_positive',
    4: 'sex_negative',
    5: 'age_positive',
    6: 'age_negative'
}

def get_mean_fa_skeleton():
    """Get path to mean FA skeleton for background"""
    mean_fa = TBSS_DIR / 'FA' / 'stats' / 'mean_FA_skeleton.nii.gz'
    if not mean_fa.exists():
        mean_fa = TBSS_DIR / 'stats' / 'mean_FA_skeleton.nii.gz'
    if not mean_fa.exists():
        raise FileNotFoundError(f"Cannot find mean_FA_skeleton.nii.gz")
    return mean_fa

def analyze_contrast(metric, contrast_num, contrast_name, mean_fa_skeleton):
    """
    Analyze a single metric/contrast combination

    Parameters
    ----------
    metric : str
        Metric name (e.g., 'FA', 'MK')
    contrast_num : int
        Contrast number (1-6)
    contrast_name : str
        Contrast name (e.g., 'mriglu_positive')
    mean_fa_skeleton : Path
        Path to mean FA skeleton for background

    Returns
    -------
    dict or None
        Results dictionary or None if files not found
    """

    # File paths
    randomise_dir = TBSS_DIR / metric / 'randomise_output'
    tstat_file = randomise_dir / f'randomise_tstat{contrast_num}.nii.gz'

    # Try TFCE corrected p-values first (preferred)
    corrp_file = randomise_dir / f'randomise_tfce_corrp_tstat{contrast_num}.nii.gz'

    # Fallback to voxel-wise corrected p-values
    if not corrp_file.exists():
        corrp_file = randomise_dir / f'randomise_vox_corrp_tstat{contrast_num}.nii.gz'

    # Check if files exist
    if not tstat_file.exists():
        logger.warning(f"Missing tstat file: {tstat_file}")
        return None

    if not corrp_file.exists():
        logger.warning(f"Missing corrp file for {metric} contrast {contrast_num}")
        return None

    # Output directory
    output_dir = OUTPUT_BASE / metric / contrast_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing {metric} - {contrast_name}")
    logger.info(f"  T-stat: {tstat_file.name}")
    logger.info(f"  Corrp: {corrp_file.name}")

    try:
        # Convert p-threshold to corrp threshold (p<0.05 -> corrp>0.95)
        corrp_threshold = 1.0 - P_THRESHOLD

        # Extract clusters with JHU atlas localization
        clusters, mean_fa_data = extract_clusters_with_atlas(
            stat_file=tstat_file,
            corrp_file=corrp_file,
            mean_fa_file=mean_fa_skeleton,
            p_thresh=corrp_threshold,
            min_size=5,
            top_n=10,
            atlas_type='jhu',  # Use JHU white matter atlas for TBSS
            use_fmrib_template=True  # Use FMRIB58_FA as background
        )

        logger.info(f"  Found {len(clusters)} significant clusters")

        if clusters:
            # Generate HTML report
            report_html = generate_enhanced_html_report(
                clusters=clusters,
                mean_fa_data=mean_fa_data,
                contrast_name=f"{metric} - {contrast_name}",
                output_dir=output_dir,
                stat_file=tstat_file,
                atlas_type='jhu',
                analysis_type='TBSS',
                study_name='IRC805',
                threshold_p=P_THRESHOLD
            )

            logger.info(f"  Report saved: {report_html}")

            # Return summary
            return {
                'metric': metric,
                'contrast': contrast_name,
                'n_clusters': len(clusters),
                'top_cluster': {
                    'size': clusters[0]['size'],
                    'peak_stat': clusters[0]['peak_stat'],
                    'peak_p': clusters[0]['peak_p'],
                    'location': clusters[0]['locations'][0]['region'] if clusters[0]['locations'] else 'Unknown'
                },
                'report': str(report_html)
            }
        else:
            logger.info(f"  No significant clusters found")
            # Create simple "no results" HTML
            report_html = output_dir / f"{metric}_{contrast_name}_report.html"
            with open(report_html, 'w') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head><title>No Significant Clusters - {metric} {contrast_name}</title></head>
<body style="font-family: Arial; padding: 40px;">
    <h1>No Significant Clusters Found</h1>
    <p><strong>Metric:</strong> {metric}</p>
    <p><strong>Contrast:</strong> {contrast_name}</p>
    <p><strong>Threshold:</strong> p &lt; {P_THRESHOLD}</p>
    <p><strong>Study:</strong> IRC805</p>
</body>
</html>
""")

            return {
                'metric': metric,
                'contrast': contrast_name,
                'n_clusters': 0,
                'report': str(report_html)
            }

    except Exception as e:
        logger.error(f"Error analyzing {metric} - {contrast_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Run batch cluster analysis on all TBSS results"""

    logger.info("="*80)
    logger.info("TBSS Batch Cluster Analysis")
    logger.info("="*80)
    logger.info(f"TBSS Directory: {TBSS_DIR}")
    logger.info(f"Output Directory: {OUTPUT_BASE}")
    logger.info(f"P-value Threshold: {P_THRESHOLD}")
    logger.info(f"Metrics: {', '.join(METRICS)}")
    logger.info(f"Contrasts: {len(CONTRASTS)}")
    logger.info("="*80)

    # Get mean FA skeleton
    try:
        mean_fa_skeleton = get_mean_fa_skeleton()
        logger.info(f"Mean FA Skeleton: {mean_fa_skeleton}")
    except FileNotFoundError as e:
        logger.error(f"Fatal error: {e}")
        return

    # Track results
    results = []
    successful = 0
    failed = 0
    no_clusters = 0

    # Process each metric and contrast
    total_analyses = len(METRICS) * len(CONTRASTS)
    current = 0

    for metric in METRICS:
        logger.info("")
        logger.info(f"Processing metric: {metric}")
        logger.info("-"*80)

        for contrast_num, contrast_name in CONTRASTS.items():
            current += 1
            logger.info(f"[{current}/{total_analyses}] {metric} - {contrast_name}")

            result = analyze_contrast(metric, contrast_num, contrast_name, mean_fa_skeleton)

            if result is not None:
                results.append(result)
                if result['n_clusters'] > 0:
                    successful += 1
                else:
                    no_clusters += 1
            else:
                failed += 1

    # Generate summary report
    logger.info("")
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total analyses: {total_analyses}")
    logger.info(f"Successful (with clusters): {successful}")
    logger.info(f"No significant clusters: {no_clusters}")
    logger.info(f"Failed: {failed}")
    logger.info("="*80)

    # Save summary JSON
    summary_file = OUTPUT_BASE / 'analysis_summary.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_analyses': total_analyses,
        'successful': successful,
        'no_clusters': no_clusters,
        'failed': failed,
        'p_threshold': P_THRESHOLD,
        'results': results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved: {summary_file}")

    # Print significant findings
    if successful > 0:
        logger.info("")
        logger.info("="*80)
        logger.info("SIGNIFICANT FINDINGS")
        logger.info("="*80)

        significant_results = [r for r in results if r['n_clusters'] > 0]
        significant_results.sort(key=lambda x: x['n_clusters'], reverse=True)

        for i, result in enumerate(significant_results, 1):
            logger.info(f"{i}. {result['metric']} - {result['contrast']}")
            logger.info(f"   Clusters: {result['n_clusters']}")
            logger.info(f"   Top cluster: {result['top_cluster']['size']} voxels, "
                       f"T={result['top_cluster']['peak_stat']:.2f}, "
                       f"p={result['top_cluster']['peak_p']:.4f}")
            logger.info(f"   Location: {result['top_cluster']['location']}")
            logger.info(f"   Report: {result['report']}")
            logger.info("")

    logger.info("All cluster reports saved to:")
    logger.info(f"  {OUTPUT_BASE}")
    logger.info("")
    logger.info("Done!")

if __name__ == '__main__':
    main()
