#!/usr/bin/env python3
"""
Batch Cluster Analysis for Functional/Anatomical Analyses
Generates enhanced cluster reports for ASL, ReHo, fALFF, and VBM results
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
        logging.FileHandler('logs/functional_cluster_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
ANALYSIS_DIR = Path('/mnt/bytopia/IRC805/analysis')
OUTPUT_BASE = ANALYSIS_DIR / 'cluster_reports'
P_THRESHOLD = 0.05  # Standard p-value threshold

# Analysis configurations
ANALYSES = {
    'ASL': {
        'dir': ANALYSIS_DIR / 'asl' / 'asl_analysis' / 'randomise_output',
        'prefix': 'randomise',
        'atlas': 'harvard-oxford',
        'n_contrasts': 6,
        'description': 'Cerebral Blood Flow (CBF)'
    },
    'ReHo': {
        'dir': ANALYSIS_DIR / 'func' / 'reho' / 'randomise',
        'prefix': 'reho_randomise',
        'atlas': 'harvard-oxford',
        'n_contrasts': 2,
        'description': 'Regional Homogeneity'
    },
    'fALFF': {
        'dir': ANALYSIS_DIR / 'func' / 'falff' / 'randomise',
        'prefix': 'falff_randomise',
        'atlas': 'harvard-oxford',
        'n_contrasts': 2,
        'description': 'Fractional ALFF'
    },
    'VBM': {
        'dir': ANALYSIS_DIR / 'anat' / 'vbm' / 'stats' / 'randomise_output',
        'prefix': 'randomise',
        'atlas': 'harvard-oxford',
        'n_contrasts': 2,
        'description': 'Voxel-Based Morphometry (Grey Matter)'
    }
}

# Contrast definitions
# Note: Not all analyses have all contrasts
ALL_CONTRASTS = {
    1: 'controlled_gt_uncontrolled',
    2: 'uncontrolled_gt_controlled',
    3: 'sex_positive',
    4: 'sex_negative',
    5: 'age_positive',
    6: 'age_negative'
}

def get_mni_template():
    """Get MNI152 T1 template for background"""
    import os
    fsldir = os.getenv('FSLDIR', '/usr/local/fsl')
    mni_template = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm.nii.gz'
    if mni_template.exists():
        return mni_template
    # Fallback to 1mm
    mni_template = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_1mm.nii.gz'
    if mni_template.exists():
        return mni_template
    raise FileNotFoundError("Cannot find MNI152 template")

def analyze_contrast(analysis_name, config, contrast_num, contrast_name, mni_template):
    """
    Analyze a single contrast for a given analysis

    Parameters
    ----------
    analysis_name : str
        Name of analysis (ASL, ReHo, fALFF, VBM)
    config : dict
        Analysis configuration
    contrast_num : int
        Contrast number (1-6)
    contrast_name : str
        Contrast name
    mni_template : Path
        Path to MNI template

    Returns
    -------
    dict or None
        Results dictionary or None if files not found
    """

    # File paths
    randomise_dir = config['dir']
    prefix = config['prefix']

    tstat_file = randomise_dir / f'{prefix}_tstat{contrast_num}.nii.gz'
    corrp_file = randomise_dir / f'{prefix}_tfce_corrp_tstat{contrast_num}.nii.gz'

    # Check if files exist
    if not tstat_file.exists():
        logger.warning(f"Missing tstat file: {tstat_file}")
        return None

    if not corrp_file.exists():
        # Try voxel-wise corrp as fallback
        corrp_file = randomise_dir / f'{prefix}_vox_corrp_tstat{contrast_num}.nii.gz'
        if not corrp_file.exists():
            logger.warning(f"Missing corrp file for {analysis_name} contrast {contrast_num}")
            return None

    # Output directory
    output_dir = OUTPUT_BASE / analysis_name / contrast_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing {analysis_name} - {contrast_name}")
    logger.info(f"  T-stat: {tstat_file.name}")
    logger.info(f"  Corrp: {corrp_file.name}")

    try:
        # Convert p-threshold to corrp threshold (p<0.05 -> corrp>0.95)
        corrp_threshold = 1.0 - P_THRESHOLD

        # Extract clusters with appropriate atlas
        clusters, bg_data = extract_clusters_with_atlas(
            stat_file=tstat_file,
            corrp_file=corrp_file,
            mean_fa_file=mni_template,
            p_thresh=corrp_threshold,
            min_size=5,
            top_n=10,
            atlas_type=config['atlas'],
            use_fmrib_template=False  # Use provided MNI template
        )

        logger.info(f"  Found {len(clusters)} significant clusters")

        if clusters:
            # Generate HTML report
            report_html = generate_enhanced_html_report(
                clusters=clusters,
                mean_fa_data=bg_data,
                contrast_name=f"{analysis_name} - {contrast_name}",
                output_dir=output_dir,
                stat_file=tstat_file,
                atlas_type=config['atlas'],
                analysis_type=analysis_name,
                study_name='IRC805',
                threshold_p=P_THRESHOLD
            )

            logger.info(f"  Report saved: {report_html}")

            # Return summary
            return {
                'analysis': analysis_name,
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
            report_html = output_dir / f"{analysis_name}_{contrast_name}_report.html"
            with open(report_html, 'w') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head><title>No Significant Clusters - {analysis_name} {contrast_name}</title></head>
<body style="font-family: Arial; padding: 40px;">
    <h1>No Significant Clusters Found</h1>
    <p><strong>Analysis:</strong> {analysis_name} - {config['description']}</p>
    <p><strong>Contrast:</strong> {contrast_name}</p>
    <p><strong>Threshold:</strong> p &lt; {P_THRESHOLD}</p>
    <p><strong>Study:</strong> IRC805</p>
</body>
</html>
""")

            return {
                'analysis': analysis_name,
                'contrast': contrast_name,
                'n_clusters': 0,
                'report': str(report_html)
            }

    except Exception as e:
        logger.error(f"Error analyzing {analysis_name} - {contrast_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Run batch cluster analysis on all functional/anatomical results"""

    logger.info("="*80)
    logger.info("Functional/Anatomical Batch Cluster Analysis")
    logger.info("="*80)
    logger.info(f"Output Directory: {OUTPUT_BASE}")
    logger.info(f"P-value Threshold: {P_THRESHOLD}")
    logger.info(f"Analyses: {', '.join(ANALYSES.keys())}")
    logger.info("="*80)

    # Get MNI template
    try:
        mni_template = get_mni_template()
        logger.info(f"MNI Template: {mni_template}")
    except FileNotFoundError as e:
        logger.error(f"Fatal error: {e}")
        return

    # Track results
    results = []
    successful = 0
    failed = 0
    no_clusters = 0

    # Process each analysis
    for analysis_name, config in ANALYSES.items():
        logger.info("")
        logger.info(f"Processing: {analysis_name} - {config['description']}")
        logger.info("-"*80)

        # Check if directory exists
        if not config['dir'].exists():
            logger.warning(f"Directory not found: {config['dir']}")
            continue

        # Process each contrast (only those that exist)
        for contrast_num in range(1, config['n_contrasts'] + 1):
            contrast_name = ALL_CONTRASTS.get(contrast_num, f'contrast_{contrast_num}')

            logger.info(f"[{analysis_name} Contrast {contrast_num}] {contrast_name}")

            result = analyze_contrast(analysis_name, config, contrast_num, contrast_name, mni_template)

            if result is not None:
                results.append(result)
                if result['n_clusters'] > 0:
                    successful += 1
                else:
                    no_clusters += 1
            else:
                failed += 1

    # Calculate total analyses
    total_analyses = sum(config['n_contrasts'] for config in ANALYSES.values())

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
    summary_file = OUTPUT_BASE / 'functional_analysis_summary.json'
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
            logger.info(f"{i}. {result['analysis']} - {result['contrast']}")
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
