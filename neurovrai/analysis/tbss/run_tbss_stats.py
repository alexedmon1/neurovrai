#!/usr/bin/env python3
"""
TBSS Statistical Analysis Workflow

Runs group-level statistical analysis on prepared TBSS data using FSL randomise.

This workflow:
1. Loads participant data and filters to match prepared subjects
2. Generates design matrix and contrasts from model formula
3. Executes FSL randomise with TFCE for nonparametric inference
4. Extracts significant clusters and generates reports
5. Supports multiple iterations with different models on same prepared data

Prerequisites:
- Completed TBSS data preparation (prepare_tbss.py)
- Participants CSV with demographic/clinical data
- Contrast specifications (inline or YAML file)

Usage:
    python -m neurovrai.analysis.tbss.run_tbss_stats \\
        --data-dir /study/analysis/tbss_FA/ \\
        --participants participants.csv \\
        --formula "age + sex + exposure" \\
        --contrasts contrasts.yaml \\
        --output-dir /study/analysis/tbss_FA/model_age_sex_exposure/
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from neurovrai.analysis.stats.design_matrix import generate_design_files
from neurovrai.analysis.stats.randomise_wrapper import run_randomise, summarize_results
from neurovrai.analysis.stats.glm_wrapper import run_fsl_glm, threshold_zstat, summarize_glm_results
from neurovrai.analysis.stats.cluster_report import generate_reports_for_all_contrasts


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"tbss_stats_{timestamp}.log"

    logger = logging.getLogger("tbss_stats")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def load_contrasts(contrasts_file: Path) -> List[Dict]:
    """
    Load contrast specifications from YAML file

    Args:
        contrasts_file: Path to contrasts YAML file

    Returns:
        List of contrast dictionaries

    Example YAML format:
        contrasts:
          - name: age_positive
            vector: [0, 1, 0, 0]
          - name: exposure_negative
            vector: [0, 0, 0, -1]
    """
    if not contrasts_file.exists():
        raise FileNotFoundError(f"Contrasts file not found: {contrasts_file}")

    with open(contrasts_file, 'r') as f:
        data = yaml.safe_load(f)

    if 'contrasts' not in data:
        raise ValueError("Contrasts YAML must have 'contrasts' key")

    return data['contrasts']


def validate_prepared_data(data_dir: Path) -> Dict:
    """
    Validate that TBSS data preparation completed successfully

    Args:
        data_dir: Directory containing prepared TBSS data

    Returns:
        Dictionary with paths to required files

    Raises:
        FileNotFoundError: If required files are missing
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Check for required files
    manifest_file = data_dir / "subject_manifest.json"
    subject_list_file = data_dir / "subject_list.txt"

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Subject manifest not found: {manifest_file}\n"
            "Did you run prepare_tbss.py?"
        )

    if not subject_list_file.exists():
        raise FileNotFoundError(
            f"Subject list not found: {subject_list_file}\n"
            "Did you run prepare_tbss.py?"
        )

    # Load manifest to get metric
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    metric = manifest.get('metric', 'FA')

    # Look for skeleton data
    skeleton_dir = data_dir / metric
    if not skeleton_dir.exists():
        skeleton_dir = data_dir / "FA"  # Fallback

    if not skeleton_dir.exists():
        raise FileNotFoundError(
            f"TBSS skeleton directory not found: {skeleton_dir}\n"
            "Did you run prepare_tbss.py?"
        )

    skeleton_file = skeleton_dir / f"all_{metric}_skeletonised.nii.gz"
    if not skeleton_file.exists():
        # Try generic name
        skeleton_file = skeleton_dir / "all_FA_skeletonised.nii.gz"

    if not skeleton_file.exists():
        raise FileNotFoundError(
            f"Skeletonised data not found in {skeleton_dir}\n"
            "Expected: all_{metric}_skeletonised.nii.gz"
        )

    # Look for skeleton mask
    mask_file = skeleton_dir / "mean_FA_skeleton_mask.nii.gz"
    if not mask_file.exists():
        logging.warning(f"Skeleton mask not found: {mask_file}")
        mask_file = None

    return {
        'data_dir': data_dir,
        'manifest_file': manifest_file,
        'subject_list_file': subject_list_file,
        'skeleton_file': skeleton_file,
        'mask_file': mask_file,
        'metric': metric,
        'n_subjects': manifest['subjects_included']
    }


def run_tbss_statistical_analysis(
    data_dir: Path,
    participants_file: Path,
    formula: str,
    contrasts: List[Dict],
    output_dir: Path,
    method: str = 'randomise',
    n_permutations: int = 5000,
    tfce: bool = True,
    cluster_threshold: float = 0.95,
    min_cluster_size: int = 10,
    z_threshold: float = 2.3,
    seed: Optional[int] = None
) -> Dict:
    """
    Main workflow: Run statistical analysis on prepared TBSS data

    Args:
        data_dir: Directory containing prepared TBSS data
        participants_file: Path to participants CSV
        formula: Model formula (e.g., "age + sex + exposure")
        contrasts: List of contrast specifications
        output_dir: Output directory for analysis results
        method: Statistical method ('randomise', 'glm', or 'both')
        n_permutations: Number of permutations for randomise
        tfce: Use Threshold-Free Cluster Enhancement (randomise only)
        cluster_threshold: Threshold for cluster extraction (0.95 = p<0.05 for randomise)
        min_cluster_size: Minimum cluster size in voxels
        z_threshold: Z-score threshold for GLM (default: 2.3 ≈ p<0.01)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with analysis results and output paths
    """
    # Validate method
    valid_methods = ['randomise', 'glm', 'both']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("TBSS STATISTICAL ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model formula: {formula}")
    logger.info(f"Number of contrasts: {len(contrasts)}")
    logger.info(f"Statistical method: {method}")

    # Step 1: Validate prepared data
    logger.info("\n" + "=" * 80)
    logger.info("[Step 1] Validating prepared TBSS data")
    logger.info("=" * 80)

    prepared_files = validate_prepared_data(data_dir)
    logger.info(f"✓ Found prepared data for {prepared_files['metric']}")
    logger.info(f"✓ Subjects included: {prepared_files['n_subjects']}")
    logger.info(f"✓ Skeleton file: {prepared_files['skeleton_file']}")

    # Step 2: Generate design matrix and contrasts
    logger.info("\n" + "=" * 80)
    logger.info("[Step 2] Generating design matrix and contrasts")
    logger.info("=" * 80)

    design_result = generate_design_files(
        participants_file=participants_file,
        formula=formula,
        contrasts=contrasts,
        output_dir=output_dir,
        subject_list_file=prepared_files['subject_list_file'],
        demean_continuous=True,
        add_intercept=True
    )

    logger.info(f"✓ Design matrix created: {design_result['n_subjects']} subjects, "
                f"{design_result['n_predictors']} predictors")

    # Check subject count matches
    if design_result['n_subjects'] != prepared_files['n_subjects']:
        logger.warning(
            f"Subject count mismatch: design={design_result['n_subjects']}, "
            f"prepared={prepared_files['n_subjects']}"
        )

    # Initialize results
    randomise_result = None
    glm_result = None
    cluster_results = None
    contrast_names = [c['name'] for c in contrasts]

    # Step 3: Run statistical analysis (randomise and/or GLM)
    if method in ['randomise', 'both']:
        logger.info("\n" + "=" * 80)
        logger.info("[Step 3a] Running FSL Randomise (Nonparametric)")
        logger.info("=" * 80)

        randomise_dir = output_dir / "randomise_output"

        randomise_result = run_randomise(
            input_file=prepared_files['skeleton_file'],
            design_mat=Path(design_result['design_mat_file']),
            contrast_con=Path(design_result['contrast_con_file']),
            output_dir=randomise_dir,
            mask=prepared_files['mask_file'],
            n_permutations=n_permutations,
            tfce=tfce,
            seed=seed
        )

        logger.info(f"✓ Randomise completed in {randomise_result['elapsed_time']:.1f} seconds")

        # Summarize randomise results
        logger.info("\n" + "=" * 80)
        logger.info("[Step 4a] Summarizing Randomise results")
        logger.info("=" * 80)

        randomise_summary = summarize_results(randomise_dir, threshold=cluster_threshold)

        # Extract clusters and generate reports
        logger.info("\n" + "=" * 80)
        logger.info("[Step 5a] Extracting clusters and generating Randomise reports")
        logger.info("=" * 80)

        cluster_reports_dir = output_dir / "cluster_reports_randomise"

        cluster_results = generate_reports_for_all_contrasts(
            randomise_output_dir=randomise_dir,
            output_dir=cluster_reports_dir,
            contrast_names=contrast_names,
            threshold=cluster_threshold,
            min_cluster_size=min_cluster_size
        )

    if method in ['glm', 'both']:
        step_num = 'Step 3b' if method == 'both' else 'Step 3'
        logger.info("\n" + "=" * 80)
        logger.info(f"[{step_num}] Running FSL GLM (Parametric)")
        logger.info("=" * 80)

        glm_dir = output_dir / "glm_output"

        glm_result = run_fsl_glm(
            input_file=prepared_files['skeleton_file'],
            design_mat=Path(design_result['design_mat_file']),
            contrast_con=Path(design_result['contrast_con_file']),
            output_dir=glm_dir,
            mask=prepared_files['mask_file']
        )

        logger.info(f"✓ FSL GLM completed in {glm_result['elapsed_time']:.1f} seconds")

        # Threshold and summarize GLM results
        step_num = 'Step 4b' if method == 'both' else 'Step 4'
        logger.info("\n" + "=" * 80)
        logger.info(f"[{step_num}] Thresholding and summarizing GLM results")
        logger.info("=" * 80)

        threshold_dir = glm_dir / "thresholded"
        threshold_result = threshold_zstat(
            zstat_file=Path(glm_result['output_files']['zstat']),
            output_dir=threshold_dir,
            z_threshold=z_threshold,
            cluster_threshold=min_cluster_size,
            mask=prepared_files['mask_file']
        )

        glm_summary = summarize_glm_results(
            output_dir=glm_dir,
            contrast_names=contrast_names,
            z_threshold=z_threshold
        )

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  Design files: {output_dir}")

    if randomise_result:
        logger.info(f"  Randomise results: {output_dir / 'randomise_output'}")
        logger.info(f"  Cluster reports (Randomise): {output_dir / 'cluster_reports_randomise'}")

    if glm_result:
        logger.info(f"  GLM results: {output_dir / 'glm_output'}")
        logger.info(f"  GLM thresholded: {output_dir / 'glm_output' / 'thresholded'}")

    logger.info("\nSignificant findings:")

    if randomise_result and cluster_results:
        n_sig = sum(1 for r in cluster_results['reports'] if r['significant'])
        logger.info(f"\n  Randomise (nonparametric):")
        logger.info(f"    {n_sig}/{len(contrasts)} contrasts showed significant clusters")

        for report in cluster_results['reports']:
            if report['significant']:
                logger.info(f"      ✓ {report['contrast_name']}: {report['n_clusters']} clusters, "
                           f"{report['total_voxels']} voxels")
            else:
                logger.info(f"      ✗ {report['contrast_name']}: No significant clusters")

    if glm_result:
        logger.info(f"\n  GLM (parametric, z > {z_threshold}):")
        n_sig_glm = sum(1 for c in glm_summary['contrasts'] if c['significant'])
        logger.info(f"    {n_sig_glm}/{len(contrasts)} contrasts showed significant voxels")

        for contrast in glm_summary['contrasts']:
            if contrast['significant']:
                logger.info(f"      ✓ {contrast['name']}: {contrast['n_positive_voxels']} pos, "
                           f"{contrast['n_negative_voxels']} neg voxels")
            else:
                logger.info(f"      ✗ {contrast['name']}: No significant voxels")

    logger.info("=" * 80)

    result_dict = {
        'success': True,
        'output_dir': str(output_dir),
        'design_result': design_result,
        'method': method
    }

    if randomise_result:
        result_dict['randomise_result'] = randomise_result
        result_dict['cluster_results'] = cluster_results
        result_dict['n_significant_contrasts_randomise'] = sum(
            1 for r in cluster_results['reports'] if r['significant']
        )

    if glm_result:
        result_dict['glm_result'] = glm_result
        result_dict['glm_summary'] = glm_summary
        result_dict['n_significant_contrasts_glm'] = sum(
            1 for c in glm_summary['contrasts'] if c['significant']
        )

    return result_dict


def main():
    """Command-line interface for TBSS statistical analysis"""
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on prepared TBSS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Randomise analysis (nonparametric, default)
  python -m neurovrai.analysis.tbss.run_tbss_stats \\
      --data-dir /study/analysis/tbss_FA/ \\
      --participants participants.csv \\
      --formula "age + sex" \\
      --contrast age_positive 0 1 0 \\
      --contrast sex_MvsF 0 0 1 \\
      --output-dir /study/analysis/tbss_FA/model1/ \\
      --method randomise

  # GLM analysis (parametric, faster)
  python -m neurovrai.analysis.tbss.run_tbss_stats \\
      --data-dir /study/analysis/tbss_FA/ \\
      --participants participants.csv \\
      --formula "age + sex + exposure" \\
      --contrasts-file contrasts.yaml \\
      --output-dir /study/analysis/tbss_FA/model_glm/ \\
      --method glm \\
      --z-threshold 2.3

  # Run both methods for comparison
  python -m neurovrai.analysis.tbss.run_tbss_stats \\
      --data-dir /study/analysis/tbss_FA/ \\
      --participants participants.csv \\
      --formula "age + sex + exposure + age*sex" \\
      --contrasts-file contrasts.yaml \\
      --output-dir /study/analysis/tbss_FA/model_both/ \\
      --method both
        """
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing prepared TBSS data'
    )

    parser.add_argument(
        '--participants',
        type=Path,
        required=True,
        help='Path to participants CSV file'
    )

    parser.add_argument(
        '--formula',
        type=str,
        required=True,
        help='Model formula (e.g., "age + sex + exposure")'
    )

    # Contrast specification (mutually exclusive)
    contrast_group = parser.add_mutually_exclusive_group(required=True)

    contrast_group.add_argument(
        '--contrasts-file',
        type=Path,
        help='Path to contrasts YAML file'
    )

    contrast_group.add_argument(
        '--contrast',
        action='append',
        nargs='+',
        metavar=('NAME', 'WEIGHT'),
        help='Inline contrast: name followed by weights (can specify multiple)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for analysis results'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['randomise', 'glm', 'both'],
        default='randomise',
        help='Statistical method: randomise (nonparametric), glm (parametric), or both (default: randomise)'
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
        help='Disable TFCE (use cluster-based thresholding)'
    )

    parser.add_argument(
        '--cluster-threshold',
        type=float,
        default=0.95,
        help='Threshold for cluster extraction (default: 0.95 = p<0.05)'
    )

    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=10,
        help='Minimum cluster size in voxels (default: 10)'
    )

    parser.add_argument(
        '--z-threshold',
        type=float,
        default=2.3,
        help='Z-score threshold for GLM (default: 2.3 ≈ p<0.01)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Parse contrasts
    if args.contrasts_file:
        contrasts = load_contrasts(args.contrasts_file)
    else:
        # Parse inline contrasts
        contrasts = []
        for contrast in args.contrast:
            name = contrast[0]
            weights = [float(w) for w in contrast[1:]]
            contrasts.append({
                'name': name,
                'vector': weights
            })

    # Run analysis
    result = run_tbss_statistical_analysis(
        data_dir=args.data_dir,
        participants_file=args.participants,
        formula=args.formula,
        contrasts=contrasts,
        output_dir=args.output_dir,
        method=args.method,
        n_permutations=args.n_permutations,
        tfce=not args.no_tfce,
        cluster_threshold=args.cluster_threshold,
        min_cluster_size=args.min_cluster_size,
        z_threshold=args.z_threshold,
        seed=args.seed
    )

    exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
