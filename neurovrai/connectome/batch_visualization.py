#!/usr/bin/env python3
"""
Batch Connectivity Visualization

Generate publication-quality visualizations for connectivity matrices.

Outputs are saved to:
    {study-root}/connectome/visualizations/{atlas}/
        ├── individual/
        │   ├── {subject}_matrix_heatmap.png
        │   └── {subject}_connectogram.png
        ├── group/
        │   ├── group_mean_heatmap.png
        │   ├── group_mean_connectogram.png
        │   └── comparison_plots.png
        └── logs/

Usage:
    # Visualize individual subject matrices
    uv run python -m neurovrai.connectome.batch_visualization \
        --study-root /mnt/bytopia/IRC805 \
        --atlas schaefer200 \
        --mode individual

    # Visualize group averages
    uv run python -m neurovrai.connectome.batch_visualization \
        --study-root /mnt/bytopia/IRC805 \
        --atlas schaefer200 \
        --mode group

    # Generate all visualizations
    uv run python -m neurovrai.connectome.batch_visualization \
        --study-root /mnt/bytopia/IRC805 \
        --atlas schaefer200 \
        --mode all
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from neurovrai.connectome.visualization import (
    plot_connectivity_matrix,
    plot_circular_connectogram,
    plot_connectivity_comparison
)


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_visualization_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")

    return logger


def load_roi_names(roi_names_file: Optional[Path]) -> Optional[List[str]]:
    """Load ROI names from file"""
    if roi_names_file is None or not roi_names_file.exists():
        return None

    with open(roi_names_file, 'r') as f:
        return [line.strip() for line in f]


def visualize_individual_matrix(
    subject: str,
    matrix_file: Path,
    roi_names_file: Optional[Path],
    output_dir: Path,
    threshold: Optional[float],
    dpi: int,
    logger: logging.Logger
) -> Dict:
    """Generate visualizations for one subject's connectivity matrix"""
    try:
        logger.info(f"Processing {subject}...")

        # Load matrix and ROI names
        matrix = np.load(matrix_file)
        roi_names = load_roi_names(roi_names_file)

        # Create heatmap
        heatmap_file = output_dir / f"{subject}_matrix_heatmap.png"
        logger.info(f"  Generating heatmap...")
        plot_connectivity_matrix(
            matrix=matrix,
            labels=roi_names,
            output_file=heatmap_file,
            title=f"Connectivity Matrix - {subject}",
            group_by_region=True,
            atlas_name=atlas,
            show_labels=(matrix.shape[0] <= 50),
            dpi=dpi
        )
        plt.close('all')

        # Create connectogram
        connectogram_file = output_dir / f"{subject}_connectogram.png"
        logger.info(f"  Generating connectogram...")
        plot_circular_connectogram(
            matrix=matrix,
            labels=roi_names,
            atlas_name=atlas,
            output_file=connectogram_file,
            title=f"Connectogram - {subject}",
            threshold=threshold,
            dpi=dpi
        )
        plt.close('all')

        logger.info(f"  ✓ Saved to {output_dir}")

        return {
            'subject': subject,
            'status': 'success',
            'heatmap': str(heatmap_file),
            'connectogram': str(connectogram_file)
        }

    except Exception as e:
        logger.error(f"  ✗ Failed: {str(e)}", exc_info=True)
        return {
            'subject': subject,
            'status': 'failed',
            'error': str(e)
        }


def visualize_group_matrices(
    atlas: str,
    group_stats_dir: Path,
    output_dir: Path,
    dpi: int,
    logger: logging.Logger
) -> Dict:
    """Generate visualizations for group average matrices"""
    try:
        logger.info(f"Processing group averages for {atlas}...")

        # Load group mean matrix
        mean_matrix_file = group_stats_dir / "group_average" / "group_mean_matrix.npy"
        if not mean_matrix_file.exists():
            logger.warning("  Group mean matrix not found, skipping")
            return {'status': 'skipped', 'reason': 'No group mean matrix'}

        mean_matrix = np.load(mean_matrix_file)

        # Create heatmap
        heatmap_file = output_dir / "group_mean_heatmap.png"
        logger.info(f"  Generating group mean heatmap...")
        plot_connectivity_matrix(
            matrix=mean_matrix,
            labels=None,
            atlas_name=atlas,
            output_file=heatmap_file,
            title=f"Group Mean Connectivity - {atlas}",
            group_by_region=True,
            show_labels=(mean_matrix.shape[0] <= 50),
            dpi=dpi
        )
        plt.close('all')

        # Create connectogram
        connectogram_file = output_dir / "group_mean_connectogram.png"
        logger.info(f"  Generating group mean connectogram...")

        # Calculate appropriate threshold (e.g., 75th percentile)
        upper_triangle = mean_matrix[np.triu_indices_from(mean_matrix, k=1)]
        threshold = np.percentile(np.abs(upper_triangle), 75)

        plot_circular_connectogram(
            matrix=mean_matrix,
            labels=None,
            atlas_name=atlas,
            output_file=connectogram_file,
            title=f"Group Mean Connectogram - {atlas}",
            threshold=threshold,
            dpi=dpi
        )
        plt.close('all')

        logger.info(f"  ✓ Saved to {output_dir}")

        return {
            'status': 'success',
            'heatmap': str(heatmap_file),
            'connectogram': str(connectogram_file)
        }

    except Exception as e:
        logger.error(f"  ✗ Failed: {str(e)}", exc_info=True)
        return {
            'status': 'failed',
            'error': str(e)
        }


def visualize_group_comparison(
    atlas: str,
    group_stats_dir: Path,
    output_dir: Path,
    dpi: int,
    logger: logging.Logger
) -> Dict:
    """Generate comparison visualizations for two groups"""
    try:
        logger.info(f"Processing group comparison for {atlas}...")

        comparison_dir = group_stats_dir / "group_comparison"

        # Load comparison matrices
        group1_file = comparison_dir / "comparison_group1_mean.npy"
        group2_file = comparison_dir / "comparison_group2_mean.npy"

        if not group1_file.exists() or not group2_file.exists():
            logger.warning("  Group comparison matrices not found, skipping")
            return {'status': 'skipped', 'reason': 'No comparison matrices'}

        group1_matrix = np.load(group1_file)
        group2_matrix = np.load(group2_file)

        # Create comparison plot
        comparison_file = output_dir / "group_comparison_plot.png"
        logger.info(f"  Generating comparison plot...")

        plot_connectivity_comparison(
            matrix1=group1_matrix,
            matrix2=group2_matrix,
            labels=None,
            atlas_name=atlas,
            group_by_region=True,
            output_file=comparison_file,
            title1="Group 1",
            title2="Group 2",
            dpi=dpi
        )
        plt.close('all')

        # Plot significant edges if available
        sig_edges_file = comparison_dir / "comparison_significant_edges.npy"
        if sig_edges_file.exists():
            sig_edges = np.load(sig_edges_file)

            sig_plot_file = output_dir / "significant_edges_heatmap.png"
            logger.info(f"  Generating significant edges plot...")

            plot_connectivity_matrix(
                matrix=sig_edges.astype(float),
                labels=None,
                atlas_name=atlas,
                output_file=sig_plot_file,
                title=f"Significant Edges - {atlas}",
                cmap='binary',
                group_by_region=True,
                show_labels=False,
                dpi=dpi
            )
            plt.close('all')

        logger.info(f"  ✓ Saved to {output_dir}")

        return {
            'status': 'success',
            'comparison_plot': str(comparison_file)
        }

    except Exception as e:
        logger.error(f"  ✗ Failed: {str(e)}", exc_info=True)
        return {
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Batch visualization for connectivity matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Path to study root directory'
    )

    parser.add_argument(
        '--atlas',
        type=str,
        required=True,
        help='Atlas name (e.g., schaefer200, aal2)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['individual', 'group', 'comparison', 'all'],
        help='Visualization mode (default: all)'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to visualize (for individual mode)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        help='Threshold for connectogram visualization'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup directories
    connectome_dir = args.study_root / "connectome"
    output_base = connectome_dir / "visualizations" / args.atlas
    output_base.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_base, verbose=args.verbose)

    logger.info("="*80)
    logger.info("BATCH CONNECTIVITY VISUALIZATION")
    logger.info("="*80)
    logger.info(f"Study root: {args.study_root}")
    logger.info(f"Atlas: {args.atlas}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"DPI: {args.dpi}")

    results = {'individual': [], 'group': None, 'comparison': None}
    successful = 0
    failed = 0

    # Individual subject visualizations
    if args.mode in ['individual', 'all']:
        logger.info("\n" + "="*80)
        logger.info("INDIVIDUAL SUBJECT VISUALIZATIONS")
        logger.info("="*80)

        individual_output_dir = output_base / "individual"
        individual_output_dir.mkdir(parents=True, exist_ok=True)

        functional_dir = connectome_dir / "functional"

        for subject_dir in sorted(functional_dir.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name == 'logs':
                continue

            subject = subject_dir.name

            # Filter by subject list if provided
            if args.subjects and subject not in args.subjects:
                continue

            atlas_dir = subject_dir / args.atlas
            matrix_file = atlas_dir / "fc_matrix.npy"
            roi_names_file = atlas_dir / "fc_roi_names.txt"

            if not matrix_file.exists():
                continue

            result = visualize_individual_matrix(
                subject=subject,
                matrix_file=matrix_file,
                roi_names_file=roi_names_file,
                output_dir=individual_output_dir,
                threshold=args.threshold,
                dpi=args.dpi,
                logger=logger
            )

            results['individual'].append(result)

            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1

        logger.info(f"\nIndividual visualizations: {successful} successful, {failed} failed")

    # Group average visualizations
    if args.mode in ['group', 'all']:
        logger.info("\n" + "="*80)
        logger.info("GROUP AVERAGE VISUALIZATIONS")
        logger.info("="*80)

        group_output_dir = output_base / "group"
        group_output_dir.mkdir(parents=True, exist_ok=True)

        group_stats_dir = connectome_dir / "group_statistics" / args.atlas

        if group_stats_dir.exists():
            result = visualize_group_matrices(
                atlas=args.atlas,
                group_stats_dir=group_stats_dir,
                output_dir=group_output_dir,
                dpi=args.dpi,
                logger=logger
            )
            results['group'] = result
        else:
            logger.warning("No group statistics found. Run group averaging first.")

    # Group comparison visualizations
    if args.mode in ['comparison', 'all']:
        logger.info("\n" + "="*80)
        logger.info("GROUP COMPARISON VISUALIZATIONS")
        logger.info("="*80)

        comparison_output_dir = output_base / "comparison"
        comparison_output_dir.mkdir(parents=True, exist_ok=True)

        group_stats_dir = connectome_dir / "group_statistics" / args.atlas

        if group_stats_dir.exists():
            result = visualize_group_comparison(
                atlas=args.atlas,
                group_stats_dir=group_stats_dir,
                output_dir=comparison_output_dir,
                dpi=args.dpi,
                logger=logger
            )
            results['comparison'] = result
        else:
            logger.warning("No group statistics found. Run group comparison first.")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_base}")

    if results['individual']:
        logger.info(f"Individual subjects: {len(results['individual'])} processed")

    if results['group']:
        logger.info(f"Group averages: {results['group']['status']}")

    if results['comparison']:
        logger.info(f"Group comparison: {results['comparison']['status']}")

    logger.info("="*80)

    sys.exit(0)


if __name__ == '__main__':
    main()
