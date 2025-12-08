#!/usr/bin/env python3
"""
Batch Graph Metrics Analysis

Compute graph-theoretic network metrics for connectivity matrices across subjects and atlases.

Outputs are saved to:
    {study-root}/connectome/graph_metrics/{atlas}/
        ├── {subject}/
        │   ├── node_metrics.csv           # Node-level metrics (degree, clustering, etc.)
        │   ├── global_metrics.json        # Global network metrics
        │   └── hub_nodes.csv              # Identified hub nodes
        ├── group_node_metrics.csv         # Averaged across subjects
        ├── group_global_metrics.json      # Group-averaged global metrics
        └── logs/

Usage:
    # Process all subjects and atlases
    uv run python -m neurovrai.connectome.batch_graph_metrics \
        --study-root /mnt/bytopia/IRC805

    # Specific atlas only
    uv run python -m neurovrai.connectome.batch_graph_metrics \
        --study-root /mnt/bytopia/IRC805 \
        --atlases schaefer200 aal2

    # Custom threshold
    uv run python -m neurovrai.connectome.batch_graph_metrics \
        --study-root /mnt/bytopia/IRC805 \
        --threshold 0.3
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from neurovrai.connectome.graph_metrics import (
    compute_node_metrics,
    compute_global_metrics,
    identify_hubs
)


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_graph_metrics_{timestamp}.log"

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


def find_connectivity_matrices(
    connectome_dir: Path,
    atlas: Optional[str] = None
) -> Dict[str, Dict[str, Path]]:
    """
    Find all connectivity matrices organized by atlas and subject

    Returns:
        Dictionary: {atlas_name: {subject_id: matrix_path}}
    """
    matrices = {}

    # Find all subject directories
    functional_dir = connectome_dir / "functional"

    if not functional_dir.exists():
        return matrices

    for subject_dir in sorted(functional_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        # Skip non-subject directories
        if subject in ['logs']:
            continue

        # Find atlas directories
        for atlas_dir in subject_dir.iterdir():
            if not atlas_dir.is_dir():
                continue

            atlas_name = atlas_dir.name

            # Filter by atlas if specified
            if atlas is not None and atlas_name != atlas:
                continue

            # Look for connectivity matrix
            matrix_file = atlas_dir / "fc_matrix.npy"
            roi_names_file = atlas_dir / "fc_roi_names.txt"

            if matrix_file.exists():
                if atlas_name not in matrices:
                    matrices[atlas_name] = {}

                matrices[atlas_name][subject] = {
                    'matrix': matrix_file,
                    'roi_names': roi_names_file if roi_names_file.exists() else None
                }

    return matrices


def load_roi_names(roi_names_file: Optional[Path]) -> Optional[List[str]]:
    """Load ROI names from file"""
    if roi_names_file is None or not roi_names_file.exists():
        return None

    with open(roi_names_file, 'r') as f:
        return [line.strip() for line in f]


def process_subject_atlas(
    subject: str,
    atlas: str,
    matrix_file: Path,
    roi_names_file: Optional[Path],
    output_dir: Path,
    threshold: Optional[float] = None,
    weighted: bool = False,
    hub_percentile: int = 90,
    logger: logging.Logger = None
) -> Dict:
    """
    Compute graph metrics for one subject-atlas combination

    Returns:
        Dictionary with results and status
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"\n{'='*80}")
    logger.info(f"Subject: {subject} | Atlas: {atlas}")
    logger.info(f"{'='*80}")

    try:
        # Create output directory
        subject_output_dir = output_dir / atlas / subject
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        # Load connectivity matrix
        logger.info(f"Loading connectivity matrix: {matrix_file.name}")
        matrix = np.load(matrix_file)
        logger.info(f"  Matrix shape: {matrix.shape}")

        # Load ROI names
        roi_names = load_roi_names(roi_names_file)
        if roi_names:
            logger.info(f"  ROI names: {len(roi_names)}")

        # Compute node metrics
        logger.info("\nComputing node-level metrics...")
        node_metrics = compute_node_metrics(
            matrix=matrix,
            threshold=threshold,
            weighted=weighted,
            roi_names=roi_names
        )

        # Save node metrics to CSV
        node_df = pd.DataFrame({
            'roi': node_metrics['roi_names'],
            'degree': node_metrics['degree'],
            'strength': node_metrics['strength'],
            'clustering_coefficient': node_metrics['clustering_coefficient'],
            'betweenness_centrality': node_metrics['betweenness_centrality']
        })

        node_csv = subject_output_dir / "node_metrics.csv"
        node_df.to_csv(node_csv, index=False)
        logger.info(f"  ✓ Saved node metrics: {node_csv.name}")

        # Identify hub nodes
        logger.info("\nIdentifying hub nodes...")
        hubs_degree = identify_hubs(node_metrics, method='degree', percentile=hub_percentile)
        hubs_betweenness = identify_hubs(node_metrics, method='betweenness', percentile=hub_percentile)

        # Save hub information
        hub_df = pd.DataFrame({
            'roi': node_metrics['roi_names'],
            'degree': node_metrics['degree'],
            'betweenness_centrality': node_metrics['betweenness_centrality'],
            'hub_degree': hubs_degree,
            'hub_betweenness': hubs_betweenness
        })

        hub_csv = subject_output_dir / "hub_nodes.csv"
        hub_df.to_csv(hub_csv, index=False)
        logger.info(f"  ✓ Saved hub nodes: {hub_csv.name}")

        # Compute global metrics
        logger.info("\nComputing global network metrics...")
        global_metrics = compute_global_metrics(
            matrix=matrix,
            threshold=threshold
        )

        # Add summary statistics
        global_metrics.update({
            'n_nodes': matrix.shape[0],
            'n_hubs_degree': int(np.sum(hubs_degree)),
            'n_hubs_betweenness': int(np.sum(hubs_betweenness)),
            'mean_degree': float(np.mean(node_metrics['degree'])),
            'mean_strength': float(np.mean(node_metrics['strength'])),
            'mean_clustering': float(np.mean(node_metrics['clustering_coefficient'])),
            'mean_betweenness': float(np.mean(node_metrics['betweenness_centrality']))
        })

        # Save global metrics
        global_json = subject_output_dir / "global_metrics.json"
        with open(global_json, 'w') as f:
            json.dump(global_metrics, f, indent=2)
        logger.info(f"  ✓ Saved global metrics: {global_json.name}")

        logger.info(f"\n✓ Success! Results saved to: {subject_output_dir}")

        return {
            'subject': subject,
            'atlas': atlas,
            'status': 'success',
            'output_dir': str(subject_output_dir),
            'n_nodes': matrix.shape[0],
            'global_efficiency': global_metrics['global_efficiency'],
            'characteristic_path_length': global_metrics['characteristic_path_length']
        }

    except Exception as e:
        logger.error(f"✗ Failed: {str(e)}", exc_info=True)
        return {
            'subject': subject,
            'atlas': atlas,
            'status': 'failed',
            'error': str(e)
        }


def compute_group_averages(
    output_dir: Path,
    atlas: str,
    logger: logging.Logger
) -> None:
    """Compute group-averaged metrics across subjects"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Computing group averages for atlas: {atlas}")
    logger.info(f"{'='*80}")

    atlas_dir = output_dir / atlas

    # Collect all subject node metrics
    all_node_metrics = []
    all_global_metrics = []

    for subject_dir in sorted(atlas_dir.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name == 'logs':
            continue

        node_csv = subject_dir / "node_metrics.csv"
        global_json = subject_dir / "global_metrics.json"

        if node_csv.exists():
            df = pd.read_csv(node_csv)
            df['subject'] = subject_dir.name
            all_node_metrics.append(df)

        if global_json.exists():
            with open(global_json, 'r') as f:
                metrics = json.load(f)
                metrics['subject'] = subject_dir.name
                all_global_metrics.append(metrics)

    if not all_node_metrics:
        logger.warning(f"  No node metrics found for {atlas}")
        return

    # Average node metrics across subjects
    logger.info(f"  Averaging {len(all_node_metrics)} subjects")

    combined_nodes = pd.concat(all_node_metrics, ignore_index=True)
    group_nodes = combined_nodes.groupby('roi').agg({
        'degree': ['mean', 'std'],
        'strength': ['mean', 'std'],
        'clustering_coefficient': ['mean', 'std'],
        'betweenness_centrality': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    group_nodes.columns = ['_'.join(col).strip('_') for col in group_nodes.columns.values]

    # Save group node metrics
    group_node_csv = atlas_dir / "group_node_metrics.csv"
    group_nodes.to_csv(group_node_csv, index=False)
    logger.info(f"  ✓ Saved group node metrics: {group_node_csv.name}")

    # Average global metrics
    if all_global_metrics:
        global_df = pd.DataFrame(all_global_metrics)

        group_global = {
            'n_subjects': len(all_global_metrics),
            'global_efficiency_mean': float(global_df['global_efficiency'].mean()),
            'global_efficiency_std': float(global_df['global_efficiency'].std()),
            'characteristic_path_length_mean': float(global_df['characteristic_path_length'].mean()),
            'characteristic_path_length_std': float(global_df['characteristic_path_length'].std()),
            'transitivity_mean': float(global_df['transitivity'].mean()),
            'transitivity_std': float(global_df['transitivity'].std()),
            'mean_degree_mean': float(global_df['mean_degree'].mean()),
            'mean_degree_std': float(global_df['mean_degree'].std()),
            'n_hubs_degree_mean': float(global_df['n_hubs_degree'].mean()),
            'n_hubs_degree_std': float(global_df['n_hubs_degree'].std())
        }

        group_global_json = atlas_dir / "group_global_metrics.json"
        with open(group_global_json, 'w') as f:
            json.dump(group_global, f, indent=2)
        logger.info(f"  ✓ Saved group global metrics: {group_global_json.name}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Batch graph metrics analysis for connectivity matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Path to study root directory'
    )

    parser.add_argument(
        '--atlases',
        nargs='+',
        help='Specific atlases to process (default: all found)'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to process (default: all found)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        help='Threshold for connectivity matrix (default: None)'
    )

    parser.add_argument(
        '--weighted',
        action='store_true',
        help='Use weighted graph metrics'
    )

    parser.add_argument(
        '--hub-percentile',
        type=int,
        default=90,
        help='Percentile for hub identification (default: 90)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup output directory
    connectome_dir = args.study_root / "connectome"
    output_dir = connectome_dir / "graph_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, verbose=args.verbose)

    logger.info("="*80)
    logger.info("BATCH GRAPH METRICS ANALYSIS")
    logger.info("="*80)
    logger.info(f"Study root: {args.study_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Threshold: {args.threshold if args.threshold else 'None'}")
    logger.info(f"Weighted: {args.weighted}")

    # Find all connectivity matrices
    logger.info("\nSearching for connectivity matrices...")
    matrices_by_atlas = find_connectivity_matrices(connectome_dir)

    if not matrices_by_atlas:
        logger.error("No connectivity matrices found!")
        sys.exit(1)

    # Filter by atlas if specified
    if args.atlases:
        matrices_by_atlas = {
            k: v for k, v in matrices_by_atlas.items()
            if k in args.atlases
        }

    # Count total analyses
    total_analyses = sum(len(subjects) for subjects in matrices_by_atlas.values())
    logger.info(f"Found {len(matrices_by_atlas)} atlases")
    logger.info(f"Total analyses to run: {total_analyses}")

    for atlas_name in sorted(matrices_by_atlas.keys()):
        logger.info(f"  {atlas_name}: {len(matrices_by_atlas[atlas_name])} subjects")

    # Process all subject-atlas combinations
    results = []
    successful = 0
    failed = 0

    for atlas_name in sorted(matrices_by_atlas.keys()):
        subjects_data = matrices_by_atlas[atlas_name]

        for subject in sorted(subjects_data.keys()):
            # Filter by subject if specified
            if args.subjects and subject not in args.subjects:
                continue

            result = process_subject_atlas(
                subject=subject,
                atlas=atlas_name,
                matrix_file=subjects_data[subject]['matrix'],
                roi_names_file=subjects_data[subject]['roi_names'],
                output_dir=output_dir,
                threshold=args.threshold,
                weighted=args.weighted,
                hub_percentile=args.hub_percentile,
                logger=logger
            )

            results.append(result)

            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1

        # Compute group averages for this atlas
        compute_group_averages(output_dir, atlas_name, logger)

    # Save batch summary
    summary_file = output_dir / 'batch_processing_summary.json'
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'study_root': str(args.study_root),
        'total_analyses': total_analyses,
        'successful': successful,
        'failed': failed,
        'threshold': args.threshold,
        'weighted': args.weighted,
        'atlases': list(matrices_by_atlas.keys()),
        'results': results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total analyses: {total_analyses}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/total_analyses*100:.1f}%")
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info(f"Results directory: {output_dir}")
    logger.info("="*80)

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
