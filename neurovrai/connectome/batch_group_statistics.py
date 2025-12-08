#!/usr/bin/env python3
"""
Batch Group Statistics for Connectomes

Perform group-level statistical analysis on functional connectivity matrices.

Analyses:
1. Group averaging with consistency thresholding
2. Two-sample t-tests (e.g., patient vs control)
3. Network-Based Statistic (NBS) with permutation testing

Outputs are saved to:
    {study-root}/connectome/group_statistics/{atlas}/
        ├── group_average/
        │   ├── mean_matrix.npy
        │   ├── std_matrix.npy
        │   ├── se_matrix.npy
        │   ├── summary.json
        │   └── mean_matrix_heatmap.png
        ├── group_comparison/
        │   ├── difference_matrix.npy
        │   ├── t_matrix.npy
        │   ├── p_matrix_fdr.npy
        │   ├── significant_edges.npy
        │   └── summary.json
        └── nbs/
            ├── nbs_t_matrix.npy
            ├── nbs_components.json
            └── nbs_null_distribution.npy

Usage:
    # Group averaging
    uv run python -m neurovrai.connectome.batch_group_statistics \
        --study-root /mnt/bytopia/IRC805 \
        --atlas schaefer200 \
        --analysis group_average

    # Two-sample comparison
    uv run python -m neurovrai.connectome.batch_group_statistics \
        --study-root /mnt/bytopia/IRC805 \
        --atlas schaefer200 \
        --analysis group_comparison \
        --group1 control_subjects.txt \
        --group2 patient_subjects.txt

    # Network-Based Statistic
    uv run python -m neurovrai.connectome.batch_group_statistics \
        --study-root /mnt/bytopia/IRC805 \
        --atlas schaefer200 \
        --analysis nbs \
        --group1 control_subjects.txt \
        --group2 patient_subjects.txt \
        --n-permutations 5000
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from neurovrai.connectome.group_analysis import (
    load_connectivity_matrices,
    average_connectivity_matrices,
    compute_group_difference,
    compute_network_based_statistic
)


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"group_statistics_{timestamp}.log"

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


def load_subject_list(file_path: Optional[Path]) -> Optional[List[str]]:
    """Load list of subject IDs from text file (one per line)"""
    if file_path is None:
        return None

    with open(file_path, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    return subjects


def extract_groups_from_csv(
    csv_file: Path,
    subject_column: str,
    group_column: str,
    group1_value: str,
    group2_value: str,
    logger: logging.Logger
) -> Tuple[List[str], List[str]]:
    """
    Extract two groups of subjects from CSV file based on group column values

    Args:
        csv_file: Path to CSV file
        subject_column: Column name containing subject IDs
        group_column: Column name containing group assignments
        group1_value: Value that identifies Group 1
        group2_value: Value that identifies Group 2
        logger: Logger instance

    Returns:
        Tuple of (group1_subjects, group2_subjects)
    """
    logger.info(f"Loading demographics from: {csv_file}")

    # Read CSV
    df = pd.read_csv(csv_file)

    # Check required columns exist
    if subject_column not in df.columns:
        raise ValueError(f"Subject column '{subject_column}' not found in CSV. Available columns: {list(df.columns)}")

    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in CSV. Available columns: {list(df.columns)}")

    logger.info(f"  Total subjects in CSV: {len(df)}")
    logger.info(f"  Subject column: {subject_column}")
    logger.info(f"  Group column: {group_column}")

    # Convert group column values to strings for comparison
    df[group_column] = df[group_column].astype(str)
    group1_value = str(group1_value)
    group2_value = str(group2_value)

    # Extract groups
    group1_df = df[df[group_column] == group1_value]
    group2_df = df[df[group_column] == group2_value]

    group1_subjects = group1_df[subject_column].astype(str).tolist()
    group2_subjects = group2_df[subject_column].astype(str).tolist()

    # Format subject IDs to match connectivity matrix filenames (add IRC805- prefix if needed)
    def format_subject_id(subj_id):
        subj_id = str(subj_id).strip()

        # Handle integer conversion (preserves leading zeros)
        # If it's stored as int, convert to padded string
        try:
            # Check if it's a number
            if '.' in subj_id:
                # It's a float, convert to int first
                subj_id = str(int(float(subj_id)))
        except:
            pass

        # Add IRC805- prefix if not present
        if not subj_id.startswith('IRC805'):
            # Pad to 7 digits if it's all numeric
            if subj_id.replace('-', '').isdigit():
                numeric_part = subj_id.replace('-', '')
                subj_id = f'IRC805-{numeric_part.zfill(7)}'
            else:
                subj_id = f'IRC805-{subj_id}'

        return subj_id

    group1_subjects = [format_subject_id(s) for s in group1_subjects]
    group2_subjects = [format_subject_id(s) for s in group2_subjects]

    logger.info(f"\n  Group 1 ({group_column}={group1_value}): {len(group1_subjects)} subjects")
    logger.info(f"  Group 2 ({group_column}={group2_value}): {len(group2_subjects)} subjects")

    # Show unique group values found
    unique_groups = df[group_column].dropna().unique()
    logger.info(f"  Unique values in {group_column}: {sorted(unique_groups)}")

    return group1_subjects, group2_subjects


def find_connectivity_matrices(
    connectome_dir: Path,
    atlas: str,
    subjects: Optional[List[str]] = None
) -> Tuple[List[Path], List[str]]:
    """
    Find connectivity matrices for specified atlas and subjects

    Returns:
        Tuple of (matrix_files, subject_ids)
    """
    functional_dir = connectome_dir / "functional"
    matrix_files = []
    subject_ids = []

    for subject_dir in sorted(functional_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        # Skip non-subject directories
        if subject in ['logs']:
            continue

        # Filter by subject list if provided
        if subjects is not None and subject not in subjects:
            continue

        # Check for atlas directory
        atlas_dir = subject_dir / atlas
        matrix_file = atlas_dir / "fc_matrix.npy"

        if matrix_file.exists():
            matrix_files.append(matrix_file)
            subject_ids.append(subject)

    return matrix_files, subject_ids


def run_group_average(
    study_root: Path,
    atlas: str,
    subjects: Optional[List[str]],
    consistency_threshold: Optional[float],
    logger: logging.Logger
) -> Dict:
    """Run group averaging analysis"""
    logger.info("="*80)
    logger.info("GROUP AVERAGE ANALYSIS")
    logger.info("="*80)

    connectome_dir = study_root / "connectome"
    output_dir = connectome_dir / "group_statistics" / atlas / "group_average"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matrices
    logger.info(f"Finding connectivity matrices for atlas: {atlas}")
    matrix_files, subject_ids = find_connectivity_matrices(
        connectome_dir, atlas, subjects
    )

    logger.info(f"Found {len(matrix_files)} subjects")

    if len(matrix_files) == 0:
        logger.error("No connectivity matrices found!")
        return {'status': 'failed', 'error': 'No matrices found'}

    # Load matrices
    matrices, loaded_subjects = load_connectivity_matrices(
        matrix_files, subject_ids
    )

    # Compute group average
    logger.info("\nComputing group average...")
    results = average_connectivity_matrices(
        matrices=matrices,
        subject_ids=loaded_subjects,
        consistency_threshold=consistency_threshold,
        output_dir=output_dir,
        output_prefix='group'
    )

    logger.info(f"\n✓ Results saved to: {output_dir}")

    return {
        'status': 'success',
        'output_dir': str(output_dir),
        'n_subjects': len(loaded_subjects),
        'subjects': loaded_subjects
    }


def run_group_comparison(
    study_root: Path,
    atlas: str,
    group1_subjects: List[str],
    group2_subjects: List[str],
    group1_name: str,
    group2_name: str,
    paired: bool,
    alpha: float,
    logger: logging.Logger
) -> Dict:
    """Run two-sample comparison analysis"""
    logger.info("="*80)
    logger.info("GROUP COMPARISON ANALYSIS")
    logger.info("="*80)

    connectome_dir = study_root / "connectome"
    output_dir = connectome_dir / "group_statistics" / atlas / "group_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matrices for group 1
    logger.info(f"\nFinding matrices for {group1_name}...")
    group1_files, group1_ids = find_connectivity_matrices(
        connectome_dir, atlas, group1_subjects
    )
    logger.info(f"  Found {len(group1_files)} subjects")

    # Find matrices for group 2
    logger.info(f"\nFinding matrices for {group2_name}...")
    group2_files, group2_ids = find_connectivity_matrices(
        connectome_dir, atlas, group2_subjects
    )
    logger.info(f"  Found {len(group2_files)} subjects")

    if len(group1_files) == 0 or len(group2_files) == 0:
        logger.error("Insufficient subjects in one or both groups!")
        return {'status': 'failed', 'error': 'Insufficient subjects'}

    # Load matrices
    logger.info("\nLoading connectivity matrices...")
    group1_matrices, _ = load_connectivity_matrices(group1_files, group1_ids)
    group2_matrices, _ = load_connectivity_matrices(group2_files, group2_ids)

    # Compute group difference
    logger.info("\nComputing group difference...")
    results = compute_group_difference(
        group1_matrices=group1_matrices,
        group2_matrices=group2_matrices,
        group1_name=group1_name,
        group2_name=group2_name,
        paired=paired,
        alpha=alpha,
        output_dir=output_dir,
        output_prefix='comparison'
    )

    logger.info(f"\n✓ Results saved to: {output_dir}")

    return {
        'status': 'success',
        'output_dir': str(output_dir),
        'n_group1': len(group1_ids),
        'n_group2': len(group2_ids),
        'n_significant_edges': results['n_significant']
    }


def run_nbs_analysis(
    study_root: Path,
    atlas: str,
    group1_subjects: List[str],
    group2_subjects: List[str],
    group1_name: str,
    group2_name: str,
    threshold: float,
    n_permutations: int,
    alpha: float,
    logger: logging.Logger
) -> Dict:
    """Run Network-Based Statistic analysis"""
    logger.info("="*80)
    logger.info("NETWORK-BASED STATISTIC (NBS)")
    logger.info("="*80)

    connectome_dir = study_root / "connectome"
    output_dir = connectome_dir / "group_statistics" / atlas / "nbs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matrices
    logger.info(f"\nFinding matrices for {group1_name}...")
    group1_files, group1_ids = find_connectivity_matrices(
        connectome_dir, atlas, group1_subjects
    )
    logger.info(f"  Found {len(group1_files)} subjects")

    logger.info(f"\nFinding matrices for {group2_name}...")
    group2_files, group2_ids = find_connectivity_matrices(
        connectome_dir, atlas, group2_subjects
    )
    logger.info(f"  Found {len(group2_files)} subjects")

    if len(group1_files) == 0 or len(group2_files) == 0:
        logger.error("Insufficient subjects in one or both groups!")
        return {'status': 'failed', 'error': 'Insufficient subjects'}

    # Load matrices
    logger.info("\nLoading connectivity matrices...")
    group1_matrices, _ = load_connectivity_matrices(group1_files, group1_ids)
    group2_matrices, _ = load_connectivity_matrices(group2_files, group2_ids)

    # Run NBS
    logger.info("\nRunning Network-Based Statistic...")
    results = compute_network_based_statistic(
        group1_matrices=group1_matrices,
        group2_matrices=group2_matrices,
        threshold=threshold,
        n_permutations=n_permutations,
        alpha=alpha,
        output_dir=output_dir
    )

    logger.info(f"\n✓ Results saved to: {output_dir}")

    return {
        'status': 'success',
        'output_dir': str(output_dir),
        'n_group1': len(group1_ids),
        'n_group2': len(group2_ids),
        'n_components': len(results['components']),
        'n_significant': results.get('n_significant', 0)
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Group-level statistical analysis for connectivity matrices",
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
        '--analysis',
        type=str,
        required=True,
        choices=['group_average', 'group_comparison', 'nbs'],
        help='Type of analysis to perform'
    )

    # Group selection
    parser.add_argument(
        '--subjects',
        type=Path,
        help='Text file with subject IDs (one per line, for group_average)'
    )

    parser.add_argument(
        '--group1',
        type=Path,
        help='Text file with Group 1 subject IDs (for comparison/NBS)'
    )

    parser.add_argument(
        '--group2',
        type=Path,
        help='Text file with Group 2 subject IDs (for comparison/NBS)'
    )

    parser.add_argument(
        '--demographics',
        type=Path,
        help='CSV file with subject demographics and group assignments'
    )

    parser.add_argument(
        '--subject-column',
        type=str,
        default='Subject',
        help='Column name for subject IDs in demographics CSV (default: Subject)'
    )

    parser.add_argument(
        '--group-column',
        type=str,
        help='Column name for group assignments in demographics CSV (required with --demographics)'
    )

    parser.add_argument(
        '--group1-value',
        help='Value in group column that defines Group 1 (e.g., 1, "control")'
    )

    parser.add_argument(
        '--group2-value',
        help='Value in group column that defines Group 2 (e.g., 2, "patient")'
    )

    parser.add_argument(
        '--group1-name',
        type=str,
        default='Group 1',
        help='Name for Group 1 (default: "Group 1")'
    )

    parser.add_argument(
        '--group2-name',
        type=str,
        default='Group 2',
        help='Name for Group 2 (default: "Group 2")'
    )

    # Analysis parameters
    parser.add_argument(
        '--consistency-threshold',
        type=float,
        help='Consistency threshold for group averaging (0-1)'
    )

    parser.add_argument(
        '--paired',
        action='store_true',
        help='Use paired t-test (for group_comparison)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )

    parser.add_argument(
        '--nbs-threshold',
        type=float,
        default=3.0,
        help='T-statistic threshold for NBS (default: 3.0)'
    )

    parser.add_argument(
        '--n-permutations',
        type=int,
        default=5000,
        help='Number of permutations for NBS (default: 5000)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup output directory
    connectome_dir = args.study_root / "connectome"
    output_base = connectome_dir / "group_statistics"
    output_base.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_base, verbose=args.verbose)

    logger.info("="*80)
    logger.info("BATCH GROUP STATISTICS")
    logger.info("="*80)
    logger.info(f"Study root: {args.study_root}")
    logger.info(f"Atlas: {args.atlas}")
    logger.info(f"Analysis: {args.analysis}")

    # Run appropriate analysis
    if args.analysis == 'group_average':
        subjects = load_subject_list(args.subjects)
        result = run_group_average(
            study_root=args.study_root,
            atlas=args.atlas,
            subjects=subjects,
            consistency_threshold=args.consistency_threshold,
            logger=logger
        )

    elif args.analysis == 'group_comparison':
        # Get group assignments from CSV or text files
        if args.demographics:
            # Use CSV file
            if not args.group_column:
                logger.error("--group-column required when using --demographics")
                sys.exit(1)
            if not args.group1_value or not args.group2_value:
                logger.error("--group1-value and --group2-value required when using --demographics")
                sys.exit(1)

            group1_subjects, group2_subjects = extract_groups_from_csv(
                csv_file=args.demographics,
                subject_column=args.subject_column,
                group_column=args.group_column,
                group1_value=args.group1_value,
                group2_value=args.group2_value,
                logger=logger
            )
        else:
            # Use text files
            if args.group1 is None or args.group2 is None:
                logger.error("Either --demographics OR --group1/--group2 required for group_comparison")
                sys.exit(1)

            group1_subjects = load_subject_list(args.group1)
            group2_subjects = load_subject_list(args.group2)

        result = run_group_comparison(
            study_root=args.study_root,
            atlas=args.atlas,
            group1_subjects=group1_subjects,
            group2_subjects=group2_subjects,
            group1_name=args.group1_name,
            group2_name=args.group2_name,
            paired=args.paired,
            alpha=args.alpha,
            logger=logger
        )

    elif args.analysis == 'nbs':
        # Get group assignments from CSV or text files
        if args.demographics:
            # Use CSV file
            if not args.group_column:
                logger.error("--group-column required when using --demographics")
                sys.exit(1)
            if not args.group1_value or not args.group2_value:
                logger.error("--group1-value and --group2-value required when using --demographics")
                sys.exit(1)

            group1_subjects, group2_subjects = extract_groups_from_csv(
                csv_file=args.demographics,
                subject_column=args.subject_column,
                group_column=args.group_column,
                group1_value=args.group1_value,
                group2_value=args.group2_value,
                logger=logger
            )
        else:
            # Use text files
            if args.group1 is None or args.group2 is None:
                logger.error("Either --demographics OR --group1/--group2 required for NBS")
                sys.exit(1)

            group1_subjects = load_subject_list(args.group1)
            group2_subjects = load_subject_list(args.group2)

        result = run_nbs_analysis(
            study_root=args.study_root,
            atlas=args.atlas,
            group1_subjects=group1_subjects,
            group2_subjects=group2_subjects,
            group1_name=args.group1_name,
            group2_name=args.group2_name,
            threshold=args.nbs_threshold,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            logger=logger
        )

    # Report final status
    if result['status'] == 'success':
        logger.info("\n✓ Analysis completed successfully!")
        sys.exit(0)
    else:
        logger.error(f"\n✗ Analysis failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
