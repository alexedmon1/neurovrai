#!/usr/bin/env python3
"""
Functional Connectivity Analysis CLI

Command-line interface for computing functional connectivity matrices from
preprocessed fMRI data.

Usage:
    # Basic usage
    python -m neurovrai.connectome.run_functional_connectivity \
        --func-file preprocessed_bold.nii.gz \
        --atlas schaefer_400.nii.gz \
        --output-dir /study/analysis/fc/subject-001/

    # With options
    python -m neurovrai.connectome.run_functional_connectivity \
        --func-file preprocessed_bold.nii.gz \
        --atlas schaefer_400.nii.gz \
        --labels schaefer_400_labels.txt \
        --mask brain_mask.nii.gz \
        --method pearson \
        --fisher-z \
        --threshold 0.3 \
        --output-dir /study/analysis/fc/subject-001/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from neurovrai.connectome.roi_extraction import extract_roi_timeseries, load_atlas
from neurovrai.connectome.functional_connectivity import compute_functional_connectivity


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"functional_connectivity_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")

    return logger


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Compute functional connectivity matrix from fMRI data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with Schaefer 400 atlas
  python -m neurovrai.connectome.run_functional_connectivity \\
      --func-file /data/derivatives/sub-001/func/preprocessed_bold.nii.gz \\
      --atlas /data/atlases/schaefer_400_MNI.nii.gz \\
      --output-dir /data/analysis/fc/sub-001/

  # With labels and mask
  python -m neurovrai.connectome.run_functional_connectivity \\
      --func-file preprocessed_bold.nii.gz \\
      --atlas schaefer_400.nii.gz \\
      --labels schaefer_400_labels.txt \\
      --mask brain_mask.nii.gz \\
      --output-dir /analysis/fc/

  # Partial correlation with thresholding
  python -m neurovrai.connectome.run_functional_connectivity \\
      --func-file preprocessed_bold.nii.gz \\
      --atlas schaefer_400.nii.gz \\
      --partial \\
      --threshold 0.3 \\
      --output-dir /analysis/fc/
        """
    )

    # Required arguments
    parser.add_argument(
        '--func-file',
        type=Path,
        required=True,
        help='Path to preprocessed 4D functional data (NIfTI)'
    )

    parser.add_argument(
        '--atlas',
        type=Path,
        required=True,
        help='Path to atlas parcellation (NIfTI)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for results'
    )

    # Optional arguments
    parser.add_argument(
        '--labels',
        type=Path,
        help='Path to atlas labels file (format: "index name")'
    )

    parser.add_argument(
        '--mask',
        type=Path,
        help='Path to brain mask (NIfTI)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='pearson',
        choices=['pearson', 'spearman'],
        help='Correlation method (default: pearson)'
    )

    parser.add_argument(
        '--partial',
        action='store_true',
        help='Compute partial correlation instead of full correlation'
    )

    parser.add_argument(
        '--fisher-z',
        action='store_true',
        default=True,
        help='Apply Fisher z-transformation (default: True)'
    )

    parser.add_argument(
        '--no-fisher-z',
        action='store_false',
        dest='fisher_z',
        help='Disable Fisher z-transformation'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        help='Threshold for sparsifying connectivity matrix'
    )

    parser.add_argument(
        '--min-voxels',
        type=int,
        default=10,
        help='Minimum voxels per ROI (default: 10)'
    )

    parser.add_argument(
        '--statistic',
        type=str,
        default='mean',
        choices=['mean', 'median', 'pca'],
        help='ROI aggregation statistic (default: mean)'
    )

    parser.add_argument(
        '--output-prefix',
        type=str,
        default='fc',
        help='Prefix for output files (default: fc)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.output_dir, verbose=args.verbose)

    logger.info("=" * 80)
    logger.info("FUNCTIONAL CONNECTIVITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Functional data: {args.func_file}")
    logger.info(f"Atlas: {args.atlas}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Method: {'partial correlation' if args.partial else args.method}")
    logger.info(f"Fisher z-transform: {args.fisher_z}")

    # Validate inputs
    if not args.func_file.exists():
        logger.error(f"Functional data file not found: {args.func_file}")
        sys.exit(1)

    if not args.atlas.exists():
        logger.error(f"Atlas file not found: {args.atlas}")
        sys.exit(1)

    if args.mask and not args.mask.exists():
        logger.error(f"Mask file not found: {args.mask}")
        sys.exit(1)

    try:
        # Step 1: Extract ROI timeseries
        logger.info("\n" + "=" * 80)
        logger.info("[Step 1] Extracting ROI timeseries")
        logger.info("=" * 80)

        timeseries, roi_names = extract_roi_timeseries(
            data_file=args.func_file,
            atlas=args.atlas,
            mask_file=args.mask,
            labels_file=args.labels,
            min_voxels=args.min_voxels,
            statistic=args.statistic
        )

        logger.info(f"Extracted timeseries shape: {timeseries.shape}")
        logger.info(f"Number of ROIs: {len(roi_names)}")

        # Step 2: Compute functional connectivity
        logger.info("\n" + "=" * 80)
        logger.info("[Step 2] Computing functional connectivity")
        logger.info("=" * 80)

        fc_results = compute_functional_connectivity(
            timeseries=timeseries,
            roi_names=roi_names,
            method=args.method,
            fisher_z=args.fisher_z,
            partial=args.partial,
            threshold=args.threshold,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix
        )

        # Step 3: Save analysis parameters
        logger.info("\n" + "=" * 80)
        logger.info("[Step 3] Saving analysis parameters")
        logger.info("=" * 80)

        params_file = args.output_dir / f"{args.output_prefix}_parameters.json"
        params = {
            'func_file': str(args.func_file),
            'atlas': str(args.atlas),
            'labels_file': str(args.labels) if args.labels else None,
            'mask': str(args.mask) if args.mask else None,
            'method': args.method,
            'partial': args.partial,
            'fisher_z': args.fisher_z,
            'threshold': args.threshold,
            'min_voxels': args.min_voxels,
            'statistic': args.statistic,
            'n_rois': len(roi_names),
            'n_timepoints': timeseries.shape[0],
            'analysis_date': datetime.now().isoformat()
        }

        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

        logger.info(f"Parameters saved to: {params_file}")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Connectivity matrix shape: {fc_results['connectivity_matrix'].shape}")
        logger.info(f"Number of ROIs: {len(fc_results['roi_names'])}")
        logger.info(f"Mean connectivity: {fc_results['summary']['mean_connectivity']:.4f}")
        logger.info(f"Non-zero edges: {fc_results['summary']['n_edges_nonzero']}")
        logger.info("\nOutput files:")
        for key, filepath in fc_results['output_files'].items():
            logger.info(f"  - {key}: {Path(filepath).name}")
        logger.info("=" * 80)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
