#!/usr/bin/env python3
"""
Calculate Cohen's d Effect Sizes from FSL Randomise Results

This script generates standardized effect size maps from existing randomise
t-statistic outputs, providing interpretable measures of effect magnitude
independent of sample size.

Example usage:
    # Basic usage with automatic detection
    python calculate_effect_sizes.py \
        --randomise-dir results/randomise_output \
        --design-file design.csv \
        --output-dir results/effect_sizes

    # Specify design details explicitly
    python calculate_effect_sizes.py \
        --randomise-dir results/tbss_randomise \
        --n1 60 --n2 60 \
        --design-type two_sample \
        --output-dir results/effect_sizes

    # With contrast names
    python calculate_effect_sizes.py \
        --randomise-dir results/randomise_output \
        --design-file design.csv \
        --contrast-names "1:GDM_vs_Control,2:Control_vs_GDM" \
        --output-dir results/effect_sizes
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurovrai.analysis.stats.effect_size import (
    create_effect_size_maps,
    batch_effect_size_calculation,
    EffectSizeError
)


def parse_contrast_names(contrast_str: str) -> Dict[int, str]:
    """
    Parse contrast names from command line string.

    Format: "1:name1,2:name2,3:name3"

    Parameters
    ----------
    contrast_str : str
        Comma-separated contrast definitions

    Returns
    -------
    dict
        Mapping of contrast numbers to names
    """
    contrasts = {}

    for item in contrast_str.split(','):
        if ':' not in item:
            continue
        num, name = item.split(':', 1)
        contrasts[int(num)] = name.strip()

    return contrasts


def main():
    """Main entry point for effect size calculation."""

    parser = argparse.ArgumentParser(
        description='Calculate Cohen\'s d effect sizes from randomise results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--randomise-dir',
        type=Path,
        required=True,
        help='Directory containing randomise output files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for effect size maps'
    )

    # Design information
    design_group = parser.add_mutually_exclusive_group()
    design_group.add_argument(
        '--design-file',
        type=Path,
        help='Design matrix file (.mat or .csv) to extract sample sizes'
    )
    design_group.add_argument(
        '--n1',
        type=int,
        help='Sample size for group 1 (or total N for one-sample)'
    )

    parser.add_argument(
        '--n2',
        type=int,
        help='Sample size for group 2 (for two-sample designs)'
    )
    parser.add_argument(
        '--design-type',
        choices=['one_sample', 'two_sample', 'paired'],
        default='two_sample',
        help='Type of statistical design (default: two_sample)'
    )

    # Contrast information
    parser.add_argument(
        '--contrast-names',
        type=str,
        help='Contrast names in format "1:name1,2:name2" (e.g., "1:GDM_vs_HC,2:HC_vs_GDM")'
    )

    # Processing options
    parser.add_argument(
        '--p-threshold',
        type=float,
        default=0.05,
        help='P-value threshold for corrected maps (default: 0.05)'
    )
    parser.add_argument(
        '--no-hedges',
        action='store_true',
        help='Skip calculation of Hedges\' g (bias-corrected effect size)'
    )
    parser.add_argument(
        '--no-ci',
        action='store_true',
        help='Skip calculation of confidence intervals'
    )

    # Output options
    parser.add_argument(
        '--create-plots',
        action='store_true',
        help='Create visualization plots for effect size maps'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Validate inputs
    if not args.randomise_dir.exists():
        logger.error(f"Randomise directory not found: {args.randomise_dir}")
        sys.exit(1)

    # Check for t-stat files
    tstat_files = list(args.randomise_dir.glob("*tstat*.nii.gz"))
    if not tstat_files:
        logger.error(f"No t-statistic files found in {args.randomise_dir}")
        logger.error("Expected files matching pattern: *tstat*.nii.gz")
        sys.exit(1)

    logger.info(f"Found {len(tstat_files)} t-statistic file(s)")

    # Parse contrast names if provided
    contrast_names = None
    if args.contrast_names:
        contrast_names = parse_contrast_names(args.contrast_names)
        logger.info(f"Parsed {len(contrast_names)} contrast names")

    try:
        # Determine how to run based on inputs
        if args.design_file:
            # Use batch processing with design file
            logger.info("Running batch effect size calculation...")
            results = batch_effect_size_calculation(
                randomise_dir=args.randomise_dir,
                design_file=args.design_file,
                output_dir=args.output_dir,
                contrast_names=contrast_names
            )

        elif args.n1 is not None:
            # Process manually with specified sample sizes
            logger.info("Processing with manually specified sample sizes...")

            design_info = {
                'n1': args.n1,
                'n2': args.n2,
                'design_type': args.design_type
            }

            results = {}
            for tstat_file in sorted(tstat_files):
                # Extract contrast number from filename
                stem = tstat_file.stem
                if 'tstat' in stem:
                    try:
                        # Handle names like "randomise_tstat1" or "tbss_tstat1"
                        parts = stem.split('tstat')
                        if len(parts) > 1:
                            contrast_num = int(''.join(c for c in parts[-1] if c.isdigit())[:1])
                        else:
                            contrast_num = 1
                    except:
                        contrast_num = 1
                else:
                    contrast_num = 1

                # Get contrast name
                if contrast_names and contrast_num in contrast_names:
                    contrast_name = contrast_names[contrast_num]
                else:
                    contrast_name = f"contrast{contrast_num}"

                logger.info(f"\nProcessing: {tstat_file.name} as {contrast_name}")

                # Look for corrected p-value file
                corrp_patterns = [
                    f"*tfce_corrp_tstat{contrast_num}.nii.gz",
                    f"*corrp_tstat{contrast_num}.nii.gz",
                    f"*tstat{contrast_num}_corrp.nii.gz"
                ]

                corrp_file = None
                for pattern in corrp_patterns:
                    matches = list(args.randomise_dir.glob(pattern))
                    if matches:
                        corrp_file = matches[0]
                        logger.info(f"  Found corrected p-values: {corrp_file.name}")
                        break

                if not corrp_file:
                    logger.info("  No corrected p-value file found")

                # Create output directory for this contrast
                contrast_output = args.output_dir / contrast_name

                # Calculate effect sizes
                result = create_effect_size_maps(
                    tstat_file=tstat_file,
                    output_dir=contrast_output,
                    design_info={**design_info, 'contrast_name': contrast_name},
                    corrp_file=corrp_file,
                    p_threshold=args.p_threshold,
                    calculate_hedges=not args.no_hedges,
                    calculate_ci=not args.no_ci
                )

                results[contrast_name] = result

        else:
            logger.error("Must provide either --design-file or --n1")
            sys.exit(1)

        # Print summary
        print("\n" + "="*60)
        print("EFFECT SIZE CALCULATION COMPLETE")
        print("="*60)

        for contrast_name, result in results.items():
            stats = result['statistics']

            print(f"\n{contrast_name}:")
            print("-" * len(contrast_name))
            print(f"  Mean Cohen's d: {stats['mean_d']:.3f}")
            print(f"  Median Cohen's d: {stats['median_d']:.3f}")
            print(f"  Std Cohen's d: {stats['std_d']:.3f}")
            print(f"  Range: [{stats['min_d']:.3f}, {stats['max_d']:.3f}]")

            print(f"\n  Effect size distribution:")
            print(f"    Small (0.2-0.5): {stats['percent_small']:.1f}%")
            print(f"    Medium (0.5-0.8): {stats['percent_medium']:.1f}%")
            print(f"    Large (>0.8): {stats['percent_large']:.1f}%")

            if 'sig_n_voxels' in stats:
                print(f"\n  Significant voxels (p<{args.p_threshold}): {stats['sig_n_voxels']:,}")
                if stats['sig_n_voxels'] > 0:
                    print(f"  Mean d (significant only): {stats['sig_mean_d']:.3f}")

            print(f"\n  Output files:")
            for key, path in result['output_files'].items():
                print(f"    {key}: {Path(path).name}")

        print(f"\n✓ All results saved to: {args.output_dir}")
        print("="*60)

    except EffectSizeError as e:
        logger.error(f"Effect size calculation failed: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()