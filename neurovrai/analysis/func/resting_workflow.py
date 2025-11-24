#!/usr/bin/env python3
"""
Resting-State fMRI Analysis Workflow

Runs ReHo and fALFF analysis on preprocessed resting-state fMRI data.
Generates standardized outputs and quality control reports.

Future additions: MELODIC (group ICA)
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import nibabel as nib

from .reho import compute_reho_map, compute_reho_zscore
from .falff import compute_falff_map, compute_falff_zscore


def run_resting_state_analysis(
    func_file: Path,
    mask_file: Optional[Path] = None,
    output_dir: Path = None,
    tr: Optional[float] = None,
    subject_id: Optional[str] = None,
    reho_neighborhood: int = 27,
    falff_low_freq: float = 0.01,
    falff_high_freq: float = 0.08,
    compute_zscore: bool = True
) -> Dict:
    """
    Run complete resting-state fMRI analysis (ReHo + fALFF)

    Args:
        func_file: Preprocessed 4D functional image
                  Should be detrended, bandpass filtered, and nuisance-regressed
        mask_file: Brain mask
        output_dir: Output directory
        tr: Repetition time (will read from header if not provided)
        subject_id: Subject identifier
        reho_neighborhood: ReHo neighborhood size (7, 19, or 27)
        falff_low_freq: fALFF lower frequency bound (Hz)
        falff_high_freq: fALFF upper frequency bound (Hz)
        compute_zscore: Whether to compute z-scored maps

    Returns:
        Dictionary with analysis results and file paths
    """
    # Create output directories first
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"resting_state_analysis_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 80)
    logging.info("RESTING-STATE fMRI ANALYSIS")
    logging.info("=" * 80)
    if subject_id:
        logging.info(f"Subject: {subject_id}")
    logging.info(f"Input: {func_file}")
    logging.info(f"Output: {output_dir}")
    logging.info("")

    reho_dir = output_dir / 'reho'
    falff_dir = output_dir / 'falff'
    reho_dir.mkdir(exist_ok=True)
    falff_dir.mkdir(exist_ok=True)

    # Store results
    results = {
        'subject_id': subject_id,
        'func_file': str(func_file),
        'mask_file': str(mask_file) if mask_file else None,
        'timestamp': timestamp,
        'reho': {},
        'falff': {}
    }

    # =========================================================================
    # ReHo Analysis
    # =========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("REHO ANALYSIS")
    logging.info("=" * 80)

    try:
        reho_output = reho_dir / 'reho.nii.gz'

        reho_data, reho_img = compute_reho_map(
            func_file=func_file,
            mask_file=mask_file,
            neighborhood=reho_neighborhood,
            output_file=reho_output
        )

        results['reho']['map'] = str(reho_output)
        results['reho']['neighborhood'] = reho_neighborhood

        # Compute z-scored version
        if compute_zscore:
            reho_zscore_output = reho_dir / 'reho_zscore.nii.gz'
            reho_zscore_data, reho_zscore_img = compute_reho_zscore(
                reho_file=reho_output,
                mask_file=mask_file,
                output_file=reho_zscore_output
            )
            results['reho']['zscore_map'] = str(reho_zscore_output)

        logging.info("✓ ReHo analysis complete")
        results['reho']['status'] = 'success'

    except Exception as e:
        logging.error(f"ReHo analysis failed: {str(e)}", exc_info=True)
        results['reho']['status'] = 'failed'
        results['reho']['error'] = str(e)

    # =========================================================================
    # fALFF Analysis
    # =========================================================================
    logging.info("\n" + "=" * 80)
    logging.info("FALFF ANALYSIS")
    logging.info("=" * 80)

    try:
        falff_result = compute_falff_map(
            func_file=func_file,
            mask_file=mask_file,
            tr=tr,
            low_freq=falff_low_freq,
            high_freq=falff_high_freq,
            output_dir=falff_dir
        )

        results['falff']['alff_map'] = str(falff_dir / 'alff.nii.gz')
        results['falff']['falff_map'] = str(falff_dir / 'falff.nii.gz')
        results['falff']['low_freq'] = falff_low_freq
        results['falff']['high_freq'] = falff_high_freq
        results['falff']['statistics'] = falff_result['statistics']

        # Compute z-scored versions
        if compute_zscore:
            zscore_result = compute_falff_zscore(
                alff_file=falff_dir / 'alff.nii.gz',
                falff_file=falff_dir / 'falff.nii.gz',
                mask_file=mask_file,
                output_dir=falff_dir
            )
            results['falff']['alff_zscore_map'] = str(falff_dir / 'alff_zscore.nii.gz')
            results['falff']['falff_zscore_map'] = str(falff_dir / 'falff_zscore.nii.gz')

        logging.info("✓ fALFF analysis complete")
        results['falff']['status'] = 'success'

    except Exception as e:
        logging.error(f"fALFF analysis failed: {str(e)}", exc_info=True)
        results['falff']['status'] = 'failed'
        results['falff']['error'] = str(e)

    # =========================================================================
    # Save Results Summary
    # =========================================================================
    summary_file = output_dir / 'resting_state_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Summary: {summary_file}")
    logging.info(f"Log: {log_file}")
    logging.info("=" * 80 + "\n")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Run resting-state fMRI analysis (ReHo + fALFF)"
    )
    parser.add_argument(
        '--func',
        type=Path,
        required=True,
        help='Preprocessed 4D functional image'
    )
    parser.add_argument(
        '--mask',
        type=Path,
        help='Brain mask'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--subject',
        type=str,
        help='Subject ID'
    )
    parser.add_argument(
        '--tr',
        type=float,
        help='Repetition time in seconds'
    )
    parser.add_argument(
        '--reho-neighborhood',
        type=int,
        default=27,
        choices=[7, 19, 27],
        help='ReHo neighborhood size (default: 27)'
    )
    parser.add_argument(
        '--low-freq',
        type=float,
        default=0.01,
        help='fALFF lower frequency (default: 0.01 Hz)'
    )
    parser.add_argument(
        '--high-freq',
        type=float,
        default=0.08,
        help='fALFF upper frequency (default: 0.08 Hz)'
    )
    parser.add_argument(
        '--no-zscore',
        action='store_true',
        help='Skip z-score normalization'
    )

    args = parser.parse_args()

    # Run analysis
    results = run_resting_state_analysis(
        func_file=args.func,
        mask_file=args.mask,
        output_dir=args.output_dir,
        tr=args.tr,
        subject_id=args.subject,
        reho_neighborhood=args.reho_neighborhood,
        falff_low_freq=args.low_freq,
        falff_high_freq=args.high_freq,
        compute_zscore=not args.no_zscore
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"ReHo: {results['reho']['status']}")
    if results['reho']['status'] == 'success':
        print(f"  Map: {results['reho']['map']}")
        if 'zscore_map' in results['reho']:
            print(f"  Z-score: {results['reho']['zscore_map']}")

    print(f"\nfALFF: {results['falff']['status']}")
    if results['falff']['status'] == 'success':
        print(f"  ALFF: {results['falff']['alff_map']}")
        print(f"  fALFF: {results['falff']['falff_map']}")
        if 'alff_zscore_map' in results['falff']:
            print(f"  ALFF z-score: {results['falff']['alff_zscore_map']}")
            print(f"  fALFF z-score: {results['falff']['falff_zscore_map']}")
    print("=" * 80)
