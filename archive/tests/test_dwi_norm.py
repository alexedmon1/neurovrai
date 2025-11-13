#!/usr/bin/env python3
"""
Test script for DWI normalization to FMRIB58_FA template.
"""

import logging
from pathlib import Path
from mri_preprocess.utils.dwi_normalization import (
    normalize_dwi_to_fmrib58,
    apply_warp_to_metrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dwi_normalization_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    # Paths
    subject = 'IRC805-0580101'
    derivatives_dir = Path(f'/mnt/bytopia/IRC805/derivatives/{subject}/dwi')

    # FA file
    fa_file = derivatives_dir / 'dti' / 'dtifit__FA.nii.gz'

    if not fa_file.exists():
        logger.error(f"FA file not found: {fa_file}")
        return

    logger.info("="*70)
    logger.info("Testing DWI Normalization to FMRIB58_FA")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"FA file: {fa_file}")
    logger.info("")

    # Step 1: Normalize FA to FMRIB58_FA
    logger.info("Step 1: Normalizing FA to FMRIB58_FA template")
    norm_results = normalize_dwi_to_fmrib58(
        fa_file=fa_file,
        output_dir=derivatives_dir,
        fmrib58_template=None  # Uses $FSLDIR default
    )

    logger.info("")
    logger.info("Normalization results:")
    logger.info(f"  Affine matrix: {norm_results['affine_mat']}")
    logger.info(f"  Forward warp: {norm_results['forward_warp']}")
    logger.info(f"  Inverse warp: {norm_results['inverse_warp']}")
    logger.info(f"  Normalized FA: {norm_results['fa_normalized']}")
    logger.info("")

    # Step 2: Collect all DWI metrics
    logger.info("Step 2: Collecting DWI metrics for normalization")
    metric_files = []

    # DTI metrics
    dti_dir = derivatives_dir / 'dti'
    if dti_dir.exists():
        for metric in ['MD', 'L1', 'L2', 'L3']:
            metric_file = list(dti_dir.glob(f'*{metric}.nii.gz'))
            if metric_file:
                metric_files.append(metric_file[0])
                logger.info(f"  Found DTI metric: {metric_file[0].name}")

    # DKI metrics
    dki_dir = derivatives_dir / 'dki'
    if dki_dir.exists():
        for metric in ['mean_kurtosis', 'axial_kurtosis', 'radial_kurtosis', 'kurtosis_fa']:
            metric_file = list(dki_dir.glob(f'*{metric}.nii.gz'))
            if metric_file:
                metric_files.append(metric_file[0])
                logger.info(f"  Found DKI metric: {metric_file[0].name}")

    # NODDI metrics
    noddi_dir = derivatives_dir / 'noddi'
    if noddi_dir.exists():
        for metric in ['ficvf', 'odi', 'fiso']:
            metric_file = list(noddi_dir.glob(f'*{metric}.nii.gz'))
            if metric_file:
                metric_files.append(metric_file[0])
                logger.info(f"  Found NODDI metric: {metric_file[0].name}")

    logger.info(f"  Total metrics to normalize: {len(metric_files)}")
    logger.info("")

    # Step 3: Apply warp to all metrics
    if metric_files:
        logger.info("Step 3: Applying normalization to all metrics")
        normalized_files = apply_warp_to_metrics(
            metric_files=metric_files,
            forward_warp=norm_results['forward_warp'],
            fmrib58_template=None,
            output_dir=derivatives_dir,
            interpolation='spline'
        )

        logger.info("")
        logger.info(f"Successfully normalized {len(normalized_files)} metrics")
        logger.info("")

    # Step 4: Verify outputs
    logger.info("Step 4: Verifying normalized outputs")
    normalized_dir = derivatives_dir / 'normalized'
    if normalized_dir.exists():
        normalized_count = len(list(normalized_dir.glob('*.nii.gz')))
        logger.info(f"  Found {normalized_count} normalized files in {normalized_dir}")

        # List first 10 files
        for i, f in enumerate(sorted(normalized_dir.glob('*.nii.gz'))[:10]):
            logger.info(f"    {f.name}")
        if normalized_count > 10:
            logger.info(f"    ... and {normalized_count - 10} more")

    logger.info("")
    logger.info("="*70)
    logger.info("DWI NORMALIZATION TEST COMPLETE!")
    logger.info("="*70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Visually inspect normalized FA overlay on FMRIB58_FA template")
    logger.info("  2. Check alignment quality at corpus callosum and CST")
    logger.info("  3. Test inverse warp by bringing JHU atlas to native DWI space")
    logger.info("")

if __name__ == '__main__':
    main()
