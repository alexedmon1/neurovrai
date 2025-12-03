#!/usr/bin/env python3
"""
DTI metric calculations and utilities.

Provides functions for calculating derived DTI metrics from eigenvalues.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import nibabel as nib
import numpy as np


logger = logging.getLogger(__name__)


def calculate_ad_rd(dti_dir: Path, prefix: str = 'dtifit__') -> Tuple[Optional[Path], Optional[Path]]:
    """
    Calculate AD and RD from L1, L2, L3 eigenvalue files.

    AD (Axial Diffusivity) = L1
    RD (Radial Diffusivity) = (L2 + L3) / 2

    Parameters
    ----------
    dti_dir : Path
        Directory containing dtifit output (L1, L2, L3)
    prefix : str, optional
        Filename prefix for dtifit outputs (default: 'dtifit__')

    Returns
    -------
    tuple
        (AD_file, RD_file) paths, or (None, None) if eigenvalue files not found

    Examples
    --------
    >>> ad_file, rd_file = calculate_ad_rd(Path('/derivatives/sub-001/dwi/dti'))
    >>> print(f"AD: {ad_file}, RD: {rd_file}")
    """
    dti_dir = Path(dti_dir)

    # Find eigenvalue files
    l1_file = dti_dir / f"{prefix}L1.nii.gz"
    l2_file = dti_dir / f"{prefix}L2.nii.gz"
    l3_file = dti_dir / f"{prefix}L3.nii.gz"

    # Check if files exist
    missing_files = []
    for f in [l1_file, l2_file, l3_file]:
        if not f.exists():
            missing_files.append(f.name)

    if missing_files:
        logger.warning(f"Cannot calculate AD/RD: missing files {missing_files}")
        return None, None

    try:
        # Load eigenvalue images
        logger.info(f"Calculating AD and RD from eigenvalues in {dti_dir}")
        l1_img = nib.load(l1_file)
        l2_img = nib.load(l2_file)
        l3_img = nib.load(l3_file)

        l1_data = l1_img.get_fdata()
        l2_data = l2_img.get_fdata()
        l3_data = l3_img.get_fdata()

        # Calculate metrics
        # AD (Axial Diffusivity) = L1
        ad_data = l1_data.copy()

        # RD (Radial Diffusivity) = (L2 + L3) / 2
        rd_data = (l2_data + l3_data) / 2.0

        # Save outputs with same prefix as input files
        ad_file = dti_dir / f"{prefix}AD.nii.gz"
        rd_file = dti_dir / f"{prefix}RD.nii.gz"

        # Use L1's header and affine
        ad_img = nib.Nifti1Image(ad_data, l1_img.affine, l1_img.header)
        rd_img = nib.Nifti1Image(rd_data, l1_img.affine, l1_img.header)

        nib.save(ad_img, ad_file)
        nib.save(rd_img, rd_file)

        logger.info(f"  ✓ Created AD: {ad_file.name}")
        logger.info(f"  ✓ Created RD: {rd_file.name}")

        return ad_file, rd_file

    except Exception as e:
        logger.error(f"Failed to calculate AD/RD: {e}")
        return None, None


def validate_dti_metrics(
    dti_dir: Path,
    required_metrics: list = None,
    prefix: str = 'dtifit__'
) -> dict:
    """
    Validate presence and basic properties of DTI metrics.

    Parameters
    ----------
    dti_dir : Path
        Directory containing DTI metrics
    required_metrics : list, optional
        List of required metric names (default: ['FA', 'MD', 'AD', 'RD'])
    prefix : str, optional
        Filename prefix (default: 'dtifit__')

    Returns
    -------
    dict
        Dictionary with validation results for each metric

    Examples
    --------
    >>> results = validate_dti_metrics(Path('/derivatives/sub-001/dwi/dti'))
    >>> print(results['FA']['exists'])
    True
    """
    if required_metrics is None:
        required_metrics = ['FA', 'MD', 'AD', 'RD']

    dti_dir = Path(dti_dir)
    results = {}

    for metric in required_metrics:
        metric_file = dti_dir / f"{prefix}{metric}.nii.gz"

        result = {
            'exists': metric_file.exists(),
            'path': metric_file if metric_file.exists() else None
        }

        if result['exists']:
            try:
                img = nib.load(metric_file)
                data = img.get_fdata()
                result['shape'] = data.shape
                result['min'] = float(np.min(data))
                result['max'] = float(np.max(data))
                result['mean'] = float(np.mean(data))
            except Exception as e:
                result['error'] = str(e)

        results[metric] = result

    return results
