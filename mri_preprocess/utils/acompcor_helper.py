#!/usr/bin/env python3
"""
ACompCor helper functions for functional preprocessing.

This module handles:
1. Quick anatomical segmentation (FAST)
2. Registration of tissue masks to functional space
3. ACompCor component extraction
4. Nuisance regression

Based on fMRIPrep and Nipype ACompCor implementations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)


def run_fast_segmentation(
    t1w_file: Path,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Run FSL FAST tissue segmentation on T1w image.

    Parameters
    ----------
    t1w_file : Path
        T1w anatomical image (should be brain-extracted)
    output_dir : Path
        Output directory for segmentation

    Returns
    -------
    dict
        Paths to tissue probability maps:
        - csf: CSF probability map
        - gm: Gray matter probability map
        - wm: White matter probability map
    """
    logger.info("Running FAST tissue segmentation...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Brain extract if needed
    brain_file = output_dir / 't1w_brain.nii.gz'
    if not brain_file.exists():
        logger.info("  Brain extracting T1w...")
        bet_cmd = [
            'bet',
            str(t1w_file),
            str(brain_file.with_suffix('')),
            '-f', '0.5',
            '-g', '0',
            '-R'
        ]
        subprocess.run(bet_cmd, check=True, capture_output=True)

    # Run FAST
    fast_base = output_dir / 't1w_brain'
    logger.info("  Running FAST (3-class segmentation)...")
    fast_cmd = [
        'fast',
        '-t', '1',  # T1-weighted
        '-n', '3',  # 3 tissue classes (CSF, GM, WM)
        '-o', str(fast_base),
        str(brain_file)
    ]
    subprocess.run(fast_cmd, check=True, capture_output=True)

    # Return tissue probability maps
    # FAST outputs: _pve_0 = CSF, _pve_1 = GM, _pve_2 = WM
    results = {
        'csf': output_dir / 't1w_brain_pve_0.nii.gz',
        'gm': output_dir / 't1w_brain_pve_1.nii.gz',
        'wm': output_dir / 't1w_brain_pve_2.nii.gz',
        'seg': output_dir / 't1w_brain_seg.nii.gz',
        't1w_brain': brain_file
    }

    logger.info(f"  Segmentation complete: {results['seg']}")
    return results


def register_masks_to_functional(
    t1w_brain: Path,
    func_ref: Path,
    csf_mask: Path,
    wm_mask: Path,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Register tissue masks from T1w space to functional space.

    Uses boundary-based registration (BBR) for accurate alignment.

    Parameters
    ----------
    t1w_brain : Path
        Brain-extracted T1w image
    func_ref : Path
        Reference functional image (e.g., mean functional)
    csf_mask : Path
        CSF probability map in T1w space
    wm_mask : Path
        WM probability map in T1w space
    output_dir : Path
        Output directory

    Returns
    -------
    tuple
        (csf_func, wm_func) - Tissue masks in functional space
    """
    logger.info("Registering tissue masks to functional space...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Register functional to T1w (BBR)
    logger.info("  Computing functional → T1w registration (BBR)...")
    xfm_file = output_dir / 'func_to_t1w.mat'

    flirt_cmd = [
        'flirt',
        '-in', str(func_ref),
        '-ref', str(t1w_brain),
        '-dof', '6',
        '-cost', 'bbr',  # Boundary-based registration
        '-omat', str(xfm_file),
        '-wmseg', str(wm_mask)  # Use WM for BBR
    ]
    subprocess.run(flirt_cmd, check=True, capture_output=True)

    # Step 2: Invert transformation (T1w → functional)
    logger.info("  Inverting transformation...")
    inv_xfm_file = output_dir / 't1w_to_func.mat'

    convert_cmd = [
        'convert_xfm',
        '-omat', str(inv_xfm_file),
        '-inverse', str(xfm_file)
    ]
    subprocess.run(convert_cmd, check=True, capture_output=True)

    # Step 3: Apply inverse transform to tissue masks
    logger.info("  Transforming CSF mask to functional space...")
    csf_func = output_dir / 'csf_func.nii.gz'

    apply_csf_cmd = [
        'flirt',
        '-in', str(csf_mask),
        '-ref', str(func_ref),
        '-init', str(inv_xfm_file),
        '-applyxfm',
        '-interp', 'nearestneighbour',
        '-out', str(csf_func)
    ]
    subprocess.run(apply_csf_cmd, check=True, capture_output=True)

    logger.info("  Transforming WM mask to functional space...")
    wm_func = output_dir / 'wm_func.nii.gz'

    apply_wm_cmd = [
        'flirt',
        '-in', str(wm_mask),
        '-ref', str(func_ref),
        '-init', str(inv_xfm_file),
        '-applyxfm',
        '-interp', 'nearestneighbour',
        '-out', str(wm_func)
    ]
    subprocess.run(apply_wm_cmd, check=True, capture_output=True)

    logger.info("  Registration complete")
    return csf_func, wm_func


def prepare_acompcor_masks(
    csf_mask: Path,
    wm_mask: Path,
    output_dir: Path,
    csf_threshold: float = 0.9,
    wm_threshold: float = 0.9,
    erode_mm: float = 2.0
) -> Tuple[Path, Path]:
    """
    Prepare tissue masks for ACompCor by thresholding and eroding.

    Parameters
    ----------
    csf_mask : Path
        CSF probability map in functional space
    wm_mask : Path
        WM probability map in functional space
    output_dir : Path
        Output directory
    csf_threshold : float
        Probability threshold for CSF (default: 0.9)
    wm_threshold : float
        Probability threshold for WM (default: 0.9)
    erode_mm : float
        Erosion amount in mm (default: 2.0)

    Returns
    -------
    tuple
        (csf_eroded, wm_eroded) - Prepared masks for ACompCor
    """
    logger.info("Preparing ACompCor masks...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Threshold CSF
    logger.info(f"  Thresholding CSF (p > {csf_threshold})...")
    csf_thresh = output_dir / 'csf_thresh.nii.gz'

    thresh_csf_cmd = [
        'fslmaths',
        str(csf_mask),
        '-thr', str(csf_threshold),
        '-bin',
        str(csf_thresh)
    ]
    subprocess.run(thresh_csf_cmd, check=True, capture_output=True)

    # Threshold WM
    logger.info(f"  Thresholding WM (p > {wm_threshold})...")
    wm_thresh = output_dir / 'wm_thresh.nii.gz'

    thresh_wm_cmd = [
        'fslmaths',
        str(wm_mask),
        '-thr', str(wm_threshold),
        '-bin',
        str(wm_thresh)
    ]
    subprocess.run(thresh_wm_cmd, check=True, capture_output=True)

    # Erode CSF
    logger.info(f"  Eroding CSF mask ({erode_mm}mm)...")
    csf_eroded = output_dir / 'csf_eroded.nii.gz'

    erode_csf_cmd = [
        'fslmaths',
        str(csf_thresh),
        '-eroF',  # Fast erosion
        str(csf_eroded)
    ]
    subprocess.run(erode_csf_cmd, check=True, capture_output=True)

    # Erode WM
    logger.info(f"  Eroding WM mask ({erode_mm}mm)...")
    wm_eroded = output_dir / 'wm_eroded.nii.gz'

    erode_wm_cmd = [
        'fslmaths',
        str(wm_thresh),
        '-eroF',
        str(wm_eroded)
    ]
    subprocess.run(erode_wm_cmd, check=True, capture_output=True)

    logger.info("  Mask preparation complete")
    return csf_eroded, wm_eroded


def extract_acompcor_components(
    func_file: Path,
    csf_mask: Path,
    wm_mask: Path,
    output_dir: Path,
    num_components: int = 5,
    variance_threshold: float = 0.5
) -> Dict[str, any]:
    """
    Extract ACompCor components from CSF and WM.

    Parameters
    ----------
    func_file : Path
        Functional 4D image (after bandpass filtering)
    csf_mask : Path
        Eroded CSF mask in functional space
    wm_mask : Path
        Eroded WM mask in functional space
    output_dir : Path
        Output directory
    num_components : int
        Number of components to extract (default: 5)
    variance_threshold : float
        Cumulative variance explained threshold (default: 0.5)

    Returns
    -------
    dict
        - components: Array of component time series (n_timepoints x n_components)
        - variance_explained: Variance explained by each component
        - n_voxels_csf: Number of CSF voxels used
        - n_voxels_wm: Number of WM voxels used
        - components_file: Path to saved components file
    """
    logger.info("=" * 70)
    logger.info("ACompCor Component Extraction")
    logger.info("=" * 70)
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading functional data and masks...")
    func_img = nib.load(func_file)
    func_data = func_img.get_fdata()
    n_vols = func_data.shape[3]

    csf_img = nib.load(csf_mask)
    csf_data = csf_img.get_fdata() > 0

    wm_img = nib.load(wm_mask)
    wm_data = wm_img.get_fdata() > 0

    # Combine masks (union)
    combined_mask = np.logical_or(csf_data, wm_data)
    n_voxels = np.sum(combined_mask)

    logger.info(f"  CSF voxels: {np.sum(csf_data)}")
    logger.info(f"  WM voxels: {np.sum(wm_data)}")
    logger.info(f"  Combined voxels: {n_voxels}")
    logger.info("")

    # Extract time series from tissue voxels
    logger.info("Extracting tissue time series...")
    tissue_signals = func_data[combined_mask, :].T  # n_timepoints x n_voxels

    # Demean
    tissue_signals_demeaned = tissue_signals - np.mean(tissue_signals, axis=0)

    # Run PCA
    logger.info(f"Running PCA (extracting {num_components} components)...")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=num_components)
    components = pca.fit_transform(tissue_signals_demeaned)

    # Variance explained
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)

    logger.info("")
    logger.info("Component Analysis:")
    for i, (var, cum_var) in enumerate(zip(variance_explained, cumulative_variance)):
        logger.info(f"  Component {i+1}: {var*100:.2f}% variance (cumulative: {cum_var*100:.2f}%)")
    logger.info("")

    # Save components
    components_file = output_dir / 'acompcor_components.txt'
    np.savetxt(components_file, components, fmt='%.6f', delimiter='\t')
    logger.info(f"Components saved: {components_file}")
    logger.info("")

    return {
        'components': components,
        'variance_explained': variance_explained,
        'cumulative_variance': cumulative_variance,
        'n_voxels_csf': int(np.sum(csf_data)),
        'n_voxels_wm': int(np.sum(wm_data)),
        'n_voxels_total': int(n_voxels),
        'components_file': components_file
    }


def regress_out_components(
    func_file: Path,
    components_file: Path,
    output_file: Path
) -> Path:
    """
    Regress out ACompCor components from functional data.

    Parameters
    ----------
    func_file : Path
        Functional 4D image
    components_file : Path
        ACompCor components file (n_timepoints x n_components)
    output_file : Path
        Output residuals file

    Returns
    -------
    Path
        Path to residuals (cleaned functional data)
    """
    logger.info("Regressing out ACompCor components...")

    # Load data
    func_img = nib.load(func_file)
    func_data = func_img.get_fdata()
    n_vols = func_data.shape[3]

    components = np.loadtxt(components_file)

    # Reshape functional data to 2D (voxels x time)
    original_shape = func_data.shape
    func_2d = func_data.reshape(-1, n_vols).T  # timepoints x voxels

    # Add intercept to components
    design_matrix = np.hstack([np.ones((n_vols, 1)), components])

    # Solve least squares: beta = (X'X)^-1 X'Y
    logger.info(f"  Regressing {components.shape[1]} components from {func_2d.shape[1]} voxels...")
    beta = np.linalg.lstsq(design_matrix, func_2d, rcond=None)[0]

    # Compute residuals: Y - X*beta
    predicted = design_matrix @ beta
    residuals = func_2d - predicted

    # Reshape back to 4D
    residuals_4d = residuals.T.reshape(original_shape)

    # Save
    logger.info(f"  Saving cleaned data: {output_file}")
    residuals_img = nib.Nifti1Image(residuals_4d, func_img.affine, func_img.header)
    nib.save(residuals_img, output_file)

    # Calculate variance explained
    original_var = np.var(func_2d)
    residual_var = np.var(residuals)
    variance_removed = 1 - (residual_var / original_var)

    logger.info(f"  Variance removed by ACompCor: {variance_removed*100:.2f}%")
    logger.info("")

    return output_file
