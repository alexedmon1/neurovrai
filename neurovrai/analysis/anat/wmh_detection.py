#!/usr/bin/env python3
"""
White Matter Hyperintensity (WMH) Detection Module

Core algorithms for detecting and quantifying white matter hyperintensities
using intensity thresholding on T2-weighted images within white matter masks.

The detection algorithm identifies voxels with intensity greater than
mean + (SD_threshold * SD) within the white matter mask, then applies
connected component labeling and minimum cluster size filtering.

Usage:
    from neurovrai.analysis.anat.wmh_detection import detect_wmh, compute_lesion_metrics

    results = detect_wmh(
        t2w_mni=Path('t2w_mni.nii.gz'),
        wm_mask_mni=Path('wm_mask_mni.nii.gz'),
        output_mask=Path('wmh_mask.nii.gz'),
        output_labeled=Path('wmh_labeled.nii.gz'),
        sd_threshold=2.5,
        min_cluster_size=3
    )
"""

import nibabel as nib
import numpy as np
from scipy import ndimage
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def detect_wmh(
    t2w_mni: Path,
    wm_mask_mni: Path,
    output_mask: Path,
    output_labeled: Path,
    sd_threshold: float = 2.5,
    min_cluster_size: int = 3
) -> Dict:
    """
    Detect white matter hyperintensities using intensity thresholding.

    Algorithm:
    1. Extract T2w voxels within WM mask
    2. Compute mean and SD of T2w intensity within WM
    3. Create threshold: threshold = mean + (sd_threshold * SD)
    4. Apply threshold to T2w within WM mask
    5. Label connected components (scipy.ndimage.label)
    6. Remove clusters smaller than min_cluster_size
    7. Relabel remaining components sequentially

    Parameters
    ----------
    t2w_mni : Path
        T2w image in MNI space
    wm_mask_mni : Path
        White matter mask in MNI space (binary)
    output_mask : Path
        Output path for binary WMH mask
    output_labeled : Path
        Output path for labeled WMH map (each lesion has unique ID)
    sd_threshold : float
        Number of SDs above mean for threshold (default: 2.5)
    min_cluster_size : int
        Minimum voxel count for a valid lesion (default: 3)

    Returns
    -------
    dict
        Detection results including:
        - n_lesions: Number of detected lesions
        - total_volume_mm3: Total WMH volume
        - mean_wm_intensity: Mean T2w intensity in WM
        - sd_wm_intensity: SD of T2w intensity in WM
        - threshold: Intensity threshold used
        - voxel_volume_mm3: Volume of single voxel
    """
    t2w_mni = Path(t2w_mni)
    wm_mask_mni = Path(wm_mask_mni)
    output_mask = Path(output_mask)
    output_labeled = Path(output_labeled)

    logger.info(f"Loading T2w image: {t2w_mni}")
    t2w_img = nib.load(t2w_mni)
    t2w_data = t2w_img.get_fdata()

    logger.info(f"Loading WM mask: {wm_mask_mni}")
    wm_img = nib.load(wm_mask_mni)
    wm_data = wm_img.get_fdata()

    # Ensure WM mask is binary
    wm_mask = wm_data > 0.5

    # Calculate voxel volume in mm³
    voxel_dims = t2w_img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(voxel_dims))
    logger.info(f"Voxel dimensions: {voxel_dims}, volume: {voxel_volume_mm3:.4f} mm³")

    # Extract T2w intensities within WM mask
    wm_voxels = t2w_data[wm_mask]
    n_wm_voxels = len(wm_voxels)
    logger.info(f"White matter voxels: {n_wm_voxels}")

    if n_wm_voxels == 0:
        logger.warning("No white matter voxels found in mask")
        # Return empty results
        return _save_empty_results(
            t2w_img, output_mask, output_labeled, voxel_volume_mm3
        )

    # Compute intensity statistics within WM
    mean_intensity = float(np.mean(wm_voxels))
    sd_intensity = float(np.std(wm_voxels))
    threshold = mean_intensity + (sd_threshold * sd_intensity)

    logger.info(f"WM intensity - Mean: {mean_intensity:.2f}, SD: {sd_intensity:.2f}")
    logger.info(f"Threshold ({sd_threshold} SD): {threshold:.2f}")

    # Apply threshold within WM mask
    candidate_mask = (t2w_data > threshold) & wm_mask

    # Label connected components
    labeled_array, n_components = ndimage.label(candidate_mask)
    logger.info(f"Initial connected components: {n_components}")

    # Filter by minimum cluster size
    if n_components > 0:
        # Get component sizes
        component_sizes = ndimage.sum(
            candidate_mask, labeled_array, range(1, n_components + 1)
        )

        # Create mask of components to keep
        keep_mask = np.zeros_like(labeled_array, dtype=bool)
        for i, size in enumerate(component_sizes, 1):
            if size >= min_cluster_size:
                keep_mask |= (labeled_array == i)

        # Relabel remaining components sequentially
        final_labeled, n_lesions = ndimage.label(keep_mask)
    else:
        final_labeled = np.zeros_like(labeled_array)
        n_lesions = 0

    logger.info(f"Lesions after size filter (>= {min_cluster_size} voxels): {n_lesions}")

    # Create binary mask
    wmh_mask = final_labeled > 0

    # Calculate total volume
    total_voxels = int(np.sum(wmh_mask))
    total_volume_mm3 = total_voxels * voxel_volume_mm3

    logger.info(f"Total WMH volume: {total_volume_mm3:.2f} mm³ ({total_voxels} voxels)")

    # Save outputs
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    wmh_mask_img = nib.Nifti1Image(
        wmh_mask.astype(np.uint8), t2w_img.affine, t2w_img.header
    )
    nib.save(wmh_mask_img, output_mask)
    logger.info(f"Saved WMH mask: {output_mask}")

    wmh_labeled_img = nib.Nifti1Image(
        final_labeled.astype(np.int32), t2w_img.affine, t2w_img.header
    )
    nib.save(wmh_labeled_img, output_labeled)
    logger.info(f"Saved labeled WMH map: {output_labeled}")

    return {
        'n_lesions': n_lesions,
        'total_volume_mm3': total_volume_mm3,
        'total_voxels': total_voxels,
        'mean_wm_intensity': mean_intensity,
        'sd_wm_intensity': sd_intensity,
        'threshold': threshold,
        'sd_threshold': sd_threshold,
        'min_cluster_size': min_cluster_size,
        'voxel_volume_mm3': voxel_volume_mm3,
        'n_wm_voxels': n_wm_voxels,
        'output_mask': str(output_mask),
        'output_labeled': str(output_labeled)
    }


def _save_empty_results(
    reference_img: nib.Nifti1Image,
    output_mask: Path,
    output_labeled: Path,
    voxel_volume_mm3: float
) -> Dict:
    """Save empty WMH results when no white matter voxels found."""
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    empty_data = np.zeros(reference_img.shape[:3], dtype=np.uint8)

    empty_mask_img = nib.Nifti1Image(empty_data, reference_img.affine, reference_img.header)
    nib.save(empty_mask_img, output_mask)

    empty_labeled_img = nib.Nifti1Image(
        empty_data.astype(np.int32), reference_img.affine, reference_img.header
    )
    nib.save(empty_labeled_img, output_labeled)

    return {
        'n_lesions': 0,
        'total_volume_mm3': 0.0,
        'total_voxels': 0,
        'mean_wm_intensity': 0.0,
        'sd_wm_intensity': 0.0,
        'threshold': 0.0,
        'sd_threshold': 0.0,
        'min_cluster_size': 0,
        'voxel_volume_mm3': voxel_volume_mm3,
        'n_wm_voxels': 0,
        'output_mask': str(output_mask),
        'output_labeled': str(output_labeled)
    }


def compute_lesion_metrics(
    wmh_labeled: Path,
    t2w_mni: Path,
    voxel_volume_mm3: Optional[float] = None
) -> list:
    """
    Compute detailed metrics for each individual lesion.

    Parameters
    ----------
    wmh_labeled : Path
        Labeled WMH map (each lesion has unique ID)
    t2w_mni : Path
        T2w image in MNI space
    voxel_volume_mm3 : float, optional
        Volume of single voxel in mm³ (computed from header if not provided)

    Returns
    -------
    list of dict
        List of metrics for each lesion:
        - lesion_id: Unique lesion identifier
        - volume_mm3: Lesion volume in mm³
        - n_voxels: Number of voxels
        - centroid_x, centroid_y, centroid_z: Centroid coordinates (MNI mm)
        - mean_t2w_intensity: Mean T2w intensity within lesion
        - max_t2w_intensity: Maximum T2w intensity within lesion
    """
    wmh_labeled = Path(wmh_labeled)
    t2w_mni = Path(t2w_mni)

    labeled_img = nib.load(wmh_labeled)
    labeled_data = labeled_img.get_fdata().astype(np.int32)

    t2w_img = nib.load(t2w_mni)
    t2w_data = t2w_img.get_fdata()

    if voxel_volume_mm3 is None:
        voxel_dims = labeled_img.header.get_zooms()[:3]
        voxel_volume_mm3 = float(np.prod(voxel_dims))

    # Get affine for coordinate transformation
    affine = labeled_img.affine

    # Find unique lesion IDs (excluding 0 = background)
    lesion_ids = np.unique(labeled_data)
    lesion_ids = lesion_ids[lesion_ids > 0]

    lesion_metrics = []

    for lesion_id in lesion_ids:
        lesion_mask = labeled_data == lesion_id
        n_voxels = int(np.sum(lesion_mask))

        # Volume
        volume_mm3 = n_voxels * voxel_volume_mm3

        # Centroid in voxel coordinates
        coords = np.array(np.where(lesion_mask))
        centroid_voxel = coords.mean(axis=1)

        # Convert to MNI coordinates (mm)
        centroid_mni = nib.affines.apply_affine(
            affine, centroid_voxel
        )

        # T2w intensity within lesion
        lesion_intensities = t2w_data[lesion_mask]
        mean_intensity = float(np.mean(lesion_intensities))
        max_intensity = float(np.max(lesion_intensities))

        lesion_metrics.append({
            'lesion_id': int(lesion_id),
            'n_voxels': n_voxels,
            'volume_mm3': volume_mm3,
            'centroid_x': float(centroid_mni[0]),
            'centroid_y': float(centroid_mni[1]),
            'centroid_z': float(centroid_mni[2]),
            'mean_t2w_intensity': mean_intensity,
            'max_t2w_intensity': max_intensity
        })

    return lesion_metrics


def get_lesion_size_distribution(
    wmh_labeled: Path,
    voxel_volume_mm3: Optional[float] = None
) -> Dict:
    """
    Get distribution statistics for lesion sizes.

    Parameters
    ----------
    wmh_labeled : Path
        Labeled WMH map
    voxel_volume_mm3 : float, optional
        Volume of single voxel in mm³

    Returns
    -------
    dict
        Size distribution statistics:
        - n_lesions: Total number of lesions
        - mean_volume_mm3: Mean lesion volume
        - median_volume_mm3: Median lesion volume
        - std_volume_mm3: Standard deviation of volumes
        - min_volume_mm3: Smallest lesion
        - max_volume_mm3: Largest lesion
        - volumes_mm3: List of all lesion volumes
    """
    wmh_labeled = Path(wmh_labeled)

    labeled_img = nib.load(wmh_labeled)
    labeled_data = labeled_img.get_fdata().astype(np.int32)

    if voxel_volume_mm3 is None:
        voxel_dims = labeled_img.header.get_zooms()[:3]
        voxel_volume_mm3 = float(np.prod(voxel_dims))

    # Find unique lesion IDs
    lesion_ids = np.unique(labeled_data)
    lesion_ids = lesion_ids[lesion_ids > 0]

    if len(lesion_ids) == 0:
        return {
            'n_lesions': 0,
            'mean_volume_mm3': 0.0,
            'median_volume_mm3': 0.0,
            'std_volume_mm3': 0.0,
            'min_volume_mm3': 0.0,
            'max_volume_mm3': 0.0,
            'volumes_mm3': []
        }

    # Calculate volumes
    volumes = []
    for lesion_id in lesion_ids:
        n_voxels = np.sum(labeled_data == lesion_id)
        volumes.append(n_voxels * voxel_volume_mm3)

    volumes = np.array(volumes)

    return {
        'n_lesions': len(volumes),
        'mean_volume_mm3': float(np.mean(volumes)),
        'median_volume_mm3': float(np.median(volumes)),
        'std_volume_mm3': float(np.std(volumes)),
        'min_volume_mm3': float(np.min(volumes)),
        'max_volume_mm3': float(np.max(volumes)),
        'volumes_mm3': volumes.tolist()
    }
