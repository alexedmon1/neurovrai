#!/usr/bin/env python3
"""
ROI Extraction Module

Modality-agnostic ROI extraction from atlas parcellations. Supports extraction
from functional (4D timeseries), anatomical (3D volumes), and diffusion (3D/4D)
data.

Key Features:
- Load and validate atlas parcellations
- Extract ROI timeseries from 4D functional data
- Extract ROI statistics from 3D volumes (mean, median, std)
- Handle both integer-labeled and probabilistic atlases
- Optional mask-based filtering

Usage:
    # Functional timeseries extraction
    timeseries = extract_roi_timeseries(
        data_file='preprocessed_bold.nii.gz',
        atlas_file='schaefer_400.nii.gz',
        mask_file='brain_mask.nii.gz'
    )

    # Anatomical value extraction
    roi_values = extract_roi_values(
        data_file='FA.nii.gz',
        atlas_file='JHU-ICBM-labels-2mm.nii.gz',
        statistic='mean'
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class Atlas:
    """
    Atlas container for managing brain parcellations

    Attributes:
        data: Numpy array with atlas labels (3D integer or 4D probabilistic)
        affine: Affine transformation matrix
        labels: Dictionary mapping label integers to region names
        is_probabilistic: Whether atlas is probabilistic (4D) or discrete (3D)
    """

    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        labels: Optional[Dict[int, str]] = None,
        is_probabilistic: bool = False
    ):
        self.data = data
        self.affine = affine
        self.labels = labels or {}
        self.is_probabilistic = is_probabilistic

        # Validate
        if is_probabilistic and data.ndim != 4:
            raise ValueError("Probabilistic atlas must be 4D")
        if not is_probabilistic and data.ndim != 3:
            raise ValueError("Discrete atlas must be 3D")

    @property
    def n_rois(self) -> int:
        """Number of ROIs in atlas"""
        if self.is_probabilistic:
            return self.data.shape[3]
        else:
            # Exclude background (label 0)
            unique_labels = np.unique(self.data)
            return len(unique_labels[unique_labels > 0])

    @property
    def roi_indices(self) -> np.ndarray:
        """Array of ROI indices/labels"""
        if self.is_probabilistic:
            return np.arange(self.n_rois)
        else:
            unique_labels = np.unique(self.data)
            return unique_labels[unique_labels > 0]

    def get_roi_name(self, roi_idx: int) -> str:
        """Get name for ROI index"""
        roi_idx = int(roi_idx)  # Ensure integer
        return self.labels.get(roi_idx, f"ROI_{roi_idx:03d}")


def load_atlas(
    atlas_file: Union[str, Path],
    labels_file: Optional[Union[str, Path]] = None,
    is_probabilistic: bool = False
) -> Atlas:
    """
    Load atlas parcellation from file

    Args:
        atlas_file: Path to atlas NIfTI file
        labels_file: Optional path to labels text file (format: "index name")
        is_probabilistic: Whether atlas is probabilistic (4D) or discrete (3D)

    Returns:
        Atlas object

    Raises:
        FileNotFoundError: If atlas file doesn't exist
        ValueError: If atlas dimensionality doesn't match expected format
    """
    atlas_file = Path(atlas_file)

    if not atlas_file.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_file}")

    logger.info(f"Loading atlas: {atlas_file}")

    # Load atlas image
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    # Auto-detect probabilistic vs discrete
    if atlas_data.ndim == 4:
        is_probabilistic = True
        logger.info(f"  Detected 4D probabilistic atlas: {atlas_data.shape}")
    elif atlas_data.ndim == 3:
        is_probabilistic = False
        logger.info(f"  Detected 3D discrete atlas: {atlas_data.shape}")
    else:
        raise ValueError(f"Atlas must be 3D or 4D, got shape {atlas_data.shape}")

    # Load labels if provided
    labels = {}
    if labels_file is not None:
        labels_file = Path(labels_file)
        if labels_file.exists():
            logger.info(f"  Loading labels from: {labels_file}")
            with open(labels_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            idx, name = parts
                            labels[int(idx)] = name
            logger.info(f"  Loaded {len(labels)} region labels")

    # Create Atlas object
    atlas = Atlas(
        data=atlas_data,
        affine=affine,
        labels=labels,
        is_probabilistic=is_probabilistic
    )

    logger.info(f"  Atlas contains {atlas.n_rois} ROIs")

    return atlas


def _validate_spatial_match(
    data_img: nib.Nifti1Image,
    atlas: Atlas
) -> bool:
    """
    Validate that data and atlas have matching spatial dimensions

    Args:
        data_img: NIfTI image with data
        atlas: Atlas object

    Returns:
        True if dimensions match

    Raises:
        ValueError: If dimensions don't match
    """
    data_shape = data_img.shape[:3]  # Spatial dimensions only
    atlas_shape = atlas.data.shape[:3]

    if data_shape != atlas_shape:
        raise ValueError(
            f"Spatial dimensions mismatch: data={data_shape}, atlas={atlas_shape}. "
            "Consider resampling atlas to match data space."
        )

    return True


def extract_roi_timeseries(
    data_file: Union[str, Path],
    atlas: Union[Atlas, str, Path],
    mask_file: Optional[Union[str, Path]] = None,
    labels_file: Optional[Union[str, Path]] = None,
    min_voxels: int = 10,
    statistic: str = 'mean'
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract ROI timeseries from 4D functional data

    Args:
        data_file: Path to 4D functional data (e.g., preprocessed BOLD)
        atlas: Atlas object or path to atlas file
        mask_file: Optional brain mask to restrict extraction
        labels_file: Optional labels file if atlas is path
        min_voxels: Minimum voxels per ROI (ROIs with fewer are excluded)
        statistic: How to aggregate voxels in ROI ('mean', 'median', 'pca')

    Returns:
        Tuple of:
            - timeseries: Array of shape (n_timepoints, n_rois)
            - roi_names: List of ROI names corresponding to columns

    Raises:
        ValueError: If data is not 4D or dimensions don't match atlas
    """
    data_file = Path(data_file)

    logger.info(f"Extracting ROI timeseries from: {data_file}")

    # Load atlas if needed
    if not isinstance(atlas, Atlas):
        atlas = load_atlas(atlas, labels_file=labels_file)

    # Load functional data
    logger.info("  Loading functional data...")
    data_img = nib.load(data_file)
    data = data_img.get_fdata()

    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {data.shape}")

    n_timepoints = data.shape[3]
    logger.info(f"  Data shape: {data.shape} ({n_timepoints} timepoints)")

    # Validate spatial match
    _validate_spatial_match(data_img, atlas)

    # Load mask if provided
    mask = None
    if mask_file is not None:
        mask_file = Path(mask_file)
        if mask_file.exists():
            logger.info(f"  Loading mask: {mask_file}")
            mask_img = nib.load(mask_file)
            mask = mask_img.get_fdata() > 0

            # Validate mask dimensions
            if mask.shape != data.shape[:3]:
                raise ValueError(
                    f"Mask dimensions {mask.shape} don't match data {data.shape[:3]}"
                )

    # Extract timeseries for each ROI
    logger.info(f"  Extracting timeseries for {atlas.n_rois} ROIs...")

    timeseries_list = []
    roi_names = []

    for roi_idx in atlas.roi_indices:
        if atlas.is_probabilistic:
            # Probabilistic atlas: use probabilities as weights
            roi_mask = atlas.data[:, :, :, roi_idx] > 0.1  # Threshold at 10%
            weights = atlas.data[:, :, :, roi_idx][roi_mask]
        else:
            # Discrete atlas: binary mask
            roi_mask = atlas.data == roi_idx
            weights = None

        # Apply brain mask if provided
        if mask is not None:
            roi_mask = roi_mask & mask

        # Check minimum voxels
        n_voxels = np.sum(roi_mask)
        if n_voxels < min_voxels:
            logger.warning(
                f"  ROI {roi_idx} has only {n_voxels} voxels (< {min_voxels}), skipping"
            )
            continue

        # Extract timeseries
        roi_data = data[roi_mask, :]  # Shape: (n_voxels, n_timepoints)

        if statistic == 'mean':
            if weights is not None:
                # Weighted mean for probabilistic atlas
                roi_timeseries = np.average(roi_data, axis=0, weights=weights)
            else:
                roi_timeseries = np.mean(roi_data, axis=0)
        elif statistic == 'median':
            roi_timeseries = np.median(roi_data, axis=0)
        elif statistic == 'pca':
            # First principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            roi_timeseries = pca.fit_transform(roi_data.T).flatten()
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        timeseries_list.append(roi_timeseries)
        roi_names.append(atlas.get_roi_name(roi_idx))

    # Stack into array (n_timepoints, n_rois)
    timeseries = np.column_stack(timeseries_list)

    logger.info(f"  Extracted timeseries shape: {timeseries.shape}")
    logger.info(f"  Successfully extracted {len(roi_names)} ROIs")

    return timeseries, roi_names


def extract_roi_values(
    data_file: Union[str, Path],
    atlas: Union[Atlas, str, Path],
    mask_file: Optional[Union[str, Path]] = None,
    labels_file: Optional[Union[str, Path]] = None,
    min_voxels: int = 10,
    statistic: str = 'mean'
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Extract ROI statistics from 3D volume data

    Useful for extracting values from anatomical images (FA, MD, GM density, etc.)

    Args:
        data_file: Path to 3D volume data
        atlas: Atlas object or path to atlas file
        mask_file: Optional brain mask to restrict extraction
        labels_file: Optional labels file if atlas is path
        min_voxels: Minimum voxels per ROI (ROIs with fewer are excluded)
        statistic: How to aggregate voxels in ROI ('mean', 'median', 'std', 'max', 'min')

    Returns:
        Tuple of:
            - roi_values: Dictionary mapping ROI names to scalar values
            - roi_voxel_counts: Dictionary mapping ROI names to voxel counts

    Raises:
        ValueError: If data is not 3D or dimensions don't match atlas
    """
    data_file = Path(data_file)

    logger.info(f"Extracting ROI values from: {data_file}")

    # Load atlas if needed
    if not isinstance(atlas, Atlas):
        atlas = load_atlas(atlas, labels_file=labels_file)

    # Load volume data
    logger.info("  Loading volume data...")
    data_img = nib.load(data_file)
    data = data_img.get_fdata()

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape {data.shape}")

    logger.info(f"  Data shape: {data.shape}")

    # Validate spatial match
    _validate_spatial_match(data_img, atlas)

    # Load mask if provided
    mask = None
    if mask_file is not None:
        mask_file = Path(mask_file)
        if mask_file.exists():
            logger.info(f"  Loading mask: {mask_file}")
            mask_img = nib.load(mask_file)
            mask = mask_img.get_fdata() > 0

    # Extract values for each ROI
    logger.info(f"  Extracting values for {atlas.n_rois} ROIs...")

    roi_values = {}
    roi_voxel_counts = {}

    for roi_idx in atlas.roi_indices:
        if atlas.is_probabilistic:
            # Probabilistic atlas: use probabilities as weights
            roi_mask = atlas.data[:, :, :, roi_idx] > 0.1
            weights = atlas.data[:, :, :, roi_idx][roi_mask]
        else:
            # Discrete atlas: binary mask
            roi_mask = atlas.data == roi_idx
            weights = None

        # Apply brain mask if provided
        if mask is not None:
            roi_mask = roi_mask & mask

        # Check minimum voxels
        n_voxels = np.sum(roi_mask)
        if n_voxels < min_voxels:
            logger.warning(
                f"  ROI {roi_idx} has only {n_voxels} voxels (< {min_voxels}), skipping"
            )
            continue

        # Extract values
        roi_data = data[roi_mask]

        # Compute statistic
        if statistic == 'mean':
            if weights is not None:
                value = np.average(roi_data, weights=weights)
            else:
                value = np.mean(roi_data)
        elif statistic == 'median':
            value = np.median(roi_data)
        elif statistic == 'std':
            value = np.std(roi_data)
        elif statistic == 'max':
            value = np.max(roi_data)
        elif statistic == 'min':
            value = np.min(roi_data)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        roi_name = atlas.get_roi_name(roi_idx)
        roi_values[roi_name] = float(value)
        roi_voxel_counts[roi_name] = int(n_voxels)

    logger.info(f"  Successfully extracted {len(roi_values)} ROIs")

    return roi_values, roi_voxel_counts


def resample_atlas_to_data(
    atlas_file: Union[str, Path],
    reference_file: Union[str, Path],
    output_file: Union[str, Path],
    interpolation: str = 'nearest'
) -> Path:
    """
    Resample atlas to match data space

    Uses FSL's flirt with identity transformation for pure resampling.

    Args:
        atlas_file: Path to atlas in standard space
        reference_file: Path to data file defining target space
        output_file: Path for resampled atlas
        interpolation: Interpolation method ('nearest' for labels, 'trilinear' for probabilistic)

    Returns:
        Path to resampled atlas

    Note:
        Requires FSL to be installed
    """
    from nipype.interfaces import fsl

    atlas_file = Path(atlas_file)
    reference_file = Path(reference_file)
    output_file = Path(output_file)

    logger.info(f"Resampling atlas to match reference space")
    logger.info(f"  Atlas: {atlas_file}")
    logger.info(f"  Reference: {reference_file}")
    logger.info(f"  Output: {output_file}")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use FLIRT with identity matrix for pure resampling
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = str(atlas_file)
    flirt.inputs.reference = str(reference_file)
    flirt.inputs.out_file = str(output_file)
    flirt.inputs.apply_xfm = True
    flirt.inputs.uses_qform = True
    flirt.inputs.interp = interpolation

    logger.info("  Running FLIRT for resampling...")
    result = flirt.run()

    logger.info(f"  Resampled atlas saved to: {output_file}")

    return output_file
