#!/usr/bin/env python3
"""
White Matter Hyperintensity (WMH) Analysis Workflow

Detects and quantifies white matter hyperintensities using T1w+T2w combined analysis.
Uses intensity thresholding on T2w images within white matter masks.

Workflow:
1. T2w preprocessing: Co-register to T1w, normalize to MNI space
2. WMH detection: Threshold T2w at mean + X*SD within WM mask
3. Lesion labeling: Connected component analysis with minimum cluster size
4. Tract-wise analysis: Map lesions to JHU white matter atlas regions

Usage:
    # Process single subject
    python -m neurovrai.analysis.anat.wmh_workflow single \
        --subject IRC805-0580101 \
        --study-root /mnt/bytopia/IRC805 \
        --output-dir /mnt/bytopia/IRC805/hyperintensities

    # Batch process all subjects
    python -m neurovrai.analysis.anat.wmh_workflow batch \
        --study-root /mnt/bytopia/IRC805 \
        --output-dir /mnt/bytopia/IRC805/hyperintensities

    # Generate group report
    python -m neurovrai.analysis.anat.wmh_workflow report \
        --hyperintensities-dir /mnt/bytopia/IRC805/hyperintensities
"""

import nibabel as nib
import numpy as np
import pandas as pd
import subprocess
import json
import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from .wmh_detection import detect_wmh, compute_lesion_metrics, get_lesion_size_distribution

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def get_fsl_dir() -> Path:
    """Get FSL directory from environment."""
    import os
    fsl_dir = os.getenv('FSLDIR', '/usr/local/fsl')
    return Path(fsl_dir)


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO):
    """Configure logging for WMH analysis."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


# ============================================================================
# T2w Preprocessing Functions
# ============================================================================

def find_t2w_image(bids_dir: Path, subject: str) -> Optional[Path]:
    """
    Find the best T2w image for a subject from BIDS directory.

    Selection criteria (in order of preference):
    1. 3D T2W CS5 sequence (highest quality)
    2. T2W Sagittal Reformat
    3. Any other T2W file

    Parameters
    ----------
    bids_dir : Path
        BIDS directory root
    subject : str
        Subject identifier

    Returns
    -------
    Path or None
        Path to T2w image, or None if not found
    """
    anat_dir = bids_dir / subject / 'anat'
    if not anat_dir.exists():
        logger.warning(f"BIDS anat directory not found: {anat_dir}")
        return None

    # Look for T2w files
    t2w_files = list(anat_dir.glob('*T2W*.nii.gz')) + list(anat_dir.glob('*T2w*.nii.gz'))

    if not t2w_files:
        logger.warning(f"No T2w files found in {anat_dir}")
        return None

    # Prefer 3D T2W CS5 (highest quality)
    for f in t2w_files:
        if 'CS5' in f.name and 'Reformat' not in f.name:
            logger.info(f"Selected T2w (3D CS5): {f.name}")
            return f

    # Fall back to first non-reformat T2w
    for f in t2w_files:
        if 'Reformat' not in f.name:
            logger.info(f"Selected T2w: {f.name}")
            return f

    # Last resort: any T2w
    logger.info(f"Selected T2w (fallback): {t2w_files[0].name}")
    return t2w_files[0]


def find_t1w_brain(derivatives_dir: Path, subject: str) -> Optional[Path]:
    """Find T1w brain image from derivatives."""
    brain_dir = derivatives_dir / subject / 'anat' / 'brain'
    if brain_dir.exists():
        brain_files = list(brain_dir.glob('*_brain.nii.gz'))
        if brain_files:
            return brain_files[0]
    return None


def find_wm_segmentation(derivatives_dir: Path, subject: str) -> Optional[Path]:
    """Find white matter segmentation from derivatives."""
    seg_dir = derivatives_dir / subject / 'anat' / 'segmentation'
    if seg_dir.exists():
        # Check for symlink or POSTERIOR_03
        wm_symlink = seg_dir / 'wm.nii.gz'
        if wm_symlink.exists():
            return wm_symlink
        posterior = seg_dir / 'POSTERIOR_03.nii.gz'
        if posterior.exists():
            return posterior
    return None


def find_csf_segmentation(derivatives_dir: Path, subject: str) -> Optional[Path]:
    """Find CSF segmentation from derivatives."""
    seg_dir = derivatives_dir / subject / 'anat' / 'segmentation'
    if seg_dir.exists():
        csf_symlink = seg_dir / 'csf.nii.gz'
        if csf_symlink.exists():
            return csf_symlink
        posterior = seg_dir / 'POSTERIOR_01.nii.gz'
        if posterior.exists():
            return posterior
    return None


def find_gm_segmentation(derivatives_dir: Path, subject: str) -> Optional[Path]:
    """Find gray matter segmentation from derivatives."""
    seg_dir = derivatives_dir / subject / 'anat' / 'segmentation'
    if seg_dir.exists():
        gm_symlink = seg_dir / 'gm.nii.gz'
        if gm_symlink.exists():
            return gm_symlink
        posterior = seg_dir / 'POSTERIOR_02.nii.gz'
        if posterior.exists():
            return posterior
    return None


def find_t1w_mni_transform(transforms_dir: Path, subject: str) -> Optional[Path]:
    """Find T1w to MNI transform from centralized transforms directory."""
    subj_transforms = transforms_dir / subject
    if subj_transforms.exists():
        # Look for ANTs composite transform
        composite = subj_transforms / 't1w-mni-composite.h5'
        if composite.exists():
            return composite
    return None


def register_t2w_to_t1w(
    t2w_file: Path,
    t1w_brain: Path,
    output_file: Path,
    output_transform: Path
) -> Tuple[Path, Path]:
    """
    Co-register T2w image to T1w space using FSL FLIRT.

    Uses correlation ratio cost function (optimized for cross-modality).
    6 DOF rigid body registration.

    Parameters
    ----------
    t2w_file : Path
        Input T2w image
    t1w_brain : Path
        T1w brain (skull-stripped) as reference
    output_file : Path
        Registered T2w in T1w space
    output_transform : Path
        Affine transformation matrix (.mat)

    Returns
    -------
    tuple
        (registered_t2w, transform_matrix)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'flirt',
        '-in', str(t2w_file),
        '-ref', str(t1w_brain),
        '-out', str(output_file),
        '-omat', str(output_transform),
        '-dof', '6',
        '-cost', 'corratio',
        '-searchrx', '-90', '90',
        '-searchry', '-90', '90',
        '-searchrz', '-90', '90'
    ]

    logger.info(f"Running FLIRT: T2w -> T1w registration")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FLIRT failed: {result.stderr}")
        raise RuntimeError(f"FLIRT registration failed: {result.stderr}")

    logger.info(f"T2w registered to T1w: {output_file}")
    return output_file, output_transform


def normalize_to_mni(
    input_file: Path,
    t1w_to_mni_transform: Path,
    output_file: Path,
    reference: Optional[Path] = None,
    interpolation: str = 'Linear'
) -> Path:
    """
    Normalize image to MNI space using existing T1w->MNI transform.

    Uses ANTs antsApplyTransforms with the composite T1w->MNI transform.

    Parameters
    ----------
    input_file : Path
        Input image in T1w native space
    t1w_to_mni_transform : Path
        T1w->MNI composite transform (.h5 for ANTs)
    output_file : Path
        Output image in MNI space
    reference : Path, optional
        MNI reference template (default: MNI152_T1_2mm_brain)
    interpolation : str
        Interpolation method ('Linear', 'NearestNeighbor', etc.)

    Returns
    -------
    Path
        Normalized file
    """
    if reference is None:
        fsl_dir = get_fsl_dir()
        reference = fsl_dir / 'data' / 'standard' / 'MNI152_T1_2mm_brain.nii.gz'

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(input_file),
        '-r', str(reference),
        '-o', str(output_file),
        '-n', interpolation,
        '-t', str(t1w_to_mni_transform)
    ]

    logger.info(f"Running antsApplyTransforms: {input_file.name} -> MNI")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"antsApplyTransforms failed: {result.stderr}")
        raise RuntimeError(f"ANTs transform failed: {result.stderr}")

    logger.info(f"Normalized to MNI: {output_file}")
    return output_file


def create_wm_mask_mni(
    wm_prob_mni: Path,
    output_file: Path,
    threshold: float = 0.5
) -> Path:
    """
    Create binary white matter mask from probability map.

    Parameters
    ----------
    wm_prob_mni : Path
        White matter probability map in MNI space
    output_file : Path
        Output binary WM mask
    threshold : float
        Probability threshold for WM (default: 0.5)

    Returns
    -------
    Path
        Binary WM mask
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'fslmaths', str(wm_prob_mni),
        '-thr', str(threshold),
        '-bin', str(output_file)
    ]

    logger.info(f"Creating binary WM mask (threshold={threshold})")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"fslmaths failed: {result.stderr}")
        raise RuntimeError(f"fslmaths failed: {result.stderr}")

    return output_file


def create_clean_wm_mask_mni(
    wm_prob_mni: Path,
    csf_prob_mni: Path,
    gm_prob_mni: Path,
    output_file: Path,
    wm_threshold: float = 0.7,
    csf_exclude_threshold: float = 0.1,
    gm_exclude_threshold: float = 0.5,
    erode_iterations: int = 1,
    csf_dilate_iterations: int = 1,
    gm_dilate_iterations: int = 0
) -> Path:
    """
    Create a clean white matter mask excluding CSF and GM regions.

    This prevents false positive WMH detection in periventricular areas
    and gray matter. The mask is also eroded to ensure detection is
    confined to core white matter.

    Parameters
    ----------
    wm_prob_mni : Path
        White matter probability map in MNI space
    csf_prob_mni : Path
        CSF probability map in MNI space
    gm_prob_mni : Path
        Gray matter probability map in MNI space
    output_file : Path
        Output binary WM mask
    wm_threshold : float
        Probability threshold for WM inclusion (default: 0.7)
    csf_exclude_threshold : float
        CSF probability threshold for exclusion (default: 0.1)
    gm_exclude_threshold : float
        GM probability threshold for exclusion (default: 0.5)
    erode_iterations : int
        Number of erosion iterations for final WM mask (default: 1)
    csf_dilate_iterations : int
        Number of dilation iterations for CSF exclusion mask (default: 1).
        Creates a buffer zone around CSF to prevent periventricular false
        positives due to registration imperfections. Each iteration adds
        approximately one voxel width (~2mm in MNI 2mm space).
    gm_dilate_iterations : int
        Number of dilation iterations for GM exclusion mask (default: 0).
        Creates a buffer zone at the cortical boundary to prevent false
        positives at the GM/WM interface.

    Returns
    -------
    Path
        Clean binary WM mask
    """
    import tempfile

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Threshold WM at higher probability
        wm_mask = tmpdir / 'wm_mask.nii.gz'
        cmd = ['fslmaths', str(wm_prob_mni), '-thr', str(wm_threshold), '-bin', str(wm_mask)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"fslmaths WM threshold failed: {result.stderr}")

        # Step 2: Create CSF exclusion mask (areas with CSF > threshold)
        csf_mask_base = tmpdir / 'csf_mask_base.nii.gz'
        cmd = ['fslmaths', str(csf_prob_mni), '-thr', str(csf_exclude_threshold), '-bin', str(csf_mask_base)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"fslmaths CSF threshold failed: {result.stderr}")

        # Step 2b: Dilate CSF mask to create buffer zone around ventricles/sulci
        csf_mask = tmpdir / 'csf_mask.nii.gz'
        if csf_dilate_iterations > 0:
            cmd = ['fslmaths', str(csf_mask_base)]
            for _ in range(csf_dilate_iterations):
                cmd.append('-dilM')
            cmd.append(str(csf_mask))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"fslmaths CSF dilation failed: {result.stderr}")
        else:
            import shutil
            shutil.copy(str(csf_mask_base), str(csf_mask))

        # Step 3: Create GM exclusion mask (areas with GM > threshold)
        gm_mask_base = tmpdir / 'gm_mask_base.nii.gz'
        cmd = ['fslmaths', str(gm_prob_mni), '-thr', str(gm_exclude_threshold), '-bin', str(gm_mask_base)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"fslmaths GM threshold failed: {result.stderr}")

        # Step 3b: Dilate GM mask to create buffer zone at cortical boundary
        gm_mask = tmpdir / 'gm_mask.nii.gz'
        if gm_dilate_iterations > 0:
            cmd = ['fslmaths', str(gm_mask_base)]
            for _ in range(gm_dilate_iterations):
                cmd.append('-dilM')
            cmd.append(str(gm_mask))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"fslmaths GM dilation failed: {result.stderr}")
        else:
            import shutil
            shutil.copy(str(gm_mask_base), str(gm_mask))

        # Step 4: Combine exclusion masks
        exclusion_mask = tmpdir / 'exclusion_mask.nii.gz'
        cmd = ['fslmaths', str(csf_mask), '-add', str(gm_mask), '-bin', str(exclusion_mask)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"fslmaths combine exclusion failed: {result.stderr}")

        # Step 5: Subtract exclusion mask from WM mask
        clean_wm = tmpdir / 'clean_wm.nii.gz'
        cmd = ['fslmaths', str(wm_mask), '-sub', str(exclusion_mask), '-thr', '0', '-bin', str(clean_wm)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"fslmaths subtract exclusion failed: {result.stderr}")

        # Step 6: Optional erosion to stay away from boundaries
        if erode_iterations > 0:
            eroded_wm = tmpdir / 'eroded_wm.nii.gz'
            cmd = ['fslmaths', str(clean_wm), '-ero', str(eroded_wm)]
            for _ in range(erode_iterations - 1):
                cmd.extend(['-ero'])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"fslmaths erosion failed: {result.stderr}")
            final_mask = eroded_wm
        else:
            final_mask = clean_wm

        # Copy to output
        import shutil
        shutil.copy(str(final_mask), str(output_file))

    # Log mask statistics
    result = subprocess.run(['fslstats', str(output_file), '-V'], capture_output=True, text=True)
    if result.returncode == 0:
        voxels, volume = result.stdout.strip().split()
        logger.info(f"Clean WM mask: {voxels} voxels, {volume} mm³")
        logger.info(f"  WM threshold: {wm_threshold}, CSF exclude: {csf_exclude_threshold}, "
                   f"CSF dilate: {csf_dilate_iterations}, GM exclude: {gm_exclude_threshold}, "
                   f"GM dilate: {gm_dilate_iterations}, erosion: {erode_iterations}")

    return output_file


# ============================================================================
# JHU Atlas Functions
# ============================================================================

# JHU white matter atlas labels (FSL's JHU-ICBM-DTI-81) - 48 regions
JHU_LABELS = {
    0: "Background",
    1: "Middle cerebellar peduncle",
    2: "Pontine crossing tract",
    3: "Genu of corpus callosum",
    4: "Body of corpus callosum",
    5: "Splenium of corpus callosum",
    6: "Fornix (column and body of fornix)",
    7: "Corticospinal tract R",
    8: "Corticospinal tract L",
    9: "Medial lemniscus R",
    10: "Medial lemniscus L",
    11: "Inferior cerebellar peduncle R",
    12: "Inferior cerebellar peduncle L",
    13: "Superior cerebellar peduncle R",
    14: "Superior cerebellar peduncle L",
    15: "Cerebral peduncle R",
    16: "Cerebral peduncle L",
    17: "Anterior limb of internal capsule R",
    18: "Anterior limb of internal capsule L",
    19: "Posterior limb of internal capsule R",
    20: "Posterior limb of internal capsule L",
    21: "Retrolenticular part of internal capsule R",
    22: "Retrolenticular part of internal capsule L",
    23: "Anterior corona radiata R",
    24: "Anterior corona radiata L",
    25: "Superior corona radiata R",
    26: "Superior corona radiata L",
    27: "Posterior corona radiata R",
    28: "Posterior corona radiata L",
    29: "Posterior thalamic radiation R",
    30: "Posterior thalamic radiation L",
    31: "Sagittal stratum R",
    32: "Sagittal stratum L",
    33: "External capsule R",
    34: "External capsule L",
    35: "Cingulum (cingulate gyrus) R",
    36: "Cingulum (cingulate gyrus) L",
    37: "Cingulum (hippocampus) R",
    38: "Cingulum (hippocampus) L",
    39: "Fornix (cres) / Stria terminalis R",
    40: "Fornix (cres) / Stria terminalis L",
    41: "Superior longitudinal fasciculus R",
    42: "Superior longitudinal fasciculus L",
    43: "Superior fronto-occipital fasciculus R",
    44: "Superior fronto-occipital fasciculus L",
    45: "Uncinate fasciculus R",
    46: "Uncinate fasciculus L",
    47: "Tapetum R",
    48: "Tapetum L"
}

# JHU White Matter Tractography Atlas - 20 major tracts
JHU_TRACTS_LABELS = {
    0: "Background",
    1: "Anterior thalamic radiation L",
    2: "Anterior thalamic radiation R",
    3: "Corticospinal tract L",
    4: "Corticospinal tract R",
    5: "Cingulum (cingulate gyrus) L",
    6: "Cingulum (cingulate gyrus) R",
    7: "Cingulum (hippocampus) L",
    8: "Cingulum (hippocampus) R",
    9: "Forceps major (splenium of corpus callosum)",
    10: "Forceps minor (genu of corpus callosum)",
    11: "Inferior fronto-occipital fasciculus L",
    12: "Inferior fronto-occipital fasciculus R",
    13: "Inferior longitudinal fasciculus L",
    14: "Inferior longitudinal fasciculus R",
    15: "Superior longitudinal fasciculus L",
    16: "Superior longitudinal fasciculus R",
    17: "Uncinate fasciculus L",
    18: "Uncinate fasciculus R",
    19: "Superior longitudinal fasciculus (temporal part) L",
    20: "Superior longitudinal fasciculus (temporal part) R"
}


def load_jhu_atlas(use_combined: bool = True) -> Tuple[Optional[nib.Nifti1Image], Dict[int, str]]:
    """
    Load JHU white matter atlas from FSL.

    Parameters
    ----------
    use_combined : bool
        If True, combine tracts and labels atlases for best coverage.
        If False, use labels atlas only.

    Returns
    -------
    tuple
        (atlas_img, atlas_labels_dict)
    """
    fsl_dir = get_fsl_dir()

    # Use 2mm atlases (matches MNI152_T1_2mm)
    tracts_path = fsl_dir / 'data' / 'atlases' / 'JHU' / 'JHU-ICBM-tracts-maxprob-thr0-2mm.nii.gz'
    labels_path = fsl_dir / 'data' / 'atlases' / 'JHU' / 'JHU-ICBM-labels-2mm.nii.gz'

    if use_combined:
        if not tracts_path.exists() or not labels_path.exists():
            logger.warning("Cannot combine atlases - using labels only")
            if labels_path.exists():
                return nib.load(str(labels_path)), JHU_LABELS
            return None, {}

        logger.info("Loading combined JHU atlas (tracts + labels)")

        tracts_img = nib.load(str(tracts_path))
        labels_img = nib.load(str(labels_path))

        tracts_data = tracts_img.get_fdata()
        labels_data = labels_img.get_fdata()

        # Combine: tracts take priority, labels fill gaps with offset
        combined_data = tracts_data.copy()
        mask = tracts_data == 0
        combined_data[mask] = labels_data[mask] + 100  # Offset labels by 100

        combined_img = nib.Nifti1Image(combined_data, tracts_img.affine, tracts_img.header)

        # Create combined labels dictionary
        combined_labels = JHU_TRACTS_LABELS.copy()
        for idx, label in JHU_LABELS.items():
            if idx > 0:
                combined_labels[idx + 100] = label

        return combined_img, combined_labels

    else:
        if labels_path.exists():
            logger.info("Loading JHU labels atlas (48 regions)")
            return nib.load(str(labels_path)), JHU_LABELS
        return None, {}


def analyze_wmh_by_tract(
    wmh_labeled: Path,
    jhu_atlas: nib.Nifti1Image,
    jhu_labels: Dict[int, str],
    voxel_volume_mm3: float
) -> pd.DataFrame:
    """
    Analyze WMH distribution across JHU white matter tracts.

    For each tract:
    1. Count number of lesions overlapping with tract
    2. Calculate total WMH volume within tract (mm3)
    3. Compute percentage of tract affected

    Parameters
    ----------
    wmh_labeled : Path
        Labeled WMH map
    jhu_atlas : nib.Nifti1Image
        JHU combined atlas
    jhu_labels : Dict[int, str]
        Atlas label names
    voxel_volume_mm3 : float
        Volume of single voxel in mm3

    Returns
    -------
    pd.DataFrame
        Columns: tract_id, tract_name, n_lesions, wmh_volume_mm3,
                 tract_volume_mm3, pct_affected
    """
    wmh_labeled = Path(wmh_labeled)

    wmh_img = nib.load(wmh_labeled)
    wmh_data = wmh_img.get_fdata().astype(np.int32)

    atlas_data = jhu_atlas.get_fdata().astype(np.int32)

    # Check dimensions match
    if wmh_data.shape != atlas_data.shape:
        logger.warning(f"Shape mismatch: WMH {wmh_data.shape} vs Atlas {atlas_data.shape}")
        # Resample atlas to WMH space if needed
        from scipy.ndimage import zoom
        zoom_factors = np.array(wmh_data.shape) / np.array(atlas_data.shape)
        atlas_data = zoom(atlas_data, zoom_factors, order=0)  # Nearest neighbor

    # Get unique tract IDs (excluding background)
    tract_ids = np.unique(atlas_data)
    tract_ids = tract_ids[tract_ids > 0]

    results = []

    for tract_id in tract_ids:
        tract_name = jhu_labels.get(tract_id, f"Unknown ({tract_id})")
        tract_mask = atlas_data == tract_id
        tract_voxels = int(np.sum(tract_mask))

        if tract_voxels == 0:
            continue

        # Find lesions overlapping this tract
        overlapping_wmh = wmh_data * tract_mask
        unique_lesions = np.unique(overlapping_wmh)
        unique_lesions = unique_lesions[unique_lesions > 0]
        n_lesions = len(unique_lesions)

        # WMH volume within tract
        wmh_in_tract = int(np.sum((wmh_data > 0) & tract_mask))
        wmh_volume_mm3 = wmh_in_tract * voxel_volume_mm3

        # Tract volume and percentage affected
        tract_volume_mm3 = tract_voxels * voxel_volume_mm3
        pct_affected = (wmh_in_tract / tract_voxels) * 100 if tract_voxels > 0 else 0

        results.append({
            'tract_id': int(tract_id),
            'tract_name': tract_name,
            'n_lesions': n_lesions,
            'wmh_volume_mm3': wmh_volume_mm3,
            'wmh_voxels': wmh_in_tract,
            'tract_volume_mm3': tract_volume_mm3,
            'tract_voxels': tract_voxels,
            'pct_affected': pct_affected
        })

    df = pd.DataFrame(results)

    # Sort by WMH volume descending
    if not df.empty:
        df = df.sort_values('wmh_volume_mm3', ascending=False)

    return df


# ============================================================================
# Single Subject Pipeline
# ============================================================================

def run_wmh_analysis_single(
    subject: str,
    study_root: Path,
    output_dir: Path,
    sd_threshold: float = 3.0,
    min_cluster_size: int = 5,
    wm_threshold: float = 0.7,
    csf_exclude_threshold: float = 0.1,
    gm_exclude_threshold: float = 0.5,
    erode_wm_mask: int = 0,
    csf_dilate: int = 1,
    gm_dilate: int = 0,
    overwrite: bool = False
) -> Dict:
    """
    Run complete WMH analysis for a single subject.

    Steps:
    1. Find T2w image in BIDS directory
    2. Co-register T2w to T1w space
    3. Normalize T2w to MNI using existing transform
    4. Create clean WM mask in MNI space (excluding CSF and GM)
    5. Detect WMH using intensity thresholding
    6. Analyze by JHU white matter tract
    7. Save outputs and metrics

    Parameters
    ----------
    subject : str
        Subject identifier
    study_root : Path
        Study root directory
    output_dir : Path
        Output directory for WMH analysis
    sd_threshold : float
        SD threshold for detection (default: 3.0)
    min_cluster_size : int
        Minimum cluster size in voxels (default: 5)
    wm_threshold : float
        WM probability threshold for inclusion (default: 0.7)
    csf_exclude_threshold : float
        CSF probability threshold for exclusion (default: 0.1)
    gm_exclude_threshold : float
        GM probability threshold for exclusion (default: 0.5)
    erode_wm_mask : int
        Number of erosion iterations for WM mask (default: 0)
    csf_dilate : int
        Number of dilation iterations for CSF exclusion mask (default: 1).
        Creates a ~2mm buffer zone per iteration around CSF to prevent
        periventricular false positives.
    gm_dilate : int
        Number of dilation iterations for GM exclusion mask (default: 0).
        Creates a ~2mm buffer zone per iteration at cortical boundary to
        prevent false positives at the GM/WM interface.
    overwrite : bool
        Overwrite existing outputs (default: False)

    Returns
    -------
    dict
        Analysis results and output paths
    """
    study_root = Path(study_root)
    output_dir = Path(output_dir)

    subj_output = output_dir / subject
    subj_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"=" * 60)
    logger.info(f"Processing subject: {subject}")
    logger.info(f"=" * 60)

    # Check for existing outputs
    metrics_file = subj_output / 'wmh_metrics.json'
    if metrics_file.exists() and not overwrite:
        logger.info(f"Outputs exist for {subject}, skipping (use --overwrite to reprocess)")
        with open(metrics_file) as f:
            return json.load(f)

    # Locate input files
    bids_dir = study_root / 'bids'
    derivatives_dir = study_root / 'derivatives'
    transforms_dir = study_root / 'transforms'

    # Find T2w
    t2w_file = find_t2w_image(bids_dir, subject)
    if t2w_file is None:
        logger.error(f"No T2w image found for {subject}")
        return {'error': 'No T2w image found', 'subject': subject}

    # Find T1w brain
    t1w_brain = find_t1w_brain(derivatives_dir, subject)
    if t1w_brain is None:
        logger.error(f"No T1w brain found for {subject}")
        return {'error': 'No T1w brain found', 'subject': subject}

    # Find tissue segmentations (WM, CSF, GM)
    wm_seg = find_wm_segmentation(derivatives_dir, subject)
    if wm_seg is None:
        logger.error(f"No WM segmentation found for {subject}")
        return {'error': 'No WM segmentation found', 'subject': subject}

    csf_seg = find_csf_segmentation(derivatives_dir, subject)
    if csf_seg is None:
        logger.error(f"No CSF segmentation found for {subject}")
        return {'error': 'No CSF segmentation found', 'subject': subject}

    gm_seg = find_gm_segmentation(derivatives_dir, subject)
    if gm_seg is None:
        logger.error(f"No GM segmentation found for {subject}")
        return {'error': 'No GM segmentation found', 'subject': subject}

    # Find T1w->MNI transform
    t1w_mni_transform = find_t1w_mni_transform(transforms_dir, subject)
    if t1w_mni_transform is None:
        logger.error(f"No T1w->MNI transform found for {subject}")
        return {'error': 'No T1w->MNI transform found', 'subject': subject}

    logger.info(f"Input files:")
    logger.info(f"  T2w: {t2w_file}")
    logger.info(f"  T1w brain: {t1w_brain}")
    logger.info(f"  WM segmentation: {wm_seg}")
    logger.info(f"  CSF segmentation: {csf_seg}")
    logger.info(f"  GM segmentation: {gm_seg}")
    logger.info(f"  T1w->MNI transform: {t1w_mni_transform}")

    # Step 1: Register T2w to T1w
    t2w_t1w = subj_output / 't2w_t1w.nii.gz'
    t2w_t1w_mat = subj_output / 't2w_to_t1w.mat'
    register_t2w_to_t1w(t2w_file, t1w_brain, t2w_t1w, t2w_t1w_mat)

    # Step 2: Normalize T2w to MNI
    t2w_mni = subj_output / 't2w_mni.nii.gz'
    normalize_to_mni(t2w_t1w, t1w_mni_transform, t2w_mni)

    # Step 3: Normalize tissue probability maps to MNI
    wm_prob_mni = subj_output / 'wm_prob_mni.nii.gz'
    csf_prob_mni = subj_output / 'csf_prob_mni.nii.gz'
    gm_prob_mni = subj_output / 'gm_prob_mni.nii.gz'

    normalize_to_mni(wm_seg, t1w_mni_transform, wm_prob_mni)
    normalize_to_mni(csf_seg, t1w_mni_transform, csf_prob_mni)
    normalize_to_mni(gm_seg, t1w_mni_transform, gm_prob_mni)

    # Step 4: Create clean WM mask (excluding CSF and GM regions)
    wm_mask_mni = subj_output / 'wm_mask_mni.nii.gz'
    create_clean_wm_mask_mni(
        wm_prob_mni=wm_prob_mni,
        csf_prob_mni=csf_prob_mni,
        gm_prob_mni=gm_prob_mni,
        output_file=wm_mask_mni,
        wm_threshold=wm_threshold,
        csf_exclude_threshold=csf_exclude_threshold,
        gm_exclude_threshold=gm_exclude_threshold,
        erode_iterations=erode_wm_mask,
        csf_dilate_iterations=csf_dilate,
        gm_dilate_iterations=gm_dilate
    )

    # Step 5: Detect WMH
    wmh_mask = subj_output / 'wmh_mask.nii.gz'
    wmh_labeled = subj_output / 'wmh_labeled.nii.gz'

    detection_results = detect_wmh(
        t2w_mni=t2w_mni,
        wm_mask_mni=wm_mask_mni,
        output_mask=wmh_mask,
        output_labeled=wmh_labeled,
        sd_threshold=sd_threshold,
        min_cluster_size=min_cluster_size
    )

    # Step 5: Tract-wise analysis using JHU atlas
    jhu_atlas, jhu_labels = load_jhu_atlas(use_combined=True)

    if jhu_atlas is not None:
        tract_df = analyze_wmh_by_tract(
            wmh_labeled=wmh_labeled,
            jhu_atlas=jhu_atlas,
            jhu_labels=jhu_labels,
            voxel_volume_mm3=detection_results['voxel_volume_mm3']
        )
        tract_csv = subj_output / 'wmh_tract_counts.csv'
        tract_df.to_csv(tract_csv, index=False)
        logger.info(f"Saved tract analysis: {tract_csv}")
    else:
        logger.warning("JHU atlas not available, skipping tract analysis")
        tract_df = pd.DataFrame()

    # Step 6: Compute individual lesion metrics
    lesion_metrics = compute_lesion_metrics(
        wmh_labeled=wmh_labeled,
        t2w_mni=t2w_mni,
        voxel_volume_mm3=detection_results['voxel_volume_mm3']
    )
    lesion_df = pd.DataFrame(lesion_metrics)
    lesion_csv = subj_output / 'wmh_lesions.csv'
    lesion_df.to_csv(lesion_csv, index=False)
    logger.info(f"Saved lesion metrics: {lesion_csv}")

    # Step 7: Get size distribution
    size_dist = get_lesion_size_distribution(
        wmh_labeled=wmh_labeled,
        voxel_volume_mm3=detection_results['voxel_volume_mm3']
    )

    # Compile final results
    results = {
        'subject': subject,
        'timestamp': datetime.now().isoformat(),
        'detection_parameters': {
            'sd_threshold': sd_threshold,
            'min_cluster_size': min_cluster_size
        },
        'mask_parameters': {
            'wm_threshold': wm_threshold,
            'csf_exclude_threshold': csf_exclude_threshold,
            'csf_dilate_iterations': csf_dilate,
            'gm_exclude_threshold': gm_exclude_threshold,
            'gm_dilate_iterations': gm_dilate,
            'erode_iterations': erode_wm_mask
        },
        'wm_intensity_stats': {
            'mean': detection_results['mean_wm_intensity'],
            'std': detection_results['sd_wm_intensity'],
            'threshold': detection_results['threshold']
        },
        'wmh_summary': {
            'n_lesions': detection_results['n_lesions'],
            'total_volume_mm3': detection_results['total_volume_mm3'],
            'total_voxels': detection_results['total_voxels'],
            'voxel_volume_mm3': detection_results['voxel_volume_mm3']
        },
        'size_distribution': size_dist,
        'input_files': {
            't2w': str(t2w_file),
            't1w_brain': str(t1w_brain),
            'wm_segmentation': str(wm_seg),
            'csf_segmentation': str(csf_seg),
            'gm_segmentation': str(gm_seg),
            't1w_mni_transform': str(t1w_mni_transform)
        },
        'output_files': {
            't2w_t1w': str(t2w_t1w),
            't2w_mni': str(t2w_mni),
            'csf_prob_mni': str(csf_prob_mni),
            'gm_prob_mni': str(gm_prob_mni),
            'wm_mask_mni': str(wm_mask_mni),
            'wmh_mask': str(wmh_mask),
            'wmh_labeled': str(wmh_labeled),
            'tract_counts': str(subj_output / 'wmh_tract_counts.csv'),
            'lesion_metrics': str(lesion_csv)
        }
    }

    # Save metrics JSON
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics: {metrics_file}")

    logger.info(f"Completed {subject}: {detection_results['n_lesions']} lesions, "
                f"{detection_results['total_volume_mm3']:.2f} mm3 total volume")

    return results


# ============================================================================
# Batch Processing
# ============================================================================

def discover_subjects(study_root: Path) -> List[str]:
    """Discover subjects with required preprocessing outputs."""
    derivatives_dir = study_root / 'derivatives'

    subjects = []
    for subj_dir in sorted(derivatives_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        subject = subj_dir.name

        # Check for required outputs
        t1w_brain = find_t1w_brain(derivatives_dir, subject)
        wm_seg = find_wm_segmentation(derivatives_dir, subject)
        t1w_mni_transform = find_t1w_mni_transform(study_root / 'transforms', subject)

        if t1w_brain and wm_seg and t1w_mni_transform:
            subjects.append(subject)
        else:
            logger.debug(f"Skipping {subject}: missing prerequisites")

    return subjects


def _process_subject_wrapper(args):
    """Wrapper for parallel processing."""
    (subject, study_root, output_dir, sd_threshold, min_cluster_size,
     wm_threshold, csf_exclude_threshold, gm_exclude_threshold, erode_wm_mask,
     csf_dilate, gm_dilate, overwrite) = args
    try:
        return run_wmh_analysis_single(
            subject=subject,
            study_root=study_root,
            output_dir=output_dir,
            sd_threshold=sd_threshold,
            min_cluster_size=min_cluster_size,
            wm_threshold=wm_threshold,
            csf_exclude_threshold=csf_exclude_threshold,
            gm_exclude_threshold=gm_exclude_threshold,
            erode_wm_mask=erode_wm_mask,
            csf_dilate=csf_dilate,
            gm_dilate=gm_dilate,
            overwrite=overwrite
        )
    except Exception as e:
        logger.error(f"Error processing {subject}: {e}")
        return {'subject': subject, 'error': str(e)}


def run_wmh_analysis_batch(
    study_root: Path,
    output_dir: Path,
    subjects: Optional[List[str]] = None,
    sd_threshold: float = 3.0,
    min_cluster_size: int = 5,
    wm_threshold: float = 0.7,
    csf_exclude_threshold: float = 0.1,
    gm_exclude_threshold: float = 0.5,
    erode_wm_mask: int = 0,
    csf_dilate: int = 1,
    gm_dilate: int = 0,
    n_jobs: int = 4,
    overwrite: bool = False
) -> Dict:
    """
    Run WMH analysis for multiple subjects.

    Parameters
    ----------
    study_root : Path
        Study root directory
    output_dir : Path
        Output directory
    subjects : List[str], optional
        List of subject IDs (auto-discover if None)
    sd_threshold : float
        SD threshold for detection (default: 3.0)
    min_cluster_size : int
        Minimum cluster size (default: 5)
    wm_threshold : float
        WM probability threshold for inclusion (default: 0.7)
    csf_exclude_threshold : float
        CSF probability threshold for exclusion (default: 0.1)
    gm_exclude_threshold : float
        GM probability threshold for exclusion (default: 0.5)
    erode_wm_mask : int
        Number of erosion iterations for WM mask (default: 0)
    csf_dilate : int
        Number of dilation iterations for CSF exclusion mask (default: 1)
    gm_dilate : int
        Number of dilation iterations for GM exclusion mask (default: 0)
    n_jobs : int
        Number of parallel jobs
    overwrite : bool
        Overwrite existing outputs

    Returns
    -------
    dict
        Batch results summary
    """
    study_root = Path(study_root)
    output_dir = Path(output_dir)

    # Discover subjects if not provided
    if subjects is None:
        subjects = discover_subjects(study_root)
        logger.info(f"Discovered {len(subjects)} subjects with preprocessing")

    if not subjects:
        logger.error("No subjects found to process")
        return {'error': 'No subjects found'}

    logger.info(f"Processing {len(subjects)} subjects with {n_jobs} workers")
    logger.info(f"Parameters: SD threshold={sd_threshold}, min cluster={min_cluster_size}")
    logger.info(f"WM mask: WM threshold={wm_threshold}, CSF exclude={csf_exclude_threshold}, "
                f"CSF dilate={csf_dilate}, GM exclude={gm_exclude_threshold}, "
                f"GM dilate={gm_dilate}, erosion={erode_wm_mask}")

    # Process subjects
    results = []
    successful = 0
    failed = 0

    if n_jobs == 1:
        # Sequential processing
        for subject in subjects:
            result = run_wmh_analysis_single(
                subject=subject,
                study_root=study_root,
                output_dir=output_dir,
                sd_threshold=sd_threshold,
                min_cluster_size=min_cluster_size,
                wm_threshold=wm_threshold,
                csf_exclude_threshold=csf_exclude_threshold,
                gm_exclude_threshold=gm_exclude_threshold,
                erode_wm_mask=erode_wm_mask,
                csf_dilate=csf_dilate,
                gm_dilate=gm_dilate,
                overwrite=overwrite
            )
            results.append(result)
            if 'error' in result:
                failed += 1
            else:
                successful += 1
    else:
        # Parallel processing
        args_list = [
            (s, study_root, output_dir, sd_threshold, min_cluster_size,
             wm_threshold, csf_exclude_threshold, gm_exclude_threshold, erode_wm_mask,
             csf_dilate, gm_dilate, overwrite)
            for s in subjects
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_process_subject_wrapper, args): args[0]
                      for args in args_list}

            for future in as_completed(futures):
                subject = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if 'error' in result:
                        failed += 1
                    else:
                        successful += 1
                except Exception as e:
                    logger.error(f"Error processing {subject}: {e}")
                    results.append({'subject': subject, 'error': str(e)})
                    failed += 1

    logger.info(f"Batch complete: {successful} successful, {failed} failed")

    return {
        'total': len(subjects),
        'successful': successful,
        'failed': failed,
        'subjects': results
    }


# ============================================================================
# Group Summary and Reporting
# ============================================================================

def generate_group_summary(hyperintensities_dir: Path) -> pd.DataFrame:
    """
    Generate group-level summary statistics.

    Parameters
    ----------
    hyperintensities_dir : Path
        Root hyperintensities output directory

    Returns
    -------
    pd.DataFrame
        Group summary statistics
    """
    hyperintensities_dir = Path(hyperintensities_dir)
    group_dir = hyperintensities_dir / 'group'
    group_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for subj_dir in sorted(hyperintensities_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name in ['group', 'logs']:
            continue

        metrics_file = subj_dir / 'wmh_metrics.json'
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        if 'error' in metrics:
            continue

        wmh_summary = metrics.get('wmh_summary', {})
        size_dist = metrics.get('size_distribution', {})

        summaries.append({
            'subject': metrics['subject'],
            'n_lesions': wmh_summary.get('n_lesions', 0),
            'total_volume_mm3': wmh_summary.get('total_volume_mm3', 0),
            'mean_lesion_volume_mm3': size_dist.get('mean_volume_mm3', 0),
            'median_lesion_volume_mm3': size_dist.get('median_volume_mm3', 0),
            'max_lesion_volume_mm3': size_dist.get('max_volume_mm3', 0)
        })

    df = pd.DataFrame(summaries)

    # Save group summary
    summary_csv = group_dir / 'wmh_summary.csv'
    df.to_csv(summary_csv, index=False)
    logger.info(f"Saved group summary: {summary_csv}")

    # Also generate tract-wise summary across subjects
    tract_summaries = []
    for subj_dir in sorted(hyperintensities_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name in ['group', 'logs']:
            continue

        tract_csv = subj_dir / 'wmh_tract_counts.csv'
        if not tract_csv.exists():
            continue

        tract_df = pd.read_csv(tract_csv)
        tract_df['subject'] = subj_dir.name
        tract_summaries.append(tract_df)

    if tract_summaries:
        all_tracts_df = pd.concat(tract_summaries, ignore_index=True)

        # Aggregate by tract
        tract_agg = all_tracts_df.groupby('tract_name').agg({
            'n_lesions': ['sum', 'mean', 'std'],
            'wmh_volume_mm3': ['sum', 'mean', 'std'],
            'pct_affected': ['mean', 'std']
        }).round(2)
        tract_agg.columns = ['_'.join(col).strip() for col in tract_agg.columns.values]
        tract_agg = tract_agg.reset_index()

        tract_summary_csv = group_dir / 'wmh_by_tract.csv'
        tract_agg.to_csv(tract_summary_csv, index=False)
        logger.info(f"Saved tract summary: {tract_summary_csv}")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='White Matter Hyperintensity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Single subject command
    single_parser = subparsers.add_parser('single', help='Process single subject')
    single_parser.add_argument('--subject', required=True, help='Subject ID')
    single_parser.add_argument('--study-root', required=True, type=Path, help='Study root directory')
    single_parser.add_argument('--output-dir', type=Path, help='Output directory (default: {study-root}/hyperintensities)')
    single_parser.add_argument('--sd-threshold', type=float, default=3.0, help='SD threshold for detection (default: 3.0)')
    single_parser.add_argument('--min-cluster-size', type=int, default=5, help='Min cluster size in voxels (default: 5)')
    single_parser.add_argument('--wm-threshold', type=float, default=0.7, help='WM probability threshold (default: 0.7)')
    single_parser.add_argument('--csf-exclude', type=float, default=0.1, help='CSF probability threshold for exclusion (default: 0.1)')
    single_parser.add_argument('--gm-exclude', type=float, default=0.5, help='GM probability threshold for exclusion (default: 0.5)')
    single_parser.add_argument('--erode-wm', type=int, default=0, help='WM mask erosion iterations (default: 0)')
    single_parser.add_argument('--csf-dilate', type=int, default=1, help='CSF exclusion dilation iterations (default: 1). Creates ~2mm buffer zone per iteration.')
    single_parser.add_argument('--gm-dilate', type=int, default=0, help='GM exclusion dilation iterations (default: 0). Creates ~2mm buffer zone per iteration at cortical boundary.')
    single_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process all subjects')
    batch_parser.add_argument('--study-root', required=True, type=Path, help='Study root directory')
    batch_parser.add_argument('--output-dir', type=Path, help='Output directory (default: {study-root}/hyperintensities)')
    batch_parser.add_argument('--subjects', nargs='+', help='Specific subjects to process')
    batch_parser.add_argument('--sd-threshold', type=float, default=3.0, help='SD threshold for detection (default: 3.0)')
    batch_parser.add_argument('--min-cluster-size', type=int, default=5, help='Min cluster size in voxels (default: 5)')
    batch_parser.add_argument('--wm-threshold', type=float, default=0.7, help='WM probability threshold (default: 0.7)')
    batch_parser.add_argument('--csf-exclude', type=float, default=0.1, help='CSF probability threshold for exclusion (default: 0.1)')
    batch_parser.add_argument('--gm-exclude', type=float, default=0.5, help='GM probability threshold for exclusion (default: 0.5)')
    batch_parser.add_argument('--erode-wm', type=int, default=0, help='WM mask erosion iterations (default: 0)')
    batch_parser.add_argument('--csf-dilate', type=int, default=1, help='CSF exclusion dilation iterations (default: 1). Creates ~2mm buffer zone per iteration.')
    batch_parser.add_argument('--gm-dilate', type=int, default=0, help='GM exclusion dilation iterations (default: 0). Creates ~2mm buffer zone per iteration at cortical boundary.')
    batch_parser.add_argument('--n-jobs', type=int, default=4, help='Number of parallel jobs (default: 4)')
    batch_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate group report')
    report_parser.add_argument('--hyperintensities-dir', required=True, type=Path, help='Hyperintensities directory')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Setup logging
    if args.command in ['single', 'batch']:
        output_dir = args.output_dir or (args.study_root / 'hyperintensities')
        log_file = output_dir / 'logs' / f'wmh_analysis_{datetime.now():%Y%m%d_%H%M%S}.log'
        setup_logging(log_file)
    else:
        setup_logging()

    if args.command == 'single':
        output_dir = args.output_dir or (args.study_root / 'hyperintensities')
        run_wmh_analysis_single(
            subject=args.subject,
            study_root=args.study_root,
            output_dir=output_dir,
            sd_threshold=args.sd_threshold,
            min_cluster_size=args.min_cluster_size,
            wm_threshold=args.wm_threshold,
            csf_exclude_threshold=args.csf_exclude,
            gm_exclude_threshold=args.gm_exclude,
            erode_wm_mask=args.erode_wm,
            csf_dilate=args.csf_dilate,
            gm_dilate=args.gm_dilate,
            overwrite=args.overwrite
        )

    elif args.command == 'batch':
        output_dir = args.output_dir or (args.study_root / 'hyperintensities')
        run_wmh_analysis_batch(
            study_root=args.study_root,
            output_dir=output_dir,
            subjects=args.subjects,
            sd_threshold=args.sd_threshold,
            min_cluster_size=args.min_cluster_size,
            wm_threshold=args.wm_threshold,
            csf_exclude_threshold=args.csf_exclude,
            gm_exclude_threshold=args.gm_exclude,
            erode_wm_mask=args.erode_wm,
            csf_dilate=args.csf_dilate,
            gm_dilate=args.gm_dilate,
            n_jobs=args.n_jobs,
            overwrite=args.overwrite
        )

    elif args.command == 'report':
        generate_group_summary(args.hyperintensities_dir)


if __name__ == '__main__':
    main()
