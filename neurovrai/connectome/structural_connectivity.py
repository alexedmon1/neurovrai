#!/usr/bin/env python3
"""
Structural Connectivity Module

Probabilistic tractography-based structural connectivity analysis using FSL probtrackx2.

This module provides:
- BEDPOSTX integration for fiber orientation modeling
- Probtrackx2 wrapper for probabilistic tractography
- ROI-to-ROI structural connectivity matrices
- Network construction from tractography results
- Tractography quality control metrics

Requirements:
- Completed DWI preprocessing (eddy correction, DTI fitting)
- Atlas parcellation in DWI space (or transformation to apply)
- FSL installed with probtrackx2 and bedpostx available

Workflow:
    1. Run BEDPOSTX on preprocessed DWI data (fiber orientation modeling)
    2. Prepare atlas/ROI masks in DWI space
    3. Run probtrackx2 in network mode (ROI-to-ROI tractography)
    4. Construct connectivity matrix from tractography outputs
    5. Threshold and analyze resulting structural network

Usage:
    # Step 1: Run BEDPOSTX
    bedpostx_dir = run_bedpostx(
        dwi_dir='derivatives/subject/dwi/',
        n_fibers=2,
        n_jumps=1250
    )

    # Step 2: Compute structural connectivity matrix
    sc_results = compute_structural_connectivity(
        bedpostx_dir=bedpostx_dir,
        atlas_file='parcellations/schaefer_400_dwi.nii.gz',
        output_dir='connectome/structural/',
        n_samples=5000
    )

    # Access connectivity matrix
    sc_matrix = sc_results['connectivity_matrix']
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StructuralConnectivityError(Exception):
    """Raised when structural connectivity analysis fails"""
    pass


# FreeSurfer ventricle labels (aparc+aseg)
VENTRICLE_LABELS = [
    4,   # Left-Lateral-Ventricle
    5,   # Left-Inf-Lat-Vent
    14,  # 3rd-Ventricle
    15,  # 4th-Ventricle
    43,  # Right-Lateral-Ventricle
    44,  # Right-Inf-Lat-Vent
    72,  # 5th-Ventricle (if present)
]


def create_ventricle_mask(
    source: Path,
    output_file: Path,
    source_type: str = 'auto',
    reference: Optional[Path] = None
) -> Path:
    """
    Create a ventricle exclusion mask for tractography

    Args:
        source: Either FreeSurfer aparc+aseg or CSF probability map
        output_file: Output mask file path
        source_type: 'freesurfer', 'csf_pve', or 'auto' (detect from file)
        reference: Reference image for resampling (e.g., DWI)

    Returns:
        Path to ventricle mask
    """
    source = Path(source)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise StructuralConnectivityError(f"Source file not found: {source}")

    # Load source image
    src_img = nib.load(source)
    src_data = src_img.get_fdata()

    # Auto-detect source type
    if source_type == 'auto':
        unique_vals = np.unique(src_data[src_data > 0])
        if len(unique_vals) > 50:  # Likely segmentation with many labels
            source_type = 'freesurfer'
        elif np.max(src_data) <= 1.0:  # Probability map
            source_type = 'csf_pve'
        else:
            source_type = 'freesurfer'

    logger.info(f"Creating ventricle mask from {source_type} source")

    if source_type == 'freesurfer':
        # Extract ventricle labels from aparc+aseg
        mask_data = np.zeros_like(src_data, dtype=np.uint8)
        for label in VENTRICLE_LABELS:
            mask_data[src_data == label] = 1

        n_voxels = np.sum(mask_data)
        logger.info(f"  Ventricle mask: {n_voxels} voxels from {len(VENTRICLE_LABELS)} labels")

    elif source_type == 'csf_pve':
        # Threshold CSF probability map
        mask_data = (src_data > 0.5).astype(np.uint8)
        n_voxels = np.sum(mask_data)
        logger.info(f"  CSF mask: {n_voxels} voxels (threshold > 0.5)")

    else:
        raise StructuralConnectivityError(f"Unknown source type: {source_type}")

    # Save mask
    mask_img = nib.Nifti1Image(mask_data, src_img.affine, src_img.header)
    nib.save(mask_img, output_file)

    # Resample to reference if provided
    if reference is not None:
        reference = Path(reference)
        if reference.exists():
            resampled_file = output_file.parent / f"{output_file.stem}_resampled.nii.gz"

            # Use FLIRT to resample (nearest neighbor for mask)
            cmd = [
                'flirt',
                '-in', str(output_file),
                '-ref', str(reference),
                '-out', str(resampled_file),
                '-applyxfm',
                '-usesqform',
                '-interp', 'nearestneighbour'
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"  Resampled to reference: {resampled_file.name}")
                return resampled_file
            except subprocess.CalledProcessError as e:
                logger.warning(f"  Resampling failed, using original: {e}")

    return output_file


def find_ventricle_mask_source(
    subject: str,
    derivatives_dir: Path,
    fs_subjects_dir: Optional[Path] = None
) -> Tuple[Optional[Path], str]:
    """
    Find best available source for ventricle mask

    Priority:
    1. FreeSurfer aparc+aseg (most accurate)
    2. CSF segmentation from anatomical preprocessing

    Args:
        subject: Subject ID
        derivatives_dir: Path to derivatives directory
        fs_subjects_dir: FreeSurfer subjects directory

    Returns:
        Tuple of (source_path, source_type) or (None, None) if not found
    """
    # Try FreeSurfer aparc+aseg first
    if fs_subjects_dir is not None:
        aparc_aseg = fs_subjects_dir / subject / 'mri' / 'aparc+aseg.mgz'
        if aparc_aseg.exists():
            return aparc_aseg, 'freesurfer'

    # Try CSF from anatomical preprocessing
    csf_paths = [
        derivatives_dir / subject / 'anat' / 'segmentation' / 'pve_0.nii.gz',
        derivatives_dir / subject / 'anat' / 'tissue_pve_0.nii.gz',
        derivatives_dir / subject / 'anat' / 'csf_pve.nii.gz',
    ]

    for csf_path in csf_paths:
        if csf_path.exists():
            return csf_path, 'csf_pve'

    return None, None


def check_fsl_installation() -> Tuple[bool, bool, bool]:
    """
    Check if FSL tools are available

    Returns:
        Tuple of (bedpostx_available, probtrackx2_available, probtrackx2_gpu_available)
    """
    try:
        bedpostx_result = subprocess.run(
            ['which', 'bedpostx'],
            capture_output=True,
            text=True
        )
        bedpostx_available = bedpostx_result.returncode == 0
    except Exception:
        bedpostx_available = False

    try:
        probtrackx_result = subprocess.run(
            ['which', 'probtrackx2'],
            capture_output=True,
            text=True
        )
        probtrackx_available = probtrackx_result.returncode == 0
    except Exception:
        probtrackx_available = False

    try:
        probtrackx_gpu_result = subprocess.run(
            ['which', 'probtrackx2_gpu'],
            capture_output=True,
            text=True
        )
        probtrackx_gpu_available = probtrackx_gpu_result.returncode == 0
    except Exception:
        probtrackx_gpu_available = False

    return bedpostx_available, probtrackx_available, probtrackx_gpu_available


# FreeSurfer white matter labels (for ACT-style WM constraint)
WM_LABELS = [
    2,    # Left-Cerebral-White-Matter
    41,   # Right-Cerebral-White-Matter
    77,   # WM-hypointensities
    251,  # CC_Posterior
    252,  # CC_Mid_Posterior
    253,  # CC_Central
    254,  # CC_Mid_Anterior
    255,  # CC_Anterior
]

# FreeSurfer subcortical gray matter labels
SUBCORTICAL_GM_LABELS = {
    'Left-Thalamus': 10,
    'Right-Thalamus': 49,
    'Left-Caudate': 11,
    'Right-Caudate': 50,
    'Left-Putamen': 12,
    'Right-Putamen': 51,
    'Left-Pallidum': 13,
    'Right-Pallidum': 52,
    'Left-Hippocampus': 17,
    'Right-Hippocampus': 53,
    'Left-Amygdala': 18,
    'Right-Amygdala': 54,
    'Left-Accumbens': 26,
    'Right-Accumbens': 58,
    'Brain-Stem': 16,
}


def create_wm_mask(
    source: Path,
    output_file: Path,
    source_type: str = 'auto',
    reference: Optional[Path] = None,
    include_cc: bool = True
) -> Path:
    """
    Create a white matter inclusion mask for ACT-style tractography

    Args:
        source: Either FreeSurfer aparc+aseg or WM probability map (pve_2)
        output_file: Output mask file path
        source_type: 'freesurfer', 'fsl_pve', or 'auto' (detect from file)
        reference: Reference image for resampling (e.g., DWI)
        include_cc: Include corpus callosum in WM mask (default: True)

    Returns:
        Path to white matter mask
    """
    source = Path(source)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise StructuralConnectivityError(f"Source file not found: {source}")

    src_img = nib.load(source)
    src_data = src_img.get_fdata()

    # Auto-detect source type
    if source_type == 'auto':
        unique_vals = np.unique(src_data[src_data > 0])
        if len(unique_vals) > 50:
            source_type = 'freesurfer'
        elif np.max(src_data) <= 1.0:
            source_type = 'fsl_pve'
        else:
            source_type = 'freesurfer'

    logger.info(f"Creating WM mask from {source_type} source")

    if source_type == 'freesurfer':
        mask_data = np.zeros_like(src_data, dtype=np.uint8)

        # Add cerebral white matter
        for label in WM_LABELS[:2]:  # Left and Right WM
            mask_data[src_data == label] = 1

        # Optionally add corpus callosum
        if include_cc:
            for label in WM_LABELS[3:]:  # CC labels
                mask_data[src_data == label] = 1

        n_voxels = np.sum(mask_data)
        logger.info(f"  WM mask: {n_voxels} voxels from FreeSurfer labels")

    elif source_type == 'fsl_pve':
        # Threshold WM probability map
        mask_data = (src_data > 0.5).astype(np.uint8)
        n_voxels = np.sum(mask_data)
        logger.info(f"  WM mask: {n_voxels} voxels (threshold > 0.5)")

    else:
        raise StructuralConnectivityError(f"Unknown source type: {source_type}")

    # Save mask
    mask_img = nib.Nifti1Image(mask_data, src_img.affine, src_img.header)
    nib.save(mask_img, output_file)

    # Resample to reference if provided
    if reference is not None:
        reference = Path(reference)
        if reference.exists():
            resampled_file = output_file.parent / f"{output_file.stem}_resampled.nii.gz"
            cmd = [
                'flirt',
                '-in', str(output_file),
                '-ref', str(reference),
                '-out', str(resampled_file),
                '-applyxfm', '-usesqform',
                '-interp', 'nearestneighbour'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"  Resampled to reference: {resampled_file}")
            return resampled_file

    return output_file


def find_wm_mask_source(
    subject: str,
    derivatives_dir: Path,
    fs_subjects_dir: Optional[Path] = None,
    prefer_freesurfer: bool = True
) -> Tuple[Optional[Path], str]:
    """
    Find best available source for white matter mask

    Args:
        subject: Subject ID
        derivatives_dir: Path to derivatives directory
        fs_subjects_dir: Optional FreeSurfer subjects directory
        prefer_freesurfer: Prefer FreeSurfer if available (default: True)

    Returns:
        Tuple of (source_path, source_type) or (None, None) if not found
    """
    derivatives_dir = Path(derivatives_dir)

    # FreeSurfer sources (preferred for precise boundaries)
    if prefer_freesurfer and fs_subjects_dir is not None:
        fs_subjects_dir = Path(fs_subjects_dir)
        aparc_paths = [
            fs_subjects_dir / subject / 'mri' / 'aparc+aseg.mgz',
            fs_subjects_dir / subject / 'mri' / 'aparc+aseg.nii.gz',
        ]
        for aparc_path in aparc_paths:
            if aparc_path.exists():
                return aparc_path, 'freesurfer'

    # FSL FAST WM probability map
    wm_paths = [
        derivatives_dir / subject / 'anat' / 'segmentation' / 'pve_2.nii.gz',
        derivatives_dir / subject / 'anat' / 'tissue_pve_2.nii.gz',
        derivatives_dir / subject / 'anat' / 'wm_pve.nii.gz',
    ]

    for wm_path in wm_paths:
        if wm_path.exists():
            return wm_path, 'fsl_pve'

    return None, None


def create_gmwmi_mask(
    fs_subject_dir: Path,
    output_file: Path,
    method: str = 'volume',
    reference: Optional[Path] = None
) -> Path:
    """
    Create gray-white matter interface (GMWMI) mask for seeding tractography

    The GMWMI is the boundary between cortical gray matter and white matter,
    providing anatomically precise seed locations for cortical connectivity.

    Args:
        fs_subject_dir: FreeSurfer subject directory
        output_file: Output mask file path
        method: 'volume' (from aparc+aseg) or 'surface' (from lh/rh.white)
        reference: Reference image for resampling (e.g., DWI)

    Returns:
        Path to GMWMI mask
    """
    fs_subject_dir = Path(fs_subject_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if method == 'volume':
        # Create GMWMI from aparc+aseg (boundary detection)
        aparc_file = fs_subject_dir / 'mri' / 'aparc+aseg.mgz'
        if not aparc_file.exists():
            aparc_file = fs_subject_dir / 'mri' / 'aparc+aseg.nii.gz'

        if not aparc_file.exists():
            raise StructuralConnectivityError(f"aparc+aseg not found in {fs_subject_dir}")

        logger.info(f"Creating GMWMI mask from aparc+aseg (volume method)")

        # Load aparc+aseg
        aparc_img = nib.load(aparc_file)
        aparc_data = aparc_img.get_fdata().astype(int)

        # Create WM mask (labels 2 and 41)
        wm_mask = np.isin(aparc_data, [2, 41])

        # Create cortical GM mask (labels 1000-2035 for aparc)
        # These are the cortical parcellation labels
        gm_mask = (aparc_data >= 1000) & (aparc_data <= 2035)

        # Also include subcortical GM for subcortical connectivity
        subcortical_labels = list(SUBCORTICAL_GM_LABELS.values())
        subcortical_mask = np.isin(aparc_data, subcortical_labels)
        gm_mask = gm_mask | subcortical_mask

        # Find boundary: GM voxels adjacent to WM voxels
        from scipy import ndimage

        # Dilate WM mask by 1 voxel
        wm_dilated = ndimage.binary_dilation(wm_mask, iterations=1)

        # GMWMI = GM voxels that are in dilated WM region
        gmwmi_data = (gm_mask & wm_dilated).astype(np.uint8)

        n_voxels = np.sum(gmwmi_data)
        logger.info(f"  GMWMI mask: {n_voxels} voxels at GM-WM boundary")

        # Save mask
        gmwmi_img = nib.Nifti1Image(gmwmi_data, aparc_img.affine, aparc_img.header)
        nib.save(gmwmi_img, output_file)

    elif method == 'surface':
        # Create GMWMI from FreeSurfer surfaces
        # This requires mri_surf2vol from FreeSurfer
        logger.info(f"Creating GMWMI mask from surfaces (surface method)")

        lh_white = fs_subject_dir / 'surf' / 'lh.white'
        rh_white = fs_subject_dir / 'surf' / 'rh.white'
        orig_mgz = fs_subject_dir / 'mri' / 'orig.mgz'

        if not all(p.exists() for p in [lh_white, rh_white, orig_mgz]):
            raise StructuralConnectivityError(
                f"FreeSurfer surface files not found in {fs_subject_dir}"
            )

        # Create temporary volume for each hemisphere
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            lh_vol = Path(tmpdir) / 'lh_white.nii.gz'
            rh_vol = Path(tmpdir) / 'rh_white.nii.gz'

            # Convert left hemisphere surface to volume
            cmd_lh = [
                'mri_surf2vol',
                '--hemi', 'lh',
                '--surf', 'white',
                '--o', str(lh_vol),
                '--template', str(orig_mgz),
                '--sd', str(fs_subject_dir.parent),
                '--identity', fs_subject_dir.name
            ]

            # Convert right hemisphere surface to volume
            cmd_rh = [
                'mri_surf2vol',
                '--hemi', 'rh',
                '--surf', 'white',
                '--o', str(rh_vol),
                '--template', str(orig_mgz),
                '--sd', str(fs_subject_dir.parent),
                '--identity', fs_subject_dir.name
            ]

            try:
                subprocess.run(cmd_lh, check=True, capture_output=True)
                subprocess.run(cmd_rh, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"mri_surf2vol failed, falling back to volume method: {e}")
                return create_gmwmi_mask(fs_subject_dir, output_file, method='volume', reference=reference)

            # Combine hemispheres
            lh_img = nib.load(lh_vol)
            rh_img = nib.load(rh_vol)

            gmwmi_data = ((lh_img.get_fdata() > 0) | (rh_img.get_fdata() > 0)).astype(np.uint8)

            n_voxels = np.sum(gmwmi_data)
            logger.info(f"  GMWMI mask: {n_voxels} voxels from surfaces")

            gmwmi_img = nib.Nifti1Image(gmwmi_data, lh_img.affine, lh_img.header)
            nib.save(gmwmi_img, output_file)

    else:
        raise StructuralConnectivityError(f"Unknown GMWMI method: {method}")

    # Resample to reference if provided
    if reference is not None:
        reference = Path(reference)
        if reference.exists():
            resampled_file = output_file.parent / f"{output_file.stem}_resampled.nii.gz"
            cmd = [
                'flirt',
                '-in', str(output_file),
                '-ref', str(reference),
                '-out', str(resampled_file),
                '-applyxfm', '-usesqform',
                '-interp', 'nearestneighbour'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"  Resampled to reference: {resampled_file}")
            return resampled_file

    return output_file


def create_gm_termination_mask(
    source: Path,
    output_file: Path,
    source_type: str = 'auto',
    reference: Optional[Path] = None
) -> Path:
    """
    Create gray matter termination mask for ACT-style tractography

    Streamlines terminate when entering gray matter, preventing
    anatomically implausible paths through cortex.

    Args:
        source: Either FreeSurfer aparc+aseg or GM probability map (pve_1)
        output_file: Output mask file path
        source_type: 'freesurfer', 'fsl_pve', or 'auto'
        reference: Reference image for resampling

    Returns:
        Path to GM termination mask
    """
    source = Path(source)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise StructuralConnectivityError(f"Source file not found: {source}")

    src_img = nib.load(source)
    src_data = src_img.get_fdata()

    # Auto-detect source type
    if source_type == 'auto':
        unique_vals = np.unique(src_data[src_data > 0])
        if len(unique_vals) > 50:
            source_type = 'freesurfer'
        elif np.max(src_data) <= 1.0:
            source_type = 'fsl_pve'
        else:
            source_type = 'freesurfer'

    logger.info(f"Creating GM termination mask from {source_type} source")

    if source_type == 'freesurfer':
        mask_data = np.zeros_like(src_data, dtype=np.uint8)

        # Cortical GM (aparc labels 1000-2035)
        cortical_gm = (src_data >= 1000) & (src_data <= 2035)
        mask_data[cortical_gm] = 1

        # Subcortical GM
        for label in SUBCORTICAL_GM_LABELS.values():
            mask_data[src_data == label] = 1

        n_voxels = np.sum(mask_data)
        logger.info(f"  GM mask: {n_voxels} voxels from FreeSurfer labels")

    elif source_type == 'fsl_pve':
        mask_data = (src_data > 0.5).astype(np.uint8)
        n_voxels = np.sum(mask_data)
        logger.info(f"  GM mask: {n_voxels} voxels (threshold > 0.5)")

    else:
        raise StructuralConnectivityError(f"Unknown source type: {source_type}")

    mask_img = nib.Nifti1Image(mask_data, src_img.affine, src_img.header)
    nib.save(mask_img, output_file)

    if reference is not None:
        reference = Path(reference)
        if reference.exists():
            resampled_file = output_file.parent / f"{output_file.stem}_resampled.nii.gz"
            cmd = [
                'flirt',
                '-in', str(output_file),
                '-ref', str(reference),
                '-out', str(resampled_file),
                '-applyxfm', '-usesqform',
                '-interp', 'nearestneighbour'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return resampled_file

    return output_file


def create_subcortical_waypoint_mask(
    fs_subject_dir: Path,
    output_file: Path,
    structures: Optional[List[str]] = None,
    reference: Optional[Path] = None
) -> Path:
    """
    Create subcortical waypoint mask from FreeSurfer parcellation

    Useful for ensuring connectivity passes through key relay structures
    like the thalamus.

    Args:
        fs_subject_dir: FreeSurfer subject directory
        output_file: Output mask file path
        structures: List of structure names (default: bilateral thalamus)
        reference: Reference image for resampling

    Returns:
        Path to subcortical waypoint mask
    """
    fs_subject_dir = Path(fs_subject_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if structures is None:
        structures = ['Left-Thalamus', 'Right-Thalamus']

    # Get labels for requested structures
    labels = []
    for struct in structures:
        if struct in SUBCORTICAL_GM_LABELS:
            labels.append(SUBCORTICAL_GM_LABELS[struct])
        else:
            logger.warning(f"Unknown subcortical structure: {struct}")

    if not labels:
        raise StructuralConnectivityError("No valid subcortical structures specified")

    # Load aparc+aseg
    aparc_file = fs_subject_dir / 'mri' / 'aparc+aseg.mgz'
    if not aparc_file.exists():
        aparc_file = fs_subject_dir / 'mri' / 'aparc+aseg.nii.gz'

    if not aparc_file.exists():
        raise StructuralConnectivityError(f"aparc+aseg not found in {fs_subject_dir}")

    logger.info(f"Creating subcortical waypoint mask for: {structures}")

    aparc_img = nib.load(aparc_file)
    aparc_data = aparc_img.get_fdata().astype(int)

    mask_data = np.isin(aparc_data, labels).astype(np.uint8)

    n_voxels = np.sum(mask_data)
    logger.info(f"  Subcortical mask: {n_voxels} voxels")

    mask_img = nib.Nifti1Image(mask_data, aparc_img.affine, aparc_img.header)
    nib.save(mask_img, output_file)

    if reference is not None:
        reference = Path(reference)
        if reference.exists():
            resampled_file = output_file.parent / f"{output_file.stem}_resampled.nii.gz"
            cmd = [
                'flirt',
                '-in', str(output_file),
                '-ref', str(reference),
                '-out', str(resampled_file),
                '-applyxfm', '-usesqform',
                '-interp', 'nearestneighbour'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return resampled_file

    return output_file


def get_tractography_config(config: Optional[Dict] = None) -> Dict:
    """
    Get tractography configuration with defaults

    Args:
        config: Optional config dictionary (from YAML)

    Returns:
        Dictionary with all tractography settings
    """
    defaults = {
        'tractography': {
            'use_gpu': True,
            'n_samples': 5000,
            'step_length': 0.5,
            'curvature_threshold': 0.2,
            'loop_check': True,
        },
        'anatomical_constraints': {
            'avoid_ventricles': True,
            'use_wm_mask': True,
            'terminate_at_gm': False,
            'wm_source': 'auto',
        },
        'freesurfer_options': {
            'use_gmwmi_seeding': False,
            'gmwmi_method': 'volume',
            'use_subcortical_waypoints': False,
            'subcortical_structures': ['Left-Thalamus', 'Right-Thalamus'],
        },
        'output': {
            'normalize': True,
            'threshold': None,
            'compute_graph_metrics': True,
        }
    }

    if config is None:
        return defaults

    # Get structural_connectivity section from config
    sc_config = config.get('structural_connectivity', {})

    # Merge with defaults
    result = defaults.copy()
    for section in ['tractography', 'anatomical_constraints', 'freesurfer_options', 'output']:
        if section in sc_config:
            result[section].update(sc_config[section])

    return result


def validate_bedpostx_inputs(
    dwi_dir: Path
) -> Dict[str, Path]:
    """
    Validate that required BEDPOSTX input files exist

    Args:
        dwi_dir: Directory containing preprocessed DWI data

    Returns:
        Dictionary with paths to required files

    Raises:
        StructuralConnectivityError: If required files are missing
    """
    dwi_dir = Path(dwi_dir)

    if not dwi_dir.exists():
        raise StructuralConnectivityError(f"DWI directory not found: {dwi_dir}")

    # Required files for BEDPOSTX
    required_files = {
        'data': 'data.nii.gz',
        'nodif_brain_mask': 'nodif_brain_mask.nii.gz',
        'bvals': 'bvals',
        'bvecs': 'bvecs'
    }

    files = {}
    missing = []

    for key, filename in required_files.items():
        filepath = dwi_dir / filename
        if filepath.exists():
            files[key] = filepath
        else:
            missing.append(filename)

    if missing:
        raise StructuralConnectivityError(
            f"Missing required BEDPOSTX files in {dwi_dir}:\n"
            f"  {', '.join(missing)}\n\n"
            f"Required files: {', '.join(required_files.values())}"
        )

    return files


def run_bedpostx(
    dwi_dir: Path,
    output_dir: Optional[Path] = None,
    n_fibers: int = 2,
    n_jumps: int = 1250,
    burn_in: int = 1000,
    sample_every: int = 25,
    use_gpu: bool = False,
    force: bool = False
) -> Path:
    """
    Run BEDPOSTX for fiber orientation modeling

    BEDPOSTX performs Bayesian Estimation of Diffusion Parameters Obtained
    using Sampling Techniques, modeling crossing fibers in each voxel.

    Args:
        dwi_dir: Directory containing preprocessed DWI data with files:
            - data.nii.gz: Eddy-corrected DWI data
            - nodif_brain_mask.nii.gz: Brain mask
            - bvals: b-values
            - bvecs: b-vectors (eddy-rotated)
        output_dir: Output directory (default: {dwi_dir}.bedpostX)
        n_fibers: Number of fibers to model per voxel (default: 2)
        n_jumps: Number of MCMC jumps (default: 1250)
        burn_in: Burn-in period (default: 1000)
        sample_every: Sample every N iterations (default: 25)
        use_gpu: Use GPU acceleration if available (default: False)
        force: Overwrite existing BEDPOSTX output (default: False)

    Returns:
        Path to BEDPOSTX output directory

    Raises:
        StructuralConnectivityError: If BEDPOSTX fails or inputs invalid

    Note:
        BEDPOSTX can take several hours to complete. Monitor progress in
        {output_dir}/logs/. Use use_gpu=True for significant speedup if
        CUDA-enabled GPU is available.
    """
    bedpostx_available, _ = check_fsl_installation()
    if not bedpostx_available:
        raise StructuralConnectivityError(
            "bedpostx not found. Ensure FSL is installed and $FSLDIR is set."
        )

    dwi_dir = Path(dwi_dir)

    # Validate inputs
    input_files = validate_bedpostx_inputs(dwi_dir)

    # Determine output directory
    if output_dir is None:
        output_dir = dwi_dir.parent / f"{dwi_dir.name}.bedpostX"
    else:
        output_dir = Path(output_dir)

    # Check if already completed
    if output_dir.exists() and not force:
        # Check for completion marker or key output files
        dyads_file = output_dir / "dyads1.nii.gz"
        mean_f1_file = output_dir / "mean_f1samples.nii.gz"

        if dyads_file.exists() and mean_f1_file.exists():
            logger.info(f"BEDPOSTX output already exists: {output_dir}")
            logger.info("Use force=True to rerun")
            return output_dir
        else:
            logger.warning(f"Incomplete BEDPOSTX output found, will rerun")

    logger.info("=" * 80)
    logger.info("Running BEDPOSTX")
    logger.info("=" * 80)
    logger.info(f"Input directory: {dwi_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of fibers: {n_fibers}")
    logger.info(f"MCMC jumps: {n_jumps}")
    logger.info(f"Burn-in: {burn_in}")
    logger.info(f"GPU acceleration: {use_gpu}")

    # Build bedpostx command
    if use_gpu:
        cmd = ['bedpostx_gpu', str(dwi_dir)]
    else:
        cmd = ['bedpostx', str(dwi_dir)]

    # BEDPOSTX reads parameters from environment or uses defaults
    # We'll create a temporary options file
    env = {}
    if output_dir != dwi_dir.parent / f"{dwi_dir.name}.bedpostX":
        logger.warning(
            "Custom output_dir requires manual setup. "
            "BEDPOSTX creates {input_dir}.bedpostX by default."
        )

    # Execute BEDPOSTX
    start_time = time.time()
    log_file = dwi_dir.parent / "bedpostx.log"

    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info("This may take several hours. Monitor progress in logs/")

        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600

        logger.info(f"✓ BEDPOSTX completed in {elapsed_hours:.1f} hours")

    except subprocess.CalledProcessError as e:
        logger.error(f"BEDPOSTX failed with exit code {e.returncode}")
        logger.error(f"Check log file: {log_file}")
        raise StructuralConnectivityError(f"BEDPOSTX execution failed: {e}")

    # Verify outputs
    expected_output_dir = dwi_dir.parent / f"{dwi_dir.name}.bedpostX"
    if not expected_output_dir.exists():
        raise StructuralConnectivityError(
            f"BEDPOSTX output directory not created: {expected_output_dir}"
        )

    # Check key output files
    dyads_file = expected_output_dir / "dyads1.nii.gz"
    if not dyads_file.exists():
        raise StructuralConnectivityError(
            f"BEDPOSTX output incomplete. Missing: {dyads_file}"
        )

    logger.info(f"✓ BEDPOSTX outputs validated: {expected_output_dir}")

    return expected_output_dir


def validate_bedpostx_outputs(
    bedpostx_dir: Path,
    n_fibers: int = 2
) -> Dict[str, Path]:
    """
    Validate BEDPOSTX outputs exist and are complete

    Args:
        bedpostx_dir: Path to BEDPOSTX output directory
        n_fibers: Expected number of fiber compartments

    Returns:
        Dictionary with paths to key output files

    Raises:
        StructuralConnectivityError: If outputs are incomplete
    """
    bedpostx_dir = Path(bedpostx_dir)

    if not bedpostx_dir.exists():
        raise StructuralConnectivityError(
            f"BEDPOSTX directory not found: {bedpostx_dir}"
        )

    # Check for required files
    outputs = {
        'merged': bedpostx_dir / 'merged',
        'nodif_brain_mask': bedpostx_dir / 'nodif_brain_mask.nii.gz'
    }

    # Check fiber orientation files for each compartment
    for i in range(1, n_fibers + 1):
        outputs[f'dyads{i}'] = bedpostx_dir / f'dyads{i}.nii.gz'
        outputs[f'mean_f{i}samples'] = bedpostx_dir / f'mean_f{i}samples.nii.gz'
        outputs[f'mean_th{i}samples'] = bedpostx_dir / f'mean_th{i}samples.nii.gz'
        outputs[f'mean_ph{i}samples'] = bedpostx_dir / f'mean_ph{i}samples.nii.gz'

    missing = []
    for key, filepath in outputs.items():
        if not filepath.exists():
            missing.append(filepath.name)

    if missing:
        raise StructuralConnectivityError(
            f"Incomplete BEDPOSTX outputs in {bedpostx_dir}:\n"
            f"  Missing: {', '.join(missing)}\n\n"
            f"BEDPOSTX may still be running. Check logs/ directory."
        )

    return outputs


def prepare_atlas_for_probtrackx(
    atlas_file: Path,
    output_dir: Path,
    min_voxels_per_roi: int = 10
) -> Tuple[Path, List[str]]:
    """
    Prepare atlas for probtrackx2 network mode

    Creates individual ROI masks and coordinate list required by probtrackx2.

    Args:
        atlas_file: Path to atlas parcellation (integer labels)
        output_dir: Output directory for ROI masks
        min_voxels_per_roi: Minimum voxels per ROI (exclude smaller ROIs)

    Returns:
        Tuple of (seeds_list_file, roi_names)
        - seeds_list_file: Text file with paths to individual ROI masks
        - roi_names: List of ROI names/labels

    Raises:
        StructuralConnectivityError: If atlas is invalid
    """
    atlas_file = Path(atlas_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not atlas_file.exists():
        raise StructuralConnectivityError(f"Atlas file not found: {atlas_file}")

    logger.info(f"Preparing atlas for probtrackx2: {atlas_file.name}")

    # Load atlas
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata().astype(int)

    # Get unique ROI labels (exclude 0 = background)
    roi_labels = np.unique(atlas_data)
    roi_labels = roi_labels[roi_labels > 0]

    logger.info(f"  Found {len(roi_labels)} ROIs in atlas")

    # Create individual ROI masks
    roi_masks = []
    roi_names = []
    valid_labels = []

    for label in roi_labels:
        roi_mask = (atlas_data == label).astype(np.uint8)
        n_voxels = np.sum(roi_mask)

        if n_voxels < min_voxels_per_roi:
            logger.warning(f"  ROI {label}: only {n_voxels} voxels, excluding")
            continue

        # Save individual ROI mask
        roi_file = output_dir / f"roi_{label:03d}.nii.gz"
        roi_img = nib.Nifti1Image(roi_mask, atlas_img.affine, atlas_img.header)
        nib.save(roi_img, roi_file)

        roi_masks.append(roi_file)
        roi_names.append(f"ROI_{label:03d}")
        valid_labels.append(label)

        logger.info(f"  ROI {label}: {n_voxels} voxels -> {roi_file.name}")

    if len(roi_masks) == 0:
        raise StructuralConnectivityError(
            f"No valid ROIs found in atlas. Check min_voxels_per_roi parameter."
        )

    # Create seeds list file (required by probtrackx2 --network option)
    seeds_list_file = output_dir / "seeds.txt"
    with open(seeds_list_file, 'w') as f:
        for roi_file in roi_masks:
            f.write(f"{roi_file}\n")

    logger.info(f"✓ Created {len(roi_masks)} ROI masks")
    logger.info(f"✓ Seeds list: {seeds_list_file}")

    # Save ROI names for later use
    roi_names_file = output_dir / "roi_names.txt"
    with open(roi_names_file, 'w') as f:
        for name in roi_names:
            f.write(f"{name}\n")

    return seeds_list_file, roi_names


def run_probtrackx2_network(
    bedpostx_dir: Path,
    seeds_list: Path,
    output_dir: Path,
    n_samples: int = 5000,
    step_length: float = 0.5,
    curvature_threshold: float = 0.2,
    loop_check: bool = True,
    distance_correct: bool = True,
    waypoint_mask: Optional[Path] = None,
    exclusion_mask: Optional[Path] = None,
    termination_mask: Optional[Path] = None,
    use_gpu: bool = True
) -> Dict[str, Path]:
    """
    Run probtrackx2 in network mode for ROI-to-ROI tractography

    Args:
        bedpostx_dir: Path to BEDPOSTX output directory
        seeds_list: Path to seeds list file (from prepare_atlas_for_probtrackx)
        output_dir: Output directory for tractography results
        n_samples: Number of samples per seed voxel (default: 5000)
        step_length: Step length in mm (default: 0.5)
        curvature_threshold: Curvature threshold, 0-1 (default: 0.2)
        loop_check: Discard looping paths (default: True)
        distance_correct: Apply distance correction (default: True)
        waypoint_mask: Optional waypoint mask (e.g., white matter)
        exclusion_mask: Optional exclusion mask (e.g., CSF)
        termination_mask: Optional termination mask (e.g., grey matter)
        use_gpu: Use probtrackx2_gpu if available (default: True)

    Returns:
        Dictionary with paths to output files

    Raises:
        StructuralConnectivityError: If probtrackx2 fails

    Note:
        Network mode runs tractography from each seed ROI to all target ROIs,
        creating an NxN connectivity matrix. This can take several hours for
        large atlases (e.g., Schaefer 400 parcellation). GPU version is
        significantly faster (10-100x speedup).
    """
    _, probtrackx_available, probtrackx_gpu_available = check_fsl_installation()
    if not probtrackx_available:
        raise StructuralConnectivityError(
            "probtrackx2 not found. Ensure FSL is installed and $FSLDIR is set."
        )

    # Determine which executable to use
    if use_gpu and probtrackx_gpu_available:
        probtrackx_cmd = 'probtrackx2_gpu'
        logger.info("Using GPU-accelerated probtrackx2_gpu")
    else:
        probtrackx_cmd = 'probtrackx2'
        if use_gpu and not probtrackx_gpu_available:
            logger.warning("GPU requested but probtrackx2_gpu not available, falling back to CPU")

    bedpostx_dir = Path(bedpostx_dir)
    seeds_list = Path(seeds_list)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate BEDPOSTX outputs
    bedpostx_outputs = validate_bedpostx_outputs(bedpostx_dir)

    # Validate seeds list
    if not seeds_list.exists():
        raise StructuralConnectivityError(f"Seeds list not found: {seeds_list}")

    # Count number of ROIs
    with open(seeds_list, 'r') as f:
        n_rois = len([line for line in f if line.strip()])

    logger.info("=" * 80)
    logger.info(f"Running {probtrackx_cmd} (Network Mode)")
    logger.info("=" * 80)
    logger.info(f"BEDPOSTX directory: {bedpostx_dir}")
    logger.info(f"Seeds list: {seeds_list}")
    logger.info(f"Number of ROIs: {n_rois}")
    logger.info(f"Samples per voxel: {n_samples}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"GPU acceleration: {use_gpu and probtrackx_gpu_available}")

    # Build probtrackx2 command
    # Note: GPU version requires = syntax and different option names
    # GPU version limitations:
    #   - --omatrix1 and --omatrix3 cannot be run simultaneously
    #   - --omatrix2 requires --target2 parameter
    cmd = [
        probtrackx_cmd,
        f'--samples={str(bedpostx_outputs["merged"])}',
        f'--mask={str(bedpostx_outputs["nodif_brain_mask"])}',
        f'--seed={str(seeds_list)}',
        '--network',  # Enable network mode
        f'--targetmasks={str(seeds_list)}',  # GPU requires explicit targets (same as seeds for network mode)
        f'--dir={str(output_dir)}',
        f'--nsamples={str(n_samples)}',
        f'--steplength={str(step_length)}',
        f'--cthr={str(curvature_threshold)}',  # GPU uses --cthr not --curvthresh
        # Note: --opd removed - causes GPU memory issues with large atlases
        '--forcedir',  # Overwrite output directory
        '--os2t',  # Output seeds to targets
        '--omatrix1',  # Output connectivity matrix (fdt_network_matrix)
        # Note: --omatrix2 removed - GPU requires --target2 which we don't need
        # Note: --omatrix3 removed - GPU incompatible with --omatrix1
        # Note: --ompl removed - causes GPU memory issues with large atlases
    ]

    # Optional parameters
    if loop_check:
        cmd.append('--loopcheck')

    if distance_correct:
        cmd.append('--distthresh=0.0')  # Enable distance correction

    if waypoint_mask is not None:
        if not Path(waypoint_mask).exists():
            raise StructuralConnectivityError(f"Waypoint mask not found: {waypoint_mask}")
        cmd.append(f'--waypoints={str(waypoint_mask)}')

    if exclusion_mask is not None:
        if not Path(exclusion_mask).exists():
            raise StructuralConnectivityError(f"Exclusion mask not found: {exclusion_mask}")
        cmd.append(f'--avoid={str(exclusion_mask)}')

    if termination_mask is not None:
        if not Path(termination_mask).exists():
            raise StructuralConnectivityError(f"Termination mask not found: {termination_mask}")
        cmd.append(f'--stop={str(termination_mask)}')

    # Execute probtrackx2
    start_time = time.time()
    log_file = output_dir / "probtrackx2.log"

    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info(f"This may take several hours for {n_rois} ROIs...")

        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600

        logger.info(f"✓ Probtrackx2 completed in {elapsed_hours:.2f} hours")

    except subprocess.CalledProcessError as e:
        logger.error(f"Probtrackx2 failed with exit code {e.returncode}")
        logger.error(f"Check log file: {log_file}")
        raise StructuralConnectivityError(f"Probtrackx2 execution failed: {e}")

    # Collect output files
    output_files = {
        'log': log_file,
        'fdt_network_matrix': output_dir / 'fdt_network_matrix',
        'waytotal': output_dir / 'waytotal',
        'fdt_paths': []
    }

    # Check for network matrix (key output)
    if not output_files['fdt_network_matrix'].exists():
        raise StructuralConnectivityError(
            f"Probtrackx2 output incomplete. Missing: fdt_network_matrix"
        )

    # Find individual seed-to-target path files
    for seed_dir in sorted(output_dir.glob('seeds_to_*')):
        output_files['fdt_paths'].append(seed_dir)

    logger.info(f"✓ Network matrix: {output_files['fdt_network_matrix']}")
    logger.info(f"✓ Found {len(output_files['fdt_paths'])} seed directories")

    return output_files


def construct_connectivity_matrix(
    probtrackx_output_dir: Path,
    roi_names: List[str],
    normalize: bool = True,
    threshold: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Construct structural connectivity matrix from probtrackx2 outputs

    Args:
        probtrackx_output_dir: Path to probtrackx2 output directory
        roi_names: List of ROI names (must match order in seeds list)
        normalize: Normalize by waytotal (default: True)
        threshold: Optional threshold for weak connections (0-1)

    Returns:
        Dictionary containing:
            - connectivity_matrix: Structural connectivity matrix (n_rois, n_rois)
            - waytotal: Number of successful samples per seed
            - connectivity_matrix_raw: Unnormalized matrix

    Raises:
        StructuralConnectivityError: If outputs are invalid
    """
    probtrackx_output_dir = Path(probtrackx_output_dir)

    # Load fdt_network_matrix (FSL format: space-separated)
    matrix_file = probtrackx_output_dir / 'fdt_network_matrix'
    if not matrix_file.exists():
        raise StructuralConnectivityError(
            f"Network matrix not found: {matrix_file}"
        )

    logger.info(f"Loading connectivity matrix: {matrix_file}")

    # Read matrix
    try:
        connectivity_raw = np.loadtxt(matrix_file)
    except Exception as e:
        raise StructuralConnectivityError(f"Failed to load network matrix: {e}")

    n_rois = len(roi_names)
    if connectivity_raw.shape != (n_rois, n_rois):
        raise StructuralConnectivityError(
            f"Matrix shape {connectivity_raw.shape} doesn't match "
            f"number of ROIs {n_rois}"
        )

    logger.info(f"  Matrix shape: {connectivity_raw.shape}")
    logger.info(f"  Total connections: {np.sum(connectivity_raw > 0)}")

    # Load waytotal (number of successful samples from each seed)
    waytotal_file = probtrackx_output_dir / 'waytotal'
    if waytotal_file.exists():
        waytotal = np.loadtxt(waytotal_file)
        logger.info(f"  Loaded waytotal: {len(waytotal)} seeds")
    else:
        logger.warning("waytotal file not found, normalization unavailable")
        waytotal = None
        normalize = False

    # Normalize by waytotal if requested
    if normalize and waytotal is not None:
        connectivity_norm = connectivity_raw.copy()
        for i in range(n_rois):
            if waytotal[i] > 0:
                connectivity_norm[i, :] /= waytotal[i]
            else:
                logger.warning(f"ROI {roi_names[i]}: waytotal = 0")

        logger.info("  Normalized by waytotal")
    else:
        connectivity_norm = connectivity_raw.copy()

    # Apply threshold if specified
    if threshold is not None:
        connections_before = np.sum(connectivity_norm > 0)
        connectivity_norm[connectivity_norm < threshold] = 0
        connections_after = np.sum(connectivity_norm > 0)

        logger.info(f"  Threshold: {threshold}")
        logger.info(f"  Connections: {connections_before} → {connections_after}")

    # Make symmetric (average i->j and j->i)
    connectivity_symmetric = (connectivity_norm + connectivity_norm.T) / 2

    logger.info(f"  Final connections: {np.sum(connectivity_symmetric > 0)}")
    logger.info(f"  Connection density: {np.sum(connectivity_symmetric > 0) / (n_rois * (n_rois - 1)):.3f}")

    return {
        'connectivity_matrix': connectivity_symmetric,
        'connectivity_matrix_raw': connectivity_raw,
        'connectivity_matrix_normalized': connectivity_norm,
        'waytotal': waytotal,
        'roi_names': roi_names
    }


def compute_structural_connectivity(
    bedpostx_dir: Path,
    atlas_file: Path,
    output_dir: Path,
    n_samples: int = 5000,
    step_length: float = 0.5,
    curvature_threshold: float = 0.2,
    normalize: bool = True,
    threshold: Optional[float] = None,
    min_voxels_per_roi: int = 10,
    waypoint_mask: Optional[Path] = None,
    exclusion_mask: Optional[Path] = None,
    avoid_ventricles: bool = True,
    ventricle_mask: Optional[Path] = None,
    subject: Optional[str] = None,
    derivatives_dir: Optional[Path] = None,
    fs_subjects_dir: Optional[Path] = None,
    use_gpu: bool = True,
    use_wm_mask: bool = True,
    wm_mask: Optional[Path] = None,
    terminate_at_gm: bool = False,
    gm_mask: Optional[Path] = None,
    use_gmwmi_seeding: bool = False,
    gmwmi_method: str = 'volume',
    config: Optional[Dict] = None
) -> Dict:
    """
    Complete workflow: Compute structural connectivity matrix using probtrackx2

    This is the main function that orchestrates the full structural connectivity
    analysis pipeline with support for anatomical constraints and GPU acceleration.

    Args:
        bedpostx_dir: Path to BEDPOSTX output directory
        atlas_file: Path to atlas parcellation in DWI space
        output_dir: Output directory for all results
        n_samples: Number of samples per seed voxel (default: 5000)
        step_length: Tractography step length in mm (default: 0.5)
        curvature_threshold: Curvature threshold 0-1 (default: 0.2)
        normalize: Normalize by waytotal (default: True)
        threshold: Optional threshold for weak connections (default: None)
        min_voxels_per_roi: Minimum voxels per ROI (default: 10)
        waypoint_mask: Optional waypoint mask (e.g., white matter)
        exclusion_mask: Optional exclusion mask (e.g., CSF) - overrides ventricle mask
        avoid_ventricles: Exclude streamlines through ventricles (default: True)
        ventricle_mask: Pre-computed ventricle mask (auto-generated if None)
        subject: Subject ID (for auto-finding ventricle source)
        derivatives_dir: Derivatives directory (for auto-finding ventricle source)
        fs_subjects_dir: FreeSurfer subjects directory (for masks from aparc+aseg)
        use_gpu: Use probtrackx2_gpu if available (default: True)
        use_wm_mask: Constrain tractography to white matter - ACT style (default: True)
        wm_mask: Pre-computed WM mask (auto-generated if None)
        terminate_at_gm: Stop streamlines when entering gray matter (default: False)
        gm_mask: Pre-computed GM mask for termination (auto-generated if None)
        use_gmwmi_seeding: Seed from gray-white matter interface (default: False)
        gmwmi_method: GMWMI creation method - 'volume' or 'surface' (default: 'volume')
        config: Optional config dictionary to override all settings

    Returns:
        Dictionary containing:
            - connectivity_matrix: Structural connectivity matrix
            - roi_names: List of ROI names
            - output_dir: Path to output directory
            - probtrackx_outputs: Dictionary of probtrackx output file paths
            - metadata: Analysis metadata

    Raises:
        StructuralConnectivityError: If any step fails
    """
    # Apply config settings if provided
    if config is not None:
        sc_config = get_tractography_config(config)
        # Override parameters from config
        use_gpu = sc_config['tractography'].get('use_gpu', use_gpu)
        n_samples = sc_config['tractography'].get('n_samples', n_samples)
        step_length = sc_config['tractography'].get('step_length', step_length)
        curvature_threshold = sc_config['tractography'].get('curvature_threshold', curvature_threshold)
        avoid_ventricles = sc_config['anatomical_constraints'].get('avoid_ventricles', avoid_ventricles)
        use_wm_mask = sc_config['anatomical_constraints'].get('use_wm_mask', use_wm_mask)
        terminate_at_gm = sc_config['anatomical_constraints'].get('terminate_at_gm', terminate_at_gm)
        use_gmwmi_seeding = sc_config['freesurfer_options'].get('use_gmwmi_seeding', use_gmwmi_seeding)
        gmwmi_method = sc_config['freesurfer_options'].get('gmwmi_method', gmwmi_method)
        normalize = sc_config['output'].get('normalize', normalize)
        threshold = sc_config['output'].get('threshold', threshold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bedpostx_dir = Path(bedpostx_dir)

    logger.info("=" * 80)
    logger.info("STRUCTURAL CONNECTIVITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"BEDPOSTX directory: {bedpostx_dir}")
    logger.info(f"Atlas: {atlas_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Avoid ventricles: {avoid_ventricles}")

    # Step 1: Validate BEDPOSTX outputs
    logger.info("\n[Step 1] Validating BEDPOSTX outputs...")
    bedpostx_outputs = validate_bedpostx_outputs(bedpostx_dir)
    logger.info("✓ BEDPOSTX outputs validated")

    # Step 1b: Prepare ventricle exclusion mask if requested
    final_exclusion_mask = exclusion_mask
    ventricle_mask_used = None

    if avoid_ventricles and exclusion_mask is None:
        logger.info("\n[Step 1b] Preparing ventricle exclusion mask...")

        # Get reference image (brain mask from BEDPOSTX)
        dwi_reference = bedpostx_dir / 'nodif_brain_mask.nii.gz'

        if ventricle_mask is not None:
            # Use provided ventricle mask
            ventricle_mask = Path(ventricle_mask)
            if ventricle_mask.exists():
                final_exclusion_mask = ventricle_mask
                ventricle_mask_used = str(ventricle_mask)
                logger.info(f"  Using provided ventricle mask: {ventricle_mask.name}")
            else:
                logger.warning(f"  Provided ventricle mask not found: {ventricle_mask}")

        if final_exclusion_mask is None and subject is not None and derivatives_dir is not None:
            # Try to auto-find ventricle source
            source_path, source_type = find_ventricle_mask_source(
                subject=subject,
                derivatives_dir=Path(derivatives_dir),
                fs_subjects_dir=Path(fs_subjects_dir) if fs_subjects_dir else None
            )

            if source_path is not None:
                # Create ventricle mask in output directory
                ventricle_output = output_dir / 'ventricle_mask.nii.gz'
                final_exclusion_mask = create_ventricle_mask(
                    source=source_path,
                    output_file=ventricle_output,
                    source_type=source_type,
                    reference=dwi_reference
                )
                ventricle_mask_used = str(final_exclusion_mask)
                logger.info(f"  Created ventricle mask from {source_type}: {final_exclusion_mask.name}")
            else:
                logger.warning("  Could not find source for ventricle mask")
                logger.warning("  Tractography will run WITHOUT ventricle exclusion")

        if final_exclusion_mask is None and avoid_ventricles:
            logger.warning("  No ventricle mask available - continuing without exclusion")

    elif exclusion_mask is not None:
        logger.info(f"  Using provided exclusion mask: {exclusion_mask}")

    # Step 1c: Prepare white matter waypoint mask (ACT-style constraint)
    final_waypoint_mask = waypoint_mask
    wm_mask_used = None

    if use_wm_mask and waypoint_mask is None:
        logger.info("\n[Step 1c] Preparing white matter waypoint mask (ACT-style)...")

        # Get reference image
        dwi_reference = bedpostx_dir / 'nodif_brain_mask.nii.gz'

        if wm_mask is not None:
            wm_mask = Path(wm_mask)
            if wm_mask.exists():
                final_waypoint_mask = wm_mask
                wm_mask_used = str(wm_mask)
                logger.info(f"  Using provided WM mask: {wm_mask.name}")

        if final_waypoint_mask is None and subject is not None and derivatives_dir is not None:
            source_path, source_type = find_wm_mask_source(
                subject=subject,
                derivatives_dir=Path(derivatives_dir),
                fs_subjects_dir=Path(fs_subjects_dir) if fs_subjects_dir else None
            )

            if source_path is not None:
                wm_output = output_dir / 'wm_mask.nii.gz'
                final_waypoint_mask = create_wm_mask(
                    source=source_path,
                    output_file=wm_output,
                    source_type=source_type,
                    reference=dwi_reference
                )
                wm_mask_used = str(final_waypoint_mask)
                logger.info(f"  Created WM mask from {source_type}: {final_waypoint_mask.name}")
            else:
                logger.warning("  Could not find source for WM mask")
                logger.warning("  Tractography will run WITHOUT WM constraint")

    # Step 1d: Prepare gray matter termination mask
    final_termination_mask = None
    gm_mask_used = None

    if terminate_at_gm:
        logger.info("\n[Step 1d] Preparing gray matter termination mask...")

        dwi_reference = bedpostx_dir / 'nodif_brain_mask.nii.gz'

        if gm_mask is not None:
            gm_mask = Path(gm_mask)
            if gm_mask.exists():
                final_termination_mask = gm_mask
                gm_mask_used = str(gm_mask)
                logger.info(f"  Using provided GM mask: {gm_mask.name}")

        if final_termination_mask is None and subject is not None and derivatives_dir is not None:
            # Try FreeSurfer first, then FSL
            if fs_subjects_dir is not None:
                aparc_path = Path(fs_subjects_dir) / subject / 'mri' / 'aparc+aseg.mgz'
                if aparc_path.exists():
                    gm_output = output_dir / 'gm_termination_mask.nii.gz'
                    final_termination_mask = create_gm_termination_mask(
                        source=aparc_path,
                        output_file=gm_output,
                        source_type='freesurfer',
                        reference=dwi_reference
                    )
                    gm_mask_used = str(final_termination_mask)

            if final_termination_mask is None:
                # Try FSL GM probability map
                gm_paths = [
                    Path(derivatives_dir) / subject / 'anat' / 'segmentation' / 'pve_1.nii.gz',
                    Path(derivatives_dir) / subject / 'anat' / 'tissue_pve_1.nii.gz',
                ]
                for gm_path in gm_paths:
                    if gm_path.exists():
                        gm_output = output_dir / 'gm_termination_mask.nii.gz'
                        final_termination_mask = create_gm_termination_mask(
                            source=gm_path,
                            output_file=gm_output,
                            source_type='fsl_pve',
                            reference=dwi_reference
                        )
                        gm_mask_used = str(final_termination_mask)
                        break

        if final_termination_mask is not None:
            logger.info(f"  GM termination mask: {final_termination_mask}")
        else:
            logger.warning("  Could not create GM termination mask")

    # Step 1e: Prepare GMWMI seeding mask (optional - for FreeSurfer-enhanced tractography)
    gmwmi_mask_used = None

    if use_gmwmi_seeding and fs_subjects_dir is not None and subject is not None:
        logger.info("\n[Step 1e] Preparing GMWMI seeding mask...")

        fs_subject_path = Path(fs_subjects_dir) / subject
        dwi_reference = bedpostx_dir / 'nodif_brain_mask.nii.gz'

        if fs_subject_path.exists():
            gmwmi_output = output_dir / 'gmwmi_seed_mask.nii.gz'
            try:
                gmwmi_mask = create_gmwmi_mask(
                    fs_subject_dir=fs_subject_path,
                    output_file=gmwmi_output,
                    method=gmwmi_method,
                    reference=dwi_reference
                )
                gmwmi_mask_used = str(gmwmi_mask)
                logger.info(f"  GMWMI mask created: {gmwmi_mask.name}")
                logger.info("  Note: GMWMI mask will be used to constrain seeding to GM-WM interface")
            except Exception as e:
                logger.warning(f"  Could not create GMWMI mask: {e}")
                logger.warning("  Continuing with standard seeding")
        else:
            logger.warning(f"  FreeSurfer subject not found: {fs_subject_path}")

    # Step 2: Prepare atlas for probtrackx2
    logger.info("\n[Step 2] Preparing atlas for probtrackx2...")
    roi_dir = output_dir / 'roi_masks'
    seeds_list, roi_names = prepare_atlas_for_probtrackx(
        atlas_file=atlas_file,
        output_dir=roi_dir,
        min_voxels_per_roi=min_voxels_per_roi
    )
    logger.info(f"✓ Created {len(roi_names)} ROI masks")

    # Step 3: Run probtrackx2 in network mode
    logger.info("\n[Step 3] Running probtrackx2 network mode...")
    logger.info(f"  GPU acceleration: {use_gpu}")
    if final_exclusion_mask:
        logger.info(f"  Exclusion mask: {final_exclusion_mask}")
    if final_waypoint_mask:
        logger.info(f"  Waypoint mask (WM): {final_waypoint_mask}")
    if final_termination_mask:
        logger.info(f"  Termination mask (GM): {final_termination_mask}")

    probtrackx_dir = output_dir / 'probtrackx_output'
    probtrackx_outputs = run_probtrackx2_network(
        bedpostx_dir=bedpostx_dir,
        seeds_list=seeds_list,
        output_dir=probtrackx_dir,
        n_samples=n_samples,
        step_length=step_length,
        curvature_threshold=curvature_threshold,
        waypoint_mask=final_waypoint_mask,
        exclusion_mask=final_exclusion_mask,
        termination_mask=final_termination_mask,
        use_gpu=use_gpu
    )
    logger.info("✓ Probtrackx2 completed")

    # Step 4: Construct connectivity matrix
    logger.info("\n[Step 4] Constructing connectivity matrix...")
    sc_results = construct_connectivity_matrix(
        probtrackx_output_dir=probtrackx_dir,
        roi_names=roi_names,
        normalize=normalize,
        threshold=threshold
    )
    logger.info("✓ Connectivity matrix constructed")

    # Save connectivity matrix and metadata
    logger.info("\n[Step 5] Saving results...")

    # Save connectivity matrix (NumPy format)
    np.save(
        output_dir / 'structural_connectivity_matrix.npy',
        sc_results['connectivity_matrix']
    )

    # Save as CSV for easy inspection
    sc_df = pd.DataFrame(
        sc_results['connectivity_matrix'],
        index=roi_names,
        columns=roi_names
    )
    sc_df.to_csv(output_dir / 'structural_connectivity_matrix.csv')

    # Save metadata
    metadata = {
        'n_rois': len(roi_names),
        'atlas_file': str(atlas_file),
        'bedpostx_dir': str(bedpostx_dir),
        'n_samples': n_samples,
        'step_length': step_length,
        'curvature_threshold': curvature_threshold,
        'normalized': normalize,
        'threshold': threshold,
        'use_gpu': use_gpu,
        'avoid_ventricles': avoid_ventricles,
        'ventricle_mask': ventricle_mask_used,
        'exclusion_mask': str(final_exclusion_mask) if final_exclusion_mask else None,
        'use_wm_mask': use_wm_mask,
        'wm_mask': wm_mask_used,
        'waypoint_mask': str(final_waypoint_mask) if final_waypoint_mask else None,
        'terminate_at_gm': terminate_at_gm,
        'gm_mask': gm_mask_used,
        'termination_mask': str(final_termination_mask) if final_termination_mask else None,
        'use_gmwmi_seeding': use_gmwmi_seeding,
        'gmwmi_method': gmwmi_method if use_gmwmi_seeding else None,
        'gmwmi_mask': gmwmi_mask_used,
        'n_connections': int(np.sum(sc_results['connectivity_matrix'] > 0)),
        'connection_density': float(np.sum(sc_results['connectivity_matrix'] > 0) / (len(roi_names) * (len(roi_names) - 1))),
        'roi_names': roi_names
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Connectivity matrix: {output_dir / 'structural_connectivity_matrix.npy'}")
    logger.info(f"✓ CSV export: {output_dir / 'structural_connectivity_matrix.csv'}")
    logger.info(f"✓ Metadata: {output_dir / 'metadata.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("STRUCTURAL CONNECTIVITY ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"ROIs: {len(roi_names)}")
    logger.info(f"Connections: {metadata['n_connections']}")
    logger.info(f"Density: {metadata['connection_density']:.3f}")

    return {
        'connectivity_matrix': sc_results['connectivity_matrix'],
        'roi_names': roi_names,
        'output_dir': str(output_dir),
        'probtrackx_outputs': {k: str(v) if not isinstance(v, list) else [str(x) for x in v]
                                for k, v in probtrackx_outputs.items()},
        'metadata': metadata
    }
