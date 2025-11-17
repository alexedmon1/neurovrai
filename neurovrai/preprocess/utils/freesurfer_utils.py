#!/usr/bin/env python3
"""
FreeSurfer Integration Utilities

**STATUS: EXPERIMENTAL / NOT PRODUCTION READY**

This module provides detection and extraction hooks for FreeSurfer outputs but
is NOT a complete integration. Critical missing components:

1. **Transform Pipeline**: No anatomical→DWI transformation implemented
   - FreeSurfer ROIs are in native anatomical space
   - Cannot be used for DWI tractography without proper warping
   - Requires integration with anatomical preprocessing transforms

2. **Space Management**: No handling of FreeSurfer native space vs. anatomical space
   - May need additional T1→FreeSurfer transform if spaces differ
   - No validation that FreeSurfer was run on same T1 as preprocessing

3. **Quality Control**: No QC for transform accuracy or ROI alignment

DO NOT ENABLE in production until transform pipeline is fully implemented.

This module does NOT run FreeSurfer - it only detects and uses existing outputs.
"""

from pathlib import Path
from typing import Optional, Dict, List
import logging
import nibabel as nib

logger = logging.getLogger(__name__)


def detect_freesurfer_subject(
    subject: str,
    subjects_dir: Optional[Path] = None,
    config: Optional[Dict] = None
) -> Optional[Path]:
    """
    Detect if FreeSurfer has been run for a subject.

    Parameters
    ----------
    subject : str
        Subject identifier
    subjects_dir : Path, optional
        FreeSurfer SUBJECTS_DIR path
        If not provided, will try to get from config or environment
    config : dict, optional
        Configuration dictionary (can specify freesurfer.subjects_dir)

    Returns
    -------
    Path or None
        Path to FreeSurfer subject directory if found, None otherwise

    Examples
    --------
    >>> fs_dir = detect_freesurfer_subject('IRC805-0580101',
    ...                                      subjects_dir=Path('/mnt/bytopia/IRC805/freesurfer'))
    >>> if fs_dir:
    ...     print(f"Found FreeSurfer outputs: {fs_dir}")
    """
    import os

    # Try to get SUBJECTS_DIR from multiple sources
    if subjects_dir is None:
        if config and 'freesurfer' in config:
            subjects_dir = config['freesurfer'].get('subjects_dir')
            if subjects_dir:
                subjects_dir = Path(subjects_dir)

        # Fall back to environment variable
        if subjects_dir is None:
            env_subjects_dir = os.environ.get('SUBJECTS_DIR')
            if env_subjects_dir:
                subjects_dir = Path(env_subjects_dir)

    if subjects_dir is None:
        logger.debug("No FreeSurfer SUBJECTS_DIR specified")
        return None

    subjects_dir = Path(subjects_dir)
    if not subjects_dir.exists():
        logger.debug(f"FreeSurfer SUBJECTS_DIR does not exist: {subjects_dir}")
        return None

    # Check if subject directory exists
    subject_dir = subjects_dir / subject
    if not subject_dir.exists():
        logger.debug(f"FreeSurfer subject directory not found: {subject_dir}")
        return None

    # Verify it's a valid FreeSurfer directory (check for key subdirectories)
    required_dirs = ['mri', 'surf', 'label']
    for req_dir in required_dirs:
        if not (subject_dir / req_dir).exists():
            logger.debug(f"Invalid FreeSurfer directory (missing {req_dir}): {subject_dir}")
            return None

    logger.info(f"✓ Found FreeSurfer outputs: {subject_dir}")
    return subject_dir


def get_freesurfer_files(fs_subject_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Get paths to common FreeSurfer output files.

    Parameters
    ----------
    fs_subject_dir : Path
        Path to FreeSurfer subject directory

    Returns
    -------
    dict
        Dictionary of FreeSurfer file paths (values are None if file doesn't exist)
        Keys include:
        - 'aparc_aseg': aparc+aseg.mgz (volumetric parcellation)
        - 'brain': brain.mgz (skull-stripped brain)
        - 'wm': wm.mgz (white matter mask)
        - 'aseg': aseg.mgz (subcortical segmentation)
        - 'norm': norm.mgz (normalized intensity)
        - 'orig': orig.mgz (original volume)
        - 'lh_white': lh.white (left hemisphere white surface)
        - 'rh_white': rh.white (right hemisphere white surface)
        - 'lh_pial': lh.pial (left hemisphere pial surface)
        - 'rh_pial': rh.pial (right hemisphere pial surface)

    Examples
    --------
    >>> fs_dir = Path('/mnt/bytopia/IRC805/freesurfer/IRC805-0580101')
    >>> files = get_freesurfer_files(fs_dir)
    >>> if files['aparc_aseg']:
    ...     print(f"Parcellation available: {files['aparc_aseg']}")
    """
    mri_dir = fs_subject_dir / 'mri'
    surf_dir = fs_subject_dir / 'surf'

    files = {
        # Volumetric files
        'aparc_aseg': mri_dir / 'aparc+aseg.mgz',
        'brain': mri_dir / 'brain.mgz',
        'wm': mri_dir / 'wm.mgz',
        'aseg': mri_dir / 'aseg.mgz',
        'norm': mri_dir / 'norm.mgz',
        'orig': mri_dir / 'orig.mgz',
        'rawavg': mri_dir / 'rawavg.mgz',

        # Surface files
        'lh_white': surf_dir / 'lh.white',
        'rh_white': surf_dir / 'rh.white',
        'lh_pial': surf_dir / 'lh.pial',
        'rh_pial': surf_dir / 'rh.pial',
    }

    # Check which files exist
    for key, path in files.items():
        if not path.exists():
            files[key] = None

    return files


def extract_freesurfer_roi(
    aparc_aseg_file: Path,
    roi_label: int,
    output_file: Path,
    reference_space: Optional[Path] = None
) -> Path:
    """
    Extract a specific ROI from FreeSurfer parcellation.

    Parameters
    ----------
    aparc_aseg_file : Path
        Path to aparc+aseg.mgz file
    roi_label : int
        FreeSurfer label number (e.g., 17 for left hippocampus)
    output_file : Path
        Output NIfTI file path
    reference_space : Path, optional
        If provided, warp ROI to this reference space (e.g., DWI space)

    Returns
    -------
    Path
        Path to output ROI mask

    Notes
    -----
    Common FreeSurfer labels:
    - Left hippocampus: 17
    - Right hippocampus: 53
    - Left thalamus: 10
    - Right thalamus: 49
    - Left caudate: 11
    - Right caudate: 50

    See: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

    Examples
    --------
    >>> extract_freesurfer_roi(
    ...     aparc_aseg_file=Path('freesurfer/sub-001/mri/aparc+aseg.mgz'),
    ...     roi_label=17,  # Left hippocampus
    ...     output_file=Path('rois/hippocampus_l.nii.gz')
    ... )
    """
    import subprocess

    logger.info(f"Extracting FreeSurfer ROI (label={roi_label}) to {output_file}")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use mri_binarize to extract the ROI
    cmd = [
        'mri_binarize',
        '--i', str(aparc_aseg_file),
        '--match', str(roi_label),
        '--o', str(output_file)
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    # TODO: If reference_space is provided, warp ROI to that space
    # This would require FreeSurfer → anatomical → reference transforms
    if reference_space:
        logger.warning("ROI warping to reference space not yet implemented")

    logger.info(f"  Extracted ROI: {output_file}")
    return output_file


def get_freesurfer_rois_for_tractography(
    fs_subject_dir: Path,
    roi_names: List[str],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Extract common ROIs from FreeSurfer for tractography.

    Parameters
    ----------
    fs_subject_dir : Path
        Path to FreeSurfer subject directory
    roi_names : list of str
        ROI names to extract (e.g., ['hippocampus_l', 'thalamus_l'])
    output_dir : Path
        Output directory for ROI masks

    Returns
    -------
    dict
        Dictionary mapping ROI names to file paths

    Examples
    --------
    >>> fs_dir = Path('/mnt/bytopia/IRC805/freesurfer/IRC805-0580101')
    >>> rois = get_freesurfer_rois_for_tractography(
    ...     fs_subject_dir=fs_dir,
    ...     roi_names=['hippocampus_l', 'hippocampus_r', 'thalamus_l'],
    ...     output_dir=Path('derivatives/tractography/rois')
    ... )
    """
    # FreeSurfer label mapping (from aparc+aseg)
    label_map = {
        'hippocampus_l': 17,
        'hippocampus_r': 53,
        'thalamus_l': 10,
        'thalamus_r': 49,
        'caudate_l': 11,
        'caudate_r': 50,
        'putamen_l': 12,
        'putamen_r': 51,
        'pallidum_l': 13,
        'pallidum_r': 52,
        'amygdala_l': 18,
        'amygdala_r': 54,
    }

    aparc_aseg = fs_subject_dir / 'mri' / 'aparc+aseg.mgz'
    if not aparc_aseg.exists():
        raise FileNotFoundError(f"FreeSurfer parcellation not found: {aparc_aseg}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rois = {}
    for roi_name in roi_names:
        if roi_name not in label_map:
            logger.warning(f"Unknown ROI name: {roi_name} (skipping)")
            continue

        label = label_map[roi_name]
        output_file = output_dir / f'{roi_name}.nii.gz'

        extract_freesurfer_roi(
            aparc_aseg_file=aparc_aseg,
            roi_label=label,
            output_file=output_file
        )

        rois[roi_name] = output_file

    return rois


def check_freesurfer_availability(config: Optional[Dict] = None) -> bool:
    """
    Check if FreeSurfer integration is enabled and available.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    bool
        True if FreeSurfer integration is enabled, False otherwise

    Examples
    --------
    >>> if check_freesurfer_availability(config):
    ...     print("FreeSurfer integration enabled")
    """
    if config is None:
        return False

    fs_config = config.get('freesurfer', {})
    enabled = fs_config.get('enabled', False)

    if not enabled:
        logger.debug("FreeSurfer integration disabled in config")
        return False

    subjects_dir = fs_config.get('subjects_dir')
    if not subjects_dir:
        logger.warning("FreeSurfer enabled but no subjects_dir specified")
        return False

    subjects_dir = Path(subjects_dir)
    if not subjects_dir.exists():
        logger.warning(f"FreeSurfer subjects_dir does not exist: {subjects_dir}")
        return False

    return True
