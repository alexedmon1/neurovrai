#!/usr/bin/env python3
"""
FreeSurfer Integration Utilities

This module provides utilities for working with FreeSurfer outputs:
- Detection of existing FreeSurfer outputs
- Extraction of ROIs and atlases from aparc+aseg
- Complete label mappings for Desikan-Killiany and subcortical structures
- Validation of FreeSurfer completion status

For transform pipelines (FreeSurfer → T1w → DWI), see:
    neurovrai.preprocess.utils.freesurfer_transforms

This module does NOT run FreeSurfer - it only works with existing outputs.
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


# =============================================================================
# Complete Label Mappings
# =============================================================================

def get_desikan_killiany_labels() -> Dict[int, str]:
    """
    Get complete Desikan-Killiany atlas label mapping.

    Returns 68 cortical regions (34 per hemisphere) plus subcortical structures
    from the FreeSurfer aparc+aseg parcellation.

    Returns
    -------
    dict
        Mapping of FreeSurfer label numbers to region names

    Notes
    -----
    Reference: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    """
    labels = {
        # Left hemisphere cortical (1000-1035)
        1001: 'ctx-lh-bankssts',
        1002: 'ctx-lh-caudalanteriorcingulate',
        1003: 'ctx-lh-caudalmiddlefrontal',
        1005: 'ctx-lh-cuneus',
        1006: 'ctx-lh-entorhinal',
        1007: 'ctx-lh-fusiform',
        1008: 'ctx-lh-inferiorparietal',
        1009: 'ctx-lh-inferiortemporal',
        1010: 'ctx-lh-isthmuscingulate',
        1011: 'ctx-lh-lateraloccipital',
        1012: 'ctx-lh-lateralorbitofrontal',
        1013: 'ctx-lh-lingual',
        1014: 'ctx-lh-medialorbitofrontal',
        1015: 'ctx-lh-middletemporal',
        1016: 'ctx-lh-parahippocampal',
        1017: 'ctx-lh-paracentral',
        1018: 'ctx-lh-parsopercularis',
        1019: 'ctx-lh-parsorbitalis',
        1020: 'ctx-lh-parstriangularis',
        1021: 'ctx-lh-pericalcarine',
        1022: 'ctx-lh-postcentral',
        1023: 'ctx-lh-posteriorcingulate',
        1024: 'ctx-lh-precentral',
        1025: 'ctx-lh-precuneus',
        1026: 'ctx-lh-rostralanteriorcingulate',
        1027: 'ctx-lh-rostralmiddlefrontal',
        1028: 'ctx-lh-superiorfrontal',
        1029: 'ctx-lh-superiorparietal',
        1030: 'ctx-lh-superiortemporal',
        1031: 'ctx-lh-supramarginal',
        1032: 'ctx-lh-frontalpole',
        1033: 'ctx-lh-temporalpole',
        1034: 'ctx-lh-transversetemporal',
        1035: 'ctx-lh-insula',

        # Right hemisphere cortical (2000-2035)
        2001: 'ctx-rh-bankssts',
        2002: 'ctx-rh-caudalanteriorcingulate',
        2003: 'ctx-rh-caudalmiddlefrontal',
        2005: 'ctx-rh-cuneus',
        2006: 'ctx-rh-entorhinal',
        2007: 'ctx-rh-fusiform',
        2008: 'ctx-rh-inferiorparietal',
        2009: 'ctx-rh-inferiortemporal',
        2010: 'ctx-rh-isthmuscingulate',
        2011: 'ctx-rh-lateraloccipital',
        2012: 'ctx-rh-lateralorbitofrontal',
        2013: 'ctx-rh-lingual',
        2014: 'ctx-rh-medialorbitofrontal',
        2015: 'ctx-rh-middletemporal',
        2016: 'ctx-rh-parahippocampal',
        2017: 'ctx-rh-paracentral',
        2018: 'ctx-rh-parsopercularis',
        2019: 'ctx-rh-parsorbitalis',
        2020: 'ctx-rh-parstriangularis',
        2021: 'ctx-rh-pericalcarine',
        2022: 'ctx-rh-postcentral',
        2023: 'ctx-rh-posteriorcingulate',
        2024: 'ctx-rh-precentral',
        2025: 'ctx-rh-precuneus',
        2026: 'ctx-rh-rostralanteriorcingulate',
        2027: 'ctx-rh-rostralmiddlefrontal',
        2028: 'ctx-rh-superiorfrontal',
        2029: 'ctx-rh-superiorparietal',
        2030: 'ctx-rh-superiortemporal',
        2031: 'ctx-rh-supramarginal',
        2032: 'ctx-rh-frontalpole',
        2033: 'ctx-rh-temporalpole',
        2034: 'ctx-rh-transversetemporal',
        2035: 'ctx-rh-insula',

        # Subcortical structures (from aseg)
        10: 'Left-Thalamus',
        11: 'Left-Caudate',
        12: 'Left-Putamen',
        13: 'Left-Pallidum',
        17: 'Left-Hippocampus',
        18: 'Left-Amygdala',
        26: 'Left-Accumbens-area',
        28: 'Left-VentralDC',

        49: 'Right-Thalamus',
        50: 'Right-Caudate',
        51: 'Right-Putamen',
        52: 'Right-Pallidum',
        53: 'Right-Hippocampus',
        54: 'Right-Amygdala',
        58: 'Right-Accumbens-area',
        60: 'Right-VentralDC',

        # Brainstem
        16: 'Brain-Stem',
    }

    return labels


def get_subcortical_labels() -> Dict[int, str]:
    """
    Get subcortical structure labels from FreeSurfer aseg.

    Returns
    -------
    dict
        Mapping of FreeSurfer label numbers to subcortical region names
    """
    return {
        # Left hemisphere
        10: 'Left-Thalamus',
        11: 'Left-Caudate',
        12: 'Left-Putamen',
        13: 'Left-Pallidum',
        17: 'Left-Hippocampus',
        18: 'Left-Amygdala',
        26: 'Left-Accumbens-area',
        28: 'Left-VentralDC',

        # Right hemisphere
        49: 'Right-Thalamus',
        50: 'Right-Caudate',
        51: 'Right-Putamen',
        52: 'Right-Pallidum',
        53: 'Right-Hippocampus',
        54: 'Right-Amygdala',
        58: 'Right-Accumbens-area',
        60: 'Right-VentralDC',

        # Midline
        16: 'Brain-Stem',
        4: 'Left-Lateral-Ventricle',
        43: 'Right-Lateral-Ventricle',
        14: '3rd-Ventricle',
        15: '4th-Ventricle',
    }


# =============================================================================
# Atlas Extraction Functions
# =============================================================================

def extract_freesurfer_atlas(
    fs_subject_dir: Path,
    atlas_type: str,
    output_file: Path
) -> Path:
    """
    Extract volumetric atlas from FreeSurfer outputs.

    Converts FreeSurfer .mgz parcellation to NIfTI format for use
    with FSL/ANTs tools.

    Parameters
    ----------
    fs_subject_dir : Path
        Path to FreeSurfer subject directory
    atlas_type : str
        Type of atlas to extract:
        - 'aparc+aseg': Desikan-Killiany cortical + subcortical (default)
        - 'aparc.a2009s+aseg': Destrieux cortical + subcortical
        - 'aseg': Subcortical only
        - 'wmparc': White matter parcellation
    output_file : Path
        Output NIfTI file path (.nii.gz)

    Returns
    -------
    Path
        Path to extracted atlas NIfTI file

    Examples
    --------
    >>> atlas = extract_freesurfer_atlas(
    ...     fs_subject_dir=Path('/freesurfer/sub-001'),
    ...     atlas_type='aparc+aseg',
    ...     output_file=Path('atlases/desikan_killiany.nii.gz')
    ... )
    """
    import subprocess

    # Map atlas type to FreeSurfer file
    atlas_files = {
        'aparc+aseg': 'aparc+aseg.mgz',
        'aparc.a2009s+aseg': 'aparc.a2009s+aseg.mgz',
        'aseg': 'aseg.mgz',
        'wmparc': 'wmparc.mgz',
    }

    if atlas_type not in atlas_files:
        raise ValueError(
            f"Unknown atlas type: {atlas_type}. "
            f"Valid options: {list(atlas_files.keys())}"
        )

    mgz_file = fs_subject_dir / 'mri' / atlas_files[atlas_type]

    if not mgz_file.exists():
        raise FileNotFoundError(f"FreeSurfer atlas not found: {mgz_file}")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting FreeSurfer atlas: {atlas_type}")
    logger.info(f"  Input: {mgz_file}")
    logger.info(f"  Output: {output_file}")

    # Convert using mri_convert
    cmd = [
        'mri_convert',
        str(mgz_file),
        str(output_file)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"mri_convert failed: {result.stderr}")

    logger.info(f"  Atlas extracted: {output_file}")

    return output_file


def validate_freesurfer_complete(fs_subject_dir: Path) -> tuple:
    """
    Validate FreeSurfer recon-all completed successfully.

    Checks for required output files from a complete recon-all run.

    Parameters
    ----------
    fs_subject_dir : Path
        Path to FreeSurfer subject directory

    Returns
    -------
    tuple of (bool, list)
        (is_complete, missing_files) - True if complete, list of missing files

    Examples
    --------
    >>> is_complete, missing = validate_freesurfer_complete(
    ...     Path('/freesurfer/sub-001')
    ... )
    >>> if not is_complete:
    ...     print(f"Missing files: {missing}")
    """
    required_files = [
        # MRI volumes
        'mri/orig.mgz',
        'mri/brain.mgz',
        'mri/aparc+aseg.mgz',
        'mri/aseg.mgz',
        'mri/wm.mgz',
        'mri/norm.mgz',

        # Surfaces
        'surf/lh.white',
        'surf/rh.white',
        'surf/lh.pial',
        'surf/rh.pial',

        # Labels
        'label/lh.aparc.annot',
        'label/rh.aparc.annot',

        # Stats
        'stats/aseg.stats',
        'stats/lh.aparc.stats',
        'stats/rh.aparc.stats',
    ]

    missing = []
    for rel_path in required_files:
        full_path = fs_subject_dir / rel_path
        if not full_path.exists():
            missing.append(rel_path)

    is_complete = len(missing) == 0

    if is_complete:
        logger.info(f"FreeSurfer validation: COMPLETE ({fs_subject_dir.name})")
    else:
        logger.warning(
            f"FreeSurfer validation: INCOMPLETE ({fs_subject_dir.name}) - "
            f"{len(missing)} files missing"
        )

    return is_complete, missing


def get_roi_labels_for_atlas(atlas_type: str) -> Dict[int, str]:
    """
    Get ROI label mapping for a specific atlas type.

    Parameters
    ----------
    atlas_type : str
        Atlas type: 'aparc+aseg', 'aseg', 'subcortical'

    Returns
    -------
    dict
        Mapping of label numbers to region names
    """
    if atlas_type in ['aparc+aseg', 'desikan_killiany']:
        return get_desikan_killiany_labels()
    elif atlas_type in ['aseg', 'subcortical']:
        return get_subcortical_labels()
    else:
        raise ValueError(f"Unknown atlas type: {atlas_type}")
