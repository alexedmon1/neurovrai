#!/usr/bin/env python3
"""
File finder utility for locating MRI data files.

Provides config-driven file matching without hardcoded sequence names.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from glob import glob


def match_sequence(filename: str, patterns: List[str]) -> bool:
    """
    Check if filename matches any of the given patterns.

    Patterns can be:
    - Exact strings (substring match)
    - Regex patterns (if starting with .*)

    Parameters
    ----------
    filename : str
        Filename or sequence name to check
    patterns : list of str
        List of patterns to match against

    Returns
    -------
    bool
        True if filename matches any pattern

    Examples
    --------
    >>> match_sequence("MPRAGE_SAG", ["MPRAGE", "T1_3D"])
    True
    >>> match_sequence("T2_SPACE", [".*T2.*", "FLAIR"])
    True
    """
    filename_upper = filename.upper()

    for pattern in patterns:
        pattern_upper = pattern.upper()

        # Check if it's a regex pattern
        if pattern_upper.startswith(".*"):
            # Use regex matching
            if re.search(pattern_upper, filename_upper):
                return True
        else:
            # Simple substring matching
            if pattern_upper in filename_upper:
                return True

    return False


def find_by_modality(
    search_dir: Path,
    modality: str,
    config: Dict,
    file_pattern: str = "*.nii.gz"
) -> List[Path]:
    """
    Find all files matching a specific modality.

    Uses sequence mappings from config to identify files.

    Parameters
    ----------
    search_dir : Path
        Directory to search in
    modality : str
        Modality to find (e.g., 't1w', 'dwi', 'rest')
    config : dict
        Configuration dictionary with sequence_mappings
    file_pattern : str
        File pattern to match (default: *.nii.gz)

    Returns
    -------
    list of Path
        List of matching files

    Examples
    --------
    >>> files = find_by_modality(Path("/data/sub-001/anat"), "t1w", config)
    >>> # Returns all T1w files based on sequence_mappings in config
    """
    search_dir = Path(search_dir)

    # Get sequence patterns from config
    if 'sequence_mappings' not in config:
        raise ValueError("Config missing 'sequence_mappings'")

    if modality not in config['sequence_mappings']:
        return []

    patterns = config['sequence_mappings'][modality]

    # Find all files matching the file pattern
    all_files = list(search_dir.rglob(file_pattern))

    # Filter by sequence name
    matching_files = []
    for file_path in all_files:
        if match_sequence(file_path.name, patterns):
            matching_files.append(file_path)

    return sorted(matching_files)


def find_subject_files(
    subject_dir: Path,
    modality: str,
    config: Dict,
    session: Optional[str] = None
) -> Dict[str, List[Path]]:
    """
    Find all files for a subject, organized by type.

    Parameters
    ----------
    subject_dir : Path
        Subject's BIDS directory
    modality : str
        Modality to find (e.g., 't1w', 'dwi', 'rest')
    config : dict
        Configuration dictionary
    session : str, optional
        Session identifier (for multi-session studies)

    Returns
    -------
    dict
        Dictionary with file types as keys:
        - 'images': NIfTI files
        - 'json': JSON sidecars
        - 'bval': b-value files (for DWI)
        - 'bvec': b-vector files (for DWI)

    Examples
    --------
    >>> files = find_subject_files(Path("/data/sub-001"), "dwi", config)
    >>> print(files['images'])  # List of DWI NIfTI files
    >>> print(files['bval'])    # Corresponding bval files
    """
    subject_dir = Path(subject_dir)

    # Build search directory
    if session:
        search_dir = subject_dir / session / modality
    else:
        search_dir = subject_dir / modality

    if not search_dir.exists():
        return {'images': [], 'json': [], 'bval': [], 'bvec': []}

    # Find images
    images = find_by_modality(search_dir, modality, config, "*.nii.gz")

    # Find corresponding sidecars
    json_files = []
    bval_files = []
    bvec_files = []

    for img in images:
        # JSON sidecar
        json_path = img.with_suffix('').with_suffix('.json')
        if json_path.exists():
            json_files.append(json_path)

        # bval/bvec (for diffusion)
        if modality == 'dwi':
            bval_path = img.with_suffix('').with_suffix('.bval')
            bvec_path = img.with_suffix('').with_suffix('.bvec')

            if bval_path.exists():
                bval_files.append(bval_path)
            if bvec_path.exists():
                bvec_files.append(bvec_path)

    return {
        'images': images,
        'json': json_files,
        'bval': bval_files,
        'bvec': bvec_files
    }


def detect_multi_echo(files: List[Path]) -> Tuple[bool, Optional[int]]:
    """
    Detect if functional data is multi-echo and count echoes.

    Looks for echo number in filename (e.g., _echo-1_, _echo-2_).

    Parameters
    ----------
    files : list of Path
        List of functional image files

    Returns
    -------
    tuple
        (is_multi_echo, num_echoes)

    Examples
    --------
    >>> files = [Path("sub-001_task-rest_echo-1_bold.nii.gz"),
    ...          Path("sub-001_task-rest_echo-2_bold.nii.gz"),
    ...          Path("sub-001_task-rest_echo-3_bold.nii.gz")]
    >>> detect_multi_echo(files)
    (True, 3)
    """
    echo_pattern = re.compile(r'echo-(\d+)', re.IGNORECASE)

    echoes = set()
    for file_path in files:
        match = echo_pattern.search(file_path.name)
        if match:
            echoes.add(int(match.group(1)))

    if len(echoes) > 1:
        return True, len(echoes)
    elif len(echoes) == 1:
        # Single echo explicitly labeled
        return False, 1
    else:
        # No echo labels - assume single echo
        return False, None


def group_by_echo(files: List[Path]) -> Dict[int, Path]:
    """
    Group multi-echo files by echo number.

    Parameters
    ----------
    files : list of Path
        List of multi-echo functional files

    Returns
    -------
    dict
        Dictionary mapping echo number -> file path

    Examples
    --------
    >>> files = [Path("sub-001_echo-2_bold.nii.gz"),
    ...          Path("sub-001_echo-1_bold.nii.gz"),
    ...          Path("sub-001_echo-3_bold.nii.gz")]
    >>> grouped = group_by_echo(files)
    >>> grouped[1]  # First echo
    Path("sub-001_echo-1_bold.nii.gz")
    """
    echo_pattern = re.compile(r'echo-(\d+)', re.IGNORECASE)

    grouped = {}
    for file_path in files:
        match = echo_pattern.search(file_path.name)
        if match:
            echo_num = int(match.group(1))
            grouped[echo_num] = file_path

    return grouped


def find_reference_files(subject_dir: Path, file_type: str) -> Optional[Path]:
    """
    Find reference files (e.g., brain mask, transformation matrix).

    Parameters
    ----------
    subject_dir : Path
        Subject's derivatives directory
    file_type : str
        Type of file to find:
        - 'brain_mask': Skull-stripped brain mask
        - 't1w_brain': Skull-stripped T1w
        - 't1w_mni_affine': Affine transformation matrix
        - 't1w_mni_warp': Nonlinear warp field

    Returns
    -------
    Path or None
        Path to file if found, None otherwise

    Examples
    --------
    >>> mask = find_reference_files(Path("/data/derivatives/sub-001"), "brain_mask")
    """
    subject_dir = Path(subject_dir)

    # Define search patterns for different file types
    patterns = {
        'brain_mask': ['*brain_mask.nii.gz', '*mask.nii.gz'],
        't1w_brain': ['*T1w_brain.nii.gz', '*t1w_brain.nii.gz'],
        't1w_mni_affine': ['*to_MNI_affine.mat', '*to_mni_affine.mat'],
        't1w_mni_warp': ['*to_MNI_warp.nii.gz', '*to_mni_warp.nii.gz'],
    }

    if file_type not in patterns:
        raise ValueError(f"Unknown file_type: {file_type}")

    # Search for files
    for pattern in patterns[file_type]:
        matches = list(subject_dir.rglob(pattern))
        if matches:
            return matches[0]  # Return first match

    return None


def validate_dwi_files(images: List[Path], bvals: List[Path], bvecs: List[Path]) -> bool:
    """
    Validate that DWI files are complete (images + bvals + bvecs).

    Parameters
    ----------
    images : list of Path
        DWI image files
    bvals : list of Path
        b-value files
    bvecs : list of Path
        b-vector files

    Returns
    -------
    bool
        True if all files present and counts match

    Raises
    ------
    ValueError
        If files are missing or counts don't match
    """
    if not images:
        raise ValueError("No DWI images found")

    if len(images) != len(bvals):
        raise ValueError(f"Number of images ({len(images)}) doesn't match bval files ({len(bvals)})")

    if len(images) != len(bvecs):
        raise ValueError(f"Number of images ({len(images)}) doesn't match bvec files ({len(bvecs)})")

    return True
