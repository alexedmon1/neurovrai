#!/usr/bin/env python3
"""
BIDS utilities for path management.

Provides functions for BIDS-like directory layout without os.chdir().
All path operations use absolute paths.
"""

from pathlib import Path
from typing import Optional, Dict
import json


def get_subject_dir(
    base_dir: Path,
    subject: str,
    session: Optional[str] = None,
    create: bool = False
) -> Path:
    """
    Get subject directory path.

    Parameters
    ----------
    base_dir : Path
        Base directory (rawdata or derivatives)
    subject : str
        Subject ID (with or without 'sub-' prefix)
    session : str, optional
        Session ID (with or without 'ses-' prefix)
    create : bool
        Create directory if it doesn't exist

    Returns
    -------
    Path
        Absolute path to subject directory

    Examples
    --------
    >>> get_subject_dir(Path("/data/rawdata"), "001")
    Path("/data/rawdata/sub-001")
    >>> get_subject_dir(Path("/data/rawdata"), "sub-001", session="01")
    Path("/data/rawdata/sub-001/ses-01")
    """
    # Ensure 'sub-' prefix
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'

    subject_dir = Path(base_dir) / subject

    if session:
        # Ensure 'ses-' prefix
        if not session.startswith('ses-'):
            session = f'ses-{session}'
        subject_dir = subject_dir / session

    # Make absolute
    subject_dir = subject_dir.resolve()

    if create:
        subject_dir.mkdir(parents=True, exist_ok=True)

    return subject_dir


def get_modality_dir(
    base_dir: Path,
    subject: str,
    modality: str,
    session: Optional[str] = None,
    create: bool = False
) -> Path:
    """
    Get modality directory path.

    Parameters
    ----------
    base_dir : Path
        Base directory (rawdata or derivatives)
    subject : str
        Subject ID
    modality : str
        Modality name (anat, dwi, func, etc.)
    session : str, optional
        Session ID
    create : bool
        Create directory if it doesn't exist

    Returns
    -------
    Path
        Absolute path to modality directory

    Examples
    --------
    >>> get_modality_dir(Path("/data/rawdata"), "001", "anat")
    Path("/data/rawdata/sub-001/anat")
    """
    subject_dir = get_subject_dir(base_dir, subject, session, create=False)
    modality_dir = subject_dir / modality

    # Make absolute
    modality_dir = modality_dir.resolve()

    if create:
        modality_dir.mkdir(parents=True, exist_ok=True)

    return modality_dir


def build_bids_filename(
    subject: str,
    modality: str,
    suffix: str,
    session: Optional[str] = None,
    task: Optional[str] = None,
    echo: Optional[int] = None,
    run: Optional[int] = None,
    extension: str = '.nii.gz'
) -> str:
    """
    Build BIDS-compliant filename.

    Parameters
    ----------
    subject : str
        Subject ID
    modality : str
        Modality (anat, func, dwi)
    suffix : str
        BIDS suffix (T1w, bold, dwi, etc.)
    session : str, optional
        Session ID
    task : str, optional
        Task name (for functional)
    echo : int, optional
        Echo number (for multi-echo)
    run : int, optional
        Run number
    extension : str
        File extension (default: .nii.gz)

    Returns
    -------
    str
        BIDS filename

    Examples
    --------
    >>> build_bids_filename("001", "anat", "T1w")
    'sub-001_T1w.nii.gz'
    >>> build_bids_filename("001", "func", "bold", task="rest", echo=1)
    'sub-001_task-rest_echo-1_bold.nii.gz'
    """
    # Ensure 'sub-' prefix
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'

    parts = [subject]

    # Add session if provided
    if session:
        if not session.startswith('ses-'):
            session = f'ses-{session}'
        parts.append(session)

    # Add task if provided
    if task:
        if not task.startswith('task-'):
            task = f'task-{task}'
        parts.append(task)

    # Add run if provided
    if run:
        parts.append(f'run-{run:02d}')

    # Add echo if provided
    if echo:
        parts.append(f'echo-{echo}')

    # Add suffix
    parts.append(suffix)

    # Join and add extension
    filename = '_'.join(parts) + extension

    return filename


def get_derivatives_dir(
    derivatives_base: Path,
    pipeline_name: str,
    subject: str,
    session: Optional[str] = None,
    create: bool = False
) -> Path:
    """
    Get derivatives directory for a specific pipeline.

    Parameters
    ----------
    derivatives_base : Path
        Base derivatives directory
    pipeline_name : str
        Name of preprocessing pipeline
    subject : str
        Subject ID
    session : str, optional
        Session ID
    create : bool
        Create directory if it doesn't exist

    Returns
    -------
    Path
        Absolute path to pipeline derivatives directory

    Examples
    --------
    >>> get_derivatives_dir(Path("/data/derivatives"), "mri-preprocess", "001")
    Path("/data/derivatives/mri-preprocess/sub-001")
    """
    pipeline_dir = Path(derivatives_base) / pipeline_name
    subject_dir = get_subject_dir(pipeline_dir, subject, session, create=False)

    # Make absolute
    subject_dir = subject_dir.resolve()

    if create:
        subject_dir.mkdir(parents=True, exist_ok=True)

    return subject_dir


def save_json_sidecar(
    image_path: Path,
    metadata: Dict,
    overwrite: bool = False
) -> Path:
    """
    Save JSON sidecar for NIfTI file.

    Parameters
    ----------
    image_path : Path
        Path to NIfTI image
    metadata : dict
        Metadata dictionary to save
    overwrite : bool
        Overwrite existing JSON file

    Returns
    -------
    Path
        Path to saved JSON file

    Examples
    --------
    >>> metadata = {'TR': 2.0, 'TE': 0.03}
    >>> save_json_sidecar(Path("sub-001_T1w.nii.gz"), metadata)
    Path("sub-001_T1w.json")
    """
    # Build JSON path (remove .nii.gz, add .json)
    json_path = image_path.with_suffix('').with_suffix('.json')

    if json_path.exists() and not overwrite:
        print(f"JSON sidecar already exists: {json_path}")
        return json_path

    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return json_path


def load_json_sidecar(image_path: Path) -> Optional[Dict]:
    """
    Load JSON sidecar for NIfTI file.

    Parameters
    ----------
    image_path : Path
        Path to NIfTI image

    Returns
    -------
    dict or None
        Metadata dictionary, or None if JSON doesn't exist

    Examples
    --------
    >>> metadata = load_json_sidecar(Path("sub-001_T1w.nii.gz"))
    >>> print(metadata['TR'])
    2.0
    """
    json_path = image_path.with_suffix('').with_suffix('.json')

    if not json_path.exists():
        return None

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def create_bids_structure(
    base_dir: Path,
    subject: str,
    modalities: list,
    session: Optional[str] = None
) -> Dict[str, Path]:
    """
    Create complete BIDS directory structure for a subject.

    Parameters
    ----------
    base_dir : Path
        Base directory
    subject : str
        Subject ID
    modalities : list
        List of modality names to create (e.g., ['anat', 'func', 'dwi'])
    session : str, optional
        Session ID

    Returns
    -------
    dict
        Dictionary mapping modality -> directory path

    Examples
    --------
    >>> dirs = create_bids_structure(
    ...     Path("/data/rawdata"),
    ...     "001",
    ...     ['anat', 'func', 'dwi']
    ... )
    >>> print(dirs['anat'])
    Path("/data/rawdata/sub-001/anat")
    """
    directories = {}

    for modality in modalities:
        mod_dir = get_modality_dir(
            base_dir,
            subject,
            modality,
            session=session,
            create=True
        )
        directories[modality] = mod_dir

    return directories


def get_workflow_dir(
    base_dir: Path,
    subject: str,
    workflow_name: str,
    session: Optional[str] = None,
    create: bool = True
) -> Path:
    """
    Get workflow working directory for Nipype.

    Parameters
    ----------
    base_dir : Path
        Base working directory
    subject : str
        Subject ID
    workflow_name : str
        Name of workflow (e.g., 'anat-prep', 'dti-prep')
    session : str, optional
        Session ID
    create : bool
        Create directory if it doesn't exist

    Returns
    -------
    Path
        Absolute path to workflow directory

    Examples
    --------
    >>> get_workflow_dir(Path("/tmp/work"), "001", "anat-prep")
    Path("/tmp/work/sub-001/anat-prep")
    """
    subject_dir = get_subject_dir(base_dir, subject, session, create=False)
    workflow_dir = subject_dir / workflow_name

    # Make absolute
    workflow_dir = workflow_dir.resolve()

    if create:
        workflow_dir.mkdir(parents=True, exist_ok=True)

    return workflow_dir


def sanitize_subject_id(subject: str) -> str:
    """
    Sanitize subject ID to remove non-BIDS-compliant characters.

    Removes 'sub-' prefix if present, keeps only alphanumeric and hyphens.

    Parameters
    ----------
    subject : str
        Raw subject ID

    Returns
    -------
    str
        Sanitized subject ID (without 'sub-' prefix)

    Examples
    --------
    >>> sanitize_subject_id("sub-001")
    '001'
    >>> sanitize_subject_id("IRC805-0580101")
    'IRC805-0580101'
    >>> sanitize_subject_id("subject_123")
    'subject-123'
    """
    # Remove 'sub-' prefix if present
    if subject.startswith('sub-'):
        subject = subject[4:]

    # Replace underscores with hyphens
    subject = subject.replace('_', '-')

    # Keep only alphanumeric and hyphens
    subject = ''.join(c for c in subject if c.isalnum() or c == '-')

    return subject
