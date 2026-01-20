"""
Study initialization module.

This module provides functions to initialize a new neuroimaging study,
including BIDS/DICOM data discovery, config generation, directory structure
setup, and template configuration.

Usage:
    from neurovrai.study_initialization import setup_study

    report = setup_study(
        study_root=Path('/path/to/study'),
        study_name='My MRI Study',
        study_code='STUDY01',
        dicom_root=Path('/path/to/dicom'),
    )
"""

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

from neurovrai.config import load_config


@dataclass
class ScanInfo:
    """Information about a single scan."""
    file_path: Path
    modality: str
    suffix: str  # e.g., 'T1w', 'T2w', 'dwi', 'bold', 'asl'
    run: Optional[str] = None
    acquisition: Optional[str] = None
    echo: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'file_path': str(self.file_path),
            'modality': self.modality,
            'suffix': self.suffix,
            'run': self.run,
            'acquisition': self.acquisition,
            'echo': self.echo
        }


@dataclass
class SessionInfo:
    """Information about a session."""
    session: str
    modalities: Dict[str, List[ScanInfo]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'session': self.session,
            'modalities': {
                mod: [s.to_dict() for s in scans]
                for mod, scans in self.modalities.items()
            }
        }


@dataclass
class SubjectInfo:
    """Information about a subject."""
    subject: str
    sessions: Dict[str, SessionInfo] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'subject': self.subject,
            'sessions': {
                ses: info.to_dict()
                for ses, info in self.sessions.items()
            }
        }


@dataclass
class BIDSManifest:
    """Complete manifest of BIDS data."""
    bids_root: Path
    subjects: Dict[str, SubjectInfo] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def n_sessions(self) -> int:
        return sum(len(s.sessions) for s in self.subjects.values())

    def get_modality_breakdown(self) -> Dict[str, int]:
        """Count sessions with each modality."""
        modalities = {}
        for subj in self.subjects.values():
            for ses_info in subj.sessions.values():
                for mod in ses_info.modalities.keys():
                    modalities[mod] = modalities.get(mod, 0) + 1
        return modalities

    def to_dict(self) -> dict:
        return {
            'bids_root': str(self.bids_root),
            'summary': {
                'n_subjects': self.n_subjects,
                'n_sessions': self.n_sessions,
                'modality_breakdown': self.get_modality_breakdown()
            },
            'subjects': {
                subj: info.to_dict()
                for subj, info in self.subjects.items()
            },
            'issues': self.issues
        }


def discover_bids_data(bids_root: Path) -> BIDSManifest:
    """
    Scan BIDS directory and return complete inventory.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS directory

    Returns
    -------
    BIDSManifest
        Complete inventory of subjects, sessions, and scans
    """
    manifest = BIDSManifest(bids_root=bids_root)

    # Define modality patterns for human MRI
    modality_patterns = {
        'anat': ['*_T1w.nii.gz', '*_T1w.nii', '*_T2w.nii.gz', '*_T2w.nii',
                 '*_FLAIR.nii.gz', '*_FLAIR.nii'],
        'dwi': ['*_dwi.nii.gz', '*_dwi.nii'],
        'func': ['*_bold.nii.gz', '*_bold.nii'],
        'asl': ['*_asl.nii.gz', '*_asl.nii', '*_m0scan.nii.gz'],
        'fmap': ['*_epi.nii.gz', '*_magnitude*.nii.gz', '*_phasediff.nii.gz',
                 '*_fieldmap.nii.gz'],
    }

    # Find all subject directories
    subject_dirs = sorted(bids_root.glob('sub-*'))

    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name
        subject_info = SubjectInfo(subject=subject)

        # Find all session directories
        session_dirs = sorted(subject_dir.glob('ses-*'))

        if not session_dirs:
            # No session structure - treat as single implicit session
            session_dirs = [subject_dir]
            has_sessions = False
        else:
            has_sessions = True

        for session_dir in session_dirs:
            if not session_dir.is_dir():
                continue

            session = session_dir.name if has_sessions else 'ses-01'
            session_info = SessionInfo(session=session)

            # Check each modality
            for modality, patterns in modality_patterns.items():
                modality_dir = session_dir / modality

                if not modality_dir.exists():
                    continue

                scans = []
                for pattern in patterns:
                    for scan_file in modality_dir.glob(pattern):
                        # Parse BIDS filename
                        scan_info = _parse_bids_filename(scan_file, modality)
                        if scan_info:
                            scans.append(scan_info)

                if scans:
                    session_info.modalities[modality] = scans

            if session_info.modalities:
                subject_info.sessions[session] = session_info
            else:
                manifest.issues.append({
                    'subject': subject,
                    'session': session,
                    'issue': 'No valid scans found',
                    'severity': 'warning'
                })

        if subject_info.sessions:
            manifest.subjects[subject] = subject_info

    return manifest


def _parse_bids_filename(file_path: Path, modality: str) -> Optional[ScanInfo]:
    """Parse BIDS filename to extract metadata."""
    name = file_path.name

    # Remove extension
    if name.endswith('.nii.gz'):
        name = name[:-7]
    elif name.endswith('.nii'):
        name = name[:-4]
    else:
        return None

    parts = name.split('_')

    # Last part is suffix (T1w, dwi, bold, etc.)
    suffix = parts[-1]

    # Parse optional fields
    run = None
    acquisition = None
    echo = None

    for part in parts:
        if part.startswith('run-'):
            run = part.replace('run-', '')
        elif part.startswith('acq-'):
            acquisition = part.replace('acq-', '')
        elif part.startswith('echo-'):
            echo = part.replace('echo-', '')

    return ScanInfo(
        file_path=file_path,
        modality=modality,
        suffix=suffix,
        run=run,
        acquisition=acquisition,
        echo=echo
    )


@dataclass
class DICOMSeriesInfo:
    """Information about a DICOM series."""
    series_dir: Path
    series_number: str
    series_description: str
    modality: Optional[str] = None
    n_files: int = 0
    patient_id: Optional[str] = None
    study_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'series_dir': str(self.series_dir),
            'series_number': self.series_number,
            'series_description': self.series_description,
            'modality': self.modality,
            'n_files': self.n_files,
            'patient_id': self.patient_id,
            'study_date': self.study_date
        }


@dataclass
class DICOMSubjectInfo:
    """Information about a DICOM subject/session."""
    subject_dir: Path
    subject_id: Optional[str] = None
    study_date: Optional[str] = None
    series: List[DICOMSeriesInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'subject_dir': str(self.subject_dir),
            'subject_id': self.subject_id,
            'study_date': self.study_date,
            'series': [s.to_dict() for s in self.series]
        }


@dataclass
class DICOMManifest:
    """Complete manifest of DICOM data."""
    dicom_root: Path
    subjects: List[DICOMSubjectInfo] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def n_series(self) -> int:
        return sum(len(s.series) for s in self.subjects)

    def get_modality_breakdown(self) -> Dict[str, int]:
        """Count series by detected modality."""
        modalities = {}
        for subj in self.subjects:
            for series in subj.series:
                mod = series.modality or 'unknown'
                modalities[mod] = modalities.get(mod, 0) + 1
        return modalities

    def to_dict(self) -> dict:
        return {
            'dicom_root': str(self.dicom_root),
            'summary': {
                'n_subjects': self.n_subjects,
                'n_series': self.n_series,
                'modality_breakdown': self.get_modality_breakdown()
            },
            'subjects': [s.to_dict() for s in self.subjects],
            'issues': self.issues
        }


def discover_dicom_data(dicom_root: Path) -> DICOMManifest:
    """
    Scan DICOM directory and return inventory of raw data.

    Supports common DICOM directory structures:
    - dicom_root/subject_folder/series_folder/
    - dicom_root/subject_folder/date_folder/series_folder/
    - dicom_root/date_folder/series_folder/

    Parameters
    ----------
    dicom_root : Path
        Root of DICOM data directory

    Returns
    -------
    DICOMManifest
        Inventory of all DICOM subjects and series
    """
    manifest = DICOMManifest(dicom_root=dicom_root)

    if not dicom_root.exists():
        manifest.issues.append({
            'issue': f'DICOM root not found: {dicom_root}',
            'severity': 'error'
        })
        return manifest

    # Try to import pydicom for detailed parsing
    try:
        import pydicom
        has_pydicom = True
    except ImportError:
        has_pydicom = False

    # Find subject directories (one level deep)
    subject_candidates = []
    for item in sorted(dicom_root.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            subject_candidates.append(item)

    for subject_dir in subject_candidates:
        subject_info = _parse_dicom_subject(subject_dir, has_pydicom)
        if subject_info and subject_info.series:
            manifest.subjects.append(subject_info)

    return manifest


def _parse_dicom_subject(subject_dir: Path, has_pydicom: bool = False) -> Optional[DICOMSubjectInfo]:
    """Parse a DICOM subject directory."""
    subject_info = DICOMSubjectInfo(subject_dir=subject_dir)

    # Try to extract subject ID from directory name
    dir_name = subject_dir.name
    subject_info.subject_id = dir_name

    # Find series directories
    # Could be directly under subject or under date subdirectories
    series_dirs = []

    for item in sorted(subject_dir.iterdir()):
        if item.is_dir():
            # Check if this looks like a series (has DICOM files)
            dcm_files = list(item.glob('*.dcm')) + list(item.glob('*.DCM'))
            if not dcm_files:
                # Could be files without extension
                dcm_files = [f for f in item.iterdir() if f.is_file() and not f.name.startswith('.')]

            if dcm_files:
                series_dirs.append((item, dcm_files))
            else:
                # Check one level deeper (date subdirectory)
                for sub_item in item.iterdir():
                    if sub_item.is_dir():
                        sub_dcm_files = list(sub_item.glob('*.dcm')) + list(sub_item.glob('*.DCM'))
                        if not sub_dcm_files:
                            sub_dcm_files = [f for f in sub_item.iterdir()
                                            if f.is_file() and not f.name.startswith('.')]
                        if sub_dcm_files:
                            series_dirs.append((sub_item, sub_dcm_files))

    # Parse each series
    for series_dir, dcm_files in series_dirs:
        series_info = _parse_dicom_series(series_dir, dcm_files, has_pydicom)
        if series_info:
            subject_info.series.append(series_info)

            # Update subject info from DICOM headers
            if series_info.patient_id and not subject_info.subject_id:
                subject_info.subject_id = series_info.patient_id
            if series_info.study_date and not subject_info.study_date:
                subject_info.study_date = series_info.study_date

    return subject_info if subject_info.series else None


def _parse_dicom_series(series_dir: Path, dcm_files: List[Path], has_pydicom: bool) -> Optional[DICOMSeriesInfo]:
    """Parse a DICOM series directory."""
    series_info = DICOMSeriesInfo(
        series_dir=series_dir,
        series_number=series_dir.name,
        series_description=series_dir.name,
        n_files=len(dcm_files)
    )

    # Try to read DICOM header for more info
    if has_pydicom and dcm_files:
        try:
            import pydicom
            dcm = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

            series_info.series_number = str(getattr(dcm, 'SeriesNumber', series_dir.name))
            series_info.series_description = getattr(dcm, 'SeriesDescription', series_dir.name)
            series_info.patient_id = getattr(dcm, 'PatientID', None)
            series_info.study_date = getattr(dcm, 'StudyDate', None)

            # Classify modality based on series description
            series_info.modality = _classify_dicom_modality(series_info.series_description)

        except Exception:
            pass

    # Fallback: classify based on directory name
    if not series_info.modality:
        series_info.modality = _classify_dicom_modality(series_dir.name)

    return series_info


def _classify_dicom_modality(description: str) -> Optional[str]:
    """Classify DICOM series into modality based on description."""
    description = description.lower()

    # T1w patterns
    t1w_patterns = ['t1', 'mprage', '3d_t1', 'ir_fspgr', 'bravo', 'mp2rage']
    if any(p in description for p in t1w_patterns):
        return 'anat_T1w'

    # T2w patterns
    t2w_patterns = ['t2w', 't2_', 'space', 't2_tse', 'flair']
    if any(p in description for p in t2w_patterns):
        if 'flair' in description:
            return 'anat_FLAIR'
        return 'anat_T2w'

    # DWI patterns
    dwi_patterns = ['dwi', 'dti', 'diff', 'ep2d_diff', 'hardi', 'dmri']
    if any(p in description for p in dwi_patterns):
        # Exclude scanner-processed maps
        exclude_patterns = ['adc', 'fa_', 'trace', 'colfa', 'tensor']
        if not any(ex in description for ex in exclude_patterns):
            return 'dwi'

    # Functional patterns
    func_patterns = ['bold', 'fmri', 'func', 'rest', 'task', 'epi']
    if any(p in description for p in func_patterns):
        return 'func'

    # ASL patterns
    asl_patterns = ['asl', 'pcasl', 'pasl', 'casl', 'cbf', 'perfusion']
    if any(p in description for p in asl_patterns):
        return 'asl'

    # Field map patterns
    fmap_patterns = ['fieldmap', 'field_map', 'b0_map', 'gre_field', 'se_epi']
    if any(p in description for p in fmap_patterns):
        return 'fmap'

    return None


def create_study_directories(study_root: Path, create_all: bool = False) -> Dict[str, Path]:
    """
    Create the study directory structure.

    Parameters
    ----------
    study_root : Path
        Root directory for the study
    create_all : bool
        If True, create all directories. If False, only create essential ones.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping directory names to paths
    """
    directories = {
        'study_root': study_root,
        'raw': study_root / 'raw',
        'dicom': study_root / 'raw' / 'dicom',
        'bids': study_root / 'raw' / 'bids',
        'derivatives': study_root / 'derivatives',
        'transforms': study_root / 'transforms',
        'qc': study_root / 'qc',
        'work': study_root / 'work',
        'analysis': study_root / 'analysis',
        'logs': study_root / 'logs',
    }

    # Create directories
    essential_dirs = ['study_root', 'raw', 'derivatives', 'transforms', 'qc', 'work', 'logs']

    for name, path in directories.items():
        if create_all or name in essential_dirs:
            path.mkdir(parents=True, exist_ok=True)

    return directories


def generate_config(
    study_root: Path,
    study_name: str,
    study_code: str,
    bids_root: Optional[Path] = None,
    dicom_root: Optional[Path] = None,
    freesurfer_subjects_dir: Optional[Path] = None,
    n_procs: int = 8,
    output_path: Optional[Path] = None
) -> Path:
    """
    Generate a study configuration file.

    Parameters
    ----------
    study_root : Path
        Root directory for the study
    study_name : str
        Human-readable study name
    study_code : str
        Short study code (e.g., 'STUDY01')
    bids_root : Path, optional
        Path to BIDS data (defaults to study_root/raw/bids)
    dicom_root : Path, optional
        Path to DICOM data (defaults to study_root/raw/dicom)
    freesurfer_subjects_dir : Path, optional
        Path to FreeSurfer SUBJECTS_DIR
    n_procs : int
        Number of processors for parallel execution
    output_path : Path, optional
        Where to save config (defaults to study_root/config.yaml)

    Returns
    -------
    Path
        Path to generated config file
    """
    if bids_root is None:
        bids_root = study_root / 'raw' / 'bids'

    if dicom_root is None:
        dicom_root = study_root / 'raw' / 'dicom'

    config = {
        'study': {
            'name': study_name,
            'code': study_code,
            'created': datetime.now().isoformat(),
        },
        'project_dir': str(study_root),
        'dicom_dir': str(dicom_root),
        'bids_dir': str(bids_root),
        'derivatives_dir': str(study_root / 'derivatives'),
        'work_dir': str(study_root / 'work'),
        'paths': {
            'logs': str(study_root / 'logs'),
            'transforms': str(study_root / 'transforms'),
            'qc': str(study_root / 'qc'),
            'analysis': str(study_root / 'analysis'),
        },
        'execution': {
            'plugin': 'MultiProc',
            'n_procs': n_procs,
            'hash_method': 'timestamp',
            'caching': True,
        },
        'templates': {
            'mni152_t1_1mm': '${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz',
            'mni152_t1_2mm': '${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz',
            'fmrib58_fa': '${FSLDIR}/data/standard/FMRIB58_FA_1mm.nii.gz',
        },
        'anatomical': {
            'bet': {
                'frac': 0.5,
                'robust': True,
            },
            'bias_correction': {
                'n_iterations': [50, 50, 30, 20],
                'shrink_factor': 3,
                'convergence_threshold': 0.001,
                'bspline_fitting_distance': 300,
            },
            'atropos': {
                'number_of_tissue_classes': 3,
                'initialization': 'KMeans',
                'n_iterations': 5,
                'convergence_threshold': 0.001,
                'mrf_smoothing_factor': 0.1,
                'mrf_radius': [1, 1, 1],
            },
            'registration': {
                'dof': 12,
                'interp': 'spline',
            },
            'run_qc': True,
        },
        'diffusion': {
            'bet': {
                'frac': 0.3,
            },
            'topup': {
                'readout_time': 0.05,
            },
            'eddy_config': {
                'use_cuda': True,
                'repol': True,
            },
            'bedpostx': {
                'enabled': True,
                'n_fibres': 3,
            },
            'advanced_models': {
                'dki': True,
                'noddi': True,
                'noddi_backend': 'amico',
            },
            'run_qc': True,
        },
        'functional': {
            'tr': 2.0,
            'bet': {
                'frac': 0.3,
            },
            'highpass': 0.01,
            'lowpass': 0.1,
            'fwhm': 6,
            'tedana': {
                'enabled': False,
                'tedpca': 'kic',
            },
            'ica_aroma': {
                'enabled': True,
            },
            'acompcor': {
                'enabled': True,
                'n_components': 5,
            },
            'run_qc': True,
        },
        'asl': {
            'bet': {
                'frac': 0.3,
            },
            'labeling_type': 'PCASL',
            'labeling_duration': 1.8,
            'post_labeling_delay': 1.8,
            'labeling_efficiency': 0.85,
            'm0_scale': 1.0,
            'run_qc': True,
        },
        'freesurfer': {
            'enabled': freesurfer_subjects_dir is not None,
            'subjects_dir': str(freesurfer_subjects_dir) if freesurfer_subjects_dir else '${SUBJECTS_DIR}',
        },
    }

    if output_path is None:
        output_path = study_root / 'config.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated config: {output_path}")

    return output_path


def setup_study(
    study_root: Path,
    study_name: str,
    study_code: str,
    bids_root: Optional[Path] = None,
    dicom_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
    freesurfer_subjects_dir: Optional[Path] = None,
    link_bids: bool = True,
    link_dicom: bool = True,
    n_procs: int = 8,
    force: bool = False
) -> Dict[str, Any]:
    """
    Initialize a complete study with directory structure, config, and data discovery.

    This is the main entry point for setting up a new study. It:
    1. Creates the directory structure
    2. Discovers raw data (DICOM) and/or BIDS data
    3. Generates or validates configuration
    4. Returns a comprehensive report

    Parameters
    ----------
    study_root : Path
        Root directory for the study
    study_name : str
        Human-readable study name
    study_code : str
        Short study code (e.g., 'STUDY01')
    bids_root : Path, optional
        Path to BIDS data. If None, looks for study_root/raw/bids
    dicom_root : Path, optional
        Path to raw DICOM data. If None, looks for study_root/raw/dicom
    config_path : Path, optional
        Path to existing config file. If provided, will validate instead of generate.
    freesurfer_subjects_dir : Path, optional
        Path to FreeSurfer SUBJECTS_DIR for connectivity analysis
    link_bids : bool
        If True and bids_root is outside study_root, create a symlink
    link_dicom : bool
        If True and dicom_root is outside study_root, create a symlink
    n_procs : int
        Number of processors for parallel execution
    force : bool
        If True, overwrite existing files/directories

    Returns
    -------
    Dict[str, Any]
        Initialization report with status of each step
    """
    study_root = Path(study_root).resolve()

    report = {
        'status': 'success',
        'study_root': str(study_root),
        'timestamp': datetime.now().isoformat(),
        'steps': {},
        'bids_manifest': None,
        'dicom_manifest': None,
        'next_steps': [],
    }

    print("=" * 70)
    print(f"Initializing Study: {study_name}")
    print(f"Study Root: {study_root}")
    print("=" * 70)

    # Step 1: Create directory structure
    print("\n[1/5] Creating directory structure...")
    try:
        directories = create_study_directories(study_root, create_all=False)
        report['steps']['directories'] = {
            'status': 'success',
            'created': [str(p) for p in directories.values() if p.exists()]
        }
        print(f"  Created {len([p for p in directories.values() if p.exists()])} directories")
    except Exception as e:
        report['steps']['directories'] = {'status': 'failed', 'error': str(e)}
        report['status'] = 'failed'
        return report

    # Step 2: Discover DICOM data
    print("\n[2/5] Discovering DICOM data...")
    if dicom_root is None:
        dicom_root = study_root / 'raw' / 'dicom'
    else:
        dicom_root = Path(dicom_root).resolve()

    # Create symlink if needed
    target_dicom = study_root / 'raw' / 'dicom'
    if dicom_root != target_dicom and link_dicom:
        if target_dicom.exists():
            if target_dicom.is_symlink():
                if force:
                    target_dicom.unlink()
                else:
                    print(f"  DICOM symlink already exists: {target_dicom}")

        if not target_dicom.exists() and dicom_root.exists():
            target_dicom.symlink_to(dicom_root)
            print(f"  Created symlink: {target_dicom} -> {dicom_root}")

    # Discover DICOM data
    if dicom_root.exists():
        dicom_manifest = discover_dicom_data(dicom_root)
        report['dicom_manifest'] = dicom_manifest.to_dict()
        report['steps']['dicom_discovery'] = {
            'status': 'success',
            'n_subjects': dicom_manifest.n_subjects,
            'n_series': dicom_manifest.n_series,
            'modalities': dicom_manifest.get_modality_breakdown(),
            'issues': len(dicom_manifest.issues)
        }
        print(f"  Found {dicom_manifest.n_subjects} subjects, {dicom_manifest.n_series} series")
        if dicom_manifest.get_modality_breakdown():
            print(f"  Modalities: {dicom_manifest.get_modality_breakdown()}")
    else:
        report['steps']['dicom_discovery'] = {
            'status': 'skipped',
            'reason': f'DICOM directory not found: {dicom_root}'
        }
        print(f"  DICOM directory not found: {dicom_root}")

    # Step 3: Handle BIDS data
    print("\n[3/5] Discovering BIDS data...")
    if bids_root is None:
        bids_root = study_root / 'raw' / 'bids'
    else:
        bids_root = Path(bids_root).resolve()

    # Create symlink if needed
    target_bids = study_root / 'raw' / 'bids'
    if bids_root != target_bids and link_bids:
        if target_bids.exists():
            if target_bids.is_symlink():
                if force:
                    target_bids.unlink()
                else:
                    print(f"  BIDS symlink already exists: {target_bids}")

        if not target_bids.exists() and bids_root.exists():
            target_bids.symlink_to(bids_root)
            print(f"  Created symlink: {target_bids} -> {bids_root}")

    # Discover BIDS data
    if bids_root.exists():
        bids_manifest = discover_bids_data(bids_root)
        report['bids_manifest'] = bids_manifest.to_dict()
        report['steps']['bids_discovery'] = {
            'status': 'success',
            'n_subjects': bids_manifest.n_subjects,
            'n_sessions': bids_manifest.n_sessions,
            'modalities': bids_manifest.get_modality_breakdown(),
            'issues': len(bids_manifest.issues)
        }
        print(f"  Found {bids_manifest.n_subjects} subjects, {bids_manifest.n_sessions} sessions")
        print(f"  Modalities: {bids_manifest.get_modality_breakdown()}")
        if bids_manifest.issues:
            print(f"  Warnings: {len(bids_manifest.issues)}")
    else:
        report['steps']['bids_discovery'] = {
            'status': 'skipped',
            'reason': f'BIDS directory not found: {bids_root}'
        }
        print(f"  BIDS directory not found: {bids_root}")
        bids_manifest = None

    # Step 4: Generate or validate config
    print("\n[4/5] Setting up configuration...")
    if config_path and Path(config_path).exists():
        # Validate existing config
        try:
            config = load_config(Path(config_path))
            report['steps']['config'] = {
                'status': 'success',
                'action': 'validated',
                'path': str(config_path)
            }
            print(f"  Validated existing config: {config_path}")
        except Exception as e:
            report['steps']['config'] = {
                'status': 'failed',
                'action': 'validation',
                'error': str(e)
            }
            report['status'] = 'partial'
            print(f"  Config validation failed: {e}")
    else:
        # Generate new config
        config_output = study_root / 'config.yaml'
        if config_output.exists() and not force:
            report['steps']['config'] = {
                'status': 'skipped',
                'reason': 'Config already exists (use force=True to overwrite)',
                'path': str(config_output)
            }
            print(f"  Config already exists: {config_output}")
        else:
            try:
                config_path = generate_config(
                    study_root=study_root,
                    study_name=study_name,
                    study_code=study_code,
                    bids_root=bids_root,
                    dicom_root=dicom_root,
                    freesurfer_subjects_dir=freesurfer_subjects_dir,
                    n_procs=n_procs,
                    output_path=config_output
                )
                report['steps']['config'] = {
                    'status': 'success',
                    'action': 'generated',
                    'path': str(config_path)
                }
            except Exception as e:
                report['steps']['config'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                report['status'] = 'partial'

    # Step 5: Save manifest and report
    print("\n[5/5] Saving study manifest...")
    try:
        manifest_path = study_root / 'study_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(report, f, indent=2)
        report['steps']['manifest'] = {
            'status': 'success',
            'path': str(manifest_path)
        }
        print(f"  Saved manifest: {manifest_path}")
    except Exception as e:
        report['steps']['manifest'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Generate next steps
    report['next_steps'] = _generate_next_steps(report, study_root)

    # Print summary
    print("\n" + "=" * 70)
    print("Study Initialization Complete!")
    print("=" * 70)
    print(f"\nStatus: {report['status'].upper()}")

    if report['next_steps']:
        print("\nNext Steps:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"  {i}. {step}")

    return report


def _generate_next_steps(report: dict, study_root: Path) -> List[str]:
    """Generate recommended next steps based on initialization results."""
    steps = []

    # Check if DICOM data was found but not BIDS
    dicom_info = report['steps'].get('dicom_discovery', {})
    bids_info = report['steps'].get('bids_discovery', {})

    has_dicom = dicom_info.get('status') == 'success' and dicom_info.get('n_series', 0) > 0
    has_bids = bids_info.get('status') == 'success' and bids_info.get('n_subjects', 0) > 0

    if has_dicom and not has_bids:
        # Need to convert DICOM to BIDS/NIfTI first
        steps.append(
            f"Convert DICOM data to NIfTI ({dicom_info.get('n_series', 0)} series found). "
            f"Run the preprocessing pipeline with --dicom-dir flag."
        )

    if has_bids:
        modalities = bids_info.get('modalities', {})
        n_subjects = bids_info.get('n_subjects', 0)

        steps.append(
            f"Run preprocessing pipeline: "
            f"uv run python run_simple_pipeline.py --subject <sub-ID> --nifti-dir {study_root}/raw/bids/<sub-ID> --config {study_root}/config.yaml"
        )

        if modalities.get('anat', 0) > 0:
            steps.append(f"  - Anatomical data available: {modalities['anat']} sessions")
        if modalities.get('dwi', 0) > 0:
            steps.append(f"  - DWI data available: {modalities['dwi']} sessions")
        if modalities.get('func', 0) > 0:
            steps.append(f"  - Functional data available: {modalities['func']} sessions")
        if modalities.get('asl', 0) > 0:
            steps.append(f"  - ASL data available: {modalities['asl']} sessions")

    elif not has_dicom:
        steps.append(f"Add raw DICOM data to: {study_root}/raw/dicom/")
        steps.append(f"Or add BIDS data to: {study_root}/raw/bids/")

    # Always recommend editing config
    steps.append(f"Review and customize config: {study_root}/config.yaml")

    return steps


def get_study_subjects(
    study_root: Path,
    modality: Optional[str] = None,
    exclude_failed: bool = True
) -> List[Dict[str, Any]]:
    """
    Get list of subjects ready for processing.

    Parameters
    ----------
    study_root : Path
        Study root directory
    modality : str, optional
        Filter by modality (e.g., 'anat', 'dwi', 'func', 'asl')
    exclude_failed : bool
        If True, exclude subjects with exclusion markers

    Returns
    -------
    List[Dict]
        List of subject info dicts with subject, session, available modalities
    """
    study_root = Path(study_root)
    bids_root = study_root / 'raw' / 'bids'

    if not bids_root.exists():
        return []

    manifest = discover_bids_data(bids_root)
    subjects = []

    for subj_id, subj_info in manifest.subjects.items():
        for ses_id, ses_info in subj_info.sessions.items():
            # Filter by modality
            if modality and modality not in ses_info.modalities:
                continue

            # Check for exclusion marker
            if exclude_failed:
                exclusion_file = (study_root / 'derivatives' / subj_id / ses_id /
                                 'anat' / '.preprocessing_failed')
                if exclusion_file.exists():
                    continue

            subjects.append({
                'subject': subj_id,
                'session': ses_id,
                'modalities': list(ses_info.modalities.keys())
            })

    return subjects


def print_study_summary(study_root: Path) -> None:
    """Print a summary of study status."""
    study_root = Path(study_root)

    print("=" * 70)
    print(f"Study Summary: {study_root.name}")
    print("=" * 70)

    # Check for manifest
    manifest_path = study_root / 'study_manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"\nInitialized: {manifest.get('timestamp', 'Unknown')}")

        # DICOM info
        dicom_info = manifest.get('steps', {}).get('dicom_discovery', {})
        if dicom_info.get('status') == 'success':
            print(f"\nDICOM Data:")
            print(f"  Subjects: {dicom_info.get('n_subjects', 0)}")
            print(f"  Series: {dicom_info.get('n_series', 0)}")
            print(f"  Modalities: {dicom_info.get('modalities', {})}")

        # BIDS info
        bids_info = manifest.get('steps', {}).get('bids_discovery', {})
        if bids_info.get('status') == 'success':
            print(f"\nBIDS Data:")
            print(f"  Subjects: {bids_info.get('n_subjects', 0)}")
            print(f"  Sessions: {bids_info.get('n_sessions', 0)}")
            print(f"  Modalities: {bids_info.get('modalities', {})}")
    else:
        print("\nStudy not initialized. Run setup_study() first.")
        return

    # Check preprocessing status
    derivatives_dir = study_root / 'derivatives'
    if derivatives_dir.exists():
        processed_subjects = list(derivatives_dir.glob('sub-*'))
        print(f"\nPreprocessing Status:")
        print(f"  Processed subjects: {len(processed_subjects)}")

        # Count by modality
        for modality in ['anat', 'dwi', 'func', 'asl']:
            count = len(list(derivatives_dir.glob(f'sub-*/{modality}')))
            if count > 0:
                print(f"    {modality}: {count}")

    print("=" * 70)
