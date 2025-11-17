#!/usr/bin/env python3
"""
BIDS conversion and organization utilities.

Handles organizing converted NIfTI files into BIDS-compliant structure.
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from neurovrai.preprocess.utils.bids import (
    get_modality_dir,
    build_bids_filename,
    save_json_sidecar
)
from neurovrai.preprocess.utils.file_finder import match_sequence


class BIDSConverter:
    """
    Convert and organize NIfTI files into BIDS structure.

    Parameters
    ----------
    config : dict
        Configuration dictionary with sequence_mappings
    rawdata_dir : Path
        BIDS rawdata directory
    subject : str
        Subject identifier
    session : str, optional
        Session identifier

    Examples
    --------
    >>> bids = BIDSConverter(config, Path("/data/rawdata"), "sub-001")
    >>> bids.organize_converted_files(
    ...     nifti_files=[Path("MPRAGE_001.nii.gz")],
    ...     source_dir=Path("/tmp/convert")
    ... )
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rawdata_dir: Path,
        subject: str,
        session: Optional[str] = None
    ):
        self.config = config
        self.rawdata_dir = Path(rawdata_dir)
        self.subject = subject
        self.session = session
        self.logger = logging.getLogger(__name__)

        # Modality directory mapping
        self.modality_bids_map = {
            't1w': ('anat', 'T1w'),
            't2w': ('anat', 'T2w'),
            'flair': ('anat', 'FLAIR'),
            'dwi': ('dwi', 'dwi'),
            'rest': ('func', 'bold'),
            'task': ('func', 'bold'),
            'asl': ('perf', 'asl'),
            'fmap': ('fmap', 'fieldmap')
        }

    def identify_modality(self, filename: str) -> Optional[Tuple[str, str, str]]:
        """
        Identify modality of a file based on sequence mappings.

        Parameters
        ----------
        filename : str
            Filename to classify

        Returns
        -------
        tuple or None
            (modality_key, bids_dir, bids_suffix) or None if unidentified

        Examples
        --------
        >>> bids = BIDSConverter(config, rawdata_dir, "sub-001")
        >>> modality, bids_dir, suffix = bids.identify_modality("MPRAGE_001.nii.gz")
        >>> # ('t1w', 'anat', 'T1w')
        """
        if 'sequence_mappings' not in self.config:
            self.logger.warning("No sequence_mappings in config")
            return None

        for modality, patterns in self.config['sequence_mappings'].items():
            if match_sequence(filename, patterns):
                if modality in self.modality_bids_map:
                    bids_dir, bids_suffix = self.modality_bids_map[modality]
                    return modality, bids_dir, bids_suffix

        return None

    def organize_file(
        self,
        source_file: Path,
        modality_key: str,
        bids_dir: str,
        bids_suffix: str,
        run: Optional[int] = None,
        echo: Optional[int] = None,
        task: Optional[str] = None
    ) -> Path:
        """
        Move and rename file to BIDS structure.

        Parameters
        ----------
        source_file : Path
            Source NIfTI file
        modality_key : str
            Modality key (e.g., 't1w', 'dwi')
        bids_dir : str
            BIDS directory name (e.g., 'anat', 'func')
        bids_suffix : str
            BIDS suffix (e.g., 'T1w', 'bold')
        run : int, optional
            Run number
        echo : int, optional
            Echo number (for multi-echo)
        task : str, optional
            Task name (for functional)

        Returns
        -------
        Path
            Path to organized file

        Examples
        --------
        >>> dest = bids.organize_file(
        ...     source_file=Path("/tmp/MPRAGE_001.nii.gz"),
        ...     modality_key='t1w',
        ...     bids_dir='anat',
        ...     bids_suffix='T1w'
        ... )
        """
        # Get modality directory
        modality_dir = get_modality_dir(
            self.rawdata_dir,
            self.subject,
            bids_dir,
            session=self.session,
            create=True
        )

        # Build BIDS filename
        bids_filename = build_bids_filename(
            subject=self.subject,
            modality=bids_dir,
            suffix=bids_suffix,
            session=self.session,
            task=task,
            echo=echo,
            run=run,
            extension='.nii.gz'
        )

        dest_file = modality_dir / bids_filename

        # Move file
        if source_file != dest_file:
            shutil.move(str(source_file), str(dest_file))
            self.logger.info(f"Organized: {source_file.name} → {dest_file}")

        # Handle JSON sidecar if it exists
        json_source = source_file.with_suffix('').with_suffix('.json')
        if json_source.exists():
            json_dest = dest_file.with_suffix('').with_suffix('.json')
            shutil.move(str(json_source), str(json_dest))
            self.logger.info(f"Moved sidecar: {json_source.name} → {json_dest}")

        # Handle bval/bvec for diffusion
        if modality_key == 'dwi':
            for ext in ['.bval', '.bvec']:
                aux_source = source_file.with_suffix('').with_suffix(ext)
                if aux_source.exists():
                    aux_dest = dest_file.with_suffix('').with_suffix(ext)
                    shutil.move(str(aux_source), str(aux_dest))
                    self.logger.info(f"Moved {ext}: {aux_source.name} → {aux_dest}")

        return dest_file

    def organize_converted_files(
        self,
        nifti_files: List[Path],
        source_dir: Path
    ) -> Dict[str, List[Path]]:
        """
        Organize all converted files into BIDS structure.

        Parameters
        ----------
        nifti_files : list of Path
            List of converted NIfTI files
        source_dir : Path
            Source directory containing files

        Returns
        -------
        dict
            Dictionary mapping modality -> list of organized files

        Examples
        --------
        >>> files = [
        ...     Path("/tmp/MPRAGE_001.nii.gz"),
        ...     Path("/tmp/DTI_002.nii.gz")
        ... ]
        >>> organized = bids.organize_converted_files(files, Path("/tmp"))
        >>> print(organized['anat'])  # List of anatomical files
        """
        organized_files = {}

        for nifti_file in nifti_files:
            # Identify modality
            modality_info = self.identify_modality(nifti_file.name)

            if modality_info is None:
                self.logger.warning(f"Could not identify modality: {nifti_file.name}")
                continue

            modality_key, bids_dir, bids_suffix = modality_info

            # Check for multi-echo (look for echo number in filename)
            echo = None
            if '_echo-' in nifti_file.name.lower():
                import re
                match = re.search(r'echo-?(\d+)', nifti_file.name, re.IGNORECASE)
                if match:
                    echo = int(match.group(1))

            # Check for run number
            run = None
            if '_run-' in nifti_file.name.lower():
                import re
                match = re.search(r'run-?(\d+)', nifti_file.name, re.IGNORECASE)
                if match:
                    run = int(match.group(1))

            # Check for task name (for functional)
            task = None
            if bids_dir == 'func':
                # Try to extract task from filename
                import re
                match = re.search(r'task-?(\w+)', nifti_file.name, re.IGNORECASE)
                if match:
                    task = match.group(1)
                else:
                    # Default task name for resting-state
                    if modality_key == 'rest':
                        task = 'rest'

            # Organize file
            try:
                dest_file = self.organize_file(
                    source_file=nifti_file,
                    modality_key=modality_key,
                    bids_dir=bids_dir,
                    bids_suffix=bids_suffix,
                    run=run,
                    echo=echo,
                    task=task
                )

                if bids_dir not in organized_files:
                    organized_files[bids_dir] = []
                organized_files[bids_dir].append(dest_file)

            except Exception as e:
                self.logger.error(f"Failed to organize {nifti_file}: {e}")

        return organized_files

    def create_dataset_description(self):
        """
        Create dataset_description.json for BIDS compliance.

        Creates the required BIDS dataset_description.json file in
        the rawdata directory if it doesn't exist.

        Examples
        --------
        >>> bids = BIDSConverter(config, rawdata_dir, "sub-001")
        >>> bids.create_dataset_description()
        """
        desc_file = self.rawdata_dir / 'dataset_description.json'

        if desc_file.exists():
            return

        study_name = self.config.get('study', {}).get('name', 'Unknown Study')
        study_code = self.config.get('study', {}).get('code', 'UNKNOWN')

        description = {
            'Name': study_name,
            'BIDSVersion': '1.8.0',
            'DatasetType': 'raw',
            'License': 'Unknown',
            'Authors': [],
            'Acknowledgements': '',
            'HowToAcknowledge': '',
            'Funding': [],
            'ReferencesAndLinks': [],
            'DatasetDOI': '',
            'GeneratedBy': [
                {
                    'Name': 'mri-preprocess',
                    'Version': '0.1.0',
                    'CodeURL': ''
                }
            ]
        }

        with open(desc_file, 'w') as f:
            json.dump(description, f, indent=2)

        self.logger.info(f"Created dataset_description.json")

    def validate_bids_structure(self) -> Tuple[bool, List[str]]:
        """
        Validate BIDS structure for subject.

        Returns
        -------
        tuple
            (is_valid, issues) - validation result and list of issues

        Examples
        --------
        >>> is_valid, issues = bids.validate_bids_structure()
        >>> if not is_valid:
        ...     for issue in issues:
        ...         print(f"Issue: {issue}")
        """
        issues = []

        # Check dataset_description.json
        desc_file = self.rawdata_dir / 'dataset_description.json'
        if not desc_file.exists():
            issues.append("Missing dataset_description.json")

        # Check subject directory exists
        from neurovrai.preprocess.utils.bids import get_subject_dir
        subject_dir = get_subject_dir(self.rawdata_dir, self.subject, self.session)
        if not subject_dir.exists():
            issues.append(f"Subject directory not found: {subject_dir}")
            return False, issues

        # Check for at least one modality directory
        modality_dirs = ['anat', 'func', 'dwi', 'fmap', 'perf']
        has_data = False
        for modality in modality_dirs:
            mod_dir = subject_dir / modality
            if mod_dir.exists() and list(mod_dir.glob('*.nii*')):
                has_data = True
                break

        if not has_data:
            issues.append("No imaging data found in any modality directory")

        return len(issues) == 0, issues


def convert_and_organize(
    dicom_dir: Path,
    rawdata_dir: Path,
    subject: str,
    config: Dict[str, Any],
    session: Optional[str] = None,
    temp_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Complete DICOM to BIDS conversion pipeline.

    Converts DICOM to NIfTI and organizes into BIDS structure.

    Parameters
    ----------
    dicom_dir : Path
        Input DICOM directory
    rawdata_dir : Path
        Output BIDS rawdata directory
    subject : str
        Subject identifier
    config : dict
        Configuration
    session : str, optional
        Session identifier
    temp_dir : Path, optional
        Temporary directory for conversion

    Returns
    -------
    dict
        Conversion results

    Examples
    --------
    >>> result = convert_and_organize(
    ...     dicom_dir=Path("/data/sourcedata/sub-001"),
    ...     rawdata_dir=Path("/data/rawdata"),
    ...     subject="sub-001",
    ...     config=config
    ... )
    >>> print(f"Organized {len(result['organized_files'])} files")
    """
    from neurovrai.preprocess.dicom.converter import convert_dicom_to_nifti
    import tempfile

    # Use temp directory for initial conversion
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix='dcm2niix_'))
    else:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Convert DICOM to NIfTI
        logger = logging.getLogger(__name__)
        logger.info("Converting DICOM to NIfTI...")

        conversion_result = convert_dicom_to_nifti(
            dicom_dir=dicom_dir,
            output_dir=temp_dir,
            config=config
        )

        if not conversion_result['success']:
            return {
                'success': False,
                'error': 'DICOM conversion failed',
                'details': conversion_result
            }

        # Step 2: Organize into BIDS
        logger.info("Organizing files into BIDS structure...")

        bids_converter = BIDSConverter(
            config=config,
            rawdata_dir=rawdata_dir,
            subject=subject,
            session=session
        )

        organized_files = bids_converter.organize_converted_files(
            nifti_files=conversion_result['nifti_files'],
            source_dir=temp_dir
        )

        # Step 3: Create dataset description
        bids_converter.create_dataset_description()

        # Step 4: Validate
        is_valid, issues = bids_converter.validate_bids_structure()

        return {
            'success': True,
            'organized_files': organized_files,
            'conversion_result': conversion_result,
            'validation': {
                'is_valid': is_valid,
                'issues': issues
            }
        }

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
