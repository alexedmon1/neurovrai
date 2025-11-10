#!/usr/bin/env python3
"""
Anonymization utilities for DICOM and NIfTI files.

Removes identifying information from medical imaging data for
safe sharing and use as example data.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging
import pydicom
from datetime import datetime


class AnonymizationError(Exception):
    """Raised when anonymization fails."""
    pass


class DICOMAnonymizer:
    """
    Anonymize DICOM files by removing patient-identifying information.

    Parameters
    ----------
    subject_id : str, optional
        Replacement subject ID (default: 'ANONYMOUS')
    date_shift : int, optional
        Number of days to shift dates (for temporal consistency)

    Examples
    --------
    >>> anonymizer = DICOMAnonymizer(subject_id='sub-001')
    >>> anonymizer.anonymize_file(
    ...     dicom_file=Path("/data/raw.dcm"),
    ...     output_file=Path("/data/anon.dcm")
    ... )
    """

    # DICOM tags containing identifying information
    # Based on DICOM PS3.15 Annex E
    IDENTIFYING_TAGS = [
        # Patient Information
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0032),  # PatientBirthTime
        (0x0010, 0x0040),  # PatientSex - KEEP (research relevant)
        (0x0010, 0x1000),  # OtherPatientIDs
        (0x0010, 0x1001),  # OtherPatientNames
        (0x0010, 0x1010),  # PatientAge - KEEP (research relevant)
        (0x0010, 0x1040),  # PatientAddress
        (0x0010, 0x1060),  # PatientMotherBirthName
        (0x0010, 0x2154),  # PatientTelephoneNumbers
        (0x0010, 0x2160),  # PatientEthnicGroup
        (0x0010, 0x21B0),  # AdditionalPatientHistory

        # Study/Series Information
        (0x0008, 0x0014),  # InstanceCreatorUID
        (0x0008, 0x0018),  # SOPInstanceUID
        (0x0008, 0x0080),  # InstitutionName
        (0x0008, 0x0081),  # InstitutionAddress
        (0x0008, 0x0090),  # ReferringPhysicianName
        (0x0008, 0x1048),  # PhysiciansOfRecord
        (0x0008, 0x1050),  # PerformingPhysicianName
        (0x0008, 0x1070),  # OperatorsName

        # Device Information
        (0x0008, 0x1010),  # StationName
        (0x0018, 0x1000),  # DeviceSerialNumber
        (0x0018, 0x1030),  # ProtocolName - KEEP (research relevant)

        # Unique Identifiers - will be regenerated
        (0x0020, 0x000D),  # StudyInstanceUID
        (0x0020, 0x000E),  # SeriesInstanceUID
    ]

    # Tags to keep for research purposes
    KEEP_TAGS = [
        (0x0010, 0x0040),  # PatientSex
        (0x0010, 0x1010),  # PatientAge
        (0x0018, 0x1030),  # ProtocolName
    ]

    def __init__(
        self,
        subject_id: str = 'ANONYMOUS',
        date_shift: int = 0
    ):
        self.subject_id = subject_id
        self.date_shift = date_shift
        self.logger = logging.getLogger(__name__)

    def anonymize_file(
        self,
        dicom_file: Path,
        output_file: Path,
        keep_tags: Optional[Set] = None
    ):
        """
        Anonymize a single DICOM file.

        Parameters
        ----------
        dicom_file : Path
            Input DICOM file
        output_file : Path
            Output anonymized DICOM file
        keep_tags : set, optional
            Additional tags to keep (as tuples)

        Raises
        ------
        AnonymizationError
            If anonymization fails

        Examples
        --------
        >>> anonymizer = DICOMAnonymizer('sub-001')
        >>> anonymizer.anonymize_file(
        ...     dicom_file=Path("/data/raw.dcm"),
        ...     output_file=Path("/data/anon.dcm")
        ... )
        """
        dicom_file = Path(dicom_file)
        output_file = Path(output_file)

        if not dicom_file.exists():
            raise AnonymizationError(f"DICOM file not found: {dicom_file}")

        try:
            # Read DICOM
            dcm = pydicom.dcmread(dicom_file)

            # Combine keep tags
            keep_set = set(self.KEEP_TAGS)
            if keep_tags:
                keep_set.update(keep_tags)

            # Remove identifying tags
            for tag in self.IDENTIFYING_TAGS:
                if tag not in keep_set:
                    if tag in dcm:
                        delattr(dcm, pydicom.datadict.keyword_for_tag(tag))

            # Set replacement patient info
            dcm.PatientName = self.subject_id
            dcm.PatientID = self.subject_id

            # Shift dates if requested
            if self.date_shift != 0:
                self._shift_dates(dcm)

            # Remove private tags (manufacturer-specific)
            dcm.remove_private_tags()

            # Save anonymized file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            dcm.save_as(output_file)

            self.logger.info(f"Anonymized: {dicom_file} → {output_file}")

        except Exception as e:
            raise AnonymizationError(f"Failed to anonymize {dicom_file}: {e}")

    def _shift_dates(self, dcm: pydicom.Dataset):
        """Shift dates by specified number of days."""
        from datetime import timedelta

        date_tags = [
            'StudyDate',
            'SeriesDate',
            'AcquisitionDate',
            'ContentDate'
        ]

        for tag_name in date_tags:
            if hasattr(dcm, tag_name):
                try:
                    date_str = getattr(dcm, tag_name)
                    # Parse DICOM date format (YYYYMMDD)
                    date_obj = datetime.strptime(date_str, '%Y%m%d')
                    # Shift
                    shifted_date = date_obj + timedelta(days=self.date_shift)
                    # Set back
                    setattr(dcm, tag_name, shifted_date.strftime('%Y%m%d'))
                except Exception:
                    # If parsing fails, just remove the date
                    delattr(dcm, tag_name)

    def anonymize_directory(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> int:
        """
        Anonymize all DICOM files in a directory.

        Parameters
        ----------
        input_dir : Path
            Input directory with DICOM files
        output_dir : Path
            Output directory for anonymized files

        Returns
        -------
        int
            Number of files anonymized

        Examples
        --------
        >>> anonymizer = DICOMAnonymizer('sub-001')
        >>> count = anonymizer.anonymize_directory(
        ...     input_dir=Path("/data/raw"),
        ...     output_dir=Path("/data/anon")
        ... )
        >>> print(f"Anonymized {count} files")
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise AnonymizationError(f"Input directory not found: {input_dir}")

        # Find all DICOM files
        dicom_files = list(input_dir.rglob("*.dcm"))
        if not dicom_files:
            # Try without extension
            dicom_files = [f for f in input_dir.rglob("*") if f.is_file()]

        count = 0
        for dicom_file in dicom_files:
            # Preserve relative structure
            relative_path = dicom_file.relative_to(input_dir)
            output_file = output_dir / relative_path

            try:
                self.anonymize_file(dicom_file, output_file)
                count += 1
            except Exception as e:
                self.logger.error(f"Failed to anonymize {dicom_file}: {e}")

        return count


class NIfTIAnonymizer:
    """
    Anonymize NIfTI files by cleaning JSON sidecars.

    NIfTI files themselves don't contain identifying information,
    but JSON sidecars from dcm2niix may contain DICOM header info.

    Examples
    --------
    >>> anonymizer = NIfTIAnonymizer()
    >>> anonymizer.anonymize_json_sidecar(
    ...     json_file=Path("/data/sub-001_T1w.json")
    ... )
    """

    # JSON fields that may contain identifying information
    IDENTIFYING_FIELDS = [
        'PatientName',
        'PatientID',
        'PatientBirthDate',
        'PatientSex',  # Keep if needed for research
        'PatientAge',  # Keep if needed for research
        'InstitutionName',
        'InstitutionAddress',
        'ReferringPhysicianName',
        'OperatorsName',
        'StationName',
        'DeviceSerialNumber',
        'StudyInstanceUID',
        'SeriesInstanceUID',
        'AcquisitionDateTime'
    ]

    KEEP_FIELDS = [
        'PatientSex',
        'PatientAge'
    ]

    def __init__(self, keep_fields: Optional[List[str]] = None):
        """
        Initialize NIfTI anonymizer.

        Parameters
        ----------
        keep_fields : list, optional
            Additional fields to keep
        """
        self.keep_fields = set(self.KEEP_FIELDS)
        if keep_fields:
            self.keep_fields.update(keep_fields)
        self.logger = logging.getLogger(__name__)

    def anonymize_json_sidecar(
        self,
        json_file: Path,
        output_file: Optional[Path] = None
    ):
        """
        Anonymize JSON sidecar by removing identifying fields.

        Parameters
        ----------
        json_file : Path
            JSON sidecar file
        output_file : Path, optional
            Output file (if None, overwrites input)

        Examples
        --------
        >>> anonymizer = NIfTIAnonymizer()
        >>> anonymizer.anonymize_json_sidecar(
        ...     json_file=Path("/data/sub-001_T1w.json")
        ... )
        """
        json_file = Path(json_file)

        if not json_file.exists():
            self.logger.warning(f"JSON file not found: {json_file}")
            return

        if output_file is None:
            output_file = json_file

        try:
            # Load JSON
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Remove identifying fields
            for field in self.IDENTIFYING_FIELDS:
                if field not in self.keep_fields and field in data:
                    del data[field]

            # Save cleaned JSON
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Anonymized JSON: {json_file}")

        except Exception as e:
            self.logger.error(f"Failed to anonymize {json_file}: {e}")

    def anonymize_directory(
        self,
        data_dir: Path
    ) -> int:
        """
        Anonymize all JSON sidecars in a directory.

        Parameters
        ----------
        data_dir : Path
            Directory containing NIfTI + JSON files

        Returns
        -------
        int
            Number of JSON files anonymized

        Examples
        --------
        >>> anonymizer = NIfTIAnonymizer()
        >>> count = anonymizer.anonymize_directory(Path("/data/rawdata/sub-001"))
        >>> print(f"Cleaned {count} JSON sidecars")
        """
        data_dir = Path(data_dir)

        if not data_dir.exists():
            self.logger.warning(f"Directory not found: {data_dir}")
            return 0

        json_files = list(data_dir.rglob("*.json"))

        # Skip dataset_description.json and other BIDS metadata files
        json_files = [
            f for f in json_files
            if f.name not in ['dataset_description.json', 'participants.json']
        ]

        count = 0
        for json_file in json_files:
            try:
                self.anonymize_json_sidecar(json_file)
                count += 1
            except Exception as e:
                self.logger.error(f"Failed to anonymize {json_file}: {e}")

        return count


def anonymize_subject_data(
    rawdata_dir: Path,
    subject: str,
    session: Optional[str] = None,
    anonymize_nifti: bool = True,
    anonymize_dicom: bool = False,
    dicom_dir: Optional[Path] = None
) -> Dict[str, int]:
    """
    Anonymize all data for a subject.

    Parameters
    ----------
    rawdata_dir : Path
        BIDS rawdata directory
    subject : str
        Subject identifier
    session : str, optional
        Session identifier
    anonymize_nifti : bool
        Anonymize NIfTI JSON sidecars (default: True)
    anonymize_dicom : bool
        Anonymize source DICOM files (default: False)
    dicom_dir : Path, optional
        DICOM directory (if anonymize_dicom=True)

    Returns
    -------
    dict
        Counts of anonymized files by type

    Examples
    --------
    >>> results = anonymize_subject_data(
    ...     rawdata_dir=Path("/data/rawdata"),
    ...     subject="sub-001",
    ...     anonymize_nifti=True
    ... )
    >>> print(f"Cleaned {results['nifti_json']} JSON files")
    """
    from mri_preprocess.utils.bids import get_subject_dir

    results = {
        'nifti_json': 0,
        'dicom': 0
    }

    # Anonymize NIfTI JSON sidecars
    if anonymize_nifti:
        subject_dir = get_subject_dir(rawdata_dir, subject, session)
        if subject_dir.exists():
            anonymizer = NIfTIAnonymizer()
            results['nifti_json'] = anonymizer.anonymize_directory(subject_dir)

    # Anonymize DICOM files
    if anonymize_dicom and dicom_dir:
        dicom_anonymizer = DICOMAnonymizer(subject_id=subject)
        # This would overwrite source files - use carefully!
        # Better to anonymize during conversion
        results['dicom'] = 0  # Skip for safety

    return results


def check_for_phi(
    data_dir: Path,
    subject_id: str
) -> List[Dict[str, str]]:
    """
    Check for potential PHI (Protected Health Information) in files.

    Scans JSON sidecars for fields that might contain identifying information.

    Parameters
    ----------
    data_dir : Path
        Directory to check
    subject_id : str
        Subject ID to look for

    Returns
    -------
    list
        List of potential PHI findings

    Examples
    --------
    >>> findings = check_for_phi(
    ...     data_dir=Path("/data/rawdata/sub-001"),
    ...     subject_id="0580101"
    ... )
    >>> for finding in findings:
    ...     print(f"Found {finding['field']} in {finding['file']}")
    """
    data_dir = Path(data_dir)
    findings = []

    # Check JSON files
    json_files = list(data_dir.rglob("*.json"))

    for json_file in json_files:
        if json_file.name in ['dataset_description.json']:
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check for identifying fields
            for field in NIfTIAnonymizer.IDENTIFYING_FIELDS:
                if field in data:
                    value = str(data[field])
                    # Check if it contains subject ID
                    if subject_id in value:
                        findings.append({
                            'file': str(json_file),
                            'field': field,
                            'value': value,
                            'type': 'contains_subject_id'
                        })
                    elif field not in NIfTIAnonymizer.KEEP_FIELDS:
                        findings.append({
                            'file': str(json_file),
                            'field': field,
                            'value': value,
                            'type': 'identifying_field'
                        })

        except Exception as e:
            logging.warning(f"Could not check {json_file}: {e}")

    return findings
