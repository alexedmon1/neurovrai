#!/usr/bin/env python3
"""
DICOM to NIfTI conversion using dcm2niix.

Provides a Python wrapper around dcm2niix with configuration support.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging


class DCM2NIIXError(Exception):
    """Raised when dcm2niix conversion fails."""
    pass


class DCM2NIIXConverter:
    """
    Wrapper for dcm2niix DICOM to NIfTI conversion.

    Parameters
    ----------
    dcm2niix_path : Path, optional
        Path to dcm2niix executable (if not in PATH)
    compression_level : int
        Gzip compression level (1-9, default: 6)
    anonymize : bool
        Strip patient information (default: False)
    merge_2d : bool
        Merge 2D slices into 3D volume (default: True)

    Examples
    --------
    >>> converter = DCM2NIIXConverter(anonymize=True)
    >>> results = converter.convert(
    ...     dicom_dir=Path("/data/DICOM/sub-001"),
    ...     output_dir=Path("/data/rawdata/sub-001")
    ... )
    >>> for nifti in results['nifti_files']:
    ...     print(nifti)
    """

    def __init__(
        self,
        dcm2niix_path: Optional[Path] = None,
        compression_level: int = 6,
        anonymize: bool = False,
        merge_2d: bool = True
    ):
        self.dcm2niix_path = dcm2niix_path or 'dcm2niix'
        self.compression_level = compression_level
        self.anonymize = anonymize
        self.merge_2d = merge_2d
        self.logger = logging.getLogger(__name__)

        # Verify dcm2niix is available
        self._check_dcm2niix()

    def _check_dcm2niix(self):
        """Check that dcm2niix is available."""
        try:
            result = subprocess.run(
                [str(self.dcm2niix_path), '-h'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise DCM2NIIXError("dcm2niix not working correctly")
        except FileNotFoundError:
            raise DCM2NIIXError(
                f"dcm2niix not found at {self.dcm2niix_path}. "
                "Please install dcm2niix: https://github.com/rordenlab/dcm2niix"
            )
        except subprocess.TimeoutExpired:
            raise DCM2NIIXError("dcm2niix timed out")

    def build_command(
        self,
        dicom_dir: Path,
        output_dir: Path,
        filename_format: str = '%f_%p_%t_%s',
        additional_flags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Build dcm2niix command with options.

        Parameters
        ----------
        dicom_dir : Path
            Input DICOM directory
        output_dir : Path
            Output directory for NIfTI files
        filename_format : str
            Output filename format (dcm2niix format string)
            Default: '%f_%p_%t_%s' (folder_protocol_time_series)
        additional_flags : list, optional
            Additional command-line flags

        Returns
        -------
        list
            Command as list of strings

        Notes
        -----
        dcm2niix format specifiers:
        - %f : folder name
        - %p : protocol name
        - %t : time
        - %s : series number
        - %d : description
        - %i : patient ID
        - %n : patient name
        """
        cmd = [str(self.dcm2niix_path)]

        # Compression level
        cmd.extend(['-z', str(self.compression_level)])

        # Output format (always NIfTI)
        cmd.extend(['-f', filename_format])

        # Output directory
        cmd.extend(['-o', str(output_dir)])

        # Merge 2D slices
        if self.merge_2d:
            cmd.append('-m')
            cmd.append('y')
        else:
            cmd.append('-m')
            cmd.append('n')

        # Anonymization
        if self.anonymize:
            cmd.extend(['-ba', 'y'])  # Anonymize BIDS sidecar

        # Create BIDS sidecar JSON
        cmd.extend(['-b', 'y'])

        # Single file per folder (avoid duplicates)
        cmd.extend(['-s', 'n'])

        # Verbose output
        cmd.extend(['-v', 'y'])

        # Additional flags
        if additional_flags:
            cmd.extend(additional_flags)

        # Input directory (last argument)
        cmd.append(str(dicom_dir))

        return cmd

    def convert(
        self,
        dicom_dir: Path,
        output_dir: Path,
        filename_format: str = '%f_%p_%t_%s',
        additional_flags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convert DICOM directory to NIfTI.

        Parameters
        ----------
        dicom_dir : Path
            Input DICOM directory
        output_dir : Path
            Output directory for NIfTI files
        filename_format : str
            Output filename format
        additional_flags : list, optional
            Additional dcm2niix flags

        Returns
        -------
        dict
            Conversion results with keys:
            - 'success': bool
            - 'nifti_files': list of Path
            - 'json_files': list of Path
            - 'stdout': str
            - 'stderr': str

        Raises
        ------
        DCM2NIIXError
            If conversion fails

        Examples
        --------
        >>> converter = DCM2NIIXConverter()
        >>> result = converter.convert(
        ...     dicom_dir=Path("/data/DICOM/001"),
        ...     output_dir=Path("/data/rawdata/sub-001/anat")
        ... )
        >>> print(f"Converted {len(result['nifti_files'])} files")
        """
        dicom_dir = Path(dicom_dir)
        output_dir = Path(output_dir)

        if not dicom_dir.exists():
            raise DCM2NIIXError(f"DICOM directory not found: {dicom_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = self.build_command(
            dicom_dir,
            output_dir,
            filename_format,
            additional_flags
        )

        self.logger.info(f"Running dcm2niix: {' '.join(cmd)}")

        # Run conversion
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            stdout = result.stdout
            stderr = result.stderr

            self.logger.debug(f"dcm2niix stdout:\n{stdout}")
            if stderr:
                self.logger.warning(f"dcm2niix stderr:\n{stderr}")

            # Check for errors
            if result.returncode != 0:
                raise DCM2NIIXError(
                    f"dcm2niix failed with return code {result.returncode}\n"
                    f"stderr: {stderr}"
                )

            # Find converted files
            nifti_files = sorted(output_dir.glob("*.nii*"))
            json_files = sorted(output_dir.glob("*.json"))

            return {
                'success': True,
                'nifti_files': nifti_files,
                'json_files': json_files,
                'stdout': stdout,
                'stderr': stderr
            }

        except subprocess.TimeoutExpired:
            raise DCM2NIIXError(f"dcm2niix timed out after 600 seconds")

    def convert_series(
        self,
        series_dirs: List[Path],
        output_dir: Path,
        filename_format: str = '%f_%p_%t_%s'
    ) -> List[Dict[str, Any]]:
        """
        Convert multiple DICOM series.

        Parameters
        ----------
        series_dirs : list of Path
            List of DICOM series directories
        output_dir : Path
            Output directory
        filename_format : str
            Filename format

        Returns
        -------
        list
            List of conversion results (one per series)

        Examples
        --------
        >>> series = [
        ...     Path("/data/DICOM/sub-001/T1w"),
        ...     Path("/data/DICOM/sub-001/T2w"),
        ...     Path("/data/DICOM/sub-001/DWI")
        ... ]
        >>> results = converter.convert_series(
        ...     series_dirs=series,
        ...     output_dir=Path("/data/rawdata/sub-001")
        ... )
        """
        results = []

        for series_dir in series_dirs:
            self.logger.info(f"Converting series: {series_dir.name}")

            try:
                result = self.convert(
                    dicom_dir=series_dir,
                    output_dir=output_dir,
                    filename_format=filename_format
                )
                result['series_dir'] = series_dir
                results.append(result)

            except DCM2NIIXError as e:
                self.logger.error(f"Failed to convert {series_dir}: {e}")
                results.append({
                    'success': False,
                    'series_dir': series_dir,
                    'error': str(e)
                })

        return results

    def get_version(self) -> str:
        """
        Get dcm2niix version.

        Returns
        -------
        str
            Version string

        Examples
        --------
        >>> converter = DCM2NIIXConverter()
        >>> print(converter.get_version())
        'v1.0.20220720'
        """
        try:
            result = subprocess.run(
                [str(self.dcm2niix_path), '-v'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Parse version from output
            for line in result.stdout.split('\n'):
                if 'version' in line.lower():
                    return line.strip()
            return result.stdout.strip()

        except Exception as e:
            self.logger.warning(f"Could not get dcm2niix version: {e}")
            return "unknown"


def convert_dicom_to_nifti(
    dicom_dir: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for DICOM to NIfTI conversion.

    Parameters
    ----------
    dicom_dir : Path
        Input DICOM directory
    output_dir : Path
        Output directory
    config : dict, optional
        Configuration dictionary with dcm2niix settings

    Returns
    -------
    dict
        Conversion results

    Examples
    --------
    >>> config = {
    ...     'dicom': {
    ...         'anonymize': True,
    ...         'compression_level': 9
    ...     }
    ... }
    >>> result = convert_dicom_to_nifti(
    ...     dicom_dir=Path("/data/DICOM/sub-001"),
    ...     output_dir=Path("/data/rawdata/sub-001"),
    ...     config=config
    ... )
    """
    # Extract settings from config
    if config and 'dicom' in config:
        dicom_config = config['dicom']
        anonymize = dicom_config.get('anonymize', False)
        compression = dicom_config.get('compression_level', 6)
        merge_2d = dicom_config.get('merge_2d', True)
    else:
        anonymize = False
        compression = 6
        merge_2d = True

    # Create converter
    converter = DCM2NIIXConverter(
        compression_level=compression,
        anonymize=anonymize,
        merge_2d=merge_2d
    )

    # Run conversion
    return converter.convert(dicom_dir, output_dir)


def batch_convert_subjects(
    sourcedata_dir: Path,
    rawdata_dir: Path,
    subjects: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Batch convert multiple subjects from sourcedata to rawdata.

    Parameters
    ----------
    sourcedata_dir : Path
        Root directory containing DICOM data
    rawdata_dir : Path
        Output BIDS rawdata directory
    subjects : list of str
        List of subject IDs
    config : dict, optional
        Configuration

    Returns
    -------
    dict
        Results for each subject

    Examples
    --------
    >>> results = batch_convert_subjects(
    ...     sourcedata_dir=Path("/data/sourcedata"),
    ...     rawdata_dir=Path("/data/rawdata"),
    ...     subjects=["001", "002", "003"],
    ...     config=config
    ... )
    >>> for subject, result in results.items():
    ...     print(f"{subject}: {len(result['nifti_files'])} files")
    """
    from neurovrai.preprocess.utils.bids import get_subject_dir

    results = {}

    for subject in subjects:
        print(f"\nConverting subject: {subject}")

        # Build paths
        subject_dicom_dir = get_subject_dir(sourcedata_dir, subject)
        subject_output_dir = get_subject_dir(rawdata_dir, subject, create=True)

        if not subject_dicom_dir.exists():
            print(f"Warning: DICOM directory not found for {subject}")
            results[subject] = {'success': False, 'error': 'Directory not found'}
            continue

        # Convert
        try:
            result = convert_dicom_to_nifti(
                dicom_dir=subject_dicom_dir,
                output_dir=subject_output_dir,
                config=config
            )
            results[subject] = result

        except Exception as e:
            print(f"Error converting {subject}: {e}")
            results[subject] = {'success': False, 'error': str(e)}

    return results
