#!/usr/bin/env python3
"""
DICOM to NIfTI Converter with Automatic Parameter Extraction

This module provides unified DICOM conversion for all MRI modalities:
- Anatomical (T1w, T2w)
- Diffusion (DWI)
- Functional (resting-state fMRI, multi-echo)
- Arterial Spin Labeling (ASL)

Features:
- Automatic modality detection based on SeriesDescription
- Parameter extraction from DICOM headers
- Organized output structure: {output_dir}/{subject}/{modality}/
- JSON sidecar files with acquisition parameters
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pydicom

logger = logging.getLogger(__name__)


class ModalityDetector:
    """Detect MRI modality from DICOM SeriesDescription."""

    # Sequence name patterns for each modality
    MODALITY_PATTERNS = {
        'anat': [
            'T2W CS5 OF1 TR2500',
            'T2W Sagittal Reformat',
            '3D_T1_TFE_SAG_CS3',
            'AX T1 MPR',
            'COR T1 MPR'
        ],
        'func': [
            'fMRI_CORRECTION_MB3_ME3_SENSE3_3mm_TR1104_TE15_DTE20',
            'RESTING ME3 MB3 SENSE3',
            'RESTING_ME3_MB3_SENSE3',
            'RESTING STATE'
        ],
        'dwi': [
            'SE_EPI Posterior',
            'DelRec - DTI_1shell_b3000_MB4',
            'DelRec - DTI_2shell_b1000_b2000_MB4',
            'dWIP DTI_32_2.37mm CLEAR',
            'facWIP DTI_32_2.37mm CLEAR',
            'DTI_32_2.37mm',
            'isoWIP DTI_32_2.37mm CLEAR'
        ],
        'asl': [
            'PCA_PRE_INT',
            'DelRec - pCASL1',
            'WIP SOURCE - DelRec - pCASL1'
        ],
        'fmap': [
            'SE_EPI Posterior',  # Field map for distortion correction
            'SE EPI'
        ]
    }

    @classmethod
    def detect_modality(cls, series_description: str) -> Optional[str]:
        """
        Detect modality from DICOM SeriesDescription.

        Args:
            series_description: DICOM SeriesDescription field

        Returns:
            Modality string ('anat', 'dwi', 'func', 'asl', 'fmap') or None
        """
        for modality, patterns in cls.MODALITY_PATTERNS.items():
            for pattern in patterns:
                if pattern in series_description:
                    return modality

        logger.warning(f"Unknown sequence: {series_description}")
        return None


class DICOMParameterExtractor:
    """Extract acquisition parameters from DICOM headers."""

    @staticmethod
    def extract_common_params(dcm: pydicom.Dataset) -> Dict:
        """Extract common parameters present in all modalities."""
        params = {
            'SeriesDescription': str(dcm.get('SeriesDescription', 'Unknown')),
            'SeriesNumber': int(dcm.get('SeriesNumber', 0)),
            'AcquisitionDate': str(dcm.get('AcquisitionDate', '')),
            'AcquisitionTime': str(dcm.get('AcquisitionTime', '')),
            'Manufacturer': str(dcm.get('Manufacturer', 'Unknown')),
            'ManufacturerModelName': str(dcm.get('ManufacturerModelName', 'Unknown')),
            'MagneticFieldStrength': float(dcm.get('MagneticFieldStrength', 0)),
            'ImageOrientationPatient': [float(x) for x in dcm.get('ImageOrientationPatient', [])],
        }
        return params

    @staticmethod
    def extract_anatomical_params(dcm: pydicom.Dataset) -> Dict:
        """Extract anatomical-specific parameters."""
        params = DICOMParameterExtractor.extract_common_params(dcm)
        params.update({
            'PixelSpacing': [float(x) for x in dcm.get('PixelSpacing', [])],
            'SliceThickness': float(dcm.get('SliceThickness', 0)),
            'RepetitionTime': float(dcm.get('RepetitionTime', 0)),
            'EchoTime': float(dcm.get('EchoTime', 0)),
            'FlipAngle': float(dcm.get('FlipAngle', 0)),
        })
        return params

    @staticmethod
    def extract_dwi_params(dcm: pydicom.Dataset) -> Dict:
        """Extract DWI-specific parameters."""
        params = DICOMParameterExtractor.extract_common_params(dcm)

        # Phase encoding direction (for TOPUP)
        phase_encoding = None
        if hasattr(dcm, 'InPlanePhaseEncodingDirection'):
            phase_encoding = dcm.InPlanePhaseEncodingDirection

        # Philips-specific tags
        pe_direction = None
        readout_time = None
        try:
            # Philips private tags
            if [0x0018, 0x9078] in dcm:  # PhaseEncodingDirectionPositive
                pe_positive = int(dcm[0x0018, 0x9078].value)
                pe_direction = 'AP' if pe_positive else 'PA'

            # Try to get effective echo spacing
            if [0x0043, 0x102c] in dcm:  # Philips private
                echo_spacing = float(dcm[0x0043, 0x102c].value)
                # Calculate readout time
                if hasattr(dcm, 'AcquisitionMatrix'):
                    matrix = dcm.AcquisitionMatrix
                    pe_steps = max([x for x in matrix if x > 0])
                    readout_time = echo_spacing * (pe_steps - 1) / 1000.0  # Convert to seconds
        except Exception as e:
            logger.warning(f"Could not extract DWI timing parameters: {e}")

        # Get b-value (try tag first, then string key)
        bval = 0
        if (0x0018, 0x9087) in dcm:
            bval = float(dcm[0x0018, 0x9087].value)
        else:
            bval = float(dcm.get('DiffusionBValue', 0))

        params.update({
            'RepetitionTime': float(dcm.get('RepetitionTime', 0)),
            'EchoTime': float(dcm.get('EchoTime', 0)),
            'PhaseEncodingDirection': phase_encoding,
            'EstimatedPEDirection': pe_direction,
            'EstimatedReadoutTime': readout_time,
            'DiffusionBValue': bval,
        })
        return params

    @staticmethod
    def extract_functional_params(dcm: pydicom.Dataset) -> Dict:
        """Extract functional MRI-specific parameters."""
        params = DICOMParameterExtractor.extract_common_params(dcm)

        # Multi-echo TEs
        echo_time = float(dcm.get('EchoTime', 0))
        echo_times = [echo_time]

        # Try to get all echo times for multi-echo sequences
        try:
            if hasattr(dcm, 'EchoNumbers'):
                # This is a multi-echo sequence
                # Echo times are typically stored in per-frame functional groups
                pass
        except Exception:
            pass

        params.update({
            'RepetitionTime': float(dcm.get('RepetitionTime', 0)) / 1000.0,  # Convert to seconds
            'EchoTime': echo_time,
            'EchoTimes': echo_times,
            'FlipAngle': float(dcm.get('FlipAngle', 0)),
            'SliceTiming': None,  # Will be extracted from slice positions if available
        })
        return params

    @staticmethod
    def extract_asl_params(dcm: pydicom.Dataset) -> Dict:
        """
        Extract ASL-specific parameters.
        Uses the validated ASL parameter extractor.
        """
        from neurovrai.preprocess.utils.dicom_asl_params import extract_asl_parameters

        params = DICOMParameterExtractor.extract_common_params(dcm)

        # Use existing validated ASL extractor
        asl_params = extract_asl_parameters(dcm)
        params.update(asl_params)

        return params


class DICOMConverter:
    """
    Convert DICOM directories to NIfTI with parameter extraction.

    This class handles:
    - Automatic modality detection
    - dcm2niix conversion
    - Parameter extraction and JSON sidecar creation
    - Organized output structure
    """

    def __init__(self, dicom_dir: Path, output_dir: Path, subject: str):
        """
        Initialize DICOM converter.

        Args:
            dicom_dir: Path to subject's DICOM directory (contains date subdirs)
            output_dir: Base output directory for NIfTI files
            subject: Subject identifier
        """
        self.dicom_dir = Path(dicom_dir)
        self.output_dir = Path(output_dir)
        self.subject = subject

        # Find all sequence directories
        self.sequence_dirs = self._find_sequence_directories()

        logger.info(f"Initialized DICOM converter for {subject}")
        logger.info(f"  DICOM directory: {dicom_dir}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Found {len(self.sequence_dirs)} sequences")

    def _find_sequence_directories(self) -> List[Path]:
        """Find all sequence directories containing DICOM files."""
        sequence_dirs = []

        # Look for date directories
        for date_dir in self.dicom_dir.iterdir():
            if not date_dir.is_dir():
                continue

            # Look for sequence directories
            for seq_dir in date_dir.iterdir():
                if not seq_dir.is_dir():
                    continue

                # Check if contains DICOM files
                dcm_files = list(seq_dir.glob('*.dcm')) + list(seq_dir.glob('*.DCM'))
                if dcm_files:
                    sequence_dirs.append(seq_dir)

        return sorted(sequence_dirs)

    def convert_all(self) -> Dict[str, List[Path]]:
        """
        Convert all DICOM sequences to NIfTI.

        Returns:
            Dictionary mapping modality to list of output NIfTI files
        """
        results = {
            'anat': [],
            'dwi': [],
            'func': [],
            'asl': [],
            'fmap': []
        }

        for seq_dir in self.sequence_dirs:
            try:
                modality, nifti_file, json_file = self.convert_sequence(seq_dir)
                if modality and nifti_file:
                    results[modality].append(nifti_file)
                    logger.info(f"  ✓ Converted: {seq_dir.name} → {modality}/{nifti_file.name}")
            except Exception as e:
                logger.error(f"  ✗ Failed to convert {seq_dir.name}: {e}")

        # Log summary
        logger.info("")
        logger.info("Conversion Summary:")
        for modality, files in results.items():
            if files:
                logger.info(f"  {modality}: {len(files)} files")

        return results

    def convert_sequence(self, seq_dir: Path) -> Tuple[Optional[str], Optional[Path], Optional[Path]]:
        """
        Convert a single DICOM sequence to NIfTI.

        Args:
            seq_dir: Path to sequence directory containing DICOM files

        Returns:
            Tuple of (modality, nifti_file, json_file)
        """
        # Read first DICOM to detect modality
        dcm_files = list(seq_dir.glob('*.dcm')) + list(seq_dir.glob('*.DCM'))
        if not dcm_files:
            logger.warning(f"No DICOM files in {seq_dir}")
            return None, None, None

        dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        series_desc = str(dcm.get('SeriesDescription', 'Unknown'))

        # Detect modality
        modality = ModalityDetector.detect_modality(series_desc)
        if not modality:
            logger.warning(f"Could not detect modality for: {series_desc}")
            return None, None, None

        # Create output directory
        # Structure: {output_dir}/{subject}/{modality}/
        output_modality_dir = self.output_dir / self.subject / modality
        output_modality_dir.mkdir(parents=True, exist_ok=True)

        # Run dcm2niix (handles multi-echo automatically)
        nifti_file = self._run_dcm2niix(seq_dir, output_modality_dir)

        if not nifti_file:
            return modality, None, None

        # Check if dcm2niix created a JSON sidecar (it usually does)
        json_file = nifti_file.with_suffix('').with_suffix('.json')

        # If no JSON exists, create one with extracted parameters
        if not json_file.exists():
            json_file = self._create_json_sidecar(dcm, modality, nifti_file)

        return modality, nifti_file, json_file

    def _run_dcm2niix(self, seq_dir: Path, output_dir: Path) -> Optional[Path]:
        """
        Run dcm2niix on DICOM directory.

        Args:
            seq_dir: DICOM sequence directory
            output_dir: Output directory for NIfTI

        Returns:
            Path to converted NIfTI file
        """
        # Create temporary directory for dcm2niix output
        temp_dir = seq_dir / 'temp_nifti'
        temp_dir.mkdir(exist_ok=True)

        try:
            # Run dcm2niix
            cmd = [
                'dcm2niix',
                '-z', 'y',  # Compress
                '-f', '%s_%p_%t_%z',  # Filename format
                '-o', str(temp_dir),
                str(seq_dir)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Find generated NIfTI files (may be multiple for multi-echo)
            nifti_files = sorted(list(temp_dir.glob('*.nii.gz')))
            if not nifti_files:
                logger.warning(f"dcm2niix did not generate NIfTI for {seq_dir}")
                return None

            # Move ALL files to output directory (handles multi-echo)
            output_files = []
            for nifti_file in nifti_files:
                output_file = output_dir / nifti_file.name
                shutil.move(str(nifti_file), str(output_file))
                output_files.append(output_file)

                # Move corresponding JSON sidecar (dcm2niix creates these)
                json_file = temp_dir / nifti_file.name.replace('.nii.gz', '.json')
                if json_file.exists():
                    shutil.move(str(json_file), str(output_dir / json_file.name))

                # Move bval/bvec if present (for DWI)
                for ext in ['bval', 'bvec']:
                    bval_file = temp_dir / f"{nifti_file.stem.replace('.nii', '')}.{ext}"
                    if bval_file.exists():
                        shutil.move(str(bval_file), str(output_dir / bval_file.name))

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            # Log multi-echo detection
            if len(output_files) > 1:
                logger.info(f"  Multi-echo: converted {len(output_files)} echoes")

            # Return first file for backward compatibility
            return output_files[0]

        except subprocess.CalledProcessError as e:
            logger.error(f"dcm2niix failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error running dcm2niix: {e}")
            return None

    def _create_json_sidecar(
        self,
        dcm: pydicom.Dataset,
        modality: str,
        nifti_file: Path
    ) -> Optional[Path]:
        """
        Create JSON sidecar with extracted parameters.

        Args:
            dcm: DICOM dataset
            modality: Detected modality
            nifti_file: Path to NIfTI file

        Returns:
            Path to JSON sidecar file
        """
        # Extract parameters based on modality
        if modality == 'anat':
            params = DICOMParameterExtractor.extract_anatomical_params(dcm)
        elif modality == 'dwi':
            params = DICOMParameterExtractor.extract_dwi_params(dcm)
        elif modality == 'func':
            params = DICOMParameterExtractor.extract_functional_params(dcm)
        elif modality == 'asl':
            params = DICOMParameterExtractor.extract_asl_params(dcm)
        else:
            params = DICOMParameterExtractor.extract_common_params(dcm)

        # Create JSON file
        json_file = nifti_file.with_suffix('').with_suffix('.json')
        with open(json_file, 'w') as f:
            json.dump(params, f, indent=2)

        logger.debug(f"Created JSON sidecar: {json_file}")
        return json_file


def convert_subject_dicoms(
    subject: str,
    dicom_dir: Path,
    output_dir: Path
) -> Dict[str, List[Path]]:
    """
    Convert all DICOM sequences for a subject to NIfTI.

    Args:
        subject: Subject identifier
        dicom_dir: Path to subject's DICOM directory
        output_dir: Base output directory

    Returns:
        Dictionary mapping modality to list of NIfTI files

    Example:
        >>> results = convert_subject_dicoms(
        ...     subject='IRC805-1580101',
        ...     dicom_dir=Path('/mnt/bytopia/IRC805/raw/dicom/IRC805-1580101'),
        ...     output_dir=Path('/mnt/bytopia/IRC805/nifti')
        ... )
        >>> print(results['anat'])
        [PosixPath('/mnt/bytopia/IRC805/nifti/IRC805-1580101/anat/t1w.nii.gz')]
    """
    logger.info("="*70)
    logger.info(f"CONVERTING DICOMS FOR {subject}")
    logger.info("="*70)

    converter = DICOMConverter(dicom_dir, output_dir, subject)
    results = converter.convert_all()

    logger.info("")
    logger.info("="*70)
    logger.info("CONVERSION COMPLETE")
    logger.info("="*70)

    return results


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Convert a subject
    subject = 'IRC805-1580101'
    dicom_dir = Path(f'/mnt/bytopia/IRC805/raw/dicom/{subject}')
    output_dir = Path('/mnt/bytopia/IRC805/nifti')

    results = convert_subject_dicoms(subject, dicom_dir, output_dir)
