#!/usr/bin/env python3
"""
Configuration auto-generation from DICOM headers.

Scans DICOM directories to automatically detect sequences and generate
study-specific configuration files.
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pydicom
import yaml
from collections import defaultdict


class DICOMScanner:
    """Scans DICOM directories to detect sequences and extract parameters."""

    def __init__(self, dicom_dir: Path):
        """
        Initialize DICOM scanner.

        Parameters
        ----------
        dicom_dir : Path
            Root directory containing DICOM files
        """
        self.dicom_dir = Path(dicom_dir)
        self.sequences = {}
        self.scanner_info = {}

    def scan_directory(self) -> Dict[str, List[str]]:
        """
        Scan DICOM directory and detect all unique sequences.

        Returns
        -------
        dict
            Dictionary mapping modality -> list of sequence names
        """
        print(f"Scanning DICOM directory: {self.dicom_dir}")

        # Find all DICOM files
        dicom_files = list(self.dicom_dir.rglob("*.dcm"))
        if not dicom_files:
            # Try without extension
            dicom_files = [f for f in self.dicom_dir.rglob("*") if f.is_file()]

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {self.dicom_dir}")

        print(f"Found {len(dicom_files)} DICOM files")

        # Collect unique sequences
        sequences_found = set()
        scanner_params = {}

        # Sample files to extract info (don't read all - can be slow)
        sample_size = min(100, len(dicom_files))
        for dcm_file in dicom_files[:sample_size]:
            try:
                dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)

                # Get sequence description
                if hasattr(dcm, 'SeriesDescription'):
                    seq_desc = dcm.SeriesDescription
                    sequences_found.add(seq_desc)

                    # Extract scanner params from first file only
                    if not scanner_params:
                        scanner_params = self._extract_scanner_params(dcm)

            except Exception as e:
                # Skip files that can't be read
                continue

        self.scanner_info = scanner_params
        print(f"Detected {len(sequences_found)} unique sequences")

        # Classify sequences by modality
        self.sequences = self._classify_sequences(list(sequences_found))

        return self.sequences

    def _extract_scanner_params(self, dcm: pydicom.Dataset) -> Dict[str, any]:
        """
        Extract scanner parameters from DICOM header.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset

        Returns
        -------
        dict
            Scanner parameters
        """
        params = {}

        # Scanner information
        if hasattr(dcm, 'Manufacturer'):
            params['manufacturer'] = dcm.Manufacturer
        if hasattr(dcm, 'ManufacturerModelName'):
            params['model'] = dcm.ManufacturerModelName
        if hasattr(dcm, 'MagneticFieldStrength'):
            params['field_strength'] = f"{dcm.MagneticFieldStrength}T"

        # Sequence parameters (if available)
        if hasattr(dcm, 'RepetitionTime'):
            params['tr'] = dcm.RepetitionTime
        if hasattr(dcm, 'EchoTime'):
            params['te'] = dcm.EchoTime
        if hasattr(dcm, 'FlipAngle'):
            params['flip_angle'] = dcm.FlipAngle

        # Image dimensions
        if hasattr(dcm, 'PixelSpacing'):
            params['pixel_spacing'] = list(dcm.PixelSpacing)
        if hasattr(dcm, 'SliceThickness'):
            params['slice_thickness'] = dcm.SliceThickness

        return params

    def _classify_sequences(self, sequences: List[str]) -> Dict[str, List[str]]:
        """
        Classify sequences by modality using pattern matching.

        Parameters
        ----------
        sequences : list
            List of sequence descriptions

        Returns
        -------
        dict
            Dictionary mapping modality -> list of sequences
        """
        # Modality patterns (from default.yaml)
        patterns = {
            't1w': ['MPRAGE', 'T1', '3D_T1', 'T1_TFE', 'T1W'],
            't2w': ['T2', 'SPACE', 'T2W', 'T2_TSE'],
            'dwi': ['DTI', 'DWI', 'ep2d_diff', 'dWIP'],
            'rest': ['REST', 'BOLD', 'fMRI', 'rs_fMRI'],
            'asl': ['ASL', 'pCASL', 'PCASL'],
            'fmap': ['field', 'map', 'gre_field', 'SE_EPI', 'FIELDMAP']
        }

        classified = defaultdict(list)
        unclassified = []

        for seq in sequences:
            seq_upper = seq.upper()
            matched = False

            for modality, keywords in patterns.items():
                for keyword in keywords:
                    if keyword.upper() in seq_upper:
                        classified[modality].append(seq)
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                unclassified.append(seq)

        # Report findings
        print("\nSequence classification:")
        for modality, seqs in sorted(classified.items()):
            print(f"  {modality}: {len(seqs)} sequences")
            for seq in seqs:
                print(f"    - {seq}")

        if unclassified:
            print(f"\n⚠ Unclassified sequences ({len(unclassified)}):")
            for seq in unclassified:
                print(f"    - {seq}")
            print("  These will need manual review and classification.")

        return dict(classified)

    def get_scanner_info(self) -> Dict[str, any]:
        """
        Get scanner information.

        Returns
        -------
        dict
            Scanner parameters
        """
        return self.scanner_info


def generate_study_config(
    study_name: str,
    study_code: str,
    base_dir: Path,
    sequences: Dict[str, List[str]],
    scanner_info: Optional[Dict[str, any]] = None,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate study-specific configuration YAML.

    Parameters
    ----------
    study_name : str
        Full study name
    study_code : str
        Short study code
    base_dir : Path
        Base directory for study data
    sequences : dict
        Dictionary mapping modality -> list of sequences
    scanner_info : dict, optional
        Scanner parameters to include as comments
    output_path : Path, optional
        Where to save YAML file (if None, returns string)

    Returns
    -------
    str
        Generated YAML configuration
    """
    config = {
        'study': {
            'name': study_name,
            'code': study_code,
            'base_dir': str(base_dir)
        },
        'paths': {
            'sourcedata': '${study.base_dir}/sourcedata',
            'rawdata': '${study.base_dir}/rawdata',
            'derivatives': '${study.base_dir}/derivatives',
            'transforms': '${study.base_dir}/transforms',
            'logs': '${study.base_dir}/logs'
        },
        'sequence_mappings': sequences
    }

    # Build YAML string with comments
    yaml_lines = []

    # Header
    yaml_lines.append(f"# {study_code}.yaml")
    yaml_lines.append(f"# Auto-generated configuration for {study_name}")
    yaml_lines.append("#")
    yaml_lines.append("# Generated by: mri-preprocess config init")
    yaml_lines.append("#")
    yaml_lines.append("# IMPORTANT: Review and edit this file before using!")
    yaml_lines.append("# - Verify sequence classifications are correct")
    yaml_lines.append("# - Add any missing sequences")
    yaml_lines.append("# - Adjust preprocessing parameters as needed")
    yaml_lines.append("#")

    # Scanner info as comments
    if scanner_info:
        yaml_lines.append("# Scanner Information:")
        for key, value in scanner_info.items():
            yaml_lines.append(f"#   {key}: {value}")
        yaml_lines.append("#")

    yaml_lines.append("")

    # Dump main config
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    yaml_lines.append(yaml_str)

    # Add helpful comments at end
    yaml_lines.append("\n# Additional configuration:")
    yaml_lines.append("# Uncomment and customize as needed\n")

    # Example diffusion params
    if 'dwi' in sequences:
        yaml_lines.append("# Diffusion preprocessing:")
        yaml_lines.append("# diffusion:")
        yaml_lines.append("#   eddy:")
        yaml_lines.append("#     acqp_file: ${study.base_dir}/configs/acqp.txt")
        yaml_lines.append("#     index_file: ${study.base_dir}/configs/index.txt")
        yaml_lines.append("#   bedpostx:")
        yaml_lines.append("#     run: true  # Enable for tractography")
        yaml_lines.append("")

    # Example execution params
    yaml_lines.append("# Execution settings:")
    yaml_lines.append("# execution:")
    yaml_lines.append("#   n_procs: 8  # Adjust based on your system")
    yaml_lines.append("")

    # Anonymization
    yaml_lines.append("# Anonymization (for shared data):")
    yaml_lines.append("# anonymization:")
    yaml_lines.append("#   enabled: true")
    yaml_lines.append("#   strip_headers: true")
    yaml_lines.append("#   dcm2niix_flags: \"-ba y\"")

    final_yaml = '\n'.join(yaml_lines)

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(final_yaml)
        print(f"\n✓ Configuration saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Review and edit: {output_path}")
        print(f"2. Verify sequence classifications")
        print(f"3. Add study-specific parameters")
        print(f"4. Validate: mri-preprocess config validate --config {output_path}")

    return final_yaml


def auto_generate_config(
    dicom_dir: Path,
    study_name: str,
    study_code: str,
    base_dir: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> str:
    """
    Automatically generate configuration from DICOM directory.

    This is the main entry point for auto-generation.

    Parameters
    ----------
    dicom_dir : Path
        Directory containing DICOM files
    study_name : str
        Full study name
    study_code : str
        Short study code
    base_dir : Path, optional
        Base directory for study (defaults to parent of dicom_dir)
    output_path : Path, optional
        Where to save YAML (if None, returns string only)

    Returns
    -------
    str
        Generated YAML configuration
    """
    if base_dir is None:
        base_dir = Path(dicom_dir).parent.parent

    # Scan DICOM directory
    scanner = DICOMScanner(dicom_dir)
    sequences = scanner.scan_directory()
    scanner_info = scanner.get_scanner_info()

    # Generate config
    config_yaml = generate_study_config(
        study_name=study_name,
        study_code=study_code,
        base_dir=base_dir,
        sequences=sequences,
        scanner_info=scanner_info,
        output_path=output_path
    )

    return config_yaml
