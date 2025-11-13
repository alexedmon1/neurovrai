#!/usr/bin/env python3
"""
Utility for extracting ASL-specific acquisition parameters from DICOM files.

This module extracts Philips-specific pCASL parameters from DICOM private tags
for accurate CBF quantification.

Key Parameters:
- Labeling Duration (τ): Time arterial blood is labeled
- Post-Labeling Delay (PLD): Time between labeling and imaging
- Labeling Efficiency (α): Effectiveness of labeling (typically 0.85 for pCASL)
- Background Suppression: Number of pulses applied
- Label-Control Order: Acquisition ordering

References:
- Alsop et al. (2015). Recommended implementation of arterial spin-labeled
  perfusion MRI for clinical applications. MRM 73(1).
- Philips ASL Documentation
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import pydicom
from pydicom.dataset import FileDataset
import json

logger = logging.getLogger(__name__)


def extract_asl_parameters(dicom_file: Path) -> Dict[str, Any]:
    """
    Extract ASL acquisition parameters from DICOM file.

    Parameters
    ----------
    dicom_file : Path
        Path to ASL DICOM file

    Returns
    -------
    dict
        Dictionary containing ASL parameters:
        - labeling_duration: τ (tau) in seconds
        - post_labeling_delay: PLD in seconds
        - labeling_type: 'pcasl' or 'pasl'
        - labeling_efficiency: α (alpha), typically 0.85 for pCASL
        - repetition_time: TR in seconds
        - echo_time: TE in milliseconds
        - flip_angle: degrees
        - matrix_size: acquisition matrix
        - slice_thickness: mm
        - num_volumes: total number of volumes
        - background_suppression: number of BS pulses
        - label_control_order: acquisition order

    Notes
    -----
    Philips stores ASL parameters in private DICOM tags:
    - (2005,10xx): Philips private creator
    - Specific ASL parameters may vary by software version
    """
    logger.info(f"Extracting ASL parameters from: {dicom_file}")

    # Load DICOM file
    dcm = pydicom.dcmread(dicom_file, force=True)

    params = {}

    # Standard DICOM tags
    params['repetition_time'] = float(dcm.RepetitionTime) / 1000.0  # Convert ms to s
    params['echo_time'] = float(dcm.EchoTime)  # Keep in ms
    params['flip_angle'] = float(dcm.FlipAngle)

    # Matrix and slice info
    if hasattr(dcm, 'AcquisitionMatrix'):
        matrix = dcm.AcquisitionMatrix
        params['matrix_size'] = [int(m) for m in matrix if m > 0]

    if hasattr(dcm, 'SliceThickness'):
        params['slice_thickness'] = float(dcm.SliceThickness)

    # Number of volumes (images in acquisition)
    if hasattr(dcm, 'NumberOfTemporalPositions'):
        params['num_volumes'] = int(dcm.NumberOfTemporalPositions)
    elif hasattr(dcm, 'ImagesInAcquisition'):
        params['num_volumes'] = int(dcm.ImagesInAcquisition)

    # ASL-specific parameters
    # Check for standard ASL tags (if present in newer scanners)
    if hasattr(dcm, 'ASLContext'):
        params['asl_context'] = str(dcm.ASLContext)

    # Try to extract from sequence name/protocol
    if hasattr(dcm, 'SeriesDescription'):
        series_desc = str(dcm.SeriesDescription).lower()
        params['series_description'] = series_desc

        # Infer labeling type from series description
        if 'pcasl' in series_desc or 'pseudo' in series_desc:
            params['labeling_type'] = 'pcasl'
        elif 'pasl' in series_desc or 'epistar' in series_desc or 'picore' in series_desc:
            params['labeling_type'] = 'pasl'
        else:
            params['labeling_type'] = 'unknown'

    # Check Philips private tags for ASL parameters
    # Philips uses (2005,xxxx) for private tags
    # Validated tags from IRC805 Philips Ingenia Elition X 3T
    try:
        # Specific ASL parameter tags (validated on Philips Ingenia Elition X)
        if (0x2005, 0x140a) in dcm:  # Labeling duration (τ)
            tau_val = float(dcm[0x2005, 0x140a].value)
            if 1.0 < tau_val < 3.0:  # Sanity check
                params['labeling_duration'] = tau_val
                logger.info(f"Found labeling duration in DICOM: {tau_val:.3f} s")

        if (0x2005, 0x1442) in dcm:  # Post-labeling delay (PLD)
            pld_val = float(dcm[0x2005, 0x1442].value)
            if 1.0 < pld_val < 4.0:  # Sanity check
                params['post_labeling_delay'] = pld_val
                logger.info(f"Found PLD in DICOM: {pld_val:.3f} s")

        if (0x2005, 0x1412) in dcm:  # Background suppression pulses
            bs_pulses = int(dcm[0x2005, 0x1412].value)
            params['background_suppression_pulses'] = bs_pulses
            logger.info(f"Found background suppression pulses: {bs_pulses}")

        if (0x2005, 0x1429) in dcm:  # Label/Control identifier
            lc_id = dcm[0x2005, 0x1429].value
            if isinstance(lc_id, bytes):
                lc_id = lc_id.decode('utf-8').strip()
            if lc_id in ['CONTROL', 'LABEL']:
                params['volume_type'] = lc_id.lower()
                logger.info(f"Volume type: {lc_id}")

        # Additional parameter (purpose unknown, but consistent)
        if (0x2005, 0x100a) in dcm:
            param_100a = float(dcm[0x2005, 0x100a].value)
            params['philips_100a'] = param_100a
            logger.debug(f"Philips tag (2005,100a): {param_100a:.3f}")

    except Exception as e:
        logger.warning(f"Error reading Philips private tags: {e}")

    # Check for MR Echo Pulse Sequence (0018,9005) - ASL labeling info
    if (0x0018, 0x9005) in dcm:
        params['pulse_sequence_name'] = str(dcm[0x0018, 0x9005].value)

    # Set defaults based on typical Philips pCASL protocols if not found
    if 'labeling_duration' not in params:
        logger.warning("Labeling duration not found in DICOM. Using default: 1.8s")
        params['labeling_duration'] = 1.8  # Typical for pCASL

    if 'post_labeling_delay' not in params:
        logger.warning("Post-labeling delay not found in DICOM. Using default: 1.8s")
        params['post_labeling_delay'] = 1.8  # Typical for pCASL

    if 'labeling_efficiency' not in params:
        # Default labeling efficiency for pCASL
        if params.get('labeling_type') == 'pcasl':
            params['labeling_efficiency'] = 0.85
        elif params.get('labeling_type') == 'pasl':
            params['labeling_efficiency'] = 0.98
        else:
            params['labeling_efficiency'] = 0.85  # Default to pCASL

    # T1 of blood at 3T (standard value)
    params['t1_blood'] = 1.65  # seconds

    # Blood-brain partition coefficient (standard value)
    params['blood_brain_partition'] = 0.9  # ml/g

    # Label-control order (try to infer from protocol)
    if 'label_control_order' not in params:
        # Default to control-first for Philips pCASL
        params['label_control_order'] = 'control_first'

    logger.info("")
    logger.info("Extracted ASL Parameters:")
    logger.info("=" * 70)
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70)
    logger.info("")

    return params


def extract_from_json_sidecar(json_file: Path) -> Optional[Dict[str, Any]]:
    """
    Extract ASL parameters from BIDS JSON sidecar file.

    BIDS format stores acquisition parameters in JSON sidecars alongside NIfTI files.

    Parameters
    ----------
    json_file : Path
        Path to JSON sidecar file

    Returns
    -------
    dict or None
        ASL parameters if found
    """
    if not json_file.exists():
        logger.warning(f"JSON sidecar not found: {json_file}")
        return None

    logger.info(f"Reading ASL parameters from JSON sidecar: {json_file}")

    with open(json_file, 'r') as f:
        data = json.load(f)

    params = {}

    # Standard BIDS ASL fields
    asl_fields = {
        'RepetitionTime': 'repetition_time',
        'EchoTime': 'echo_time',
        'FlipAngle': 'flip_angle',
        'LabelingDuration': 'labeling_duration',
        'PostLabelDelay': 'post_labeling_delay',
        'ArterialSpinLabelingType': 'labeling_type',
        'LabelingEfficiency': 'labeling_efficiency',
        'BackgroundSuppression': 'background_suppression',
        'M0Type': 'm0_type',
        'TotalAcquiredPairs': 'num_pairs'
    }

    for bids_key, param_key in asl_fields.items():
        if bids_key in data:
            params[param_key] = data[bids_key]

    logger.info("")
    logger.info("Extracted ASL Parameters from JSON:")
    logger.info("=" * 70)
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70)
    logger.info("")

    return params if params else None


def save_parameters_to_config(params: Dict[str, Any], output_file: Path):
    """
    Save extracted ASL parameters to YAML config file.

    Parameters
    ----------
    params : dict
        ASL parameters dictionary
    output_file : Path
        Output YAML config file path
    """
    import yaml

    config = {
        'asl': {
            'labeling_type': params.get('labeling_type', 'pcasl'),
            'labeling_duration': params.get('labeling_duration', 1.8),
            'post_labeling_delay': params.get('post_labeling_delay', 1.8),
            'labeling_efficiency': params.get('labeling_efficiency', 0.85),
            't1_blood': params.get('t1_blood', 1.65),
            'blood_brain_partition': params.get('blood_brain_partition', 0.9),
            'label_control_order': params.get('label_control_order', 'control_first')
        }
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved ASL parameters to: {output_file}")


if __name__ == '__main__':
    """Example usage"""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python dicom_asl_params.py <dicom_file_or_json>")
        print("Example: python dicom_asl_params.py /path/to/asl_001.dcm")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if input_file.suffix == '.json':
        params = extract_from_json_sidecar(input_file)
    else:
        params = extract_asl_parameters(input_file)

    if params:
        # Save to config file
        output_config = Path('asl_params.yaml')
        save_parameters_to_config(params, output_config)
        print(f"\nASL parameters saved to: {output_config}")
