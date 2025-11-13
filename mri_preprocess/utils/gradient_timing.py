"""
Gradient timing extraction and estimation for diffusion MRI.

This module provides functions to extract gradient timing parameters (δ, Δ)
from DICOM headers or estimate them from sequence parameters for AMICO STEJSKALTANNER scheme.
"""

import pydicom
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def extract_gradient_timing_from_dicom(
    dicom_file: Path
) -> Dict[str, Optional[float]]:
    """
    Extract gradient timing parameters from a DICOM file.

    Attempts to extract:
    - TE (Echo Time) in seconds
    - δ (small delta, gradient pulse duration) in seconds
    - Δ (big delta, gradient pulse separation) in seconds

    Args:
        dicom_file: Path to DICOM file

    Returns:
        Dictionary with 'TE', 'delta', 'Delta' keys. Values are None if not found.
    """
    timing = {'TE': None, 'delta': None, 'Delta': None}

    try:
        ds = pydicom.dcmread(dicom_file)

        # Extract TE (standard DICOM tag)
        if 'EchoTime' in ds:
            timing['TE'] = float(ds.EchoTime) / 1000.0  # Convert ms to seconds

        # Try to find δ and Δ in Philips private tags
        # Note: Exact tags may vary by scanner model and software version

        # Common Philips private tag locations
        philips_delta_tags = [
            (0x0019, 0x100c),  # Possible small delta
            (0x0019, 0x100e),  # Possible gradient duration
        ]

        philips_Delta_tags = [
            (0x0019, 0x100d),  # Possible big delta
        ]

        for tag in philips_delta_tags:
            if tag in ds:
                try:
                    val = float(ds[tag].value)
                    # Check if value is reasonable (typically 5-30 ms)
                    if 0.005 < val < 0.1:  # 5-100 ms
                        timing['delta'] = val
                        logger.info(f"Found δ (small delta) = {val*1000:.1f} ms in tag {tag}")
                        break
                except (ValueError, TypeError):
                    pass

        for tag in philips_Delta_tags:
            if tag in ds:
                try:
                    val = float(ds[tag].value)
                    # Check if value is reasonable (typically 20-60 ms)
                    if 0.01 < val < 0.15:  # 10-150 ms
                        timing['Delta'] = val
                        logger.info(f"Found Δ (big delta) = {val*1000:.1f} ms in tag {tag}")
                        break
                except (ValueError, TypeError):
                    pass

    except Exception as e:
        logger.warning(f"Error reading DICOM: {e}")

    return timing


def extract_gradient_timing_from_bids_json(
    json_file: Path
) -> Dict[str, Optional[float]]:
    """
    Extract gradient timing parameters from BIDS JSON sidecar.

    Args:
        json_file: Path to BIDS JSON file

    Returns:
        Dictionary with 'TE', 'delta', 'Delta' keys. Values are None if not found.
    """
    timing = {'TE': None, 'delta': None, 'Delta': None}

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract TE
        if 'EchoTime' in data:
            timing['TE'] = float(data['EchoTime'])

        # Check for gradient timing fields (not standard BIDS, but some converters add them)
        if 'DiffusionGradientDuration' in data:
            timing['delta'] = float(data['DiffusionGradientDuration'])

        if 'DiffusionGradientSeparation' in data:
            timing['Delta'] = float(data['DiffusionGradientSeparation'])

        # Alternative field names
        if timing['delta'] is None and 'SmallDelta' in data:
            timing['delta'] = float(data['SmallDelta'])

        if timing['Delta'] is None and 'BigDelta' in data:
            timing['Delta'] = float(data['BigDelta'])

    except Exception as e:
        logger.warning(f"Error reading BIDS JSON: {e}")

    return timing


def estimate_gradient_timing_philips(
    TE: float,
    manufacturer_model: str = "Philips",
    sequence_type: str = "spin_echo"
) -> Tuple[float, float]:
    """
    Estimate gradient timing parameters for Philips scanners.

    Uses empirical relationships and typical values for clinical Philips scanners.
    These are ESTIMATES and may not be exact for your specific protocol.

    For Philips DWI spin-echo EPI:
    - δ (small delta): Typically 10-20 ms (gradient pulse duration)
    - Δ (big delta): Typically TE/2 to 2*TE/3

    Args:
        TE: Echo time in seconds
        manufacturer_model: Scanner manufacturer and model
        sequence_type: Sequence type ("spin_echo" or "gradient_echo")

    Returns:
        Tuple of (delta, Delta) in seconds
    """
    if "Philips" in manufacturer_model or "Ingenia" in manufacturer_model:
        if sequence_type == "spin_echo":
            # Philips spin-echo EPI typical values
            # δ: Gradient pulse duration, typically 15-25 ms for clinical scanners
            delta = 0.020  # 20 ms (typical for Philips clinical systems)

            # Δ: Gradient separation
            # For spin-echo: Δ ≈ TE/2 - δ/2 to TE - δ
            # Conservative estimate: Δ ≈ TE/2
            Delta = TE / 2.0

            logger.info("Using Philips spin-echo EPI estimates:")
            logger.info(f"  δ (small delta) = {delta*1000:.1f} ms (typical clinical value)")
            logger.info(f"  Δ (big delta) = {Delta*1000:.1f} ms (≈ TE/2)")
            logger.info("  WARNING: These are ESTIMATES. Verify with scanner protocol if possible.")

            return delta, Delta

    # Generic fallback estimates
    logger.warning("Using generic gradient timing estimates")
    delta = 0.025  # 25 ms
    Delta = TE / 2.0
    logger.info(f"  δ = {delta*1000:.1f} ms, Δ = {Delta*1000:.1f} ms")

    return delta, Delta


def get_gradient_timing(
    bids_json: Optional[Path] = None,
    dicom_file: Optional[Path] = None,
    TE: Optional[float] = None,
    manufacturer_model: Optional[str] = None,
    allow_estimation: bool = True
) -> Tuple[float, float, float]:
    """
    Get gradient timing parameters (TE, δ, Δ) from available sources.

    Attempts extraction in order:
    1. BIDS JSON sidecar
    2. DICOM header
    3. Estimation (if allow_estimation=True)

    Args:
        bids_json: Path to BIDS JSON file
        dicom_file: Path to DICOM file
        TE: Echo time in seconds (if known)
        manufacturer_model: Scanner manufacturer/model (for estimation)
        allow_estimation: Whether to estimate missing values

    Returns:
        Tuple of (TE, delta, Delta) in seconds

    Raises:
        ValueError: If required parameters cannot be determined
    """
    timing = {'TE': TE, 'delta': None, 'Delta': None}

    # Try BIDS JSON first
    if bids_json and bids_json.exists():
        logger.info(f"Checking BIDS JSON: {bids_json}")
        json_timing = extract_gradient_timing_from_bids_json(bids_json)

        if timing['TE'] is None:
            timing['TE'] = json_timing['TE']
        if timing['delta'] is None:
            timing['delta'] = json_timing['delta']
        if timing['Delta'] is None:
            timing['Delta'] = json_timing['Delta']

    # Try DICOM if values still missing
    if dicom_file and dicom_file.exists():
        if timing['TE'] is None or timing['delta'] is None or timing['Delta'] is None:
            logger.info(f"Checking DICOM: {dicom_file}")
            dicom_timing = extract_gradient_timing_from_dicom(dicom_file)

            if timing['TE'] is None:
                timing['TE'] = dicom_timing['TE']
            if timing['delta'] is None:
                timing['delta'] = dicom_timing['delta']
            if timing['Delta'] is None:
                timing['Delta'] = dicom_timing['Delta']

    # Estimate missing values if allowed
    if allow_estimation and (timing['delta'] is None or timing['Delta'] is None):
        if timing['TE'] is not None:
            logger.warning("Gradient timing parameters not found in DICOM/JSON")
            logger.warning("Using estimated values based on TE and scanner type")

            delta_est, Delta_est = estimate_gradient_timing_philips(
                timing['TE'],
                manufacturer_model or "Philips",
                "spin_echo"
            )

            if timing['delta'] is None:
                timing['delta'] = delta_est
            if timing['Delta'] is None:
                timing['Delta'] = Delta_est

    # Validate we have all required values
    if timing['TE'] is None:
        raise ValueError("Echo Time (TE) could not be determined")
    if timing['delta'] is None:
        raise ValueError("Gradient pulse duration (δ) could not be determined")
    if timing['Delta'] is None:
        raise ValueError("Gradient pulse separation (Δ) could not be determined")

    logger.info("Gradient timing parameters:")
    logger.info(f"  TE = {timing['TE']*1000:.1f} ms")
    logger.info(f"  δ (small delta) = {timing['delta']*1000:.1f} ms")
    logger.info(f"  Δ (big delta) = {timing['Delta']*1000:.1f} ms")

    return timing['TE'], timing['delta'], timing['Delta']


def create_amico_scheme_with_timing(
    bval_file: Path,
    bvec_file: Path,
    output_scheme: Path,
    TE: float,
    delta: float,
    Delta: float
) -> Path:
    """
    Create AMICO STEJSKALTANNER scheme file with gradient timing.

    Format (7 columns):
    bx by bz G δ Δ TE

    Where:
    - bx, by, bz: Gradient directions (unit vectors for b>0, zeros for b=0)
    - G: Gradient strength [T/m] (computed from b-value)
    - δ: Gradient pulse duration [seconds]
    - Δ: Gradient pulse separation [seconds]
    - TE: Echo time [seconds]

    Args:
        bval_file: Path to FSL bval file
        bvec_file: Path to FSL bvec file
        output_scheme: Path for output scheme file
        TE: Echo time in seconds
        delta: Gradient pulse duration in seconds
        Delta: Gradient pulse separation in seconds

    Returns:
        Path to created scheme file
    """
    import numpy as np

    # Load bvals and bvecs
    bvals = np.loadtxt(bval_file)
    bvecs = np.loadtxt(bvec_file)

    if bvecs.shape[0] == 3:
        bvecs = bvecs.T  # Transpose to Nx3

    # Gyromagnetic ratio for protons (rad/s/T)
    GAMMA = 267.513e6  # 2π × 42.576 MHz/T

    # Compute gradient strength from b-value
    # b = (γ × G × δ)² × (Δ - δ/3)
    # G = sqrt(b / ((γ × δ)² × (Δ - δ/3)))

    scheme_lines = [f"VERSION: STEJSKALTANNER"]

    for i, (bval, bvec) in enumerate(zip(bvals, bvecs)):
        bval_si = bval * 1e6  # Convert from s/mm² to s/m²

        if bval < 10:  # b=0 volume
            G = 0.0
            bvec = [0.0, 0.0, 0.0]
        else:
            # Calculate gradient strength
            G = np.sqrt(bval_si / ((GAMMA * delta)**2 * (Delta - delta/3)))

        # Format: bx by bz G δ Δ TE
        line = f"{bvec[0]:.6f}\t{bvec[1]:.6f}\t{bvec[2]:.6f}\t{G:.6f}\t{delta:.6f}\t{Delta:.6f}\t{TE:.6f}"
        scheme_lines.append(line)

    # Write scheme file
    with open(output_scheme, 'w') as f:
        f.write('\n'.join(scheme_lines))

    logger.info(f"Created STEJSKALTANNER scheme: {output_scheme}")
    logger.info(f"  Volumes: {len(bvals)}")
    logger.info(f"  TE = {TE*1000:.1f} ms")
    logger.info(f"  δ = {delta*1000:.1f} ms")
    logger.info(f"  Δ = {Delta*1000:.1f} ms")

    return output_scheme
