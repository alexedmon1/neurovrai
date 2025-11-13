"""
ASL CBF Quantification Utilities

Provides functions for cerebral blood flow (CBF) quantification from ASL data
using the standard single-compartment kinetic model (Alsop et al., MRM 2015).
"""

import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def separate_label_control(
    asl_4d: np.ndarray,
    order: str = 'control_first'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate label and control volumes from 4D ASL time series.

    Parameters
    ----------
    asl_4d : np.ndarray
        4D ASL time series (x, y, z, time)
    order : str
        Label-control order:
        - 'control_first': C-L-C-L-C-L... (default)
        - 'label_first': L-C-L-C-L-C...

    Returns
    -------
    tuple
        (control_volumes, label_volumes) as 4D arrays

    Examples
    --------
    >>> asl_data = nib.load('asl.nii.gz').get_fdata()
    >>> control, label = separate_label_control(asl_data, order='control_first')
    >>> print(f"Control: {control.shape}, Label: {label.shape}")
    """
    n_volumes = asl_4d.shape[3]

    if n_volumes % 2 != 0:
        logger.warning(f"Odd number of volumes ({n_volumes}). Last volume will be discarded.")
        asl_4d = asl_4d[..., :-1]
        n_volumes -= 1

    if order == 'control_first':
        # Even indices (0, 2, 4, ...) are control
        control_volumes = asl_4d[..., 0::2]
        # Odd indices (1, 3, 5, ...) are label
        label_volumes = asl_4d[..., 1::2]
    elif order == 'label_first':
        # Even indices are label
        label_volumes = asl_4d[..., 0::2]
        # Odd indices are control
        control_volumes = asl_4d[..., 1::2]
    else:
        raise ValueError(f"Invalid order: {order}. Must be 'control_first' or 'label_first'")

    n_pairs = control_volumes.shape[3]
    logger.info(f"Separated {n_volumes} volumes into {n_pairs} label-control pairs")

    return control_volumes, label_volumes


def compute_perfusion_weighted_signal(
    control_volumes: np.ndarray,
    label_volumes: np.ndarray,
    method: str = 'simple'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute perfusion-weighted signal (ΔM) from label-control subtraction.

    Parameters
    ----------
    control_volumes : np.ndarray
        4D array of control volumes
    label_volumes : np.ndarray
        4D array of label volumes
    method : str
        Subtraction method:
        - 'simple': Control - Label (standard)
        - 'surround': (Control_before + Control_after)/2 - Label

    Returns
    -------
    tuple
        (delta_m_4d, delta_m_mean) - 4D perfusion time series and mean

    Notes
    -----
    The perfusion-weighted signal ΔM is proportional to CBF.
    Positive ΔM indicates blood flow (control has more signal than label).
    """
    if method == 'simple':
        # Standard pairwise subtraction: C1-L1, C2-L2, C3-L3, ...
        delta_m_4d = control_volumes - label_volumes
    elif method == 'surround':
        # Surround subtraction: (C0+C2)/2 - L1, (C2+C4)/2 - L3, ...
        # More robust to motion but loses first and last pairs
        logger.warning("Surround subtraction not yet implemented, using simple subtraction")
        delta_m_4d = control_volumes - label_volumes
    else:
        raise ValueError(f"Invalid method: {method}")

    # Compute mean across all pairs
    delta_m_mean = np.mean(delta_m_4d, axis=3)

    logger.info(f"Computed perfusion-weighted signal (ΔM)")
    logger.info(f"  Shape: {delta_m_4d.shape}")
    logger.info(f"  Mean ΔM range: [{np.min(delta_m_mean):.2f}, {np.max(delta_m_mean):.2f}]")

    return delta_m_4d, delta_m_mean


def quantify_cbf(
    delta_m: np.ndarray,
    m0: np.ndarray,
    labeling_duration: float = 1.8,
    post_labeling_delay: float = 1.8,
    labeling_efficiency: float = 0.85,
    t1_blood: float = 1.65,
    blood_brain_partition: float = 0.9,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Quantify cerebral blood flow (CBF) from perfusion-weighted signal.

    Uses the standard single-compartment kinetic model from the ASL
    white paper (Alsop et al., MRM 2015).

    Parameters
    ----------
    delta_m : np.ndarray
        Perfusion-weighted signal (Control - Label) in image units
    m0 : np.ndarray
        Equilibrium magnetization (from control image or proton density)
    labeling_duration : float
        Arterial transit time (τ) in seconds (default: 1.8s for pCASL)
    post_labeling_delay : float
        Post-labeling delay (PLD or TI) in seconds (default: 1.8s)
    labeling_efficiency : float
        Labeling efficiency (α) for pCASL (default: 0.85)
    t1_blood : float
        T1 relaxation time of blood at 3T in seconds (default: 1.65s)
    blood_brain_partition : float
        Blood-brain partition coefficient (λ) in ml/g (default: 0.9)
    mask : np.ndarray, optional
        Brain mask to restrict CBF calculation

    Returns
    -------
    np.ndarray
        CBF map in ml/100g/min

    Notes
    -----
    The standard ASL equation:

    CBF = (λ · ΔM · e^(PLD/T1_blood)) /
          (2 · α · T1_blood · M0 · (1 - e^(-τ/T1_blood)))

    Typical CBF values:
    - Gray matter: 40-60 ml/100g/min
    - White matter: 20-30 ml/100g/min

    References
    ----------
    Alsop et al. (2015). Recommended implementation of arterial spin-labeled
    perfusion MRI for clinical applications. Magnetic Resonance in Medicine, 73(1).
    """
    logger.info("Quantifying CBF using standard kinetic model")
    logger.info(f"  Labeling duration (τ): {labeling_duration} s")
    logger.info(f"  Post-labeling delay (PLD): {post_labeling_delay} s")
    logger.info(f"  Labeling efficiency (α): {labeling_efficiency}")
    logger.info(f"  T1 blood: {t1_blood} s")
    logger.info(f"  Partition coefficient (λ): {blood_brain_partition} ml/g")

    # Avoid division by zero
    m0_safe = np.where(m0 > 0, m0, np.nan)

    # Compute CBF in ml/g/s
    numerator = blood_brain_partition * delta_m * np.exp(post_labeling_delay / t1_blood)
    denominator = (2 * labeling_efficiency * t1_blood * m0_safe *
                   (1 - np.exp(-labeling_duration / t1_blood)))

    cbf_ml_g_s = numerator / denominator

    # Convert to ml/100g/min (standard units)
    cbf = cbf_ml_g_s * 100 * 60

    # Apply mask if provided
    if mask is not None:
        cbf = np.where(mask > 0, cbf, 0)

    # Replace NaN/Inf with 0
    cbf = np.nan_to_num(cbf, nan=0.0, posinf=0.0, neginf=0.0)

    # Report statistics
    if mask is not None:
        masked_cbf = cbf[mask > 0]
        logger.info(f"  CBF statistics (within mask):")
        logger.info(f"    Mean: {np.mean(masked_cbf):.2f} ml/100g/min")
        logger.info(f"    Median: {np.median(masked_cbf):.2f} ml/100g/min")
        logger.info(f"    Std: {np.std(masked_cbf):.2f} ml/100g/min")
        logger.info(f"    Range: [{np.min(masked_cbf):.2f}, {np.max(masked_cbf):.2f}]")
    else:
        logger.info(f"  CBF range: [{np.min(cbf):.2f}, {np.max(cbf):.2f}] ml/100g/min")

    return cbf


def compute_tissue_specific_cbf(
    cbf: np.ndarray,
    gm_mask: np.ndarray,
    wm_mask: np.ndarray,
    csf_mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute tissue-specific CBF statistics.

    Parameters
    ----------
    cbf : np.ndarray
        CBF map in ml/100g/min
    gm_mask : np.ndarray
        Gray matter mask
    wm_mask : np.ndarray
        White matter mask
    csf_mask : np.ndarray
        CSF mask

    Returns
    -------
    dict
        Dictionary with tissue-specific CBF statistics:
        - 'gm': {'mean', 'median', 'std', 'n_voxels'}
        - 'wm': {'mean', 'median', 'std', 'n_voxels'}
        - 'csf': {'mean', 'median', 'std', 'n_voxels'}

    Notes
    -----
    Expected CBF values:
    - Gray matter: 40-60 ml/100g/min
    - White matter: 20-30 ml/100g/min
    - CSF: ~0 ml/100g/min (no perfusion)
    """
    results = {}

    for tissue, mask in [('gm', gm_mask), ('wm', wm_mask), ('csf', csf_mask)]:
        tissue_cbf = cbf[mask > 0]

        if len(tissue_cbf) > 0:
            results[tissue] = {
                'mean': float(np.mean(tissue_cbf)),
                'median': float(np.median(tissue_cbf)),
                'std': float(np.std(tissue_cbf)),
                'min': float(np.min(tissue_cbf)),
                'max': float(np.max(tissue_cbf)),
                'n_voxels': int(np.sum(mask > 0))
            }

            logger.info(f"{tissue.upper()} CBF: {results[tissue]['mean']:.2f} ± "
                       f"{results[tissue]['std']:.2f} ml/100g/min "
                       f"(median: {results[tissue]['median']:.2f}, n={results[tissue]['n_voxels']})")
        else:
            results[tissue] = None
            logger.warning(f"No voxels found in {tissue.upper()} mask")

    return results


def save_asl_outputs(
    output_dir: Path,
    subject: str,
    control_mean: Optional[np.ndarray] = None,
    label_mean: Optional[np.ndarray] = None,
    delta_m_mean: Optional[np.ndarray] = None,
    cbf: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    header: Optional[nib.Nifti1Header] = None
) -> Dict[str, Path]:
    """
    Save ASL processing outputs to NIfTI files.

    Parameters
    ----------
    output_dir : Path
        Output directory
    subject : str
        Subject identifier
    control_mean : np.ndarray, optional
        Mean control image
    label_mean : np.ndarray, optional
        Mean label image
    delta_m_mean : np.ndarray, optional
        Mean perfusion-weighted image
    cbf : np.ndarray, optional
        CBF map
    affine : np.ndarray, optional
        Affine matrix for NIfTI header
    header : nib.Nifti1Header, optional
        NIfTI header template

    Returns
    -------
    dict
        Dictionary mapping output names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save each output if provided
    if control_mean is not None:
        control_file = output_dir / f'{subject}_control_mean.nii.gz'
        nib.save(nib.Nifti1Image(control_mean, affine, header), control_file)
        saved_files['control_mean'] = control_file
        logger.info(f"Saved mean control image: {control_file}")

    if label_mean is not None:
        label_file = output_dir / f'{subject}_label_mean.nii.gz'
        nib.save(nib.Nifti1Image(label_mean, affine, header), label_file)
        saved_files['label_mean'] = label_file
        logger.info(f"Saved mean label image: {label_file}")

    if delta_m_mean is not None:
        delta_m_file = output_dir / f'{subject}_perfusion_mean.nii.gz'
        nib.save(nib.Nifti1Image(delta_m_mean, affine, header), delta_m_file)
        saved_files['perfusion_mean'] = delta_m_file
        logger.info(f"Saved mean perfusion image: {delta_m_file}")

    if cbf is not None:
        cbf_file = output_dir / f'{subject}_cbf.nii.gz'
        nib.save(nib.Nifti1Image(cbf, affine, header), cbf_file)
        saved_files['cbf'] = cbf_file
        logger.info(f"Saved CBF map: {cbf_file}")

    return saved_files
