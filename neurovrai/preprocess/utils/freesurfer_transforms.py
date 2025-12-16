#!/usr/bin/env python3
"""
FreeSurfer Transform Pipeline

Provides transform chain from FreeSurfer conformed space to DWI native space.

Transform Chain:
    FreeSurfer (conformed 256³, 1mm) → T1w (preprocessing) → DWI (native)

This module enables using FreeSurfer parcellations (aparc+aseg) for DWI
tractography by properly transforming ROIs to DWI space.

Dependencies:
    - FSL (FLIRT, convert_xfm, applywarp)
    - FreeSurfer (mri_convert for .mgz → .nii.gz conversion)
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


def convert_mgz_to_nifti(
    mgz_file: Path,
    output_file: Path
) -> Path:
    """
    Convert FreeSurfer .mgz file to NIfTI format.

    Parameters
    ----------
    mgz_file : Path
        Input .mgz file
    output_file : Path
        Output .nii.gz file

    Returns
    -------
    Path
        Path to output NIfTI file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'mri_convert',
        str(mgz_file),
        str(output_file)
    ]

    logger.debug(f"Converting {mgz_file.name} to NIfTI")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"mri_convert failed: {result.stderr}")

    return output_file


def compute_fs_to_t1w_transform(
    fs_orig_mgz: Path,
    t1w_brain: Path,
    output_dir: Path,
    cost: str = 'corratio'
) -> Tuple[Path, Path]:
    """
    Compute FreeSurfer conformed space to T1w preprocessing space transform.

    Uses FLIRT with correlation ratio cost function for T1-T1 registration.
    If FreeSurfer was run on the same T1 as preprocessing, this should be
    near-identity (translation/rotation only).

    Parameters
    ----------
    fs_orig_mgz : Path
        FreeSurfer orig.mgz file (conformed T1w)
    t1w_brain : Path
        Preprocessed T1w brain from anatomical workflow
    output_dir : Path
        Directory to save transform matrices
    cost : str
        FLIRT cost function (default: corratio)

    Returns
    -------
    tuple of (Path, Path)
        (fs_to_t1w_mat, t1w_to_fs_mat) - forward and inverse affine matrices
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert FreeSurfer orig.mgz to NIfTI if needed
    fs_orig_nii = output_dir / 'fs_orig.nii.gz'
    if not fs_orig_nii.exists() or fs_orig_mgz.stat().st_mtime > fs_orig_nii.stat().st_mtime:
        convert_mgz_to_nifti(fs_orig_mgz, fs_orig_nii)

    # Output paths
    fs_to_t1w_mat = output_dir / 'fs_to_t1w.mat'
    t1w_to_fs_mat = output_dir / 't1w_to_fs.mat'
    fs_in_t1w = output_dir / 'fs_orig_in_t1w.nii.gz'

    logger.info("Computing FreeSurfer → T1w transform")
    logger.info(f"  FreeSurfer orig: {fs_orig_mgz}")
    logger.info(f"  T1w reference: {t1w_brain}")

    # Run FLIRT: FS orig → T1w preprocessing space
    cmd = [
        'flirt',
        '-in', str(fs_orig_nii),
        '-ref', str(t1w_brain),
        '-omat', str(fs_to_t1w_mat),
        '-out', str(fs_in_t1w),
        '-cost', cost,
        '-dof', '6',  # Rigid body (should be near-identity if same scan)
        '-searchrx', '-30', '30',
        '-searchry', '-30', '30',
        '-searchrz', '-30', '30'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FLIRT fs→t1w failed: {result.stderr}")

    logger.info(f"  Forward transform: {fs_to_t1w_mat}")

    # Compute inverse transform
    cmd_inv = [
        'convert_xfm',
        '-omat', str(t1w_to_fs_mat),
        '-inverse', str(fs_to_t1w_mat)
    ]

    result = subprocess.run(cmd_inv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"convert_xfm inverse failed: {result.stderr}")

    logger.info(f"  Inverse transform: {t1w_to_fs_mat}")

    return fs_to_t1w_mat, t1w_to_fs_mat


def compute_t1w_to_dwi_transform(
    t1w_brain: Path,
    dwi_b0_brain: Path,
    output_dir: Path,
    cost: str = 'corratio',
    dof: int = 6
) -> Tuple[Path, Path]:
    """
    Compute T1w to DWI native space transform.

    Uses FLIRT with correlation ratio cost function. The transform maps
    T1w anatomical space to DWI native space for atlas warping.

    Parameters
    ----------
    t1w_brain : Path
        Preprocessed T1w brain (skull-stripped)
    dwi_b0_brain : Path
        DWI b0 brain (skull-stripped, from eddy output)
    output_dir : Path
        Directory to save transform matrices
    cost : str
        FLIRT cost function (default: corratio)
    dof : int
        Degrees of freedom (6=rigid, 12=affine)

    Returns
    -------
    tuple of (Path, Path)
        (t1w_to_dwi_mat, dwi_to_t1w_mat) - forward and inverse affine matrices
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    t1w_to_dwi_mat = output_dir / 't1w_to_dwi.mat'
    dwi_to_t1w_mat = output_dir / 'dwi_to_t1w.mat'
    t1w_in_dwi = output_dir / 't1w_brain_in_dwi.nii.gz'

    logger.info("Computing T1w → DWI transform")
    logger.info(f"  T1w brain: {t1w_brain}")
    logger.info(f"  DWI b0 reference: {dwi_b0_brain}")

    # Run FLIRT: T1w → DWI
    cmd = [
        'flirt',
        '-in', str(t1w_brain),
        '-ref', str(dwi_b0_brain),
        '-omat', str(t1w_to_dwi_mat),
        '-out', str(t1w_in_dwi),
        '-cost', cost,
        '-dof', str(dof),
        '-searchrx', '-30', '30',
        '-searchry', '-30', '30',
        '-searchrz', '-30', '30'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FLIRT t1w→dwi failed: {result.stderr}")

    logger.info(f"  Forward transform: {t1w_to_dwi_mat}")

    # Compute inverse transform
    cmd_inv = [
        'convert_xfm',
        '-omat', str(dwi_to_t1w_mat),
        '-inverse', str(t1w_to_dwi_mat)
    ]

    result = subprocess.run(cmd_inv, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"convert_xfm inverse failed: {result.stderr}")

    logger.info(f"  Inverse transform: {dwi_to_t1w_mat}")

    return t1w_to_dwi_mat, dwi_to_t1w_mat


def compose_fs_to_dwi_transform(
    fs_to_t1w_mat: Path,
    t1w_to_dwi_mat: Path,
    output_dir: Path
) -> Path:
    """
    Compose FreeSurfer → T1w → DWI into a single transform.

    Parameters
    ----------
    fs_to_t1w_mat : Path
        FreeSurfer to T1w affine matrix
    t1w_to_dwi_mat : Path
        T1w to DWI affine matrix
    output_dir : Path
        Directory to save composite transform

    Returns
    -------
    Path
        Path to composite FS → DWI transform matrix
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fs_to_dwi_mat = output_dir / 'fs_to_dwi.mat'

    logger.info("Composing FS → T1w → DWI transform")

    # convert_xfm -concat applies transforms in order: B then A
    # So: -concat t1w_to_dwi fs_to_t1w gives: t1w_to_dwi(fs_to_t1w(x))
    cmd = [
        'convert_xfm',
        '-omat', str(fs_to_dwi_mat),
        '-concat', str(t1w_to_dwi_mat), str(fs_to_t1w_mat)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"convert_xfm concat failed: {result.stderr}")

    logger.info(f"  Composite transform: {fs_to_dwi_mat}")

    return fs_to_dwi_mat


def apply_transform_to_volume(
    input_file: Path,
    reference_file: Path,
    transform_mat: Path,
    output_file: Path,
    interpolation: str = 'nearestneighbour'
) -> Path:
    """
    Apply linear transform to a volume.

    Parameters
    ----------
    input_file : Path
        Input volume to transform
    reference_file : Path
        Reference volume defining output space
    transform_mat : Path
        FLIRT transform matrix
    output_file : Path
        Output transformed volume
    interpolation : str
        Interpolation method: 'nearestneighbour' (for labels),
        'trilinear', 'spline'

    Returns
    -------
    Path
        Path to output transformed volume
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'flirt',
        '-in', str(input_file),
        '-ref', str(reference_file),
        '-applyxfm',
        '-init', str(transform_mat),
        '-out', str(output_file),
        '-interp', interpolation
    ]

    logger.debug(f"Applying transform to {input_file.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FLIRT applyxfm failed: {result.stderr}")

    return output_file


def transform_atlas_to_dwi(
    atlas_file: Path,
    dwi_reference: Path,
    transform_mat: Path,
    output_file: Path
) -> Path:
    """
    Transform a parcellation atlas to DWI native space.

    Uses nearest-neighbor interpolation to preserve label values.

    Parameters
    ----------
    atlas_file : Path
        Input atlas/parcellation (e.g., aparc+aseg.nii.gz)
    dwi_reference : Path
        DWI reference volume (defines output space)
    transform_mat : Path
        Transform matrix (FS→DWI or MNI→DWI)
    output_file : Path
        Output atlas in DWI space

    Returns
    -------
    Path
        Path to transformed atlas
    """
    logger.info(f"Transforming atlas to DWI space")
    logger.info(f"  Input: {atlas_file}")
    logger.info(f"  Reference: {dwi_reference}")
    logger.info(f"  Transform: {transform_mat}")

    result = apply_transform_to_volume(
        input_file=atlas_file,
        reference_file=dwi_reference,
        transform_mat=transform_mat,
        output_file=output_file,
        interpolation='nearestneighbour'
    )

    logger.info(f"  Output: {output_file}")

    return result


def validate_fs_t1w_alignment(
    fs_orig_nii: Path,
    t1w_brain: Path,
    transform_mat: Path,
    qc_dir: Path
) -> Dict[str, Any]:
    """
    Validate FreeSurfer and T1w preprocessing alignment.

    Computes cross-correlation before and after registration to verify
    that FreeSurfer was run on the same T1 scan as preprocessing.

    Parameters
    ----------
    fs_orig_nii : Path
        FreeSurfer orig volume (NIfTI format)
    t1w_brain : Path
        Preprocessed T1w brain
    transform_mat : Path
        FS → T1w transform matrix
    qc_dir : Path
        Directory to save QC outputs

    Returns
    -------
    dict
        QC metrics including:
        - correlation_before: Cross-correlation before registration
        - correlation_after: Cross-correlation after registration
        - is_same_scan: Boolean indicating likely same scan
        - warning: Warning message if alignment is poor
    """
    import nibabel as nib
    from scipy import ndimage

    qc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Validating FreeSurfer ↔ T1w alignment")

    # Load images
    fs_img = nib.load(fs_orig_nii)
    t1w_img = nib.load(t1w_brain)

    fs_data = fs_img.get_fdata()
    t1w_data = t1w_img.get_fdata()

    # Apply transform to FS
    fs_in_t1w_file = qc_dir / 'fs_orig_in_t1w_qc.nii.gz'
    apply_transform_to_volume(
        input_file=fs_orig_nii,
        reference_file=t1w_brain,
        transform_mat=transform_mat,
        output_file=fs_in_t1w_file,
        interpolation='trilinear'
    )

    fs_transformed = nib.load(fs_in_t1w_file).get_fdata()

    # Compute correlation after registration
    # Mask to brain region
    mask = t1w_data > np.percentile(t1w_data[t1w_data > 0], 10)

    t1w_masked = t1w_data[mask]
    fs_masked = fs_transformed[mask]

    # Normalize
    t1w_norm = (t1w_masked - np.mean(t1w_masked)) / np.std(t1w_masked)
    fs_norm = (fs_masked - np.mean(fs_masked)) / np.std(fs_masked)

    correlation_after = np.corrcoef(t1w_norm, fs_norm)[0, 1]

    # Determine if same scan (correlation should be very high)
    is_same_scan = correlation_after > 0.95
    warning = None

    if correlation_after < 0.90:
        warning = (
            f"LOW CORRELATION ({correlation_after:.3f}): FreeSurfer may have been "
            "run on a DIFFERENT T1 scan than preprocessing. Transforms may be inaccurate."
        )
        logger.warning(warning)
    elif correlation_after < 0.95:
        warning = (
            f"MODERATE CORRELATION ({correlation_after:.3f}): FreeSurfer and preprocessing "
            "T1w images are similar but not identical. Check registration QC carefully."
        )
        logger.warning(warning)
    else:
        logger.info(f"  Correlation: {correlation_after:.4f} (same scan confirmed)")

    results = {
        'correlation_after': float(correlation_after),
        'is_same_scan': is_same_scan,
        'warning': warning,
        'fs_in_t1w_file': str(fs_in_t1w_file)
    }

    # Save QC metrics
    import json
    qc_file = qc_dir / 'fs_t1w_alignment_qc.json'
    with open(qc_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"  QC metrics saved: {qc_file}")

    return results


def compute_all_transforms(
    subject: str,
    fs_subject_dir: Path,
    t1w_brain: Path,
    dwi_b0_brain: Path,
    output_dir: Path,
    run_qc: bool = True
) -> Dict[str, Path]:
    """
    Compute complete transform chain from FreeSurfer to DWI space.

    This is the main entry point for FreeSurfer transform computation.

    Parameters
    ----------
    subject : str
        Subject identifier
    fs_subject_dir : Path
        FreeSurfer subject directory
    t1w_brain : Path
        Preprocessed T1w brain
    dwi_b0_brain : Path
        DWI b0 brain (skull-stripped)
    output_dir : Path
        Output directory for transforms
    run_qc : bool
        Whether to run alignment validation

    Returns
    -------
    dict
        Dictionary of transform file paths:
        - 'fs_to_t1w': FS → T1w matrix
        - 't1w_to_fs': T1w → FS matrix (inverse)
        - 't1w_to_dwi': T1w → DWI matrix
        - 'dwi_to_t1w': DWI → T1w matrix (inverse)
        - 'fs_to_dwi': Composite FS → DWI matrix
        - 'qc': QC results (if run_qc=True)
    """
    logger.info("=" * 60)
    logger.info(f"Computing FreeSurfer → DWI Transforms for {subject}")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check FreeSurfer files exist
    fs_orig = fs_subject_dir / 'mri' / 'orig.mgz'
    if not fs_orig.exists():
        raise FileNotFoundError(f"FreeSurfer orig.mgz not found: {fs_orig}")

    results = {}

    # Step 1: FreeSurfer → T1w
    logger.info("")
    logger.info("Step 1: FreeSurfer → T1w registration")
    fs_to_t1w, t1w_to_fs = compute_fs_to_t1w_transform(
        fs_orig_mgz=fs_orig,
        t1w_brain=t1w_brain,
        output_dir=output_dir
    )
    results['fs_to_t1w'] = fs_to_t1w
    results['t1w_to_fs'] = t1w_to_fs

    # Step 2: T1w → DWI
    logger.info("")
    logger.info("Step 2: T1w → DWI registration")
    t1w_to_dwi, dwi_to_t1w = compute_t1w_to_dwi_transform(
        t1w_brain=t1w_brain,
        dwi_b0_brain=dwi_b0_brain,
        output_dir=output_dir
    )
    results['t1w_to_dwi'] = t1w_to_dwi
    results['dwi_to_t1w'] = dwi_to_t1w

    # Step 3: Compose FS → DWI
    logger.info("")
    logger.info("Step 3: Composing FS → T1w → DWI")
    fs_to_dwi = compose_fs_to_dwi_transform(
        fs_to_t1w_mat=fs_to_t1w,
        t1w_to_dwi_mat=t1w_to_dwi,
        output_dir=output_dir
    )
    results['fs_to_dwi'] = fs_to_dwi

    # Step 4: QC validation (optional)
    if run_qc:
        logger.info("")
        logger.info("Step 4: Validating alignment")
        qc_dir = output_dir / 'qc'
        fs_orig_nii = output_dir / 'fs_orig.nii.gz'
        qc_results = validate_fs_t1w_alignment(
            fs_orig_nii=fs_orig_nii,
            t1w_brain=t1w_brain,
            transform_mat=fs_to_t1w,
            qc_dir=qc_dir
        )
        results['qc'] = qc_results

    logger.info("")
    logger.info("Transform computation complete")
    logger.info(f"  Outputs: {output_dir}")

    return results
