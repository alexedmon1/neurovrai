"""
Functional MRI Spatial Normalization Utilities

Provides functions for normalizing functional data to MNI152 standard space by
reusing anatomical transforms and BBR transform from ACompCor.

This implements a transform reuse strategy:
1. BBR transform (func→anat) computed during ACompCor - REUSED, not recomputed
2. Anatomical transforms (anat→MNI152) from anatomical preprocessing - REUSED
3. Concatenate transforms for efficient single-step normalization
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)


def normalize_func_to_mni152(
    func_file: Path,
    func_to_anat_bbr: Path,
    t1w_to_mni_affine: Path,
    t1w_to_mni_warp: Path,
    output_dir: Path,
    mni152_template: Optional[Path] = None,
    interpolation: str = 'spline'
) -> Dict[str, Path]:
    """
    Normalize functional data to MNI152 standard space using pre-computed transforms.

    This function implements efficient transform reuse by concatenating:
    - func→anat (BBR transform from ACompCor)
    - anat→MNI152 (affine + warp from anatomical preprocessing)

    Parameters
    ----------
    func_file : Path
        Preprocessed functional 4D image (after TEDANA, bandpass, smoothing)
    func_to_anat_bbr : Path
        BBR transform from functional to anatomical space (from ACompCor)
    t1w_to_mni_affine : Path
        Affine transformation matrix from anatomical preprocessing
    t1w_to_mni_warp : Path
        Nonlinear warp field from anatomical preprocessing (FNIRT)
    output_dir : Path
        Output directory for normalized functional data and transforms
    mni152_template : Path, optional
        Path to MNI152 template. If None, uses $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz
    interpolation : str
        Interpolation method ('spline', 'trilinear', 'nn'). Default: 'spline'

    Returns
    -------
    dict
        Dictionary containing paths to:
        - func_to_mni_warp: Concatenated warp field (func→MNI152)
        - func_normalized: Normalized functional data in MNI152 space

    Notes
    -----
    Transform reuse strategy:
    - BBR transform computed once during ACompCor, saved for reuse here
    - Anatomical transforms computed once during anatomical preprocessing
    - No redundant computation, only concatenation and application

    Examples
    --------
    >>> results = normalize_func_to_mni152(
    ...     func_file=Path('bold_preprocessed.nii.gz'),
    ...     func_to_anat_bbr=Path('func_to_anat_bbr.mat'),
    ...     t1w_to_mni_affine=Path('T1w_to_MNI152_affine.mat'),
    ...     t1w_to_mni_warp=Path('T1w_to_MNI152_warp.nii.gz'),
    ...     output_dir=Path('/derivatives/func')
    ... )
    """
    logger.info("="*70)
    logger.info("Functional Normalization to MNI152")
    logger.info("="*70)
    logger.info("")

    # Setup output directories
    output_dir = Path(output_dir)
    transforms_dir = output_dir / 'transforms'
    normalized_dir = output_dir / 'normalized'

    transforms_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    # Get MNI152 template
    if mni152_template is None:
        fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')
        mni152_template = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm.nii.gz'

    if not Path(mni152_template).exists():
        raise FileNotFoundError(f"MNI152 template not found: {mni152_template}")

    logger.info(f"Functional input: {func_file}")
    logger.info(f"Template: {mni152_template}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Reusing transforms:")
    logger.info(f"  1. BBR (func→anat): {func_to_anat_bbr}")
    logger.info(f"  2. Affine (anat→MNI): {t1w_to_mni_affine}")
    logger.info(f"  3. Warp (anat→MNI): {t1w_to_mni_warp}")
    logger.info("")

    # Step 1: Concatenate transforms (func→anat→MNI152)
    logger.info("Step 1: Concatenating transforms (func→anat→MNI152)...")
    func_to_mni_warp = transforms_dir / 'func_to_MNI152_warp.nii.gz'

    convertwarp_cmd = [
        'convertwarp',
        '--ref=' + str(mni152_template),
        '--premat=' + str(func_to_anat_bbr),
        '--warp1=' + str(t1w_to_mni_warp),
        '--out=' + str(func_to_mni_warp)
    ]

    logger.info(f"  Running: {' '.join(convertwarp_cmd)}")
    result = subprocess.run(convertwarp_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"convertwarp failed: {result.stderr}")
        raise RuntimeError(f"Transform concatenation failed")

    logger.info(f"  Concatenated warp saved: {func_to_mni_warp}")
    logger.info("")

    # Step 2: Apply concatenated warp to functional data
    logger.info("Step 2: Applying normalization to functional data...")
    logger.info("  This may take 2-5 minutes for 4D time series...")
    func_normalized = normalized_dir / 'bold_normalized.nii.gz'

    applywarp_cmd = [
        'applywarp',
        '--in=' + str(func_file),
        '--ref=' + str(mni152_template),
        '--warp=' + str(func_to_mni_warp),
        '--out=' + str(func_normalized),
        '--interp=' + interpolation
    ]

    logger.info(f"  Running applywarp...")
    result = subprocess.run(applywarp_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"applywarp failed: {result.stderr}")
        raise RuntimeError(f"Functional normalization failed")

    logger.info(f"  Normalized functional data saved: {func_normalized}")
    logger.info("")

    logger.info("Functional normalization complete!")
    logger.info("")

    return {
        'func_to_mni_warp': func_to_mni_warp,
        'func_normalized': func_normalized
    }
