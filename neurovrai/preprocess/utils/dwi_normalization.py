"""
DWI Spatial Normalization Utilities

Provides functions for normalizing DWI data to standard space using FMRIB58_FA template.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

logger = logging.getLogger(__name__)


def normalize_dwi_to_fmrib58(
    fa_file: Path,
    output_dir: Path,
    fmrib58_template: Optional[Path] = None,
    study_root: Optional[Path] = None,
    subject: Optional[str] = None
) -> Dict[str, Path]:
    """
    Normalize FA map to FMRIB58_FA standard space.

    Generates both forward warp (for group analyses) and inverse warp
    (for bringing atlas ROIs to native DWI space for tractography).

    Parameters
    ----------
    fa_file : Path
        FA map in native DWI space
    output_dir : Path
        Output directory for transforms and normalized FA
    fmrib58_template : Path, optional
        Path to FMRIB58_FA template. If None, uses $FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz
    study_root : Path, optional
        Study root directory. If provided with subject, transforms will be copied
        to the standardized location: {study_root}/transforms/{subject}/
    subject : str, optional
        Subject identifier. Required if study_root is provided.

    Returns
    -------
    dict
        Dictionary containing paths to:
        - affine_mat: Affine transformation matrix
        - forward_warp: FA → FMRIB58 warp (for group analyses)
        - inverse_warp: FMRIB58 → FA warp (for tractography ROIs)
        - fa_normalized: FA in FMRIB58 space

    Notes
    -----
    Uses FLIRT (affine) + FNIRT (nonlinear) registration.
    Forward warp is used to normalize all DWI metrics to standard space.
    Inverse warp is used to warp atlas ROIs to native DWI space for tractography.

    If study_root and subject are provided, transforms are also saved to the
    standardized location using the naming convention:
    - fa-fmrib58-affine.mat
    - fa-fmrib58-warp.nii.gz
    - fmrib58-fa-warp.nii.gz
    """
    logger.info("=" * 70)
    logger.info("DWI Normalization to FMRIB58_FA")
    logger.info("=" * 70)
    logger.info("")

    # Setup output directories
    output_dir = Path(output_dir)
    transforms_dir = output_dir / 'transforms'
    normalized_dir = output_dir / 'normalized'

    transforms_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    # Get FMRIB58_FA template
    if fmrib58_template is None:
        fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')
        fmrib58_template = Path(fsldir) / 'data' / 'standard' / 'FMRIB58_FA_1mm.nii.gz'

    if not Path(fmrib58_template).exists():
        raise FileNotFoundError(f"FMRIB58_FA template not found: {fmrib58_template}")

    logger.info(f"FA input: {fa_file}")
    logger.info(f"Template: {fmrib58_template}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Step 1: Affine registration (FLIRT)
    logger.info("Step 1: Computing affine registration (FLIRT)...")
    affine_mat = transforms_dir / 'FA_to_FMRIB58_affine.mat'
    fa_affine = transforms_dir / 'FA_to_FMRIB58_affine.nii.gz'

    flirt_cmd = [
        'flirt',
        '-in', str(fa_file),
        '-ref', str(fmrib58_template),
        '-omat', str(affine_mat),
        '-out', str(fa_affine),
        '-dof', '12',
        '-cost', 'corratio',
        '-searchcost', 'corratio'
    ]

    logger.info(f"  Running: {' '.join(flirt_cmd)}")
    result = subprocess.run(flirt_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FLIRT failed: {result.stderr}")
        raise RuntimeError(f"FLIRT registration failed")

    logger.info(f"  Affine transform saved: {affine_mat}")
    logger.info("")

    # Step 2: Nonlinear registration (FNIRT)
    logger.info("Step 2: Computing nonlinear registration (FNIRT)...")
    forward_warp = transforms_dir / 'FA_to_FMRIB58_warp.nii.gz'
    fa_normalized = normalized_dir / 'FA_normalized.nii.gz'

    fnirt_cmd = [
        'fnirt',
        '--in=' + str(fa_file),
        '--ref=' + str(fmrib58_template),
        '--aff=' + str(affine_mat),
        '--cout=' + str(forward_warp),
        '--iout=' + str(fa_normalized),
        '--config=FA_2_FMRIB58_1mm'  # FSL config for FA→FMRIB58
    ]

    logger.info(f"  Running FNIRT (this may take 5-10 minutes)...")
    result = subprocess.run(fnirt_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FNIRT failed: {result.stderr}")
        raise RuntimeError(f"FNIRT registration failed")

    logger.info(f"  Forward warp saved: {forward_warp}")
    logger.info(f"  Normalized FA saved: {fa_normalized}")
    logger.info("")

    # Step 3: Compute inverse warp (for tractography ROIs)
    logger.info("Step 3: Computing inverse warp (FMRIB58 → native DWI)...")
    inverse_warp = transforms_dir / 'FMRIB58_to_FA_warp.nii.gz'

    invwarp_cmd = [
        'invwarp',
        '--ref=' + str(fa_file),
        '--warp=' + str(forward_warp),
        '--out=' + str(inverse_warp)
    ]

    logger.info(f"  Running: {' '.join(invwarp_cmd)}")
    result = subprocess.run(invwarp_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"invwarp failed: {result.stderr}")
        raise RuntimeError(f"Inverse warp computation failed")

    logger.info(f"  Inverse warp saved: {inverse_warp}")
    logger.info("")

    logger.info("DWI normalization complete!")
    logger.info("")

    # Copy transforms to standardized location if study_root and subject provided
    if study_root and subject:
        import shutil
        std_transforms_dir = Path(study_root) / 'transforms' / subject
        std_transforms_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Copying transforms to standardized location...")
        logger.info(f"  Location: {std_transforms_dir}")

        # Standardized naming: {source}-{target}-{type}.{ext}
        std_affine = std_transforms_dir / 'fa-fmrib58-affine.mat'
        std_forward = std_transforms_dir / 'fa-fmrib58-warp.nii.gz'
        std_inverse = std_transforms_dir / 'fmrib58-fa-warp.nii.gz'

        shutil.copy2(affine_mat, std_affine)
        shutil.copy2(forward_warp, std_forward)
        shutil.copy2(inverse_warp, std_inverse)

        logger.info(f"  Copied: fa-fmrib58-affine.mat")
        logger.info(f"  Copied: fa-fmrib58-warp.nii.gz")
        logger.info(f"  Copied: fmrib58-fa-warp.nii.gz")
        logger.info("")

    return {
        'affine_mat': affine_mat,
        'forward_warp': forward_warp,
        'inverse_warp': inverse_warp,
        'fa_normalized': fa_normalized
    }


def apply_warp_to_metrics(
    metric_files: List[Path],
    forward_warp: Path,
    fmrib58_template: Optional[Path],
    output_dir: Path,
    interpolation: str = 'spline'
) -> List[Path]:
    """
    Apply forward warp to DWI metric maps.

    Parameters
    ----------
    metric_files : list of Path
        List of DWI metric files to normalize (FA, MD, AD, RD, MK, etc.)
    forward_warp : Path
        Forward warp field (FA → FMRIB58)
    fmrib58_template : Path
        FMRIB58_FA template (reference space)
    output_dir : Path
        Output directory for normalized metrics
    interpolation : str
        Interpolation method ('spline', 'trilinear', 'nn')

    Returns
    -------
    list of Path
        Paths to normalized metric files
    """
    logger.info("=" * 70)
    logger.info("Applying Normalization to DWI Metrics")
    logger.info("=" * 70)
    logger.info("")

    output_dir = Path(output_dir)
    normalized_dir = output_dir / 'normalized'
    normalized_dir.mkdir(parents=True, exist_ok=True)

    # Get FMRIB58_FA template
    if fmrib58_template is None:
        fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')
        fmrib58_template = Path(fsldir) / 'data' / 'standard' / 'FMRIB58_FA_1mm.nii.gz'

    normalized_files = []

    for metric_file in metric_files:
        metric_file = Path(metric_file)
        metric_name = metric_file.stem.replace('.nii', '')
        output_file = normalized_dir / f'{metric_name}_normalized.nii.gz'

        logger.info(f"Normalizing {metric_name}...")

        applywarp_cmd = [
            'applywarp',
            '--in=' + str(metric_file),
            '--ref=' + str(fmrib58_template),
            '--warp=' + str(forward_warp),
            '--out=' + str(output_file),
            '--interp=' + interpolation
        ]

        result = subprocess.run(applywarp_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"applywarp failed for {metric_name}: {result.stderr}")
            continue

        logger.info(f"  Saved: {output_file}")
        normalized_files.append(output_file)

    logger.info("")
    logger.info(f"Normalized {len(normalized_files)} metrics")
    logger.info("")

    return normalized_files
