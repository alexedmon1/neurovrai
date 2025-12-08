"""
Transform MNI-space atlases to subject functional space using registration transforms.
"""
from pathlib import Path
from typing import Tuple
import nibabel as nib
import subprocess
import logging

logger = logging.getLogger(__name__)


def transform_mni_atlas_to_func(
    atlas_mni: Path,
    reference_func: Path,
    t1w_brain: Path,
    mni_to_t1w_warp: Path,
    t1w_to_func_mat: Path,
    output_file: Path
) -> Path:
    """
    Transform MNI-space atlas to subject functional space.

    Transform chain: MNI → T1w → func

    Uses a two-step approach to handle mixed ANTs/FSL transforms:
    1. MNI → T1w: ANTs inverse of composite transform
    2. T1w → func: FSL matrix (pre-computed during functional preprocessing)

    Parameters
    ----------
    atlas_mni : Path
        Atlas label map in MNI space
    reference_func : Path
        Reference functional image (defines target space)
    t1w_brain : Path
        T1w brain image (intermediate reference)
    mni_to_t1w_warp : Path
        ANTs composite transform (anat→MNI forward, will be inverted)
    t1w_to_func_mat : Path
        FSL T1w→func matrix (pre-computed during functional preprocessing)
    output_file : Path
        Output path for transformed atlas

    Returns
    -------
    output_file : Path
        Path to transformed atlas
    """
    logger.info("Transforming atlas from MNI to functional space")
    logger.info(f"  Atlas: {atlas_mni.name}")
    logger.info(f"  Reference: {reference_func.name}")

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Intermediate file: atlas in T1w space
    atlas_in_t1w = output_file.parent / f"{output_file.stem}_in_t1w.nii.gz"

    # Step 1: MNI → T1w using ANTs inverse
    logger.info(f"  Step 1: MNI → T1w (ANTs inverse)")

    ants_cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_mni),
        '-r', str(t1w_brain),
        '-o', str(atlas_in_t1w),
        '-n', 'GenericLabel',  # Preserve integer labels
        '-t', f'Inverse[{str(mni_to_t1w_warp)}]'  # Inverse[] computes inverse on-the-fly
    ]

    logger.info(f"    Running: {' '.join(ants_cmd)}")

    try:
        result = subprocess.run(ants_cmd, check=True, capture_output=True, text=True)
        logger.info(f"    ✓ Transformed to T1w space: {atlas_in_t1w.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"  ANTs inverse failed: {e.stderr}")
        raise RuntimeError(f"Failed to transform atlas MNI→T1w: {e.stderr}")

    # Step 2: T1w → func using pre-computed FSL matrix
    logger.info(f"  Step 2: T1w → func (using pre-computed matrix)")

    # Apply the t1w_to_func transform directly
    flirt_cmd = [
        'flirt',
        '-in', str(atlas_in_t1w),
        '-ref', str(reference_func),
        '-applyxfm',
        '-init', str(t1w_to_func_mat),
        '-out', str(output_file),
        '-interp', 'nearestneighbour'  # Preserve label values
    ]

    logger.info(f"    Running: {' '.join(flirt_cmd)}")

    try:
        result = subprocess.run(flirt_cmd, check=True, capture_output=True, text=True)
        logger.info(f"  ✓ Transformed to functional space: {output_file.name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"  FLIRT failed: {e.stderr}")
        raise RuntimeError(f"Failed to transform atlas T1w→func: {e.stderr}")

    # Clean up intermediate files
    if atlas_in_t1w.exists():
        atlas_in_t1w.unlink()

    return output_file


def find_subject_transforms(derivatives_dir: Path, subject: str) -> Tuple[Path, Path, Path, Path]:
    """
    Find required transformation files for a subject.

    Parameters
    ----------
    derivatives_dir : Path
        Path to derivatives directory (e.g., /mnt/bytopia/IRC805/derivatives)
    subject : str
        Subject ID

    Returns
    -------
    t1w_brain : Path
        T1w brain image
    mni_to_t1w_warp : Path
        MNI → T1w ANTs transform
    t1w_to_func_mat : Path
        T1w → func FSL transform
    reference_func : Path
        Reference functional image
    """
    subject_deriv = derivatives_dir / subject

    # Find T1w brain - check brain directory
    t1w_brain_candidates = [
        subject_deriv / "anat" / "brain.nii.gz",
        list((subject_deriv / "anat" / "brain").glob("*brain.nii.gz"))[0] if (subject_deriv / "anat" / "brain").exists() else None,
    ]

    t1w_brain = None
    for candidate in t1w_brain_candidates:
        if candidate and candidate.exists():
            t1w_brain = candidate
            break

    if t1w_brain is None:
        raise FileNotFoundError(f"T1w brain not found in: {subject_deriv / 'anat'}")

    # Find MNI → T1w transform (inverse of T1w → MNI)
    mni_to_t1w_warp = subject_deriv / "anat" / "transforms" / "ants_Composite.h5"
    if not mni_to_t1w_warp.exists():
        raise FileNotFoundError(f"ANTs composite transform not found: {mni_to_t1w_warp}")

    # Find T1w → func transform (should use pre-computed t1w_to_func.mat, not func_to_t1w.mat)
    t1w_to_func_mat = subject_deriv / "func" / "acompcor" / "t1w_to_func.mat"
    if not t1w_to_func_mat.exists():
        raise FileNotFoundError(f"T1w→func transform not found: {t1w_to_func_mat}")

    # Find reference functional image (optcom preferred)
    ref_candidates = [
        subject_deriv / "func" / "tedana" / "tedana_desc-optcom_bold.nii.gz",
        subject_deriv / "func" / "tedana" / "tedana_desc-denoised_bold.nii.gz",
        subject_deriv / "func" / f"{subject}_bold_preprocessed.nii.gz",
    ]

    reference_func = None
    for ref in ref_candidates:
        if ref.exists():
            reference_func = ref
            break

    if reference_func is None:
        raise FileNotFoundError("No reference functional image found")

    return t1w_brain, mni_to_t1w_warp, t1w_to_func_mat, reference_func
