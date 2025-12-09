"""
Functional Registration Utilities using ANTs

This module provides ANTs-based registration for functional MRI data,
including:
- fMRI → T1w registration on raw motion-corrected data
- Transform concatenation (fMRI → T1w → MNI)
- Inverse transform computation (MNI → fMRI for atlas transformation)

Key principle: Registration is performed on RAW motion-corrected functional
data (before any filtering) to preserve structural information for alignment.
"""

import logging
import subprocess
from pathlib import Path
from typing import Tuple, Optional

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def compute_func_mean(func_file: Path, output_file: Path) -> Path:
    """
    Compute temporal mean of 4D functional data.

    Args:
        func_file: 4D functional timeseries
        output_file: Output path for mean image

    Returns:
        Path to mean functional image
    """
    logger.info(f"Computing temporal mean: {func_file.name}")

    img = nib.load(func_file)
    # Use dataobj for memory efficiency (memory-mapped, not loaded fully)
    data = np.asarray(img.dataobj)

    # Compute mean across time (4th dimension)
    mean_data = np.mean(data, axis=3)

    # Save as 3D image
    mean_img = nib.Nifti1Image(mean_data, img.affine, img.header)
    nib.save(mean_img, output_file)

    logger.info(f"  Mean image saved: {output_file}")
    return output_file


def register_func_to_t1w_ants(
    func_mean: Path,
    t1w_brain: Path,
    output_dir: Path,
    output_prefix: str = "func_to_t1w",
    use_quick: bool = True
) -> Tuple[Path, Path]:
    """
    Register functional mean to T1w using ANTs.

    Uses antsRegistrationSyNQuick.sh for robust EPI → T1w alignment.
    This performs rigid + affine + deformable SyN registration.

    Args:
        func_mean: Mean functional image (motion-corrected, unfiltered)
        t1w_brain: T1w brain-extracted reference
        output_dir: Output directory for transforms
        output_prefix: Prefix for output files
        use_quick: Use antsRegistrationSyNQuick.sh (recommended)

    Returns:
        Tuple of (composite_transform, warped_image)

    Note:
        The composite transform includes both affine and deformable components.
        For inverse transforms, ANTs will automatically handle both components.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ANTs Functional → T1w Registration")
    logger.info("=" * 70)
    logger.info(f"Moving image: {func_mean}")
    logger.info(f"Reference: {t1w_brain}")
    logger.info("")

    # Output files from antsRegistrationSyNQuick.sh
    # It creates: prefix0GenericAffine.mat, prefix1Warp.nii.gz, prefixWarped.nii.gz
    affine_transform = output_dir / f"{output_prefix}0GenericAffine.mat"
    warp_transform = output_dir / f"{output_prefix}1Warp.nii.gz"
    warped_image = output_dir / f"{output_prefix}Warped.nii.gz"

    # Check if already computed
    if affine_transform.exists() and warp_transform.exists() and warped_image.exists():
        logger.info("Registration already completed - using cached results")
        logger.info(f"  Affine: {affine_transform}")
        logger.info(f"  Warp: {warp_transform}")
        logger.info(f"  Warped: {warped_image}")
        # Return list of transforms - ANTs applies in reverse order
        return [warp_transform, affine_transform], warped_image

    if use_quick:
        # Use antsRegistrationSyNQuick.sh for robust EPI → T1w
        # Using affine-only (no deformable) since T1w→MNI already has non-linear component
        logger.info("Running antsRegistrationSyNQuick.sh (rigid + affine, no deformable)...")

        cmd = [
            'antsRegistrationSyNQuick.sh',
            '-d', '3',                           # 3D
            '-f', str(t1w_brain),                # Fixed image (T1w)
            '-m', str(func_mean),                # Moving image (functional)
            '-t', 'a',                           # Transform type: a = affine (rigid+affine, no deformable)
            '-o', str(output_dir / output_prefix)  # Output prefix
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("  Registration completed successfully")

        except subprocess.CalledProcessError as e:
            logger.error(f"antsRegistrationSyNQuick.sh failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(
                f"ANTs registration failed. Check that moving and fixed images "
                f"have sufficient overlap. You may need to manually check orientation."
            ) from e
    else:
        # Full antsRegistration command (more control, slower)
        raise NotImplementedError("Custom antsRegistration not yet implemented")

    # Verify outputs exist
    if not affine_transform.exists():
        raise FileNotFoundError(
            f"Expected affine transform not found: {affine_transform}"
        )
    if not warped_image.exists():
        raise FileNotFoundError(
            f"Expected warped image not found: {warped_image}"
        )

    # For affine-only (-t a), only affine transform is created
    # For SyN (-t s), both affine and warp are created
    if warp_transform.exists():
        logger.info(f"  Affine: {affine_transform}")
        logger.info(f"  Warp: {warp_transform}")
        logger.info(f"  Warped: {warped_image}")
        logger.info("")
        # Return list of transforms - ANTs applies in reverse order: [warp, affine]
        return [warp_transform, affine_transform], warped_image
    else:
        # Affine-only registration
        logger.info(f"  Affine: {affine_transform}")
        logger.info(f"  Warped: {warped_image}")
        logger.info("")
        # Return single transform
        return affine_transform, warped_image


def concatenate_transforms_ants(
    transform1: Path | list[Path],
    transform2: Path,
    output_transform: Path,
    reference_image: Path
) -> Path:
    """
    Concatenate two ANTs transforms into a single composite transform.

    This creates a single transform that applies transform1 followed by transform2.
    For fMRI → MNI: transform1 = func→T1w, transform2 = T1w→MNI

    Args:
        transform1: First transform(s) - can be single Path or list of Paths
                   (e.g., [warp.nii.gz, affine.mat] or just affine.mat)
        transform2: Second transform (e.g., T1w_to_MNI_Composite.h5)
        output_transform: Output composite transform
        reference_image: Reference image for the final space (e.g., MNI template)

    Returns:
        Path to concatenated transform

    Note:
        ANTs applies transforms in REVERSE order, so the command syntax is:
        antsApplyTransforms ... -t transform2 -t transform1
        This applies transform1 first, then transform2.
    """
    logger.info("Concatenating transforms...")

    # Handle both single Path and list of Paths
    if isinstance(transform1, list):
        logger.info(f"  Transform 1: {[t.name for t in transform1]}")
    else:
        logger.info(f"  Transform 1: {transform1.name}")
        transform1 = [transform1]  # Convert to list for uniform handling

    logger.info(f"  Transform 2: {transform2.name}")
    logger.info(f"  Output: {output_transform.name}")

    # Check if already exists
    if output_transform.exists():
        logger.info("  Concatenated transform already exists - using cached version")
        return output_transform

    output_transform.parent.mkdir(parents=True, exist_ok=True)

    # Create a composite transform by applying transforms in sequence
    # We'll use antsApplyTransforms with --default-value to create the composite
    # Note: ANTs doesn't have a direct "concatenate" command, so we create
    # a composite by saving the transform chain

    # For ANTs composite transforms, we need to use antsApplyTransforms
    # to generate a combined transform. However, the better approach is
    # to simply pass both transforms in the correct order to antsApplyTransforms
    # when applying to an image.

    # Create a text file listing the transforms in application order
    transform_list_file = output_transform.parent / "transform_list.txt"
    with open(transform_list_file, 'w') as f:
        # ANTs applies in reverse order listed
        f.write(f"{transform2}\n")
        for t in reversed(transform1):  # Handle list of transforms
            f.write(f"{t}\n")

    logger.info(f"  Transform list saved: {transform_list_file}")
    logger.info("  Note: For ANTs, pass transforms to antsApplyTransforms in reverse order")

    # Actually, let's create a proper composite using ANTs
    # We'll use antsApplyTransforms to create a displacement field
    # that represents the combined transformation

    logger.info("  Creating composite displacement field...")

    # Build command with all transforms
    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-r', str(reference_image),
        '-t', str(transform2),
    ]

    # Add all transform1 components in reverse order (ANTs applies in reverse)
    for t in reversed(transform1):
        cmd.extend(['-t', str(t)])

    cmd.extend(['-o', f'[{output_transform},1]'])  # Output displacement field

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("  Composite transform created successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create composite transform: {e}")
        logger.error(f"STDERR: {e.stderr.decode()}")
        raise

    return output_transform


def invert_transform_ants(
    forward_transform: Path,
    output_transform: Path,
    reference_image: Path
) -> Path:
    """
    Compute inverse of an ANTs transform.

    For composite transforms (.h5), ANTs can compute the inverse directly.
    This is used to create MNI → fMRI transforms for atlas transformation.

    Args:
        forward_transform: Forward transform (e.g., func_to_mni_Composite.h5)
        output_transform: Output inverse transform (e.g., mni_to_func_Composite.h5)
        reference_image: Reference image in source space (e.g., functional mean)

    Returns:
        Path to inverse transform

    Note:
        For ANTs composite transforms, the inverse is computed automatically
        when using antsApplyTransforms with the flag: -t [transform,1]
        where the "1" indicates "use inverse"
    """
    logger.info("Computing inverse transform...")
    logger.info(f"  Forward: {forward_transform.name}")
    logger.info(f"  Output: {output_transform.name}")

    # Check if already exists
    if output_transform.exists():
        logger.info("  Inverse transform already exists - using cached version")
        return output_transform

    output_transform.parent.mkdir(parents=True, exist_ok=True)

    # For ANTs, we create an inverse displacement field
    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-r', str(reference_image),
        '-t', f'[{forward_transform},1]',  # Use inverse of transform
        '-o', f'[{output_transform},1]'     # Output as displacement field
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("  Inverse transform computed successfully")
        logger.info(f"  Saved: {output_transform}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to compute inverse transform: {e}")
        logger.error(f"STDERR: {e.stderr.decode()}")
        raise

    return output_transform


def apply_transform_to_atlas(
    atlas_file: Path,
    transform: Path,
    reference_image: Path,
    output_file: Path,
    interpolation: str = 'NearestNeighbor'
) -> Path:
    """
    Apply ANTs transform to bring atlas from MNI to functional space.

    Args:
        atlas_file: Atlas in MNI space
        transform: Inverse transform (MNI → functional)
        reference_image: Functional space reference
        output_file: Output atlas in functional space
        interpolation: Interpolation method (NearestNeighbor for labels)

    Returns:
        Path to transformed atlas
    """
    logger.info("Transforming atlas to functional space...")
    logger.info(f"  Atlas: {atlas_file.name}")
    logger.info(f"  Transform: {transform.name}")
    logger.info(f"  Reference: {reference_image.name}")

    if output_file.exists():
        logger.info("  Transformed atlas already exists - using cached version")
        return output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(atlas_file),
        '-r', str(reference_image),
        '-t', str(transform),
        '-n', interpolation,
        '-o', str(output_file)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  Atlas transformed: {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to transform atlas: {e}")
        logger.error(f"STDERR: {e.stderr.decode()}")
        raise

    return output_file


def apply_inverse_transform_to_masks(
    csf_mask: Path,
    wm_mask: Path,
    t1w_to_func_transform: Path,
    reference_image: Path,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Apply inverse ANTs transform to bring T1w-space masks to functional space.

    This is used for ACompCor: transform CSF and WM masks from T1w to
    functional space using the inverse of the func→T1w registration.

    Args:
        csf_mask: CSF probability map in T1w space
        wm_mask: WM probability map in T1w space
        t1w_to_func_transform: Inverse transform (T1w → functional)
        reference_image: Functional space reference
        output_dir: Output directory for transformed masks

    Returns:
        Tuple of (csf_func, wm_func) - masks in functional space
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Transforming tissue masks to functional space...")
    logger.info(f"  CSF mask: {csf_mask.name}")
    logger.info(f"  WM mask: {wm_mask.name}")
    logger.info(f"  Transform: {t1w_to_func_transform.name}")
    logger.info(f"  Reference: {reference_image.name}")

    # Transform CSF mask
    csf_func = output_dir / 'csf_func.nii.gz'
    if not csf_func.exists():
        logger.info("  Transforming CSF mask...")
        cmd = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(csf_mask),
            '-r', str(reference_image),
            '-t', f'[{t1w_to_func_transform},1]',  # Use inverse
            '-n', 'Linear',  # Linear for probability maps
            '-o', str(csf_func)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"    Saved: {csf_func}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to transform CSF mask: {e}")
            logger.error(f"STDERR: {e.stderr.decode()}")
            raise
    else:
        logger.info(f"  Using cached CSF mask: {csf_func}")

    # Transform WM mask
    wm_func = output_dir / 'wm_func.nii.gz'
    if not wm_func.exists():
        logger.info("  Transforming WM mask...")
        cmd = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(wm_mask),
            '-r', str(reference_image),
            '-t', f'[{t1w_to_func_transform},1]',  # Use inverse
            '-n', 'Linear',  # Linear for probability maps
            '-o', str(wm_func)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"    Saved: {wm_func}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to transform WM mask: {e}")
            logger.error(f"STDERR: {e.stderr.decode()}")
            raise
    else:
        logger.info(f"  Using cached WM mask: {wm_func}")

    logger.info("  Mask transformation complete")
    logger.info("")

    return csf_func, wm_func


def create_func_to_mni_transforms(
    func_mean: Path,
    t1w_brain: Path,
    t1w_to_mni_transform: Path,
    mni_template: Path,
    output_dir: Path,
    recompute: bool = False
) -> dict:
    """
    Create fMRI → T1w → MNI transform pipeline for preprocessing.

    This function sets up forward transforms for functional preprocessing.
    The inverse MNI → fMRI transform is computed separately in the
    connectivity pipeline where it's needed for atlas transformation.

    Performs:
    1. fMRI → T1w registration (affine, on raw motion-corrected mean)
    2. Concatenate with T1w → MNI (from anatomical preprocessing)

    Args:
        func_mean: Motion-corrected functional mean (unfiltered!)
        t1w_brain: T1w brain-extracted reference
        t1w_to_mni_transform: Existing T1w → MNI composite transform
        mni_template: MNI template image
        output_dir: Output directory for transforms
        recompute: Force recomputation even if transforms exist

    Returns:
        Dictionary with keys:
            'func_to_t1w': func → T1w transform
            'func_to_mni': func → MNI composite transform
            'func_warped_to_t1w': Warped functional image in T1w space
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Creating Functional → MNI Transform Pipeline")
    logger.info("=" * 70)
    logger.info("")

    results = {}

    # Step 1: Register fMRI → T1w
    logger.info("Step 1: Registering functional → T1w...")
    func_to_t1w, func_warped = register_func_to_t1w_ants(
        func_mean=func_mean,
        t1w_brain=t1w_brain,
        output_dir=output_dir,
        output_prefix="func_to_t1w"
    )
    results['func_to_t1w'] = func_to_t1w
    results['func_warped_to_t1w'] = func_warped
    logger.info("")

    # Step 2: Concatenate func → T1w → MNI
    logger.info("Step 2: Concatenating func → T1w → MNI...")
    func_to_mni = output_dir / "func_to_mni_Composite.h5"

    if not func_to_mni.exists() or recompute:
        func_to_mni = concatenate_transforms_ants(
            transform1=func_to_t1w,
            transform2=t1w_to_mni_transform,
            output_transform=func_to_mni,
            reference_image=mni_template
        )
    else:
        logger.info("  Concatenated transform already exists - using cached version")
        logger.info(f"  {func_to_mni}")

    results['func_to_mni'] = func_to_mni
    logger.info("")

    logger.info("=" * 70)
    logger.info("Transform Pipeline Complete")
    logger.info("=" * 70)
    logger.info(f"func → T1w: {func_to_t1w}")
    logger.info(f"func → MNI: {func_to_mni}")
    logger.info("")
    logger.info("Note: Inverse MNI → func transform will be computed")
    logger.info("      in connectivity pipeline when needed for atlas transformation")
    logger.info("")

    return results
