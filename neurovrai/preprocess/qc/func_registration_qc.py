"""
Quality Control for Functional Registration

This module provides visualization functions to validate ANTs-based
functional registration quality, including:
- fMRI → T1w alignment
- fMRI → MNI alignment
- Tissue mask alignment (CSF, WM) in functional space
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import subprocess

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


def create_overlay_mosaic(
    background: Path,
    overlay: Path,
    output_file: Path,
    title: str = "Registration QC",
    overlay_alpha: float = 0.5,
    num_slices: int = 9,
    cmap_overlay: str = 'hot',
    vmin_percentile: float = 2,
    vmax_percentile: float = 98
) -> Path:
    """
    Create mosaic visualization of overlay on background image.

    Parameters
    ----------
    background : Path
        Background image (e.g., T1w or functional reference)
    overlay : Path
        Overlay image (e.g., warped functional or tissue mask)
    output_file : Path
        Output path for PNG mosaic
    title : str
        Title for the mosaic
    overlay_alpha : float
        Alpha transparency for overlay (0-1)
    num_slices : int
        Number of slices to show
    cmap_overlay : str
        Colormap for overlay
    vmin_percentile : float
        Lower percentile for intensity scaling
    vmax_percentile : float
        Upper percentile for intensity scaling

    Returns
    -------
    Path
        Output file path
    """
    logger.info(f"Creating overlay mosaic: {title}")
    logger.info(f"  Background: {background.name}")
    logger.info(f"  Overlay: {overlay.name}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load images (use dataobj for memory efficiency - don't load full array at once)
    bg_img = nib.load(background)
    bg_data = np.asarray(bg_img.dataobj)  # Memory-mapped, not copied

    overlay_img = nib.load(overlay)
    overlay_data = np.asarray(overlay_img.dataobj)  # Memory-mapped, not copied

    # Ensure same shape (pad or resample if needed)
    if bg_data.shape != overlay_data.shape:
        logger.warning(f"Shape mismatch: background {bg_data.shape} vs overlay {overlay_data.shape}")
        logger.warning("Attempting to continue with available data...")

    # Normalize background to 0-1
    bg_min = np.percentile(bg_data[bg_data > 0], vmin_percentile)
    bg_max = np.percentile(bg_data[bg_data > 0], vmax_percentile)
    bg_norm = np.clip((bg_data - bg_min) / (bg_max - bg_min), 0, 1)

    # Normalize overlay to 0-1
    if overlay_data.max() > 0:
        overlay_min = np.percentile(overlay_data[overlay_data > 0], vmin_percentile)
        overlay_max = np.percentile(overlay_data[overlay_data > 0], vmax_percentile)
        overlay_norm = np.clip((overlay_data - overlay_min) / (overlay_max - overlay_min), 0, 1)
    else:
        overlay_norm = overlay_data

    # Create figure with 3 rows (axial, coronal, sagittal)
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, num_slices, hspace=0.3, wspace=0.05)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Axial slices
    z_slices = np.linspace(10, bg_data.shape[2] - 10, num_slices, dtype=int)
    for i, z in enumerate(z_slices):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(bg_norm[:, :, z].T, cmap='gray', origin='lower', aspect='auto')
        if overlay_norm.shape[2] > z:
            ax.imshow(overlay_norm[:, :, z].T, cmap=cmap_overlay, alpha=overlay_alpha,
                     origin='lower', aspect='auto')
        ax.axis('off')
        if i == 0:
            ax.set_title('Axial', fontsize=10, loc='left')

    # Coronal slices
    y_slices = np.linspace(10, bg_data.shape[1] - 10, num_slices, dtype=int)
    for i, y in enumerate(y_slices):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(bg_norm[:, y, :].T, cmap='gray', origin='lower', aspect='auto')
        if overlay_norm.shape[1] > y:
            ax.imshow(overlay_norm[:, y, :].T, cmap=cmap_overlay, alpha=overlay_alpha,
                     origin='lower', aspect='auto')
        ax.axis('off')
        if i == 0:
            ax.set_title('Coronal', fontsize=10, loc='left')

    # Sagittal slices
    x_slices = np.linspace(10, bg_data.shape[0] - 10, num_slices, dtype=int)
    for i, x in enumerate(x_slices):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(bg_norm[x, :, :].T, cmap='gray', origin='lower', aspect='auto')
        if overlay_norm.shape[0] > x:
            ax.imshow(overlay_norm[x, :, :].T, cmap=cmap_overlay, alpha=overlay_alpha,
                     origin='lower', aspect='auto')
        ax.axis('off')
        if i == 0:
            ax.set_title('Sagittal', fontsize=10, loc='left')

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"  Saved: {output_file}")
    return output_file


def create_edge_overlay(
    background: Path,
    overlay: Path,
    output_file: Path,
    title: str = "Edge Overlay QC",
    num_slices: int = 9
) -> Path:
    """
    Create edge-based overlay for registration QC.

    Shows edges of overlay in color on background grayscale image.
    Useful for checking boundary alignment.

    Parameters
    ----------
    background : Path
        Background image
    overlay : Path
        Overlay image
    output_file : Path
        Output path for PNG
    title : str
        Title for the visualization
    num_slices : int
        Number of slices to display

    Returns
    -------
    Path
        Output file path
    """
    logger.info(f"Creating edge overlay: {title}")

    from scipy import ndimage

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load images (use dataobj for memory efficiency - don't load full array at once)
    bg_img = nib.load(background)
    bg_data = np.asarray(bg_img.dataobj)  # Memory-mapped, not copied

    overlay_img = nib.load(overlay)
    overlay_data = np.asarray(overlay_img.dataobj)  # Memory-mapped, not copied

    # Normalize background
    bg_norm = (bg_data - bg_data.min()) / (bg_data.max() - bg_data.min() + 1e-8)

    # Compute edges of overlay
    overlay_edges = ndimage.sobel(overlay_data)
    overlay_edges = overlay_edges / (overlay_edges.max() + 1e-8)

    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, num_slices, hspace=0.3, wspace=0.05)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Axial
    z_slices = np.linspace(10, bg_data.shape[2] - 10, num_slices, dtype=int)
    for i, z in enumerate(z_slices):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(bg_norm[:, :, z].T, cmap='gray', origin='lower', aspect='auto')
        if overlay_edges.shape[2] > z:
            ax.contour(overlay_edges[:, :, z].T, levels=[0.3], colors='red',
                      linewidths=1, origin='lower')
        ax.axis('off')
        if i == 0:
            ax.set_title('Axial', fontsize=10, loc='left')

    # Coronal
    y_slices = np.linspace(10, bg_data.shape[1] - 10, num_slices, dtype=int)
    for i, y in enumerate(y_slices):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(bg_norm[:, y, :].T, cmap='gray', origin='lower', aspect='auto')
        if overlay_edges.shape[1] > y:
            ax.contour(overlay_edges[:, y, :].T, levels=[0.3], colors='red',
                      linewidths=1, origin='lower')
        ax.axis('off')
        if i == 0:
            ax.set_title('Coronal', fontsize=10, loc='left')

    # Sagittal
    x_slices = np.linspace(10, bg_data.shape[0] - 10, num_slices, dtype=int)
    for i, x in enumerate(x_slices):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(bg_norm[x, :, :].T, cmap='gray', origin='lower', aspect='auto')
        if overlay_edges.shape[0] > x:
            ax.contour(overlay_edges[x, :, :].T, levels=[0.3], colors='red',
                      linewidths=1, origin='lower')
        ax.axis('off')
        if i == 0:
            ax.set_title('Sagittal', fontsize=10, loc='left')

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"  Saved: {output_file}")
    return output_file


def qc_func_to_t1w_registration(
    func_mean: Path,
    t1w_brain: Path,
    func_to_t1w_transform: Path,
    output_dir: Path
) -> dict:
    """
    Generate QC visualizations for fMRI → T1w registration.

    Creates:
    1. Overlay of warped functional on T1w
    2. Edge overlay for boundary checking

    Parameters
    ----------
    func_mean : Path
        Functional mean image (in native functional space)
    t1w_brain : Path
        T1w brain reference
    func_to_t1w_transform : Path
        ANTs composite transform (func → T1w)
    output_dir : Path
        Output directory for QC images

    Returns
    -------
    dict
        Paths to generated QC images
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("QC: Functional → T1w Registration")
    logger.info("=" * 70)

    # Apply transform to warp functional to T1w space
    func_warped = output_dir / 'func_mean_in_t1w.nii.gz'

    if not func_warped.exists():
        logger.info("Warping functional mean to T1w space...")
        cmd = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(func_mean),
            '-r', str(t1w_brain),
            '-t', str(func_to_t1w_transform),
            '-o', str(func_warped)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  Warped: {func_warped}")
    else:
        logger.info(f"Using cached warped image: {func_warped}")

    # Create overlay mosaic
    overlay_file = output_dir / 'func_to_t1w_overlay.png'
    create_overlay_mosaic(
        background=t1w_brain,
        overlay=func_warped,
        output_file=overlay_file,
        title="Functional → T1w Registration QC (Functional overlay on T1w)",
        overlay_alpha=0.4,
        cmap_overlay='hot'
    )

    # Create edge overlay
    edge_file = output_dir / 'func_to_t1w_edges.png'
    create_edge_overlay(
        background=t1w_brain,
        overlay=func_warped,
        output_file=edge_file,
        title="Functional → T1w Edge Alignment (Red = functional edges)"
    )

    logger.info("")

    return {
        'func_warped_to_t1w': func_warped,
        'overlay_mosaic': overlay_file,
        'edge_overlay': edge_file
    }


def qc_func_to_mni_registration(
    func_mean: Path,
    mni_template: Path,
    func_to_mni_transform: Path,
    output_dir: Path
) -> dict:
    """
    Generate QC visualizations for fMRI → MNI registration.

    Parameters
    ----------
    func_mean : Path
        Functional mean image
    mni_template : Path
        MNI152 template
    func_to_mni_transform : Path
        ANTs composite transform (func → MNI)
    output_dir : Path
        Output directory for QC images

    Returns
    -------
    dict
        Paths to generated QC images
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("QC: Functional → MNI Registration")
    logger.info("=" * 70)

    # Apply transform to warp functional to MNI space
    func_warped = output_dir / 'func_mean_in_mni.nii.gz'

    if not func_warped.exists():
        logger.info("Warping functional mean to MNI space...")
        cmd = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(func_mean),
            '-r', str(mni_template),
            '-t', str(func_to_mni_transform),
            '-o', str(func_warped)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  Warped: {func_warped}")
    else:
        logger.info(f"Using cached warped image: {func_warped}")

    # Create overlay mosaic
    overlay_file = output_dir / 'func_to_mni_overlay.png'
    create_overlay_mosaic(
        background=mni_template,
        overlay=func_warped,
        output_file=overlay_file,
        title="Functional → MNI Registration QC (Functional overlay on MNI)",
        overlay_alpha=0.4,
        cmap_overlay='hot'
    )

    # Create edge overlay
    edge_file = output_dir / 'func_to_mni_edges.png'
    create_edge_overlay(
        background=mni_template,
        overlay=func_warped,
        output_file=edge_file,
        title="Functional → MNI Edge Alignment (Red = functional edges)"
    )

    logger.info("")

    return {
        'func_warped_to_mni': func_warped,
        'overlay_mosaic': overlay_file,
        'edge_overlay': edge_file
    }


def qc_tissue_masks_in_func(
    func_mean: Path,
    csf_mask_func: Path,
    wm_mask_func: Path,
    output_dir: Path
) -> dict:
    """
    Generate QC visualizations for tissue masks in functional space.

    Shows CSF and WM masks overlaid on functional reference to verify
    proper alignment for ACompCor.

    Parameters
    ----------
    func_mean : Path
        Functional mean reference
    csf_mask_func : Path
        CSF mask in functional space
    wm_mask_func : Path
        WM mask in functional space
    output_dir : Path
        Output directory for QC images

    Returns
    -------
    dict
        Paths to generated QC images
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("QC: Tissue Masks in Functional Space")
    logger.info("=" * 70)

    # CSF mask overlay
    csf_overlay_file = output_dir / 'csf_mask_overlay.png'
    create_overlay_mosaic(
        background=func_mean,
        overlay=csf_mask_func,
        output_file=csf_overlay_file,
        title="CSF Mask in Functional Space (for ACompCor)",
        overlay_alpha=0.5,
        cmap_overlay='Blues'
    )

    # WM mask overlay
    wm_overlay_file = output_dir / 'wm_mask_overlay.png'
    create_overlay_mosaic(
        background=func_mean,
        overlay=wm_mask_func,
        output_file=wm_overlay_file,
        title="WM Mask in Functional Space (for ACompCor)",
        overlay_alpha=0.5,
        cmap_overlay='Reds'
    )

    # Combined masks overlay
    combined_file = output_dir / 'tissue_masks_combined.png'
    _create_combined_tissue_overlay(
        func_mean=func_mean,
        csf_mask=csf_mask_func,
        wm_mask=wm_mask_func,
        output_file=combined_file
    )

    logger.info("")

    return {
        'csf_overlay': csf_overlay_file,
        'wm_overlay': wm_overlay_file,
        'combined_overlay': combined_file
    }


def _create_combined_tissue_overlay(
    func_mean: Path,
    csf_mask: Path,
    wm_mask: Path,
    output_file: Path,
    num_slices: int = 9
) -> Path:
    """Create overlay with both CSF (blue) and WM (red) masks."""

    logger.info("Creating combined tissue mask overlay...")

    # Load images (use dataobj for memory efficiency)
    func_img = nib.load(func_mean)
    func_data = np.asarray(func_img.dataobj)  # Memory-mapped

    csf_img = nib.load(csf_mask)
    csf_data = np.asarray(csf_img.dataobj)  # Memory-mapped

    wm_img = nib.load(wm_mask)
    wm_data = np.asarray(wm_img.dataobj)  # Memory-mapped

    # Normalize functional
    func_norm = (func_data - func_data.min()) / (func_data.max() - func_data.min() + 1e-8)

    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, num_slices, hspace=0.3, wspace=0.05)

    fig.suptitle("Combined Tissue Masks (Blue=CSF, Red=WM)", fontsize=16, fontweight='bold')

    # Axial
    z_slices = np.linspace(10, func_data.shape[2] - 10, num_slices, dtype=int)
    for i, z in enumerate(z_slices):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(func_norm[:, :, z].T, cmap='gray', origin='lower', aspect='auto')
        # CSF in blue
        ax.imshow(csf_data[:, :, z].T, cmap='Blues', alpha=0.3, origin='lower', aspect='auto')
        # WM in red
        ax.imshow(wm_data[:, :, z].T, cmap='Reds', alpha=0.3, origin='lower', aspect='auto')
        ax.axis('off')
        if i == 0:
            ax.set_title('Axial', fontsize=10, loc='left')

    # Coronal
    y_slices = np.linspace(10, func_data.shape[1] - 10, num_slices, dtype=int)
    for i, y in enumerate(y_slices):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(func_norm[:, y, :].T, cmap='gray', origin='lower', aspect='auto')
        ax.imshow(csf_data[:, y, :].T, cmap='Blues', alpha=0.3, origin='lower', aspect='auto')
        ax.imshow(wm_data[:, y, :].T, cmap='Reds', alpha=0.3, origin='lower', aspect='auto')
        ax.axis('off')
        if i == 0:
            ax.set_title('Coronal', fontsize=10, loc='left')

    # Sagittal
    x_slices = np.linspace(10, func_data.shape[0] - 10, num_slices, dtype=int)
    for i, x in enumerate(x_slices):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(func_norm[x, :, :].T, cmap='gray', origin='lower', aspect='auto')
        ax.imshow(csf_data[x, :, :].T, cmap='Blues', alpha=0.3, origin='lower', aspect='auto')
        ax.imshow(wm_data[x, :, :].T, cmap='Reds', alpha=0.3, origin='lower', aspect='auto')
        ax.axis('off')
        if i == 0:
            ax.set_title('Sagittal', fontsize=10, loc='left')

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"  Saved: {output_file}")
    return output_file


def generate_registration_qc_report(
    func_mean: Path,
    t1w_brain: Path,
    mni_template: Path,
    func_to_t1w_transform: Path,
    func_to_mni_transform: Path,
    csf_mask_func: Optional[Path] = None,
    wm_mask_func: Optional[Path] = None,
    output_dir: Path = None
) -> dict:
    """
    Generate complete registration QC report.

    Creates all QC visualizations for functional registration pipeline:
    - fMRI → T1w alignment
    - fMRI → MNI alignment
    - Tissue masks in functional space

    Parameters
    ----------
    func_mean : Path
        Functional mean reference
    t1w_brain : Path
        T1w brain image
    mni_template : Path
        MNI152 template
    func_to_t1w_transform : Path
        ANTs transform (func → T1w)
    func_to_mni_transform : Path
        ANTs transform (func → MNI)
    csf_mask_func : Path, optional
        CSF mask in functional space
    wm_mask_func : Path, optional
        WM mask in functional space
    output_dir : Path
        Output directory for all QC images

    Returns
    -------
    dict
        Dictionary of all generated QC files
    """
    if output_dir is None:
        output_dir = Path.cwd() / 'registration_qc'

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Generating Complete Registration QC Report")
    logger.info("=" * 70)
    logger.info("")

    results = {}

    # 1. fMRI → T1w QC
    t1w_qc = qc_func_to_t1w_registration(
        func_mean=func_mean,
        t1w_brain=t1w_brain,
        func_to_t1w_transform=func_to_t1w_transform,
        output_dir=output_dir / 'func_to_t1w'
    )
    results['func_to_t1w'] = t1w_qc

    # 2. fMRI → MNI QC
    mni_qc = qc_func_to_mni_registration(
        func_mean=func_mean,
        mni_template=mni_template,
        func_to_mni_transform=func_to_mni_transform,
        output_dir=output_dir / 'func_to_mni'
    )
    results['func_to_mni'] = mni_qc

    # 3. Tissue masks QC (if available)
    if csf_mask_func and wm_mask_func:
        tissue_qc = qc_tissue_masks_in_func(
            func_mean=func_mean,
            csf_mask_func=csf_mask_func,
            wm_mask_func=wm_mask_func,
            output_dir=output_dir / 'tissue_masks'
        )
        results['tissue_masks'] = tissue_qc

    logger.info("=" * 70)
    logger.info("Registration QC Report Complete")
    logger.info("=" * 70)
    logger.info(f"QC images saved to: {output_dir}")
    logger.info("")

    return results
