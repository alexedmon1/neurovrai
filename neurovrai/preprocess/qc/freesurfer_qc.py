#!/usr/bin/env python3
"""
FreeSurfer Alignment Quality Control

Validates the alignment between FreeSurfer outputs and preprocessing T1w,
ensuring accurate registration for downstream analyses.

Key checks:
- FreeSurfer orig.mgz alignment with preprocessed T1w brain
- Cross-correlation and mutual information metrics
- Visual overlay generation for manual QC
- Detect if FreeSurfer was run on a different T1w scan
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class FreeSurferQCError(Exception):
    """Raised when FreeSurfer QC fails"""
    pass


def compute_image_similarity(
    img1_data: np.ndarray,
    img2_data: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute similarity metrics between two images

    Args:
        img1_data: First image data
        img2_data: Second image data
        mask: Optional mask to restrict computation

    Returns:
        Dictionary with similarity metrics
    """
    if mask is not None:
        v1 = img1_data[mask > 0].flatten()
        v2 = img2_data[mask > 0].flatten()
    else:
        v1 = img1_data.flatten()
        v2 = img2_data.flatten()

    # Remove zeros and NaNs
    valid = (v1 > 0) & (v2 > 0) & np.isfinite(v1) & np.isfinite(v2)
    v1 = v1[valid]
    v2 = v2[valid]

    if len(v1) < 100:
        logger.warning("Too few valid voxels for similarity computation")
        return {
            'correlation': 0.0,
            'n_voxels': len(v1),
        }

    # Pearson correlation
    correlation = np.corrcoef(v1, v2)[0, 1]

    # Normalized mutual information (simplified)
    # Using histogram-based estimation
    n_bins = 64
    hist_2d, _, _ = np.histogram2d(v1, v2, bins=n_bins)

    # Joint probability
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # Marginal entropies
    px_pos = px[px > 0]
    py_pos = py[py > 0]
    hx = -np.sum(px_pos * np.log2(px_pos))
    hy = -np.sum(py_pos * np.log2(py_pos))

    # Joint entropy
    pxy_pos = pxy[pxy > 0]
    hxy = -np.sum(pxy_pos * np.log2(pxy_pos))

    # Normalized mutual information
    nmi = (hx + hy) / hxy if hxy > 0 else 0.0

    return {
        'correlation': float(correlation),
        'nmi': float(nmi),
        'n_voxels': len(v1),
    }


def validate_fs_t1w_alignment(
    fs_orig_nii: Path,
    t1w_brain: Path,
    transform_mat: Optional[Path] = None,
    qc_dir: Optional[Path] = None
) -> Dict:
    """
    Validate alignment between FreeSurfer orig and preprocessed T1w

    Args:
        fs_orig_nii: FreeSurfer orig.mgz converted to NIfTI (or registered to T1w)
        t1w_brain: Preprocessed T1w brain
        transform_mat: Optional FLIRT transform matrix (FS to T1w)
        qc_dir: Optional output directory for QC outputs

    Returns:
        Dictionary with QC results
    """
    logger.info("Validating FreeSurfer to T1w alignment...")

    fs_orig_nii = Path(fs_orig_nii)
    t1w_brain = Path(t1w_brain)

    if not fs_orig_nii.exists():
        raise FreeSurferQCError(f"FreeSurfer image not found: {fs_orig_nii}")
    if not t1w_brain.exists():
        raise FreeSurferQCError(f"T1w brain not found: {t1w_brain}")

    # Load images
    fs_img = nib.load(fs_orig_nii)
    t1w_img = nib.load(t1w_brain)

    fs_data = fs_img.get_fdata()
    t1w_data = t1w_img.get_fdata()

    # Check dimensions match
    if fs_data.shape != t1w_data.shape:
        logger.warning(
            f"Image dimensions differ: FS {fs_data.shape} vs T1w {t1w_data.shape}. "
            f"Registration may be needed."
        )
        # If transform exists, apply it
        if transform_mat is not None:
            from neurovrai.preprocess.utils.freesurfer_transforms import apply_transform_to_volume
            import tempfile

            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
                registered_path = Path(tmp.name)

            apply_transform_to_volume(
                input_file=fs_orig_nii,
                reference_file=t1w_brain,
                transform_mat=transform_mat,
                output_file=registered_path,
                interpolation='spline'
            )

            fs_img = nib.load(registered_path)
            fs_data = fs_img.get_fdata()

    # Create brain mask from T1w
    mask = t1w_data > (np.percentile(t1w_data[t1w_data > 0], 5))

    # Compute similarity metrics
    metrics = compute_image_similarity(fs_data, t1w_data, mask)

    # Determine quality flags
    flags = []

    if metrics['correlation'] < 0.90:
        flags.append('LOW_CORRELATION')
        logger.warning(f"Low correlation ({metrics['correlation']:.3f}) - images may not match")

    if metrics['correlation'] < 0.70:
        flags.append('VERY_LOW_CORRELATION')
        logger.error(
            f"Very low correlation ({metrics['correlation']:.3f}) - "
            f"FreeSurfer may have been run on different T1w!"
        )

    if metrics.get('nmi', 0) < 1.0:
        flags.append('LOW_NMI')

    # Generate QC outputs
    if qc_dir is not None:
        qc_dir = Path(qc_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = qc_dir / 'fs_t1w_alignment_metrics.json'
        qc_results = {
            'fs_image': str(fs_orig_nii),
            't1w_image': str(t1w_brain),
            'transform_applied': str(transform_mat) if transform_mat else None,
            'metrics': metrics,
            'flags': flags,
            'pass': len([f for f in flags if 'VERY_LOW' in f]) == 0,
        }
        with open(metrics_file, 'w') as f:
            json.dump(qc_results, f, indent=2)

        logger.info(f"QC metrics saved: {metrics_file}")

        # Generate overlay image if matplotlib available
        try:
            _generate_alignment_overlay(
                fs_data=fs_data,
                t1w_data=t1w_data,
                output_file=qc_dir / 'fs_t1w_alignment_overlay.png',
                title=f"FreeSurfer-T1w Alignment (r={metrics['correlation']:.3f})"
            )
            logger.info("Overlay image generated")
        except Exception as e:
            logger.warning(f"Could not generate overlay: {e}")

    results = {
        'metrics': metrics,
        'flags': flags,
        'pass': len([f for f in flags if 'VERY_LOW' in f]) == 0,
    }

    if results['pass']:
        logger.info(f"FreeSurfer-T1w alignment: PASS (r={metrics['correlation']:.3f})")
    else:
        logger.warning(f"FreeSurfer-T1w alignment: FAIL (r={metrics['correlation']:.3f})")

    return results


def _generate_alignment_overlay(
    fs_data: np.ndarray,
    t1w_data: np.ndarray,
    output_file: Path,
    title: str = "FreeSurfer-T1w Alignment"
):
    """Generate overlay visualization for alignment QC"""
    import matplotlib.pyplot as plt

    # Get center slices
    mid_x = fs_data.shape[0] // 2
    mid_y = fs_data.shape[1] // 2
    mid_z = fs_data.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14)

    # FreeSurfer slices
    axes[0, 0].imshow(fs_data[mid_x, :, :].T, cmap='gray', origin='lower')
    axes[0, 0].set_title('FreeSurfer - Sagittal')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fs_data[:, mid_y, :].T, cmap='gray', origin='lower')
    axes[0, 1].set_title('FreeSurfer - Coronal')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(fs_data[:, :, mid_z].T, cmap='gray', origin='lower')
    axes[0, 2].set_title('FreeSurfer - Axial')
    axes[0, 2].axis('off')

    # T1w slices with FreeSurfer edge overlay
    from scipy import ndimage

    # Compute edges from FS
    fs_edges_sag = ndimage.sobel(fs_data[mid_x, :, :])
    fs_edges_cor = ndimage.sobel(fs_data[:, mid_y, :])
    fs_edges_ax = ndimage.sobel(fs_data[:, :, mid_z])

    # Show T1w with edges
    axes[1, 0].imshow(t1w_data[mid_x, :, :].T, cmap='gray', origin='lower')
    axes[1, 0].contour(fs_edges_sag.T, colors='red', alpha=0.5, linewidths=0.5)
    axes[1, 0].set_title('T1w + FS edges - Sagittal')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(t1w_data[:, mid_y, :].T, cmap='gray', origin='lower')
    axes[1, 1].contour(fs_edges_cor.T, colors='red', alpha=0.5, linewidths=0.5)
    axes[1, 1].set_title('T1w + FS edges - Coronal')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(t1w_data[:, :, mid_z].T, cmap='gray', origin='lower')
    axes[1, 2].contour(fs_edges_ax.T, colors='red', alpha=0.5, linewidths=0.5)
    axes[1, 2].set_title('T1w + FS edges - Axial')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def validate_freesurfer_outputs(
    fs_subject_dir: Path,
    qc_dir: Optional[Path] = None
) -> Dict:
    """
    Validate FreeSurfer recon-all outputs are complete

    Args:
        fs_subject_dir: Path to FreeSurfer subject directory
        qc_dir: Optional output directory for QC outputs

    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating FreeSurfer outputs: {fs_subject_dir}")

    fs_subject_dir = Path(fs_subject_dir)
    if not fs_subject_dir.exists():
        raise FreeSurferQCError(f"FreeSurfer directory not found: {fs_subject_dir}")

    # Required files for structural connectivity
    required_files = {
        'mri/orig.mgz': 'Original T1w',
        'mri/brain.mgz': 'Brain extracted',
        'mri/aparc+aseg.mgz': 'Parcellation (Desikan-Killiany)',
        'mri/aparc.a2009s+aseg.mgz': 'Parcellation (Destrieux)',
        'surf/lh.white': 'Left white surface',
        'surf/rh.white': 'Right white surface',
        'surf/lh.pial': 'Left pial surface',
        'surf/rh.pial': 'Right pial surface',
    }

    results = {
        'subject_dir': str(fs_subject_dir),
        'files': {},
        'missing': [],
        'complete': True,
    }

    for filepath, description in required_files.items():
        full_path = fs_subject_dir / filepath
        exists = full_path.exists()
        results['files'][filepath] = {
            'description': description,
            'exists': exists,
            'path': str(full_path) if exists else None,
        }
        if not exists:
            results['missing'].append(filepath)
            results['complete'] = False
            logger.warning(f"Missing: {filepath} ({description})")
        else:
            logger.debug(f"Found: {filepath}")

    # Check for recon-all done marker
    recon_done = (fs_subject_dir / 'scripts' / 'recon-all.done').exists()
    results['recon_all_complete'] = recon_done

    if not recon_done:
        results['complete'] = False
        logger.warning("recon-all.done marker not found - processing may be incomplete")

    # Save results
    if qc_dir is not None:
        qc_dir = Path(qc_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)

        results_file = qc_dir / 'freesurfer_validation.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Validation results saved: {results_file}")

    if results['complete']:
        logger.info("FreeSurfer outputs: COMPLETE")
    else:
        logger.warning(f"FreeSurfer outputs: INCOMPLETE (missing {len(results['missing'])} files)")

    return results


class FreeSurferAlignmentQC:
    """
    Complete FreeSurfer alignment quality control

    Usage:
        qc = FreeSurferAlignmentQC(
            subject='IRC805-0580101',
            fs_subject_dir='/study/freesurfer/IRC805-0580101',
            t1w_brain='/study/derivatives/IRC805-0580101/anat/brain.nii.gz',
            qc_dir='/study/qc/IRC805-0580101/freesurfer'
        )
        results = qc.run_qc()
    """

    def __init__(
        self,
        subject: str,
        fs_subject_dir: Path,
        t1w_brain: Path,
        qc_dir: Path,
        transform_mat: Optional[Path] = None
    ):
        self.subject = subject
        self.fs_subject_dir = Path(fs_subject_dir)
        self.t1w_brain = Path(t1w_brain)
        self.qc_dir = Path(qc_dir)
        self.transform_mat = Path(transform_mat) if transform_mat else None

        self.qc_dir.mkdir(parents=True, exist_ok=True)

    def run_qc(self) -> Dict:
        """Run complete FreeSurfer QC"""
        logger.info(f"Running FreeSurfer QC for {self.subject}")

        results = {
            'subject': self.subject,
            'qc_dir': str(self.qc_dir),
        }

        # Step 1: Validate FreeSurfer outputs
        try:
            validation = validate_freesurfer_outputs(
                fs_subject_dir=self.fs_subject_dir,
                qc_dir=self.qc_dir
            )
            results['validation'] = validation
        except Exception as e:
            logger.error(f"FreeSurfer validation failed: {e}")
            results['validation'] = {'error': str(e), 'complete': False}

        # Step 2: Check alignment if FreeSurfer outputs exist
        if results.get('validation', {}).get('complete', False):
            try:
                # Convert orig.mgz to nii.gz for comparison
                orig_mgz = self.fs_subject_dir / 'mri' / 'orig.mgz'
                orig_nii = self.qc_dir / 'orig.nii.gz'

                if not orig_nii.exists():
                    import subprocess
                    subprocess.run([
                        'mri_convert', str(orig_mgz), str(orig_nii)
                    ], check=True, capture_output=True)

                alignment = validate_fs_t1w_alignment(
                    fs_orig_nii=orig_nii,
                    t1w_brain=self.t1w_brain,
                    transform_mat=self.transform_mat,
                    qc_dir=self.qc_dir
                )
                results['alignment'] = alignment

            except Exception as e:
                logger.error(f"Alignment check failed: {e}")
                results['alignment'] = {'error': str(e), 'pass': False}
        else:
            results['alignment'] = {'skipped': True, 'reason': 'FreeSurfer incomplete'}

        # Overall pass/fail
        results['overall_pass'] = (
            results.get('validation', {}).get('complete', False) and
            results.get('alignment', {}).get('pass', False)
        )

        # Save summary
        summary_file = self.qc_dir / 'freesurfer_qc_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"QC summary saved: {summary_file}")

        return results
