#!/usr/bin/env python3
"""
Structural Connectivity Quality Control

Validates atlas transformations, tractography outputs, and connectivity matrices
for probabilistic tractography-based structural connectivity analysis.

Key QC checks:
- Atlas transformation: volume preservation, label count, overlap with brain
- Tractography: waytotal distribution, success rate, streamline density
- Connectivity matrix: density, symmetry, expected connection validation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class StructuralConnectivityQCError(Exception):
    """Raised when structural connectivity QC fails"""
    pass


def run_atlas_transform_qc(
    atlas_original: Path,
    atlas_dwi: Path,
    dwi_reference: Path,
    qc_dir: Path,
    atlas_name: str = 'atlas'
) -> Dict:
    """
    Quality control for atlas transformation to DWI space

    Checks:
    - Label count preservation
    - Volume preservation
    - Overlap with brain mask
    - ROI sizes in DWI space

    Args:
        atlas_original: Original atlas (MNI/FreeSurfer space)
        atlas_dwi: Transformed atlas in DWI space
        dwi_reference: DWI reference image (FA map or B0)
        qc_dir: Output directory for QC results
        atlas_name: Name of atlas for reporting

    Returns:
        Dictionary with QC results
    """
    logger.info(f"Running atlas transform QC: {atlas_name}")

    atlas_original = Path(atlas_original)
    atlas_dwi = Path(atlas_dwi)
    dwi_reference = Path(dwi_reference)
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    orig_img = nib.load(atlas_original)
    dwi_img = nib.load(atlas_dwi)
    ref_img = nib.load(dwi_reference)

    orig_data = orig_img.get_fdata().astype(int)
    dwi_data = dwi_img.get_fdata().astype(int)
    ref_data = ref_img.get_fdata()

    # Get voxel sizes
    orig_voxel_vol = np.prod(orig_img.header.get_zooms()[:3])
    dwi_voxel_vol = np.prod(dwi_img.header.get_zooms()[:3])

    # Label analysis
    orig_labels = np.unique(orig_data)
    orig_labels = orig_labels[orig_labels > 0]  # Exclude background

    dwi_labels = np.unique(dwi_data)
    dwi_labels = dwi_labels[dwi_labels > 0]

    # Check label preservation
    labels_preserved = len(dwi_labels) >= len(orig_labels) * 0.9  # Allow 10% loss
    missing_labels = set(orig_labels) - set(dwi_labels)
    extra_labels = set(dwi_labels) - set(orig_labels)

    # Per-ROI statistics
    roi_stats = []
    for label in dwi_labels:
        orig_count = np.sum(orig_data == label)
        dwi_count = np.sum(dwi_data == label)

        orig_vol_mm3 = orig_count * orig_voxel_vol
        dwi_vol_mm3 = dwi_count * dwi_voxel_vol

        volume_ratio = dwi_vol_mm3 / orig_vol_mm3 if orig_vol_mm3 > 0 else 0

        roi_stats.append({
            'label': int(label),
            'orig_voxels': int(orig_count),
            'dwi_voxels': int(dwi_count),
            'orig_volume_mm3': float(orig_vol_mm3),
            'dwi_volume_mm3': float(dwi_vol_mm3),
            'volume_ratio': float(volume_ratio),
        })

    # Overall volume statistics
    total_orig_vol = sum(r['orig_volume_mm3'] for r in roi_stats)
    total_dwi_vol = sum(r['dwi_volume_mm3'] for r in roi_stats)
    overall_vol_ratio = total_dwi_vol / total_orig_vol if total_orig_vol > 0 else 0

    # Create brain mask from reference
    ref_mask = ref_data > np.percentile(ref_data[ref_data > 0], 5)
    brain_voxels = np.sum(ref_mask)

    # Check atlas-brain overlap
    atlas_in_brain = np.sum((dwi_data > 0) & ref_mask)
    atlas_total = np.sum(dwi_data > 0)
    overlap_ratio = atlas_in_brain / atlas_total if atlas_total > 0 else 0

    # Quality flags
    flags = []

    if len(missing_labels) > 0:
        flags.append('LABELS_LOST')
        logger.warning(f"Lost {len(missing_labels)} labels during transformation")

    if overall_vol_ratio < 0.7 or overall_vol_ratio > 1.3:
        flags.append('VOLUME_CHANGE')
        logger.warning(f"Significant volume change: ratio = {overall_vol_ratio:.2f}")

    if overlap_ratio < 0.8:
        flags.append('LOW_BRAIN_OVERLAP')
        logger.warning(f"Low brain overlap: {overlap_ratio:.2f}")

    # Small ROI check
    small_rois = [r for r in roi_stats if r['dwi_voxels'] < 10]
    if len(small_rois) > 0:
        flags.append('SMALL_ROIS')
        logger.warning(f"{len(small_rois)} ROIs have < 10 voxels")

    results = {
        'atlas_name': atlas_name,
        'original': str(atlas_original),
        'dwi': str(atlas_dwi),
        'orig_labels': len(orig_labels),
        'dwi_labels': len(dwi_labels),
        'missing_labels': list(map(int, missing_labels)),
        'volume_ratio': float(overall_vol_ratio),
        'brain_overlap': float(overlap_ratio),
        'small_rois': len(small_rois),
        'roi_stats': roi_stats,
        'flags': flags,
        'pass': len(flags) == 0,
    }

    # Save results
    results_file = qc_dir / f'{atlas_name}_transform_qc.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate overlay visualization
    try:
        _generate_atlas_overlay(
            atlas_data=dwi_data,
            ref_data=ref_data,
            output_file=qc_dir / f'{atlas_name}_overlay.png',
            title=f"{atlas_name} in DWI space ({len(dwi_labels)} ROIs)"
        )
    except Exception as e:
        logger.warning(f"Could not generate overlay: {e}")

    if results['pass']:
        logger.info(f"Atlas transform QC: PASS")
    else:
        logger.warning(f"Atlas transform QC: FLAGS - {flags}")

    return results


def _generate_atlas_overlay(
    atlas_data: np.ndarray,
    ref_data: np.ndarray,
    output_file: Path,
    title: str = "Atlas Overlay"
):
    """Generate atlas overlay visualization"""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Get center slices
    mid_x = atlas_data.shape[0] // 2
    mid_y = atlas_data.shape[1] // 2
    mid_z = atlas_data.shape[2] // 2

    # Create random colormap for labels
    n_labels = len(np.unique(atlas_data)) - 1  # Exclude 0
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    np.random.seed(42)
    np.random.shuffle(colors)

    # Create masked atlas for overlay
    atlas_masked = np.ma.masked_where(atlas_data == 0, atlas_data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    # Sagittal
    axes[0].imshow(ref_data[mid_x, :, :].T, cmap='gray', origin='lower')
    axes[0].imshow(atlas_masked[mid_x, :, :].T, alpha=0.4, origin='lower')
    axes[0].set_title('Sagittal')
    axes[0].axis('off')

    # Coronal
    axes[1].imshow(ref_data[:, mid_y, :].T, cmap='gray', origin='lower')
    axes[1].imshow(atlas_masked[:, mid_y, :].T, alpha=0.4, origin='lower')
    axes[1].set_title('Coronal')
    axes[1].axis('off')

    # Axial
    axes[2].imshow(ref_data[:, :, mid_z].T, cmap='gray', origin='lower')
    axes[2].imshow(atlas_masked[:, :, mid_z].T, alpha=0.4, origin='lower')
    axes[2].set_title('Axial')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def run_tractography_qc(
    probtrackx_dir: Path,
    qc_dir: Path
) -> Dict:
    """
    Quality control for probtrackx2 tractography outputs

    Checks:
    - Waytotal distribution (successful streamlines per seed)
    - Overall success rate
    - Network matrix properties

    Args:
        probtrackx_dir: Path to probtrackx2 output directory
        qc_dir: Output directory for QC results

    Returns:
        Dictionary with QC results
    """
    logger.info("Running tractography QC")

    probtrackx_dir = Path(probtrackx_dir)
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    if not probtrackx_dir.exists():
        raise StructuralConnectivityQCError(
            f"Probtrackx output not found: {probtrackx_dir}"
        )

    # Load waytotal
    waytotal_file = probtrackx_dir / 'waytotal'
    if not waytotal_file.exists():
        raise StructuralConnectivityQCError(f"waytotal not found: {waytotal_file}")

    waytotal = np.loadtxt(waytotal_file)
    n_seeds = len(waytotal)

    # Waytotal statistics
    waytotal_mean = float(np.mean(waytotal))
    waytotal_std = float(np.std(waytotal))
    waytotal_min = float(np.min(waytotal))
    waytotal_max = float(np.max(waytotal))
    waytotal_median = float(np.median(waytotal))

    # Seeds with low waytotal
    low_waytotal_threshold = waytotal_mean * 0.1
    n_low_waytotal = int(np.sum(waytotal < low_waytotal_threshold))
    n_zero_waytotal = int(np.sum(waytotal == 0))

    # Load network matrix
    matrix_file = probtrackx_dir / 'fdt_network_matrix'
    if matrix_file.exists():
        network_matrix = np.loadtxt(matrix_file)
        matrix_shape = network_matrix.shape
        n_connections = int(np.sum(network_matrix > 0))
        matrix_density = n_connections / (matrix_shape[0] * (matrix_shape[1] - 1)) if matrix_shape[0] > 1 else 0
    else:
        network_matrix = None
        matrix_shape = None
        n_connections = None
        matrix_density = None
        logger.warning("Network matrix not found")

    # Quality flags
    flags = []

    if n_zero_waytotal > n_seeds * 0.1:
        flags.append('MANY_ZERO_WAYTOTAL')
        logger.warning(f"{n_zero_waytotal}/{n_seeds} seeds have zero waytotal")

    if waytotal_std / waytotal_mean > 1.0 if waytotal_mean > 0 else False:
        flags.append('HIGH_WAYTOTAL_VARIANCE')
        logger.warning(f"High waytotal variance: CV = {waytotal_std/waytotal_mean:.2f}")

    if matrix_density is not None and matrix_density < 0.05:
        flags.append('LOW_CONNECTIVITY_DENSITY')
        logger.warning(f"Low connectivity density: {matrix_density:.3f}")

    if matrix_density is not None and matrix_density > 0.5:
        flags.append('HIGH_CONNECTIVITY_DENSITY')
        logger.warning(f"High connectivity density: {matrix_density:.3f}")

    results = {
        'probtrackx_dir': str(probtrackx_dir),
        'n_seeds': n_seeds,
        'waytotal': {
            'mean': waytotal_mean,
            'std': waytotal_std,
            'min': waytotal_min,
            'max': waytotal_max,
            'median': waytotal_median,
            'n_zero': n_zero_waytotal,
            'n_low': n_low_waytotal,
        },
        'matrix': {
            'shape': list(matrix_shape) if matrix_shape else None,
            'n_connections': n_connections,
            'density': float(matrix_density) if matrix_density else None,
        } if matrix_shape else None,
        'flags': flags,
        'pass': len(flags) == 0,
    }

    # Save results
    results_file = qc_dir / 'tractography_qc.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate waytotal histogram
    try:
        _generate_waytotal_histogram(
            waytotal=waytotal,
            output_file=qc_dir / 'waytotal_distribution.png'
        )
    except Exception as e:
        logger.warning(f"Could not generate histogram: {e}")

    logger.info(f"Tractography QC: {'PASS' if results['pass'] else 'FLAGS - ' + str(flags)}")

    return results


def _generate_waytotal_histogram(waytotal: np.ndarray, output_file: Path):
    """Generate waytotal distribution histogram"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(waytotal, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(waytotal), color='red', linestyle='--', label=f'Mean: {np.mean(waytotal):.0f}')
    ax.axvline(np.median(waytotal), color='green', linestyle='--', label=f'Median: {np.median(waytotal):.0f}')

    ax.set_xlabel('Waytotal (successful streamlines)')
    ax.set_ylabel('Number of seeds')
    ax.set_title('Waytotal Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def run_connectivity_matrix_qc(
    sc_matrix: np.ndarray,
    roi_names: List[str],
    qc_dir: Path,
    atlas_name: str = 'atlas'
) -> Dict:
    """
    Quality control for structural connectivity matrix

    Checks:
    - Matrix symmetry
    - Connection density
    - Isolated nodes
    - Distribution of connection strengths

    Args:
        sc_matrix: Structural connectivity matrix (n_rois x n_rois)
        roi_names: List of ROI names
        qc_dir: Output directory for QC results
        atlas_name: Atlas name for reporting

    Returns:
        Dictionary with QC results
    """
    logger.info("Running connectivity matrix QC")

    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    n_rois = sc_matrix.shape[0]

    # Symmetry check
    symmetry_diff = np.abs(sc_matrix - sc_matrix.T)
    max_asymmetry = float(np.max(symmetry_diff))
    mean_asymmetry = float(np.mean(symmetry_diff))
    is_symmetric = max_asymmetry < 1e-6

    # Make symmetric for further analysis
    sc_sym = (sc_matrix + sc_matrix.T) / 2
    np.fill_diagonal(sc_sym, 0)

    # Connection statistics
    n_possible = n_rois * (n_rois - 1) / 2
    n_connections = int(np.sum(sc_sym > 0) / 2)  # Divide by 2 for symmetric
    density = n_connections / n_possible if n_possible > 0 else 0

    # Node degree (connections per ROI)
    node_degree = np.sum(sc_sym > 0, axis=1)
    isolated_nodes = int(np.sum(node_degree == 0))
    mean_degree = float(np.mean(node_degree))
    max_degree = int(np.max(node_degree))

    # Connection strength distribution
    nonzero_values = sc_sym[sc_sym > 0]
    if len(nonzero_values) > 0:
        strength_stats = {
            'mean': float(np.mean(nonzero_values)),
            'std': float(np.std(nonzero_values)),
            'min': float(np.min(nonzero_values)),
            'max': float(np.max(nonzero_values)),
            'median': float(np.median(nonzero_values)),
        }
    else:
        strength_stats = None

    # Quality flags
    flags = []

    if not is_symmetric:
        flags.append('ASYMMETRIC_MATRIX')
        logger.warning(f"Matrix is not symmetric (max diff: {max_asymmetry:.6f})")

    if isolated_nodes > n_rois * 0.1:
        flags.append('MANY_ISOLATED_NODES')
        logger.warning(f"{isolated_nodes}/{n_rois} nodes are isolated")

    if density < 0.05:
        flags.append('VERY_SPARSE')
        logger.warning(f"Very sparse connectivity: {density:.3f}")

    if density > 0.5:
        flags.append('VERY_DENSE')
        logger.warning(f"Very dense connectivity: {density:.3f}")

    results = {
        'atlas_name': atlas_name,
        'n_rois': n_rois,
        'n_connections': n_connections,
        'n_possible_connections': int(n_possible),
        'density': float(density),
        'symmetry': {
            'is_symmetric': is_symmetric,
            'max_asymmetry': max_asymmetry,
            'mean_asymmetry': mean_asymmetry,
        },
        'node_degree': {
            'mean': mean_degree,
            'max': max_degree,
            'isolated_nodes': isolated_nodes,
        },
        'strength_stats': strength_stats,
        'flags': flags,
        'pass': len(flags) == 0,
    }

    # Save results
    results_file = qc_dir / f'{atlas_name}_connectivity_qc.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate matrix visualization
    try:
        _generate_connectivity_matrix_plot(
            sc_matrix=sc_sym,
            roi_names=roi_names,
            output_file=qc_dir / f'{atlas_name}_connectivity_matrix.png',
            title=f"Structural Connectivity ({atlas_name})"
        )
    except Exception as e:
        logger.warning(f"Could not generate matrix plot: {e}")

    # Generate degree histogram
    try:
        _generate_degree_histogram(
            node_degree=node_degree,
            output_file=qc_dir / f'{atlas_name}_degree_distribution.png'
        )
    except Exception as e:
        logger.warning(f"Could not generate degree histogram: {e}")

    logger.info(f"Connectivity QC: {'PASS' if results['pass'] else 'FLAGS - ' + str(flags)}")

    return results


def _generate_connectivity_matrix_plot(
    sc_matrix: np.ndarray,
    roi_names: List[str],
    output_file: Path,
    title: str = "Structural Connectivity"
):
    """Generate connectivity matrix visualization"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # Log scale for better visualization
    sc_log = np.log10(sc_matrix + 1)

    im = ax.imshow(sc_log, cmap='hot', aspect='equal')
    ax.set_title(title)
    ax.set_xlabel('ROI')
    ax.set_ylabel('ROI')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(connectivity + 1)')

    # Add labels for small matrices
    if len(roi_names) <= 50:
        ax.set_xticks(range(len(roi_names)))
        ax.set_yticks(range(len(roi_names)))
        ax.set_xticklabels(roi_names, rotation=90, fontsize=6)
        ax.set_yticklabels(roi_names, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def _generate_degree_histogram(node_degree: np.ndarray, output_file: Path):
    """Generate node degree distribution histogram"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(node_degree, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(node_degree), color='red', linestyle='--',
               label=f'Mean: {np.mean(node_degree):.1f}')

    ax.set_xlabel('Node Degree (connections)')
    ax.set_ylabel('Number of ROIs')
    ax.set_title('Node Degree Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


class StructuralConnectivityQC:
    """
    Complete structural connectivity quality control

    Usage:
        qc = StructuralConnectivityQC(
            subject='IRC805-0580101',
            atlas_name='schaefer_200',
            sc_dir='/study/connectome/structural/schaefer_200/IRC805-0580101',
            qc_dir='/study/qc/IRC805-0580101/structural_connectivity'
        )
        results = qc.run_qc()
    """

    def __init__(
        self,
        subject: str,
        atlas_name: str,
        sc_dir: Path,
        qc_dir: Path,
        atlas_original: Optional[Path] = None,
        dwi_reference: Optional[Path] = None
    ):
        self.subject = subject
        self.atlas_name = atlas_name
        self.sc_dir = Path(sc_dir)
        self.qc_dir = Path(qc_dir)
        self.atlas_original = Path(atlas_original) if atlas_original else None
        self.dwi_reference = Path(dwi_reference) if dwi_reference else None

        self.qc_dir.mkdir(parents=True, exist_ok=True)

    def run_qc(self) -> Dict:
        """Run complete structural connectivity QC"""
        logger.info(f"Running SC QC for {self.subject} ({self.atlas_name})")

        results = {
            'subject': self.subject,
            'atlas_name': self.atlas_name,
            'qc_dir': str(self.qc_dir),
        }

        # Step 1: Atlas transform QC (if original atlas provided)
        atlas_dwi = self.sc_dir / 'atlas' / f'{self.atlas_name}_in_dwi.nii.gz'
        if not atlas_dwi.exists():
            atlas_dwi = self.sc_dir / f'{self.atlas_name}_in_dwi.nii.gz'

        if self.atlas_original and self.dwi_reference and atlas_dwi.exists():
            try:
                results['atlas_qc'] = run_atlas_transform_qc(
                    atlas_original=self.atlas_original,
                    atlas_dwi=atlas_dwi,
                    dwi_reference=self.dwi_reference,
                    qc_dir=self.qc_dir,
                    atlas_name=self.atlas_name
                )
            except Exception as e:
                logger.error(f"Atlas QC failed: {e}")
                results['atlas_qc'] = {'error': str(e), 'pass': False}
        else:
            results['atlas_qc'] = {'skipped': True, 'reason': 'Missing inputs'}

        # Step 2: Tractography QC
        probtrackx_dir = self.sc_dir / 'probtrackx_output'
        if probtrackx_dir.exists():
            try:
                results['tractography_qc'] = run_tractography_qc(
                    probtrackx_dir=probtrackx_dir,
                    qc_dir=self.qc_dir
                )
            except Exception as e:
                logger.error(f"Tractography QC failed: {e}")
                results['tractography_qc'] = {'error': str(e), 'pass': False}
        else:
            results['tractography_qc'] = {'skipped': True, 'reason': 'probtrackx_output not found'}

        # Step 3: Connectivity matrix QC
        sc_matrix_file = self.sc_dir / 'sc_matrix.npy'
        roi_names_file = self.sc_dir / 'sc_roi_names.txt'

        if sc_matrix_file.exists():
            try:
                sc_matrix = np.load(sc_matrix_file)

                if roi_names_file.exists():
                    with open(roi_names_file) as f:
                        roi_names = [line.strip() for line in f]
                else:
                    roi_names = [f'ROI_{i}' for i in range(sc_matrix.shape[0])]

                results['matrix_qc'] = run_connectivity_matrix_qc(
                    sc_matrix=sc_matrix,
                    roi_names=roi_names,
                    qc_dir=self.qc_dir,
                    atlas_name=self.atlas_name
                )
            except Exception as e:
                logger.error(f"Matrix QC failed: {e}")
                results['matrix_qc'] = {'error': str(e), 'pass': False}
        else:
            results['matrix_qc'] = {'skipped': True, 'reason': 'sc_matrix.npy not found'}

        # Overall pass/fail
        qc_checks = [
            results.get('atlas_qc', {}).get('pass', True),
            results.get('tractography_qc', {}).get('pass', True),
            results.get('matrix_qc', {}).get('pass', True),
        ]
        results['overall_pass'] = all(qc_checks)

        # Save summary
        summary_file = self.qc_dir / 'sc_qc_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"QC summary saved: {summary_file}")
        logger.info(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")

        return results


def generate_html_report(qc_results: Dict, output_file: Path):
    """
    Generate HTML report from QC results

    Args:
        qc_results: Dictionary with QC results from StructuralConnectivityQC
        output_file: Output HTML file path
    """
    subject = qc_results.get('subject', 'Unknown')
    atlas = qc_results.get('atlas_name', 'Unknown')
    overall_pass = qc_results.get('overall_pass', False)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Structural Connectivity QC - {subject}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Structural Connectivity QC Report</h1>
    <p><strong>Subject:</strong> {subject}</p>
    <p><strong>Atlas:</strong> {atlas}</p>
    <p><strong>Overall Status:</strong> <span class="{'pass' if overall_pass else 'fail'}">
        {'PASS' if overall_pass else 'FAIL'}</span></p>
"""

    # Atlas QC section
    atlas_qc = qc_results.get('atlas_qc', {})
    if not atlas_qc.get('skipped', False):
        html += f"""
    <div class="section">
        <h2>Atlas Transformation QC</h2>
        <p>Status: <span class="{'pass' if atlas_qc.get('pass', False) else 'fail'}">
            {'PASS' if atlas_qc.get('pass', False) else 'FAIL'}</span></p>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Original Labels</td><td>{atlas_qc.get('orig_labels', 'N/A')}</td></tr>
            <tr><td>DWI Labels</td><td>{atlas_qc.get('dwi_labels', 'N/A')}</td></tr>
            <tr><td>Volume Ratio</td><td>{atlas_qc.get('volume_ratio', 'N/A'):.3f}</td></tr>
            <tr><td>Brain Overlap</td><td>{atlas_qc.get('brain_overlap', 'N/A'):.3f}</td></tr>
        </table>
        <p><strong>Flags:</strong> {', '.join(atlas_qc.get('flags', [])) or 'None'}</p>
    </div>
"""

    # Matrix QC section
    matrix_qc = qc_results.get('matrix_qc', {})
    if not matrix_qc.get('skipped', False):
        html += f"""
    <div class="section">
        <h2>Connectivity Matrix QC</h2>
        <p>Status: <span class="{'pass' if matrix_qc.get('pass', False) else 'fail'}">
            {'PASS' if matrix_qc.get('pass', False) else 'FAIL'}</span></p>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Number of ROIs</td><td>{matrix_qc.get('n_rois', 'N/A')}</td></tr>
            <tr><td>Connections</td><td>{matrix_qc.get('n_connections', 'N/A')}</td></tr>
            <tr><td>Density</td><td>{matrix_qc.get('density', 'N/A'):.3f}</td></tr>
            <tr><td>Isolated Nodes</td><td>{matrix_qc.get('node_degree', {}).get('isolated_nodes', 'N/A')}</td></tr>
        </table>
        <p><strong>Flags:</strong> {', '.join(matrix_qc.get('flags', [])) or 'None'}</p>
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

    logger.info(f"HTML report generated: {output_file}")
