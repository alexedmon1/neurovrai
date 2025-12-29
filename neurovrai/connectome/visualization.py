#!/usr/bin/env python3
"""
Connectome Visualization Module

Visualization tools for connectivity matrices and brain networks.

Key Features:
- Connectivity matrix heatmaps with hierarchical clustering
- Circular connectograms for network visualization
- Degree distribution plots
- Edge weight distributions
- Group comparison visualizations

Usage:
    from neurovrai.connectome.visualization import (
        plot_connectivity_matrix,
        plot_circular_connectogram
    )

    # Plot connectivity matrix heatmap
    plot_connectivity_matrix(
        matrix=fc_matrix,
        labels=roi_names,
        output_file='fc_heatmap.png'
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


def plot_connectivity_matrix(
    matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    output_file: Optional[Union[str, Path]] = None,
    title: str = 'Connectivity Matrix',
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cluster: bool = False,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
) -> plt.Figure:
    """
    Plot connectivity matrix as heatmap

    Args:
        matrix: Connectivity matrix (n_rois, n_rois)
        labels: Optional ROI labels
        output_file: Path to save figure
        title: Figure title
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cluster: Apply hierarchical clustering
        show_labels: Show ROI labels on axes
        figsize: Figure size in inches
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    n_rois = matrix.shape[0]

    if labels is None:
        labels = [f"ROI_{i:03d}" for i in range(n_rois)]

    logger.info(f"Plotting connectivity matrix: {matrix.shape}")

    # Apply clustering if requested
    if cluster:
        logger.info("Applying hierarchical clustering...")
        # Convert to distance matrix
        dist_matrix = 1 - np.abs(matrix)
        np.fill_diagonal(dist_matrix, 0)

        # Hierarchical clustering
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage = hierarchy.linkage(condensed_dist, method='average')
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        order = dendro['leaves']

        # Reorder matrix and labels
        matrix = matrix[order, :][:, order]
        labels = [labels[i] for i in order]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Connectivity', rotation=270, labelpad=20)

    # Set labels
    if show_labels and n_rois <= 50:
        ax.set_xticks(range(n_rois))
        ax.set_yticks(range(n_rois))
        ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xlabel(f'{n_rois} ROIs')
        ax.set_ylabel(f'{n_rois} ROIs')

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid
    ax.set_xticks(np.arange(n_rois) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rois) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved connectivity matrix plot: {output_file}")

    return fig


def plot_circular_connectogram(
    matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    output_file: Optional[Union[str, Path]] = None,
    title: str = 'Circular Connectogram',
    threshold: Optional[float] = None,
    edge_cmap: str = 'coolwarm',
    node_colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 12),
    dpi: int = 300,
    line_width_scale: float = 2.0
) -> plt.Figure:
    """
    Plot circular connectogram

    Args:
        matrix: Connectivity matrix (n_rois, n_rois)
        labels: Optional ROI labels
        output_file: Path to save figure
        title: Figure title
        threshold: Threshold for displaying edges (absolute value)
        edge_cmap: Colormap for edge colors
        node_colors: Optional list of colors for nodes
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        line_width_scale: Scaling factor for line widths

    Returns:
        Matplotlib figure object
    """
    n_rois = matrix.shape[0]

    if labels is None:
        labels = [f"ROI_{i:03d}" for i in range(n_rois)]

    logger.info(f"Plotting circular connectogram: {n_rois} nodes")

    # Threshold matrix if requested
    if threshold is not None:
        plot_matrix = matrix.copy()
        plot_matrix[np.abs(plot_matrix) < threshold] = 0
        n_edges = np.sum(plot_matrix != 0) / 2
        logger.info(f"  Threshold: {threshold}, Edges: {n_edges}")
    else:
        plot_matrix = matrix

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')

    # Node positions (evenly spaced around circle)
    angles = np.linspace(0, 2 * np.pi, n_rois, endpoint=False)

    # Plot nodes
    if node_colors is None:
        node_colors = ['steelblue'] * n_rois

    ax.scatter(angles, np.ones(n_rois), c=node_colors, s=100, zorder=10)

    # Add labels
    for angle, label in zip(angles, labels):
        rotation = np.degrees(angle)
        if rotation > 90 and rotation < 270:
            rotation = rotation + 180
            ha = 'right'
        else:
            ha = 'left'

        ax.text(angle, 1.15, label, rotation=rotation, ha=ha, va='center',
                fontsize=6, rotation_mode='anchor')

    # Plot edges
    cmap = plt.get_cmap(edge_cmap)
    vmin = plot_matrix[plot_matrix != 0].min() if np.any(plot_matrix != 0) else -1
    vmax = plot_matrix[plot_matrix != 0].max() if np.any(plot_matrix != 0) else 1

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            if plot_matrix[i, j] != 0:
                norm_value = (plot_matrix[i, j] - vmin) / (vmax - vmin)
                color = cmap(norm_value)
                line_width = np.abs(plot_matrix[i, j]) * line_width_scale

                theta = np.linspace(angles[i], angles[j], 100)
                r = np.ones(100)

                ax.plot(theta, r, color=color, linewidth=line_width,
                       alpha=0.6, zorder=1)

    # Configure plot
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved connectogram: {output_file}")

    return fig


def plot_connectivity_comparison(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    labels: Optional[List[str]] = None,
    output_file: Optional[Union[str, Path]] = None,
    title1: str = 'Group 1',
    title2: str = 'Group 2',
    cmap: str = 'RdBu_r',
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 300
) -> plt.Figure:
    """Plot side-by-side comparison of two connectivity matrices"""
    n_rois = matrix1.shape[0]

    if labels is None:
        labels = [f"ROI_{i:03d}" for i in range(n_rois)]

    logger.info("Plotting connectivity comparison...")

    diff_matrix = matrix1 - matrix2

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    vmin = min(matrix1.min(), matrix2.min())
    vmax = max(matrix1.max(), matrix2.max())

    im1 = axes[0].imshow(matrix1, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(title1, fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(matrix2, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(title2, fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    diff_max = np.abs(diff_matrix).max()
    im3 = axes[2].imshow(diff_matrix, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, aspect='auto')
    axes[2].set_title('Difference (1 - 2)', fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label('Difference', rotation=270, labelpad=20)

    for ax in axes:
        if n_rois <= 50:
            ax.set_xticks(range(n_rois))
            ax.set_yticks(range(n_rois))
            ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_xlabel(f'{n_rois} ROIs')
            ax.set_ylabel(f'{n_rois} ROIs')

    plt.tight_layout()

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved comparison plot: {output_file}")

    return fig


# TODO: Circos plot implementation is in development
# Current limitations:
# - Basic bezier curves for edges (not true circos arcs)
# - No anatomical label support yet
# - Edge bundling not implemented
# - Needs refinement for publication quality
def plot_circos_connectome(
    matrix: np.ndarray,
    roi_names: List[str],
    output_file: Optional[Union[str, Path]] = None,
    title: str = 'Connectivity',
    threshold_pct: float = 90.0,
    cmap: str = 'RdBu_r',
    figsize: Tuple[int, int] = (14, 14),
    dpi: int = 300
) -> plt.Figure:
    """
    Create a circos-style connectivity plot.

    NOTE: This function is in development and may change.

    Args:
        matrix: Connectivity matrix (n_rois, n_rois)
        roi_names: List of ROI names (format: ROI_XXXX)
        output_file: Path to save figure
        title: Figure title
        threshold_pct: Percentile threshold for displaying edges (0-100)
        cmap: Colormap for edges
        figsize: Figure size in inches
        dpi: Resolution for saved figure

    Returns:
        Matplotlib figure object
    """
    import matplotlib.colors as mcolors

    n_rois = len(roi_names)

    # Extract hemisphere info from ROI names
    def get_hemisphere(roi_name):
        roi_num = int(roi_name.replace('ROI_', ''))
        if 1000 <= roi_num <= 1999:
            return 'L'
        elif 2000 <= roi_num <= 2999:
            return 'R'
        else:
            return 'Sub'

    hemispheres = [get_hemisphere(r) for r in roi_names]

    # Sort ROIs by hemisphere for visualization
    left_idx = [i for i, h in enumerate(hemispheres) if h == 'L']
    right_idx = [i for i, h in enumerate(hemispheres) if h == 'R']
    sub_idx = [i for i, h in enumerate(hemispheres) if h == 'Sub']
    order = left_idx + sub_idx + right_idx[::-1]

    matrix_ordered = matrix[np.ix_(order, order)]
    hemi_ordered = [hemispheres[i] for i in order]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    angles = np.linspace(0, 2 * np.pi, n_rois, endpoint=False)

    # Node colors by hemisphere
    hemi_colors = {'L': '#1f77b4', 'R': '#d62728', 'Sub': '#2ca02c'}
    node_colors = [hemi_colors[h] for h in hemi_ordered]

    # Plot nodes
    for angle, color in zip(angles, node_colors):
        ax.scatter([angle], [1.0], c=[color], s=80, zorder=10)

    # Get threshold
    upper_tri = matrix_ordered[np.triu_indices(n_rois, k=1)]
    valid_edges = upper_tri[~np.isnan(upper_tri) & (upper_tri != 0)]
    threshold = np.percentile(np.abs(valid_edges), threshold_pct) if len(valid_edges) > 0 else 0

    # Setup colormap
    cmap_obj = plt.get_cmap(cmap)
    valid_values = matrix_ordered[~np.isnan(matrix_ordered) & (matrix_ordered != 0)]
    if len(valid_values) > 0:
        vmax = np.percentile(np.abs(valid_values), 99)
        vmin = -vmax if np.min(valid_values) < 0 else 0
    else:
        vmax, vmin = 1, 0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot edges as bezier curves
    edges_plotted = 0
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            val = matrix_ordered[i, j]
            if not np.isnan(val) and np.abs(val) >= threshold:
                theta1, theta2 = angles[i], angles[j]

                # Bezier curve through center
                t = np.linspace(0, 1, 50)
                r_control = 0.3
                theta_mid = (theta1 + theta2) / 2
                if abs(theta2 - theta1) > np.pi:
                    theta_mid += np.pi

                theta = (1-t)**2 * theta1 + 2*(1-t)*t * theta_mid + t**2 * theta2
                r = (1-t)**2 * 0.95 + 2*(1-t)*t * r_control + t**2 * 0.95

                color = cmap_obj(norm(val))
                width = 0.5 + 1.5 * (np.abs(val) / vmax) if vmax > 0 else 0.5

                ax.plot(theta, r, color=color, linewidth=width, alpha=0.6, zorder=1)
                edges_plotted += 1

    # Labels
    ax.annotate('Left Hemisphere', xy=(np.pi/2, 1.2), fontsize=12,
                ha='center', fontweight='bold', color='#1f77b4')
    ax.annotate('Right Hemisphere', xy=(3*np.pi/2, 1.2), fontsize=12,
                ha='center', fontweight='bold', color='#d62728')

    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08,
                        fraction=0.03, aspect=40)
    cbar.set_label('Connectivity Strength', fontsize=10)

    plt.title(f'{title}\n({edges_plotted} edges, {n_rois} ROIs)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved circos plot: {output_file}")

    return fig
