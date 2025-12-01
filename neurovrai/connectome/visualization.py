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
