#!/usr/bin/env python3
"""
Graph Theory Metrics Module

Compute graph-theoretic network metrics for brain connectivity analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


def threshold_and_binarize(matrix, threshold=None, density=None, binarize=True):
    """Threshold connectivity matrix"""
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    
    if threshold is not None:
        matrix[np.abs(matrix) < threshold] = 0
    elif density is not None:
        n_edges_target = int(density * matrix.size)
        flat_abs = np.abs(matrix.flatten())
        threshold_val = np.sort(flat_abs)[-n_edges_target]
        matrix[np.abs(matrix) < threshold_val] = 0
    
    if binarize:
        matrix = (matrix != 0).astype(float)
    
    return matrix


def matrix_to_graph(matrix, weighted=True):
    """Convert connectivity matrix to NetworkX graph"""
    if weighted:
        G = nx.from_numpy_array(np.abs(matrix))
    else:
        binary = (matrix != 0).astype(int)
        G = nx.from_numpy_array(binary)
    
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def compute_node_degree(matrix, weighted=False):
    """Compute node degree"""
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    
    if weighted:
        degree = np.sum(np.abs(matrix), axis=1)
    else:
        degree = np.sum(matrix != 0, axis=1)
    
    return degree


def compute_node_strength(matrix):
    """Compute node strength"""
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    return np.sum(np.abs(matrix), axis=1)


def compute_clustering_coefficient(matrix, weighted=False):
    """Compute clustering coefficient"""
    G = matrix_to_graph(matrix, weighted=weighted)
    
    if weighted:
        clustering = nx.clustering(G, weight='weight')
    else:
        clustering = nx.clustering(G)
    
    return np.array([clustering[i] for i in range(len(clustering))])


def compute_betweenness_centrality(matrix, weighted=False, normalized=True):
    """Compute betweenness centrality"""
    G = matrix_to_graph(matrix, weighted=weighted)
    
    if weighted:
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=normalized)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=normalized)
    
    return np.array([betweenness[i] for i in range(len(betweenness))])


def compute_node_metrics(matrix, threshold=None, weighted=False, roi_names=None):
    """Compute comprehensive node-level metrics"""
    n_nodes = matrix.shape[0]
    
    if roi_names is None:
        roi_names = [f"Node_{i:03d}" for i in range(n_nodes)]
    
    logger.info(f"Computing node metrics for {n_nodes} nodes...")
    
    if threshold is not None:
        matrix_thresh = threshold_and_binarize(matrix, threshold=threshold, binarize=not weighted)
    else:
        matrix_thresh = matrix.copy()
    
    degree = compute_node_degree(matrix_thresh, weighted=weighted)
    strength = compute_node_strength(matrix_thresh)
    clustering = compute_clustering_coefficient(matrix_thresh, weighted=weighted)
    betweenness = compute_betweenness_centrality(matrix_thresh, weighted=weighted)
    
    logger.info("  OK Computed node metrics")
    
    return {
        'roi_names': roi_names,
        'degree': degree,
        'strength': strength,
        'clustering_coefficient': clustering,
        'betweenness_centrality': betweenness,
        'threshold': threshold,
        'weighted': weighted
    }


def compute_global_efficiency(matrix):
    """Compute global efficiency"""
    G = matrix_to_graph(matrix, weighted=True)
    
    for u, v, data in G.edges(data=True):
        if data['weight'] > 0:
            data['distance'] = 1.0 / data['weight']
        else:
            data['distance'] = np.inf
    
    return float(nx.global_efficiency(G))


def compute_characteristic_path_length(matrix):
    """Compute characteristic path length"""
    G = matrix_to_graph(matrix, weighted=True)
    
    for u, v, data in G.edges(data=True):
        if data['weight'] > 0:
            data['distance'] = 1.0 / data['weight']
        else:
            data['distance'] = np.inf
    
    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G, weight='distance')
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc)
        path_length = nx.average_shortest_path_length(G_sub, weight='distance')
        logger.warning(f"Graph disconnected, using largest component")
    
    return float(path_length)


def compute_transitivity(matrix):
    """Compute global clustering coefficient"""
    G = matrix_to_graph(matrix, weighted=False)
    return float(nx.transitivity(G))


def compute_global_metrics(matrix, threshold=None):
    """Compute comprehensive global network metrics"""
    logger.info("Computing global network metrics...")
    
    if threshold is not None:
        matrix_thresh = threshold_and_binarize(matrix, threshold=threshold, binarize=True)
    else:
        matrix_thresh = matrix.copy()
    
    global_eff = compute_global_efficiency(matrix_thresh)
    char_path = compute_characteristic_path_length(matrix_thresh)
    transitivity = compute_transitivity(matrix_thresh)
    
    logger.info(f"  Global efficiency: {global_eff:.4f}")
    logger.info(f"  Path length: {char_path:.4f}")
    logger.info(f"  Transitivity: {transitivity:.4f}")
    
    return {
        'global_efficiency': global_eff,
        'characteristic_path_length': char_path,
        'transitivity': transitivity,
        'threshold': threshold
    }


def identify_hubs(node_metrics, method='degree', percentile=90):
    """Identify hub nodes"""
    metric_values = node_metrics[method] if method == 'degree' else node_metrics[f'{method}_centrality']
    threshold = np.percentile(metric_values, percentile)
    hubs = metric_values >= threshold

    logger.info(f"Identified {np.sum(hubs)} hub nodes using {method}")

    return hubs


def compute_graph_metrics(
    connectivity_matrix: np.ndarray,
    roi_names: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    weighted: bool = False
) -> Dict:
    """
    Compute comprehensive graph metrics for a connectivity matrix.

    Combines node-level and global network metrics into a single
    JSON-serializable dictionary.

    Args:
        connectivity_matrix: NxN connectivity matrix
        roi_names: List of ROI names (default: Node_000, Node_001, ...)
        threshold: Optional threshold to apply before computing metrics
        weighted: Use weighted metrics (default: False for binary)

    Returns:
        Dictionary with node_metrics, global_metrics, and summary
    """
    logger.info("Computing comprehensive graph metrics...")

    n_nodes = connectivity_matrix.shape[0]

    # Ensure symmetric matrix
    matrix = (connectivity_matrix + connectivity_matrix.T) / 2
    np.fill_diagonal(matrix, 0)

    # Compute node-level metrics
    node_metrics = compute_node_metrics(
        matrix,
        threshold=threshold,
        weighted=weighted,
        roi_names=roi_names
    )

    # Compute global metrics
    global_metrics = compute_global_metrics(matrix, threshold=threshold)

    # Identify hub nodes
    hubs_by_degree = identify_hubs(node_metrics, method='degree', percentile=90)
    hubs_by_between = identify_hubs(node_metrics, method='betweenness', percentile=90)

    # Convert arrays to lists for JSON serialization
    node_metrics_json = {
        'roi_names': list(node_metrics['roi_names']),
        'degree': node_metrics['degree'].tolist(),
        'strength': node_metrics['strength'].tolist(),
        'clustering_coefficient': node_metrics['clustering_coefficient'].tolist(),
        'betweenness_centrality': node_metrics['betweenness_centrality'].tolist(),
    }

    # Summary statistics
    summary = {
        'n_nodes': n_nodes,
        'n_edges': int(np.sum(matrix > 0) / 2),  # Undirected
        'density': float(np.sum(matrix > 0) / (n_nodes * (n_nodes - 1))),
        'mean_degree': float(np.mean(node_metrics['degree'])),
        'mean_strength': float(np.mean(node_metrics['strength'])),
        'mean_clustering': float(np.mean(node_metrics['clustering_coefficient'])),
        'n_hubs_degree': int(np.sum(hubs_by_degree)),
        'n_hubs_betweenness': int(np.sum(hubs_by_between)),
    }

    # Hub node names
    hub_names_degree = [node_metrics['roi_names'][i] for i in np.where(hubs_by_degree)[0]]
    hub_names_between = [node_metrics['roi_names'][i] for i in np.where(hubs_by_between)[0]]

    results = {
        'node_metrics': node_metrics_json,
        'global_metrics': global_metrics,
        'summary': summary,
        'hubs': {
            'by_degree': hub_names_degree,
            'by_betweenness': hub_names_between
        },
        'parameters': {
            'threshold': threshold,
            'weighted': weighted
        }
    }

    logger.info(f"✓ Graph metrics computed: {summary['n_edges']} edges, density={summary['density']:.3f}")

    return results
