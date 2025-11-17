"""
neurovrai.connectome: Connectivity and network analysis

This module provides structural/functional connectivity analysis and
graph-theoretic network metrics.

Planned Submodules
------------------
structural : Probabilistic tractography and structural connectivity matrices
    - Uses BEDPOSTX outputs from neurovrai.preprocess
    - Anatomical constraints (WM masks, CSF exclusion)
    - NxN connectivity matrix generation

functional : Functional connectivity matrices
    - Seed-based correlation
    - ROI-to-ROI FC matrices
    - Dynamic functional connectivity

multimodal : Structure-function coupling
    - SC-FC correlations
    - Communication models

network : Graph theory metrics
    - Node metrics (degree, betweenness, efficiency)
    - Global metrics (clustering, path length, small-worldness)
    - Modularity and rich club analysis

parcellation : Brain atlas management
    - Atlas registration to subject space
    - Custom parcellation support

visualization : Connectome visualization
    - Connectograms
    - Glass brain plots
    - Circular connectivity plots

Status
------
This module is under active development (Phase 4).
See docs/NEUROVRAI_ARCHITECTURE.md for implementation roadmap.
"""

__version__ = "2.0.0-alpha"
__all__ = []  # Will be populated as submodules are implemented

# Placeholder - will be populated in Phase 4
