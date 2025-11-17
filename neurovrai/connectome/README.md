# neurovrai.connectome

**Status**: Planned (Phase 4)

## Overview

Connectivity matrices and network neuroscience analyses.

## Planned Submodules

### Structural Connectivity
- **Tractography**: Probabilistic tractography using BEDPOSTX outputs from preprocessing
- **Anatomical Constraints**: WM masks, CSF exclusion, GM interface constraints
- **Matrix Builder**: NxN connectivity matrix generation from whole-brain parcellations
- **QC**: Streamline count thresholds, distance correction, tract density validation
- **Group Analysis**: Average connectivity matrices, population patterns

### Functional Connectivity
- **Seed-Based**: Correlation maps from seed regions
- **ROI-to-ROI**: Full NxN functional connectivity matrices
- **Dynamic FC**: Sliding window connectivity
- **Group Analysis**: Average FC matrices, network modules

### Multi-Modal Integration
- **SC-FC Coupling**: Structure-function relationships
- **Communication Models**: Shortest paths, diffusion, navigation efficiency

### Network Analysis (Graph Theory)
- **Node Metrics**: Degree, betweenness centrality, local efficiency
- **Global Metrics**: Clustering coefficient, characteristic path length, small-worldness
- **Modularity**: Community detection, participation coefficient
- **Rich Club**: Hub identification and rich club organization
- **Resilience**: Network robustness, attack tolerance

### Parcellation Management
- **Atlas Registry**: HarvardOxford, Desikan-Killiany, Schaefer, AAL, JHU
- **Warping**: Transform atlases to subject native space
- **Custom Parcellations**: Support for user-defined ROIs

### Visualization
- **Connectograms**: Circular connectivity plots
- **Glass Brain**: 3D connectivity visualization
- **Matrix Plots**: Connectivity matrix heatmaps

## Key Design Principles

1. **Separation from Preprocessing**: BEDPOSTX (fiber orientation estimation) stays in preprocessing; tractography (connectivity analysis) lives here
2. **Anatomical Constraints**: Enforce biologically plausible streamlines (no CSF crossing, WM-bound paths)
3. **Quality Control**: Rigorous validation of connectivity estimates
4. **Group Consistency**: Threshold connections present in minimum % of subjects

## Implementation Timeline

**Start**: After analysis module is complete
**Estimated Duration**: 6-8 weeks
**Priority**: Structural connectivity → Functional connectivity → Graph metrics → Visualization → Multi-modal

## Development Notes

See `docs/NEUROVRAI_ARCHITECTURE.md` for detailed architecture and implementation plan.
