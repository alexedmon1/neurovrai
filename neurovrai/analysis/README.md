# neurovrai.analysis

**Status**: Planned (Phase 3)

## Overview

Group-level statistical analyses for neuroimaging data.

## Planned Submodules

### VBM (Voxel-Based Morphometry)
- Gray/white matter concentration analysis
- Statistical comparison of brain structure across groups
- FSL or ANTs implementation

### TBSS (Tract-Based Spatial Statistics)
- FA skeleton-based voxelwise statistics
- Supports all DTI metrics (FA, MD, AD, RD) and advanced metrics (MK, FICVF, ODI)
- Permutation testing with FSL randomise

### MELODIC (Group ICA)
- Identify consistent resting-state networks across subjects
- Temporal concatenation group ICA
- Dual regression for subject-specific networks

### Functional Analysis
- ReHo (Regional Homogeneity)
- fALFF (Fractional Amplitude of Low-Frequency Fluctuations)
- Seed-based connectivity group analysis

### Perfusion Analysis
- Group-level CBF statistics
- Arterial transit time analysis
- Perfusion-connectivity analysis

## Implementation Timeline

**Start**: After preprocessing module is complete and tested
**Estimated Duration**: 4-6 weeks
**Priority**: TBSS → VBM → MELODIC → ReHo/fALFF → ASL group analysis

## Development Notes

See `docs/NEUROVRAI_ARCHITECTURE.md` for detailed architecture and implementation plan.
