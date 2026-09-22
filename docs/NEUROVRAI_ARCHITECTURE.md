# neurovrai Architecture & Roadmap

**Date**: 2025-11-16
**Status**: Planning Document

## Project Rename

**Current Name**: `neurovrai`
**New Name**: `neurovrai`

The project is being renamed to better reflect its comprehensive scope beyond just preprocessing. "neurovrai" (French: "true neuro") emphasizes the complete, accurate neuroimaging analysis pipeline.

---

## neurovrai: Three-Part Architecture

neurovrai will be organized as a **single integrated package** with three main submodules, sharing configuration, data formats, and directory hierarchies.

### Part 1: `neurovrai.preprocess` (Current - Mostly Complete ✅)

**Purpose**: Subject-level preprocessing for all MRI modalities

**Scope**:
- **Anatomical**: T1w/T2w preprocessing (N4 bias correction, skull stripping, segmentation, registration to MNI152)
- **Diffusion**: Multi-shell/single-shell DWI (TOPUP, eddy, DTI, DKI, NODDI, AMICO models)
  - Includes **BEDPOSTX** for fiber orientation estimation (GPU-accelerated)
  - BEDPOSTX output used by tractography in `neurovrai.connectome`
- **Functional**: Multi-echo (TEDANA) and single-echo (ICA-AROMA) resting-state fMRI
- **ASL**: Perfusion imaging (M0 calibration, CBF quantification, PVC)
- **Quality Control**: Automated QC for all modalities with HTML reports

**Why BEDPOSTX stays in preprocessing**:
- BEDPOSTX is fiber orientation estimation (preprocessing step)
- Generates `.bedpostX` directory with orientation distributions
- Similar to tensor fitting (dtifit) - estimates tissue microstructure
- Tractography/streamline generation moved to `neurovrai.connectome`

**Status**: Production-ready for all modalities

**Current Location**: `neurovrai/` (will be renamed to `neurovrai/preprocess/`)

---

### Part 2: `neurovrai.analysis` (Planned - Group-Level Statistics)

**Purpose**: Voxelwise and ROI-based group statistical analyses

**Scope**:

#### Anatomical Analysis
- **VBM (Voxel-Based Morphometry)**: Statistical comparison of brain structure
  - Gray matter and white matter concentration analysis
  - FSL (fslvbm) or ANTs implementation
  - Multi-subject group studies
- **Myelin Mapping**: T1w/T2w ratio images
  - Myelin content proxy from intensity ratios
  - Modernize existing legacy implementation

#### Diffusion Analysis
- **TBSS (Tract-Based Spatial Statistics)**: FA group analysis pipeline
  - Skeleton-based voxelwise statistics across subjects
  - Multi-subject white matter comparisons
  - Supports FA, MD, AD, RD, and advanced metrics (MK, FICVF, ODI)

#### Functional Analysis
- **MELODIC (Group ICA)**: Identify consistent resting-state networks across subjects
  - Temporal concatenation approach
  - Component spatial maps and time courses
- **ReHo (Regional Homogeneity)**: Local functional connectivity
  - Kendall's coefficient of concordance
  - Voxelwise or ROI-based measurements
- **fALFF (Fractional ALFF)**: Low-frequency fluctuation analysis
  - Ratio of low-frequency to total power
  - Frequency-domain connectivity measures

#### Perfusion Analysis
- **Group-Level CBF Analysis**: Perfusion comparisons and modeling
  - CBF group statistics and test-retest reliability
  - Arterial transit time analysis
  - Perfusion-based connectivity

**Inputs**: Preprocessed subjects from `neurovrai.preprocess`
**Outputs**: Statistical maps, group comparison results, cluster tables

**Proposed Structure**:
```
neurovrai/analysis/
├── vbm/              # Anatomical group analysis
│   ├── vbm_workflow.py
│   └── dartel.py     # Optional DARTEL registration
├── tbss/             # DWI skeleton-based statistics
│   ├── tbss_workflow.py
│   └── randomise.py  # Permutation testing
├── melodic/          # Functional group ICA
│   ├── group_ica.py
│   └── dual_regression.py
├── functional/       # ReHo, fALFF, etc.
│   ├── reho.py
│   ├── falff.py
│   └── seed_based.py
├── perfusion/        # ASL group analysis
│   ├── group_cbf.py
│   └── att_analysis.py
├── myelin/           # T1w/T2w ratio
│   └── myelin_workflow.py
└── utils/            # Shared analysis utilities
    ├── randomise_wrapper.py
    └── cluster_extraction.py
```

**Status**: Not yet implemented (planned after preprocessing completion)

---

### Part 3: `neurovrai.connectome` (Planned - Connectivity & Networks)

**Purpose**: Connectivity matrices and network neuroscience analyses

**Scope**:

#### Structural Connectivity
- **Subject-Level**:
  - **NxN Connectivity Matrices**: All-to-all ROI probabilistic tractography
  - **BEDPOSTX Integration**: Uses fiber orientations from `neurovrai.preprocess`
  - **Anatomical Constraints**:
    - White matter masks (keep streamlines in WM)
    - CSF exclusion masks (prevent ventricular crossing)
    - Gray matter interface constraints
    - Exclusion zones for implausible pathways
  - **Parcellation-Based**: Whole-brain atlases (Desikan-Killiany, Schaefer, AAL, etc.)
  - **Quality Control**: Streamline count thresholds, distance correction, tract density maps
- **Group-Level**:
  - Average connectivity matrices across subjects
  - Group-level graph metrics
  - Population connectivity patterns
  - Consistency thresholding (connections present in N% of subjects)

#### Functional Connectivity
- **Subject-Level**:
  - Seed-based correlation analysis
  - ROI-to-ROI functional connectivity (FC) matrices
  - Time-lagged connectivity
- **Group-Level**:
  - Average FC matrices
  - Network module identification
  - Dynamic functional connectivity

#### Multi-Modal Connectivity
- **Structure-Function Coupling**: Correlation between SC and FC
- **Integrated Analyses**: Combining structural and functional network properties
- **Communication Models**: Shortest paths, diffusion, navigation

#### Network Analysis (Graph Theory)
- **Node Metrics**: Degree, betweenness centrality, local efficiency
- **Global Metrics**: Clustering coefficient, path length, small-worldness
- **Modularity**: Community detection, participation coefficient
- **Rich Club**: Hub identification and rich club organization
- **Resilience**: Attack tolerance, robustness

**Inputs**:
- BEDPOSTX fiber orientations from `neurovrai.preprocess` (`.bedpostX` directory)
- Preprocessed BOLD time-series from `neurovrai.preprocess.func_preprocess`
- Brain parcellations/atlases (FreeSurfer, Desikan-Killiany, Schaefer, AAL, custom)
- Anatomical masks (WM, GM, CSF) from `neurovrai.preprocess.anat_preprocess`

**Outputs**:
- Connectivity matrices (`.npy`, `.mat`, `.csv`)
- Network metrics (node-level and graph-level)
- Visualizations (connectograms, glass brains, circular plots)
- Group statistics

**Proposed Structure**:
```
neurovrai/connectome/
├── structural/                   # Structural connectivity
│   ├── tractography.py           # Probabilistic tractography (probtrackx2_gpu)
│   ├── anatomical_constraints.py # WM masks, CSF exclusion, GM interface
│   ├── matrix_builder.py         # NxN connectivity matrix construction
│   ├── group_sc.py               # Group structural connectivity
│   └── qc.py                     # Tractography quality control
├── functional/                   # Functional connectivity
│   ├── seed_based.py             # Seed correlation maps
│   ├── roi_roi.py                # ROI-to-ROI FC matrix
│   ├── dynamic_fc.py             # Sliding window FC
│   └── group_fc.py               # Group functional connectivity
├── multimodal/                   # SC-FC coupling
│   ├── coupling.py               # Structure-function relationships
│   └── communication.py          # Network communication models
├── network/                      # Graph theory
│   ├── metrics.py                # Node and graph metrics
│   ├── modularity.py             # Community detection
│   ├── rich_club.py              # Hub analysis
│   └── resilience.py             # Network robustness
├── parcellation/                 # Atlas management
│   ├── atlas_registry.py
│   ├── atlas_warp.py             # Warp atlases to subject space
│   └── custom_parcellation.py
├── visualization/                # Connectome visualization
│   ├── connectogram.py
│   ├── glass_brain.py
│   └── circular_plot.py
└── utils/
    ├── matrix_utils.py
    ├── threshold.py
    └── distance_correction.py
```

**Dependencies**:
- NetworkX (graph analysis)
- Brain Connectivity Toolbox (BCT) Python port
- NiBabel (parcellation I/O)
- Nilearn (visualization)

**Status**: Not yet implemented (planned after `neurovrai.analysis`)

---

## Data Flow

```
Subject: STUDY-0580101

┌─────────────────────────────────────────────────────────────┐
│ 1. neurovrai.preprocess (Subject-Level)                    │
│    Raw DICOM → Preprocessed images + QC                    │
│    - T1w: brain, tissue masks (WM/GM/CSF), MNI transform   │
│    - DWI: eddy-corrected, FA/MD/MK/FICVF, BEDPOSTX         │
│    - BOLD: denoised, motion-corrected, smoothed            │
│    - ASL: CBF maps, M0 calibration, PVC                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────┴──────────────────┐
         ↓                                     ↓
┌──────────────────────────┐    ┌──────────────────────────────┐
│ 2a. neurovrai.analysis  │    │ 2b. neurovrai.connectome     │
│     (Group-Level Stats)  │    │     (Connectivity & Networks)│
│                          │    │                              │
│ All subjects combined:   │    │ Subject-Level:               │
│ - TBSS: FA skeleton      │    │ - BEDPOSTX → Tractography    │
│ - VBM: GM concentration  │    │   (with WM/CSF constraints)  │
│ - MELODIC: RSN networks  │    │ - NxN SC matrix (parcellation│
│ - Group CBF maps         │    │ - BOLD → FC matrix           │
│                          │    │ - Graph metrics              │
│ Output: Statistical maps │    │                              │
│         p-value maps     │    │ Group-Level:                 │
│         cluster tables   │    │ - Average connectivity       │
│                          │    │ - Network topology           │
│                          │    │ - Module identification      │
└──────────────────────────┘    └──────────────────────────────┘
```

---

## Implementation Architecture Decisions

### 1. Shared Utilities ✅ **DECIDED**

**Decision**: Each submodule has its own `utils/` directory for module-specific functions. Shared code that's used across multiple modules stays in the base `neurovrai/utils/`.

**Structure**:
```
neurovrai/
├── utils/                      # SHARED utilities
│   ├── workflow.py             # Common Nipype helpers
│   ├── transforms.py           # Spatial transformation registry
│   └── qc_base.py              # Base QC classes
├── preprocess/
│   ├── workflows/
│   └── utils/                  # Preprocessing-specific
│       ├── topup_helper.py
│       ├── dwi_normalization.py
│       └── acompcor_helper.py
├── analysis/
│   └── utils/                  # Analysis-specific
│       ├── randomise_wrapper.py
│       └── cluster_extraction.py
└── connectome/
    └── utils/                  # Connectome-specific
        ├── matrix_utils.py
        └── threshold.py
```

**Rationale**:
- Avoids circular dependencies
- Clear separation of concerns
- Module-specific functions don't clutter shared utilities
- Easy to identify what's reusable vs specialized

---

### 2. Configuration Management ✅ **DECIDED**

**Decision**: Use a single `config.yaml` for all three submodules, with clear section hierarchy.

**Extended Config Structure**:
```yaml
# Project paths (shared across all modules)
project_dir: /path/to/study
rawdata_dir: ${project_dir}/subjects
derivatives_dir: ${project_dir}/derivatives
work_dir: ${project_dir}/work
qc_dir: ${project_dir}/qc

# Execution settings (shared)
execution:
  plugin: MultiProc
  n_procs: 6

# Templates (shared)
templates:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
  fmrib58_fa: /usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz

# ============================================================
# PREPROCESSING (neurovrai.preprocess)
# ============================================================
anatomical:
  bet:
    frac: 0.5
    reduce_bias: true
    robust: true
  segmentation:
    n_iterations: 5
    mrf_weight: 0.1
  run_qc: true

diffusion:
  topup:
    readout_time: 0.05
  eddy:
    use_cuda: true
  tractography:
    n_samples: 5000
    n_steps: 2000
    step_length: 0.5
    curvature_threshold: 0.2
  run_qc: true

functional:
  tr: 1.029
  te: [10.0, 30.0, 50.0]
  highpass: 0.001
  lowpass: 0.08
  fwhm: 6
  tedana:
    enabled: true
  acompcor:
    enabled: true
    num_components: 5
  run_qc: true

asl:
  labeling_duration: 1.8
  post_labeling_delay: 2.0
  lambda_blood: 0.9
  t1_blood: 1.65
  alpha: 0.85
  run_qc: true

# ============================================================
# ANALYSIS (neurovrai.analysis)
# ============================================================
analysis:
  tbss:
    threshold: 0.2              # FA threshold for skeleton
    use_nonlinear_reg: true
    randomise:
      n_permutations: 5000
      tfce: true                # Threshold-Free Cluster Enhancement

  vbm:
    modulation: true            # Modulated GM concentration
    fwhm_smooth: 6              # Smoothing kernel (mm)
    template: dartel            # or 'spm' or 'ants'

  melodic:
    n_components: 20            # Number of ICA components
    tr: 1.029
    highpass_cutoff: 100        # Temporal filtering (seconds)

  functional:
    reho:
      n_neighbors: 27           # 3D neighborhood (27 = 3x3x3 cube)
    falff:
      low_freq: 0.01
      high_freq: 0.08

  perfusion:
    gm_threshold: 0.7           # GM probability threshold
    outlier_sd: 3.0             # SD threshold for outlier detection

# ============================================================
# CONNECTOMICS (neurovrai.connectome)
# ============================================================
connectome:
  parcellation:
    default_atlas: 'HarvardOxford-cortical'  # or 'Desikan-Killiany', 'AAL', etc.
    custom_atlas: null          # Path to custom parcellation (optional)

  structural:
    weight_by: 'streamline_count'  # or 'fa_weighted', 'length_weighted'
    min_streamlines: 3          # Minimum streamlines for valid connection
    length_correction: true     # Correct for fiber length bias

  functional:
    method: 'correlation'       # or 'partial_correlation', 'mutual_info'
    fisher_z: true              # Fisher Z-transform correlations
    global_signal_regression: false

  network:
    threshold_method: 'proportional'  # or 'absolute', 'mst', 'density'
    threshold_value: 0.1        # Keep top 10% of connections
    weighted: true              # Use weighted graph metrics
    binary: false               # Also compute binary metrics

  visualization:
    node_size_metric: 'degree'
    edge_threshold: 0.3
    colormap: 'viridis'
```

**Config Loading**:
```python
from neurovrai.config import load_config

# All modules use the same loader
config = load_config('config.yaml')

# Each module accesses its own section
preprocess_config = config['anatomical']
analysis_config = config['analysis']['tbss']
connectome_config = config['connectome']['network']
```

**Rationale**:
- Single source of truth for all parameters
- Easy to share paths, templates, execution settings
- Clear hierarchical organization
- Can override per-module if needed
- Backward compatible (existing preprocess configs still work)

---

### 3. Package Structure ✅ **DECIDED**

**Decision**: Single integrated package (monorepo) with three submodules.

**Package Name**: `neurovrai`

**Directory Structure**:
```
neurovrai/                      # Root package
├── __init__.py
├── config.py                   # Shared config loader
├── utils/                      # Shared utilities
│   ├── workflow.py
│   ├── transforms.py
│   └── qc_base.py
│
├── preprocess/                 # Submodule 1: Preprocessing
│   ├── __init__.py
│   ├── workflows/
│   │   ├── anat_preprocess.py
│   │   ├── dwi_preprocess.py
│   │   ├── func_preprocess.py
│   │   ├── asl_preprocess.py
│   │   ├── advanced_diffusion.py
│   │   ├── amico_models.py
│   │   └── tractography.py
│   ├── qc/
│   └── utils/
│
├── analysis/                   # Submodule 2: Group Analysis
│   ├── __init__.py
│   ├── vbm/
│   ├── tbss/
│   ├── melodic/
│   ├── functional/
│   ├── perfusion/
│   ├── myelin/
│   └── utils/
│
└── connectome/                 # Submodule 3: Connectivity
    ├── __init__.py
    ├── structural/
    ├── functional/
    ├── multimodal/
    ├── network/
    ├── parcellation/
    ├── visualization/
    └── utils/
```

**Import Style**:
```python
# Preprocessing
from neurovrai.preprocess.workflows import anat_preprocess, dwi_preprocess

# Analysis
from neurovrai.analysis.tbss import tbss_workflow

# Connectomics
from neurovrai.connectome.structural import matrix_builder
```

**Rationale**:
- **Shared infrastructure**: Same config, same directory hierarchy, same data formats
- **Version coherence**: All modules versioned together, no compatibility issues
- **Simpler for users**: `pip install neurovrai` gets everything
- **Code reuse**: Easy to import across submodules
- **Documentation**: Single unified docs site
- **Development**: Easier to maintain consistency

**Alternative considered**: Separate repos (`neurovrai-preprocess`, `neurovrai-analysis`, `neurovrai-connectome`)
- **Rejected because**: Would require managing dependencies between repos, separate versioning, duplicated utilities

---

## Implementation Priority

### Phase 1: Complete Preprocessing ✅ (Current)
- ✅ Anatomical, DWI, functional, ASL workflows production-ready
- ✅ QC modules for all modalities
- 🔄 Final testing and bug fixes (in progress)

### Phase 2: Project Rename & Restructure (Next - Est. 1 week)
1. Rename repository: `neurovrai` → `neurovrai`
2. Restructure package: `neurovrai/` → `neurovrai/preprocess/`
3. Update all imports and documentation
4. Extend `config.yaml` with analysis/connectome sections (with defaults)
5. Update README, CLAUDE.md, all docs

### Phase 3: Analysis Module (Est. 4-6 weeks)
**Priority order** (most requested → least):
1. **TBSS** (DWI skeleton analysis) - High demand, well-established method
2. **VBM** (Structural analysis) - Common in clinical studies
3. **MELODIC** (Functional networks) - Complements single-subject preprocessing
4. **ReHo/fALFF** (Functional metrics) - Quick to implement, useful metrics
5. **Myelin mapping** (Legacy code modernization)
6. **Group CBF** (ASL group analysis)

### Phase 4: Connectome Module (Est. 6-8 weeks)
**Priority order**:
1. **Structural connectivity matrices** - Uses existing tractography
2. **Functional connectivity matrices** - Essential for network analysis
3. **Basic graph metrics** - Degree, clustering, path length
4. **Visualization** - Connectograms, glass brains
5. **Advanced metrics** - Modularity, rich club
6. **Multi-modal coupling** - SC-FC relationships

---

## Future: neurofaune (Animal MRI Pipeline)

**Project**: `neurofaune`
**Purpose**: MRI preprocessing and analysis for animal (rodent) neuroimaging
**Status**: Planned (after neurovrai completion)

### Key Differences from neurovrai

**Species-Specific Considerations**:
- **Rodent brain templates**: Allen Brain Atlas, Waxholm Space (rat), SIGMA (mouse)
- **Resolution**: Higher resolution (e.g., 100-200 μm vs 1-2 mm in humans)
- **Acquisition**: Often ex vivo or anesthetized in vivo imaging
- **Brain size**: Smaller volumes, different atlas parcellations
- **Field strength**: Often 7T, 9.4T, or higher (vs 3T in humans)

**Architecture**: Identical three-part structure
```
neurofaune/
├── preprocess/    # Rodent-specific preprocessing
├── analysis/      # Rodent group analysis
└── connectome/    # Rodent connectivity
```

**Shared Components**:
- Same workflow engine (Nipype)
- Same config-driven architecture
- Same directory hierarchy
- Similar QC framework

**Species-Specific Modules**:
- Template registration (Allen CCFv3, Waxholm)
- Skull stripping optimized for rodent anatomy
- Parcellation based on rodent atlases
- Species-specific connectivity analysis

**Timeline**: Begin development after neurovrai analysis module is stable

**Collaboration Potential**: Share utilities, QC framework, visualization tools between neurovrai and neurofaune

---

## Development Guidelines

### Code Organization
- **Preprocessing**: `neurovrai/preprocess/` (current `neurovrai/`)
- **Analysis**: `neurovrai/analysis/` (new)
- **Connectomics**: `neurovrai/connectome/` (new)
- **Shared**: `neurovrai/utils/`, `neurovrai/config.py`

### Testing Strategy
- Unit tests: `tests/preprocess/`, `tests/analysis/`, `tests/connectome/`
- Integration tests: Full pipeline runs on test datasets
- Regression tests: Ensure analyses match established tools (FSL, SPM)

### Documentation
- **User Guide**: `docs/user_guide.md` - How to use neurovrai
- **API Reference**: Auto-generated from docstrings
- **Workflows**: `docs/workflows.md` - Detailed pipeline descriptions
- **Examples**: `examples/` - Complete analysis examples

### Version Control
- Semantic versioning: `v1.0.0` (preprocessing), `v2.0.0` (+ analysis), `v3.0.0` (+ connectome)
- Changelog: Track all changes, especially API modifications
- Git branches: `main` (stable), `develop` (active), feature branches

---

## Questions & Decisions Summary

| Question | Decision | Rationale |
|----------|----------|-----------|
| **1. Shared utilities?** | Each module has `utils/`, shared code in base `neurovrai/utils/` | Clear separation, avoid circular deps |
| **2. Configuration?** | Single `config.yaml` with sections | Single source of truth, easier management |
| **3. Package structure?** | Monorepo (single `neurovrai` package) | Shared infrastructure, version coherence, simpler |
| **4. Tractography location?** | Move from `preprocess` to `connectome` | NxN matrices need constraints, better domain separation |

---

## Tractography Migration Plan

### Current State
**File**: `neurovrai/preprocess/workflows/tractography.py`
**Status**: Implemented but not production-ready for connectomics

**Current Implementation**:
- Hypothesis-driven seed→target connectivity
- Manual ROI specification required
- No anatomical constraints (WM masks, CSF exclusion)
- Not optimized for NxN matrix generation
- Uses atlas-based ROIs (Harvard-Oxford, JHU)

### Why Migration is Needed
1. **Connectomics Requirements**:
   - Need NxN all-to-all connectivity (current: seed→target pairs)
   - Need anatomical constraints to prevent false positives (CSF crossing, etc.)
   - Need distance correction and quality thresholding
   - Need optimized parallel processing for whole-brain parcellations

2. **Architectural Clarity**:
   - BEDPOSTX = preprocessing (fiber orientation estimation)
   - Tractography = connectivity analysis (uses orientations to build networks)
   - Separation matches domain boundaries

### Migration Strategy

**Phase 1: Preserve Current Functionality** (During restructure to neurovrai)
- Mark `neurovrai/preprocess/workflows/tractography.py` as **deprecated**
- Add warning message when imported
- Keep functional for backward compatibility
- Document migration path in deprecation notice

**Phase 2: Implement in `neurovrai.connectome`** (Part 3 development)
- Build proper connectomics-focused tractography:
  - `neurovrai/connectome/structural/tractography.py` - NxN matrix generation
  - `neurovrai/connectome/structural/anatomical_constraints.py` - WM/CSF/GM masks
  - `neurovrai/connectome/structural/matrix_builder.py` - Connectivity matrix construction
- Reuse BEDPOSTX outputs from preprocessing
- Add proper QC and validation

**Phase 3: Remove Legacy Code** (After `neurovrai.connectome` is stable)
- Remove deprecated `tractography.py` from preprocessing
- Update documentation to point to connectome module
- Provide migration guide for existing users

### Deprecation Notice (to be added)
```python
# neurovrai/preprocess/workflows/tractography.py

import warnings

warnings.warn(
    "tractography.py in neurovrai.preprocess.workflows is deprecated and will be "
    "removed in v3.0.0. Use neurovrai.connectome.structural for connectomics-ready "
    "tractography with anatomical constraints and NxN matrix generation. "
    "See migration guide: docs/TRACTOGRAPHY_MIGRATION.md",
    DeprecationWarning,
    stacklevel=2
)
```

---

## Next Steps

1. **Immediate**: Finish preprocessing bug fixes and testing
2. **Week 1**: Rename project to neurovrai, restructure package
3. **Week 2-7**: Implement `neurovrai.analysis` module (TBSS → VBM → MELODIC → ReHo/fALFF)
4. **Week 8-15**: Implement `neurovrai.connectome` module (SC matrices → FC matrices → graph metrics)
5. **Future**: Begin neurofaune planning and development

---

**Document Status**: Planning & Architecture
**Last Updated**: 2025-11-16
**Next Review**: After preprocessing completion (Est. 2025-11-20)
