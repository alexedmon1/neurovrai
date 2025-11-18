# Analysis Module - Project Status
**Last Updated**: 2025-11-17

## 🎯 Project Overview

Implementation of group-level analysis tools for the neurovrai pipeline, focusing on VBM (Voxel-Based Morphometry) and TBSS (Tract-Based Spatial Statistics) with FSL randomise permutation testing.

**Module Location**: `neurovrai/analysis/`

**Priority**: High (Top priority after preprocessing completion)

---

## 📊 Current Status: Planning Phase

**Phase**: Requirements gathering and architecture design
**Progress**: 0% (Planning complete, ready to begin implementation)

---

## 🎯 Core Requirements

### Statistical Framework
- ✅ **Primary Tool**: FSL randomise (permutation testing)
- ✅ **Correction Method**: TFCE (Threshold-Free Cluster Enhancement)
- ✅ **Design Philosophy**: General-purpose (not specific to two-group comparisons)
- ✅ **Input Format**: FSL design matrix (`.mat`) and contrast files (`.con`)

### Required Outputs
- Statistical maps with TFCE correction
- Cluster tables (size, peak coordinates, p-values, anatomical labels)
- QC reports with visualizations
- Publication-ready figures

### Flexibility Requirements
- **Templates**: Both MNI152/FMRIB58 standard templates AND study-specific template creation
- **Smoothing**: Default 6mm FWHM, user-adjustable
- **Atlases**: Customizable atlas selection from implemented list
- **TBSS Metrics**: User-specified list (can run multiple: FA, MD, AD, RD, etc.)

---

## 📋 Implementation Roadmap

### Phase 1: Infrastructure & Design Matrix Handling

#### Task 1.1: Module Structure Setup
**Status**: ⏳ Not Started
**Estimated Time**: 30 minutes

- [ ] Create `neurovrai/analysis/` directory
- [ ] Create `neurovrai/analysis/__init__.py`
- [ ] Create `neurovrai/analysis/utils/` subdirectory
- [ ] Create `neurovrai/analysis/utils/__init__.py`

#### Task 1.2: Design Matrix Management
**Status**: ⏳ Not Started
**Estimated Time**: 2-3 hours

**File**: `neurovrai/analysis/utils/design_matrix.py`

- [ ] Implement `validate_design_matrix(mat_file: Path) -> bool`
  - Check FSL .mat file format
  - Validate matrix dimensions
  - Check for proper headers
- [ ] Implement `validate_contrast_file(con_file: Path, mat_file: Path) -> bool`
  - Check FSL .con file format
  - Verify contrast dimensions match design matrix
  - Validate contrast weights
- [ ] Implement `parse_design_matrix(mat_file: Path) -> Dict`
  - Extract group information
  - Return dict with: n_subjects, n_groups, group_labels, covariates
- [ ] Implement `create_simple_two_group_design(group1_n: int, group2_n: int, output_dir: Path) -> Tuple[Path, Path]`
  - Helper function for convenience
  - Generate .mat and .con files
  - Return paths to created files
- [ ] Add comprehensive docstrings and type hints

#### Task 1.3: FSL Helpers
**Status**: ⏳ Not Started
**Estimated Time**: 3-4 hours

**File**: `neurovrai/analysis/utils/fsl_helpers.py`

- [ ] Implement `run_randomise(input_4d: Path, design_mat: Path, contrasts: Path, output_prefix: Path, config: Dict) -> Dict`
  - Wrapper for FSL randomise command
  - Handle TFCE parameters from config
  - Set number of permutations
  - Capture and parse output
  - Return dict with output file paths
- [ ] Implement `extract_tfce_results(randomise_output_dir: Path, contrast_num: int) -> Dict`
  - Parse TFCE-corrected p-value maps
  - Extract raw t/F-stat maps
  - Return dict with file paths and summary stats
- [ ] Implement `generate_cluster_table(tfce_map: Path, threshold: float, atlas: str, output_file: Path) -> pd.DataFrame`
  - Use FSL cluster command
  - Apply cluster size threshold
  - Add anatomical labels from specified atlas
  - Save as CSV and return DataFrame
- [ ] Add error handling for missing FSL installation
- [ ] Add logging throughout

#### Task 1.4: Config System Extension
**Status**: ⏳ Not Started
**Estimated Time**: 1 hour

**Files**: `configs/config.yaml`, `create_config.py`

- [ ] Add `analysis` section to config template:
  ```yaml
  analysis:
    vbm:
      template: "MNI152"  # Options: "MNI152", "study_specific"
      smooth_fwhm: 6      # Smoothing kernel in mm
      tissue: "GM"        # Options: "GM", "WM", "both"
    tbss:
      skeleton_threshold: 0.2
      template: "FMRIB58"  # Options: "FMRIB58", "study_specific"
      metrics: ["FA"]      # List: FA, MD, AD, RD, MK, etc.
    randomise:
      n_permutations: 5000
      tfce: true
      tfce_H: 2           # TFCE height parameter
      tfce_E: 0.5         # TFCE extent parameter
      cluster_threshold: 0.95  # p-value threshold
      voxel_threshold: null    # Set to null for TFCE-only
    atlases:
      primary: "HarvardOxford-cort-maxprob-thr25-2mm"
      available:
        - "HarvardOxford-cort-maxprob-thr25-2mm"
        - "HarvardOxford-sub-maxprob-thr25-2mm"
        - "JHU-ICBM-labels-2mm"
        - "JHU-ICBM-tracts-maxprob-thr25-2mm"
        - "AAL3"
    output:
      save_corrected_p: true
      save_raw_stats: true
      cluster_min_size: 10  # Minimum cluster size in voxels
  ```
- [ ] Update `create_config.py` to include analysis defaults
- [ ] Add validation for analysis config section

---

### Phase 2: VBM Implementation

#### Task 2.1: VBM Data Preparation
**Status**: ⏳ Not Started
**Estimated Time**: 3-4 hours

**File**: `neurovrai/analysis/vbm.py`

- [ ] Implement `prepare_vbm_inputs(subjects: List[str], derivatives_dir: Path, tissue: str) -> Dict`
  - Collect GM/WM probability maps from anatomical preprocessing
  - Verify all subjects processed successfully
  - Create subject list file
  - Return dict with: subject_files, missing_subjects, tissue_type
- [ ] Implement `create_study_template(subject_files: List[Path], output_dir: Path, config: Dict) -> Path`
  - Option 1: Use fslvbm_1_prepare for study-specific template
  - Option 2: Return path to MNI152 template
  - Controlled by config['analysis']['vbm']['template']
  - Log which approach was used
- [ ] Implement `normalize_to_template(subject_files: List[Path], template: Path, output_dir: Path, smooth_fwhm: float) -> Tuple[Path, List[Path]]`
  - Warp all subjects to template (FNIRT)
  - Apply smoothing (configurable FWHM)
  - Modulate by Jacobian determinants
  - Concatenate into 4D volume
  - Return 4D file path and list of individual warped files
- [ ] Add progress logging and estimated time reporting

#### Task 2.2: VBM Statistical Analysis
**Status**: ⏳ Not Started
**Estimated Time**: 2-3 hours

**File**: `neurovrai/analysis/vbm.py` (continued)

- [ ] Implement `run_vbm_randomise(input_4d: Path, design_mat: Path, contrasts: Path, output_dir: Path, config: Dict) -> Dict`
  - Call `fsl_helpers.run_randomise()` with VBM-specific settings
  - Apply mask (brain mask or explicit GM/WM mask)
  - Generate TFCE-corrected statistical maps
  - Extract significant clusters
  - Return dict with output paths and statistics
- [ ] Implement `generate_vbm_cluster_table(tfce_results: Dict, atlas: str, output_dir: Path) -> pd.DataFrame`
  - Use `fsl_helpers.generate_cluster_table()`
  - Add tissue-specific information
  - Calculate cluster volumes in mm³
  - Save detailed CSV with all cluster info
- [ ] Implement `run_vbm_analysis(subjects: List[str], design_mat: Path, contrasts: Path, derivatives_dir: Path, output_dir: Path, config: Dict) -> Dict`
  - High-level wrapper combining all VBM steps
  - Preparation → Template → Normalization → Statistics
  - Generate all outputs in organized directory structure
  - Return comprehensive results dict

#### Task 2.3: VBM Quality Control
**Status**: ⏳ Not Started
**Estimated Time**: 2-3 hours

**File**: `neurovrai/analysis/vbm.py` (continued)

- [ ] Implement `create_vbm_qc_report(vbm_results: Dict, output_dir: Path) -> Path`
  - Visualize study template or MNI152 template used
  - Show registration quality across subjects (checkerboard overlays)
  - Display mean images per group
  - Plot distribution of tissue volumes per group
  - Create summary statistics table
  - Generate HTML report
  - Return path to HTML file
- [ ] Create visualization helpers:
  - [ ] `plot_registration_quality()` - Sample subjects overlaid on template
  - [ ] `plot_group_means()` - Mean tissue probability per group
  - [ ] `plot_tissue_distributions()` - Boxplots of total tissue volume

---

### Phase 3: TBSS Implementation

#### Task 3.1: TBSS Data Preparation
**Status**: ⏳ Not Started
**Estimated Time**: 3-4 hours

**File**: `neurovrai/analysis/tbss.py`

- [ ] Implement `prepare_tbss_inputs(subjects: List[str], derivatives_dir: Path, metric: str) -> Dict`
  - Collect FA (or other metric) maps from DWI preprocessing
  - Verify data quality and completeness
  - Create subject list file
  - Return dict with: subject_files, missing_subjects, metric_type
- [ ] Implement `run_tbss_preproc(subject_files: List[Path], output_dir: Path) -> Path`
  - Wrapper for `tbss_1_preproc`
  - Erode FA images slightly
  - Align to common space
  - Return FA directory path
- [ ] Implement `run_tbss_registration(fa_dir: Path, template: str, output_dir: Path, config: Dict) -> Path`
  - Wrapper for `tbss_2_reg`
  - Option 1: Register to FMRIB58_FA template
  - Option 2: Create study-specific target (find most representative subject)
  - Controlled by config['analysis']['tbss']['template']
  - Run nonlinear registration for all subjects
  - Return target template path

#### Task 3.2: TBSS Skeleton Creation
**Status**: ⏳ Not Started
**Estimated Time**: 2-3 hours

**File**: `neurovrai/analysis/tbss.py` (continued)

- [ ] Implement `run_tbss_postreg(fa_dir: Path, threshold: float, output_dir: Path) -> Tuple[Path, Path]`
  - Wrapper for `tbss_3_postreg`
  - Create mean FA image across all subjects
  - Generate mean FA skeleton
  - Apply threshold (default 0.2, configurable)
  - Return paths to mean_FA and skeleton
- [ ] Implement `run_tbss_projection(fa_dir: Path, skeleton: Path, output_dir: Path) -> Path`
  - Wrapper for `tbss_4_prestats`
  - Project all subjects' FA onto skeleton
  - Create 4D skeletonized data (all_FA_skeletonised.nii.gz)
  - Return path to 4D file

#### Task 3.3: TBSS Statistical Analysis
**Status**: ⏳ Not Started
**Estimated Time**: 3-4 hours

**File**: `neurovrai/analysis/tbss.py` (continued)

- [ ] Implement `run_tbss_randomise(skeleton_4d: Path, design_mat: Path, contrasts: Path, skeleton_mask: Path, output_dir: Path, config: Dict) -> Dict`
  - Call `fsl_helpers.run_randomise()` with TBSS-specific settings
  - Use mean FA skeleton as mask
  - Generate TFCE-corrected statistical maps
  - Extract significant clusters on skeleton
  - Return dict with output paths and statistics
- [ ] Implement `tbss_fill_stats(tfce_results: Path, mean_fa: Path, skeleton: Path, output_dir: Path) -> Path`
  - "Thicken" skeleton results for visualization
  - Use `tbss_fill` to project results into local tract
  - Create filled statistical maps
  - Return path to filled maps
- [ ] Implement `generate_tbss_cluster_table(tfce_results: Dict, atlas: str, output_dir: Path) -> pd.DataFrame`
  - Use JHU white matter atlas for tract labels
  - Report cluster details on skeleton
  - Include nearest tract labels
  - Save detailed CSV

#### Task 3.4: TBSS Multi-Modal Support
**Status**: ⏳ Not Started
**Estimated Time**: 2-3 hours

**File**: `neurovrai/analysis/tbss.py` (continued)

- [ ] Implement `project_metric_to_skeleton(metric_files: List[Path], fa_skeleton: Path, skeleton_mask: Path, metric_name: str, output_dir: Path) -> Path`
  - Project MD, AD, RD, or other metrics onto FA-derived skeleton
  - Use FA registration transforms (already computed)
  - Create 4D skeletonized data for new metric
  - Return path to metric 4D skeleton file
- [ ] Implement `run_multimodal_tbss(subjects: List[str], metrics: List[str], derivatives_dir: Path, design_mat: Path, contrasts: Path, output_dir: Path, config: Dict) -> Dict`
  - High-level wrapper for multi-metric TBSS
  - Run TBSS pipeline for FA first (creates skeleton)
  - Project additional metrics (MD, AD, RD, etc.) onto FA skeleton
  - Run randomise for each metric
  - Generate cluster tables for each
  - Return comprehensive results dict with all metrics

#### Task 3.5: TBSS Quality Control
**Status**: ⏳ Not Started
**Estimated Time**: 2-3 hours

**File**: `neurovrai/analysis/tbss.py` (continued)

- [ ] Implement `create_tbss_qc_report(tbss_results: Dict, output_dir: Path) -> Path`
  - Show registration quality (sample subjects on template)
  - Display mean FA image and skeleton
  - Show group differences overlay on skeleton
  - Plot FA distribution along skeleton per group
  - Create cluster statistics table
  - Generate HTML report
  - Return path to HTML file
- [ ] Create visualization helpers:
  - [ ] `plot_skeleton_overlay()` - Skeleton on mean FA background
  - [ ] `plot_registration_montage()` - Multiple subjects aligned
  - [ ] `plot_skeleton_profile()` - Mean metric values along skeleton per group
  - [ ] `plot_significant_clusters()` - Overlay TFCE results on skeleton

---

### Phase 4: Cluster Reporting & Visualization

#### Task 4.1: Cluster Analysis Utilities
**Status**: ⏳ Not Started
**Estimated Time**: 3-4 hours

**File**: `neurovrai/analysis/utils/cluster_analysis.py`

- [ ] Implement `extract_clusters(tfce_map: Path, threshold: float, min_size: int) -> List[Dict]`
  - Parse FSL cluster output
  - Apply cluster size threshold
  - Extract peak coordinates (MNI space)
  - Extract peak statistical values
  - Return list of cluster dicts
- [ ] Implement `add_anatomical_labels(clusters: List[Dict], atlas: str) -> List[Dict]`
  - Use FSL atlasquery to get labels
  - Support multiple atlases (Harvard-Oxford, JHU, AAL)
  - Add labels and overlap percentages to cluster dicts
  - Return enhanced cluster list
- [ ] Implement `cluster_to_table(clusters: List[Dict], output_file: Path) -> pd.DataFrame`
  - Create publication-ready table
  - Columns: Cluster ID, Size (voxels), Size (mm³), Peak T/F-stat, MNI Coordinates (x,y,z), P-value (corrected), Anatomical Label, % Overlap
  - Sort by cluster size or peak statistic
  - Save as CSV
  - Return DataFrame
- [ ] Implement `generate_cluster_report(clusters: List[Dict], statistical_maps: Dict, output_file: Path) -> Path`
  - Create HTML report with embedded cluster details
  - Include slice views of significant clusters
  - Add anatomical context
  - Interactive table with sortable columns
  - Return path to HTML file

#### Task 4.2: Statistical Map Visualization
**Status**: ⏳ Not Started
**Estimated Time**: 4-5 hours

**File**: `neurovrai/analysis/utils/visualization.py`

- [ ] Implement `plot_statistical_maps(stat_map: Path, background: Path, threshold: float, output_file: Path, **kwargs) -> Path`
  - Overlay TFCE results on anatomical template
  - Create multi-slice montages (axial, sagittal, coronal)
  - Use proper colormap (red-yellow for positive, blue-cyan for negative)
  - Add colorbar with statistical values
  - Save high-resolution PNG/SVG
  - Return path to saved file
- [ ] Implement `create_glass_brain(stat_map: Path, threshold: float, output_file: Path) -> Path`
  - Generate glass brain visualization using nilearn
  - Project 3D results onto brain surface
  - Create multiple views (left, right, top, bottom)
  - Return path to saved file
- [ ] Implement `plot_effect_sizes(data_4d: Path, mask: Path, clusters: List[Dict], group_labels: List[int], output_file: Path) -> Path`
  - Extract values from significant clusters
  - Create boxplots or violin plots per group
  - Show individual subject points
  - Add statistical annotations (p-values, effect sizes)
  - Save publication-ready figure
  - Return path to saved file
- [ ] Implement `create_montage(images: List[Path], titles: List[str], output_file: Path, layout: str) -> Path`
  - Combine multiple visualizations into single figure
  - Support grid layouts
  - Add titles and annotations
  - Return path to montage image

---

### Phase 5: Integration & Testing

#### Task 5.1: Command-Line Interface
**Status**: ⏳ Not Started
**Estimated Time**: 3-4 hours

**File**: `neurovrai/analysis/cli.py`

- [ ] Implement VBM CLI:
  ```bash
  neurovrai-analysis vbm \
    --subjects sub-001 sub-002 sub-003 \
    --derivatives /path/to/derivatives \
    --design design.mat \
    --contrast contrast.con \
    --output /path/to/analysis/vbm \
    --config config.yaml
  ```
- [ ] Implement TBSS CLI:
  ```bash
  neurovrai-analysis tbss \
    --subjects sub-001 sub-002 sub-003 \
    --derivatives /path/to/derivatives \
    --design design.mat \
    --contrast contrast.con \
    --metrics FA MD AD RD \
    --output /path/to/analysis/tbss \
    --config config.yaml
  ```
- [ ] Add common options:
  - `--dry-run` - Show what would be done without executing
  - `--verbose` - Detailed logging
  - `--n-jobs` - Parallel processing where applicable
- [ ] Add helpful error messages and validation
- [ ] Create `--help` documentation for all commands

#### Task 5.2: Comprehensive Testing
**Status**: ⏳ Not Started
**Estimated Time**: 4-6 hours (ongoing)

**Test Plan**:
- [ ] **Unit Tests**:
  - [ ] Test design matrix validation
  - [ ] Test contrast validation
  - [ ] Test cluster extraction
  - [ ] Test atlas labeling
- [ ] **Integration Tests**:
  - [ ] Test complete VBM pipeline on small dataset
  - [ ] Test complete TBSS pipeline on small dataset
  - [ ] Test multi-modal TBSS (FA + MD)
- [ ] **Real-World Validation**:
  - [ ] Run VBM on actual research dataset
  - [ ] Compare results with FSL GUI workflows
  - [ ] Verify cluster tables against manual inspection
  - [ ] Validate statistical maps visually
- [ ] **Edge Cases**:
  - [ ] Missing subjects
  - [ ] Invalid design matrix
  - [ ] Failed preprocessing outputs
  - [ ] Memory constraints with large datasets

#### Task 5.3: Documentation
**Status**: ⏳ Not Started
**Estimated Time**: 4-5 hours

**Files to Create**:

- [ ] **`docs/analysis/VBM_GUIDE.md`**:
  - Introduction to VBM
  - Step-by-step workflow
  - Design matrix creation
  - Running the analysis
  - Interpreting results
  - Common issues and troubleshooting
  - Example commands and outputs

- [ ] **`docs/analysis/TBSS_GUIDE.md`**:
  - Introduction to TBSS
  - Step-by-step workflow
  - Multi-modal analysis
  - Running the analysis
  - Interpreting skeleton results
  - Common issues and troubleshooting
  - Example commands and outputs

- [ ] **`docs/analysis/DESIGN_MATRIX_GUIDE.md`**:
  - FSL design matrix format explanation
  - Creating .mat files (Text, Glm_gui, Python)
  - Creating .con files (contrasts)
  - Common designs:
    - Two-group comparison
    - One-sample t-test
    - Regression with continuous variable
    - ANCOVA (group + covariates)
  - Troubleshooting design matrix issues

- [ ] **`docs/analysis/CLUSTER_INTERPRETATION.md`**:
  - Understanding TFCE
  - Reading cluster tables
  - Anatomical labeling
  - Effect size interpretation
  - Publication guidelines

- [ ] **Update `README.md`**:
  - Add analysis section
  - Link to analysis guides
  - Quick start examples

---

## 🎨 Design Decisions (Confirmed)

### Templates
- **VBM**: Both MNI152 and study-specific template options available
  - Default: MNI152 (faster, standard space)
  - Study-specific: Better for homogeneous samples, more accurate
- **TBSS**: Both FMRIB58_FA and study-specific options available
  - Default: FMRIB58_FA (standard space)
  - Study-specific: Better for age-specific or clinical populations

### Smoothing
- **Default**: 6mm FWHM for VBM
- **Configurable**: User can adjust via config
- **TBSS**: No smoothing (inherent in skeleton projection)

### Anatomical Atlases
**Customizable atlas selection with implemented options**:
- **Cortical**: Harvard-Oxford cortical atlas
- **Subcortical**: Harvard-Oxford subcortical atlas
- **White Matter Tracts**: JHU ICBM-DTI-81 atlas
- **White Matter Labels**: JHU white matter tractography atlas
- **Whole Brain**: AAL3 (Automated Anatomical Labeling)

Users can select primary atlas in config and override per analysis.

### TBSS Metrics
- **User-specified**: Metrics provided as list in config or CLI
- **Multi-modal support**: Can run multiple metrics in single analysis
- **Example**: `metrics: ["FA", "MD", "AD", "RD"]` runs all four

---

## 🚀 Future Goals (Post-MVP)

### Enhanced Analysis Methods
- **Longitudinal VBM**: Paired comparisons, rate of change
- **TBSS with Multiple Time Points**: Longitudinal skeleton analysis
- **Advanced Contrasts**: F-tests for multiple groups, interaction terms
- **Non-parametric Combination (NPC)**: Combine results from multiple modalities
- **Bayesian Analysis**: Alternative to frequentist randomise

### Additional Modalities
- **SBM (Surface-Based Morphometry)**: Cortical thickness, surface area (FreeSurfer)
- **Functional Connectivity**: Seed-based correlation, ICA, graph theory
- **ASL Group Analysis**: Perfusion group comparisons, CBF normalization

### Visualization Enhancements
- **Interactive 3D Viewers**: Web-based result exploration
- **Automated Report Generation**: Complete PDF/HTML reports with all QC and results
- **Publication-Ready Figures**: One-click generation of manuscript figures

### Performance Optimization
- **Parallel Processing**: Multi-threading for randomise, registration
- **Memory Optimization**: Chunked processing for large datasets
- **GPU Acceleration**: Explore GPU-based permutation testing

### Integration Features
- **BIDS Stats Models**: Support for BIDS statistical model specification
- **Result Sharing**: Export to NeuroVault, standardized formats
- **Meta-Analysis Support**: Tools for combining studies

---

## 📊 Success Metrics

### Phase Completion Criteria

**Phase 1 Complete When**:
- ✅ All infrastructure code written and tested
- ✅ Design matrix validation working
- ✅ Config system extended and documented

**Phase 2 Complete When**:
- ✅ VBM pipeline runs end-to-end on test data
- ✅ Statistical maps and cluster tables generated
- ✅ QC report created

**Phase 3 Complete When**:
- ✅ TBSS pipeline runs end-to-end on test data
- ✅ Multi-modal analysis working (FA + at least one other metric)
- ✅ Skeleton-based statistical maps generated

**Phase 4 Complete When**:
- ✅ Cluster tables have anatomical labels
- ✅ All visualization functions produce publication-quality output
- ✅ HTML reports generated successfully

**Phase 5 Complete When**:
- ✅ CLI functional and documented
- ✅ All tests passing
- ✅ Documentation complete and reviewed
- ✅ Real-world validation successful

### Project Complete When
- ✅ VBM and TBSS both production-ready
- ✅ Results match FSL GUI workflows
- ✅ Documentation comprehensive
- ✅ Successfully used on actual research project
- ✅ All edge cases handled gracefully

---

## ⚠️ Known Challenges & Mitigation

### Challenge 1: FSL Version Compatibility
**Issue**: Different FSL versions may have slightly different randomise behavior
**Mitigation**: Document tested FSL version, add version checking, test on multiple versions

### Challenge 2: Memory Requirements
**Issue**: Large datasets (many subjects, high resolution) may exceed RAM
**Mitigation**: Add memory usage estimation, implement chunked processing, provide guidance on HPC usage

### Challenge 3: Template Registration Quality
**Issue**: Poor registration can bias results
**Mitigation**: Comprehensive QC for registration, allow manual inspection, provide re-registration tools

### Challenge 4: Statistical Power
**Issue**: Small sample sizes may lack power for permutation testing
**Mitigation**: Add sample size estimation tools, provide guidance on minimum samples, support variance smoothing

### Challenge 5: Multiple Comparisons
**Issue**: Testing multiple metrics/contrasts inflates false positive rate
**Mitigation**: Document proper correction approaches, support Bonferroni correction, enable NPC for future

---

## 📝 Development Notes

### Session Tracking

**Session 1**: (Planned)
- Create module structure
- Implement design matrix validation
- Set up config system

**Session 2**: (Planned)
- Implement VBM data preparation
- Create template handling

**Session 3**: (Planned)
- Complete VBM statistical analysis
- Implement VBM QC

...continue as development progresses...

### Important Conventions
- **Logging**: Use Python logging module, write to `logs/analysis_<timestamp>.log`
- **Error Handling**: Graceful failures with informative messages
- **Type Hints**: Full type annotations for all public functions
- **Docstrings**: Google-style docstrings for all modules, classes, functions
- **Testing**: Pytest for unit and integration tests

### Resources
- **FSL Randomise**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise
- **TBSS**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS
- **VBM**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLVBM
- **TFCE**: Smith & Nichols (2009), "Threshold-free cluster enhancement"

---

## 📞 Questions for Research Team

1. **Sample Size**: What is the expected range of subjects per analysis? (Important for randomise parameters)
2. **Covariates**: Will analyses need to control for age, sex, education? (Important for design matrix templates)
3. **Multiple Comparisons**: How should we handle multiple contrasts in single analysis?
4. **Output Preferences**: Preferred format for cluster tables? (CSV, Excel, LaTeX?)
5. **Atlas Preferences**: Any institutional preference for anatomical atlas?

---

**Next Step**: Begin Phase 1 implementation - Module structure and design matrix handling
