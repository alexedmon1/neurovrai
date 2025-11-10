# MRI Preprocessing Pipeline Refactoring Plan

## Overview
Refactor the existing MRI preprocessing codebase to:
1. Separate processing logic from file paths and sequence names
2. Create clean CLI and Python API interfaces
3. Use YAML configuration for study-specific parameters
4. Enable both interactive and batch processing workflows
5. Implement multi-echo fMRI preprocessing with TEDANA
6. **Implement transformation reuse**: Compute T1w→MNI transforms once, reuse across all modalities
7. **Add probtrackx2 for structural connectivity**: Enable tractography and connectivity matrices
8. **Plan for analysis pipelines**: Framework for post-preprocessing analyses (connectivity, ReHo, fALFF, VBM, TBSS) to be implemented after core preprocessing is complete

## Guiding Principles
- **Clean break**: No backward compatibility with old scripts (archive them)
- **FSL-based**: Use FSL for speed (not FreeSurfer for coregistration)
- **Config-driven**: Sequence names and study paths in YAML, not hardcoded
- **Dual interface**: Both CLI and Python API support
- **DRY (Don't Repeat Yourself)**: Compute transformations once, reuse everywhere
- **Dependency management**: Anatomical workflow runs first, others reuse its outputs
- **Commit often**: Git commit after each completed step

---

## Phase 1: Project Structure Setup

### Step 1.1: Create new directory structure
- [ ] Create `mri_preprocess/` package directory
- [ ] Create subdirectories: `workflows/`, `utils/`, `converters/`, `analysis/`
- [ ] Create `configs/` directory for YAML files
- [ ] Create `scripts/` directory for high-level orchestration
- [ ] Add `__init__.py` files to make proper Python package
- **Commit**: "Create new package structure"

### Step 1.2: Archive old code
- [ ] Move `anat/`, `dwi/`, `rest/`, `myelin/` to `archive/` directory
- [ ] Move `dicom/`, `analysis/` to `archive/` as well
- [ ] Keep `archive/` structure: `archive/anat/`, `archive/dwi/`, etc.
- [ ] Update `.gitignore` to exclude `archive/` from future changes
- **Commit**: "Archive original code maintaining directory structure"

### Step 1.3: Update pyproject.toml
- [ ] Add new dependencies: `click` (CLI), `pyyaml` (configs), `python-dotenv` (optional)
- [ ] Define package entry points for CLI commands
- [ ] Set package name and console scripts
- **Commit**: "Update pyproject.toml with CLI dependencies and entry points"

---

## Phase 2: Configuration System

### Step 2.1: Create default configuration
- [ ] Create `configs/default.yaml` with all preprocessing parameters
- [ ] Include: FSL paths, templates, sequence patterns, workflow defaults
- [ ] Document each parameter with inline comments
- **Commit**: "Add default.yaml configuration file"

### Step 2.2: Create study configuration example
- [ ] Create `configs/example_study.yaml` showing how to override defaults
- [ ] Include study-specific paths, sequence mappings, subject lists
- [ ] Add example for custom parameters per modality
- **Commit**: "Add example study configuration"

### Step 2.3: Build config loader
- [ ] Create `mri_preprocess/config.py`
- [ ] Implement YAML loading with inheritance (study overrides defaults)
- [ ] Add environment variable substitution (`${VAR}` syntax)
- [ ] Add config validation function
- **Commit**: "Implement configuration loader with validation"

---

## Phase 3: Utility Modules

### Step 3.1: File finder utility
- [ ] Create `mri_preprocess/utils/file_finder.py`
- [ ] Implement sequence name matching (regex-based from config)
- [ ] Add functions: `find_by_modality()`, `find_subject_files()`, `match_sequence()`
- [ ] Remove all hardcoded sequence names from matching logic
- **Commit**: "Add file finder utility with config-based sequence matching"

### Step 3.2: BIDS utilities
- [ ] Create `mri_preprocess/utils/bids.py`
- [ ] Functions for BIDS-like directory layout
- [ ] Path builders: `get_subject_dir()`, `get_modality_dir()`, etc.
- [ ] No `os.chdir()` - all absolute path handling
- **Commit**: "Add BIDS utilities for path management"

### Step 3.3: Workflow helpers
- [ ] Create `mri_preprocess/utils/workflow.py`
- [ ] Nipype workflow builder helpers
- [ ] Common node configurations (FSL defaults, output types)
- [ ] Logging setup utilities
- **Commit**: "Add workflow helper utilities"

### Step 3.4: Transformation registry
- [ ] Create `mri_preprocess/utils/transforms.py`
- [ ] Class: `TransformRegistry` to save/load transformation files
- [ ] Methods: `save_transform()`, `load_transform()`, `check_transform_exists()`
- [ ] Standard naming: `{subject}_t1w_to_mni_affine.mat`, `{subject}_t1w_to_mni_warp.nii.gz`
- [ ] Enable transform reuse across workflows (DRY principle)
- **Commit**: "Add transformation registry for reusing computed transforms"

---

## Phase 4: DICOM Conversion Module

### Step 4.1: Refactor DICOM converter
- [ ] Create `mri_preprocess/converters/dicom.py`
- [ ] Refactor `dcm2niix` class to take explicit paths (no `os.chdir()`)
- [ ] Use config for sequence mappings instead of hardcoded
- [ ] Return structured output (dict of modality → files)
- **Commit**: "Refactor DICOM converter with config-based routing"

### Step 4.2: Add DICOM converter tests
- [ ] Create test for sequence detection logic
- [ ] Test with sample DICOM headers
- [ ] Validate file organization output
- **Commit**: "Add DICOM converter unit tests"

---

## Phase 5: Anatomical Preprocessing Workflow

### Step 5.1: Refactor anatomical workflow
- [ ] Create `mri_preprocess/workflows/anatomical.py`
- [ ] Class: `AnatomicalPreprocessor(t1w_file, output_dir, config)`
- [ ] Remove `os.chdir()`, use absolute paths
- [ ] Build workflow from config parameters
- [ ] **Save transformation files**: T1w→MNI affine (.mat) and warp field (.nii.gz)
- [ ] Use `TransformRegistry` to save transforms for reuse by other workflows
- [ ] Method: `build_workflow()` returns Nipype workflow
- [ ] Method: `run()` executes workflow with config settings
- **Commit**: "Refactor anatomical preprocessing workflow with transform saving"

### Step 5.2: Add FreeSurfer wrapper (optional)
- [ ] Create `FreeSurferReconAll` class in anatomical.py
- [ ] Config-driven: only runs if `freesurfer.enabled: true`
- [ ] Takes T1w, optional T2w from config
- **Commit**: "Add FreeSurfer recon-all wrapper (configurable)"

### Step 5.3: Test anatomical workflow
- [ ] Create simple test with mock data
- [ ] Verify workflow graph generation
- **Commit**: "Add anatomical workflow tests"

---

## Phase 6: Diffusion Preprocessing Workflow

### Step 6.1: Refactor DTI workflow
- [ ] Create `mri_preprocess/workflows/diffusion.py`
- [ ] Class: `DiffusionPreprocessor(dwi_files, bvals, bvecs, output_dir, config, transform_registry)`
- [ ] Handle multi-shell and single-shell cases
- [ ] Shell merging logic from config
- [ ] Eddy parameters (acqp, index) from config
- [ ] Compute DWI→T1w transformation
- [ ] **Reuse T1w→MNI transforms** from anatomical workflow (load via `TransformRegistry`)
- [ ] Concatenate transformations: DWI→T1w→MNI (single interpolation step)
- **Commit**: "Refactor diffusion preprocessing workflow with transform reuse"

### Step 6.2: Add BEDPOSTX
- [ ] Add optional BEDPOSTX step (config-controlled)
- [ ] GPU/CUDA settings from config
- [ ] Output fiber orientation distributions for probabilistic tractography
- **Commit**: "Add BEDPOSTX to diffusion workflow"

### Step 6.3: Add probtrackx2 tractography
- [ ] Class: `ProbabilisticTractography` for probtrackx2 wrapper
- [ ] Seed-based tractography from ROIs/masks
- [ ] Network mode for structural connectivity matrices
- [ ] Config options: number of samples, curvature threshold, loopcheck
- [ ] Support for waypoint/exclusion/termination masks
- [ ] Output: probability maps, waytotal stats, connectivity matrices
- [ ] **Note**: Requires BEDPOSTX output and T1w→MNI transforms for ROI warping
- **Commit**: "Add probtrackx2 for structural connectivity analysis"

---

## Phase 7: Functional Preprocessing Workflow (Multi-Echo + TEDANA)

### Step 7.1: Create base functional workflow
- [ ] Create `mri_preprocess/workflows/functional.py`
- [ ] Class: `FunctionalPreprocessor(func_files, t1w_file, output_dir, config, transform_registry)`
- [ ] Detect single-echo vs multi-echo from file count
- [ ] Basic preprocessing: reorient, skull strip
- **Commit**: "Create base functional preprocessing workflow"

### Step 7.2: Implement multi-echo TEDANA integration
- [ ] Add multi-echo detection logic
- [ ] Per-echo motion correction (middle echo as reference)
- [ ] Apply motion transforms to all echoes
- [ ] TEDANA node with config parameters (tedpca, fittype, etc.)
- [ ] Output: optimally combined and denoised time series
- **Commit**: "Implement multi-echo preprocessing with TEDANA"

### Step 7.3: Add post-TEDANA processing
- [ ] Structural coregistration (func → T1w)
- [ ] **Reuse T1w→MNI transforms** from anatomical workflow (load via `TransformRegistry`)
- [ ] Concatenate transforms: func→T1w→MNI (single interpolation step)
- [ ] Apply combined transformation to TEDANA output
- [ ] Spatial smoothing (post-TEDANA)
- [ ] ICA-AROMA on denoised data
- [ ] ACompCor nuisance regression (using T1w segmentation from anatomical workflow)
- [ ] Temporal filtering (bandpass)
- **Commit**: "Add post-TEDANA processing pipeline with transform reuse"

### Step 7.4: Handle single-echo fallback
- [ ] Single-echo path: standard preprocessing without TEDANA
- [ ] Config flag: `functional.multi_echo.enabled: false` overrides auto-detect
- [ ] Ensure both paths work from same class
- **Commit**: "Add single-echo preprocessing fallback"

---

## Phase 8: Myelin Mapping Workflow

### Step 8.1: Refactor myelin workflow
- [ ] Create `mri_preprocess/workflows/myelin.py`
- [ ] Class: `MyelinMapper(t1w_file, t2w_file, output_dir, config, transform_registry)`
- [ ] Coregister T2w to T1w space
- [ ] Compute T1w/T2w ratio in native T1w space
- [ ] **Reuse T1w→MNI transforms** from anatomical workflow (no recomputation!)
- [ ] Apply T1w→MNI transform to myelin map
- **Commit**: "Refactor myelin mapping workflow with transform reuse"

---

## Phase 9: CLI Interface

### Step 9.1: Create main CLI entry point
- [ ] Create `mri_preprocess/cli.py`
- [ ] Use Click for CLI framework
- [ ] Main command group: `mri-preprocess`
- [ ] Global options: `--config`, `--verbose`, `--dry-run`
- **Commit**: "Create main CLI entry point with Click"

### Step 9.2: Add subcommands
- [ ] `mri-preprocess convert` - DICOM to NIfTI
- [ ] `mri-preprocess anat` - anatomical preprocessing
- [ ] `mri-preprocess dwi` - diffusion preprocessing
- [ ] `mri-preprocess func` - functional preprocessing (auto-detects multi-echo)
- [ ] `mri-preprocess myelin` - myelin mapping
- [ ] Each command: `--subject`, `--bids-dir`, `--output-dir` options
- **Commit**: "Add preprocessing subcommands to CLI"

### Step 9.3: Add pipeline orchestration command
- [ ] `mri-preprocess run` - run full pipeline
- [ ] Options: `--steps` (all, convert, anat, dwi, func, myelin)
- [ ] `--subjects` (single or list)
- [ ] Calls subcommands in sequence
- **Commit**: "Add pipeline orchestration command"

---

## Phase 10: Orchestration Layer

### Step 10.1: Create pipeline orchestrator
- [ ] Create `mri_preprocess/orchestrator.py`
- [ ] Class: `PipelineOrchestrator(config)`
- [ ] Initialize `TransformRegistry` for the study
- [ ] **Workflow dependency management**: anatomical must run before func/dwi/myelin
- [ ] Methods: `run_subject()`, `run_batch()`, `run_step()`
- [ ] Check for required transforms before running dependent workflows
- [ ] Handles subject iteration, step selection, error tracking
- **Commit**: "Create pipeline orchestrator with dependency management"

### Step 10.2: Add batch processing script
- [ ] Create `scripts/batch_process.py`
- [ ] Reads subject list from config or file
- [ ] Parallel execution support (joblib or multiprocessing)
- [ ] Status tracking (resume failed runs)
- [ ] Progress logging
- **Commit**: "Add batch processing script"

---

## Phase 11: Analysis Pipelines (Post-Preprocessing)

**Note**: This phase covers analysis workflows that run AFTER preprocessing is complete. These include group-level statistical analyses, connectivity analyses, and other higher-level analyses.

### Step 11.1: Refactor cluster analysis utilities
- [ ] Create `mri_preprocess/analysis/statistics.py`
- [ ] Move cluster analysis logic from original code
- [ ] FSL randomise result aggregation
- [ ] Config-driven input paths
- **Commit**: "Refactor statistical analysis utilities"

### Step 11.2: Plan structural connectivity analysis (Future)
- [ ] **TODO**: Structural connectivity matrix generation from probtrackx2
- [ ] **TODO**: Graph theory metrics (network efficiency, modularity, etc.)
- [ ] **TODO**: Group-level connectivity analyses
- [ ] **TODO**: Integration with network analysis packages (NetworkX, BCT)
- **Note**: Implementation deferred to future phase after core preprocessing complete

### Step 11.3: Plan functional connectivity and resting-state metrics (Future)
- [ ] **TODO**: Seed-based correlation analysis
- [ ] **TODO**: ROI-to-ROI connectivity matrices
- [ ] **TODO**: Dual regression (for ICA-based analyses)
- [ ] **TODO**: Dynamic connectivity (sliding window)
- [ ] **TODO**: ReHo (Regional Homogeneity) - Kendall's coefficient of concordance
- [ ] **TODO**: fALFF (fractional Amplitude of Low Frequency Fluctuations)
- [ ] **TODO**: ALFF (Amplitude of Low Frequency Fluctuations)
- [ ] **TODO**: Integration with AFNI/DPABI tools for ReHo/fALFF calculation
- **Note**: Implementation deferred to future phase after core preprocessing complete

### Step 11.4: Plan group-level VBM/TBSS analysis (Future)
- [ ] **TODO**: TBSS pipeline wrapper (FA, MD, RD, AD analyses)
- [ ] **TODO**: VBM pipeline wrapper (GMV, cortical thickness)
- [ ] **TODO**: Design matrix generation from demographic files
- [ ] **TODO**: FSL randomise integration with multiple comparison correction
- **Note**: Implementation deferred to future phase after core preprocessing complete

### Step 11.5: Create analysis pipeline placeholder
- [ ] Document planned analysis workflows in README
- [ ] Create `mri_preprocess/analysis/connectivity.py` stub file
- [ ] Create `mri_preprocess/analysis/resting_state.py` stub file (ReHo, fALFF, ALFF)
- [ ] Create `mri_preprocess/analysis/vbm_tbss.py` stub file
- [ ] Add TODOs and docstrings describing future functionality
- **Commit**: "Add analysis pipeline stubs and documentation"

---

## Phase 12: Documentation and Examples

### Step 12.1: Update README
- [ ] Installation instructions with `uv`
- [ ] Quick start guide
- [ ] CLI usage examples
- [ ] Python API examples
- **Commit**: "Update README with usage documentation"

### Step 12.2: Create example workflows
- [ ] `examples/single_subject.py` - Python API example
- [ ] `examples/batch_subjects.sh` - CLI batch example
- [ ] `examples/interactive.ipynb` - Jupyter notebook example
- **Commit**: "Add example workflows"

### Step 12.3: Update CLAUDE.md
- [ ] Document new architecture
- [ ] CLI commands
- [ ] Config file structure
- [ ] Development guidelines
- **Commit**: "Update CLAUDE.md for refactored codebase"

---

## Phase 13: Testing and Validation

### Step 13.1: Integration testing
- [ ] Test full pipeline with sample data
- [ ] Verify multi-echo TEDANA workflow
- [ ] Test config overrides
- [ ] Validate output file structures
- **Commit**: "Add integration tests"

### Step 13.2: Compare outputs with original code
- [ ] Run same subject through old and new pipelines
- [ ] Compare preprocessing outputs
- [ ] Document any differences
- **Commit**: "Validate outputs against original implementation"

---

## Phase 14: Final Cleanup

### Step 14.1: Code review and polish
- [ ] Remove any remaining hardcoded paths
- [ ] Ensure consistent error handling
- [ ] Add docstrings to all public functions
- [ ] Type hints where appropriate
- **Commit**: "Code cleanup and documentation"

### Step 14.2: Performance optimization
- [ ] Profile workflow execution
- [ ] Optimize file I/O
- [ ] Tune parallel processing settings
- **Commit**: "Performance optimizations"

---

## Success Criteria

✅ Full pipeline runs from single CLI command: `mri-preprocess run --config study.yaml --subject 001 --steps all`

✅ Multi-echo functional data processed with TEDANA automatically

✅ **T1w→MNI transformations computed once in anatomical workflow, reused by func/dwi/myelin**

✅ **Orchestrator enforces dependencies**: anatomical runs before other modalities

✅ **Diffusion tractography**: probtrackx2 integration for structural connectivity matrices

✅ No hardcoded sequence names in code (all in config)

✅ No hardcoded file paths in code (all in config or CLI args)

✅ No `os.chdir()` calls in workflow code

✅ Both interactive (single subject) and batch modes work

✅ Clean Python API for programmatic use

✅ Comprehensive config validation with helpful error messages

---

## Future Work (Post-Refactoring)

After core preprocessing is complete, the following analysis pipelines will be implemented:

🔮 **Structural connectivity analysis**: Connectivity matrices from probtrackx2, graph theory metrics

🔮 **Functional connectivity analysis**: Seed-based correlation, ROI-to-ROI matrices, dual regression, dynamic connectivity

🔮 **Resting-state metrics**: ReHo (regional homogeneity), fALFF/ALFF (amplitude of low-frequency fluctuations)

🔮 **Group-level morphometry**: TBSS pipeline for FA/diffusivity, VBM for gray matter volume

🔮 **Network analysis**: Integration with NetworkX/BCT for graph-theoretic measures

🔮 **Statistical workflows**: Design matrix generation, FSL randomise wrappers, multiple comparison correction

These will be added as Phase 15+ after validating the preprocessing pipeline.

---

## Notes

- Commit after EVERY completed step (not just phases)
- Test each module independently before moving to next phase
- Keep archive/ directory for reference but don't modify it
- Document any deviations from this plan in commit messages
- Analysis pipelines (Phase 11) marked as future work - implement after preprocessing validated
