# Neurovrai Development Roadmap

**Last Updated:** 2025-12-11
**Current Version:** 2.0.0-alpha

## Project Status Overview

### ✅ Production-Ready Modules

**Preprocessing (neurovrai/preprocess/):**
- ✅ Anatomical (T1w/T2w): N4 bias correction, BET, registration, segmentation
- ✅ DWI: Multi-shell/single-shell, TOPUP, GPU eddy, DTI, normalization
- ✅ Advanced Diffusion: DKI, NODDI (DIPY), AMICO (100x speedup)
- ✅ Functional: Multi-echo (TEDANA), single-echo (ICA-AROMA), ACompCor, bandpass, smoothing
- ✅ ASL: Motion correction, CBF quantification, M0 calibration, PVC

**Analysis (neurovrai/analysis/):**
- ✅ VBM: Voxel-Based Morphometry with FSL randomise or nilearn GLM
- ✅ TBSS: Tract-Based Spatial Statistics with FSL randomise
- ✅ Resting-State fMRI: ReHo, fALFF/ALFF, MELODIC group ICA
- ✅ Enhanced Reporting: Atlas-based cluster localization with HTML visualization

**Quality Control (neurovrai/preprocess/qc/):**
- ✅ DWI QC: TOPUP, motion, DTI metrics, skull stripping
- ✅ Anatomical QC: Skull stripping, segmentation, registration
- ✅ Functional QC: Motion, tSNR, DVARS, skull stripping
- ✅ ASL QC: Motion, CBF distributions, tSNR, skull stripping

### ⚠️ Incomplete / Not Production Ready

**FreeSurfer Integration:**
- ⚠️ Detection and extraction hooks only
- ❌ Anatomical→DWI transform pipeline (CRITICAL for tractography ROIs)
- ❌ FreeSurfer native space handling
- ❌ Transform quality control
- ❌ T1 validation (FreeSurfer vs preprocessing)
- **Status:** DO NOT enable `freesurfer.enabled = true`
- **Estimated Work:** 2-3 full sessions

### 🚧 Recent Completions (2025-12-11)

- ✅ **Functional Preprocessing Fix**: Moved BET to Phase 1 before TEDANA
- ✅ **ApplyXFM4D Interface**: Fixed MCFLIRT mat_file connectivity issue
- ✅ **Project Cleanup**: Organized scripts, removed old logs
- ✅ **Documentation**: Comprehensive session summaries and resolution docs

---

## 🔴 Critical / High Priority

### 1. Complete Functional Preprocessing Validation
**Priority:** IMMEDIATE
**Status:** Phase 1 validated, TEDANA running, Phase 2 pending
**Estimated Time:** 1-2 hours (waiting for TEDANA)

**Tasks:**
- [ ] Wait for TEDANA to complete
- [ ] Validate Phase 2 workflow (bandpass + smoothing)
- [ ] Verify directory structure and final outputs
- [ ] Run QC on complete pipeline
- [ ] Document final results
- [ ] Update CLAUDE.md with validated workflow

**Success Criteria:**
- All nodes complete without errors
- Outputs in correct directories (work/ vs derivatives/)
- QC metrics within expected ranges
- Workflow can resume from failures

**Dependencies:** None (currently running)

---

### 2. FreeSurfer Integration ⚠️ **CRITICAL GAP**
**Priority:** HIGH
**Status:** Partially implemented, not production ready
**Estimated Time:** 2-3 full development sessions

**Why Critical:**
- Explicitly marked as incomplete in CLAUDE.md
- Blocks FreeSurfer-based ROI extraction for tractography
- Required for cortical parcellation-based connectivity
- Users may assume it works (it doesn't)

**Missing Components:**

#### A. Anatomical→DWI Transform Pipeline
**Estimated Time:** 1 session

**Tasks:**
- [ ] Design transform application workflow
- [ ] Extract FreeSurfer T1 and validate against preprocessing T1
- [ ] Compute FreeSurfer→native anatomical transform
- [ ] Compose FreeSurfer→DWI transform chain
- [ ] Apply transforms to aparc+aseg labels
- [ ] Validate ROI alignment with DWI data

**Files to Modify:**
- `neurovrai/preprocess/utils/freesurfer_utils.py` (create)
- `neurovrai/preprocess/workflows/dwi_preprocess.py` (integrate)

#### B. FreeSurfer Native Space Handling
**Estimated Time:** 0.5 session

**Tasks:**
- [ ] Implement FreeSurfer coordinate system conversion
- [ ] Handle RAS vs LAS orientation differences
- [ ] Validate voxel-to-world transformations
- [ ] Add orientation checking utilities

#### C. Transform Quality Control
**Estimated Time:** 0.5 session

**Tasks:**
- [ ] Visual overlay QC (labels on DWI)
- [ ] Dice coefficient for validation ROIs
- [ ] Edge alignment metrics
- [ ] HTML QC report generation

**Files to Create:**
- `neurovrai/preprocess/qc/freesurfer_qc.py`

#### D. Documentation & Testing
**Estimated Time:** 0.5 session

**Tasks:**
- [ ] Document FreeSurfer requirements and setup
- [ ] Add usage examples
- [ ] Create test script with sample data
- [ ] Update CLAUDE.md to mark as production-ready

**Success Criteria:**
- FreeSurfer ROIs accurately aligned with DWI data
- Transform QC shows <2mm error for key regions
- All tests pass on validation dataset
- Documentation complete and tested

**Dependencies:**
- FreeSurfer recon-all outputs must exist
- Preprocessing T1w must be available

---

## 🟡 Medium Priority

### 3. Functional Preprocessing Enhancements
**Priority:** MEDIUM
**Status:** Core workflow validated, enhancements needed
**Estimated Time:** 2-3 sessions

**Building on 2025-12-11 BET fix:**

#### A. Brain Mask Quality Control
**Estimated Time:** 0.5 session

**Tasks:**
- [ ] Generate overlay plots (mask on mean functional)
- [ ] Calculate edge statistics (brain/non-brain contrast)
- [ ] Compare volumes across subjects
- [ ] Add automated mask quality flags
- [ ] Create HTML QC report

**Files to Modify:**
- `neurovrai/preprocess/qc/func_qc.py`

#### B. Configurable Parameters
**Estimated Time:** 0.5 session

**Tasks:**
- [ ] Add BET threshold (`frac`) to config
- [ ] Add bandpass frequency range to config
- [ ] Add smoothing FWHM to config
- [ ] Document optimal values for different datasets
- [ ] Add parameter validation

**Files to Modify:**
- `neurovrai/preprocess/workflows/func_preprocess.py`
- Config schema documentation

#### C. Motion Outlier Detection
**Estimated Time:** 1 session

**Tasks:**
- [ ] Integrate framewise displacement (FD) calculation
- [ ] Add DVARS computation
- [ ] Flag high-motion volumes
- [ ] Generate motion outlier masks
- [ ] Add motion summary statistics

**Files to Create:**
- `neurovrai/preprocess/utils/motion_outliers.py`

#### D. Functional→T1w Registration Integration
**Estimated Time:** 1 session

**Tasks:**
- [ ] Move functional→T1w registration to Phase 1
- [ ] Use anatomical brain mask in functional space
- [ ] Enable anatomical-guided brain extraction
- [ ] Add registration QC
- [ ] Support optional MNI normalization

**Files to Modify:**
- `neurovrai/preprocess/workflows/func_preprocess.py`
- `neurovrai/preprocess/utils/func_normalization.py`

**Success Criteria:**
- Brain mask QC catches poor extractions
- Users can tune parameters via config
- Motion outliers automatically detected
- Registration integrated into workflow

---

### 4. Enhanced QC Framework
**Priority:** MEDIUM
**Status:** Basic QC exists, needs enhancement
**Estimated Time:** 3-4 sessions

**Current State:**
- Basic HTML reports for each modality
- Manual inspection required
- No cross-subject comparison
- No automated quality metrics

**Improvements:**

#### A. Interactive HTML Dashboards
**Estimated Time:** 2 sessions

**Tasks:**
- [ ] Design interactive QC interface (Plotly/Bokeh)
- [ ] Add zoomable/pannable image viewers
- [ ] Enable side-by-side comparisons
- [ ] Add metric plots (scatter, histograms)
- [ ] Create subject-level and group-level views

**Files to Create:**
- `neurovrai/preprocess/qc/dashboard.py`
- HTML/CSS templates

#### B. Automated Outlier Detection
**Estimated Time:** 1 session

**Tasks:**
- [ ] Calculate z-scores for QC metrics
- [ ] Flag outliers (>2-3 SD from group mean)
- [ ] Identify failed runs (missing outputs, crashes)
- [ ] Generate outlier summary report
- [ ] Add recommended exclusion criteria

**Files to Modify:**
- `neurovrai/preprocess/qc/anat/`, `dwi/`, `func_qc.py`, `asl_qc.py`

#### C. QC Metrics Database
**Estimated Time:** 1 session

**Tasks:**
- [ ] Define QC metrics schema (JSON/CSV)
- [ ] Collect metrics across all subjects
- [ ] Generate group-level statistics
- [ ] Add metric visualization
- [ ] Export to CSV for external analysis

**Files to Create:**
- `neurovrai/preprocess/qc/metrics_database.py`

**Success Criteria:**
- Interactive dashboards for all modalities
- Automated outlier flagging with justification
- Centralized QC metrics for group analysis
- <5 minutes to QC 20 subjects

---

### 5. BIDS Compliance
**Priority:** MEDIUM
**Status:** BIDS-compatible but not formally validated
**Estimated Time:** 2 sessions

**Current State:**
- Outputs follow BIDS-like structure
- Not formally validated with BIDS validator
- Missing required metadata files

**Tasks:**

#### A. BIDS Validation
**Estimated Time:** 0.5 session

- [ ] Install and run BIDS validator
- [ ] Fix any structural issues
- [ ] Ensure derivatives follow BIDS derivatives spec
- [ ] Document BIDS structure

#### B. Metadata Files
**Estimated Time:** 1 session

- [ ] Create dataset_description.json
- [ ] Generate participants.tsv
- [ ] Add README with dataset description
- [ ] Create CHANGES log
- [ ] Add .bidsignore for exclusions

#### C. Derivative Metadata
**Estimated Time:** 0.5 session

- [ ] Add JSON sidecars for derivatives
- [ ] Document processing parameters
- [ ] Include software versions
- [ ] Add provenance tracking

**Files to Create:**
- `neurovrai/utils/bids_tools.py`

**Success Criteria:**
- BIDS validator passes with 0 errors
- Derivatives compatible with BIDS apps
- Complete metadata for reproducibility

---

## 🟢 Lower Priority / Future Enhancements

### 6. Containerization
**Priority:** LOW
**Estimated Time:** 2-3 sessions

**Benefits:**
- Reproducible environments
- Easy deployment
- HPC cluster compatibility
- Version control for dependencies

**Tasks:**
- [ ] Create Dockerfile
- [ ] Add Singularity recipe
- [ ] Document container usage
- [ ] Add to CI/CD pipeline
- [ ] Publish to Docker Hub / Singularity Hub

**Files to Create:**
- `Dockerfile`
- `Singularity.def`
- `docs/CONTAINER_USAGE.md`

---

### 7. HPC/Cluster Integration
**Priority:** LOW
**Estimated Time:** 2-3 sessions

**Features:**
- SLURM job submission
- Parallel subject processing
- Resource management
- Job monitoring and restart

**Tasks:**
- [ ] Create SLURM job templates
- [ ] Add subject-level parallelization
- [ ] Implement job status tracking
- [ ] Add automatic restart on failure
- [ ] Document cluster usage

**Files to Create:**
- `neurovrai/cluster/slurm_tools.py`
- `templates/slurm_job.sh`

---

### 8. Advanced Analysis Features
**Priority:** LOW
**Estimated Time:** 4-6 sessions (varies by feature)

**Potential Features:**

#### A. Network-Based Statistic (NBS)
- Connectivity matrix group comparisons
- Family-wise error correction
- Component visualization

#### B. Tract-Specific Analysis
- Along-tract statistics
- AFQ-style profiling
- White matter bundle quantification

#### C. Longitudinal Analysis
- Within-subject change detection
- Mixed-effects models
- Trajectory analysis

#### D. Multi-Modal Integration
- Combine anat/dwi/func metrics
- Joint dimensionality reduction
- Multi-modal prediction models

---

### 9. Web-Based QC Interface
**Priority:** LOW
**Estimated Time:** 4-5 sessions

**Features:**
- Flask/Django web app
- Rating system for data quality
- Collaborative QC workflows
- Export QC decisions
- User authentication

**Tasks:**
- [ ] Design web interface
- [ ] Implement backend API
- [ ] Create database schema
- [ ] Add user management
- [ ] Deploy to server

---

## 📋 Recommended Implementation Plan

### Immediate (Current Session)
1. ✅ Monitor functional preprocessing test completion
2. ✅ Validate Phase 2 workflow
3. ✅ Document final test results
4. ✅ Update status documents

### Next Session (Choose One Path)

#### **Path A: Complete Critical Gap** (Recommended)
**Focus:** FreeSurfer Integration

**Session 1:**
- Design FreeSurfer transform pipeline architecture
- Implement anatomical→DWI transform workflow
- Add basic transform validation

**Session 2:**
- Add FreeSurfer native space handling
- Implement transform QC visualization
- Test with real FreeSurfer outputs

**Session 3:**
- Documentation and testing
- Update CLAUDE.md to mark as production-ready
- Create usage examples

**Why Recommended:**
- Closes critical gap explicitly documented in CLAUDE.md
- Unlocks important feature (FreeSurfer ROIs for tractography)
- Well-defined scope (2-3 sessions)
- High impact for cortical parcellation users

#### **Path B: Build on Current Momentum**
**Focus:** Functional Preprocessing Enhancements

**Session 1:**
- Add brain mask QC visualization
- Implement configurable BET/bandpass parameters
- Add parameter validation

**Session 2:**
- Implement motion outlier detection
- Add framewise displacement and DVARS
- Generate motion summary reports

**Session 3:**
- Integrate functional→T1w registration into Phase 1
- Add registration QC
- Update documentation

**Why Consider:**
- Builds on today's successful work
- Polishes recently-fixed workflow
- Immediate user-facing improvements

### Following Sessions (2-4 weeks)
- Enhanced QC framework (interactive dashboards)
- BIDS validation and compliance
- Documentation improvements
- User guides and tutorials

### Long-Term (1-3 months)
- Containerization (Docker/Singularity)
- HPC cluster integration
- Advanced analysis features
- Web-based QC interface

---

## 🎯 Success Metrics

### Short-Term (1 month)
- [ ] FreeSurfer integration production-ready
- [ ] Functional preprocessing fully validated
- [ ] Interactive QC dashboards for all modalities
- [ ] BIDS validation passing

### Medium-Term (3 months)
- [ ] Docker/Singularity containers available
- [ ] HPC cluster support documented
- [ ] Comprehensive user documentation
- [ ] 5+ external users successfully running pipelines

### Long-Term (6 months)
- [ ] Publication-ready analysis examples
- [ ] Multi-site validation completed
- [ ] Advanced analysis features integrated
- [ ] Web-based QC deployed

---

## 📝 Notes

### Testing Strategy
- Always test on IRC805-0580101 first (validation dataset)
- Run full pipeline after major changes
- Generate QC reports for all test runs
- Compare outputs with previous versions

### Documentation Standards
- Update CLAUDE.md for production changes
- Create session summaries for major work
- Document breaking changes
- Add usage examples for new features

### Code Quality
- Follow existing patterns (see CLAUDE.md)
- Use pathlib.Path for all file operations
- Add type hints for new functions
- Write docstrings for public APIs
- Keep workflow functions modular

### Workflow Outputs
- All intermediates → work/ directory
- Only finals → derivatives/
- Full Nipype provenance tracking
- DataSink for organized outputs

---

## 🔄 This Document

**Maintenance:**
- Review and update monthly
- Add completed items to archive
- Re-prioritize based on user needs
- Track actual vs estimated time

**Related Documents:**
- `CLAUDE.md` - Development guidelines and architecture
- `docs/FUNCTIONAL_PREPROCESSING_RESOLVED.md` - Recent completion
- `docs/sessions/SESSION_2025-12-11_FUNCTIONAL_WORKFLOW_FIX.md` - Latest session
- `docs/status/` - Historical status documents

**Supersedes:**
- Individual TODO lists scattered in docs/
- Ad-hoc feature requests in session notes
- Informal priority discussions

---

**Last Review:** 2025-12-11
**Next Review:** 2026-01-11
**Owner:** Development Team
