# Documentation Index

**Last Updated**: 2025-11-14

This document provides a curated index of all documentation for the Human MRI Preprocessing Pipeline, organized by category and relevance to current pipeline status.

---

## 📚 Essential Documentation (Start Here)

### For Users

1. **[README.md](../README.md)** - Project overview, installation, and quick start guide
   - Installation instructions
   - How to run the pipeline
   - Configuration examples
   - Production status and recent updates

2. **[PROJECT_STATUS.md](../PROJECT_STATUS.md)** - Current implementation status and roadmap
   - Production-ready workflows
   - In-progress features
   - Known issues and limitations
   - Recent activity log

3. **[configuration.md](configuration.md)** - Complete configuration reference
   - YAML configuration format
   - All available parameters
   - Advanced configuration options

4. **[workflows.md](workflows.md)** - Detailed workflow documentation
   - Anatomical preprocessing
   - DWI preprocessing
   - Functional preprocessing
   - ASL preprocessing

### For Developers

1. **[CLAUDE.md](../CLAUDE.md)** - AI assistant guidelines and project context
   - Project goals and status
   - Architecture overview
   - Development environment setup
   - Code style notes
   - Validated workflows

2. **[DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)** - Standardized directory hierarchy
   - Output organization
   - Work directory structure
   - QC directory layout

---

## 🔬 Workflow-Specific Documentation

### Diffusion Preprocessing (Production-Ready)

- **[DWI_PROCESSING_GUIDE.md](DWI_PROCESSING_GUIDE.md)** - Complete DWI preprocessing guide
  - TOPUP distortion correction
  - Multi-shell vs single-shell processing
  - Advanced models (DKI, NODDI)
  - Tractography

- **[DWI_QC_SPECIFICATION.md](DWI_QC_SPECIFICATION.md)** - Quality control for DWI
  - TOPUP QC metrics
  - Motion QC metrics
  - DTI QC metrics

- **[DWI_TOPUP_TEST_RESULTS.md](DWI_TOPUP_TEST_RESULTS.md)** - TOPUP validation results
  - Test data and parameters
  - Performance benchmarks
  - Quality metrics

### ASL Preprocessing (Production-Ready)

- **[implementation/asl_dicom_integration.md](implementation/asl_dicom_integration.md)** - DICOM parameter extraction
  - Automated τ and PLD extraction
  - Scanner-specific considerations

- **[implementation/asl_preprocessing_plan.md](implementation/asl_preprocessing_plan.md)** - ASL workflow design
  - M0 calibration strategy
  - Partial volume correction
  - CBF quantification

- **[implementation/asl_qc_summary.md](implementation/asl_qc_summary.md)** - ASL quality control
  - Motion metrics
  - CBF distribution analysis
  - tSNR computation

### Functional Preprocessing (95% Complete)

- **[implementation/RESTING_STATE_IMPLEMENTATION.md](implementation/RESTING_STATE_IMPLEMENTATION.md)** - Multi-echo fMRI workflow
  - TEDANA integration
  - ICA-AROMA auto-detection
  - ACompCor nuisance regression

- **[implementation/TEDANA_VS_AROMA.md](implementation/TEDANA_VS_AROMA.md)** - TEDANA vs ICA-AROMA comparison
  - When to use each method
  - Multi-echo vs single-echo considerations

- **[implementation/ACOMPCOR_IMPLEMENTATION.md](implementation/ACOMPCOR_IMPLEMENTATION.md)** - ACompCor implementation
  - Tissue mask registration
  - Component extraction
  - Nuisance regression

### Advanced Diffusion Models (Production-Ready)

- **[amico/AMICO_MODELS_DOCUMENTATION.md](amico/AMICO_MODELS_DOCUMENTATION.md)** - AMICO implementation guide
  - NODDI (100x speedup)
  - SANDI (soma and neurite density)
  - ActiveAx (axon diameter)

- **[archive/AMICO_INTEGRATION_COMPLETE.md](archive/AMICO_INTEGRATION_COMPLETE.md)** - AMICO integration summary
  - Performance benchmarks
  - Usage examples
  - Testing results

- **[implementation/GRADIENT_TIMING_SOLUTION.md](implementation/GRADIENT_TIMING_SOLUTION.md)** - Gradient timing for AMICO
  - TE, δ, Δ extraction
  - Estimation methods
  - Validation approach

### Spatial Normalization (Production-Ready)

- **[implementation/normalization_strategy.md](implementation/normalization_strategy.md)** - Normalization implementation
  - DWI → FMRIB58_FA
  - Functional → MNI152
  - Transform reuse strategy

---

## 🛠️ Technical Implementation

### Core Infrastructure

- **[cli.md](cli.md)** - Command-line interface documentation
  - Available commands
  - Usage examples

- **[TESTING.md](TESTING.md)** - Testing framework
  - Test organization
  - Running tests
  - Validation procedures

- **[TESTING_RESULTS.md](TESTING_RESULTS.md)** - Test results and validation
  - Workflow validation
  - Performance benchmarks
  - Known issues

### Quality Control

- **[DWI_QC_SPECIFICATION.md](DWI_QC_SPECIFICATION.md)** - DWI QC framework
  - Automated metrics
  - Visualization outputs
  - Integration with workflows

---

## 📊 Status and Progress Tracking

### Current Status

- **[status/IMPLEMENTATION_STATUS.md](status/IMPLEMENTATION_STATUS.md)** - Detailed implementation tracking
  - Completed phases
  - Git commit history
  - Key features implemented

- **[status/OVERNIGHT_RUN_STATUS.md](status/OVERNIGHT_RUN_STATUS.md)** - Recent overnight run results
  - Test subject processing
  - Workflow performance
  - Issues encountered

- **[status/PIPELINE_VERIFICATION_REPORT.md](status/PIPELINE_VERIFICATION_REPORT.md)** - Pipeline verification results
  - Single-shell vs multi-shell validation
  - Auto-detection testing

### Session Reports

- **[status/SESSION_REPORT_2025-11-13.md](status/SESSION_REPORT_2025-11-13.md)** - Development session summary
  - Features implemented
  - Bugs fixed
  - Next steps

---

## 🗂️ Archived Documentation

Documents in `docs/archive/` contain historical information but may be outdated:

- **[archive/AMICO_INTEGRATION_COMPLETE.md](archive/AMICO_INTEGRATION_COMPLETE.md)** - AMICO integration completion report (2025-11-12)
- **[archive/PROJECT_PLAN.md](archive/PROJECT_PLAN.md)** - Original project planning (superseded by PROJECT_STATUS.md)
- **[archive/QUICK_START_NEXT_SESSION.md](archive/QUICK_START_NEXT_SESSION.md)** - Session startup guide (superseded by CLAUDE.md)

---

## 🎯 Documentation by Use Case

### "I want to preprocess my data"
1. Read [README.md](../README.md)
2. Check [configuration.md](configuration.md) for config file setup
3. Follow examples in [workflows.md](workflows.md)
4. Review modality-specific guides (DWI, ASL, functional)

### "I want to understand the pipeline architecture"
1. Read [CLAUDE.md](../CLAUDE.md) - Project overview section
2. Review [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)
3. Check [workflows.md](workflows.md) for workflow details

### "I want to contribute/modify the code"
1. Read [CLAUDE.md](../CLAUDE.md) - Complete developer guide
2. Review [status/IMPLEMENTATION_STATUS.md](status/IMPLEMENTATION_STATUS.md)
3. Check [TESTING.md](TESTING.md) for testing approach

### "I need quality control information"
1. Check [DWI_QC_SPECIFICATION.md](DWI_QC_SPECIFICATION.md) for DWI
2. Review [implementation/asl_qc_summary.md](implementation/asl_qc_summary.md) for ASL
3. See workflow-specific docs for QC integration

### "I have issues with TOPUP/eddy"
1. Read [DWI_PROCESSING_GUIDE.md](DWI_PROCESSING_GUIDE.md)
2. Check [DWI_TOPUP_TEST_RESULTS.md](DWI_TOPUP_TEST_RESULTS.md)
3. Review [status/OVERNIGHT_RUN_STATUS.md](status/OVERNIGHT_RUN_STATUS.md)

### "I want to use AMICO for faster NODDI"
1. Read [amico/AMICO_MODELS_DOCUMENTATION.md](amico/AMICO_MODELS_DOCUMENTATION.md)
2. Check [archive/AMICO_INTEGRATION_COMPLETE.md](archive/AMICO_INTEGRATION_COMPLETE.md) for performance
3. Review [implementation/GRADIENT_TIMING_SOLUTION.md](implementation/GRADIENT_TIMING_SOLUTION.md) if using SANDI/ActiveAx

---

## 📈 Pipeline Status Summary (2025-11-14)

### ✅ Production Ready
- **Anatomical**: T1w preprocessing with N4, BET, segmentation, MNI registration
- **DWI**: Multi-shell/single-shell with optional TOPUP, GPU eddy, DTI/DKI/NODDI, tractography
  - AMICO support (100x faster NODDI)
- **ASL**: pCASL with M0 calibration, PVC, automated DICOM extraction

### 🔄 In Final Testing (95%)
- **Functional**: Multi-echo TEDANA preprocessing
  - TEDANA upgraded to 25.1.0 (NumPy 2.x compatible)
  - Currently validating on IRC805-0580101

### ⚠️ Experimental (Not Production Ready)
- **FreeSurfer Integration**: Hooks only, transform pipeline incomplete

---

## 🔗 Quick Links

- **GitHub Repository**: https://github.com/yourusername/human-mri-preprocess
- **Issue Tracker**: https://github.com/yourusername/human-mri-preprocess/issues
- **Latest Release**: See [README.md](../README.md) for version info

---

## 📝 Contributing to Documentation

When adding new documentation:
1. Place user guides in `docs/`
2. Place implementation details in `docs/implementation/`
3. Place status updates in `docs/status/`
4. Archive outdated docs in `docs/archive/`
5. Update this index with the new document
6. Link from relevant sections in README.md or CLAUDE.md

For questions about documentation organization, see [CLAUDE.md](../CLAUDE.md) section on "Project Organization".
