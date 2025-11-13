# Documentation Overview

This directory contains all technical documentation for the human-mri-preprocess pipeline.

## Directory Structure

### `/` (Root Documentation)
- **User Guides**: General usage and workflow documentation
  - `cli.md` - Command-line interface reference
  - `configuration.md` - Configuration options
  - `workflows.md` - Workflow descriptions
  - `DIRECTORY_STRUCTURE.md` - Output directory organization
  - `TESTING.md` - Testing procedures
  - `TESTING_RESULTS.md` - Validation results

### `/implementation/` (Implementation Details)
Technical implementation documentation for specific features:
- `ACOMPCOR_IMPLEMENTATION.md` - ACompCor nuisance regression for fMRI
- `ACTIVEAX_IMPLEMENTATION.md` - ActiveAx diffusion model
- `GRADIENT_TIMING_SOLUTION.md` - Gradient timing corrections
- `RESTING_STATE_IMPLEMENTATION.md` - Resting-state fMRI preprocessing
- `RESTING_STATE_PLAN.md` - Resting-state implementation plan
- `TEDANA_VS_AROMA.md` - Design decision: TEDANA vs ICA-AROMA

### `/status/` (Project Status)
Current implementation status and progress tracking:
- `RESTING_STATE_STATUS.md` - Resting-state fMRI status
- `DWI_ADVANCED_STATUS.md` - Advanced diffusion models status
- `IMPLEMENTATION_STATUS.md` - Overall project status
- `SESSION_STATUS.md` - Development session notes

### `/amico/` (AMICO Integration)
Documentation for AMICO (Accelerated Microstructure Imaging via Convex Optimization):
- `AMICO_FINDINGS.md` - Research findings
- `AMICO_INTEGRATION_COMPLETE.md` - Integration completion report
- `AMICO_INTEGRATION_SUMMARY.md` - Summary of integration
- `AMICO_MODELS_DOCUMENTATION.md` - Model specifications
- `AMICO_TODO.md` - Outstanding tasks

### `/archive/` (Archived Documentation)
Outdated or superseded documentation:
- `PROJECT_PLAN.md` - Original project plan (superseded by README)
- `QUICK_START_NEXT_SESSION.md` - Session-specific notes (outdated)

## DWI-Specific Documentation

- `DWI_PROCESSING_GUIDE.md` - Complete DWI processing guide
- `DWI_QC_SPECIFICATION.md` - Quality control specifications
- `DWI_ROADMAP.md` - DWI feature roadmap
- `DWI_TOPUP_TEST_RESULTS.md` - TOPUP validation results

## Quick Reference

### For Users
Start with:
1. `../README.md` (main project README)
2. `configuration.md` - Setup and configuration
3. `cli.md` - Command-line usage
4. `workflows.md` - Understanding the pipelines

### For Developers
Technical implementation:
1. `implementation/` - Feature implementation details
2. `status/` - Current development status
3. `TESTING.md` - Testing procedures

### For Contributors
1. `../CLAUDE.md` - AI-assisted development guidelines
2. `implementation/` - Implementation patterns
3. `status/` - What's currently in progress
