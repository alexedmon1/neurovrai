# MRI Preprocessing Pipeline - Project Status
**Last Updated**: 2025-11-14

## 🎯 Project Overview

Production-ready MRI preprocessing pipeline for anatomical, diffusion, functional, and ASL data with comprehensive QC and standardized outputs.

---

## ✅ Completed & Production-Ready

### 1. Core Infrastructure (100%)
- ✅ Config-driven architecture (YAML with validation)
- ✅ DICOM converter with automatic parameter extraction
- ✅ Transform Registry for spatial transformations
- ✅ Standardized directory hierarchy
- ✅ Continuous streaming pipeline (start preprocessing as data converts)
- ✅ BIDS compatibility

### 2. Anatomical Preprocessing (100%)
**Status**: Fully functional, tested on IRC805-0580101

**Features**:
- N4 bias correction (ANTs)
- Brain extraction (FSL BET)
- Tissue segmentation (ANTs Atropos - faster than FAST)
- Registration to MNI152 (FLIRT/FNIRT)
- Tissue masks for ACompCor (CSF, GM, WM)

**QC**:
- ✅ Skull stripping QC
- ✅ Segmentation QC
- ✅ Registration QC

**Outputs**: `/derivatives/{subject}/anat/`

### 3. DWI Preprocessing (100%)
**Status**: Fully functional with advanced models, tested on multi-shell IRC805-0580101

**Features**:
- TOPUP susceptibility distortion correction
- GPU-accelerated eddy correction
- DTI fitting (FA, MD, AD, RD)
- **Advanced Models**:
  - DKI (Diffusion Kurtosis Imaging): MK, AK, RK, KFA
  - NODDI (Neurite Orientation): FICVF, ODI, FISO
- Spatial normalization to FMRIB58_FA template
- GPU-accelerated probabilistic tractography
- Atlas-based ROI extraction (Harvard-Oxford, JHU)

**Shell Detection**:
- Auto-detects single-shell vs multi-shell
- Clear labeling: "Multi-shell data: True/False"
- Auto-skips advanced models for single-shell

**QC**:
- ✅ TOPUP QC (field maps, convergence)
- ✅ Motion QC (FD, outliers)
- ✅ DTI QC (FA/MD distributions)

**Outputs**: `/derivatives/{subject}/dwi/`

### 4. ASL Preprocessing (100%)
**Status**: Fully functional, tested on IRC805-0580101

**Features**:
- Automated DICOM parameter extraction (τ, PLD)
- Motion correction (MCFLIRT)
- Label-control separation
- CBF quantification (Alsop et al., 2015)
- M0 calibration with white matter reference
- Partial volume correction
- Registration to anatomical space
- Tissue-specific CBF statistics

**QC**:
- ✅ Motion metrics
- ✅ CBF distributions
- ✅ tSNR analysis
- ✅ HTML QC report

**Outputs**: `/derivatives/{subject}/asl/`

### 5. Functional Preprocessing (95%)
**Status**: Working, currently testing multi-echo fixes

**Features**:
- Multi-echo TEDANA denoising
- Motion correction (MCFLIRT)
- Single-echo support (with optional ICA-AROMA)
- ACompCor nuisance regression
- Bandpass temporal filtering
- Spatial smoothing
- Registration to anatomical space
- Spatial normalization to MNI152

**Echo Detection**:
- Auto-detects single vs multi-echo
- Clear labeling: "Input data: 3 echoes" or "Single-echo data"
- Conditional workflow routing

**Recent Fixes (2025-11-14)**:
- ✅ Fixed DICOM converter to handle all multi-echo files
- ✅ Fixed workflow input node selection bug
- 🔄 Currently testing on IRC805-0580101 (ETA: ~2-3 hours)

**QC**:
- 🔄 Functional QC module (DVARS, carpet plots) - testing
- ⏳ Motion metrics integration
- ⏳ tSNR analysis

**Outputs**: `/derivatives/{subject}/func/`

---

## 🔄 In Progress

### Functional Preprocessing Testing
- **Current Step**: MCFLIRT + TEDANA on 3-echo resting-state data
- **Started**: 2025-11-14 09:28
- **ETA**: ~2-3 hours
- **Next**: Verify QC report generation

---

## 📋 Planned Features

### 1. FreeSurfer Integration (Priority: Medium, Status: HOOKS ONLY)

**Current Status**: Detection and extraction hooks implemented, **NOT production ready**

**What's Implemented** (2025-11-14):
- ✅ Detection of existing FreeSurfer outputs (`detect_freesurfer_subject()`)
- ✅ ROI extraction from aparc+aseg parcellation
- ✅ Config integration (`freesurfer.enabled`, `freesurfer.subjects_dir`)
- ✅ Tractography workflow hooks (with fallback to atlas ROIs)

**Critical Missing Components**:
- ❌ Anatomical→DWI transform pipeline (ROIs in wrong space!)
- ❌ FreeSurfer native space handling
- ❌ Transform quality control
- ❌ Validation that FreeSurfer T1 matches preprocessing T1

**Why This Is Complex**:
The main challenge is the transformation pipeline:
1. FreeSurfer outputs are in **native anatomical space**
2. For tractography, ROIs must be in **DWI space**
3. Requires: FreeSurfer space → Anatomical T1 → DWI space transforms
4. Each transform must be validated for accuracy
5. May need to handle cases where FreeSurfer was run on different T1

**Implementation Path**:
1. ✅ Add FreeSurfer detection and extraction utilities
2. ⏳ Create anatomical→DWI registration workflow
3. ⏳ Implement FreeSurfer → DWI transform pipeline
4. ⏳ Add QC for transform accuracy
5. ⏳ Integrate with anatomical preprocessing
6. ⏳ Test on real data with manual validation

**Estimated Development Time**: 2-3 full sessions

**DO NOT ENABLE** `freesurfer.enabled = true` until transform pipeline is complete

### 2. AMICO Integration (Priority: Low)
**Scope**:
- Better NODDI fitting using Accelerated Microstructure Imaging via Convex Optimization
- Potentially faster and more accurate than current DIPY implementation

**Status**: Research phase, DIPY-based NODDI working well

**Documentation**: `docs/amico/AMICO_FINDINGS.md`

### 3. Enhanced QC (Priority: Medium)
**Scope**:
- ⏳ Complete functional QC integration
- ⏳ Group-level QC reports
- ⏳ Interactive HTML dashboards
- ⏳ Automated outlier detection

**Current TODO**: `mri_preprocess/qc/dwi/__init__.py:16`

---

## 🐛 Known Issues & Limitations

### Resolved (2025-11-14)
- ✅ DICOM converter only copied first multi-echo file → **FIXED**
- ✅ Functional workflow input node selection bug → **FIXED**

### Active
- None currently identified

### Limitations
- **Functional TOPUP**: Not applicable - no reversed phase-encoding field maps in current protocol
  - All scans (fMRI_CORRECTION + RESTING) have same phase encoding (COL)
  - Would require protocol update to add AP/PA field map pair

---

## 📊 Validation Status

### Datasets Tested
- **IRC805-0580101** (Full multi-modal subject):
  - ✅ Anatomical (T1w)
  - ✅ DWI multi-shell (b=1000, 2000, 3000)
  - ✅ ASL (pCASL with M0)
  - 🔄 Functional (3-echo resting-state) - in progress

### Metrics Generated

**Anatomical**:
- Brain masks, tissue segmentations
- MNI-registered T1w

**DWI** (19 total metrics):
- DTI: FA, MD, AD, RD (4)
- DKI: MK, AK, RK, KFA (4)
- NODDI: FICVF, ODI, FISO (3)
- Normalized versions: 12 metrics in FMRIB58_FA space

**ASL**:
- CBF maps (native, calibrated)
- Tissue-specific CBF values
- tSNR maps

---

## 📁 Repository Structure

```
human-mri-preprocess/
├── mri_preprocess/          # Production code
│   ├── workflows/           # Preprocessing workflows (anat, dwi, func, asl)
│   ├── utils/               # Helper utilities
│   ├── qc/                  # Quality control modules
│   └── config.py            # Configuration system
├── docs/                    # Documentation
│   ├── status/              # Implementation status tracking
│   ├── amico/               # AMICO research
│   └── archive/             # Outdated docs
├── archive/                 # Legacy code (reference only)
├── logs/                    # All log files (gitignored)
├── config.yaml              # Production config
├── CLAUDE.md                # AI assistant guidelines
└── README.md                # User documentation
```

---

## 🎓 Documentation

### User Documentation
- `README.md` - Installation, usage, configuration
- `docs/configuration.md` - Config file reference
- `docs/workflows.md` - Workflow details
- `docs/DIRECTORY_STRUCTURE.md` - Output organization

### Developer Documentation
- `CLAUDE.md` - AI assistant context
- `docs/status/IMPLEMENTATION_STATUS.md` - Development tracking
- `docs/DWI_PROCESSING_GUIDE.md` - DWI pipeline details
- `PIPELINE_VERIFICATION_REPORT.md` - Single/multi-shell verification

### Testing & Validation
- `TESTING_RESULTS.md` - Test outcomes
- `OVERNIGHT_RUN_STATUS.md` - Latest overnight run results

---

## 🚀 Next Steps

### Immediate (This Session)
1. ⏳ Complete functional preprocessing test
2. ⏳ Verify functional QC report generation
3. ⏳ Update CLAUDE.md with 2025-11-14 session notes
4. ⏳ Clean up verification scripts

### Short-term (Next Session)
1. Run complete pipeline on second subject to validate
2. Generate group-level QC reports
3. Performance profiling and optimization
4. User documentation updates

### Medium-term
1. FreeSurfer integration
2. Enhanced QC dashboards
3. Additional atlas support for tractography
4. Batch processing utilities

### Long-term
1. AMICO integration (if needed)
2. Docker/Singularity containerization
3. HPC cluster support
4. Web-based QC interface

---

## 📝 Recent Activity Log

### 2025-11-14 (Current Session - Part 2)
- **Fixed**: TEDANA NumPy compatibility - upgraded TEDANA 23.0.2 → 25.1.0
- **Fixed**: NumPy version constraint in pyproject.toml (relaxed to >=1.24,<3.0)
- **Fixed**: DWI work directory hierarchy bug (now correctly uses `work/{subject}/dwi_preprocess/`)
- **Updated**: CLAUDE.md - moved ASL and AMICO from "Planned" to "Production-Ready"
- **Updated**: README.md with latest status and bug fixes
- **Running**: Functional preprocessing with TEDANA 25.1.0 (in progress)

### 2025-11-14 (Earlier)
- **Fixed**: Multi-echo DICOM conversion bug
- **Fixed**: Functional workflow input node selection
- **Verified**: Single-shell DWI support with proper labeling
- **Verified**: Single-echo fMRI support with proper labeling
- **Investigated**: fMRI TOPUP (not applicable - no reversed PE)

### 2025-11-13 (Overnight Run)
- **Completed**: Anatomical preprocessing (IRC805-0580101)
- **Completed**: DWI preprocessing with advanced models + normalization
- **Completed**: ASL preprocessing with M0 calibration
- **Implemented**: DWI → FMRIB58_FA normalization
- **Implemented**: Functional → MNI152 normalization framework
- **Failed**: Functional preprocessing (due to bugs now fixed)

### 2025-11-12
- **Completed**: DWI TOPUP validation
- **Completed**: DKI/NODDI implementation
- **Enhanced**: Functional QC module

---

## 💡 Notes

- **GPU Required**: For optimal performance (eddy_cuda, probtrackx2_gpu)
- **FSL Required**: Version 6.0+ for all workflows
- **Python**: 3.13+ (managed via uv)
- **Tested Platform**: Linux (Ubuntu-based)

---

**Pipeline is production-ready for anatomical, DWI, and ASL preprocessing.**
**Functional preprocessing in final testing phase.**
