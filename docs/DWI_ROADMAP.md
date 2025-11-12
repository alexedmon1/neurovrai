# DWI Preprocessing Roadmap & Progress Tracker

**Last Updated**: November 11, 2025

## Current Status

✅ **COMPLETED**: Multi-shell DWI preprocessing with TOPUP distortion correction
- Pipeline tested and validated on IRC805-0580101
- TOPUP, eddy, and DTIFit working successfully
- Standardized directory structure implemented
- Full documentation available

## Immediate Next Steps (High Priority)

### 1. ✅ Quality Control & Validation [IN PROGRESS]
**Status**: Starting implementation
**Priority**: HIGH
**Estimated Time**: 1-2 weeks

**Sub-tasks**:
- [ ] Visual QC of TOPUP outputs
  - [ ] Inspect field maps and corrected images in FSLeyes
  - [ ] Compare before/after TOPUP to quantify distortion correction
  - [ ] Check for residual artifacts or registration issues

- [ ] Quantitative metrics
  - [ ] Calculate motion parameters from eddy outputs
  - [ ] Compute SNR improvements from TOPUP
  - [ ] Generate QC reports with outlier detection

- [ ] Multi-subject validation
  - [ ] Test pipeline on additional subjects from IRC805 study
  - [ ] Verify consistency across different acquisition parameters
  - [ ] Document any subject-specific issues

**Outputs**: QC reports, validation metrics, visual comparison images
**Dependencies**: None
**Blocker Issues**: None

---

### 2. ⏳ Complete Advanced Diffusion Models
**Status**: Stub implementations exist
**Priority**: MEDIUM
**Estimated Time**: 2-3 weeks

**Sub-tasks**:
- [ ] DKI (Diffusion Kurtosis Imaging)
  - [ ] Implement using DIPY or MRtrix3
  - [ ] Calculate mean kurtosis (MK), axial kurtosis (AK), radial kurtosis (RK)
  - [ ] Validate on multi-shell data (requires b=2500-3000)
  - **Location**: `mri_preprocess/workflows/advanced_diffusion.py:34`

- [ ] NODDI (Neurite Orientation Dispersion and Density Imaging)
  - [ ] Implement using AMICO or NODDI Matlab toolbox
  - [ ] Calculate ODI, NDI, isotropic volume fraction
  - [ ] Validate multi-shell requirements
  - **Location**: `mri_preprocess/workflows/advanced_diffusion.py:221`

**Outputs**: DKI maps (MK, AK, RK), NODDI maps (ODI, NDI, ISO)
**Dependencies**: Multi-shell DWI data with high b-values
**Blocker Issues**: Need to verify b-value coverage in IRC805 data

---

### 3. ⏳ Probabilistic Tractography Integration
**Status**: Framework exists, needs completion
**Priority**: MEDIUM
**Estimated Time**: 2-3 weeks

**Sub-tasks**:
- [ ] Complete probtrackx2 workflow
  - [ ] Integrate with BEDPOSTX outputs
  - [ ] Add network-based tracking
  - [ ] Implement waypoint and exclusion masks
  - **Location**: `mri_preprocess/workflows/tractography.py`

- [ ] Atlas-based ROI tracking
  - [ ] Use existing `atlas_rois.py` utilities
  - [ ] Generate connectivity matrices
  - [ ] Support major white matter tract atlases (JHU, Juelich, etc.)

**Outputs**: Tractography results, connectivity matrices, tract maps
**Dependencies**: BEDPOSTX completion (Step 5)
**Blocker Issues**: None

---

### 4. ⏳ Registration to Anatomical/MNI Space
**Status**: Not started
**Priority**: HIGH
**Estimated Time**: 1-2 weeks

**Sub-tasks**:
- [ ] DWI→T1w registration
  - [ ] Implement BBR (boundary-based registration) for better accuracy
  - [ ] Register FA map to T1w space
  - [ ] Apply transforms to all diffusion metrics

- [ ] Warp to MNI space
  - [ ] Integrate with TransformRegistry (compute-once, reuse)
  - [ ] Use T1w→MNI transforms from anatomical workflow
  - [ ] Compose transforms: DWI→T1w→MNI
  - [ ] Validate against standard space templates

**Outputs**: DWI metrics in T1w and MNI space, transformation matrices
**Dependencies**: Anatomical preprocessing workflow
**Blocker Issues**: Need to test TransformRegistry integration

---

## Medium Priority

### 5. ⏳ BEDPOSTX Implementation & Testing
**Status**: Code exists, disabled in testing
**Priority**: MEDIUM
**Estimated Time**: 1 week + processing time

**Sub-tasks**:
- [ ] Enable BEDPOSTX in test configuration
- [ ] Test GPU-accelerated BEDPOSTX
- [ ] Validate outputs for tractography
- [ ] Document processing time (typically 4-8 hours per subject)
- [ ] Optimize parameters for multi-shell data

**Outputs**: BEDPOSTX outputs (dyads, f-samples, mean-samples)
**Dependencies**: Eddy-corrected DWI
**Blocker Issues**: Long processing time requires careful scheduling

---

### 6. ⏳ Multi-Shell Optimization
**Status**: Not started
**Priority**: MEDIUM
**Estimated Time**: 2-3 weeks

**Sub-tasks**:
- [ ] Optimize eddy parameters for multi-shell data
  - [ ] Test different outlier detection thresholds
  - [ ] Evaluate slice-to-volume vs volume-to-volume correction
  - [ ] Fine-tune motion model parameters

- [ ] Shell-specific processing
  - [ ] Extract and process individual shells separately
  - [ ] Generate shell-specific DTI metrics for comparison
  - [ ] Implement multi-shell multi-tissue CSD (if using MRtrix3)

**Outputs**: Optimized parameters, shell-specific metrics
**Dependencies**: Multi-subject testing
**Blocker Issues**: None

---

### 7. ⏳ Batch Processing Infrastructure
**Status**: Not started
**Priority**: MEDIUM
**Estimated Time**: 1-2 weeks

**Sub-tasks**:
- [ ] Subject iteration
  - [ ] Create batch processing script for multiple subjects
  - [ ] Add parallel subject processing (using multiprocessing)
  - [ ] Implement resume capability for failed subjects

- [ ] Error handling & logging
  - [ ] Centralized logging for batch runs
  - [ ] Email notifications for completion/errors
  - [ ] Generate batch-level QC reports

**Outputs**: Batch processing scripts, automated QC reports
**Dependencies**: QC infrastructure (Step 1)
**Blocker Issues**: None

---

## Lower Priority / Future Enhancements

### 8. ⏳ Additional Diffusion Analysis Tools
**Status**: Not started
**Priority**: LOW
**Estimated Time**: 3-4 weeks

**Sub-tasks**:
- [ ] TBSS (Tract-Based Spatial Statistics)
  - [ ] Group-level FA analysis
  - [ ] Statistical comparison between groups
  - [ ] Skeletonization and registration to FMRIB58_FA

- [ ] Fixel-based analysis (MRtrix3)
  - [ ] More advanced than voxel-based or TBSS
  - [ ] Resolves crossing fibers
  - [ ] Provides fiber-specific metrics

**Outputs**: TBSS results, group statistics, fixel metrics
**Dependencies**: Multiple subjects, MNI registration
**Blocker Issues**: Requires MRtrix3 installation

---

### 9. ⏳ Alternative Preprocessing Tools
**Status**: Not started
**Priority**: LOW
**Estimated Time**: 4-6 weeks

**Sub-tasks**:
- [ ] MRtrix3 integration
  - [ ] Add dwi2tensor, dwi2fod workflows
  - [ ] Implement MSMT-CSD (multi-shell multi-tissue)
  - [ ] Generate FOD-based tractography

- [ ] TORTOISE integration
  - [ ] Alternative to FSL eddy
  - [ ] Better handling of motion and distortions
  - [ ] Requires DICOM or raw k-space data

**Outputs**: Alternative processing pipelines
**Dependencies**: Software installation and testing
**Blocker Issues**: Requires additional dependencies

---

### 10. ⏳ Microstructural Models
**Status**: Not started
**Priority**: LOW
**Estimated Time**: 6-8 weeks

**Sub-tasks**:
- [ ] CHARMED (Composite Hindered and Restricted Model of Diffusion)
- [ ] AxCaliber (axon diameter mapping)
- [ ] SMT (Standard Model of diffusion in white matter)

**Outputs**: Microstructural parameter maps
**Dependencies**: Specialized software and validation data
**Blocker Issues**: Complexity and validation requirements

---

## Recommended Timeline

### Week 1-2 (Current)
1. ✅ Visual QC and validation (Step 1)
2. ⏳ Multi-subject testing (Step 1)
3. ⏳ DWI→T1w→MNI registration (Step 4)

### Week 3-4
4. ⏳ BEDPOSTX testing (Step 5)
5. ⏳ Complete probabilistic tractography (Step 3)
6. ⏳ Batch processing infrastructure (Step 7)

### Month 2
7. ⏳ DKI implementation (Step 2)
8. ⏳ NODDI implementation (Step 2)
9. ⏳ TBSS group analysis (Step 8)

### Month 3+
10. ⏳ Advanced models and alternative tools (Steps 9-10)

---

## Progress Legend

- ✅ **COMPLETED**: Task finished and tested
- 🔄 **IN PROGRESS**: Currently being worked on
- ⏳ **PLANNED**: Scheduled but not started
- ⚠️ **BLOCKED**: Waiting on dependencies or issues
- ❌ **CANCELLED**: No longer planned

---

## Notes

- All new features should follow standardized directory structure
- QC outputs should be integrated into batch processing
- Document processing times and resource requirements
- Maintain backward compatibility with existing workflows
- Update test suite for each new feature
