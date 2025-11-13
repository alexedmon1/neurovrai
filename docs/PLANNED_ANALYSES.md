# Planned Downstream Analyses

This document outlines planned group-level analyses for each MRI modality, with special attention to **spatial normalization requirements**.

**Notation**:
- ✅ **REQUIRES NORMALIZATION**: Must have data in standard space (MNI152/atlas)
- ⚠️ **OPTIONAL NORMALIZATION**: Can work in native space, but standard space recommended for group comparisons
- ❌ **NO NORMALIZATION**: Works in native space only

---

## Anatomical (T1w/T2w) Analyses

### Current Preprocessing Status
- ✅ N4 bias correction
- ✅ ANTs Atropos tissue segmentation (GM, WM, CSF)
- ✅ FSL BET skull stripping
- ✅ **FLIRT/FNIRT registration to MNI152** → Standard space transformations available
- ✅ Spatial normalization: **IMPLEMENTED**

### Planned Analyses

#### 1. Voxel-Based Morphometry (VBM) ✅ **REQUIRES NORMALIZATION**
**Description**: Group-level statistical comparisons of gray/white matter volume

**Requirements**:
- All subjects' tissue probability maps in MNI152 space
- Modulated (preserve volume) or unmodulated segmentations
- Smoothing (typically 6-8mm FWHM)

**Implementation**:
- Apply FNIRT transformations to tissue segmentation outputs
- FSL randomise with TFCE for multiple comparisons correction
- Reference: `archive/anat/vbm_randomise.py`

**Normalization Status**: ✅ Preprocessing includes FNIRT to MNI152

---

#### 2. FreeSurfer Cortical Analysis ⚠️ **OPTIONAL NORMALIZATION**
**Description**: Cortical surface reconstruction and thickness measurements

**Requirements**:
- Surface-based analysis: Works in native space (NO normalization needed)
- Subcortical volumes: Standard space helpful for group comparisons
- ROI extraction: Can warp ROIs to native space OR warp brain to atlas space

**Implementation**:
- FreeSurfer `recon-all` with T1w+T2w inputs
- Integration with DWI/fMRI: warp FreeSurfer ROIs to native DWI/func space
- Alternative: warp DWI/func to FreeSurfer space (T1w native)

**Normalization Status**: ⚠️ Optional - depends on analysis approach

---

#### 3. Myelin Mapping (T1w/T2w Ratio) ✅ **REQUIRES NORMALIZATION**
**Description**: T1w/T2w ratio as myelin proxy

**Requirements**:
- T1w and T2w coregistered
- Both images in MNI152 space for group comparisons
- Masked division to compute ratio maps

**Implementation**:
- Apply FNIRT transformations to both T1w and T2w
- Compute ratio in standard space
- Reference: `archive/myelin/myelin_workflow.py`

**Normalization Status**: ✅ Preprocessing includes FNIRT to MNI152

---

## Diffusion (DWI/DTI/DKI/NODDI) Analyses

### Current Preprocessing Status
- ✅ TOPUP distortion correction
- ✅ GPU eddy correction with motion parameters
- ✅ DTI metrics (FA, MD, AD, RD)
- ✅ DKI metrics (MK, AK, RK, KFA)
- ✅ NODDI metrics (ODI, FICVF, FISO)
- ✅ Probabilistic tractography with atlas-based ROIs
- ❌ **Spatial normalization: NOT IMPLEMENTED**
- ⚠️ **ACTION NEEDED**: Add DWI→MNI152 registration to preprocessing

### Planned Analyses

#### 1. Tract-Based Spatial Statistics (TBSS) ✅ **REQUIRES NORMALIZATION**
**Description**: Whole-brain voxelwise FA/MD group comparisons on white matter skeleton

**Requirements**:
- All subjects' FA maps in MNI152 space
- TBSS registration (FA → FMRIB58_FA template)
- Skeleton projection (4D skeletonized data)

**Implementation**:
- FSL TBSS pipeline: `tbss_1_preproc` → `tbss_2_reg` → `tbss_3_postreg` → `tbss_4_prestats`
- FSL randomise for permutation testing
- Apply skeleton to MD, AD, RD, DKI, NODDI metrics
- Reference: `archive/dwi/tbss_randomise.py`, `archive/dwi/tbss_design_matrix.py`

**Normalization Status**: ❌ **NOT IMPLEMENTED**
- **Recommendation**: Add FLIRT/FNIRT to register FA to MNI152 in preprocessing
- Alternative: TBSS performs its own registration, but having MNI transforms helps with ROI extraction

---

#### 2. Advanced Model Group Analyses (DKI/NODDI) ✅ **REQUIRES NORMALIZATION**
**Description**: Group comparisons of DKI and NODDI metrics

**Requirements**:
- MK, AK, RK, KFA (DKI) in MNI152 space
- ODI, FICVF, FISO (NODDI) in MNI152 space
- ROI-based extraction using atlas regions
- Voxel-wise statistics (randomise)

**Implementation**:
- Apply DWI→MNI152 transformations to all metric maps
- ROI-based: Warp atlas to native DWI space OR warp metrics to MNI152
- Correlation with behavioral/clinical measures

**Normalization Status**: ❌ **NOT IMPLEMENTED**
- **Recommendation**: Add DWI→MNI152 registration in `dwi_preprocess.py`
- **Implementation**: After DTI fitting, register FA to MNI152, then apply to all metrics

---

#### 3. Tractography Connectivity ⚠️ **OPTIONAL NORMALIZATION**
**Description**: Seed-to-target connectivity matrices using probabilistic tractography

**Requirements**:
- BEDPOSTX fiber orientation distributions
- Atlas-based ROIs (Harvard-Oxford, JHU, AAL2)
- Can work in native DWI space OR standard space

**Implementation Options**:
- **Option A (Current)**: Warp atlas ROIs to native DWI space → tractography in native space
- **Option B**: Register DWI to MNI152 → tractography in standard space
- **Option C**: Tractography in native space → warp connectivity maps to standard space

**Normalization Status**: ⚠️ **OPTIONAL** - Current approach warps ROIs to native space
- **Current**: Atlas ROIs warped to native DWI space (implemented in `tractography.py`)
- **Alternative**: Could add DWI→MNI152 registration for standard-space tractography

---

#### 4. AMICO Integration ❌ **NO NORMALIZATION**
**Description**: Accelerated microstructure imaging via convex optimization (faster NODDI fitting)

**Requirements**:
- Preprocessing step, not analysis
- Works in native DWI space
- Outputs same metrics as NODDI (ODI, FICVF) but ~10-100x faster

**Implementation**:
- Replace DIPY NODDI fitting with AMICO
- Integration into `advanced_diffusion.py`

**Normalization Status**: ❌ N/A - This is preprocessing, normalization handled in analysis step

---

## Functional (Resting-State fMRI) Analyses

### Current Preprocessing Status
- ✅ Multi-echo MCFLIRT motion correction
- ✅ TEDANA multi-echo denoising (v25.1.0)
- 🔄 ACompCor nuisance regression (planned)
- 🔄 Bandpass filtering (0.001-0.08 Hz) (planned)
- ❌ **Spatial normalization: NOT IMPLEMENTED**
- ⚠️ **ACTION NEEDED**: Add func→MNI152 registration to preprocessing

### Planned Analyses

#### 1. Network Analysis with Dual Regression ✅ **REQUIRES NORMALIZATION**
**Description**: ICA-based resting-state network extraction using dual regression

**Requirements**:
- All subjects' preprocessed BOLD in MNI152 space
- Group ICA spatial maps (from group-level MELODIC)
- Dual regression: Stage 1 (spatial regression) → Stage 2 (temporal regression)

**Implementation**:
- FSL MELODIC for group ICA (requires MNI152 space)
- FSL dual_regression for subject-level network maps
- Randomise for group comparisons of network maps
- Reference: `archive/rest/rest_dualregress.py`

**Normalization Status**: ❌ **NOT IMPLEMENTED**
- **Recommendation**: Add func→MNI152 registration in `func_preprocess.py`
- **Implementation**: After TEDANA, register to anatomical T1w, then apply anat→MNI152 transform

---

#### 2. Seed-Based Connectivity ✅ **REQUIRES NORMALIZATION**
**Description**: ROI-to-ROI functional connectivity using correlation matrices

**Requirements**:
- Preprocessed BOLD in MNI152 space OR atlas ROIs warped to native func space
- Atlas-based parcellation (AAL2, Harvard-Oxford, Power264, Schaefer)
- Time series extraction from ROIs
- Correlation matrix (Pearson or partial correlation)

**Implementation Options**:
- **Option A (Recommended)**: Normalize BOLD to MNI152 → extract ROI time series
- **Option B**: Warp atlas ROIs to native func space → extract time series
- Reference: `archive/rest/rest_corrmat_generation.py`

**Normalization Status**: ❌ **NOT IMPLEMENTED**
- **Recommendation**: Add func→MNI152 registration for consistency with dual regression
- **Alternative**: Implement Option B (warp ROIs to native space) if normalization unavailable

---

#### 3. Graph Theory Measures ⚠️ **OPTIONAL NORMALIZATION**
**Description**: Network metrics (efficiency, modularity, clustering)

**Requirements**:
- Connectivity matrix (from seed-based connectivity)
- Works in native or standard space (depends on ROI definition)

**Implementation**:
- Compute graph metrics from connectivity matrices
- Brain Connectivity Toolbox (BCT) in MATLAB/Python
- Group comparisons of network properties

**Normalization Status**: ⚠️ Depends on seed-based connectivity approach

---

#### 4. Multimodal Integration (Structure-Function Coupling) ⚠️ **OPTIONAL NORMALIZATION**
**Description**: Comparison of structural (DWI tractography) and functional (fMRI) connectivity

**Requirements**:
- Structural connectivity matrix from tractography
- Functional connectivity matrix from resting-state fMRI
- Both matrices using same ROI parcellation

**Implementation Options**:
- **Option A**: Both in native DWI space (warp func to DWI)
- **Option B**: Both in native func space (warp DWI to func)
- **Option C**: Both in MNI152 space (warp both to standard space)

**Normalization Status**: ⚠️ Flexible - depends on analysis design

---

## Summary: Normalization Requirements by Modality

### Anatomical
- ✅ **Normalization IMPLEMENTED** (FLIRT/FNIRT to MNI152)
- ✅ VBM ready for analysis
- ✅ Myelin mapping ready for analysis
- ⚠️ FreeSurfer: Optional normalization

### Diffusion
- ❌ **Normalization NOT IMPLEMENTED**
- ⚠️ **HIGH PRIORITY**: Add DWI→MNI152 registration to `dwi_preprocess.py`
- Required for: TBSS, DKI/NODDI group analyses
- Optional for: Tractography (current approach warps ROIs to native space)

### Functional
- ❌ **Normalization NOT IMPLEMENTED**
- ⚠️ **HIGH PRIORITY**: Add func→MNI152 registration to `func_preprocess.py`
- Required for: Dual regression, seed-based connectivity
- Should use: func→anat→MNI152 (two-step registration with anatomical as intermediate)

---

## Recommended Preprocessing Enhancements

### Priority 1: Add Normalization to DWI Preprocessing
**Location**: `mri_preprocess/workflows/dwi_preprocess.py`

**Steps**:
1. After DTI fitting, register FA map to MNI152 template
   - Use FLIRT (affine) + FNIRT (non-linear)
   - Template: `$FSLDIR/data/standard/FMRIB58_FA_1mm.nii.gz`
2. Apply transformation to all metric maps:
   - DTI: FA, MD, AD, RD, L1, L2, L3
   - DKI: MK, AK, RK, KFA
   - NODDI: ODI, FICVF, FISO
3. Save transformations for applying to ROIs/atlases
4. Create normalized versions in: `{derivatives}/{subject}/dwi/normalized/`

**Benefits**:
- Enables TBSS analysis
- Enables voxel-wise group comparisons
- Standardizes ROI placement across subjects

---

### Priority 2: Add Normalization to Functional Preprocessing
**Location**: `mri_preprocess/workflows/func_preprocess.py`

**Steps**:
1. After TEDANA denoising, coregister to anatomical T1w
   - Use FSL FLIRT (BBR - boundary-based registration)
   - Use middle echo or optimal combination as reference
2. Apply anatomical→MNI152 transformation (from anat preprocessing)
   - Concatenate func→anat + anat→MNI152 transforms
   - Use FSL applywarp with combined transformation
3. Apply to preprocessed BOLD time series
4. Save normalized version in: `{derivatives}/{subject}/func/normalized/`
5. Resample to standard resolution (e.g., 3mm isotropic for resting-state)

**Benefits**:
- Enables group ICA and dual regression
- Enables standard-space seed-based connectivity
- Consistent with fMRIPrep and other standard pipelines

---

### Priority 3: FreeSurfer Integration (All Modalities)
**Location**: New module or integration into existing workflows

**Steps**:
1. Run FreeSurfer `recon-all` on T1w (optionally with T2w)
2. Generate subject-specific cortical/subcortical parcellations
3. **For DWI**: Warp FreeSurfer ROIs to native DWI space
4. **For fMRI**: Warp FreeSurfer ROIs to native func space OR func to FreeSurfer space
5. Extract ROI-based metrics using subject-specific anatomy

**Benefits**:
- Subject-specific ROIs (more accurate than atlas-based)
- Surface-based analysis (avoids normalization artifacts)
- Integration with structural connectivity (tractography endpoints)

---

## Implementation Timeline

### Phase 1: Add Spatial Normalization (HIGH PRIORITY)
- [ ] Implement DWI→MNI152 registration in `dwi_preprocess.py`
- [ ] Implement func→MNI152 registration in `func_preprocess.py`
- [ ] Test on IRC805-0580101 dataset
- [ ] Validate output directory structure

### Phase 2: Implement Group-Level Analyses
- [ ] TBSS pipeline for FA/MD/DKI/NODDI
- [ ] VBM pipeline for tissue volume comparisons
- [ ] Dual regression for resting-state networks
- [ ] Seed-based connectivity with AAL2/Power264 atlases

### Phase 3: Advanced Features
- [ ] FreeSurfer integration
- [ ] AMICO integration for fast NODDI
- [ ] Multimodal structure-function coupling
- [ ] Automated QC report generation

---

## References

### Archive Scripts (Legacy Code)
- `archive/anat/vbm_randomise.py`: VBM permutation testing
- `archive/dwi/tbss_randomise.py`: TBSS permutation testing
- `archive/dwi/tbss_design_matrix.py`: Design matrix creation
- `archive/rest/rest_dualregress.py`: Dual regression implementation
- `archive/rest/rest_corrmat_generation.py`: AAL2 connectivity matrices
- `archive/myelin/myelin_workflow.py`: Myelin water imaging

### Documentation
- `CLAUDE.md`: Project overview and design decisions
- `README.md`: User guide and configuration
- `docs/DIRECTORY_STRUCTURE.md`: Output organization
- `docs/implementation/`: Technical implementation details

---

## Notes

**Decision Point**: Should we add normalization to preprocessing OR to analysis scripts?

**Recommendation**: **Add to preprocessing**
- Ensures consistent preprocessing across all analyses
- Simplifies analysis scripts (no need to handle normalization)
- Follows fMRIPrep/C-PAC conventions (output both native and standard space)
- Future analyses can choose native or normalized versions as needed

**Trade-offs**:
- Preprocessing takes longer (additional registration steps)
- Larger output files (both native and normalized versions)
- More disk space required

**Alternative**: Keep preprocessing in native space, add normalization to each analysis
- Pro: More flexible (can use different templates/methods per analysis)
- Con: Duplicated code, inconsistent normalization across analyses
- Con: More complex analysis scripts

**Conclusion**: Preprocessing should output both native and normalized versions for maximum flexibility.
