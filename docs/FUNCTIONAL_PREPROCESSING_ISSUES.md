# Functional Preprocessing Issues - Session Findings

**Date**: 2025-12-08
**Session Focus**: Atlas transformation for functional connectivity analysis
**Critical Discovery**: Fundamental issues with functional→anatomical registration

---

## Executive Summary

The functional preprocessing pipeline has a critical flaw in the registration approach that prevents accurate transformation of functional data to anatomical (T1w) and standard (MNI152) space. The current implementation uses preprocessed functional data for registration, which produces poor alignment. **The entire functional preprocessing pipeline needs to be refactored.**

---

## Current Problems

### 1. Registration Performed on Wrong Data
**Location**: `neurovrai/preprocess/utils/acompcor_helper.py:134-146`

**Current approach (INCORRECT)**:
```python
# Using PREPROCESSED functional data for registration
flirt_cmd = [
    'flirt',
    '-in', str(mean_func),      # Mean of BANDPASS-FILTERED data
    '-ref', str(t1w_brain),
    '-dof', '6',
    '-cost', 'corratio',        # Simple correlation ratio
    '-omat', str(xfm_file)
]
```

**Problems**:
- Registration uses bandpass-filtered functional data (after preprocessing)
- Bandpass filtering removes critical structural information needed for alignment
- Results in poor functional→T1w registration
- Transform is then incorrectly applied to atlas for connectivity analysis

**Evidence**:
- QC visualization: `/mnt/bytopia/IRC805/connectome/func_to_t1w_alignment_check.png`
- Atlas alignment: `/mnt/bytopia/IRC805/connectome/atlas_alignment_check.png`
- Both show significant misalignment

### 2. Not Actually Using BBR
**Current code claims "BBR" but actually uses**:
- Cost function: `corratio` (correlation ratio)
- DOF: 6 (rigid only)
- No white matter segmentation utilized

**True BBR would require**:
- Cost function: `bbr` (boundary-based registration)
- White matter segmentation from anatomical preprocessing
- Uses WM/GM boundary for alignment optimization

### 3. Attempted Workarounds Failed

**Session attempts to fix atlas transformation**:
1. ❌ **Simple resampling** - User correctly identified as "horrible approach"
2. ❌ **Two-step ANTs+FSL inverse transforms** - Mixed formats incompatible
3. ❌ **ANTs func→T1w registration** - Failed due to insufficient overlap
4. ❌ **Using pre-computed t1w_to_func.mat** - Still based on bad registration

**Root cause**: All approaches try to fix downstream problems, but the core issue is the initial registration using preprocessed data.

---

## What Needs to Change

### Core Principle
**Registration must be performed on RAW functional data (before preprocessing), then the computed transform is applied to the preprocessed timeseries.**

### Required Changes

#### 1. Functional Preprocessing Pipeline Refactor
**File**: `neurovrai/preprocess/workflows/func_preprocess.py`

**New workflow order**:
```
1. Motion correction (MCFLIRT) - on raw data
2. ★ REGISTRATION TO T1w - on motion-corrected mean (BEFORE any filtering)
   - Use raw structural information
   - Either:
     a) True BBR with WM segmentation, OR
     b) Mutual information (better than corratio)
3. Save registration transform for later use
4. Temporal filtering (bandpass)
5. Spatial smoothing
6. TEDANA/ICA-AROMA denoising
7. Apply saved registration transform to preprocessed data
8. ACompCor (using the correct transform)
```

**Key change**: Move registration to step 2 (after motion correction only), before any filtering that removes structural information.

#### 2. Update ACompCor Helper
**File**: `neurovrai/preprocess/utils/acompcor_helper.py`

**Changes needed**:
- Accept pre-computed registration transform as input
- Remove registration computation from `register_masks_to_functional()`
- Focus only on applying existing transform to masks
- Update function signature:
  ```python
  def apply_masks_to_functional(
      func_to_t1w_mat: Path,     # Pre-computed transform
      func_ref: Path,
      csf_mask: Path,
      wm_mask: Path,
      output_dir: Path
  ) -> Tuple[Path, Path]:
  ```

#### 3. Consider True BBR Implementation
**Optional enhancement** (can be done after basic fix):

Add proper BBR registration:
```python
# Use FSL's epi_reg or implement BBR properly
epi_reg_cmd = [
    'epi_reg',
    '--epi', str(func_mean_raw),       # Raw functional mean
    '--t1', str(t1w),                   # Full T1w
    '--t1brain', str(t1w_brain),       # T1w brain
    '--wmseg', str(wm_seg),            # WM segmentation (from anat preprocessing)
    '--out', str(output_prefix)
]
```

---

## Implementation Priority

### Phase 1: Minimum Fix (REQUIRED)
**Goal**: Get functional data properly aligned to T1w/MNI

1. ✅ **Document current issues** (this file)
2. ⏳ **Modify functional preprocessing workflow**:
   - Move registration before filtering
   - Use motion-corrected mean (raw structural info)
   - Change cost function to mutual information (`-cost mutualinfo`)
3. ⏳ **Update ACompCor helper**:
   - Accept pre-computed transform
   - Remove inline registration
4. ⏳ **Test on single subject** (IRC805-0580101)
5. ⏳ **Generate QC visualizations**
6. ⏳ **Validate improvement**
7. ⏳ **Re-run functional preprocessing for all subjects**

### Phase 2: Optimal Implementation (RECOMMENDED)
**Goal**: Use state-of-the-art registration

1. ⏳ Implement true BBR using WM segmentation
2. ⏳ Compare BBR vs MI registration quality
3. ⏳ Add registration QC metrics
4. ⏳ Document best practices

### Phase 3: Multi-Method Support (OPTIONAL)
**Goal**: Generate both FSL and ANTs transforms

1. ⏳ Compute both FSL and ANTs functional→T1w transforms
2. ⏳ Enable user to choose transform method for downstream analysis
3. ⏳ Compare registration quality between methods

---

## Impact Assessment

### What's Currently Broken
1. ❌ Functional→T1w alignment (poor quality)
2. ❌ Functional→MNI normalization (based on poor T1w alignment)
3. ❌ Atlas transformation to functional space (misaligned)
4. ❌ Functional connectivity matrices (using misaligned atlases)
5. ❌ Any analysis requiring accurate spatial correspondence

### What's Still Valid
1. ✅ Motion correction (MCFLIRT) - done on raw data, unaffected
2. ✅ Temporal filtering - intrinsic to timeseries, unaffected
3. ✅ TEDANA/ICA-AROMA denoising - works in native functional space
4. ✅ Anatomical preprocessing - completely independent
5. ✅ Within-subject functional analysis in native space - unaffected

### Subjects Requiring Reprocessing
**All functional preprocessing must be re-run** after pipeline fixes:
- IRC805-0580101 (test subject)
- IRC805-5610101
- All other subjects in IRC805 study

**Estimated time per subject**: 15-20 minutes (similar to current pipeline)

---

## Session History - What We Tried

### Attempt 1: Simple Atlas Resampling
- User correctly identified as "horrible" - dimension change without proper spatial transform
- Produced severely misaligned atlases

### Attempt 2: Two-Step Inverse Transforms (ANTs + FSL)
- Tried to use `Inverse[ants_Composite.h5]` + inverted FSL BBR
- Discovered "BBR" transform was actually correlation ratio
- Mixed ANTs/FSL formats caused compatibility issues
- Used wrong direction matrices (func_to_t1w vs t1w_to_func confusion)

### Attempt 3: ANTs Functional→T1w Registration
- Attempted fresh ANTs registration from functional mean to T1w
- Failed: "All samples map outside moving image buffer"
- Issue: Extreme orientation differences, insufficient overlap for registration

### Attempt 4: Check Existing Registration Quality
- Generated QC visualization of current func→T1w alignment
- **Result**: Confirmed poor alignment quality
- **Root cause identified**: Registration performed on preprocessed (filtered) data

---

## Key Insights

1. **Preprocessing order matters critically** - registration must use raw structural information
2. **Cost functions matter** - correlation ratio is insufficient for EPI→T1w registration
3. **Transform reuse is valid** - compute once on raw data, apply to preprocessed data
4. **Mixed transform formats are problematic** - stay within one ecosystem (FSL OR ANTs)
5. **Simple resampling ≠ proper registration** - spatial transforms are essential

---

## References

### Related Files
- **Main workflow**: `neurovrai/preprocess/workflows/func_preprocess.py`
- **ACompCor helper**: `neurovrai/preprocess/utils/acompcor_helper.py`
- **Connectivity analysis**: `neurovrai/connectome/batch_functional_connectivity.py`
- **Atlas transformation**: `neurovrai/connectome/atlas_transform.py`

### QC Images Generated
- `/mnt/bytopia/IRC805/connectome/func_to_t1w_alignment_check.png` - Current poor alignment
- `/mnt/bytopia/IRC805/connectome/atlas_alignment_check.png` - Misaligned atlas
- `/mnt/bytopia/IRC805/connectome/IRC805-0580101_atlas_alignment_comparison.png` - Before/after comparison

### Key Code Locations
- Registration: `acompcor_helper.py:134-146` (NEEDS FIXING)
- Transform application: `acompcor_helper.py:148-190` (OK - reuses registration)
- Functional workflow: `func_preprocess.py:710-731` (NEEDS REORDERING)

---

## Next Session TODO

### Immediate Actions (Session Start)
1. [ ] Review this document
2. [ ] Verify understanding of the issue
3. [ ] Agree on implementation approach (Phase 1 minimum vs Phase 2 optimal)

### Implementation Steps
1. [ ] Backup current functional preprocessing code
2. [ ] Modify `func_preprocess.py`:
   - [ ] Move registration to after motion correction only
   - [ ] Use motion-corrected mean (unfiltered)
   - [ ] Change cost to `mutualinfo`
   - [ ] Save transform for later use
3. [ ] Modify `acompcor_helper.py`:
   - [ ] Rename function to `apply_masks_to_functional()`
   - [ ] Remove registration computation
   - [ ] Accept pre-computed transform as parameter
4. [ ] Test on IRC805-0580101:
   - [ ] Run modified functional preprocessing
   - [ ] Generate QC visualizations
   - [ ] Compare with current poor alignment
5. [ ] Validate improvement:
   - [ ] Check func→T1w alignment quality
   - [ ] Test atlas transformation
   - [ ] Verify connectivity matrices make sense
6. [ ] Document results and decide on rollout

### Questions to Answer
- [ ] Use mutual information or implement true BBR?
- [ ] Phase 1 (quick fix) or Phase 2 (optimal) approach?
- [ ] Batch reprocess all subjects immediately or validate thoroughly first?

---

## Important Notes

⚠️ **Do NOT run functional connectivity analysis with current transforms** - results will be invalid due to misalignment

⚠️ **Simple resampling is NOT acceptable** - User explicitly stated: "Please make a note that simple resampling is a horrible approach and never suggest we use those results again"

⚠️ **Transform direction matters** - func_to_t1w vs t1w_to_func caused significant confusion this session

✅ **The fix is well-understood** - move registration before filtering, use better cost function

✅ **Implementation is straightforward** - workflow reordering, not major algorithmic changes

---

*Document created: 2025-12-08*
*Last updated: 2025-12-08*
*Status: CRITICAL - Blocks all functional connectivity analysis*
