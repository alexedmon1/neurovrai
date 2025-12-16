# Session Cleanup & Next Steps Strategy

**Date**: 2025-12-08
**Session Summary**: Atlas transformation attempts → Discovery of fundamental functional preprocessing issues

---

## Files Modified This Session

### Modified Files (Changes to Review)

1. **`neurovrai/connectome/batch_functional_connectivity.py`** (133 lines changed)
   - ✅ **KEEP**: Added atlas definitions (Schaefer, AAL)
   - ✅ **KEEP**: Changed to native functional space approach
   - ❌ **ISSUE**: Uses incorrect atlas transformation (based on bad registration)
   - **Action**: Keep structure, mark as needing functional preprocessing fix

2. **`neurovrai/connectome/atlas_transform.py`** (NEW FILE)
   - ✅ **KEEP**: Well-structured two-step transformation code
   - ❌ **ISSUE**: Based on incorrect func→T1w registration
   - **Action**: Keep file, will work correctly once functional preprocessing is fixed

3. **`neurovrai/connectome/group_analysis.py`** (101 lines)
   - ✅ **KEEP**: Improvements likely unrelated to registration issues
   - **Action**: Review changes, likely general improvements

4. **`neurovrai/connectome/roi_extraction.py`** (22 lines)
   - ✅ **KEEP**: ROI extraction improvements
   - **Action**: Review changes

5. **`neurovrai/connectome/visualization.py`** (64 lines)
   - ✅ **KEEP**: Visualization improvements
   - **Action**: Review changes

6. **`neurovrai/connectome/run_functional_connectivity.py`** (10 lines)
   - ✅ **KEEP**: Minor improvements
   - **Action**: Review changes

7. **`neurovrai/connectome/README.md`** (7 lines)
   - ✅ **KEEP**: Documentation updates
   - **Action**: Review changes

8. **`neurovrai/preprocess/utils/func_normalization.py`** (6 lines)
   - ❓ **REVIEW**: Check if related to incorrect registration approach
   - **Action**: Review carefully

### New Files (Untracked)

1. **`docs/FUNCTIONAL_PREPROCESSING_ISSUES.md`** (NEW)
   - ✅ **KEEP & COMMIT**: Critical documentation of session findings
   - **Action**: Add and commit immediately

2. **`neurovrai/connectome/atlas_labels.py`** (NEW)
   - ✅ **KEEP**: Utility file for atlas labels
   - **Action**: Review and commit if useful

3. **`neurovrai/connectome/batch_graph_metrics.py`** (NEW)
   - ✅ **KEEP**: Graph analysis functionality
   - **Action**: Review and commit

4. **`neurovrai/connectome/batch_group_statistics.py`** (NEW)
   - ✅ **KEEP**: Group statistics functionality
   - **Action**: Review and commit

5. **`neurovrai/connectome/batch_visualization.py`** (NEW)
   - ✅ **KEEP**: Batch visualization utilities
   - **Action**: Review and commit

---

## Current State Assessment

### What Works ✅
- **Atlas transformation infrastructure** - Code is well-structured and will work once registration is fixed
- **Connectivity computation** - ROI extraction and connectivity matrices are valid in native space
- **Group analysis framework** - Analysis code is independent of registration issues
- **Visualization tools** - QC and results visualization work correctly

### What's Broken ❌
- **Functional→T1w registration** - Uses preprocessed data (correlation ratio on bandpass-filtered data)
- **Atlas transformation accuracy** - Based on the broken registration
- **Functional→MNI normalization** - Propagates registration errors
- **Any cross-space analysis** - Cannot trust spatial correspondence

### Root Cause
**Functional preprocessing performs registration on bandpass-filtered data instead of raw motion-corrected data**, removing the structural information needed for accurate alignment.

---

## Strategy for Next Session

### Option 1: Clean Slate Approach (RECOMMENDED)
**Goal**: Start fresh on functional preprocessing fix without worrying about connectivity code

**Steps**:
1. ✅ **Keep documentation**: Commit `FUNCTIONAL_PREPROCESSING_ISSUES.md`
2. ⏸️ **Stash connectivity changes**: Save but don't commit modified files
   ```bash
   git stash push -m "Connectivity work - needs functional preprocessing fix"
   ```
3. ✅ **Commit new utilities**: Add useful new files that aren't registration-dependent
   ```bash
   git add neurovrai/connectome/atlas_labels.py
   git add neurovrai/connectome/batch_*.py
   git commit -m "Add connectivity analysis utilities"
   ```
4. 🎯 **Focus next session**: Fix functional preprocessing registration
5. 🔄 **After fix**: Unstash and update connectivity code to use correct registration

**Advantages**:
- Clean separation of concerns
- Easy to track what depends on functional preprocessing fix
- Can abandon connectivity changes if needed
- Focused next session on core issue

### Option 2: Mark-and-Continue Approach
**Goal**: Keep all work, clearly document what needs updating

**Steps**:
1. ✅ **Add TODO comments** to files that use incorrect registration:
   ```python
   # TODO: This uses incorrect func→T1w registration (see FUNCTIONAL_PREPROCESSING_ISSUES.md)
   # Will work correctly once functional preprocessing is fixed
   ```
2. ✅ **Commit everything** with clear message:
   ```bash
   git add -A
   git commit -m "WIP: Connectivity analysis (requires functional preprocessing fix)

   - Added atlas transformation infrastructure
   - Added connectivity analysis tools
   - NOTE: Currently uses incorrect func→T1w registration
   - See docs/FUNCTIONAL_PREPROCESSING_ISSUES.md for details
   - Must fix functional preprocessing before using these results"
   ```
3. 🎯 **Next session**: Fix functional preprocessing, then update connectivity code

**Advantages**:
- No loss of work
- All context preserved
- Clear documentation of status

**Disadvantages**:
- Easy to accidentally use incorrect registration results
- More clutter in git history

### Option 3: Branch-and-Refocus Approach
**Goal**: Save work on separate branch, clean master for core fix

**Steps**:
1. 🌿 **Create feature branch**:
   ```bash
   git checkout -b feature/connectivity-analysis-needs-registration-fix
   git add -A
   git commit -m "Connectivity analysis (blocked by registration issue)"
   ```
2. 🔄 **Return to master**:
   ```bash
   git checkout master
   ```
3. ✅ **Commit only documentation**:
   ```bash
   git add docs/FUNCTIONAL_PREPROCESSING_ISSUES.md
   git commit -m "Document critical functional preprocessing registration issue"
   ```
4. 🎯 **Next session**: Fix functional preprocessing on master
5. 🔀 **After fix**: Merge or cherry-pick from feature branch

**Advantages**:
- Master branch stays clean
- Clear separation in git history
- Easy to compare before/after registration fix

---

## Recommended Strategy: **Option 1 (Clean Slate)**

**Reasoning**:
1. Core issue (registration) is independent of connectivity code
2. Connectivity code is extensive but all depends on correct registration
3. Simpler to fix registration first, then apply to working connectivity framework
4. Stashing preserves work without cluttering git history
5. Can always retrieve stashed work if needed

---

## Immediate Actions (End of Current Session)

### 1. Commit Critical Documentation
```bash
cd /home/edm9fd/sandbox/neurovrai

# Add and commit the issue documentation
git add docs/FUNCTIONAL_PREPROCESSING_ISSUES.md
git commit -m "CRITICAL: Document functional preprocessing registration issue

- Registration performed on preprocessed (filtered) data
- Should use raw motion-corrected data for registration
- Blocks accurate functional→T1w and functional→MNI normalization
- See docs/FUNCTIONAL_PREPROCESSING_ISSUES.md for full analysis
- Must fix before any cross-space connectivity analysis"
```

### 2. Review and Commit New Utilities (Optional)
```bash
# Review new files that are useful regardless of registration
git add neurovrai/connectome/atlas_labels.py
git add neurovrai/connectome/batch_graph_metrics.py
git add neurovrai/connectome/batch_group_statistics.py
git add neurovrai/connectome/batch_visualization.py

# Commit if they're general-purpose utilities
git commit -m "Add connectome analysis utilities

- atlas_labels.py: Atlas label management
- batch_graph_metrics.py: Graph theory metrics
- batch_group_statistics.py: Group-level statistics
- batch_visualization.py: Batch visualization tools

These utilities are independent of registration issues."
```

### 3. Stash Registration-Dependent Work
```bash
# Stash everything else that depends on correct registration
git stash push -m "Connectivity analysis - awaiting functional preprocessing fix

Files stashed:
- neurovrai/connectome/batch_functional_connectivity.py (atlas transformation)
- neurovrai/connectome/atlas_transform.py (NEW - transform infrastructure)
- neurovrai/connectome/group_analysis.py (improvements)
- neurovrai/connectome/roi_extraction.py (improvements)
- neurovrai/connectome/visualization.py (improvements)
- neurovrai/connectome/run_functional_connectivity.py (updates)
- neurovrai/connectome/README.md (documentation)
- neurovrai/preprocess/utils/func_normalization.py (normalization)

These changes are structurally sound but depend on accurate
functional→T1w registration. Will unstash and update after fixing
functional preprocessing pipeline.

See: docs/FUNCTIONAL_PREPROCESSING_ISSUES.md"
```

### 4. Verify Clean State
```bash
# Should show only the documentation committed
git status

# Should show one stash with connectivity work
git stash list
```

---

## Next Session Start (Recommended Workflow)

### 1. Review Documentation
```bash
# Read the issue documentation
cat docs/FUNCTIONAL_PREPROCESSING_ISSUES.md

# Check what's stashed
git stash show -p
```

### 2. Focus on Functional Preprocessing
- **PRIMARY GOAL**: Fix registration in functional preprocessing
- **DO NOT**: Unstash connectivity work yet
- **STAY FOCUSED**: One issue at a time

### 3. After Functional Preprocessing Fix
```bash
# Test the fixed preprocessing on one subject
# Verify registration quality with QC images
# Once confirmed working:

# Unstash connectivity work
git stash pop

# Update connectivity code to use correct registration
# Test connectivity analysis with correct registration
# Commit final working connectivity analysis
```

---

## Files to Delete/Cleanup

### Temporary Files Created This Session
```bash
# QC images (can regenerate)
rm /mnt/bytopia/IRC805/connectome/atlas_alignment_check.png
rm /mnt/bytopia/IRC805/connectome/func_to_t1w_alignment_check.png

# Temporary test outputs (already logged)
rm /tmp/test_*.log
rm /tmp/atlas_*.nii.gz
rm /tmp/func_*.nii.gz
rm /tmp/check_*.py
rm /tmp/visualize_*.py
```

### Old/Incorrect Analysis Results
```bash
# Connectivity results using incorrect registration
# (keep for comparison, but mark as invalid)
mkdir -p /mnt/bytopia/IRC805/connectome/functional/INVALID_REGISTRATION
mv /mnt/bytopia/IRC805/connectome/functional/IRC805-*/ \
   /mnt/bytopia/IRC805/connectome/functional/INVALID_REGISTRATION/

# Add README explaining why these are invalid
cat > /mnt/bytopia/IRC805/connectome/functional/INVALID_REGISTRATION/README.txt << 'EOF'
These connectivity results were generated with incorrect functional→T1w registration.

ISSUE: Registration was performed on bandpass-filtered functional data,
       which lacks structural information for accurate alignment.

STATUS: INVALID - Do not use these results

SEE: /home/edm9fd/sandbox/neurovrai/docs/FUNCTIONAL_PREPROCESSING_ISSUES.md

These results are preserved only for comparison after fixing the registration.
EOF
```

---

## Summary of Recommended Actions

### End of Current Session ✅
1. [x] Document issue comprehensively (DONE - `FUNCTIONAL_PREPROCESSING_ISSUES.md`)
2. [ ] Commit issue documentation
3. [ ] Review and commit new general-purpose utilities
4. [ ] Stash registration-dependent connectivity work
5. [ ] Move incorrect analysis results to INVALID folder
6. [ ] Clean up temporary files

### Start of Next Session 🎯
1. [ ] Read `FUNCTIONAL_PREPROCESSING_ISSUES.md`
2. [ ] Fix `neurovrai/preprocess/workflows/func_preprocess.py` (move registration before filtering)
3. [ ] Fix `neurovrai/preprocess/utils/acompcor_helper.py` (remove inline registration)
4. [ ] Test on IRC805-0580101
5. [ ] Verify registration quality with QC
6. [ ] Once validated, reprocess all subjects
7. [ ] Unstash connectivity work and update
8. [ ] Re-run connectivity analysis with correct registration

---

## Key Lessons for Future

1. ✅ **Registration order matters** - Always use raw/minimally processed data
2. ✅ **Document critical issues immediately** - This document will save hours next session
3. ✅ **Don't commit broken pipelines** - Stash work until core issues are fixed
4. ✅ **QC early and often** - Caught this issue through QC visualizations
5. ✅ **Simple resampling ≠ proper registration** - User's critical feedback was correct

---

*Document created: 2025-12-08*
*Status: Ready for session cleanup and next session planning*
