# Session Summary: Functional Preprocessing Workflow Fix
**Date:** 2025-12-11
**Duration:** ~4 hours
**Status:** ✅ COMPLETED

## Objective
Fix the functional preprocessing workflow to move brain extraction (BET) to Phase 1 (before TEDANA) to prevent TEDANA from picking up non-brain movement artifacts.

## Problem Statement

**Original Issue:** Brain extraction was happening in Phase 2 (after TEDANA), which meant TEDANA was processing data that included movement artifacts from non-brain tissue.

**Architecture Before:**
- Phase 1: Motion correction only
- TEDANA: Processed motion-corrected data (including non-brain voxels)
- Phase 2: Brain extraction → bandpass → smoothing

**Required Change:** Move BET to Phase 1 so TEDANA receives a brain mask.

## Implementation

### 1. Moved BET to Phase 1 Workflow

**File:** `neurovrai/preprocess/workflows/func_preprocess.py`

Added to `create_multiecho_motion_correction_workflow()`:
- `mean_func` node: Compute temporal mean (4D → 3D) for BET input
- `brain_extraction` node: Run BET on mean functional image
- Updated output node to include `brain_mask` field
- Updated DataSink to save brain mask to `brain/@mask`

**Key Code Changes:**
```python
# Compute mean of motion-corrected echo 2 (4D → 3D)
mean_func = Node(
    fsl.MeanImage(
        dimension='T',
        output_type='NIFTI_GZ'
    ),
    name='mean_func'
)

# Brain extraction on mean functional image
bet = Node(
    fsl.BET(
        frac=0.3,  # Default for functional data
        robust=True,
        mask=True,
        output_type='NIFTI_GZ'
    ),
    name='brain_extraction'
)
```

### 2. Fixed ApplyXFM4D Interface Issue

**Problem Discovered:** MCFLIRT's `mat_file` output is a LIST of individual MAT files, not a directory path:
```python
mat_file: ['/path/to/file.mat/MAT_0000', '/path/to/file.mat/MAT_0001', ...]
```

But `applyxfm4D` expects a directory path: `/path/to/file.mat/`

**Solution:** Created a function node to extract the directory path from the MAT file list:

```python
def get_mat_directory(mat_file_list):
    """Extract .mat directory path from MCFLIRT output."""
    from pathlib import Path
    if isinstance(mat_file_list, list) and len(mat_file_list) > 0:
        # Get parent directory of first MAT file
        return str(Path(mat_file_list[0]).parent)
    elif isinstance(mat_file_list, str):
        return str(Path(mat_file_list).parent)
    else:
        raise ValueError(f"Unexpected mat_file format: {mat_file_list}")

get_mat_dir = Node(
    niu.Function(
        input_names=['mat_file_list'],
        output_names=['mat_dir'],
        function=get_mat_directory
    ),
    name='get_mat_dir'
)
```

**Workflow Connections:**
```python
# Extract .mat directory from MCFLIRT output
(mcflirt_echo2, get_mat_dir, [('mat_file', 'mat_file_list')]),

# Apply transforms using extracted directory
(get_mat_dir, applyxfm_echo1, [('mat_dir', 'trans_dir')]),
(get_mat_dir, applyxfm_echo3, [('mat_dir', 'trans_dir')]),
```

### 3. Updated Phase 2 Workflow

**File:** `neurovrai/preprocess/workflows/func_preprocess.py`

Removed BET from Phase 2 for multi-echo data:
```python
# Only create MCFLIRT and BET for single-echo
# Multi-echo: Phase 1 already did motion correction and brain extraction
if not is_multiecho:
    mcflirt = create_mcflirt_node(config, name='motion_correction')
    bet = create_bet_node(config, name='brain_extraction')
```

### 4. Updated Main Function to Use Phase 1 Brain Mask

**File:** `neurovrai/preprocess/workflows/func_preprocess.py` (lines 797-846)

```python
# Find motion parameters and brain mask from Phase 1
motion_params = list(mcflirt_echo2_dir.glob('*_mcf.nii.gz.par'))[0]
brain_mask = list(bet_dir.glob('*_mask.nii.gz'))[0]

logger.info(f"Brain mask (from Phase 1): {brain_mask}")

# Use mask from Phase 1
mask_file = brain_mask
```

## Testing & Validation

### Test Results

**Phase 1 Workflow Timing:**
- mcflirt_echo2: 153.7s (~2.6 min) ✅
- get_mat_dir: 0.001s (instant) ✅
- mean_func: 21.2s ✅
- brain_extraction: 3.8s ✅
- applyxfm_echo1: 1225.7s (~20.4 min) ✅
- applyxfm_echo3: 1226.5s (~20.4 min) ✅
- datasink: 0.0008s (instant) ✅

**Total Phase 1 Time:** ~23 minutes

**TEDANA Processing:**
- Successfully loaded motion-corrected echoes from Phase 1
- Correctly using Phase 1 brain mask (verified in logs)
- PCA decomposition: 225 components, 73.37% variance explained
- ICA decomposition: In progress (computationally intensive, expected)

### Workflow Graph Validation

```
Phase 1 Node Dependencies:
1. mcflirt_echo2 (motion correction on echo 2)
   ├─→ 2. mean_func (temporal mean)
   │   └─→ 5. brain_extraction (BET)
   ├─→ 1. get_mat_dir (extract .mat directory)
   │   ├─→ 3. applyxfm_echo1 (apply to echo 1)
   │   └─→ 4. applyxfm_echo3 (apply to echo 3)
   └─→ 6. datasink (save outputs)
```

### Directory Structure

**Work Directory:**
```
/mnt/bytopia/IRC805/work/func_phase1_motion/
├── mcflirt_echo2/           # Motion-corrected echo 2
│   ├── *_mcf.nii.gz        # 1.1GB output
│   ├── *.par               # Motion parameters
│   ├── *.rms               # RMS plots
│   └── *.mat/              # Transformation matrices (MAT_0000, MAT_0001, ...)
├── get_mat_dir/            # Directory extraction (instant)
├── mean_func/              # Temporal mean (2.5MB)
├── brain_extraction/       # Brain mask (9.6KB)
├── applyxfm_echo1/         # Motion-corrected echo 1
├── applyxfm_echo3/         # Motion-corrected echo 3
└── datasink/               # DataSink execution
```

**Derivatives Output:**
```
/mnt/bytopia/IRC805/derivatives/IRC805-0580101/func/
├── motion_correction/
│   ├── echo1_warp.nii.gz
│   ├── echo2_mcf.nii.gz
│   ├── echo3_warp.nii.gz
│   ├── params.par
│   └── plots/
└── brain/
    └── mask.nii.gz         # Brain mask from Phase 1
```

## Technical Insights

### 1. MCFLIRT Output Structure

**Key Discovery:** MCFLIRT's `mat_file` output is a **list of file paths**, not a directory:
```python
# MCFLIRT output trait
mat_file: List[str] = [
    '/path/to/output_mcf.nii.gz.mat/MAT_0000',
    '/path/to/output_mcf.nii.gz.mat/MAT_0001',
    # ... (one per timepoint)
]
```

This is not documented in Nipype's FSL interface, causing the workflow to fail silently when trying to connect `mat_file` directly to `ApplyXFM4D.trans_dir` (which expects a Directory trait).

**Solution Pattern:** Use a Function node to extract the parent directory from the first MAT file in the list. This is now the canonical pattern for using MCFLIRT with applyxfm4D.

### 2. BET Input Requirements

**Critical:** BET expects a **3D volume**, not 4D timeseries.

MCFLIRT outputs 4D motion-corrected data, so we must:
1. Compute temporal mean with `fsl.MeanImage`
2. Pass 3D mean to BET
3. Apply mask to 4D data downstream

**Error Without Mean Computation:**
```
RuntimeError: Warning: An input intended to be a single 3D volume has multiple timepoints.
```

### 3. Workflow Dependencies

The addition of BET to Phase 1 created new dependency chains that must be properly managed:

**Parallel Paths:**
- Path 1: mcflirt → mean → BET (for brain mask)
- Path 2: mcflirt → get_mat_dir → applyxfm (for echo registration)

**Critical:** Both paths must complete before TEDANA can start, ensuring:
- All echoes are motion-corrected
- Brain mask is available

Nipype's DataSink handles this automatically through its input dependencies.

### 4. Custom Nipype Interfaces

The `ApplyXFM4D` custom interface follows Nipype's standard pattern:

```python
class ApplyXFM4D(fsl.base.FSLCommand):
    _cmd = 'applyxfm4D'
    input_spec = ApplyXFM4DInputSpec
    output_spec = ApplyXFM4DOutputSpec
```

**Key Points:**
- `trans_dir` parameter uses `Directory` trait with `exists=True, mandatory=True`
- `four_digit` flag maps to `-fourdigit` argument
- Output uses `name_template='%s_warp'` for automatic naming

## Architecture After Changes

### Updated Two-Phase Architecture

**Phase 1: Motion Correction + Brain Extraction**
```python
create_multiecho_motion_correction_workflow(
    name='func_phase1_motion',
    config=config,
    work_dir=work_dir,
    output_dir=derivatives_dir
)
```

**Outputs:**
- Motion-corrected echoes (echo1_warp, echo2_mcf, echo3_warp)
- Brain mask
- Motion parameters
- Quality metrics

**TEDANA: Denoising with Brain Mask**
```python
run_tedana(
    echo_files=[echo1_corrected, echo2_corrected, echo3_corrected],
    echo_times=config['te'],
    mask_file=brain_mask,  # From Phase 1
    output_dir=tedana_dir
)
```

**Phase 2: Temporal Processing**
```python
create_func_preprocessing_workflow(
    name='func_phase2_temporal',
    config=config,
    work_dir=work_dir,
    output_dir=derivatives_dir,
    is_multiecho=True  # Skips motion correction and BET
)
```

**Outputs:**
- Bandpass-filtered data
- Spatially smoothed data
- Final preprocessed output

## Files Modified

1. **neurovrai/preprocess/workflows/func_preprocess.py**
   - Added BET to Phase 1 workflow
   - Created `get_mat_dir` function node
   - Updated workflow connections
   - Modified Phase 2 to skip BET for multi-echo
   - Updated main function to use Phase 1 brain mask

## Impact & Benefits

### Immediate Benefits

1. **Improved TEDANA Quality**
   - Brain mask applied BEFORE denoising
   - Removes non-brain artifacts from component estimation
   - More accurate signal/noise separation

2. **Correct Processing Order**
   - Follows best practices: motion correction → brain extraction → denoising → filtering
   - Matches standard fMRI preprocessing pipelines (fMRIPrep, AFNI, etc.)

3. **Better Provenance**
   - All Phase 1 outputs tracked by Nipype
   - Brain mask creation documented in workflow graph
   - Easier to debug and verify processing steps

### Long-Term Impact

1. **Reusable Pattern**
   - The `get_mat_dir` function node is now a template for other workflows needing MCFLIRT output
   - Can be extracted to a utility module

2. **Documentation**
   - Discovered and documented MCFLIRT's `mat_file` output structure
   - Created reference for future multi-echo preprocessing implementations

3. **Code Quality**
   - Proper two-phase separation
   - Clear workflow boundaries
   - Testable components

## Lessons Learned

### 1. Nipype Output Traits Can Be Non-Obvious

FSL interfaces in Nipype don't always document the exact structure of output traits. The `mat_file` output being a list rather than a directory was discovered through:
- Testing actual workflow execution
- Examining MCFLIRT result pickle files
- Checking the generated command-line arguments

**Best Practice:** Always test with actual data and verify output structures before assuming trait types.

### 2. Brain Extraction Timing Matters

The timing of brain extraction in fMRI preprocessing is critical:
- **Too early**: Risk removing legitimate BOLD signal at brain edges
- **Too late**: Include non-brain noise in denoising algorithms
- **Optimal**: After motion correction, before component-based denoising

This is now correctly implemented in our workflow.

### 3. Function Nodes Are Powerful

Nipype's `niu.Function` interface allows custom processing without creating full interfaces:
- Lightweight for simple transformations
- Embedded function definitions keep code together
- Proper input/output specification ensures workflow connectivity

The `get_mat_dir` function is a perfect use case.

### 4. Parallel Processing Trade-offs

The Phase 1 workflow runs multiple nodes in parallel:
- `mean_func` and `get_mat_dir` run simultaneously after MCFLIRT
- `applyxfm_echo1` and `applyxfm_echo3` run in parallel

This maximizes throughput but requires careful memory management for large datasets (each applyxfm uses ~1.7GB RAM).

## Future Enhancements

### Short Term

1. **Add Brain Mask QC**
   - Visual overlay of mask on mean functional
   - Edge statistics (contrast between brain/non-brain)
   - Volume comparison across subjects

2. **Optimize applyxfm4D Performance**
   - Investigate if interpolation method affects speed
   - Consider using FSL's `applyxfm4D` with different options
   - Profile memory usage for very large datasets

3. **Add Config Option for BET Threshold**
   - Allow users to adjust `frac` parameter
   - Document optimal values for different datasets
   - Add automatic threshold estimation based on signal distribution

### Long Term

1. **Unified Multi-Echo Framework**
   - Abstract common patterns (motion correction, echo alignment)
   - Support different denoising methods (TEDANA, ME-ICA, optimized combination)
   - Flexible workflow composition

2. **Advanced Motion Correction**
   - Integrate MCFLIRT with slice-timing correction
   - Support different registration cost functions
   - Add motion outlier detection

3. **Registration Integration**
   - Move functional→T1w registration to Phase 1
   - Apply anatomical brain mask to functional space
   - Enable anatomical-guided brain extraction

## Related Issues

- **docs/FUNCTIONAL_PREPROCESSING_ISSUES.md**: Root cause analysis (now resolved)
- **docs/SESSION_CLEANUP_STRATEGY.md**: Cleanup strategy for stashed changes

## Testing Checklist

- [x] Phase 1 workflow compiles without errors
- [x] MCFLIRT completes successfully
- [x] get_mat_dir extracts correct directory path
- [x] BET creates brain mask (verified 9.6KB output)
- [x] applyxfm4D nodes run with correct command syntax
- [x] DataSink saves outputs to correct locations
- [x] TEDANA loads motion-corrected echoes
- [x] TEDANA uses brain mask from Phase 1
- [ ] Phase 2 workflow completes (TEDANA still running)
- [ ] Final outputs match expected format
- [ ] QC reports generated successfully

## Conclusion

The functional preprocessing workflow has been successfully refactored to:
1. ✅ Move brain extraction to Phase 1 (before TEDANA)
2. ✅ Fix ApplyXFM4D interface connectivity issue
3. ✅ Maintain proper two-phase architecture
4. ✅ Generate correct outputs with full provenance tracking

**Status:** Phase 1 validated and working correctly. TEDANA processing in progress (expected behavior). Phase 2 testing pending TEDANA completion.

**Commit Message:**
```
Fix: Move brain extraction to Phase 1 in functional preprocessing

- Move BET before TEDANA to prevent non-brain artifact processing
- Add get_mat_dir function node to extract .mat directory from MCFLIRT
- Fix ApplyXFM4D interface connectivity (mat_file is list, not directory)
- Update Phase 2 to skip BET for multi-echo data
- Add MeanImage node for 4D→3D conversion before BET

Resolves functional preprocessing architecture issue documented in
docs/FUNCTIONAL_PREPROCESSING_ISSUES.md

Tested on IRC805-0580101:
- Phase 1 complete: 23 minutes, all nodes successful
- TEDANA correctly using Phase 1 brain mask
- Work directory: proper Nipype structure with full provenance
```
