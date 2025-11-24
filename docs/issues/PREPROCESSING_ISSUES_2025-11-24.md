# DWI Preprocessing Workflow Issues

**Date**: 2025-11-24
**Context**: Investigation of 6 missing subjects in TBSS analysis
**Status**: Root causes identified, fixes needed

---

## Summary

While investigating missing DTI subjects in TBSS analysis, we discovered multiple issues in the preprocessing workflow that prevent processing of:
- Single-shell DWI data (b=800, b=1000, etc.)
- Data without reverse phase-encoding images (no TOPUP)
- Data with scanner-processed derivative maps mixed with raw data

**Impact**: 5 of 6 missing subjects have recoverable raw DWI data but cannot be preprocessed with current workflow.

---

## Issue 1: Scanner-Processed Map Filtering ✅ FIXED

### Problem
BIDS directories can contain both raw DWI data and scanner-processed derivative maps:
- **Raw DWI**: `*_WIP_DTI_*.nii.gz` with `.bval` and `.bvec` files
- **Scanner-processed**: `*__ADC.nii.gz`, `*dWIP_DTI*.nii.gz`, `*facWIP_DTI*.nii.gz`, `*isoWIP_DTI*.nii.gz`

The file discovery pattern `dwi_dir.glob('*DTI*.nii.gz')` matched BOTH types, causing:
```
ERROR: *.bval not found (scanner-processed maps don't have bval/bvec files)
```

### Fix Applied
**File**: `run_simple_pipeline.py` lines 118-137

Added filtering to exclude scanner-processed maps:
```python
# Filter out scanner-processed maps (ADC, dWIP, facWIP, isoWIP)
scanner_processed_patterns = ['__ADC', 'dWIP_DTI', 'facWIP_DTI', 'isoWIP_DTI']

dwi_files = []
for f in all_dwi_files:
    is_scanner_processed = any(pattern in f.name for pattern in scanner_processed_patterns)
    if not is_scanner_processed:
        dwi_files.append(f)
    else:
        logger.info(f"  Skipping scanner-processed map: {f.name}")
```

### Testing
✅ Tested on IRC805-2350101: Correctly filtered 4 scanner-processed maps, identified 1 raw DWI file

---

## Issue 2: Missing Config Path - `paths.logs` ✅ FIXED

### Problem
Preprocessing workflows require `config['paths']['logs']` but this was not documented or included in example configs.

**Error**:
```python
KeyError: 'logs'
# In: log_dir = Path(config['paths']['logs'])
```

**Affected Files**:
- `neurovrai/preprocess/workflows/dwi_preprocess.py` line 1055
- `neurovrai/preprocess/workflows/anat_preprocess.py` line 440
- Likely all preprocessing workflows

### Fix Applied
**File**: `config.yaml`

Added to paths section:
```yaml
paths:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
  mni152_t1_1mm: /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
  fmrib58_fa: /usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz
  fmrib58_fa_mask: /usr/local/fsl/data/standard/FMRIB58_FA-skeleton_1mm.nii.gz
  logs: ${project_dir}/logs  # <-- ADDED
```

### Recommendation
Update `CLAUDE.md` and example configs to include `paths.logs` as a required field.

---

## Issue 3: Eddy Without TOPUP - Missing acqparams.txt ❌ NEEDS FIX

### Problem
When running eddy correction **without** TOPUP (no reverse phase-encoding images), the workflow fails:

**Error**:
```
ERROR: The 'in_acqp' trait of an EddyInputSpec instance must be a pathlike
object or string representing an existing file, but a value of 'None' was specified.
```

**Root Cause**: FSL's `eddy` always requires an `acqparams.txt` file, even without TOPUP. The workflow only generates this file when TOPUP is enabled.

### Expected Behavior
For single-direction acquisitions (AP or PA only, no reverse PE):
1. Auto-generate `acqparams.txt` with single line for the acquisition direction
2. Auto-generate `index.txt` mapping all volumes to line 1
3. Pass these files to eddy

### Example Files Needed

**acqparams.txt** (for AP-only acquisition):
```
0 -1 0 0.05
```
Where:
- `0 -1 0` = phase-encoding direction (AP)
- `0.05` = readout time in seconds

**index.txt** (for 33-volume acquisition):
```
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```

### Affected Code
**File**: `neurovrai/preprocess/workflows/dwi_preprocess.py`

The workflow has this logic:
```python
if rev_phase_files:
    # Generate TOPUP files and run TOPUP
    # Then run eddy with TOPUP integration
else:
    # Skip TOPUP, run eddy without distortion correction
    # ❌ BUG: acqparams.txt and index.txt are NOT generated
    eddy_node = ... # in_acqp=None → FAILS
```

### Proposed Fix
**Location**: `neurovrai/preprocess/workflows/dwi_preprocess.py`

Add a branch for eddy-without-TOPUP:
```python
if rev_phase_files:
    # Existing TOPUP path
    acqparams_file, index_file = create_topup_files_for_multishell(...)
    run_topup(...)
    run_eddy_with_topup(acqparams_file, index_file, topup_results)
else:
    # NEW: Generate files for eddy-only mode
    logger.info("Generating acquisition parameters for eddy (no TOPUP)")
    acqparams_file, index_file = create_eddy_files_single_direction(
        dwi_files=merged_dwi,
        pe_direction=config['diffusion']['topup']['pe_direction'],  # e.g., 'AP'
        readout_time=config['diffusion']['topup']['readout_time'],
        output_dir=work_dir / 'eddy_params'
    )
    run_eddy_without_topup(acqparams_file, index_file)
```

### New Helper Function Needed
**File**: `neurovrai/preprocess/utils/topup_helper.py`

```python
def create_eddy_files_single_direction(
    dwi_file: Path,
    pe_direction: str,
    readout_time: float,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Generate acqparams.txt and index.txt for eddy without TOPUP.

    For single phase-encoding direction acquisitions where no
    reverse PE images are available.

    Args:
        dwi_file: Path to merged DWI file
        pe_direction: Phase encoding direction ('AP', 'PA', 'LR', 'RL')
        readout_time: Total readout time in seconds
        output_dir: Directory to save parameter files

    Returns:
        Tuple of (acqparams_file, index_file) paths
    """
    # Get number of volumes
    img = nib.load(dwi_file)
    n_volumes = img.shape[3]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map PE direction to vector
    pe_vectors = {
        'AP': '0 -1 0',
        'PA': '0 1 0',
        'LR': '-1 0 0',
        'RL': '1 0 0'
    }
    pe_vector = pe_vectors[pe_direction]

    # Create acqparams.txt (single line)
    acqparams_file = output_dir / 'acqparams.txt'
    with open(acqparams_file, 'w') as f:
        f.write(f"{pe_vector} {readout_time}\n")

    # Create index.txt (all volumes map to line 1)
    index_file = output_dir / 'index.txt'
    with open(index_file, 'w') as f:
        f.write(' '.join(['1'] * n_volumes) + '\n')

    logger.info(f"  Created acqparams: {acqparams_file}")
    logger.info(f"  Created index: {index_file}")
    logger.info(f"  Direction: {pe_direction}, Volumes: {n_volumes}")

    return acqparams_file, index_file
```

### Config Requirements
Add to diffusion config:
```yaml
diffusion:
  topup:
    readout_time: 0.05  # Required
    pe_direction: AP     # Required: 'AP', 'PA', 'LR', or 'RL'
```

---

## Issue 4: Silent Failures - Error Handling ❌ NEEDS FIX

### Problem
The preprocessing pipeline reports success even when preprocessing fails internally:

**Batch script output**:
```bash
ERROR: ✗ DWI preprocessing failed: 'logs'
...
✓ IRC805-2350101 completed successfully  # <-- Wrong!
```

This is because:
1. Python scripts catch exceptions and return exit code 0
2. Bash script only checks exit codes: `if [ $? -eq 0 ]`
3. FA map check happens but is advisory only

### Proposed Fix
**File**: `run_simple_pipeline.py`

Change error handling to propagate failures:
```python
def preprocess_dwi(...):
    try:
        results = run_dwi_multishell_topup_preprocessing(...)

        # Verify critical outputs exist
        fa_file = output_dir / subject / 'dwi' / 'dti_FA.nii.gz'
        if not fa_file.exists():
            logger.error("Preprocessing completed but FA map not found")
            return None  # Signal failure

        logger.info("✓ DWI preprocessing complete")
        return results

    except Exception as e:
        logger.error(f"✗ DWI preprocessing failed: {e}")
        return None  # Signal failure

def main():
    # ...
    dwi_results = preprocess_dwi(...)

    if dwi_results is None:
        logger.error("Pipeline failed at DWI preprocessing")
        sys.exit(1)  # <-- Return non-zero exit code
```

**Benefits**:
- Bash scripts can properly detect failures
- Failed subjects can be retried automatically
- CI/CD pipelines can catch errors

---

## Issue 5: Inconsistent Orientations Warning

### Problem
Some subjects have multiple acquisitions with inconsistent orientations:

```
WARNING: Inconsistent orientations for individual images when attempting to merge.
         Merge will use voxel-based orientation which is probably incorrect - *PLEASE CHECK*!
```

**Example**: IRC805-3580101 has 2 b=800 acquisitions (likely one truncated/repeated)

### Current Behavior
FSL's `fslmerge` continues with voxel-based orientation, which may produce incorrect results.

### Recommended Enhancement
Add validation and handling options:

**File**: `neurovrai/preprocess/workflows/dwi_preprocess.py`

```python
def validate_dwi_orientations(dwi_files: List[Path]) -> Dict:
    """
    Check if DWI files have consistent orientations.

    Returns:
        {
            'consistent': bool,
            'orientations': List[str],  # Orientation codes
            'warning': Optional[str]
        }
    """
    orientations = []
    for dwi_file in dwi_files:
        img = nib.load(dwi_file)
        orient = nib.aff2axcodes(img.affine)
        orientations.append(orient)

    consistent = len(set(orientations)) == 1

    if not consistent:
        warning = (
            f"Inconsistent orientations detected: {orientations}\n"
            f"Files: {[f.name for f in dwi_files]}\n"
            f"Consider using only one acquisition or manually reorienting."
        )
        return {'consistent': False, 'orientations': orientations, 'warning': warning}

    return {'consistent': True, 'orientations': orientations, 'warning': None}

# In main workflow:
if len(dwi_files) > 1:
    orientation_check = validate_dwi_orientations(dwi_files)
    if not orientation_check['consistent']:
        logger.warning(orientation_check['warning'])

        # Option 1: Fail and require manual intervention
        if config['diffusion'].get('strict_orientation_check', False):
            raise ValueError("Inconsistent orientations - manual review required")

        # Option 2: Use only the first acquisition
        if config['diffusion'].get('use_first_acquisition_only', False):
            logger.warning(f"Using only first acquisition: {dwi_files[0].name}")
            dwi_files = [dwi_files[0]]
            # Also filter bval/bvec files
```

---

## Issue 6: Size Mismatch During Merge

### Problem
Some subjects have multiple acquisitions with different matrix sizes that cannot be merged:

```
Error in size-match along non-concatenated dimension for input file
```

**Example**: IRC805-4960101 has 3 separate b=800 acquisitions with incompatible dimensions

### Current Behavior
FSL's `fslmerge` fails, pipeline returns error.

### Recommended Enhancement
Add pre-merge validation:

```python
def validate_dwi_dimensions(dwi_files: List[Path]) -> Dict:
    """
    Check if DWI files have compatible dimensions for merging.

    Returns:
        {
            'compatible': bool,
            'shapes': List[Tuple],  # [(x,y,z,t), ...]
            'error': Optional[str]
        }
    """
    shapes = []
    for dwi_file in dwi_files:
        img = nib.load(dwi_file)
        shapes.append(img.shape[:3])  # Spatial dimensions only

    compatible = len(set(shapes)) == 1

    if not compatible:
        error = (
            f"Incompatible matrix sizes detected:\n" +
            "\n".join([f"  {f.name}: {s}" for f, s in zip(dwi_files, shapes)]) +
            "\n\nThese acquisitions cannot be merged. "
            "Consider processing separately or excluding."
        )
        return {'compatible': False, 'shapes': shapes, 'error': error}

    return {'compatible': True, 'shapes': shapes, 'error': None}
```

---

## Missing Subject Summary

| Subject | Data Type | Status | Issue |
|---------|-----------|--------|-------|
| IRC805-2350101 | Single-shell b=800 | ⚠️ Has data | Needs eddy-without-TOPUP fix |
| IRC805-2990202 | None | ❌ No data | No DTI acquisition |
| IRC805-3280201 | Single-shell b=800 | ⚠️ Has data | Needs eddy-without-TOPUP fix |
| IRC805-3580101 | 2x b=800 | ⚠️ Has data | Orientation mismatch, eddy fix needed |
| IRC805-3840101 | Multi-shell + TOPUP | ✅ Ready | Should work after paths.logs fix |
| IRC805-4960101 | 3x b=800 | ❌ Incompatible | Size mismatch, cannot merge |

---

## Testing Plan

### Phase 1: Config and Error Handling
1. ✅ Add `paths.logs` to config.yaml
2. ✅ Test config validation passes
3. ⬜ Fix return codes in `run_simple_pipeline.py`
4. ⬜ Test batch script properly detects failures

### Phase 2: Eddy Without TOPUP
1. ⬜ Implement `create_eddy_files_single_direction()`
2. ⬜ Add eddy-without-TOPUP branch to `dwi_preprocess.py`
3. ⬜ Test on IRC805-2350101 (single-shell, no reverse PE)
4. ⬜ Test on IRC805-3280201 (single-shell, no reverse PE)
5. ⬜ Verify FA maps are created and look reasonable

### Phase 3: TOPUP Path
1. ⬜ Test IRC805-3840101 (multi-shell + TOPUP)
2. ⬜ Verify TOPUP distortion correction works
3. ⬜ Verify eddy with TOPUP integration works
4. ⬜ Compare FA maps with/without TOPUP

### Phase 4: Edge Cases
1. ⬜ Add orientation validation
2. ⬜ Add dimension validation
3. ⬜ Test IRC805-3580101 with orientation check
4. ⬜ Document IRC805-4960101 as unprocessable

### Phase 5: Integration
1. ⬜ Run batch processing with all fixes
2. ⬜ Verify FA maps in derivatives
3. ⬜ Run TBSS preparation with new subjects
4. ⬜ Compare TBSS results: 17 → 21 subjects

---

## Documentation Updates Needed

1. **CLAUDE.md**:
   - Add `paths.logs` to required config fields
   - Document scanner-processed map filtering
   - Note single-shell vs multi-shell handling

2. **README.md**:
   - Update config.yaml example
   - Add troubleshooting section for common errors

3. **config.yaml examples**:
   - Add `paths.logs: ${project_dir}/logs`
   - Add `diffusion.topup.pe_direction` requirement

---

## Priority

**High Priority** (blocking):
1. Eddy-without-TOPUP fix (Issue 3)
2. Error handling fix (Issue 4)

**Medium Priority** (quality of life):
3. Orientation validation (Issue 5)
4. Dimension validation (Issue 6)

**Low Priority** (documentation):
5. Update docs and examples
6. Add integration tests

---

## Related Files

**Preprocessing Workflows**:
- `neurovrai/preprocess/workflows/dwi_preprocess.py`
- `run_simple_pipeline.py`

**Utilities**:
- `neurovrai/preprocess/utils/topup_helper.py`

**Configuration**:
- `config.yaml`

**Documentation**:
- `CLAUDE.md`
- `README.md`

---

## Contact

Issues discovered during: TBSS missing subjects investigation
Session date: 2025-11-24
Context: Preparing IRC805 DTI data for tract-based spatial statistics
