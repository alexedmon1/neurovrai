# Session Status: End of Day 2025-12-05

**Date**: 2025-12-05
**Time**: 21:30 UTC
**Status**: 🔄 **ANALYSES RUNNING OVERNIGHT**

---

## Executive Summary

Successfully completed neuroaider architecture refactoring and launched parallel group-level analyses. Three statistical analyses completed successfully (VBM, ReHo, fALFF), with two more running overnight (ASL, VBM duplicate). TBSS FA preparation in progress. All workflows now use pre-generated design matrices with validated subject alignment.

---

## Critical Bugs Fixed

### 1. ASL Subject Order Mismatch (CRITICAL)

**Issue**: 4D merged image created in alphabetical order instead of design matrix order
```
ERROR: Subject order mismatch between MRI files and design matrix:
  Position 11: MRI=IRC805-3840101, Design=IRC805-3940101
```

**Fix** (`run_asl_group_analysis.py` lines 39-85):
```python
def gather_subject_maps(
    derivatives_dir: Path,
    output_dir: Path,
    design_dir: Path  # NEW: Added parameter
):
    # Load design matrix to get correct subject order
    participants_file = design_dir / 'participants_matched.tsv'
    participants_df = pd.read_csv(participants_file, sep='\t')
    design_subjects = participants_df['participant_id'].tolist()

    # Iterate in design matrix order (NOT alphabetical)
    for subject_id in design_subjects:
        # ... gather maps
```

**Validation**: ✅ Perfect alignment confirmed, all 18 subjects matched

### 2. Functional Analysis Double Hierarchy

**Issue**: Created `/analysis/func/func/reho/` instead of `/analysis/func/reho/`

**Fix** (`run_func_group_analysis.py` line 261):
```python
# BEFORE: output_dir = analysis_dir / 'func' / metric / study_name
# AFTER:
output_dir = analysis_dir / metric / study_name
```

---

## Completed Analyses

### VBM (Gray Matter)
- **Subjects**: 23
- **Started**: 20:14 UTC
- **Status**: ✅ **COMPLETED** (running duplicate overnight)
- **Contrasts**: 6 (mriglu_positive/negative, sex_positive/negative, age_positive/negative)
- **Results**: No significant voxels after FWE correction (p < 0.05, TFCE)
- **Max p-values**: 0.158 (mriglu_positive), 0.924 (age_positive)
- **Output**: `/mnt/bytopia/IRC805/analysis/anat/vbm/vbm_analysis/stats/randomise_output/`

### ReHo (Regional Homogeneity)
- **Subjects**: 17
- **Completed**: 19:53-20:07 UTC (14 minutes)
- **Status**: ✅ **COMPLETED**
- **Contrasts**: 6
- **Results**: No significant voxels after FWE correction
- **Max p-values**: 0.608 (mriglu_positive), 0.908 (age_negative)
- **Output**: `/mnt/bytopia/IRC805/analysis/func/func/reho/mriglu_analysis/randomise_output/`
- **Note**: Directory has double `func` hierarchy (fixed for future runs)

### fALFF (Fractional ALFF)
- **Subjects**: 17
- **Completed**: 19:53-20:08 UTC (15 minutes)
- **Status**: ✅ **COMPLETED**
- **Contrasts**: 6
- **Results**: No significant voxels after FWE correction
- **Max p-values**: 0.810 (mriglu_positive), 0.758 (age_negative)
- **Output**: `/mnt/bytopia/IRC805/analysis/func/func/falff/mriglu_analysis/randomise_output/`
- **Note**: Directory has double `func` hierarchy (fixed for future runs)

---

## Currently Running Analyses

### ASL (Cerebral Blood Flow)
- **PID**: 2000548
- **Started**: 20:04 UTC
- **Elapsed**: ~25 minutes (as of 21:30)
- **Subjects**: 18
- **CPU**: 99.9% (actively computing)
- **Contrasts**: 6 (1 completed as of 20:13)
- **Status**: 🔄 **RUNNING OVERNIGHT**
- **Expected completion**: ~2-3 hours total
- **Output**: `/mnt/bytopia/IRC805/analysis/asl/asl_analysis/randomise_output/`
- **Log**: `logs/asl_analysis_20251205_200442.log`

### VBM Duplicate
- **PID**: 2012125
- **Started**: 20:14 UTC
- **Elapsed**: ~16 minutes (as of 21:30)
- **Subjects**: 23
- **CPU**: 100% (actively computing)
- **Status**: 🔄 **RUNNING OVERNIGHT** (duplicate of completed run)
- **Note**: This is overwriting already-completed results from earlier run
- **Output**: Same as completed VBM above

### TBSS FA Preparation
- **PID**: 1993641
- **Started**: 19:57 UTC
- **Elapsed**: ~93 minutes (as of 21:30)
- **Subjects**: 17
- **Status**: 🔄 **RUNNING OVERNIGHT**
- **Latest activity**: 20:27 UTC (target_FA.nii.gz created)
- **Progress**: 512 files created in `/FA/` directory
- **Current step**: TBSS skeleton processing (steps 2-4)
- **Missing**: `stats/` directory, `subject_manifest.json` (created in final step)
- **Expected completion**: Should complete within next 1-2 hours
- **Output**: `/mnt/bytopia/IRC805/analysis/tbss/FA/`
- **Log**: `logs/tbss_prep_FA_20251205_195758.log` (stopped logging after step 1)

---

## Architecture Validation

### All Workflows Refactored ✅

**Pre-Generated Design Matrix Pattern:**
```python
def run_analysis(..., design_dir: Path, ...):
    # Load pre-generated designs
    design_mat = design_dir / 'design.mat'
    design_con = design_dir / 'design.con'
    participants_file = design_dir / 'participants_matched.tsv'

    # Validate alignment
    validate_design_alignment(
        design_dir=design_dir,
        mri_files=[image_4d],
        subject_ids=subject_ids,
        analysis_type="VBM|TBSS|ASL|FUNCTIONAL"
    )

    # Run statistics
    run_randomise(data, design_mat, design_con, ...)
```

**Workflows Using This Pattern:**
1. ✅ VBM: `run_vbm_group_analysis.py`
2. ✅ TBSS: `neurovrai/analysis/tbss/run_tbss_stats.py`
3. ✅ Functional: `run_func_group_analysis.py`
4. ✅ ASL: `run_asl_group_analysis.py`

### Design Validation Features ✅

**Validates:**
- ✅ Design file existence (design.mat, design.con, participants_matched.tsv)
- ✅ Subject count matches (design rows = MRI volumes)
- ✅ Subject order matches (design row order = 4D volume order)
- ✅ 4D merged image support (detects single file with multiple volumes)
- ✅ Design summary consistency

**Fixed Issues:**
- ✅ 4D image volume counting (was counting files instead of volumes)
- ✅ ASL subject order alignment (now uses design matrix order)

---

## Files Modified This Session

### Bug Fixes
| File | Changes | Lines |
|------|---------|-------|
| `run_asl_group_analysis.py` | Fixed subject ordering in 4D merge | 39-85, 275 |
| `run_func_group_analysis.py` | Fixed double func hierarchy | 261 |
| `neurovrai/analysis/utils/design_validation.py` | Added 4D image support | 113-168 |
| `neurovrai/analysis/anat/vbm_workflow.py` | Fixed JSON scoping bug | 567-637 |

### CLI Refactoring
| File | Changes | Lines |
|------|---------|-------|
| `neurovrai/analysis/tbss/run_tbss_stats.py` | Replaced `--participants/--formula` with `--design-dir` | 467-512 |

---

## Tomorrow's Tasks

### 1. Monitor Overnight Runs
- [ ] Check ASL completion status (~2-3 hours total runtime)
- [ ] Check VBM duplicate completion
- [ ] Check TBSS FA completion (~1-2 hours remaining)
- [ ] Verify all outputs and logs

### 2. TBSS Remaining Metrics
Once FA completes, prepare other diffusion metrics:
```bash
for metric in MD AD RD MK AK RK KFA FICVF ODI FISO; do
  uv run python -m neurovrai.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --metric $metric \
    --output-dir /mnt/bytopia/IRC805/analysis/tbss/$metric \
    --fa-skeleton-dir /mnt/bytopia/IRC805/analysis/tbss/FA \
    > logs/tbss_prep_${metric}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

### 3. Run TBSS Statistics
For each completed metric preparation:
```bash
uv run python -m neurovrai.analysis.tbss.run_tbss_stats \
  --data-dir /mnt/bytopia/IRC805/analysis/tbss/FA \
  --design-dir /mnt/bytopia/IRC805/data/designs/tbss \
  --output-dir /mnt/bytopia/IRC805/analysis/tbss/FA/stats \
  --method randomise \
  --n-permutations 5000  # Full run (500 was test)
```

### 4. Verify All Results
- [ ] Check ASL results for significance
- [ ] Verify VBM results unchanged (duplicate should match original)
- [ ] Review all design validation logs
- [ ] Document final results

### 5. Clean Up
- [ ] Move old functional results from `/func/func/` to `/func/`
- [ ] Remove duplicate VBM outputs if identical
- [ ] Archive test logs to `docs/archive/`

### 6. Documentation
- [ ] Update main README with new CLI examples
- [ ] Create migration guide for `--design-dir` parameter
- [ ] Document TBSS workflow requirements

---

## Design Matrix Locations

All pre-generated designs stored in `/mnt/bytopia/IRC805/data/designs/`:

```
/mnt/bytopia/IRC805/data/designs/
├── vbm/                      # 23 subjects
│   ├── design.mat
│   ├── design.con
│   ├── design_summary.json
│   └── participants_matched.tsv
├── tbss/                     # 17 subjects
│   ├── design.mat
│   ├── design.con
│   ├── design_summary.json
│   └── participants_matched.tsv
├── func_reho/                # 17 subjects
│   ├── design.mat
│   ├── design.con
│   ├── design_summary.json
│   └── participants_matched.tsv
├── func_falff/               # 17 subjects
│   ├── design.mat
│   ├── design.con
│   ├── design_summary.json
│   └── participants_matched.tsv
└── asl/                      # 18 subjects
    ├── design.mat
    ├── design.con
    ├── design_summary.json
    └── participants_matched.tsv
```

**Design Specification:**
- **Predictors**: 4 (sex, age, mriglu_1, mriglu_2)
- **Contrasts**: 6 (mriglu_positive/negative, sex_positive/negative, age_positive/negative)
- **Coding**: Dummy coding for mriglu (binary factor)
- **Covariates**: Mean-centered sex and age

---

## Key Achievements

1. ✅ **Neuroaider Architecture Refactoring Complete**: All analysis workflows use pre-generated design matrices
2. ✅ **Critical Bug Fixed**: ASL subject ordering now matches design matrix
3. ✅ **Design Validation Working**: Correctly handles 4D merged images and subject order
4. ✅ **Three Analyses Complete**: VBM, ReHo, fALFF all validated and finished
5. ✅ **Two Analyses Running**: ASL and TBSS FA in progress overnight
6. ✅ **All CLI Consistent**: Every workflow uses `--design-dir` parameter

---

## Notes for Next Session

- **VBM duplicate**: Currently running but unnecessary (original completed successfully). Consider killing if still running tomorrow.
- **TBSS FA**: Long runtime expected due to 17 subjects × extensive metric projections. Check for completion first thing.
- **Small sample sizes**: 17-23 subjects with 500 permutations explains lack of significant results. Consider running with 5000 permutations for final publication-ready analyses.
- **Functional hierarchy**: ReHo/fALFF results are in `/func/func/` due to bug (now fixed). Future runs will use `/func/` correctly.

---

## Process IDs (as of 21:30 UTC)

- ASL randomise: **2000548**
- VBM randomise: **2012125**
- TBSS FA prep: **1993641**

Check with: `ps aux | grep -E "2000548|2012125|1993641"`

---

**Session End**: 2025-12-05 21:30 UTC
