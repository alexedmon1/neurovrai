# Session Summary: Neuroaider Architecture Refactor
**Date**: 2025-12-05
**Session Focus**: Refactor analysis pipeline to use neuroaider as pre-analysis setup tool

---

## Overview

Successfully refactored the statistical analysis architecture to use **neuroaider for design matrix generation BEFORE running analyses** instead of generating designs on-the-fly during each analysis.

**Key Principle**: Neuroaider is now used as a setup tool to create validated design matrices that are then used by all downstream analyses.

---

## ✅ COMPLETED WORK

### 1. Design Generation Infrastructure

**Created**: `generate_design_matrices.py`

Standalone script using neuroaider to generate FSL-compatible design matrices:
- Auto-detects binary categorical groups
- Uses dummy coding WITHOUT intercept for direct group comparison
- Auto-generates 6 contrasts for binary mriglu variable:
  - 2 primary group contrasts (mriglu_positive/negative)
  - 4 covariate contrasts (sex_positive/negative, age_positive/negative)
- Outputs: `design.mat`, `design.con`, `design_summary.json`

**Usage**:
```bash
# Generate all designs
python generate_design_matrices.py --all --study-root /mnt/bytopia/IRC805

# Generate specific design
python generate_design_matrices.py \
    --participants /path/to/participants.tsv \
    --output-dir /path/to/output \
    --formula 'mriglu+sex+age'
```

### 2. Design Validation Infrastructure

**Created**: `neurovrai/analysis/utils/design_validation.py`

Comprehensive validation system ensuring design-to-data alignment:
- Validates design matrix dimensions match MRI data dimensions
- Checks subject counts match across design, participants file, and MRI files
- **CRITICAL**: Verifies subject ORDER matches (essential for FSL randomise)
- Validates 4D image volume counts
- Raises errors if any alignment check fails
- Creates detailed subject order reports

**Key Functions**:
- `validate_design_alignment()`: Main validation function
- `parse_fsl_design_mat()`: Extract design dimensions
- `parse_fsl_design_con()`: Extract contrast count
- `create_subject_order_report()`: Generate verification reports

### 3. All Design Matrices Generated

Successfully generated and validated designs for all analyses:

| Analysis | Subjects | Predictors | Contrasts | Location |
|----------|----------|------------|-----------|----------|
| VBM      | 23       | 4          | 6         | `/mnt/bytopia/IRC805/data/designs/vbm/` |
| ASL      | 18       | 4          | 6         | `/mnt/bytopia/IRC805/data/designs/asl/` |
| ReHo     | 17       | 4          | 6         | `/mnt/bytopia/IRC805/data/designs/func_reho/` |
| fALFF    | 17       | 4          | 6         | `/mnt/bytopia/IRC805/data/designs/func_falff/` |
| TBSS     | 17       | 4          | 6         | `/mnt/bytopia/IRC805/data/designs/tbss/` |

**Design Matrix Structure** (all):
- Columns: `[sex, age, mriglu_1, mriglu_2]`
- Coding: Dummy WITHOUT intercept
- Contrasts:
  1. `mriglu_positive [0, 0, 1, -1]`: Controlled > Uncontrolled
  2. `mriglu_negative [0, 0, -1, 1]`: Uncontrolled > Controlled
  3. `sex_positive [1, 0, 0, 0]`: Positive sex effect
  4. `sex_negative [-1, 0, 0, 0]`: Negative sex effect
  5. `age_positive [0, 1, 0, 0]`: Positive age effect
  6. `age_negative [0, -1, 0, 0]`: Negative age effect

### 4. VBM Fully Refactored

**Modified Files**:
- `run_vbm_group_analysis.py`:
  - Added `--design-dir` parameter (defaults to `{study-root}/data/designs/vbm/`)
  - Validates design files exist before execution
  - Updated function call to pass `design_dir`

- `neurovrai/analysis/anat/vbm_workflow.py`:
  - Changed function signature: `run_vbm_analysis(vbm_dir, design_dir, ...)`
  - **Removed 101 lines** of neuroaider design generation code
  - Replaced with design loading from pre-generated files
  - Integrated `validate_design_alignment()` function
  - Copies design files to stats directory for record-keeping
  - Removed unused `DesignHelper` import

**New VBM Workflow**:
```bash
# Step 1: Generate design (ONCE, review before running)
python generate_design_matrices.py \
    --participants /mnt/bytopia/IRC805/data/designs/vbm/participants_matched.tsv \
    --output-dir /mnt/bytopia/IRC805/data/designs/vbm \
    --formula 'mriglu+sex+age'

# Step 2: Review design (check design_summary.json)
cat /mnt/bytopia/IRC805/data/designs/vbm/design_summary.json

# Step 3: Run VBM analysis (uses validated pre-generated design)
python run_vbm_group_analysis.py \
    --study-root /mnt/bytopia/IRC805 \
    --tissue GM \
    --method randomise \
    --n-permutations 5000
```

### 5. TBSS Fully Refactored

**Modified Files**:
- `neurovrai/analysis/tbss/run_tbss_stats.py`:
  - Changed function signature: `run_tbss_statistical_analysis(data_dir, design_dir, output_dir, ...)`
  - **Removed 114 lines** of neuroaider design generation code
  - Replaced with design loading + validation
  - Updated CLI: replaced `--formula`, `--participants`, `--contrasts-file` with `--design-dir`
  - Removed unused `DesignHelper` import
  - Integrated design validation for all TBSS metrics

**New TBSS Workflow**:
```bash
# Step 1: Generate design (ONCE for all metrics)
python generate_design_matrices.py \
    --participants /mnt/bytopia/IRC805/data/designs/tbss/participants_matched.tsv \
    --output-dir /mnt/bytopia/IRC805/data/designs/tbss \
    --formula 'mriglu+sex+age'

# Step 2: Run TBSS stats for each metric
python -m neurovrai.analysis.tbss.run_tbss_stats \
    --data-dir /mnt/bytopia/IRC805/analysis/tbss/FA \
    --design-dir /mnt/bytopia/IRC805/data/designs/tbss \
    --output-dir /mnt/bytopia/IRC805/analysis/tbss/FA/stats \
    --n-permutations 5000
```

---

## 📊 BENEFITS OF NEW ARCHITECTURE

### 1. Separation of Concerns
- **Design creation** (neuroaider) is separate from **statistical analysis** (FSL/nilearn)
- Clear two-step workflow: design → analyze
- Follows FSL's standard practice

### 2. Transparency & Review
- Designs can be reviewed before running expensive analyses
- `design_summary.json` provides human-readable design specification
- Easy to verify contrasts are testing the correct hypotheses

### 3. Validation & Safety
- Comprehensive alignment checking prevents silent errors
- Subject order validation catches mismatches before running randomise
- Design-to-data validation ensures statistical validity

### 4. Reproducibility
- Same design used every time (no regeneration variance)
- Design files versioned with analysis outputs
- Easy to share exact design specification

### 5. Efficiency
- No redundant design generation across multiple analyses
- TBSS uses same design for all 11 metrics (FA, MD, AD, RD, MK, AK, RK, KFA, FICVF, ODI, FISO)

### 6. Error Prevention
- Validates subject counts match
- Validates subject ORDER (critical for randomise)
- Catches dimension mismatches
- Prevents running analyses with incorrect designs

---

## 🔧 IMPLEMENTATION DETAILS

### Files Created
1. `generate_design_matrices.py` - Design generation script
2. `neurovrai/analysis/utils/design_validation.py` - Validation utilities

### Files Modified
1. `run_vbm_group_analysis.py` - Updated CLI and workflow call
2. `neurovrai/analysis/anat/vbm_workflow.py` - Refactored to load designs
3. `neurovrai/analysis/tbss/run_tbss_stats.py` - Refactored to load designs
4. `run_func_group_analysis.py` - Added CLI argument parsing (partial)

### Code Removed
- VBM: 101 lines of design generation code
- TBSS: 114 lines of design generation code
- Total: 215 lines of redundant code removed

### Code Added
- Design generation: ~200 lines
- Design validation: ~250 lines
- Design loading: ~70 lines (per workflow)

---

## ⚠️ REMAINING WORK

### High Priority
1. **Functional (ReHo/fALFF)**: Refactor to load pre-generated designs (~20 min)
2. **ASL**: Refactor to load pre-generated designs (~20 min)
3. **Parallel Script**: Update `run_all_analyses_parallel.sh` to use new architecture (~15 min)

### Testing
4. **VBM Test**: Run end-to-end with real data
5. **TBSS Test**: Run FA analysis with real data
6. **Validation Test**: Verify error catching works (intentional mismatches)

**Estimated Remaining Time**: 1-1.5 hours

---

## 📚 RELATED DOCUMENTS

- **Architecture Details**: [`neuroaider_architecture_refactor.md`](./neuroaider_architecture_refactor.md)
- **Progress Summary**: [`refactor_progress_summary.md`](./refactor_progress_summary.md)
- **Validation Utility**: [`neurovrai/analysis/utils/design_validation.py`](../neurovrai/analysis/utils/design_validation.py)
- **Design Generation**: [`generate_design_matrices.py`](../generate_design_matrices.py)

---

## 📝 NEXT SESSION TODO

1. Complete functional (ReHo/fALFF) refactor
2. Complete ASL refactor
3. Update parallel analysis script
4. Test VBM end-to-end
5. Test TBSS FA end-to-end
6. Document new workflow in README
7. Create usage examples

---

## 🎯 LONG-TERM IMPROVEMENTS

1. Add design matrix visualization (heatmap of design.mat)
2. Add contrast visualization (show what each contrast tests)
3. Add interactive design review tool
4. Support for continuous group variables
5. Support for interaction terms
6. Multi-level designs (random effects)

---

## 💡 KEY LESSONS

1. **Always validate alignment** - Subject order mismatches are silent but catastrophic
2. **Separate setup from analysis** - Makes debugging easier, improves reproducibility
3. **Design review is critical** - Catching design errors before running saves hours
4. **FSL conventions matter** - Design matrix format, contrast vectors must match exactly
5. **Automation with validation** - Auto-generate but always verify

---

**Session Completed**: 2025-12-05
**Next Session**: Complete remaining refactors and testing
