# Session Summary: Dummy Coding Implementation for Group Comparisons
**Date:** 2025-12-05
**Status:** In Progress - Code Complete, Re-analysis Pending

---

## Critical Issue Identified

**FUNDAMENTAL STATISTICAL ERROR IN ALL PREVIOUS ANALYSES**

Previous analyses incorrectly treated the binary categorical variable `mriglu` (1=Controlled, 2=Uncontrolled) as a **continuous predictor**. This is statistically inappropriate for group mean comparisons.

**Correct Approach:**
- Use **dummy coding WITHOUT intercept** for binary group comparisons
- Design matrix: `[group1_indicator, group2_indicator, covariates...]`
- Contrasts: `[1, -1, 0, 0]` for Group1 > Group2, `[-1, 1, 0, 0]` for Group2 > Group1

---

## What Was Accomplished

### 1. Code Implementation ✅

**Modified Files (Committed: 6d678c4):**

1. **`neurovrai/analysis/stats/design_matrix.py` (lines 212-216)**
   - Changed dummy variable creation logic
   - When `add_intercept=False`: keeps ALL categorical levels (no drop_first)
   - When `add_intercept=True`: drops first level (avoids collinearity)
   - Enables proper dummy coding for direct group comparisons

2. **`neurovrai/analysis/anat/vbm_workflow.py`**
   - **Binary group detection** (lines 610-627):
     - Parses formula to identify first variable
     - Checks if it has exactly 2 unique values
     - Sets `use_dummy_coding=True` and `add_intercept=False`
     - Marks variable as categorical with `C()` notation

   - **Intelligent contrast generation** (lines 651-681):
     - Detects group columns in design matrix
     - Creates `[1, -1, 0, 0]` for positive contrast (Group1 > Group2)
     - Creates `[-1, 1, 0, 0]` for negative contrast (Group2 > Group1)
     - Validates exactly 2 group columns exist

   - **Command-line parsing cleanup** (lines 994-1003):
     - Simplified contrast initialization
     - Contrasts now built dynamically after design matrix creation

### 2. Data Discovery ✅

**Found Missing Subject:** IRC805-0580101
- **Location:** `/mnt/bytopia/IRC805/data/gludata.csv` (listed as "0580101")
- **Demographics:**
  - mriglu: 2 (Uncontrolled group)
  - sex: 2
  - age: ~29 years
- **Status:** EXISTS in derivatives directory but MISSING from all participant TSV files
- **Impact:** Affects N for ALL modalities

---

## What Remains To Be Done

### Phase 1: Fix Participant Files

1. **Add IRC805-0580101 to all participant TSV files:**
   - `/mnt/bytopia/IRC805/data/designs/vbm/participants_matched.tsv` (N: 22→23)
   - `/mnt/bytopia/IRC805/data/designs/dki/participants_matched.tsv` (N: 17→18)
   - `/mnt/bytopia/IRC805/data/designs/func_reho/participants_matched.tsv` (N: 16→17)
   - `/mnt/bytopia/IRC805/data/designs/func_falff/participants_matched.tsv` (N: 16→17)
   - Check ASL participant file (if exists)

2. **Demographics to add:**
   ```
   IRC805-0580101    2    2    29.08493151
   ```

### Phase 2: Apply Dummy Coding to Other Workflows

**Need to modify these workflows with same logic as VBM:**

1. **TBSS Stats** (`neurovrai/analysis/tbss/run_tbss_stats.py`):
   - Add binary group detection
   - Set `add_intercept=False` for group comparisons
   - Generate proper contrasts `[1,-1,0,0]` and `[-1,1,0,0]`

2. **Functional ReHo** (group analysis script):
   - Apply dummy coding logic
   - Update contrast generation

3. **Functional fALFF** (group analysis script):
   - Apply dummy coding logic
   - Update contrast generation

4. **ASL** (if group analysis exists):
   - Verify if ASL has group-level analysis
   - Apply dummy coding if applicable

### Phase 3: Re-run All Analyses

**All previous analyses INVALID - must re-run with corrected design:**

1. **VBM (N=23)**
   ```bash
   python -m neurovrai.analysis.anat.vbm_workflow analyze \
     --vbm-dir /mnt/bytopia/IRC805/analysis/anat/vbm \
     --participants /mnt/bytopia/IRC805/data/designs/vbm/participants_matched.tsv \
     --design "mriglu+sex+age" \
     --contrasts "mriglu_positive,mriglu_negative" \
     --method randomise \
     --n-permutations 5000
   ```

2. **TBSS - All 11 Metrics (N=18)**
   - FA, MD, AD, RD (DTI)
   - MK, AK, RK, KFA (DKI)
   - FICVF, ODI, FISO (NODDI)
   - 5000 permutations each

3. **Functional - ReHo (N=17)**
   - Regional Homogeneity analysis
   - 5000 permutations

4. **Functional - fALFF (N=17)**
   - Fractional ALFF analysis
   - 5000 permutations

5. **ASL (N=TBD)**
   - Verify subject count
   - Re-run if group analysis exists

### Phase 4: Generate Reports

**After all analyses complete:**
- Generate cluster reports for all modalities
- Compare results with old (incorrect) analyses
- Document any changes in significant findings

---

## Technical Details

### Dummy Coding Mathematics

**Incorrect Model (Previous):**
```
Y = β₀ + β₁(mriglu) + β₂(sex) + β₃(age)
```
- Treats mriglu=1,2 as continuous values
- β₁ represents slope, not group difference
- INVALID for categorical comparison

**Correct Model (New):**
```
Y = β₁(controlled) + β₂(uncontrolled) + β₃(sex) + β₄(age)
```
- No intercept term
- β₁ = mean for controlled group
- β₂ = mean for uncontrolled group
- Contrast [1,-1,0,0] tests: controlled > uncontrolled

### Design Matrix Example

**For N=22 subjects (13 controlled, 9 uncontrolled):**

```
Design matrix (22 x 4):
  controlled  uncontrolled  sex  age
  1           0             1    45.2  # Subject 1: controlled
  1           0             2    38.1  # Subject 2: controlled
  0           1             2    29.5  # Subject 3: uncontrolled
  ...
```

**Contrast Vectors:**
- `mriglu_positive`: `[1, -1, 0, 0]` → controlled > uncontrolled
- `mriglu_negative`: `[-1, 1, 0, 0]` → uncontrolled > controlled

---

## Current Participant Counts

| Modality | Current N | With IRC805-0580101 | Target N |
|----------|-----------|---------------------|----------|
| VBM      | 22        | 23                  | 23       |
| TBSS/DKI | 17        | 18                  | 18       |
| ReHo     | 16        | 17                  | 17       |
| fALFF    | 16        | 17                  | 17       |
| ASL      | ?         | ?                   | ?        |

---

## Files Modified (Git Commit 6d678c4)

1. `neurovrai/analysis/stats/design_matrix.py`
2. `neurovrai/analysis/anat/vbm_workflow.py`
3. `neurovrai/analysis/stats/enhanced_cluster_report.py`

**Commit Message:**
```
Feature: Add dummy coding support for categorical group comparisons

Implements proper statistical modeling for binary group comparisons.
Fixes fundamental error where categorical groups were treated as
continuous predictors.
```

---

## Next Session Checklist

- [ ] Extract demographics for IRC805-0580101 from gludata.csv
- [ ] Add subject to all participant TSV files
- [ ] Verify final N values match targets
- [ ] Modify TBSS stats workflow with dummy coding
- [ ] Modify Functional (ReHo & fALFF) workflows with dummy coding
- [ ] Check ASL workflow and apply dummy coding if needed
- [ ] Re-run VBM analysis (N=23, 5000 permutations)
- [ ] Re-run TBSS for all 11 metrics (N=18, 5000 permutations each)
- [ ] Re-run Functional analyses (N=17, 5000 permutations each)
- [ ] Re-run ASL analysis (if applicable)
- [ ] Generate cluster reports for all new results
- [ ] Document any significant findings

---

## Important Notes

1. **All previous analyses are INVALID** due to incorrect statistical model
2. **IRC805-0580101 must be added** before re-running any analysis
3. **Dummy coding is only implemented in VBM** - other workflows still need updates
4. **Running time:** Each randomise analysis with 5000 permutations takes ~45 minutes
5. **Total re-analysis time estimate:** ~8-10 hours for all modalities

---

## References

- Session Date: 2025-12-05
- Git Commit: 6d678c4
- Previous Session: SESSION_HISTORY_2025-11.md
