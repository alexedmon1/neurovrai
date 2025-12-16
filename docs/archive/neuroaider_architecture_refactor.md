# Neuroaider Architecture Refactor - Status

## Overview

Refactoring the analysis pipeline to use neuroaider as a **pre-analysis setup tool** instead of generating designs on-the-fly during each analysis.

**New Workflow**:
1. **Step 1**: Generate design matrices ONCE using `generate_design_matrices.py`
2. **Step 2**: Review and validate the designs (inspect design_summary.json)
3. **Step 3**: Run analyses using pre-generated designs

**Benefits**:
- Consistency: Same designs used every time
- Transparency: Review designs before running expensive analyses
- Speed: No redundant design generation
- FSL Standard: Separate design and analysis steps

---

## Completed Work

### ✅ 1. Created Design Generation Script

**File**: `/home/edm9fd/sandbox/neurovrai/generate_design_matrices.py`

**Usage**:
```bash
# Generate designs for all analyses
python generate_design_matrices.py --all --study-root /mnt/bytopia/IRC805

# Generate design for specific analysis
python generate_design_matrices.py \
    --participants /path/to/participants.tsv \
    --output-dir /path/to/design/output \
    --formula 'mriglu+sex+age'
```

**Features**:
- Uses neuroaider DesignHelper
- Auto-detects binary groups for dummy coding (no intercept)
- Auto-generates 6 contrasts for binary mriglu comparison
- Outputs: design.mat, design.con, design_summary.json

### ✅ 2. Generated All Design Matrices

Successfully generated designs for:
- **VBM**: 23 subjects, 6 contrasts → `/mnt/bytopia/IRC805/data/designs/vbm/`
- **ASL**: 18 subjects, 6 contrasts → `/mnt/bytopia/IRC805/data/designs/asl/`
- **ReHo**: 17 subjects, 6 contrasts → `/mnt/bytopia/IRC805/data/designs/func_reho/`
- **fALFF**: 17 subjects, 6 contrasts → `/mnt/bytopia/IRC805/data/designs/func_falff/`
- **TBSS**: 17 subjects, 6 contrasts → `/mnt/bytopia/IRC805/data/designs/tbss/`

**All designs verified correct**:
- Design matrix: `[sex, age, mriglu_1, mriglu_2]` (dummy coding, no intercept)
- Contrasts:
  1. `mriglu_positive [0, 0, 1, -1]`: Controlled > Uncontrolled
  2. `mriglu_negative [0, 0, -1, 1]`: Uncontrolled > Controlled
  3. `sex_positive [1, 0, 0, 0]`: Positive sex effect
  4. `sex_negative [-1, 0, 0, 0]`: Negative sex effect
  5. `age_positive [0, 1, 0, 0]`: Positive age effect
  6. `age_negative [0, -1, 0, 0]`: Negative age effect

### ✅ 3. Updated VBM Runner (Partially Complete)

**File**: `/home/edm9fd/sandbox/neurovrai/run_vbm_group_analysis.py`

**Changes**:
- Added `--design-dir` parameter (optional, defaults to `{study-root}/data/designs/vbm/`)
- Validates design files exist before running
- Updated function call to pass `design_dir` instead of `formula` and `contrasts`
- Modified function signature: `run_vbm_analysis(vbm_dir, design_dir, ...)`

**Status**: ⚠️ Runner updated, but workflow function needs update to load pre-generated designs

---

## Remaining Work

### ⚠️ 4. Complete VBM Workflow Function Refactor

**File**: `/home/edm9fd/sandbox/neurovrai/neurovrai/analysis/anat/vbm_workflow.py`

**Needed**:
- Replace design generation code (lines ~597-703) with design loading code
- Load design.mat, design.con from design_dir
- Copy design files to stats directory
- Extract contrast names from design_summary.json

**Approach**: Replace the large neuroaider generation block with:
```python
# Load pre-generated designs
design_mat_file = design_dir / 'design.mat'
design_con_file = design_dir / 'design.con'
design_summary = json.load(open(design_dir / 'design_summary.json'))

# Copy to stats directory
shutil.copy(design_mat_file, stats_dir / 'design.mat')
shutil.copy(design_con_file, stats_dir / 'design.con')

# Extract contrast names
contrast_names = design_summary['contrasts']
```

### 5. Refactor TBSS to Use Pre-Generated Designs

**File**: `/home/edm9fd/sandbox/neurovrai/neurovrai/analysis/tbss/run_tbss_stats.py`

**Needed**:
- Already has `--contrasts-file` parameter (now optional)
- Update to use design_dir with pre-generated designs
- Similar pattern to VBM

### 6. Refactor Functional to Use Pre-Generated Designs

**Files**:
- `/home/edm9fd/sandbox/neurovrai/run_func_group_analysis.py` (already updated with CLI args)

**Needed**:
- Update to load pre-generated designs from design_dir
- Remove neuroaider generation code from `create_design_matrices()` function

### 7. Refactor ASL to Use Pre-Generated Designs

**File**: `/home/edm9fd/sandbox/neurovrai/run_asl_group_analysis.py`

**Needed**:
- Add `--design-dir` parameter
- Update to load pre-generated designs
- Remove neuroaider generation code

### 8. Update Parallel Script

**File**: `/home/edm9fd/sandbox/neurovrai/run_all_analyses_parallel.sh`

**Needed**:
- Add design generation step FIRST
- Update all analysis commands to use `--design-dir` parameter
- Remove `--formula` parameter (no longer needed)

**New Structure**:
```bash
# Step 1: Generate all designs (if not already done)
if [ ! -f /mnt/bytopia/IRC805/data/designs/vbm/design.mat ]; then
    echo "Generating design matrices..."
    uv run python generate_design_matrices.py --all
fi

# Step 2: Run analyses with pre-generated designs
uv run python run_vbm_group_analysis.py \
    --study-root /mnt/bytopia/IRC805 \
    --design-dir /mnt/bytopia/IRC805/data/designs/vbm
```

---

## Testing Plan

Once refactor is complete:

1. Verify designs are loaded correctly (check logs)
2. Verify same statistical results as before (compare output maps)
3. Run full parallel analysis pipeline
4. Review cluster reports to ensure contrasts are labeled correctly

---

## Next Steps

**Option A - Complete VBM Refactor First**:
1. Finish VBM workflow function to load pre-generated designs
2. Test VBM analysis end-to-end
3. Apply same pattern to TBSS, Functional, ASL

**Option B - Use Hybrid Approach (Faster)**:
1. Keep existing workflow functions as-is (they work)
2. Have them check if design files already exist in stats_dir
3. If yes → use them directly
4. If no → generate on-the-fly (backward compatible)
5. Update runners to copy pre-generated designs to stats_dir BEFORE calling workflow

**Recommendation**: Option B is faster and maintains backward compatibility while supporting the new architecture.

---

## Files Modified So Far

1. ✅ Created: `generate_design_matrices.py`
2. ✅ Generated: 5 design directories with design.mat/design.con/design_summary.json
3. ✅ Modified: `run_vbm_group_analysis.py` (added --design-dir, updated function call)
4. ⚠️ Modified: `neurovrai/analysis/anat/vbm_workflow.py` (function signature only)
5. ✅ Modified: `run_func_group_analysis.py` (added CLI argument parsing)
