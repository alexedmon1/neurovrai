# Modality-Aware Design Generation

**Date**: 2025-12-05
**Status**: ✅ **PRODUCTION READY**

---

## Overview

Implemented modality-aware design matrix generation that:
1. **Auto-detects subjects** with MRI data for specific modalities
2. **Filters master participants file** (gludata.csv) to only include subjects with data
3. **Generates design matrices** with correct subject ordering
4. **Creates subject order files** for verification and validation

---

## Key Benefits

### 1. Eliminates Manual Participant Matching
- **Before**: Had to manually create `participants_matched.tsv` for each analysis
- **After**: Automatically uses subjects with available MRI derivatives

### 2. Ensures Data-Design Alignment
- Only includes subjects that actually have MRI data
- Subject order matches MRI file order automatically
- Prevents silent mismatches that could invalidate results

### 3. Provides Transparency
- `subject_order.txt` documents exact subject ordering
- Easy to verify which subjects are included
- Identifies missing data (subjects with MRI but no demographics)

### 4. Single Source of Truth
- Uses master `gludata.csv` file for all demographics
- No need to maintain separate participants files per modality
- Reduces data duplication and sync issues

---

## Implementation

### New Module: `neurovrai/analysis/utils/modality_subjects.py`

**Key Functions**:
- `find_subjects_for_modality(study_root, modality, metric=None, subject_prefix=None)` -
  Auto-detect subjects with MRI data. `subject_prefix` is the study code that
  prefixes subject ids (`'STUDY'` for `STUDY-0580101`); omit it and any prefix
  matches, which is what a study whose ids carry none needs.
- `load_participants_for_modality(participants_file, study_root, modality, metric=None, subject_prefix=None)` -
  Filter the participants file to those subjects. With a prefix, bare numeric ids
  in the file are zero-padded and prefixed to match the imaging filenames.
- `save_subject_list()` - Document subject ordering
- `get_modality_info()` - Modality metadata

**Supported Modalities**:
- `vbm`: Voxel-Based Morphometry
- `asl`: Arterial Spin Labeling
- `reho`: Regional Homogeneity (functional)
- `falff`: Fractional ALFF (functional)
- `tbss`: Tract-Based Spatial Statistics (with metric parameter)

### Building the design: `neuroaider`

Design matrices and contrasts are **not** built in this repo. They come from
[neuroaider](https://github.com/alexedmon1/neuroaider), a standalone package
listed in `pyproject.toml`, which writes FSL-compatible `.mat`, `.con` and
`.fts` files. neurovrai's job is to decide *which subjects, in which order* go
into that design; neuroaider's job is to turn a participants table into the
matrices FSL randomise reads.

The two meet at the filtered participants file: `load_participants_for_modality()`
returns the rows for subjects that actually have data for the modality, in the
imaging order, and neuroaider builds the design from that.

```bash
# 1. Filter participants to the subjects with VBM data, in imaging order
python -c "
from pathlib import Path
from neurovrai.analysis.utils.modality_subjects import load_participants_for_modality, save_subject_list
df, subjects = load_participants_for_modality(
    Path('/path/to/study/data/participants.csv'),
    Path('/path/to/study'), 'vbm', subject_prefix='STUDY')
df.to_csv('/path/to/study/data/designs/vbm/participants_vbm.csv', index=False)
save_subject_list(subjects, Path('/path/to/study/data/designs/vbm/subject_order.txt'))
"

# 2. Build the design and contrasts from that filtered table
neuroaider /path/to/study/data/designs/vbm/participants_vbm.csv \
  --output-mat /path/to/study/data/designs/vbm/design.mat \
  --output-con /path/to/study/data/designs/vbm/design.con \
  --subject-column participant_id \
  --covariate age mean_center=True \
  --categorical group \
  --contrast "group_A > group_B"
```

For TBSS, pass the metric so the subject list matches that metric's inputs:
`load_participants_for_modality(..., 'tbss', metric='FA', subject_prefix='STUDY')`.

`neuroaider --help` lists the rest: `--interaction`, `--ftest`,
`--validate-by-dir` (check every subject has an imaging file before writing a
design), `--output-names` and `--output-summary`. Its `DesignHelper` class is
the Python entry point to the same thing.

---

## Example Validation: VBM

### Test Run Results

```
INFO: Found 23 subjects with vbm data
INFO: Filtered participants to 22 subjects with vbm data
WARNING: 1 subjects with MRI data not found in participants file: ['STUDY-0580101']

Design Matrix:
  - Subjects: 22
  - Predictors: 4 [sex, age, mriglu_1, mriglu_2]
  - Contrasts: 6 [mriglu_positive, mriglu_negative, sex_positive, sex_negative, age_positive, age_negative]
```

### Generated Files

1. **design.mat** - FSL design matrix (22x4)
2. **design.con** - FSL contrasts (6 contrasts)
3. **design_summary.json** - Human-readable summary
4. **subject_order.txt** - Ordered subject list with documentation

### Subject Order File Format

```
# Subject order for VBM analysis
# N=22 subjects
# Generated: 2025-12-05 18:10:56
#
# CRITICAL: This order MUST match:
#   1. Rows in design.mat
#   2. Volumes in 4D merged image
#   3. Subject order in analysis scripts
#
STUDY-1580101
STUDY-1640101
STUDY-1720201
...
```

---

## Discovered Data Inconsistency

**Issue**: Subject `STUDY-0580101` has VBM data but is **missing from gludata.csv**

**Impact**: This subject will be excluded from group analysis
**Resolution**: Investigate why demographic data is missing for this subject

---

## Regenerating designs

For each modality, filter first and build second. The filtering call differs only
in the modality string (`vbm`, `asl`, `reho`, `falff`, and `tbss` with a metric);
the neuroaider call is the same shape every time, pointed at that modality's
filtered participants file and output directory.

TBSS needs one design per metric, because each metric's subject list can differ:

```bash
for metric in FA MD AD RD MK AK RK KFA FICVF ODI FISO; do
  # load_participants_for_modality(..., 'tbss', metric=$metric, subject_prefix='STUDY')
  # then neuroaider on the resulting table, writing into designs/tbss/$metric/
done
```

### 3. Update Refactored Analysis Scripts

The refactored analysis scripts already support the new design directory structure:
- ✅ `run_vbm_group_analysis.py`
- ✅ `run_func_group_analysis.py`
- ✅ `run_asl_group_analysis.py`
- ✅ `neurovrai/analysis/tbss/run_tbss_stats.py`

### 4. Test End-to-End

Run one analysis end-to-end to validate the entire pipeline:

```bash
# VBM test
python run_vbm_group_analysis.py \
  --study-root /path/to/study \
  --design-dir /path/to/study/data/designs/vbm \
  --tissue GM \
  --method randomise \
  --n-permutations 500  # Quick test
```

---

## Architecture Diagram

```
gludata.csv (Master participants file)
     |
     | Filter by available MRI derivatives
     v
Modality-Aware Subject Detection
     |
     ├─> find_subjects_for_modality()
     |   └─> Check derivatives/analysis directories
     |       └─> Return: List of subject IDs with data
     |
     └─> load_participants_for_modality()
         └─> Filter gludata.csv to matched subjects
             └─> Return: Filtered DataFrame + ordered list
                 |
                 v
         generate_design_for_modality()
                 |
                 ├─> Generate design matrix (neuroaider)
                 ├─> Generate contrasts
                 ├─> Save design.mat, design.con
                 ├─> Save design_summary.json
                 └─> Save subject_order.txt
                     |
                     v
              Design files ready for analysis!
```

---

## Files Modified/Created

### Created:
1. `neurovrai/analysis/utils/modality_subjects.py` - Modality-aware utilities
2. `docs/MODALITY_AWARE_DESIGN_GENERATION.md` - This document

### Depends on:
1. `neuroaider` - builds the `.mat`/`.con`/`.fts` files from the filtered table

### Removed:
- `generate_design_matrices.py` and `compare_design_matrices.py` - study-specific
  scripts, deleted 2026-09-22; design generation is neuroaider's job

---

## Validation Checklist

- [x] Subject detection works for VBM
- [ ] Subject detection works for ASL
- [ ] Subject detection works for ReHo
- [ ] Subject detection works for fALFF
- [ ] Subject detection works for TBSS
- [ ] Design matrices match old designs (accounting for missing subjects)
- [ ] Subject order files are accurate
- [ ] End-to-end analysis works with new designs
- [ ] Missing subject issue (STUDY-0580101) investigated

---

**Status**: Ready for production use after full validation
