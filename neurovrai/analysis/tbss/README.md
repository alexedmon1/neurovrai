# TBSS Analysis Workflow

Tract-Based Spatial Statistics (TBSS) for group-level DTI analysis.

## Overview

This workflow is split into two phases:

### Phase 1: Data Preparation (`prepare_tbss.py`)
- Collects DTI metrics from preprocessed subjects
- Runs FSL TBSS pipeline (steps 1-4)
- Handles missing data gracefully
- Generates skeleton-projected 4D volumes
- **Run once per metric**

### Phase 2: Statistical Analysis (`run_tbss_stats.py`)
- Creates design matrices from participant data
- Executes FSL randomise with TFCE
- Extracts and reports significant clusters
- **Run multiple times with different models**

## Prerequisites

1. **Completed DWI preprocessing** for all subjects
2. **FSL installed** and `$FSLDIR` set
3. **Config file** with paths (see `config.yaml`)
4. **Participants CSV** with demographic/clinical data

## Phase 1: Prepare Data

### TBSS Workflow Overview

The FSL TBSS pipeline follows this structure:
1. **Run `tbss_1_preproc`** in work directory → creates `FA/` subdirectory with processed files
2. **CD into `FA/` directory**
3. **Run `tbss_2_reg -T`** from within `FA/` → registers to FMRIB58_FA template
4. **Run `tbss_3_postreg -S`** from within `FA/` → creates mean FA and skeleton
5. **Run `tbss_4_prestats 0.2`** from within `FA/` → projects FA onto skeleton

For **non-FA metrics** (MD, AD, RD):
1. Copy metric files to `<work_dir>/<metric>/` (e.g., `tbss_work/MD/`)
2. Run `tbss_non_FA <metric>` from work directory → projects onto existing FA skeleton

### Basic Usage

```bash
# Prepare FA analysis
uv run python -m neurovrai.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --metric FA \
    --output-dir /study/analysis/tbss/
```

### Options

- `--metric`: DTI metric to analyze (`FA`, `MD`, `AD`, `RD`)
- `--output-dir`: Where to save prepared analysis
- `--subjects`: Optional list of specific subjects (otherwise discovers all)

### What It Does

1. **Discovers subjects** in derivatives directory
2. **Validates data** - checks each subject has the specified metric file
3. **Logs missing data**:
   ```
   ✓ IRC805-0580101: Found FA at IRC805-0580101/dwi/dti/FA.nii.gz
   ✗ IRC805-0580102: FA file not found in IRC805-0580102/dwi/
   ```
4. **Copies valid files** to TBSS structure
5. **Runs FSL TBSS**:
   - Step 1: Preprocessing (erosion, zero-ending)
   - Step 2: Registration to FMRIB58_FA template
   - Step 3: Creates mean FA and skeleton (threshold=0.2)
   - Step 4: Projects metric onto skeleton
6. **Generates outputs**:
   - `subject_manifest.json` - Included/excluded subjects with reasons
   - `subject_list.txt` - Simple list of included subjects
   - `all_FA_skeletonised.nii.gz` - 4D volume ready for randomise
   - `mean_FA_skeleton.nii.gz` - Skeleton mask

### Output Structure

```
/study/analysis/tbss_FA/
├── origdata/                       # Original metric files (copied)
│   ├── IRC805-0580101_FA.nii.gz
│   ├── IRC805-0580103_FA.nii.gz
│   └── ...
├── FA/                             # TBSS outputs (created by FSL)
│   ├── all_FA_skeletonised.nii.gz # Ready for randomise
│   ├── mean_FA_skeleton.nii.gz    # Skeleton mask
│   └── ...
├── subject_manifest.json           # Included/excluded subjects
├── subject_list.txt                # Order of subjects in 4D volume
└── logs/
    └── prepare_tbss_20250121_143022.log
```

### Handling Missing Data

The workflow automatically:
- ✅ **Logs** which subjects are missing data
- ✅ **Documents** exclusion reasons in manifest
- ✅ **Continues** with available subjects
- ✅ **Outputs** clean subject list for design matrix

**Example manifest:**
```json
{
  "subjects_included": 45,
  "subjects_excluded": 3,
  "included_subject_ids": ["IRC805-0580101", "IRC805-0580103", ...],
  "excluded_subjects": [
    {
      "subject_id": "IRC805-0580102",
      "reason": "FA file not found"
    }
  ]
}
```

### What to Do Next

1. **Review `subject_manifest.json`** - Check which subjects were excluded and why
2. **Filter participants.csv** - Remove excluded subjects to match `subject_list.txt`
3. **Proceed to Phase 2** - Run statistical analysis with different models

## Phase 2: Statistical Analysis

> **Note**: `run_tbss_stats.py` is not yet implemented. Coming soon!

Preview of planned usage:

```bash
# Run analysis with age + sex model
uv run python -m neurovrai.analysis.tbss.run_tbss_stats \
    --data-dir /study/analysis/tbss_FA/ \
    --participants participants_filtered.csv \
    --design "age + sex + exposure" \
    --contrasts contrasts.yaml \
    --output-dir /study/analysis/tbss_FA/model_age_sex_exposure/

# Try different model
uv run python -m neurovrai.analysis.tbss.run_tbss_stats \
    --data-dir /study/analysis/tbss_FA/ \
    --participants participants_filtered.csv \
    --design "age + sex + exposure + age*sex" \
    --contrasts contrasts_interaction.yaml \
    --output-dir /study/analysis/tbss_FA/model_interaction/
```

## Tips

### Multiple Metrics

Prepare each metric separately:

```bash
for metric in FA MD AD RD; do
    uv run python -m neurovrai.analysis.tbss.prepare_tbss \
        --config config.yaml \
        --metric $metric \
        --output-dir /study/analysis/tbss_${metric}/
done
```

### Specific Subjects Only

If you want to exclude certain subjects manually:

```bash
uv run python -m neurovrai.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --metric FA \
    --subjects IRC805-0580101 IRC805-0580103 IRC805-0580105 \
    --output-dir /study/analysis/tbss_FA/
```

### Quality Control

Before running TBSS preparation:
1. Review DWI QC reports for motion/artifacts
2. Identify subjects to exclude based on QC
3. Either use `--subjects` flag or let workflow discover all and review manifest

## Troubleshooting

### "FA file not found"
- Check that DWI preprocessing completed successfully
- Verify file naming: expects `{subject}/dwi/dti/FA.nii.gz`
- Check `derivatives_dir` in config.yaml is correct

### TBSS pipeline fails
- Ensure FSL is loaded: `echo $FSLDIR`
- Check FSL version: `cat $FSLDIR/etc/fslversion`
- Review log file in `output_dir/logs/`

### No subjects included
- Check `derivatives_dir` path in config
- Verify subject directories exist: `ls derivatives_dir/`
- Check file structure matches expected pattern

## References

- FSL TBSS: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide
- TBSS paper: Smith et al. (2006) NeuroImage
