# Configuration Setup - Summary

## What We Fixed

### 1. ✅ Config Generator Created
- **File:** `create_config.py`
- **Purpose:** Auto-generate `config.yaml` for your study
- **Usage:** `python create_config.py --study-root /mnt/bytopia/IRC805`
- **Creates:** `/mnt/bytopia/IRC805/config.yaml` (in study root by default)

### 2. ✅ Updated config.yaml
**Old config had:**
- ❌ Missing `dicom_dir` (didn't specify where DICOMs are)
- ❌ Missing `bids_dir` (didn't specify NIfTI output location)
- ❌ Referenced wrong paths (`subjects/` instead of `raw/dicom/`)

**New config has:**
- ✅ `dicom_dir: /mnt/bytopia/IRC805/raw/dicom` (where DICOMs are)
- ✅ `bids_dir: /mnt/bytopia/IRC805/bids` (where NIfTI files go)
- ✅ `derivatives_dir: /mnt/bytopia/IRC805/derivatives` (preprocessed outputs)
- ✅ All required sections with proper defaults
- ✅ Comments explaining directory structure

### 3. ✅ Documentation Created
- **CONFIG_SETUP.md** - Complete config guide with examples
- **create_config.py** - Interactive config generator
- **Updated QUICKSTART.md** - Now includes config creation as step 1

## Current config.yaml Structure

```yaml
# Correct paths for IRC805
project_dir: /mnt/bytopia/IRC805
dicom_dir: /mnt/bytopia/IRC805/raw/dicom      # ← NOW SPECIFIED!
bids_dir: /mnt/bytopia/IRC805/bids            # ← NOW SPECIFIED!
derivatives_dir: /mnt/bytopia/IRC805/derivatives
work_dir: /mnt/bytopia/IRC805/work

paths:
  logs: /mnt/bytopia/IRC805/logs
  transforms: /mnt/bytopia/IRC805/transforms
  qc: /mnt/bytopia/IRC805/qc

execution:
  plugin: MultiProc
  n_procs: 6

templates:
  mni152_t1_2mm: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
  mni152_t1_1mm: /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz
  fmrib58_fa: /usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz

# Plus complete anatomical, diffusion, functional, and ASL sections...
```

## Directory Structure

The config now correctly maps to this structure:

```
/mnt/bytopia/IRC805/                    # project_dir
├── raw/
│   └── dicom/                          # dicom_dir ← DICOMs go here
│       ├── IRC805-0580101/
│       ├── IRC805-1580201/
│       └── ...
├── bids/                               # bids_dir ← NIfTI files created here
│   ├── IRC805-0580101/
│   │   ├── anat/
│   │   ├── dwi/
│   │   ├── func/
│   │   └── asl/
│   └── ...
├── derivatives/                        # Preprocessed outputs
│   ├── IRC805-0580101/
│   │   ├── anat/
│   │   ├── dwi/
│   │   └── func/
│   └── ...
├── work/                               # Temporary Nipype files
├── qc/                                 # QC reports
├── logs/                               # Log files
└── transforms/                         # Spatial transforms
```

## What You Need to Check

### ⚠️ Critical Parameters to Verify

Open `config.yaml` and check:

1. **DWI readout_time** (line ~44)
   ```yaml
   diffusion:
     topup:
       readout_time: 0.05  # ← Check your MRI protocol!
   ```
   - Must match your scanner's EPI readout time
   - Typical values: 0.03-0.08 seconds
   - Find in: MRI protocol document

2. **Functional TR** (line ~64)
   ```yaml
   functional:
     tr: 1.029  # ← Check your protocol!
   ```
   - Repetition time in seconds
   - Must match your functional sequence

3. **Functional TE** (line ~65)
   ```yaml
   functional:
     te: [10.0, 30.0, 50.0]  # ← Check your protocol!
   ```
   - Single echo: one value `te: 30.0`
   - Multi-echo: list `te: [10.0, 30.0, 50.0]`

4. **Number of processors** (line ~17)
   ```yaml
   execution:
     n_procs: 6  # ← Adjust for your system
   ```
   - Set to number of CPU cores available
   - Typical: 4-12

5. **TEDANA PCA Components (multi-echo only)** (line ~84)
   ```yaml
   functional:
     tedana:
       tedpca: 225  # ← Important for ICA convergence!
   ```
   - Recommended: `num_volumes / 2` (e.g., 225 for 450 volumes)
   - Options: integer (exact components), float 0-1 (variance %), or string ('kundu', 'aic', etc.)
   - If TEDANA fails to converge with 'kundu', use manual value

## Validation Steps

### 1. Check Config Syntax
```bash
uv run python -c "from mri_preprocess.config import load_config; load_config('config.yaml'); print('✓ Config valid')"
```

### 2. Verify DICOM Directory
```bash
# Check DICOM directory exists with subjects
ls /mnt/bytopia/IRC805/raw/dicom/

# Should show subject directories:
# IRC805-0580101/
# IRC805-1580201/
# etc.
```

### 3. Check FSL Templates
```bash
# Verify template files exist
ls -lh /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
ls -lh /usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz
```

## Next Steps

### If Starting Fresh

```bash
# 1. Create config (if you haven't already)
python create_config.py --study-root /mnt/bytopia/IRC805

# 2. Review and edit /mnt/bytopia/IRC805/config.yaml
#    - Check readout_time
#    - Check TR and TE
#    - Verify paths

# 3. Ensure DICOM files are in place
ls /mnt/bytopia/IRC805/raw/dicom/*/

# 4. Run pipeline
uv run python run_simple_pipeline.py \
    --subject IRC805-0580101 \
    --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-0580101 \
    --config /mnt/bytopia/IRC805/config.yaml
```

### If Updating Existing Config

```bash
# Backup old config
mv config.yaml config_backup.yaml

# Create new config
python create_config.py --study-root /mnt/bytopia/IRC805

# Copy over your custom settings from backup
# (TR, TE, readout_time, etc.)
```

## Files Reference

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration (ready to use!) |
| `config_old.yaml` | Previous config (backup) |
| `create_config.py` | Config generator script |
| `CONFIG_SETUP.md` | Complete configuration guide |
| `QUICKSTART.md` | Updated with config step |
| `CONFIG_SUMMARY.md` | This file |

## Common Questions

**Q: Where do I put my DICOM files?**
A: `/mnt/bytopia/IRC805/raw/dicom/{subject_id}/`

**Q: Where will NIfTI files be created?**
A: `/mnt/bytopia/IRC805/bids/{subject_id}/anat/`, `dwi/`, `func/`, `asl/`

**Q: Where are the final preprocessed outputs?**
A: `/mnt/bytopia/IRC805/derivatives/{subject_id}/`

**Q: Do I need to create the bids/ and derivatives/ directories?**
A: No! The pipeline creates them automatically.

**Q: Can I use different paths?**
A: Yes! Edit `config.yaml` paths as needed. Just ensure `dicom_dir` points to your DICOMs.

**Q: How do I update sequence-specific parameters?**
A: Edit `config.yaml` and update `tr`, `te`, `readout_time` based on your MRI protocol.

## Troubleshooting

**Problem:** Pipeline can't find DICOM files
**Solution:** Check `dicom_dir` in config.yaml matches actual location

**Problem:** Wrong directory structure created
**Solution:** Config likely has wrong paths. Regenerate with `create_config.py`

**Problem:** Not sure about readout_time
**Solution:** Check MRI protocol PDF or contact MRI physicist

**Problem:** Multi-echo but using single-echo settings
**Solution:** Update `te:` to list of values: `te: [10.0, 30.0, 50.0]`

## Summary

✅ Config generator created (`create_config.py`)
✅ Config updated with correct IRC805 paths
✅ DICOM directory properly specified
✅ BIDS output directory specified
✅ Complete documentation provided
✅ QUICKSTART updated with config step

**You're ready to run the pipeline!** Just verify the critical parameters (TR, TE, readout_time) and go.
