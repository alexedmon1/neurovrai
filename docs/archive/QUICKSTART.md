# Quick Start Guide

**TL;DR**: How to run the pipeline in 4 steps

## Step 1: Create Config (First Time Only)

```bash
cd /home/edm9fd/sandbox/human-mri-preprocess

# Create config.yaml for your study (creates /mnt/bytopia/IRC805/config.yaml)
python create_config.py --study-root /mnt/bytopia/IRC805

# Review and edit if needed
nano /mnt/bytopia/IRC805/config.yaml  # or vim/code
```

## Step 2: Install Dependencies (First Time Only)

```bash
uv sync
```

## Step 3: Verify Installation

```bash
uv run python verify_environment.py
```

**Expected:** ✓ All required dependencies are installed and ready!

## Step 4: Run Pipeline

### Single Subject

```bash
uv run python run_simple_pipeline.py \
    --subject IRC805-0580101 \
    --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-0580101 \
    --config /mnt/bytopia/IRC805/config.yaml
```

### All Subjects

```bash
uv run python run_batch_simple.py --config /mnt/bytopia/IRC805/config.yaml
```

## That's It!

**Key Points:**
- ✅ Always use `uv run python` (not just `python`)
- ✅ Run `uv sync` after pulling updates
- ✅ Check logs in `logs/simple_pipeline.log`
- ✅ Outputs go to `/mnt/bytopia/IRC805/derivatives/{subject}/`

## Troubleshooting

**Problem:** "Module not found" error
**Solution:** `uv sync`

**Problem:** Old/stale packages
**Solution:** `rm -rf .venv && uv sync`

**Problem:** "FSLDIR not set"
**Solution:** `module load fsl` or `source /usr/local/fsl/etc/fslconf/fsl.sh`

## Full Documentation

- **Complete setup:** `SETUP_GUIDE.md`
- **Pipeline usage:** `SIMPLE_PIPELINE_GUIDE.md`
- **Config reference:** `README.md`
