# Simple Pipeline Usage Guide

This guide covers the simplified, human-readable pipeline scripts.

## Overview

Two new scripts provide a streamlined workflow:

1. **`run_simple_pipeline.py`** - Process a single subject
2. **`run_batch_simple.py`** - Process multiple subjects

## Key Simplifications

✓ **No monitoring code** - Straightforward sequential execution
✓ **No parallel processing** - One modality at a time, easy to debug
✓ **No class complexity** - Simple functions with clear names
✓ **Human-readable** - Easy to understand the flow
✓ **Linear execution** - DICOM → Anat → DWI → Func → ASL

## Single Subject Processing

### Basic Usage

```bash
# From DICOM
python run_simple_pipeline.py \
    --subject IRC805-0580101 \
    --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-0580101 \
    --config config.yaml

# From existing NIfTI
python run_simple_pipeline.py \
    --subject IRC805-0580101 \
    --nifti-dir /mnt/bytopia/IRC805/bids/IRC805-0580101 \
    --config config.yaml
```

### Skip Specific Modalities

```bash
# Only run anatomical and DWI
python run_simple_pipeline.py \
    --subject IRC805-0580101 \
    --dicom-dir /path/to/dicom \
    --config config.yaml \
    --skip-func \
    --skip-asl

# Only run anatomical
python run_simple_pipeline.py \
    --subject IRC805-0580101 \
    --dicom-dir /path/to/dicom \
    --config config.yaml \
    --skip-dwi \
    --skip-func \
    --skip-asl
```

## Batch Processing

### Process All Subjects

```bash
# Automatically finds all subjects in DICOM directory
python run_batch_simple.py --config config.yaml
```

### Process Specific Subjects

```bash
# Only process selected subjects
python run_batch_simple.py \
    --config config.yaml \
    --subjects IRC805-0580101 IRC805-1580201 IRC805-2570202
```

## Pipeline Flow

The simple pipeline executes these steps in order:

```
1. DICOM → NIfTI Conversion
   ↓
2. Anatomical Preprocessing (required)
   ↓
3. DWI Preprocessing (optional)
   ↓
4. Functional Preprocessing (optional)
   ↓
5. ASL Preprocessing (optional)
```

### Notes

- **Anatomical preprocessing is required** for functional and ASL (provides reference images and tissue masks)
- **Each step must complete** before the next begins
- **If a step fails**, the pipeline stops (single subject) or continues to next subject (batch)
- **All outputs** go to the standard directories from `config.yaml`

## Output Structure

```
{study_root}/
├── bids/                    # NIfTI files (if converted from DICOM)
│   └── {subject}/
│       ├── anat/
│       ├── dwi/
│       ├── func/
│       └── asl/
├── derivatives/             # Preprocessed outputs
│   └── {subject}/
│       ├── anat/
│       ├── dwi/
│       ├── func/
│       └── asl/
├── work/                    # Temporary Nipype files
│   └── {subject}/
└── qc/                      # Quality control reports
    └── {subject}/
```

## Logging

- **Console output**: Shows progress and results
- **Log file**: `logs/simple_pipeline.log`

## Debugging Tips

1. **Check the log file** if a step fails: `tail -f logs/simple_pipeline.log`
2. **Run with single subject first** before batch processing
3. **Use skip flags** to test individual modalities
4. **Check work directory** for Nipype intermediate files if something fails

## Comparison with Original Scripts

| Feature | Original (`run_full_pipeline.py`) | Simple (`run_simple_pipeline.py`) |
|---------|-----------------------------------|-----------------------------------|
| Code structure | Class-based (600+ lines) | Function-based (~350 lines) |
| Execution | Parallel workflows | Sequential |
| Monitoring | Progress tracking, summaries | Simple success/fail |
| Complexity | High (orchestrator, futures) | Low (linear flow) |
| Debugging | Harder | Easier |
| Readability | Moderate | High |

## When to Use Which Script

**Use `run_simple_pipeline.py` when:**
- You want clear, understandable code
- You're debugging a specific modality
- You don't need parallel processing
- You want to easily modify the pipeline

**Use `run_full_pipeline.py` when:**
- You need parallel processing for speed
- You want detailed monitoring and summaries
- You need the orchestrator features

## Example Session

```bash
# Create logs directory
mkdir -p logs

# Test on single subject first
python run_simple_pipeline.py \
    --subject IRC805-0580101 \
    --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-0580101 \
    --config config.yaml

# If successful, run batch on all subjects
python run_batch_simple.py --config config.yaml
```

## Customization

The simple pipeline is designed to be easily modified:

1. **Add custom steps**: Insert new function calls in `main()`
2. **Modify order**: Rearrange function calls as needed
3. **Add parameters**: Extend argument parser
4. **Change error handling**: Modify try/except blocks

All preprocessing logic is in `mri_preprocess/workflows/`, keeping the main script clean and readable.
