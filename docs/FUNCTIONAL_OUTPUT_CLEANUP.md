# Functional Preprocessing Output File Changes

**Date**: 2025-12-15
**Purpose**: Document differences between old and new functional preprocessing outputs to enable safe cleanup

## Critical Changes in New Pipeline

### Pipeline Order Fix
- **OLD**: Registration performed on filtered data (INCORRECT)
- **NEW**: Registration in Phase 1 before filtering (CORRECT)

### Processing Order
- **OLD**: Motion → BET → Registration → TEDANA/AROMA → Bandpass → Smooth
- **NEW**: Motion → BET → TEDANA/AROMA → ACompCor → Bandpass → Smooth → Registration (to T1w/MNI)

### Transform Management
- **OLD**: Direct ANTs/FSL transform files
- **NEW**: Uses TransformRegistry for centralized transform storage

## File Structure Comparison

### OLD Pipeline Outputs (DELETE THESE)

```
derivatives/{subject}/func/
├── func_brain.nii.gz                                    # OLD: 3D mean brain
├── func_mask.nii.gz                                     # OLD: 3D brain mask
├── {series}_mcf_bp_smooth.nii.gz                        # OLD: Final output (wrong naming)
├── filtered/
│   └── {series}_mcf_bp.nii.gz                           # OLD: Bandpass filtered only
├── brain/
│   └── {series}_mcf_mean_brain_mask.nii.gz              # OLD: Mask from motion-corrected mean
├── motion_correction/
│   ├── {series}_mcf.nii.gz                              # OLD: Motion-corrected 4D (unfiltered)
│   └── {series}_mcf.nii.gz.par                          # OLD: Motion parameters
└── registration/
    ├── transform_list.txt                               # OLD: Legacy transform tracking
    └── func_to_mni_Composite.h5                         # OLD: Registration on FILTERED data (WRONG!)
```

**Key Issues with OLD outputs:**
1. Registration transforms (`func_to_mni_Composite.h5`) were computed on **filtered data** (lacks structural detail for accurate alignment)
2. Missing ACompCor confound removal
3. Inconsistent file naming conventions
4. No integration with TransformRegistry

### NEW Pipeline Outputs (KEEP THESE)

```
derivatives/{subject}/func/
├── {subject}_bold_preprocessed.nii.gz                   # NEW: Final preprocessed output (correct pipeline order)
├── {subject}_bold_bandpass_filtered.nii.gz              # NEW: Intermediate - after ACompCor + bandpass, before smoothing
├── motion_correction/
│   ├── {series}_mcf.nii.gz                              # Motion-corrected 4D (raw, before filtering)
│   ├── {series}_mcf.nii.gz.par                          # Motion parameters (MCFLIRT format)
│   └── func_mean.nii.gz                                 # Temporal mean (for registration)
├── brain/
│   └── {series}_mcf_mean_brain_mask.nii.gz              # Brain mask from motion-corrected mean
├── denoised/
│   └── denoised_func_data_nonaggr.nii.gz                # NEW: TEDANA/AROMA denoised output
├── registration/
│   ├── func_mean.nii.gz                                 # NEW: Motion-corrected mean (UNFILTERED - correct for registration!)
│   ├── func_to_t1w0GenericAffine.mat                    # NEW: Functional → T1w linear transform (ANTs)
│   ├── func_to_t1wWarped.nii.gz                         # NEW: Functional registered to T1w space
│   └── func_to_t1wInverseWarped.nii.gz                  # NEW: Inverse transform (for bringing T1w to func space)
└── qc/                                                   # NEW: Comprehensive QC reports
    ├── motion_qc.json                                    # Motion statistics (FD, displacement)
    ├── tsnr_qc.json                                      # Temporal SNR metrics
    ├── dvars_qc.json                                     # Artifact detection
    ├── skull_strip_qc.json                               # Brain extraction quality
    ├── carpet_plot.png                                   # Visual QC for artifacts
    └── {subject}_func_qc_report.html                     # Integrated HTML report
```

**Key Improvements in NEW outputs:**
1. Registration transforms computed on **motion-corrected unfiltered data** (preserves structural detail)
2. ACompCor confound removal included
3. Consistent subject-based naming (`{subject}_bold_preprocessed.nii.gz`)
4. Integrated with TransformRegistry (transforms saved to `/mnt/bytopia/IRC805/transforms/`)
5. Comprehensive QC framework

## TransformRegistry Integration

**NEW pipeline** saves transforms to centralized location:

```
/mnt/bytopia/IRC805/transforms/{subject}/
├── func_to_T1w_0GenericAffine.mat           # Linear transform: func → T1w (from new pipeline)
├── T1w_to_MNI152_0GenericAffine.mat         # Linear transform: T1w → MNI (from anat preprocessing)
└── T1w_to_MNI152_1Warp.nii.gz               # Nonlinear warp: T1w → MNI (from anat preprocessing)
```

These transforms can be **concatenated** for direct func → MNI normalization without resampling intermediate steps.

## Cleanup Strategy

### Step 1: Identify Subjects with OLD Outputs

Check for presence of OLD file patterns:

```bash
# Find subjects with old-style outputs
for subject_dir in /mnt/bytopia/IRC805/derivatives/IRC805-*/func; do
    subject=$(basename $(dirname $subject_dir))

    # Check for OLD naming pattern
    if ls $subject_dir/*_mcf_bp_smooth.nii.gz 2>/dev/null | grep -q .; then
        echo "OLD: $subject"
    fi

    # Check for NEW naming pattern
    if ls $subject_dir/${subject}_bold_preprocessed.nii.gz 2>/dev/null | grep -q .; then
        echo "NEW: $subject"
    fi
done
```

### Step 2: Verify NEW Pipeline Completion

Before deleting OLD files, ensure NEW pipeline completed successfully:

**Required files for valid NEW output:**
1. `{subject}_bold_preprocessed.nii.gz` - Final preprocessed data
2. `registration/func_to_t1w0GenericAffine.mat` - Registration transform
3. `qc/{subject}_func_qc_report.html` - QC report

**Verification script:**

```bash
#!/bin/bash

subject="$1"
func_dir="/mnt/bytopia/IRC805/derivatives/${subject}/func"

# Check required files
required_files=(
    "${func_dir}/${subject}_bold_preprocessed.nii.gz"
    "${func_dir}/registration/func_to_t1w0GenericAffine.mat"
    "${func_dir}/qc/${subject}_func_qc_report.html"
)

all_present=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "MISSING: $file"
        all_present=false
    fi
done

if [ "$all_present" = true ]; then
    echo "✓ NEW pipeline output complete for $subject"
    exit 0
else
    echo "✗ NEW pipeline output INCOMPLETE for $subject"
    exit 1
fi
```

### Step 3: Delete OLD Files

**ONLY delete after verifying NEW pipeline completed successfully!**

```bash
#!/bin/bash
# cleanup_old_func_outputs.sh
#
# Usage: ./cleanup_old_func_outputs.sh <subject_id>
#
# DANGER: This deletes old functional preprocessing outputs!
# Run verification script first!

subject="$1"
func_dir="/mnt/bytopia/IRC805/derivatives/${subject}/func"

echo "Cleaning up OLD functional outputs for $subject..."

# Delete OLD file patterns
rm -f "$func_dir/func_brain.nii.gz"
rm -f "$func_dir/func_mask.nii.gz"
rm -f "$func_dir"/*_mcf_bp_smooth.nii.gz
rm -rf "$func_dir/filtered"

# Delete OLD registration files (wrong - computed on filtered data)
rm -f "$func_dir/registration/transform_list.txt"
rm -f "$func_dir/registration/func_to_mni_Composite.h5"

echo "✓ Cleanup complete for $subject"
echo ""
echo "Remaining files:"
ls -lh "$func_dir"
```

### Step 4: Archive Before Deletion (RECOMMENDED)

Create backup before deletion:

```bash
#!/bin/bash
# archive_old_func_outputs.sh
#
# Archive OLD functional outputs before deletion

subject="$1"
func_dir="/mnt/bytopia/IRC805/derivatives/${subject}/func"
archive_dir="/mnt/bytopia/IRC805/archive/old_func_outputs/${subject}"

mkdir -p "$archive_dir"

# Move OLD files to archive
mv "$func_dir/func_brain.nii.gz" "$archive_dir/" 2>/dev/null
mv "$func_dir/func_mask.nii.gz" "$archive_dir/" 2>/dev/null
mv "$func_dir"/*_mcf_bp_smooth.nii.gz "$archive_dir/" 2>/dev/null
mv "$func_dir/filtered" "$archive_dir/" 2>/dev/null

# Move OLD registration files
mkdir -p "$archive_dir/registration"
mv "$func_dir/registration/transform_list.txt" "$archive_dir/registration/" 2>/dev/null
mv "$func_dir/registration/func_to_mni_Composite.h5" "$archive_dir/registration/" 2>/dev/null

echo "✓ OLD outputs archived to: $archive_dir"
```

## Subject-Specific Status (as of 2025-12-15)

### IRC805-2350101 (Test Subject)
- **Status**: Mixed OLD and NEW outputs
- **Action**: Verify NEW output complete, then delete/archive OLD files
- **OLD files present**:
  - `func_brain.nii.gz`, `func_mask.nii.gz`
  - `601_WIP_RESTING_STATE_20220719153553__mcf_bp_smooth.nii.gz`
  - `filtered/601_WIP_RESTING_STATE_20220719153553__mcf_bp.nii.gz`
  - `registration/func_to_mni_Composite.h5` (WRONG - computed on filtered data)
- **NEW files present**:
  - `IRC805-2350101_bold_preprocessed.nii.gz` ✓
  - `registration/func_to_t1w0GenericAffine.mat` ✓
  - `denoised/denoised_func_data_nonaggr.nii.gz` ✓

### IRC805-0580101
- **Status**: Currently being processed by NEW pipeline (batch run started 13:20:46)
- **Action**: Wait for completion, verify outputs

### Remaining 19 Subjects
- **Status**: Will be processed by NEW pipeline
- **Action**: No cleanup needed (no OLD outputs)

## Important Notes

1. **DO NOT delete files from IRC805-2350101 until NEW pipeline batch run completes** - we need to verify the new outputs are correct

2. **Registration accuracy**: The most critical fix is that registration is now performed on **motion-corrected unfiltered data** instead of filtered data. This ensures accurate anatomical alignment.

3. **ACompCor inclusion**: NEW pipeline includes ACompCor confound removal, which OLD pipeline was missing.

4. **Transform reusability**: NEW pipeline integrates with TransformRegistry, allowing transform reuse across modalities.

5. **QC integration**: NEW pipeline includes comprehensive quality control that OLD pipeline lacked.

## Verification Checklist

Before using NEW outputs for analysis:

- [ ] Verify `{subject}_bold_preprocessed.nii.gz` exists
- [ ] Verify registration transform exists (`registration/func_to_t1w0GenericAffine.mat`)
- [ ] Check QC report (`qc/{subject}_func_qc_report.html`)
- [ ] Verify transform in TransformRegistry (`/mnt/bytopia/IRC805/transforms/{subject}/func_to_T1w_0GenericAffine.mat`)
- [ ] Check motion QC metrics (FD, displacement) are reasonable
- [ ] Verify tSNR is in expected range
- [ ] Review carpet plot for remaining artifacts

## Questions?

See `docs/FUNCTIONAL_PREPROCESSING_ISSUES.md` for detailed analysis of the registration bug that necessitated this reprocessing.
