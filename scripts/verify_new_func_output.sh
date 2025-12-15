#!/bin/bash
# verify_new_func_output.sh
#
# Verify that NEW functional preprocessing pipeline completed successfully
#
# Usage: ./verify_new_func_output.sh <subject_id>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <subject_id>"
    echo "Example: $0 IRC805-0580101"
    exit 1
fi

subject="$1"
func_dir="/mnt/bytopia/IRC805/derivatives/${subject}/func"

echo "=========================================="
echo "Verifying NEW pipeline output for $subject"
echo "=========================================="
echo ""

# Check if func directory exists
if [ ! -d "$func_dir" ]; then
    echo "✗ Functional directory not found: $func_dir"
    exit 1
fi

# Check required files
required_files=(
    "${func_dir}/${subject}_bold_preprocessed.nii.gz"
    "${func_dir}/registration/func_to_t1w0GenericAffine.mat"
    "${func_dir}/qc/${subject}_func_qc_report.html"
)

optional_files=(
    "${func_dir}/${subject}_bold_bandpass_filtered.nii.gz"
    "${func_dir}/denoised/denoised_func_data_nonaggr.nii.gz"
    "${func_dir}/motion_correction/*.par"
    "${func_dir}/qc/motion_qc.json"
    "${func_dir}/qc/tsnr_qc.json"
)

all_present=true

echo "Checking REQUIRED files..."
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ MISSING: $file"
        all_present=false
    fi
done

echo ""
echo "Checking OPTIONAL files..."
for pattern in "${optional_files[@]}"; do
    # Use glob expansion
    found=false
    for file in $pattern; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            echo "  ✓ $file ($size)"
            found=true
        fi
    done
    if [ "$found" = false ]; then
        echo "  ⚠ Not found: $pattern"
    fi
done

echo ""
echo "Checking TransformRegistry..."
transform_file="/mnt/bytopia/IRC805/transforms/${subject}/func_to_T1w_0GenericAffine.mat"
if [ -f "$transform_file" ]; then
    echo "  ✓ Transform registered: $transform_file"
else
    echo "  ⚠ Transform not in registry (may be saved locally only)"
fi

echo ""
echo "=========================================="
if [ "$all_present" = true ]; then
    echo "✓ NEW pipeline output COMPLETE for $subject"
    echo "=========================================="
    exit 0
else
    echo "✗ NEW pipeline output INCOMPLETE for $subject"
    echo "=========================================="
    exit 1
fi
