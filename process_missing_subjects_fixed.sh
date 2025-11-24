#!/bin/bash
#
# Batch process missing DTI subjects for TBSS (fixed version)
# Excludes IRC805-4960101 (incompatible matrix sizes)
#

set -e

SUBJECTS=(
    "IRC805-2350101"
    "IRC805-3280201"
    "IRC805-3580101"
    "IRC805-3840101"
)

echo "========================================================================"
echo "BATCH DWI PREPROCESSING FOR MISSING TBSS SUBJECTS"
echo "========================================================================"
echo "Processing ${#SUBJECTS[@]} subjects (excluding IRC805-4960101 - incompatible data)"
echo ""

for subject in "${SUBJECTS[@]}"; do
    echo "========================================================================"
    echo "Processing: $subject"
    echo "========================================================================"
    echo ""

    uv run python run_simple_pipeline.py \
        --subject "$subject" \
        --nifti-dir "/mnt/bytopia/IRC805/bids/$subject" \
        --config config.yaml \
        --skip-anat \
        --skip-func \
        --skip-asl

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Check if FA map was actually created
        fa_file="/mnt/bytopia/IRC805/derivatives/$subject/dwi/dti_FA.nii.gz"
        if [ -f "$fa_file" ]; then
            echo ""
            echo "✓ $subject completed successfully - FA map created"
            echo ""
        else
            echo ""
            echo "⚠ $subject completed but no FA map found"
            echo ""
        fi
    else
        echo ""
        echo "✗ $subject failed with exit code $exit_code"
        echo ""
    fi
done

echo "========================================================================"
echo "BATCH PROCESSING COMPLETE"
echo "========================================================================"
echo "Check derivatives for FA maps:"
for subject in "${SUBJECTS[@]}"; do
    fa_file="/mnt/bytopia/IRC805/derivatives/$subject/dwi/dti_FA.nii.gz"
    if [ -f "$fa_file" ]; then
        echo "  ✓ $subject"
    else
        echo "  ✗ $subject (no FA map)"
    fi
done
