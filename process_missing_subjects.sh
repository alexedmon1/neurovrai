#!/bin/bash
#
# Batch process missing DTI subjects for TBSS
# These subjects have raw b=800 DTI data that was previously skipped due to scanner-processed map filtering issue
#

set -e

SUBJECTS=(
    "IRC805-2350101"
    "IRC805-3280201"
    "IRC805-3580101"
    "IRC805-3840101"
    "IRC805-4960101"
)

echo "========================================================================"
echo "BATCH DWI PREPROCESSING FOR MISSING TBSS SUBJECTS"
echo "========================================================================"
echo "Processing ${#SUBJECTS[@]} subjects"
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

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $subject completed successfully"
        echo ""
    else
        echo ""
        echo "✗ $subject failed"
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
