#!/bin/bash
#
# Batch Functional Preprocessing for All IRC805 Subjects
# Only processes functional data (skips anat/dwi/asl)
#
# Usage: bash batch_func_only.sh

set -e

CONFIG="/mnt/bytopia/IRC805/config.yaml"
STUDY_ROOT="/mnt/bytopia/IRC805"
BIDS_DIR="$STUDY_ROOT/bids"
LOG_DIR="$STUDY_ROOT/logs/batch_func"

mkdir -p "$LOG_DIR"

# Find all subjects with BIDS data
subjects=$(ls -d $BIDS_DIR/IRC805-* | xargs -n 1 basename | sort)

echo "========================================================================"
echo "Batch Functional Preprocessing"
echo "========================================================================"
echo "Study: $STUDY_ROOT"
echo "Config: $CONFIG"
echo "Subjects found: $(echo "$subjects" | wc -l)"
echo ""
echo "Starting: $(date)"
echo "========================================================================"
echo ""

success_count=0
failed_count=0
failed_subjects=""

for subject in $subjects; do
    echo "------------------------------------------------------------------------"
    echo "Processing: $subject"
    echo "------------------------------------------------------------------------"

    log_file="$LOG_DIR/${subject}_func_preprocess.log"

    # Run functional preprocessing only
    if uv run python run_simple_pipeline.py \
        --subject "$subject" \
        --config "$CONFIG" \
        --nifti-dir "$BIDS_DIR/$subject" \
        --skip-anat \
        --skip-dwi \
        --skip-asl \
        2>&1 | tee "$log_file"; then

        echo "✓ SUCCESS: $subject"
        ((success_count++))
    else
        echo "✗ FAILED: $subject"
        ((failed_count++))
        failed_subjects="$failed_subjects\n  - $subject"
    fi

    echo ""
done

echo ""
echo "========================================================================"
echo "Batch Processing Complete"
echo "========================================================================"
echo "Finished: $(date)"
echo ""
echo "Total subjects: $(echo "$subjects" | wc -l)"
echo "Successful: $success_count"
echo "Failed: $failed_count"
echo ""

if [ $failed_count -gt 0 ]; then
    echo "Failed subjects:"
    echo -e "$failed_subjects"
    echo ""
fi

echo "Logs saved to: $LOG_DIR"
echo "========================================================================"

exit $failed_count
