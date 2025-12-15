#!/bin/bash
# identify_old_func_outputs.sh
#
# Identify subjects with OLD vs NEW functional preprocessing outputs
#
# Usage: ./identify_old_func_outputs.sh

echo "=========================================="
echo "Scanning for OLD and NEW functional outputs"
echo "=========================================="
echo ""

derivatives_dir="/mnt/bytopia/IRC805/derivatives"

subjects_with_old=()
subjects_with_new=()
subjects_with_both=()
subjects_with_func=()

# Scan all subject directories
for subject_dir in "$derivatives_dir"/IRC805-*/func; do
    if [ ! -d "$subject_dir" ]; then
        continue
    fi

    subject=$(basename $(dirname "$subject_dir"))
    subjects_with_func+=("$subject")

    has_old=false
    has_new=false

    # Check for OLD patterns
    if ls "$subject_dir"/*_mcf_bp_smooth.nii.gz 2>/dev/null | grep -q .; then
        has_old=true
    fi

    if [ -f "$subject_dir/func_brain.nii.gz" ] || [ -f "$subject_dir/func_mask.nii.gz" ]; then
        has_old=true
    fi

    if [ -f "$subject_dir/registration/func_to_mni_Composite.h5" ]; then
        has_old=true
    fi

    # Check for NEW patterns
    if [ -f "$subject_dir/${subject}_bold_preprocessed.nii.gz" ]; then
        has_new=true
    fi

    # Categorize
    if [ "$has_old" = true ] && [ "$has_new" = true ]; then
        subjects_with_both+=("$subject")
    elif [ "$has_old" = true ]; then
        subjects_with_old+=("$subject")
    elif [ "$has_new" = true ]; then
        subjects_with_new+=("$subject")
    fi
done

# Report results
echo "Summary:"
echo "  Total subjects with functional data: ${#subjects_with_func[@]}"
echo "  - NEW outputs only: ${#subjects_with_new[@]}"
echo "  - OLD outputs only: ${#subjects_with_old[@]}"
echo "  - BOTH old and new (needs cleanup): ${#subjects_with_both[@]}"
echo ""

if [ ${#subjects_with_both[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Subjects with BOTH old and new outputs"
    echo "(These need cleanup after verification)"
    echo "=========================================="
    for subject in "${subjects_with_both[@]}"; do
        echo "  $subject"
    done
    echo ""
fi

if [ ${#subjects_with_old[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Subjects with OLD outputs only"
    echo "(These need reprocessing with NEW pipeline)"
    echo "=========================================="
    for subject in "${subjects_with_old[@]}"; do
        echo "  $subject"
    done
    echo ""
fi

if [ ${#subjects_with_new[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Subjects with NEW outputs only"
    echo "(No cleanup needed)"
    echo "=========================================="
    for subject in "${subjects_with_new[@]}"; do
        echo "  $subject"
    done
    echo ""
fi

# Provide next steps
if [ ${#subjects_with_both[@]} -gt 0 ]; then
    echo "=========================================="
    echo "Recommended Cleanup Steps:"
    echo "=========================================="
    echo ""
    echo "For each subject with BOTH outputs:"
    echo ""
    echo "1. Verify NEW pipeline completed successfully:"
    echo "   ./verify_new_func_output.sh <subject_id>"
    echo ""
    echo "2. Archive OLD outputs (safe, reversible):"
    echo "   ./archive_old_func_outputs.sh <subject_id>"
    echo ""
    echo "Example for ${subjects_with_both[0]}:"
    echo "  ./verify_new_func_output.sh ${subjects_with_both[0]}"
    echo "  ./archive_old_func_outputs.sh ${subjects_with_both[0]}"
    echo ""
fi
