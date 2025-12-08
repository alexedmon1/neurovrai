#!/bin/bash
# Run FSL randomise for all TBSS metrics in parallel
# Date: 2025-12-07

set -e

STATS_DIR="/mnt/bytopia/IRC805/analysis/tbss/stats"
DESIGN_DIR="/mnt/bytopia/IRC805/data/designs/tbss"
OUTPUT_BASE="/mnt/bytopia/IRC805/analysis/tbss"
SKELETON_MASK="/mnt/bytopia/IRC805/analysis/tbss/FA/stats/mean_FA_skeleton_mask.nii.gz"
N_PERM=5000

echo "========================================="
echo "FSL Randomise - All TBSS Metrics"
echo "========================================="
echo "Design: $DESIGN_DIR"
echo "Stats Directory: $STATS_DIR"
echo "Permutations: $N_PERM"
echo "========================================="
echo

# Define all metrics to analyze
ALL_METRICS="FA MD AD RD MK AK RK KFA FICVF ODI FISO"

# Function to run randomise for a single metric
run_randomise() {
    local metric=$1
    local input="$STATS_DIR/all_${metric}_skeletonised.nii.gz"
    local output_dir="$OUTPUT_BASE/$metric/randomise_output"
    local output_prefix="$output_dir/randomise"

    echo "[$(date +%H:%M:%S)] Starting randomise for $metric..."

    # Create output directory
    mkdir -p "$output_dir"

    # Run randomise with TFCE (no mask needed for skeleton data)
    # -T: TFCE correction
    # --T2: Two-tailed testing (tests both positive and negative tails)
    # -x: Don't output raw stat images (only corrected p-values)
    randomise \
        -i "$input" \
        -o "$output_prefix" \
        -d "$DESIGN_DIR/design.mat" \
        -t "$DESIGN_DIR/design.con" \
        -n "$N_PERM" \
        -T \
        --T2 \
        -x \
        > "$output_dir/randomise.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ $metric completed"
        echo "  Output: $output_dir"
    else
        echo "[$(date +%H:%M:%S)] ✗ $metric FAILED"
        tail -20 "$output_dir/randomise.log"
    fi
}

# Export function for parallel execution
export -f run_randomise
export STATS_DIR DESIGN_DIR OUTPUT_BASE SKELETON_MASK N_PERM

# Run all metrics in parallel (4 at a time to avoid overloading)
echo "=== Running randomise for all metrics (4 parallel jobs) ==="
echo "$ALL_METRICS" | tr ' ' '\n' | xargs -P 4 -I {} bash -c 'run_randomise {}'

echo
echo "========================================="
echo "All Randomise Jobs Complete"
echo "========================================="
echo

# Verify outputs
echo "=== Verification ==="
for metric in $ALL_METRICS; do
    output_dir="$OUTPUT_BASE/$metric/randomise_output"
    if [ -f "$output_dir/randomise_tfce_corrp_tstat1.nii.gz" ]; then
        n_files=$(ls "$output_dir"/randomise_*.nii.gz 2>/dev/null | wc -l)
        echo "✓ $metric: $n_files output files"
    else
        echo "✗ $metric: Output files NOT found"
    fi
done

echo
echo "========================================="
echo "Next Steps:"
echo "  1. Review significant results (TFCE p < 0.05)"
echo "  2. Generate cluster reports for significant contrasts"
echo "  3. Create HTML summary reports"
echo "========================================="
