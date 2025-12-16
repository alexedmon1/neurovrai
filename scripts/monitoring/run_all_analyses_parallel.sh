#!/bin/bash
#
# Run all group-level analyses in parallel
#
# This script launches all statistical analyses as background jobs, allowing
# parallel execution up to system resource limits.
#
# NEW ARCHITECTURE (2025-12-05):
# - Uses pre-generated design matrices from generate_design_matrices.py
# - Validates design-to-data alignment before running analyses
# - Separates design creation from statistical analysis
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/parallel_analyses_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

STUDY_ROOT="/mnt/bytopia/IRC805"

echo "=========================================================================="
echo "PARALLEL ANALYSIS EXECUTION"
echo "=========================================================================="
echo "Study root: $STUDY_ROOT"
echo "Log directory: $LOG_DIR"
echo ""

# =============================================================================
# Step 0: Ensure design matrices exist
# =============================================================================
echo "=========================================================================="
echo "CHECKING DESIGN MATRICES"
echo "=========================================================================="

DESIGN_ROOT="$STUDY_ROOT/data/designs"
DESIGNS_NEEDED=(vbm asl func_reho func_falff tbss)
DESIGNS_MISSING=()

for design in "${DESIGNS_NEEDED[@]}"; do
    if [ ! -f "$DESIGN_ROOT/$design/design.mat" ]; then
        DESIGNS_MISSING+=("$design")
        echo "  ✗ Missing: $design"
    else
        echo "  ✓ Found: $design"
    fi
done

if [ ${#DESIGNS_MISSING[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Missing design matrices for: ${DESIGNS_MISSING[@]}"
    echo "Please run generate_design_matrices.py first:"
    echo "  uv run python generate_design_matrices.py --all --study-root $STUDY_ROOT"
    exit 1
fi

echo ""
echo "✓ All design matrices found"
echo ""

# Track PIDs of background jobs
declare -a PIDS
declare -a JOBS

# Function to launch analysis
launch_analysis() {
    local name="$1"
    local logfile="$2"
    shift 2
    local cmd="$@"

    echo "Launching: $name"
    echo "  Command: $cmd"
    echo "  Log: $logfile"

    # Run in background, redirect output to log
    eval "$cmd" > "$logfile" 2>&1 &
    local pid=$!

    PIDS+=($pid)
    JOBS+=("$name")

    echo "  PID: $pid"
    echo ""
}

# =============================================================================
# 1. VBM Analysis (23 subjects, ~45 min)
# =============================================================================
launch_analysis \
    "VBM" \
    "$LOG_DIR/vbm.log" \
    "uv run python run_vbm_group_analysis.py \
        --study-root $STUDY_ROOT \
        --design-dir $DESIGN_ROOT/vbm \
        --tissue GM \
        --method randomise \
        --n-permutations 5000"

# =============================================================================
# 2. TBSS Analysis - All 11 metrics (17 subjects, ~8 hours total)
# =============================================================================
# Run all TBSS metrics in parallel
METRICS=(FA MD AD RD MK AK RK KFA FICVF ODI FISO)

for metric in "${METRICS[@]}"; do
    launch_analysis \
        "TBSS_${metric}" \
        "$LOG_DIR/tbss_${metric}.log" \
        "uv run python -m neurovrai.analysis.tbss.run_tbss_stats \
            --data-dir $STUDY_ROOT/analysis/tbss/${metric} \
            --design-dir $DESIGN_ROOT/tbss \
            --output-dir $STUDY_ROOT/analysis/tbss/${metric}/stats \
            --n-permutations 5000"
done

# =============================================================================
# 3. Functional ReHo (17 subjects, ~45 min)
# =============================================================================
launch_analysis \
    "ReHo" \
    "$LOG_DIR/reho.log" \
    "uv run python run_func_group_analysis.py \
        --metric reho \
        --derivatives-dir $STUDY_ROOT/derivatives \
        --analysis-dir $STUDY_ROOT/analysis/func/reho \
        --design-dir $DESIGN_ROOT/func_reho \
        --method randomise \
        --n-permutations 5000"

# =============================================================================
# 4. Functional fALFF (17 subjects, ~45 min)
# =============================================================================
launch_analysis \
    "fALFF" \
    "$LOG_DIR/falff.log" \
    "uv run python run_func_group_analysis.py \
        --metric falff \
        --derivatives-dir $STUDY_ROOT/derivatives \
        --analysis-dir $STUDY_ROOT/analysis/func/falff \
        --design-dir $DESIGN_ROOT/func_falff \
        --method randomise \
        --n-permutations 5000"

# =============================================================================
# 5. ASL Analysis (18 subjects, ~45 min)
# =============================================================================
launch_analysis \
    "ASL" \
    "$LOG_DIR/asl.log" \
    "uv run python run_asl_group_analysis.py \
        --study-root $STUDY_ROOT \
        --design-dir $DESIGN_ROOT/asl \
        --method randomise \
        --n-permutations 5000"

# =============================================================================
# Monitor progress
# =============================================================================
echo "=========================================================================="
echo "LAUNCHED ${#PIDS[@]} ANALYSES"
echo "=========================================================================="

for i in "${!PIDS[@]}"; do
    echo "  ${JOBS[$i]}: PID ${PIDS[$i]}"
done

echo ""
echo "Monitoring progress... (Ctrl+C to stop monitoring, jobs will continue)"
echo ""
echo "To monitor individual logs:"
echo "  tail -f $LOG_DIR/<analysis>.log"
echo ""
echo "To check running processes:"
echo "  ps aux | grep -E '(run_vbm|run_tbss|run_func|run_asl)'"
echo ""

# Wait for all jobs to complete
completed=0
total=${#PIDS[@]}

while [ $completed -lt $total ]; do
    completed=0

    for i in "${!PIDS[@]}"; do
        if ! kill -0 ${PIDS[$i]} 2>/dev/null; then
            completed=$((completed + 1))
        fi
    done

    echo -ne "\rProgress: $completed/$total analyses complete"
    sleep 10
done

echo ""
echo ""
echo "=========================================================================="
echo "ALL ANALYSES COMPLETE"
echo "=========================================================================="

# Check exit status of each job
echo ""
echo "Exit status:"
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    status=$?
    if [ $status -eq 0 ]; then
        echo "  ✓ ${JOBS[$i]}: SUCCESS"
    else
        echo "  ✗ ${JOBS[$i]}: FAILED (exit code $status)"
    fi
done

echo ""
echo "Logs saved to: $LOG_DIR"
echo ""
