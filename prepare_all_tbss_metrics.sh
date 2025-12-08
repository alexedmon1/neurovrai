#!/bin/bash
# Prepare all TBSS metrics using the FA skeleton
# Date: 2025-12-07

set -e

CONFIG="/home/edm9fd/sandbox/neurovrai/config.yaml"
FA_SKELETON="/mnt/bytopia/IRC805/analysis/tbss/FA"
OUTPUT_BASE="/mnt/bytopia/IRC805/analysis/tbss"
LOG_DIR="logs/tbss_prep_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

echo "=================================="
echo "TBSS Metric Preparation - Parallel"
echo "=================================="
echo "FA Skeleton: $FA_SKELETON"
echo "Output Base: $OUTPUT_BASE"
echo "Log Directory: $LOG_DIR"
echo "=================================="
echo

# Define all metrics to process
DTI_METRICS="MD AD RD"
DKI_METRICS="MK AK RK KFA"
NODDI_METRICS="FICVF ODI FISO"

ALL_METRICS="$DTI_METRICS $DKI_METRICS $NODDI_METRICS"

# Function to prepare a single metric
prepare_metric() {
    local metric=$1
    local logfile="$LOG_DIR/${metric}_prep.log"

    echo "[$(date +%H:%M:%S)] Starting $metric preparation..."

    uv run python -m neurovrai.analysis.tbss.prepare_tbss \
        --config "$CONFIG" \
        --metric "$metric" \
        --fa-skeleton-dir "$FA_SKELETON" \
        --output-dir "$OUTPUT_BASE" \
        > "$logfile" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ $metric completed"
    else
        echo "[$(date +%H:%M:%S)] ✗ $metric FAILED - see $logfile"
    fi
}

# Export function for parallel execution
export -f prepare_metric
export CONFIG FA_SKELETON OUTPUT_BASE LOG_DIR

# Run DTI metrics in parallel (4 cores)
echo "=== Phase 1: DTI Metrics (MD, AD, RD) ==="
echo "$DTI_METRICS" | tr ' ' '\n' | xargs -P 3 -I {} bash -c 'prepare_metric {}'
echo

# Run DKI metrics in parallel (4 cores)
echo "=== Phase 2: DKI Metrics (MK, AK, RK, KFA) ==="
echo "$DKI_METRICS" | tr ' ' '\n' | xargs -P 4 -I {} bash -c 'prepare_metric {}'
echo

# Run NODDI metrics in parallel (3 cores)
echo "=== Phase 3: NODDI Metrics (FICVF, ODI, FISO) ==="
echo "$NODDI_METRICS" | tr ' ' '\n' | xargs -P 3 -I {} bash -c 'prepare_metric {}'
echo

echo "=================================="
echo "All TBSS Preparations Complete"
echo "=================================="
echo "Logs saved to: $LOG_DIR"
echo

# Check results
echo "=== Verification ==="
for metric in $ALL_METRICS; do
    if [ -f "$OUTPUT_BASE/$metric/stats/all_${metric}_skeletonised.nii.gz" ]; then
        echo "✓ $metric: Skeleton ready"
    else
        echo "✗ $metric: Skeleton NOT found"
    fi
done
