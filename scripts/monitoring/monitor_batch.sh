#!/bin/bash
#
# Monitor Batch Functional Preprocessing
#
# Usage: bash monitor_batch.sh

LOG="/mnt/bytopia/IRC805/logs/batch_func_processing.log"
BATCH_LOG_DIR="/mnt/bytopia/IRC805/logs/batch_func"

echo "========================================================================"
echo "Batch Functional Preprocessing Monitor"
echo "========================================================================"
echo ""

# Check if batch process is running
if ps aux | grep -q "[b]atch_func_only.sh"; then
    echo "✓ Batch processing is RUNNING"
    echo ""

    # Show current subject being processed
    current=$(tail -100 "$LOG" | grep "Processing:" | tail -1)
    if [ -n "$current" ]; then
        echo "Current: $current"
    fi
    echo ""

    # Count completed/failed
    success=$(grep "✓ SUCCESS" "$LOG" 2>/dev/null | wc -l)
    failed=$(grep "✗ FAILED" "$LOG" 2>/dev/null | wc -l)

    echo "Progress:"
    echo "  Successful: $success"
    echo "  Failed: $failed"
    echo ""

    # Show recent output
    echo "Recent output (last 20 lines):"
    echo "------------------------------------------------------------------------"
    tail -20 "$LOG"
    echo "------------------------------------------------------------------------"
else
    echo "✗ Batch processing is NOT running"
    echo ""

    # Show final summary if exists
    if [ -f "$LOG" ]; then
        echo "Final Summary:"
        echo "------------------------------------------------------------------------"
        tail -30 "$LOG"
        echo "------------------------------------------------------------------------"
    fi
fi

echo ""
echo "Full log: $LOG"
echo "Individual logs: $BATCH_LOG_DIR/"
echo ""
echo "Monitor in real-time: tail -f $LOG"
echo "========================================================================"
