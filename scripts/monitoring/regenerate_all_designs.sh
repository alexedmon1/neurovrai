#!/bin/bash
#
# Regenerate all design matrices using modality-aware approach
#
# This script generates design matrices for all modalities using:
# - gludata.csv as the master participants file
# - Auto-detection of subjects from available MRI derivatives
# - Proper subject ID formatting (with leading zeros)
#

set -e

STUDY_ROOT="/mnt/bytopia/IRC805"
PARTICIPANTS="$STUDY_ROOT/data/gludata.csv"
DESIGNS_DIR="$STUDY_ROOT/data/designs"
FORMULA="mriglu+sex+age"

echo "========================================================================"
echo "REGENERATING ALL DESIGN MATRICES"
echo "========================================================================"
echo "Study root: $STUDY_ROOT"
echo "Participants file: $PARTICIPANTS"
echo "Formula: $FORMULA"
echo ""

# Track results
declare -a SUCCESS=()
declare -a FAILED=()

# Function to generate design
generate_design() {
    local modality="$1"
    local metric="$2"
    local output_dir="$3"

    echo ""
    echo "------------------------------------------------------------------------"
    echo "Generating design for: $modality${metric:+ $metric}"
    echo "------------------------------------------------------------------------"

    if [ -n "$metric" ]; then
        # TBSS with metric
        uv run python generate_design_matrices.py \
            --modality "$modality" \
            --metric "$metric" \
            --participants "$PARTICIPANTS" \
            --study-root "$STUDY_ROOT" \
            --output-dir "$output_dir" \
            --formula "$FORMULA"
    else
        # Other modalities
        uv run python generate_design_matrices.py \
            --modality "$modality" \
            --participants "$PARTICIPANTS" \
            --study-root "$STUDY_ROOT" \
            --output-dir "$output_dir" \
            --formula "$FORMULA"
    fi

    if [ $? -eq 0 ]; then
        SUCCESS+=("$modality${metric:+ $metric}")
        echo "✓ SUCCESS: $modality${metric:+ $metric}"
    else
        FAILED+=("$modality${metric:+ $metric}")
        echo "✗ FAILED: $modality${metric:+ $metric}"
    fi
}

# 1. VBM
generate_design "vbm" "" "$DESIGNS_DIR/vbm"

# 2. ASL
generate_design "asl" "" "$DESIGNS_DIR/asl"

# 3. ReHo
generate_design "reho" "" "$DESIGNS_DIR/func_reho"

# 4. fALFF
generate_design "falff" "" "$DESIGNS_DIR/func_falff"

# 5. TBSS - FA only (for now, can add other metrics later)
generate_design "tbss" "FA" "$DESIGNS_DIR/tbss"

# Summary
echo ""
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""
echo "SUCCESS (${#SUCCESS[@]}):"
for item in "${SUCCESS[@]}"; do
    echo "  ✓ $item"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILED (${#FAILED[@]}):"
    for item in "${FAILED[@]}"; do
        echo "  ✗ $item"
    done
fi

echo ""
echo "Design matrices saved to: $DESIGNS_DIR"
echo ""

# List all generated designs
echo "Generated design directories:"
ls -lh "$DESIGNS_DIR"

echo ""
echo "Next steps:"
echo "  1. Review subject_order.txt files in each design directory"
echo "  2. Compare with old designs (if needed)"
echo "  3. Run test analysis to validate"
echo ""
