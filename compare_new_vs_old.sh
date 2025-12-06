#!/bin/bash

BACKUP_DIR="/mnt/bytopia/IRC805/data/designs_backup_20251205_180011"
NEW_DIR="/mnt/bytopia/IRC805/data/designs"

echo "========================================================================"
echo "COMPARING NEW VS OLD DESIGN MATRICES"
echo "========================================================================"
echo ""

for modality in vbm asl func_reho func_falff tbss; do
    echo "------------------------------------------------------------------------"
    echo "$modality"
    echo "------------------------------------------------------------------------"
    
    old_summary="$BACKUP_DIR/$modality/design_summary.json"
    new_summary="$NEW_DIR/$modality/design_summary.json"
    
    if [ -f "$old_summary" ] && [ -f "$new_summary" ]; then
        old_n=$(jq -r '.n_subjects' "$old_summary")
        new_n=$(jq -r '.n_subjects' "$new_summary")
        old_cols=$(jq -r '.columns' "$old_summary")
        new_cols=$(jq -r '.columns' "$new_summary")
        old_contrasts=$(jq -r '.contrasts' "$old_summary")
        new_contrasts=$(jq -r '.contrasts' "$new_summary")
        
        echo "  OLD: $old_n subjects"
        echo "  NEW: $new_n subjects"
        
        if [ "$old_n" != "$new_n" ]; then
            echo "  ⚠ DIFFERENCE: Subject count changed"
        else
            echo "  ✓ Subject count: SAME"
        fi
        
        if [ "$old_cols" == "$new_cols" ]; then
            echo "  ✓ Columns: SAME"
        else
            echo "  ⚠ DIFFERENCE: Columns changed"
            echo "     OLD: $old_cols"
            echo "     NEW: $new_cols"
        fi
        
        if [ "$old_contrasts" == "$new_contrasts" ]; then
            echo "  ✓ Contrasts: SAME"
        else
            echo "  ⚠ DIFFERENCE: Contrasts changed"
        fi
    else
        echo "  ⚠ Missing summary file(s)"
    fi
    echo ""
done

echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo ""
echo "Old designs location: $BACKUP_DIR"
echo "New designs location: $NEW_DIR"
echo ""
