#!/bin/bash
# archive_old_func_outputs.sh
#
# Archive OLD functional preprocessing outputs before deletion
# This script moves (not deletes) old files to an archive directory
#
# Usage: ./archive_old_func_outputs.sh <subject_id>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <subject_id>"
    echo "Example: $0 IRC805-2350101"
    exit 1
fi

subject="$1"
func_dir="/mnt/bytopia/IRC805/derivatives/${subject}/func"
archive_dir="/mnt/bytopia/IRC805/archive/old_func_outputs/${subject}"

echo "=========================================="
echo "Archiving OLD functional outputs for $subject"
echo "=========================================="
echo ""

# Check if func directory exists
if [ ! -d "$func_dir" ]; then
    echo "✗ Functional directory not found: $func_dir"
    exit 1
fi

# First, verify NEW pipeline completed successfully
echo "Verifying NEW pipeline completed..."
if ! bash "$(dirname $0)/verify_new_func_output.sh" "$subject"; then
    echo ""
    echo "✗ NEW pipeline output incomplete - ABORTING archive"
    echo "  Complete NEW pipeline processing before archiving OLD outputs"
    exit 1
fi

echo ""
echo "Creating archive directory..."
mkdir -p "$archive_dir/registration"
echo "  ✓ $archive_dir"

# Count files to be archived
file_count=0

# Archive OLD file patterns
echo ""
echo "Archiving OLD files..."

# 1. Old 3D brain/mask files
if [ -f "$func_dir/func_brain.nii.gz" ]; then
    mv "$func_dir/func_brain.nii.gz" "$archive_dir/"
    echo "  ✓ Archived: func_brain.nii.gz"
    ((file_count++))
fi

if [ -f "$func_dir/func_mask.nii.gz" ]; then
    mv "$func_dir/func_mask.nii.gz" "$archive_dir/"
    echo "  ✓ Archived: func_mask.nii.gz"
    ((file_count++))
fi

# 2. Old final output with wrong naming pattern
for file in "$func_dir"/*_mcf_bp_smooth.nii.gz; do
    if [ -f "$file" ]; then
        mv "$file" "$archive_dir/"
        echo "  ✓ Archived: $(basename $file)"
        ((file_count++))
    fi
done

# 3. Old filtered directory
if [ -d "$func_dir/filtered" ]; then
    mv "$func_dir/filtered" "$archive_dir/"
    echo "  ✓ Archived: filtered/ directory"
    ((file_count++))
fi

# 4. OLD registration files (WRONG - computed on filtered data)
if [ -f "$func_dir/registration/transform_list.txt" ]; then
    mv "$func_dir/registration/transform_list.txt" "$archive_dir/registration/"
    echo "  ✓ Archived: registration/transform_list.txt"
    ((file_count++))
fi

if [ -f "$func_dir/registration/func_to_mni_Composite.h5" ]; then
    mv "$func_dir/registration/func_to_mni_Composite.h5" "$archive_dir/registration/"
    echo "  ✓ Archived: registration/func_to_mni_Composite.h5 (INCORRECT - used filtered data)"
    ((file_count++))
fi

echo ""
echo "=========================================="
if [ $file_count -gt 0 ]; then
    echo "✓ Archived $file_count OLD file(s) to: $archive_dir"
else
    echo "⚠ No OLD files found to archive (subject may already be cleaned up)"
fi
echo "=========================================="
echo ""
echo "Archive contents:"
du -sh "$archive_dir"
echo ""
echo "Remaining functional outputs (NEW only):"
ls -lh "$func_dir"
