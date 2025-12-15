#!/bin/bash
# Check echo counts for functional subjects

for subject_dir in /mnt/bytopia/IRC805/bids/IRC805-*/func; do
    subject=$(basename $(dirname $subject_dir))
    n_files=$(ls $subject_dir/*.nii.gz 2>/dev/null | wc -l)
    echo "$subject: $n_files files"
    ls $subject_dir/*.nii.gz 2>/dev/null | head -3 | xargs -n1 basename
    echo ""
done | head -50
