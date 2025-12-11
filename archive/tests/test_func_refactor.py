#!/usr/bin/env python3
"""Test script for refactored functional preprocessing workflow."""

from pathlib import Path
from neurovrai.config import load_config
from neurovrai.preprocess.workflows.func_preprocess import run_func_preprocessing

# Configuration - Use study folder config with correct TEDANA settings
config = load_config(Path('/mnt/bytopia/IRC805/config.yaml'))
subject = 'IRC805-0580101'
study_root = Path('/mnt/bytopia/IRC805')

# Functional files (multi-echo resting state)
func_files = [
    study_root / 'bids' / subject / 'func' / '501_WIP_RESTING_ME3_MB3_SENSE3_20220301134414_e1.nii.gz',
    study_root / 'bids' / subject / 'func' / '501_WIP_RESTING_ME3_MB3_SENSE3_20220301134414_e2.nii.gz',
    study_root / 'bids' / subject / 'func' / '501_WIP_RESTING_ME3_MB3_SENSE3_20220301134414_e3.nii.gz'
]

# Anatomical derivatives (for tissue masks)
anat_derivatives = study_root / 'derivatives' / subject / 'anat'

print("=" * 80)
print("TESTING REFACTORED FUNCTIONAL PREPROCESSING WORKFLOW")
print("=" * 80)
print(f"Subject: {subject}")
print(f"Functional files: {len(func_files)} echoes")
print(f"Study root: {study_root}")
print("")

# Run preprocessing
results = run_func_preprocessing(
    config=config,
    subject=subject,
    func_file=func_files,
    output_dir=study_root / 'derivatives',
    anat_derivatives=anat_derivatives
)

print("")
print("=" * 80)
print("PREPROCESSING COMPLETE!")
print("=" * 80)
print("Results:")
for key, value in results.items():
    print(f"  {key}: {value}")
