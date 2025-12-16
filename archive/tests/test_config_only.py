#!/usr/bin/env python3
"""Test config loading only"""
import sys
from pathlib import Path
from neurovrai.config import load_config

print("1. Loading config...", flush=True)
config_file = Path('/mnt/bytopia/IRC805/config.yaml')
config = load_config(config_file)
print("   ✓ Config loaded", flush=True)

print("2. Extracting functional config...", flush=True)
func_config = config.get('functional', {})
print(f"   ✓ Found {len(func_config)} functional config keys", flush=True)

print("3. Finding functional file...", flush=True)
subject = 'IRC805-2350101'
study_root = Path('/mnt/bytopia/IRC805')
nifti_dir = study_root / 'bids' / subject / 'func'
func_files = sorted(nifti_dir.glob('*RESTING*.nii.gz'))
print(f"   ✓ Found {len(func_files)} functional files", flush=True)

print("4. Setting up paths...", flush=True)
anat_derivatives = study_root / 'derivatives' / subject / 'anat'
output_dir = study_root / 'derivatives'
print(f"   ✓ Output dir: {output_dir}", flush=True)
print(f"   ✓ Anat derivatives exists: {anat_derivatives.exists()}", flush=True)

print("\n✓ All setup successful - ready to call run_func_preprocessing()", flush=True)
