#!/usr/bin/env python3
"""Test with verbose output to identify where it hangs"""

from pathlib import Path
import sys
from neurovrai.config import load_config

print("=" * 70, flush=True)
print("VERBOSE TEST - Single Echo Functional Preprocessing", flush=True)
print("=" * 70, flush=True)

# Load configuration
print("\n[1/6] Loading config...", flush=True)
config_file = Path('/mnt/bytopia/IRC805/config.yaml')
config = load_config(config_file)
print("  ✓ Config loaded", flush=True)

# Subject and paths
subject = 'IRC805-2350101'
study_root = Path('/mnt/bytopia/IRC805')

# Find functional file
print("\n[2/6] Finding functional file...", flush=True)
nifti_dir = study_root / 'bids' / subject / 'func'
func_files = sorted(nifti_dir.glob('*RESTING*.nii.gz'))
func_file = func_files[0]
print(f"  ✓ Found: {func_file.name}", flush=True)

# Anatomical derivatives
print("\n[3/6] Setting up directories...", flush=True)
anat_derivatives = study_root / 'derivatives' / subject / 'anat'
output_dir = study_root / 'derivatives'
print(f"  ✓ Output dir: {output_dir}", flush=True)

# Disable QC to speed up
print("\n[4/6] Configuring preprocessing...", flush=True)
# Modify the full config (not just functional section)
if 'functional' not in config:
    config['functional'] = {}
config['functional']['run_qc'] = False  # Disable QC for faster testing
print("  ✓ QC disabled", flush=True)

# Import workflow (after all prints to avoid import delays)
print("\n[5/6] Importing workflow...", flush=True)
from neurovrai.preprocess.workflows.func_preprocess import run_func_preprocessing
print("  ✓ Import complete", flush=True)

# Run preprocessing with detailed output
print("\n[6/6] Starting preprocessing...", flush=True)
print("=" * 70, flush=True)
sys.stdout.flush()

try:
    results = run_func_preprocessing(
        config=config,  # Pass full config (refactored to expect full config like DWI)
        subject=subject,
        func_file=func_file,
        output_dir=output_dir,
        anat_derivatives=anat_derivatives
    )

    print("\n" + "=" * 70, flush=True)
    print("SUCCESS!", flush=True)
    print("=" * 70, flush=True)
    if 'preprocessed' in results:
        print(f"Preprocessed output: {results['preprocessed']}", flush=True)
    else:
        print("Warning: 'preprocessed' not in results", flush=True)
        print(f"Available keys: {list(results.keys())}", flush=True)

except Exception as e:
    print("\n" + "=" * 70, flush=True)
    print("ERROR!", flush=True)
    print("=" * 70, flush=True)
    print(f"Error type: {type(e).__name__}", flush=True)
    print(f"Error message: {e}", flush=True)
    import traceback
    traceback.print_exc()
