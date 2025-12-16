#!/usr/bin/env python3
"""Simplified test with debug output"""

from pathlib import Path
from neurovrai.config import load_config
from neurovrai.preprocess.workflows.func_preprocess import run_func_preprocessing

# Load configuration
config_file = Path('/mnt/bytopia/IRC805/config.yaml')
config = load_config(config_file)

# Subject and paths
subject = 'IRC805-2350101'
study_root = Path('/mnt/bytopia/IRC805')

# Find functional file
nifti_dir = study_root / 'bids' / subject / 'func'
func_files = sorted(nifti_dir.glob('*RESTING*.nii.gz'))
func_file = func_files[0]

# Anatomical derivatives
anat_derivatives = study_root / 'derivatives' / subject / 'anat'

# Output directory
output_dir = study_root / 'derivatives'

print("=" * 70)
print("Starting simplified functional preprocessing test")
print("=" * 70)
print(f"Subject: {subject}")
print(f"Functional file: {func_file}")
print(f"Output dir: {output_dir}")
print("")

# Disable QC to speed up
func_config = config.get('functional', {})
func_config['run_qc'] = False  # Disable QC for faster testing

try:
    print("Calling run_func_preprocessing...")
    results = run_func_preprocessing(
        config=func_config,
        subject=subject,
        func_file=func_file,
        output_dir=output_dir,
        anat_derivatives=anat_derivatives
    )

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    if 'preprocessed' in results:
        print(f"Preprocessed output: {results['preprocessed']}")
    else:
        print("Warning: 'preprocessed' not in results")
        print(f"Available keys: {list(results.keys())}")

except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR!")
    print("=" * 70)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()
