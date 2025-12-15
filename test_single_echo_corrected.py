#!/usr/bin/env python3
"""Test corrected single-echo pipeline with proper order: AROMA → ACompCor → Bandpass → Smooth"""

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

if not func_files:
    print(f"No functional files found in {nifti_dir}")
    exit(1)

# Use the first file (single-echo)
func_file = func_files[0]
print(f"Processing: {func_file}")

# Anatomical derivatives
anat_derivatives = study_root / 'derivatives' / subject / 'anat'

# Output directory (study root for standardized hierarchy)
output_dir = study_root / 'derivatives'

# Run preprocessing
results = run_func_preprocessing(
    config=config.get('functional', {}),
    subject=subject,
    func_file=func_file,
    output_dir=output_dir,
    anat_derivatives=anat_derivatives
)

print("\n" + "=" * 70)
print("Preprocessing complete!")
print("=" * 70)
print(f"Preprocessed output: {results['preprocessed']}")
if 'qc_report' in results:
    print(f"QC report: {results['qc_report']}")
