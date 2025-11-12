#!/usr/bin/env python3
"""
Example script for running multi-shell DWI preprocessing with TOPUP distortion correction.

This script demonstrates how to use the new DWI preprocessing pipeline that:
1. Applies TOPUP correction using reverse phase-encoding images
2. Merges multiple DWI shells (e.g., b1000, b2000, b3000)
3. Runs eddy correction with TOPUP integration
4. Performs DTI fitting and optional BEDPOSTX

Usage:
    python examples/run_dwi_with_topup.py

Requirements:
    - FSL installed and $FSLDIR set
    - CUDA for GPU-accelerated eddy (optional but recommended)
    - TOPUP acquisition parameters file (acqparams.txt)
"""

import sys
from pathlib import Path
from glob import glob
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing
from mri_preprocess.utils.workflow import load_config


def main():
    """Run DWI preprocessing with TOPUP on multiple subjects."""

    # Load configuration
    config_file = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    config = load_config(config_file)

    # Study directory
    study_dir = Path('/mnt/bytopia/IRC805')
    subjects_dir = study_dir / 'subjects'

    # Get subject list
    subject_folders = sorted(glob(str(subjects_dir / '*/nifti')))

    print(f"Found {len(subject_folders)} subjects to process")

    # Process each subject
    for subject_folder in subject_folders:
        subject_path = Path(subject_folder)
        subject_id = subject_path.parent.name
        dwi_folder = subject_path / 'dwi'

        print(f"\n{'='*60}")
        print(f"Processing {subject_id}")
        print(f"{'='*60}")

        # Check if DWI folder exists
        if not dwi_folder.exists():
            print(f"  Skipping {subject_id}: no DWI folder found")
            continue

        # Find DWI files based on sequence names
        # Adjust these patterns based on your acquisition protocol
        try:
            # Shell 1: b1000-b2000
            dwi_b1000_b2000 = list(dwi_folder.glob('*DTI_2shell_b1000_b2000_MB4*.nii.gz'))
            bval_b1000_b2000 = list(dwi_folder.glob('*DTI_2shell_b1000_b2000_MB4*.bval'))
            bvec_b1000_b2000 = list(dwi_folder.glob('*DTI_2shell_b1000_b2000_MB4*.bvec'))
            rev_b1000_b2000 = list(dwi_folder.glob('*SE_EPI*b1000_b2000*.nii.gz'))

            # Shell 2: b3000
            dwi_b3000 = list(dwi_folder.glob('*DTI_1shell_b3000_MB4*.nii.gz'))
            bval_b3000 = list(dwi_folder.glob('*DTI_1shell_b3000_MB4*.bval'))
            bvec_b3000 = list(dwi_folder.glob('*DTI_1shell_b3000_MB4*.bvec'))
            rev_b3000 = list(dwi_folder.glob('*SE_EPI*b3000*.nii.gz'))

            # Verify all files exist
            if not all([
                dwi_b1000_b2000, bval_b1000_b2000, bvec_b1000_b2000, rev_b1000_b2000,
                dwi_b3000, bval_b3000, bvec_b3000, rev_b3000
            ]):
                print(f"  Skipping {subject_id}: missing required files")
                print(f"    Found: {len(dwi_b1000_b2000)} b1000-b2000 DWI, "
                      f"{len(dwi_b3000)} b3000 DWI")
                print(f"    Found: {len(rev_b1000_b2000)} b1000-b2000 reverse, "
                      f"{len(rev_b3000)} b3000 reverse")
                continue

            # Collect files
            dwi_files = [dwi_b1000_b2000[0], dwi_b3000[0]]
            bval_files = [bval_b1000_b2000[0], bval_b3000[0]]
            bvec_files = [bvec_b1000_b2000[0], bvec_b3000[0]]
            rev_phase_files = [rev_b1000_b2000[0], rev_b3000[0]]

            print(f"  Found all required files:")
            print(f"    Shell 1 (b1000-b2000): {dwi_b1000_b2000[0].name}")
            print(f"    Shell 2 (b3000): {dwi_b3000[0].name}")
            print(f"    Reverse PE files: {len(rev_phase_files)}")

            # Set up directories
            output_dir = study_dir / 'derivatives'
            work_dir = study_dir / 'work'

            # Run preprocessing
            print(f"\n  Starting TOPUP preprocessing...")
            results = run_dwi_multishell_topup_preprocessing(
                config=config,
                subject=subject_id,
                dwi_files=dwi_files,
                bval_files=bval_files,
                bvec_files=bvec_files,
                rev_phase_files=rev_phase_files,
                output_dir=output_dir,
                work_dir=work_dir,
                run_bedpostx=False  # Set to True if you want to run BEDPOSTX
            )

            print(f"\n  Preprocessing completed successfully!")
            print(f"  Outputs:")
            print(f"    FA map: {results['fa']}")
            print(f"    MD map: {results['md']}")
            print(f"    Eddy corrected: {results['eddy_corrected']}")

        except Exception as e:
            print(f"  ERROR processing {subject_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("All subjects processed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
