#!/usr/bin/env python3
"""
Example 1: Basic Single-Modality Preprocessing

This example demonstrates how to run preprocessing for individual modalities
using the production workflows.
"""

from pathlib import Path
from mri_preprocess.config import load_config

# ============================================================================
# Example 1a: Anatomical Preprocessing
# ============================================================================

from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing

# Load configuration
config = load_config(Path('config.yaml'))

# Define paths
subject = 'IRC805-0580101'
study_root = Path('/mnt/bytopia/IRC805')
nifti_dir = study_root / 'bids' / subject / 'anat'

# Find T1w file
t1w_files = list(nifti_dir.glob('*T1*.nii.gz'))

if t1w_files:
    results = run_anat_preprocessing(
        config=config,
        subject=subject,
        t1w_file=t1w_files[0],
        output_dir=study_root / 'derivatives',
        work_dir=study_root / 'work' / subject
    )

    print(f"✓ Anatomical preprocessing complete!")
    print(f"  Brain: {results['brain']}")
    print(f"  Brain mask: {results['brain_mask']}")
    print(f"  QC report: {results.get('qc_report')}")


# ============================================================================
# Example 1b: DWI Preprocessing
# ============================================================================

from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

# Find DWI files
dwi_dir = study_root / 'bids' / subject / 'dwi'
dwi_files = sorted(list(dwi_dir.glob('*DTI*.nii.gz')))
bval_files = sorted(list(dwi_dir.glob('*DTI*.bval')))
bvec_files = sorted(list(dwi_dir.glob('*DTI*.bvec')))
rev_phase_files = sorted(list(dwi_dir.glob('*SE_EPI*.nii.gz')))

if dwi_files and bval_files and bvec_files:
    results = run_dwi_multishell_topup_preprocessing(
        config=config,
        subject=subject,
        dwi_files=dwi_files,
        bval_files=bval_files,
        bvec_files=bvec_files,
        rev_phase_files=rev_phase_files if rev_phase_files else None,
        output_dir=study_root / 'derivatives',
        work_dir=study_root / 'work' / subject
    )

    print(f"✓ DWI preprocessing complete!")
    print(f"  Eddy-corrected: {results['eddy_corrected']}")
    print(f"  FA map: {results['fa']}")
    print(f"  QC reports: {study_root}/qc/{subject}/dwi/")


# ============================================================================
# Example 1c: Functional Preprocessing
# ============================================================================

from mri_preprocess.workflows.func_preprocess import run_func_preprocessing

# Find functional files
func_dir = study_root / 'bids' / subject / 'func'
func_files = sorted(list(func_dir.glob('*RESTING*.nii.gz')))
anat_derivatives = study_root / 'derivatives' / subject / 'anat'

if func_files and anat_derivatives.exists():
    results = run_func_preprocessing(
        config=config,
        subject=subject,
        func_file=func_files,  # Can be single file or list of echoes
        output_dir=study_root / 'derivatives',
        anat_derivatives=anat_derivatives,
        work_dir=study_root / 'work' / subject
    )

    print(f"✓ Functional preprocessing complete!")
    print(f"  Preprocessed BOLD: {results.get('preprocessed')}")
    print(f"  QC report: {results.get('qc_report')}")


# ============================================================================
# Example 1d: ASL Preprocessing
# ============================================================================

from mri_preprocess.workflows.asl_preprocess import run_asl_preprocessing

# Find ASL files
asl_dir = study_root / 'bids' / subject / 'asl'
asl_files = list(asl_dir.glob('*ASL*.nii.gz'))
dicom_dir = study_root / 'dicoms' / subject / 'asl'

# Anatomical outputs needed for ASL
t1w_brain = anat_derivatives / 'brain' / 'brain.nii.gz'
gm_mask = anat_derivatives / 'segmentation' / 'POSTERIOR_02.nii.gz'
wm_mask = anat_derivatives / 'segmentation' / 'POSTERIOR_03.nii.gz'

if asl_files and t1w_brain.exists():
    results = run_asl_preprocessing(
        config=config,
        subject=subject,
        asl_file=asl_files[0],
        output_dir=study_root / 'derivatives',
        t1w_brain=t1w_brain,
        gm_mask=gm_mask,
        wm_mask=wm_mask,
        dicom_dir=dicom_dir if dicom_dir.exists() else None,
        normalize_to_mni=True
    )

    print(f"✓ ASL preprocessing complete!")
    print(f"  CBF map: {results.get('cbf')}")
    print(f"  QC report: {results.get('qc_report')}")
