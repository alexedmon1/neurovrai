#!/usr/bin/env python3
"""
Example 2: Complete Multi-Modality Pipeline

This example demonstrates how to run the complete preprocessing pipeline
for a single subject across all modalities with proper dependency management.
"""

import logging
from pathlib import Path
from neurovrai.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config(Path('config.yaml'))

# Define paths
subject = 'IRC805-0580101'
study_root = Path('/mnt/bytopia/IRC805')
derivatives_dir = study_root / 'derivatives'
work_dir = study_root / 'work' / subject

# Create directories
derivatives_dir.mkdir(parents=True, exist_ok=True)
work_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Step 1: Anatomical Preprocessing (REQUIRED - runs first)
# ============================================================================

logger.info("="*70)
logger.info("Step 1: Anatomical Preprocessing")
logger.info("="*70)

from neurovrai.preprocess.workflows.anat_preprocess import run_anat_preprocessing

nifti_dir = study_root / 'bids' / subject / 'anat'
t1w_files = list(nifti_dir.glob('*T1*.nii.gz'))

if not t1w_files:
    logger.error(f"No T1w files found in {nifti_dir}")
    exit(1)

anat_results = run_anat_preprocessing(
    config=config,
    subject=subject,
    t1w_file=t1w_files[0],
    output_dir=derivatives_dir,
    work_dir=work_dir
)

logger.info(f"✓ Anatomical preprocessing complete")
logger.info(f"  Outputs: {derivatives_dir / subject / 'anat'}")
logger.info(f"  QC: {study_root}/qc/{subject}/anat/")
logger.info("")

# Anatomical outputs needed by other modalities
anat_derivatives = derivatives_dir / subject / 'anat'
t1w_brain = anat_results['brain']
gm_mask = list((anat_derivatives / 'segmentation').glob('*POSTERIOR_02.nii.gz'))[0]
wm_mask = list((anat_derivatives / 'segmentation').glob('*POSTERIOR_03.nii.gz'))[0]

# ============================================================================
# Step 2: DWI Preprocessing (independent of other modalities)
# ============================================================================

logger.info("="*70)
logger.info("Step 2: DWI Preprocessing")
logger.info("="*70)

from neurovrai.preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

dwi_dir = study_root / 'bids' / subject / 'dwi'
dwi_files = sorted(list(dwi_dir.glob('*DTI*.nii.gz')))
bval_files = sorted(list(dwi_dir.glob('*DTI*.bval')))
bvec_files = sorted(list(dwi_dir.glob('*DTI*.bvec')))
rev_phase_files = sorted(list(dwi_dir.glob('*SE_EPI*.nii.gz')))

if dwi_files:
    dwi_results = run_dwi_multishell_topup_preprocessing(
        config=config,
        subject=subject,
        dwi_files=dwi_files,
        bval_files=bval_files,
        bvec_files=bvec_files,
        rev_phase_files=rev_phase_files if rev_phase_files else None,
        output_dir=derivatives_dir,
        work_dir=work_dir,
        run_bedpostx=False  # Set to True for tractography
    )

    logger.info(f"✓ DWI preprocessing complete")
    logger.info(f"  FA: {dwi_results['fa']}")
    logger.info(f"  Eddy-corrected: {dwi_results['eddy_corrected']}")
    if 'advanced_models' in dwi_results:
        logger.info(f"  Advanced models: DKI, NODDI complete")
    logger.info(f"  QC: {study_root}/qc/{subject}/dwi/")
    logger.info("")
else:
    logger.warning("No DWI files found, skipping DWI preprocessing")
    logger.info("")

# ============================================================================
# Step 3: Functional Preprocessing (depends on anatomical)
# ============================================================================

logger.info("="*70)
logger.info("Step 3: Functional Preprocessing")
logger.info("="*70)

from neurovrai.preprocess.workflows.func_preprocess import run_func_preprocessing

func_dir = study_root / 'bids' / subject / 'func'
func_files = sorted(list(func_dir.glob('*RESTING*.nii.gz')))

if func_files:
    func_results = run_func_preprocessing(
        config=config,
        subject=subject,
        func_file=func_files,  # Auto-detects multi-echo vs single-echo
        output_dir=derivatives_dir,
        anat_derivatives=anat_derivatives,
        work_dir=work_dir
    )

    logger.info(f"✓ Functional preprocessing complete")
    if 'tedana_denoised' in func_results:
        logger.info(f"  TEDANA denoised: {func_results['tedana_denoised']}")
    logger.info(f"  Preprocessed: {func_results.get('preprocessed')}")
    logger.info(f"  QC: {study_root}/qc/{subject}/func/")
    logger.info("")
else:
    logger.warning("No functional files found, skipping functional preprocessing")
    logger.info("")

# ============================================================================
# Step 4: ASL Preprocessing (depends on anatomical)
# ============================================================================

logger.info("="*70)
logger.info("Step 4: ASL Preprocessing")
logger.info("="*70)

from neurovrai.preprocess.workflows.asl_preprocess import run_asl_preprocessing

asl_dir = study_root / 'bids' / subject / 'asl'
asl_files = list(asl_dir.glob('*ASL*.nii.gz'))
dicom_dir = study_root / 'dicoms' / subject / 'asl'

if asl_files:
    asl_results = run_asl_preprocessing(
        config=config,
        subject=subject,
        asl_file=asl_files[0],
        output_dir=derivatives_dir,
        t1w_brain=t1w_brain,
        gm_mask=gm_mask,
        wm_mask=wm_mask,
        dicom_dir=dicom_dir if dicom_dir.exists() else None,
        normalize_to_mni=True
    )

    logger.info(f"✓ ASL preprocessing complete")
    logger.info(f"  CBF: {asl_results.get('cbf')}")
    logger.info(f"  QC: {study_root}/qc/{subject}/asl/")
    logger.info("")
else:
    logger.warning("No ASL files found, skipping ASL preprocessing")
    logger.info("")

# ============================================================================
# Summary
# ============================================================================

logger.info("="*70)
logger.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
logger.info("="*70)
logger.info(f"Subject: {subject}")
logger.info(f"Derivatives: {derivatives_dir / subject}/")
logger.info(f"QC Reports: {study_root}/qc/{subject}/")
logger.info(f"Work Directory: {work_dir}")
logger.info("="*70)
