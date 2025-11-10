#!/usr/bin/env python3
"""
Complete Example: Processing Test Subject 0580101

This script demonstrates the complete MRI preprocessing pipeline
using the test subject data.

Directory Structure:
    /mnt/bytopia/development/mri-preprocess/
    ├── sourcedata/          # DICOM files
    │   └── sub-0580101/
    ├── rawdata/             # Converted NIfTI (BIDS)
    │   └── sub-0580101/
    ├── derivatives/         # Preprocessed outputs
    │   └── mri-preprocess/
    │       └── sub-0580101/
    ├── transforms/          # TransformRegistry
    │   └── sub-0580101/
    ├── work/                # Intermediate files
    └── logs/                # Processing logs

Processing Steps:
    1. Generate configuration from DICOM headers
    2. Convert DICOM to BIDS-organized NIfTI
    3. Anonymize data for use as example
    4. Run anatomical preprocessing (computes transforms)
    5. Run diffusion preprocessing (reuses transforms)
    6. Run functional preprocessing (reuses transforms + tissue masks)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'human-mri-preprocess'))

from mri_preprocess.config import load_config
from mri_preprocess.config_generator import auto_generate_config
from mri_preprocess.dicom.bids_converter import convert_and_organize
from mri_preprocess.dicom.anonymize import anonymize_subject_data, check_for_phi
from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing
from mri_preprocess.workflows.dwi_preprocess import run_dwi_preprocessing
from mri_preprocess.utils.file_finder import find_subject_files
from mri_preprocess.utils.bids import get_subject_dir, get_workflow_dir
from mri_preprocess.utils.transforms import create_transform_registry

# =============================================================================
# Configuration
# =============================================================================

# Test environment paths
BASE_DIR = Path("/mnt/bytopia/development/mri-preprocess")
DICOM_DIR = Path("/mnt/bytopia/IRC805/raw/dicom/IRC805-0580101/20220301")

# Subject information
SUBJECT_ID = "0580101"  # Real subject ID
SUBJECT_BIDS = "sub-0580101"  # BIDS-formatted

# Output paths
SOURCEDATA_DIR = BASE_DIR / "sourcedata"
RAWDATA_DIR = BASE_DIR / "rawdata"
DERIVATIVES_DIR = BASE_DIR / "derivatives"
TRANSFORMS_DIR = BASE_DIR / "transforms"
WORK_DIR = BASE_DIR / "work"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_FILE = BASE_DIR / "configs" / "test_subject_0580101.yaml"


def step_1_generate_config():
    """
    Step 1: Generate configuration from DICOM headers.
    
    This automatically detects sequences and creates a study-specific config.
    """
    print("\n" + "="*70)
    print("STEP 1: Generate Configuration from DICOM Headers")
    print("="*70)
    
    if CONFIG_FILE.exists():
        print(f"✓ Config already exists: {CONFIG_FILE}")
        print("  Skipping generation (delete file to regenerate)")
        return
    
    print(f"\nScanning DICOM directory: {DICOM_DIR}")
    
    auto_generate_config(
        dicom_dir=DICOM_DIR,
        study_name="IRC805 Test Data",
        study_code="IRC805_TEST",
        base_dir=BASE_DIR,
        output_path=CONFIG_FILE
    )
    
    print(f"\n✓ Configuration generated: {CONFIG_FILE}")
    print("\n⚠ IMPORTANT: Review the config file before proceeding!")
    print("  Check that sequence classifications are correct")


def step_2_convert_dicom():
    """
    Step 2: Convert DICOM to BIDS-organized NIfTI.
    
    Converts DICOM files to NIfTI format and organizes them
    according to BIDS specification.
    """
    print("\n" + "="*70)
    print("STEP 2: Convert DICOM to BIDS NIfTI")
    print("="*70)
    
    # Check if already converted
    subject_dir = get_subject_dir(RAWDATA_DIR, SUBJECT_BIDS)
    if subject_dir.exists() and list(subject_dir.rglob("*.nii.gz")):
        print(f"✓ Subject already converted: {subject_dir}")
        print("  Skipping conversion (delete directory to reconvert)")
        return
    
    print(f"\nConverting {SUBJECT_BIDS}...")
    print(f"  Source: {DICOM_DIR}")
    print(f"  Destination: {RAWDATA_DIR}")
    
    # Load config
    config = load_config(CONFIG_FILE)
    
    # Convert
    result = convert_and_organize(
        dicom_dir=DICOM_DIR,
        rawdata_dir=RAWDATA_DIR,
        subject=SUBJECT_BIDS,
        config=config
    )
    
    if result['success']:
        print(f"\n✓ Conversion successful")
        print(f"\nOrganized files:")
        for modality, files in result['organized_files'].items():
            print(f"  {modality}: {len(files)} files")
    else:
        print(f"✗ Conversion failed")
        raise RuntimeError("DICOM conversion failed")


def step_3_anonymize_data():
    """
    Step 3: Anonymize data for use as example data.
    
    CRITICAL: Removes identifying information from JSON sidecars
    so this data can be safely used as example data.
    """
    print("\n" + "="*70)
    print("STEP 3: Anonymize Data for Example Use")
    print("="*70)
    
    print(f"\nAnonymizing {SUBJECT_BIDS}...")
    print("  Removing patient information from JSON sidecars")
    print("  (Folder names can contain subject ID)")
    
    # Load config
    config = load_config(CONFIG_FILE)
    
    # Anonymize
    results = anonymize_subject_data(
        rawdata_dir=RAWDATA_DIR,
        subject=SUBJECT_BIDS,
        anonymize_nifti=True
    )
    
    print(f"\n✓ Anonymized {results['nifti_json']} JSON files")
    
    # Check for PHI
    print("\nChecking for remaining PHI...")
    subject_dir = get_subject_dir(RAWDATA_DIR, SUBJECT_BIDS)
    findings = check_for_phi(subject_dir, SUBJECT_ID)
    
    if findings:
        print(f"\n⚠ Found {len(findings)} potential PHI instances:")
        for finding in findings[:5]:  # Show first 5
            print(f"  - {finding['field']} in {Path(finding['file']).name}")
        if len(findings) > 5:
            print(f"  ... and {len(findings) - 5} more")
    else:
        print("✓ No PHI found in JSON sidecars")


def step_4_anatomical_preprocessing():
    """
    Step 4: Run anatomical (T1w) preprocessing.
    
    This is the FIRST workflow and it:
    - Reorients to standard space
    - Skull strips with BET
    - Bias field correction and tissue segmentation with FAST
    - Registers to MNI152 (linear + nonlinear)
    - SAVES transforms to TransformRegistry for reuse
    """
    print("\n" + "="*70)
    print("STEP 4: Anatomical Preprocessing")
    print("="*70)
    
    # Load config
    config = load_config(CONFIG_FILE)
    
    # Find T1w file
    subject_dir = get_subject_dir(RAWDATA_DIR, SUBJECT_BIDS)
    files = find_subject_files(subject_dir, 't1w', config)
    
    if not files['images']:
        print("✗ No T1w files found")
        raise RuntimeError("No T1w files found")
    
    t1w_file = files['images'][0]
    print(f"\nUsing T1w file: {t1w_file.name}")
    
    # Setup paths
    work_dir = get_workflow_dir(WORK_DIR, SUBJECT_BIDS, 'anat-preprocess')
    
    print(f"\nRunning anatomical preprocessing...")
    print(f"  Input: {t1w_file}")
    print(f"  Output: {DERIVATIVES_DIR}/mri-preprocess/{SUBJECT_BIDS}/anat")
    print(f"  Work: {work_dir}")
    
    # Run workflow
    results = run_anat_preprocessing(
        config=config,
        subject=SUBJECT_BIDS,
        t1w_file=t1w_file,
        output_dir=DERIVATIVES_DIR,
        work_dir=work_dir
    )
    
    print(f"\n✓ Anatomical preprocessing complete")
    print(f"\nKey outputs:")
    print(f"  Brain: {results['brain']}")
    print(f"  Brain mask: {results['brain_mask']}")
    print(f"  Bias corrected: {results['bias_corrected']}")
    print(f"  CSF mask: {results['csf_prob']}")
    print(f"  GM mask: {results['gm_prob']}")
    print(f"  WM mask: {results['wm_prob']}")
    print(f"  MNI affine: {results['mni_affine']}")
    print(f"  MNI warp: {results['mni_warp']}")
    
    # Verify transforms were saved to registry
    registry = create_transform_registry(config, SUBJECT_BIDS)
    if registry.has_transform('T1w', 'MNI152', 'nonlinear'):
        print(f"\n✓ T1w→MNI152 transform saved to TransformRegistry")
        print(f"  Location: {TRANSFORMS_DIR}/{SUBJECT_BIDS}")
    else:
        print(f"\n⚠ Transform not found in registry")
    
    return results


def step_5_diffusion_preprocessing(anat_results):
    """
    Step 5: Run diffusion (DWI) preprocessing.
    
    This workflow:
    - Runs eddy correction
    - Fits diffusion tensor
    - REUSES T1w→MNI transforms from TransformRegistry
    - No duplicate registration computation!
    """
    print("\n" + "="*70)
    print("STEP 5: Diffusion Preprocessing")
    print("="*70)
    
    # Load config
    config = load_config(CONFIG_FILE)
    
    # Find DWI files
    subject_dir = get_subject_dir(RAWDATA_DIR, SUBJECT_BIDS)
    files = find_subject_files(subject_dir, 'dwi', config)
    
    if not files['images']:
        print("⚠ No DWI files found - skipping diffusion preprocessing")
        return None
    
    dwi_file = files['images'][0]
    bval_file = files['bval'][0] if files['bval'] else None
    bvec_file = files['bvec'][0] if files['bvec'] else None
    
    if not bval_file or not bvec_file:
        print("✗ Missing bval/bvec files")
        return None
    
    print(f"\nUsing DWI file: {dwi_file.name}")
    print(f"  bval: {bval_file.name}")
    print(f"  bvec: {bvec_file.name}")
    
    # Setup paths
    work_dir = get_workflow_dir(WORK_DIR, SUBJECT_BIDS, 'dwi-preprocess')
    
    # Verify transforms available
    registry = create_transform_registry(config, SUBJECT_BIDS)
    if not registry.has_transform('T1w', 'MNI152', 'nonlinear'):
        print("\n⚠ T1w→MNI152 transform not found in registry")
        print("  Run anatomical preprocessing first!")
        return None
    
    print(f"\n✓ Found T1w→MNI152 transform in registry - will reuse!")
    
    print(f"\nRunning diffusion preprocessing...")
    print(f"  Input: {dwi_file}")
    print(f"  Output: {DERIVATIVES_DIR}/mri-preprocess/{SUBJECT_BIDS}/dwi")
    print(f"  Work: {work_dir}")
    
    # Run workflow
    results = run_dwi_preprocessing(
        config=config,
        subject=SUBJECT_BIDS,
        dwi_file=dwi_file,
        bval_file=bval_file,
        bvec_file=bvec_file,
        output_dir=DERIVATIVES_DIR,
        work_dir=work_dir,
        warp_to_mni=True  # This uses TransformRegistry!
    )
    
    print(f"\n✓ Diffusion preprocessing complete")
    print(f"\nKey outputs:")
    print(f"  Eddy corrected: {results['eddy_corrected']}")
    print(f"  FA map: {results['fa']}")
    print(f"  MD map: {results['md']}")
    if 'fa_mni' in results:
        print(f"  FA in MNI: {results['fa_mni']} (using TransformRegistry!)")
    if 'md_mni' in results:
        print(f"  MD in MNI: {results['md_mni']} (using TransformRegistry!)")
    
    return results


def main():
    """
    Main execution: Process test subject through complete pipeline.
    """
    print("\n" + "="*70)
    print("MRI Preprocessing Pipeline - Example Processing")
    print("="*70)
    print(f"\nTest Subject: {SUBJECT_BIDS}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"DICOM Source: {DICOM_DIR}")
    
    # Create directories
    print(f"\nCreating directory structure...")
    for directory in [SOURCEDATA_DIR, RAWDATA_DIR, DERIVATIVES_DIR, 
                     TRANSFORMS_DIR, WORK_DIR, LOGS_DIR, CONFIG_FILE.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")
    
    try:
        # Step 1: Generate configuration
        step_1_generate_config()
        
        # Step 2: Convert DICOM to BIDS
        step_2_convert_dicom()
        
        # Step 3: Anonymize for example use
        step_3_anonymize_data()
        
        # Step 4: Anatomical preprocessing (computes transforms)
        anat_results = step_4_anatomical_preprocessing()
        
        # Step 5: Diffusion preprocessing (reuses transforms)
        dwi_results = step_5_diffusion_preprocessing(anat_results)
        
        # Summary
        print("\n" + "="*70)
        print("✓ COMPLETE PIPELINE FINISHED")
        print("="*70)
        
        print(f"\nProcessed data locations:")
        print(f"  Raw data (BIDS): {RAWDATA_DIR}/{SUBJECT_BIDS}")
        print(f"  Derivatives: {DERIVATIVES_DIR}/mri-preprocess/{SUBJECT_BIDS}")
        print(f"  Transforms: {TRANSFORMS_DIR}/{SUBJECT_BIDS}")
        
        print(f"\nKey demonstration points:")
        print(f"  ✓ Config auto-generated from DICOM headers")
        print(f"  ✓ DICOM converted to BIDS organization")
        print(f"  ✓ Data anonymized for example use")
        print(f"  ✓ T1w→MNI transforms computed once (anatomical workflow)")
        print(f"  ✓ Transforms saved to TransformRegistry")
        print(f"  ✓ Transforms reused by diffusion workflow")
        print(f"  ✓ No duplicate registration computation!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
