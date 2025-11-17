#!/usr/bin/env python3
"""
Simple MRI Preprocessing Pipeline

A straightforward, human-readable pipeline that processes MRI data.

Usage:
    # Sequential execution (default)
    python run_simple_pipeline.py --subject IRC805-0580101 --dicom-dir /path/to/dicom --config config.yaml

    # Parallel execution (faster, more resource intensive)
    python run_simple_pipeline.py --subject IRC805-0580101 --dicom-dir /path/to/dicom --config config.yaml --parallel-modalities

Steps:
    1. Convert DICOM to NIfTI (if needed)
    2. Anatomical preprocessing (required first)
    3-5. DWI/Functional/ASL preprocessing (run sequentially by default, or in parallel with --parallel-modalities)

Parallel Execution:
    When --parallel-modalities is enabled, DWI, functional, and ASL workflows run simultaneously
    after anatomical preprocessing completes. This can significantly reduce total pipeline time
    when processing subjects with multiple modalities, but requires sufficient system resources
    (CPU cores, RAM, GPU if using CUDA).
"""

import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import configuration and workflow functions
from neurovrai.config import load_config
from neurovrai.preprocess.utils.dicom_converter import convert_subject_dicoms
from neurovrai.preprocess.workflows.anat_preprocess import run_anat_preprocessing
from neurovrai.preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing
from neurovrai.preprocess.workflows.func_preprocess import run_func_preprocessing
from neurovrai.preprocess.workflows.asl_preprocess import run_asl_preprocessing


# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simple_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def convert_dicom_to_nifti(subject, dicom_dir, output_dir):
    """Convert DICOM files to NIfTI format."""
    logger.info("="*70)
    logger.info("STEP 1: Converting DICOM to NIfTI")
    logger.info("="*70)

    try:
        convert_subject_dicoms(
            subject=subject,
            dicom_dir=dicom_dir,
            output_dir=output_dir / 'bids'
        )
        logger.info("✓ DICOM conversion complete\n")
        return output_dir / 'bids' / subject

    except Exception as e:
        logger.error(f"✗ DICOM conversion failed: {e}\n")
        return None


def preprocess_anatomical(subject, config, nifti_dir, derivatives_dir, work_dir):
    """Run anatomical (T1w) preprocessing."""
    logger.info("="*70)
    logger.info("STEP 2: Anatomical Preprocessing")
    logger.info("="*70)

    # Find T1w file
    anat_dir = nifti_dir / 'anat'
    t1w_files = list(anat_dir.glob('*T1*.nii.gz'))

    if not t1w_files:
        logger.error("✗ No T1w file found\n")
        return None

    t1w_file = t1w_files[0]
    logger.info(f"Input: {t1w_file.name}")

    try:
        results = run_anat_preprocessing(
            config=config,
            subject=subject,
            t1w_file=t1w_file,
            output_dir=derivatives_dir,
            work_dir=work_dir
        )
        logger.info("✓ Anatomical preprocessing complete\n")
        return results

    except Exception as e:
        logger.error(f"✗ Anatomical preprocessing failed: {e}\n")
        return None


def preprocess_dwi(subject, config, nifti_dir, derivatives_dir, work_dir):
    """Run diffusion (DWI) preprocessing."""
    logger.info("="*70)
    logger.info("STEP 3: DWI Preprocessing")
    logger.info("="*70)

    # Find DWI files
    dwi_dir = nifti_dir / 'dwi'
    if not dwi_dir.exists():
        logger.info("⊘ No DWI data found - skipping\n")
        return None

    dwi_files = sorted(list(dwi_dir.glob('*DTI*.nii.gz')))
    if not dwi_files:
        logger.info("⊘ No DWI files found - skipping\n")
        return None

    # Get corresponding bval/bvec files
    bval_files = [f.with_suffix('').with_suffix('.bval') for f in dwi_files]
    bvec_files = [f.with_suffix('').with_suffix('.bvec') for f in dwi_files]

    # Find reverse phase encoding files for TOPUP
    rev_phase_files = list(dwi_dir.glob('*SE_EPI*.nii.gz'))

    logger.info(f"Found {len(dwi_files)} DWI files")
    logger.info(f"Found {len(rev_phase_files)} reverse phase files")

    try:
        results = run_dwi_multishell_topup_preprocessing(
            config=config,
            subject=subject,
            dwi_files=dwi_files,
            bval_files=bval_files,
            bvec_files=bvec_files,
            rev_phase_files=rev_phase_files if rev_phase_files else None,
            output_dir=derivatives_dir,
            work_dir=work_dir
        )
        logger.info("✓ DWI preprocessing complete\n")
        return results

    except Exception as e:
        logger.error(f"✗ DWI preprocessing failed: {e}\n")
        return None


def preprocess_functional(subject, config, nifti_dir, derivatives_dir, work_dir):
    """Run functional (resting-state fMRI) preprocessing."""
    logger.info("="*70)
    logger.info("STEP 4: Functional Preprocessing")
    logger.info("="*70)

    # Find functional files
    func_dir = nifti_dir / 'func'
    if not func_dir.exists():
        logger.info("⊘ No functional data found - skipping\n")
        return None

    func_files = list(func_dir.glob('*RESTING*.nii.gz'))
    if not func_files:
        logger.info("⊘ No functional files found - skipping\n")
        return None

    # Group files by run/series (series number is the prefix before first underscore)
    from collections import defaultdict
    import re
    runs = defaultdict(list)
    for f in func_files:
        # Extract series number (e.g., "401" from "401_WIP_RESTING...")
        series_num = f.name.split('_')[0]
        runs[series_num].append(f)

    logger.info(f"Found {len(func_files)} functional files across {len(runs)} run(s)")

    # Select the latest COMPLETE run
    # A complete multi-echo run should have 3+ echoes (e1, e2, e3, ...)
    # Extract timestamp from filename and select the most recent complete run
    complete_runs = {}
    for series_num, files in runs.items():
        # Check if multi-echo (has _e1, _e2, _e3 pattern)
        echo_nums = [int(re.search(r'_e(\d+)\.nii', f.name).group(1)) for f in files if re.search(r'_e(\d+)\.nii', f.name)]

        if echo_nums:
            # Multi-echo: check if we have consecutive echoes starting from e1
            expected_echoes = list(range(1, max(echo_nums) + 1))
            is_complete = sorted(echo_nums) == expected_echoes
        else:
            # Single-echo: 1 file is complete
            is_complete = len(files) == 1

        # Extract timestamp (format: YYYYMMDDHHMMSS in filename)
        timestamp_match = re.search(r'_(\d{14})_', files[0].name)
        timestamp = timestamp_match.group(1) if timestamp_match else "00000000000000"

        logger.info(f"  Run {series_num}: {len(files)} file(s), timestamp={timestamp}, complete={is_complete}")

        if is_complete:
            complete_runs[series_num] = {'files': files, 'timestamp': timestamp}

    if not complete_runs:
        logger.error("✗ No complete functional runs found - all runs appear incomplete\n")
        return None

    # Select the run with the latest timestamp
    selected_series = max(complete_runs.items(), key=lambda x: x[1]['timestamp'])[0]
    func_files_selected = sorted(complete_runs[selected_series]['files'])

    logger.info(f"Selected latest complete run: {selected_series} ({len(func_files_selected)} file(s))")

    # Check that anatomical preprocessing was done
    anat_dir = derivatives_dir / subject / 'anat'

    # Find brain file (may be in brain/ subdirectory)
    brain_files = list(anat_dir.rglob('*brain.nii.gz'))
    if not brain_files:
        logger.error("✗ Anatomical preprocessing required first\n")
        return None
    t1w_brain = brain_files[0]

    # Detect multi-echo based on selected files
    is_multi_echo = len(func_files_selected) > 1 or 'ME' in func_files_selected[0].name
    logger.info(f"Multi-echo: {is_multi_echo}")

    try:
        results = run_func_preprocessing(
            config=config,
            subject=subject,
            func_file=func_files_selected,  # Pass only selected run's files
            output_dir=derivatives_dir,
            work_dir=work_dir
        )
        logger.info("✓ Functional preprocessing complete\n")
        return results

    except Exception as e:
        logger.error(f"✗ Functional preprocessing failed: {e}\n")
        return None


def preprocess_asl(subject, config, nifti_dir, derivatives_dir, work_dir, dicom_dir):
    """Run ASL (perfusion) preprocessing."""
    logger.info("="*70)
    logger.info("STEP 5: ASL Preprocessing")
    logger.info("="*70)

    # Find ASL files
    asl_dir = nifti_dir / 'asl'
    if not asl_dir.exists():
        logger.info("⊘ No ASL data found - skipping\n")
        return None

    asl_files = list(asl_dir.glob('*pCASL*.nii.gz'))
    if not asl_files:
        logger.info("⊘ No ASL files found - skipping\n")
        return None

    # Use SOURCE file if available
    source_files = [f for f in asl_files if 'SOURCE' in f.name]
    asl_file = source_files[0] if source_files else asl_files[0]

    # Check that anatomical preprocessing was done
    anat_dir = derivatives_dir / subject / 'anat'
    seg_dir = anat_dir / 'segmentation'

    # Find brain file (may be in brain/ subdirectory)
    brain_files = list(anat_dir.rglob('*brain.nii.gz'))
    if not brain_files:
        logger.error("✗ Anatomical preprocessing required first\n")
        return None
    t1w_brain = brain_files[0]

    gm_mask = seg_dir / 'POSTERIOR_02.nii.gz'
    wm_mask = seg_dir / 'POSTERIOR_03.nii.gz'
    csf_mask = seg_dir / 'POSTERIOR_01.nii.gz'

    # Find DICOM directory for parameter extraction
    dicom_asl_dir = None
    if dicom_dir:
        for date_dir in dicom_dir.glob('*'):
            asl_subdirs = list(date_dir.glob('*pCASL*'))
            if asl_subdirs:
                dicom_asl_dir = asl_subdirs[0]
                break

    logger.info(f"Input: {asl_file.name}")
    logger.info(f"DICOM parameters: {'Available' if dicom_asl_dir else 'Using defaults from config'}")

    try:
        results = run_asl_preprocessing(
            config=config,
            subject=subject,
            asl_file=asl_file,
            output_dir=derivatives_dir,
            t1w_brain=t1w_brain,
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            csf_mask=csf_mask,
            dicom_dir=dicom_asl_dir
        )
        logger.info("✓ ASL preprocessing complete\n")
        return results

    except Exception as e:
        logger.error(f"✗ ASL preprocessing failed: {e}\n")
        return None


def run_modalities_parallel(subject, config, nifti_dir, derivatives_dir, work_dir, dicom_dir, skip_flags):
    """
    Run DWI, functional, and ASL preprocessing in parallel.

    Parameters
    ----------
    subject : str
        Subject ID
    config : dict
        Configuration dictionary
    nifti_dir : Path
        NIfTI directory
    derivatives_dir : Path
        Derivatives output directory
    work_dir : Path
        Working directory
    dicom_dir : Path
        DICOM directory (for ASL parameter extraction)
    skip_flags : dict
        Dictionary with 'skip_dwi', 'skip_func', 'skip_asl' flags

    Returns
    -------
    dict
        Results from each modality workflow
    """
    logger.info("="*70)
    logger.info("Running post-anatomical workflows in PARALLEL")
    logger.info("="*70)
    logger.info("")

    # Build list of workflows to run
    workflows = []

    if not skip_flags['skip_dwi']:
        workflows.append(('DWI', preprocess_dwi, (subject, config, nifti_dir, derivatives_dir, work_dir)))

    if not skip_flags['skip_func']:
        workflows.append(('Functional', preprocess_functional, (subject, config, nifti_dir, derivatives_dir, work_dir)))

    if not skip_flags['skip_asl']:
        workflows.append(('ASL', preprocess_asl, (subject, config, nifti_dir, derivatives_dir, work_dir, dicom_dir)))

    if not workflows:
        logger.info("No post-anatomical workflows to run\n")
        return {}

    logger.info(f"Submitting {len(workflows)} workflows for parallel execution:")
    for name, _, _ in workflows:
        logger.info(f"  - {name}")
    logger.info("")

    results = {}

    # Use ThreadPoolExecutor (not ProcessPoolExecutor) since workflows use multiprocessing internally
    with ThreadPoolExecutor(max_workers=len(workflows)) as executor:
        # Submit all workflows
        future_to_workflow = {
            executor.submit(func, *args): name
            for name, func, args in workflows
        }

        # Process results as they complete
        for future in as_completed(future_to_workflow):
            workflow_name = future_to_workflow[future]
            try:
                result = future.result()
                results[workflow_name] = result
                if result:
                    logger.info(f"✓ {workflow_name} workflow completed successfully")
                else:
                    logger.warning(f"⚠ {workflow_name} workflow completed with warnings")
            except Exception as e:
                logger.error(f"✗ {workflow_name} workflow failed: {e}")
                results[workflow_name] = None

    logger.info("")
    logger.info("="*70)
    logger.info(f"Parallel execution complete: {sum(1 for r in results.values() if r)}/{len(workflows)} succeeded")
    logger.info("="*70)
    logger.info("")

    return results


def main():
    """Main pipeline execution."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Simple MRI preprocessing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--subject', required=True, help='Subject ID (e.g., IRC805-0580101)')
    parser.add_argument('--config', type=Path, default=Path('config.yaml'), help='Config file')
    parser.add_argument('--dicom-dir', type=Path, help='DICOM directory (if converting from DICOM)')
    parser.add_argument('--nifti-dir', type=Path, help='NIfTI directory (if already converted)')
    parser.add_argument('--skip-anat', action='store_true', help='Skip anatomical preprocessing')
    parser.add_argument('--skip-dwi', action='store_true', help='Skip DWI preprocessing')
    parser.add_argument('--skip-func', action='store_true', help='Skip functional preprocessing')
    parser.add_argument('--skip-asl', action='store_true', help='Skip ASL preprocessing')
    parser.add_argument('--parallel-modalities', action='store_true',
                        help='Run DWI/functional/ASL in parallel (faster, more resource intensive)')

    args = parser.parse_args()

    # Validate inputs
    if not args.dicom_dir and not args.nifti_dir:
        logger.error("Must provide either --dicom-dir or --nifti-dir")
        sys.exit(1)

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Get study root from config
    study_root = Path(config['project_dir'])
    logger.info(f"Study root: {study_root}")

    # Setup directories
    derivatives_dir = study_root / 'derivatives'
    work_dir = study_root / 'work' / args.subject

    derivatives_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Subject: {args.subject}")
    logger.info(f"Output: {derivatives_dir / args.subject}")
    logger.info("")

    # Step 1: DICOM Conversion (if needed)
    if args.dicom_dir:
        nifti_dir = convert_dicom_to_nifti(args.subject, args.dicom_dir, study_root)
        if not nifti_dir:
            logger.error("Pipeline failed at DICOM conversion")
            sys.exit(1)
    else:
        nifti_dir = args.nifti_dir
        logger.info("Using existing NIfTI files\n")

    # Step 2: Anatomical Preprocessing (required for other modalities)
    if not args.skip_anat:
        anat_results = preprocess_anatomical(args.subject, config, nifti_dir, derivatives_dir, work_dir)
        if not anat_results:
            logger.error("Pipeline failed at anatomical preprocessing")
            sys.exit(1)
    else:
        logger.info("Skipping anatomical preprocessing\n")

    # Steps 3-5: Post-anatomical workflows (DWI, Functional, ASL)
    if args.parallel_modalities:
        # Run DWI/functional/ASL in parallel
        skip_flags = {
            'skip_dwi': args.skip_dwi,
            'skip_func': args.skip_func,
            'skip_asl': args.skip_asl
        }
        run_modalities_parallel(args.subject, config, nifti_dir, derivatives_dir, work_dir, args.dicom_dir, skip_flags)
    else:
        # Run sequentially (default)
        # Step 3: DWI Preprocessing
        if not args.skip_dwi:
            preprocess_dwi(args.subject, config, nifti_dir, derivatives_dir, work_dir)

        # Step 4: Functional Preprocessing
        if not args.skip_func:
            preprocess_functional(args.subject, config, nifti_dir, derivatives_dir, work_dir)

        # Step 5: ASL Preprocessing
        if not args.skip_asl:
            preprocess_asl(args.subject, config, nifti_dir, derivatives_dir, work_dir, args.dicom_dir)

    # Summary
    logger.info("="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Results: {derivatives_dir / args.subject}")
    logger.info(f"Working files: {work_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
