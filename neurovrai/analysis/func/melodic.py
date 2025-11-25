#!/usr/bin/env python3
"""
MELODIC Group ICA Analysis for Resting-State fMRI

Performs group-level Independent Component Analysis (ICA) using FSL MELODIC.
Identifies spatial patterns of functional connectivity across multiple subjects.

Features:
- Automatic subject data collection and validation
- Temporal concatenation or tensor ICA
- Automated dimensionality estimation or fixed dimensions
- Optional dual regression for subject-specific networks
- QC reports with component visualizations

Usage:
    python melodic.py --subjects-file subjects.txt --output-dir /study/analysis/melodic

Or from Python:
    from neurovrai.analysis.func.melodic import run_melodic_group_ica

    results = run_melodic_group_ica(
        subject_files=['sub-001/func/preprocessed.nii.gz', ...],
        output_dir=Path('/study/analysis/melodic'),
        mask_file=Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'),
        tr=1.029,
        n_components=20  # or 'auto' for automatic estimation
    )
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Union
import shutil

import nibabel as nib
from nipype.interfaces import fsl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_subject_data(
    subject_files: List[Path],
    mask_file: Optional[Path] = None
) -> Dict:
    """
    Validate that all subject files exist and have compatible dimensions

    Args:
        subject_files: List of preprocessed 4D functional images
        mask_file: Optional brain mask (will check dimensions match)

    Returns:
        Dictionary with validation results
    """
    logger.info("Validating subject data...")

    results = {
        'n_subjects': len(subject_files),
        'valid_files': [],
        'invalid_files': [],
        'dimensions': None,
        'tr': None
    }

    reference_shape = None
    reference_tr = None

    for subj_file in subject_files:
        if not subj_file.exists():
            logger.warning(f"File not found: {subj_file}")
            results['invalid_files'].append(str(subj_file))
            continue

        try:
            img = nib.load(subj_file)
            shape = img.shape

            # Check it's 4D
            if len(shape) != 4:
                logger.warning(f"Not a 4D image: {subj_file}")
                results['invalid_files'].append(str(subj_file))
                continue

            # Get TR from header
            tr = img.header.get_zooms()[3]

            # Check dimensions match across subjects
            if reference_shape is None:
                reference_shape = shape[:3]  # Spatial dimensions only
                reference_tr = tr
                results['dimensions'] = shape
                results['tr'] = float(tr)
            else:
                if shape[:3] != reference_shape:
                    logger.warning(f"Dimension mismatch: {subj_file} ({shape[:3]} vs {reference_shape})")
                    results['invalid_files'].append(str(subj_file))
                    continue

                # Allow TR variations up to 50ms (typical scanner timing precision)
                if abs(tr - reference_tr) > 0.05:
                    logger.warning(f"TR mismatch: {subj_file} ({tr} vs {reference_tr})")
                    results['invalid_files'].append(str(subj_file))
                    continue
                elif abs(tr - reference_tr) > 0.001:
                    logger.info(f"Minor TR variation: {subj_file} ({tr} vs {reference_tr}) - within tolerance")

            results['valid_files'].append(str(subj_file))

        except Exception as e:
            logger.warning(f"Error reading {subj_file}: {e}")
            results['invalid_files'].append(str(subj_file))

    # Validate mask if provided
    if mask_file and mask_file.exists():
        try:
            mask_img = nib.load(mask_file)
            if mask_img.shape != reference_shape:
                logger.warning(f"Mask dimensions {mask_img.shape} don't match data {reference_shape}")
                results['mask_valid'] = False
            else:
                results['mask_valid'] = True
        except Exception as e:
            logger.warning(f"Error reading mask: {e}")
            results['mask_valid'] = False

    logger.info(f"Validation complete: {len(results['valid_files'])}/{len(subject_files)} files valid")

    return results


def run_melodic_group_ica(
    subject_files: List[Path],
    output_dir: Path,
    mask_file: Optional[Path] = None,
    tr: Optional[float] = None,
    n_components: Union[int, str] = 'auto',
    approach: str = 'concat',
    bg_image: Optional[Path] = None,
    validate_inputs: bool = True,
    sep_vn: bool = True,
    mm_thresh: float = 0.5,
    report: bool = True
) -> Dict:
    """
    Run MELODIC group ICA analysis

    Args:
        subject_files: List of preprocessed 4D functional images (should be in MNI space)
        output_dir: Output directory for MELODIC results
        mask_file: Brain mask (default: MNI152 2mm mask)
        tr: Repetition time in seconds (will read from header if not provided)
        n_components: Number of ICA components
                     - int: Fixed number of components
                     - 'auto': Automatic estimation by MELODIC
        approach: ICA approach
                 - 'concat': Temporal concatenation (default, fastest)
                 - 'tica': Tensor ICA (more complex, slower)
        bg_image: Background image for reporting (default: MNI152 2mm brain)
        validate_inputs: Whether to validate subject data before running
        sep_vn: Separate variance normalization for each subject (recommended)
        mm_thresh: Mixture model threshold (0-1, default 0.5)
        report: Generate HTML report

    Returns:
        Dictionary with analysis results and file paths

    Outputs:
        {output_dir}/
            melodic_IC.nii.gz - Independent component spatial maps (4D)
            melodic_mix - Time courses matrix
            melodic_FTmix - Frequency domain time courses
            stats/ - Statistical images and thresholds
            report/ - HTML report and visualizations
            log.txt - MELODIC log file
            melodic_list.txt - List of input files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"melodic_ica_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("MELODIC GROUP ICA ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Number of subjects: {len(subject_files)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Components: {n_components}")
    logger.info(f"Approach: {approach}")
    logger.info("")

    # Validate inputs
    if validate_inputs:
        validation = validate_subject_data(subject_files, mask_file)

        if len(validation['valid_files']) < 2:
            raise ValueError(f"Need at least 2 valid subjects, found {len(validation['valid_files'])}")

        # Use only valid files
        subject_files = [Path(f) for f in validation['valid_files']]

        # Use TR from data if not provided
        if tr is None:
            tr = validation['tr']
            logger.info(f"Using TR from data: {tr:.4f} seconds")

    # Set defaults for FSL standard files
    if mask_file is None:
        mask_file = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
        logger.info(f"Using default mask: {mask_file}")

    if bg_image is None:
        bg_image = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz')
        logger.info(f"Using default background: {bg_image}")

    # Create subject list file
    subject_list_file = output_dir / 'melodic_list.txt'
    with open(subject_list_file, 'w') as f:
        for subj_file in subject_files:
            f.write(f'{subj_file}\n')

    logger.info(f"Created subject list: {subject_list_file}")

    # Setup MELODIC interface
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING MELODIC")
    logger.info("=" * 80)

    melodic = fsl.MELODIC()
    melodic.inputs.in_files = [str(f) for f in subject_files]
    melodic.inputs.out_dir = str(output_dir)
    melodic.inputs.approach = approach
    melodic.inputs.no_bet = True  # Data should already be skull-stripped
    melodic.inputs.tr_sec = float(tr)

    # Only set mask and bg_image if data is already in standard space
    # For native space data, MELODIC will register internally
    # Check first subject to see if it's in MNI space (2mm isotropic)
    first_img = nib.load(subject_files[0])
    voxel_sizes = first_img.header.get_zooms()[:3]
    is_mni = all(abs(vox - 2.0) < 0.1 for vox in voxel_sizes)  # MNI152 2mm

    if is_mni:
        logger.info("Data appears to be in MNI152 2mm space")
        logger.info("Using standard space mask and background image")
        melodic.inputs.mask = str(mask_file)
        melodic.inputs.bg_image = str(bg_image)
    else:
        logger.info("Data appears to be in native space")
        logger.info("MELODIC will perform registration internally")
        logger.info("This may take longer but handles native space data correctly")
        # Don't set mask or bg_image - let MELODIC handle registration

    melodic.inputs.report = report
    melodic.inputs.sep_vn = sep_vn
    melodic.inputs.mm_thresh = mm_thresh
    melodic.inputs.out_all = True  # Save all outputs
    melodic.inputs.output_type = 'NIFTI_GZ'

    # Set dimensionality
    if n_components == 'auto':
        # Let MELODIC estimate automatically
        logger.info("Using automatic dimensionality estimation")
    else:
        melodic.inputs.dim = int(n_components)
        logger.info(f"Using fixed dimensionality: {n_components}")

    # Additional arguments for better results
    melodic.inputs.args = '--verbose'

    logger.info("\nMELODIC Parameters:")
    logger.info(f"  Approach: {approach}")
    logger.info(f"  TR: {tr} seconds")
    logger.info(f"  Components: {n_components}")
    logger.info(f"  Separate variance normalization: {sep_vn}")
    logger.info(f"  Mixture model threshold: {mm_thresh}")
    logger.info(f"  Subjects: {len(subject_files)}")
    logger.info("")

    # Run MELODIC
    try:
        logger.info("Starting MELODIC (this may take 30-60 minutes)...")
        result = melodic.run()
        logger.info("✓ MELODIC completed successfully")
    except Exception as e:
        logger.error(f"MELODIC failed: {str(e)}", exc_info=True)
        raise

    # Parse results
    results = {
        'n_subjects': len(subject_files),
        'n_components': n_components,
        'approach': approach,
        'tr': tr,
        'output_dir': str(output_dir),
        'timestamp': timestamp,
        'subject_list': str(subject_list_file),
        'outputs': {}
    }

    # Check for expected outputs
    expected_outputs = {
        'component_maps': output_dir / 'melodic_IC.nii.gz',
        'mixing_matrix': output_dir / 'melodic_mix',
        'ft_mixing_matrix': output_dir / 'melodic_FTmix',
        'log': output_dir / 'log.txt',
        'report_dir': output_dir / 'report',
        'stats_dir': output_dir / 'stats'
    }

    for name, path in expected_outputs.items():
        if path.exists():
            results['outputs'][name] = str(path)
            logger.info(f"✓ Created: {path}")
        else:
            logger.warning(f"✗ Missing: {path}")

    # Save results summary
    summary_file = output_dir / 'melodic_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("MELODIC ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Component maps: {results['outputs'].get('component_maps', 'N/A')}")
    logger.info(f"Summary: {summary_file}")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 80 + "\n")

    return results


def prepare_subjects_for_melodic(
    derivatives_dir: Path,
    subjects: List[str],
    output_file: Optional[Path] = None
) -> List[Path]:
    """
    Collect preprocessed functional data from multiple subjects

    Args:
        derivatives_dir: Base derivatives directory
        subjects: List of subject IDs
        output_file: Optional file to write subject list to

    Returns:
        List of paths to preprocessed functional images

    Example:
        subjects = ['IRC805-0580101', 'IRC805-0590101', ...]
        func_files = prepare_subjects_for_melodic(
            Path('/study/derivatives'),
            subjects
        )
    """
    logger.info(f"Collecting preprocessed data for {len(subjects)} subjects...")

    func_files = []
    missing = []

    for subject in subjects:
        # Look for preprocessed functional data in standardized location
        # Try several possible filenames
        possible_files = [
            derivatives_dir / subject / 'func' / 'preprocessed_bold.nii.gz',
            derivatives_dir / subject / 'func' / 'preprocessed_bold_mni.nii.gz',
            derivatives_dir / subject / 'func' / 'bold_preprocessed_mni.nii.gz',
        ]

        found = False
        for func_file in possible_files:
            if func_file.exists():
                func_files.append(func_file)
                logger.info(f"✓ {subject}: {func_file.name}")
                found = True
                break

        if not found:
            logger.warning(f"✗ {subject}: No preprocessed data found")
            missing.append(subject)

    logger.info(f"\nFound data for {len(func_files)}/{len(subjects)} subjects")

    if missing:
        logger.warning(f"Missing data for {len(missing)} subjects:")
        for subj in missing[:10]:  # Show first 10
            logger.warning(f"  - {subj}")
        if len(missing) > 10:
            logger.warning(f"  ... and {len(missing) - 10} more")

    # Write subject list if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for func_file in func_files:
                f.write(f'{func_file}\n')
        logger.info(f"\nWrote subject list to: {output_file}")

    return func_files


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MELODIC group ICA analysis on resting-state fMRI data"
    )
    parser.add_argument(
        '--subjects-file',
        type=Path,
        help='Text file with list of preprocessed functional images (one per line)'
    )
    parser.add_argument(
        '--derivatives-dir',
        type=Path,
        help='Derivatives directory (will search for preprocessed data)'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        help='List of subject IDs (use with --derivatives-dir)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for MELODIC results'
    )
    parser.add_argument(
        '--mask',
        type=Path,
        help='Brain mask (default: MNI152 2mm mask)'
    )
    parser.add_argument(
        '--tr',
        type=float,
        help='Repetition time in seconds (will read from header if not provided)'
    )
    parser.add_argument(
        '--n-components',
        default='auto',
        help='Number of ICA components (default: auto)'
    )
    parser.add_argument(
        '--approach',
        choices=['concat', 'tica'],
        default='concat',
        help='ICA approach (default: concat)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip input validation'
    )

    args = parser.parse_args()

    # Collect subject files
    if args.subjects_file:
        # Read from file
        with open(args.subjects_file) as f:
            subject_files = [Path(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(subject_files)} subjects from {args.subjects_file}")

    elif args.derivatives_dir and args.subjects:
        # Search derivatives directory
        subject_files = prepare_subjects_for_melodic(
            args.derivatives_dir,
            args.subjects,
            output_file=args.output_dir / 'subject_list.txt'
        )

    else:
        parser.error("Must provide either --subjects-file or (--derivatives-dir and --subjects)")

    # Convert n_components
    try:
        n_components = int(args.n_components)
    except ValueError:
        n_components = 'auto'

    # Run MELODIC
    results = run_melodic_group_ica(
        subject_files=subject_files,
        output_dir=args.output_dir,
        mask_file=args.mask,
        tr=args.tr,
        n_components=n_components,
        approach=args.approach,
        validate_inputs=not args.no_validate
    )

    # Print summary
    print("\n" + "=" * 80)
    print("MELODIC GROUP ICA ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Subjects analyzed: {results['n_subjects']}")
    print(f"Components: {results['n_components']}")
    print(f"Output directory: {results['output_dir']}")
    print(f"\nKey outputs:")
    for name, path in results['outputs'].items():
        print(f"  {name}: {path}")
    print("=" * 80)
