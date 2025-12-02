#!/usr/bin/env python3
"""
Dual Regression Analysis for Resting-State fMRI

Performs dual regression to obtain subject-specific spatial maps and time courses
from group ICA components. This allows for subject-level analysis and group-level
statistical inference on independent component networks.

The dual regression approach:
1. Spatial Regression: Regress group ICA spatial maps against each subject's 4D data
   to obtain subject-specific time courses
2. Temporal Regression: Regress subject-specific time courses against subject's 4D data
   to obtain subject-specific spatial maps
3. Statistical Analysis: Perform group-level statistics on subject-specific maps

Usage:
    from neurovrai.analysis.func.dual_regression import run_dual_regression

    results = run_dual_regression(
        group_ic_maps=Path('/study/melodic/melodic_IC.nii.gz'),
        subject_files=[Path('sub-001/preprocessed.nii.gz'), ...],
        output_dir=Path('/study/dual_regression'),
        design_mat=Path('/study/design.mat'),  # Optional
        contrast_con=Path('/study/design.con')  # Optional
    )
"""

import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import shutil

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dual_regression_inputs(
    group_ic_maps: Path,
    subject_files: List[Path]
) -> Dict:
    """
    Validate inputs for dual regression

    Args:
        group_ic_maps: Group ICA spatial maps (4D)
        subject_files: List of preprocessed subject 4D images

    Returns:
        Dictionary with validation results
    """
    logger.info("Validating dual regression inputs...")

    results = {
        'valid': True,
        'errors': [],
        'n_components': None,
        'n_subjects': len(subject_files),
        'spatial_dims': None
    }

    # Check group IC maps exist
    if not group_ic_maps.exists():
        results['valid'] = False
        results['errors'].append(f"Group IC maps not found: {group_ic_maps}")
        return results

    # Load group ICA maps to get dimensions
    try:
        ic_img = nib.load(group_ic_maps)
        ic_shape = ic_img.shape

        if len(ic_shape) != 4:
            results['valid'] = False
            results['errors'].append(f"Group IC maps should be 4D, got shape {ic_shape}")
            return results

        results['n_components'] = ic_shape[3]
        results['spatial_dims'] = ic_shape[:3]
        logger.info(f"  Group ICA: {results['n_components']} components, shape {results['spatial_dims']}")

    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Failed to load group IC maps: {e}")
        return results

    # Validate subject files
    valid_subjects = []
    for subj_file in subject_files:
        if not subj_file.exists():
            logger.warning(f"  Subject file not found: {subj_file}")
            continue

        try:
            subj_img = nib.load(subj_file)
            subj_shape = subj_img.shape

            if len(subj_shape) != 4:
                logger.warning(f"  Not 4D: {subj_file}")
                continue

            if subj_shape[:3] != results['spatial_dims']:
                logger.warning(f"  Dimension mismatch: {subj_file} (expected {results['spatial_dims']})")
                continue

            valid_subjects.append(subj_file)

        except Exception as e:
            logger.warning(f"  Failed to load: {subj_file} - {e}")

    results['valid_subjects'] = valid_subjects
    results['n_valid_subjects'] = len(valid_subjects)

    if len(valid_subjects) < 2:
        results['valid'] = False
        results['errors'].append(f"Need at least 2 valid subjects, found {len(valid_subjects)}")

    logger.info(f"  Valid subjects: {len(valid_subjects)} / {len(subject_files)}")

    return results


def run_dual_regression(
    group_ic_maps: Path,
    subject_files: List[Path],
    output_dir: Path,
    design_mat: Optional[Path] = None,
    contrast_con: Optional[Path] = None,
    n_permutations: int = 5000,
    demean: bool = True,
    des_norm: bool = False,
    validate_inputs: bool = True
) -> Dict:
    """
    Run FSL dual regression analysis

    Args:
        group_ic_maps: Group ICA component maps from MELODIC (4D .nii.gz)
        subject_files: List of preprocessed 4D subject images (must be in same space as ICA maps)
        output_dir: Output directory for results
        design_mat: Design matrix for group statistics (optional, FSL .mat format)
        contrast_con: Contrast file for group statistics (optional, FSL .con format)
        n_permutations: Number of permutations for randomise (default: 5000)
        demean: Demean time courses (default: True)
        des_norm: Variance normalize time courses (default: False)
        validate_inputs: Validate inputs before running (default: True)

    Returns:
        Dictionary with analysis results and file paths

    Outputs:
        {output_dir}/
            dr_stage1_{subject}/ - Stage 1: Subject-specific time courses
            dr_stage2_{subject}.nii.gz - Stage 2: Subject-specific spatial maps (one per subject)
            dr_stage3_ic{N}_*.nii.gz - Stage 3: Group statistics (one set per component)
            subject_list.txt - List of input files
            command.log - Dual regression command and output
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"dual_regression_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("DUAL REGRESSION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Group IC maps: {group_ic_maps}")
    logger.info(f"Number of subjects: {len(subject_files)}")
    logger.info(f"Output directory: {output_dir}")
    if design_mat:
        logger.info(f"Design matrix: {design_mat}")
    if contrast_con:
        logger.info(f"Contrasts: {contrast_con}")
    logger.info("")

    # Validate inputs
    if validate_inputs:
        validation = validate_dual_regression_inputs(group_ic_maps, subject_files)

        if not validation['valid']:
            error_msg = "Validation failed:\n" + "\n".join(validation['errors'])
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Use only valid subjects
        subject_files = validation['valid_subjects']
        n_components = validation['n_components']
    else:
        # Load just to get n_components
        ic_img = nib.load(group_ic_maps)
        n_components = ic_img.shape[3]

    logger.info(f"Processing {len(subject_files)} subjects with {n_components} components")

    # Create subject list file
    subject_list_file = output_dir / 'subject_list.txt'
    with open(subject_list_file, 'w') as f:
        for subj_file in subject_files:
            f.write(f'{subj_file.resolve()}\n')

    logger.info(f"Created subject list: {subject_list_file}")

    # Build dual_regression command
    cmd = [
        'dual_regression',
        str(group_ic_maps.resolve()),
        '1' if demean else '0',
        str(design_mat.resolve()) if design_mat else '-1',
        str(n_permutations) if design_mat else '1',
        str(output_dir.resolve()),
        *(str(f.resolve()) for f in subject_files)
    ]

    # Add des_norm if needed
    if des_norm:
        cmd.insert(3, '1')
    else:
        cmd.insert(3, '0')

    logger.info("Running dual regression...")
    logger.info(f"Command: {' '.join(cmd)}")

    # Run dual regression
    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"✓ Dual regression completed in {elapsed:.1f} seconds")

        # Save command output
        cmd_log_file = output_dir / 'command.log'
        with open(cmd_log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Dual regression failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise

    # Collect output files
    outputs = {
        'group_ic_maps': str(group_ic_maps),
        'n_components': n_components,
        'n_subjects': len(subject_files),
        'output_dir': str(output_dir),
        'subject_list': str(subject_list_file),
        'elapsed_time': elapsed,
        'stage1_dirs': [],  # Subject-specific time courses
        'stage2_files': [],  # Subject-specific spatial maps
        'stage3_files': {}  # Group statistics (per component)
    }

    # Find stage 1 outputs (time courses)
    for subj_idx in range(len(subject_files)):
        stage1_dir = output_dir / f'dr_stage1_subject{subj_idx:05d}'
        if stage1_dir.exists():
            outputs['stage1_dirs'].append(str(stage1_dir))

    # Find stage 2 outputs (subject spatial maps)
    for stage2_file in sorted(output_dir.glob('dr_stage2_subject*.nii.gz')):
        outputs['stage2_files'].append(str(stage2_file))

    # Find stage 3 outputs (group statistics, if design provided)
    if design_mat:
        for ic_idx in range(n_components):
            ic_files = list(output_dir.glob(f'dr_stage3_ic{ic_idx:04d}_*.nii.gz'))
            if ic_files:
                outputs['stage3_files'][f'ic_{ic_idx}'] = [str(f) for f in ic_files]

    logger.info(f"\nOutputs:")
    logger.info(f"  Stage 1 (time courses): {len(outputs['stage1_dirs'])} directories")
    logger.info(f"  Stage 2 (spatial maps): {len(outputs['stage2_files'])} files")
    if outputs['stage3_files']:
        logger.info(f"  Stage 3 (statistics): {len(outputs['stage3_files'])} components")

    # Save results summary
    results_file = output_dir / 'dual_regression_results.json'
    with open(results_file, 'w') as f:
        json.dump(outputs, f, indent=2)

    logger.info(f"\n✓ Results saved: {results_file}")
    logger.info(f"✓ Log file: {log_file}")
    logger.info("=" * 80 + "\n")

    # Remove file handler
    logger.removeHandler(file_handler)
    file_handler.close()

    return outputs


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Dual regression analysis for resting-state fMRI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--group-ics',
        type=Path,
        required=True,
        help='Group ICA spatial maps (melodic_IC.nii.gz)'
    )
    parser.add_argument(
        '--subjects-file',
        type=Path,
        required=True,
        help='Text file with list of subject 4D images (one per line)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--design',
        type=Path,
        help='Design matrix (.mat) for group statistics (optional)'
    )
    parser.add_argument(
        '--contrasts',
        type=Path,
        help='Contrast file (.con) for group statistics (optional)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=5000,
        help='Number of permutations for randomise (default: 5000)'
    )
    parser.add_argument(
        '--no-demean',
        action='store_true',
        help='Do not demean time courses'
    )
    parser.add_argument(
        '--des-norm',
        action='store_true',
        help='Variance normalize time courses'
    )

    args = parser.parse_args()

    # Load subject files
    with open(args.subjects_file) as f:
        subject_files = [Path(line.strip()) for line in f if line.strip()]

    # Run dual regression
    results = run_dual_regression(
        group_ic_maps=args.group_ics,
        subject_files=subject_files,
        output_dir=args.output_dir,
        design_mat=args.design,
        contrast_con=args.contrasts,
        n_permutations=args.n_permutations,
        demean=not args.no_demean,
        des_norm=args.des_norm
    )

    print("\n" + "=" * 80)
    print("DUAL REGRESSION COMPLETE")
    print("=" * 80)
    print(f"Components: {results['n_components']}")
    print(f"Subjects: {results['n_subjects']}")
    print(f"Output: {results['output_dir']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
