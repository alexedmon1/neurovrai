#!/usr/bin/env python3
"""
Create config.yaml for MRI Preprocessing Pipeline

This script creates config.yaml in the study root directory by default.

Usage:
    # Creates /mnt/bytopia/IRC805/config.yaml
    python create_config.py --study-root /mnt/bytopia/IRC805

    # Interactive mode
    python create_config.py
"""

import argparse
import sys
from pathlib import Path
import yaml


def create_config_template(study_root: Path, output_file: Path = None):
    """
    Create a complete config.yaml template.

    By default, creates config.yaml in the study root directory.
    """

    if output_file is None:
        output_file = study_root / 'config.yaml'

    config = {
        # Project paths
        'project_dir': str(study_root),
        'dicom_dir': f'{study_root}/raw/dicom',  # Where DICOM files are located
        'bids_dir': f'{study_root}/bids',  # Where NIfTI files will be created
        'derivatives_dir': f'{study_root}/derivatives',  # Preprocessed outputs
        'work_dir': f'{study_root}/work',  # Temporary Nipype files

        # Path structure for workflows
        'paths': {
            'logs': f'{study_root}/logs',
            'transforms': f'{study_root}/transforms',
            'qc': f'{study_root}/qc'
        },

        # Execution settings
        'execution': {
            'plugin': 'MultiProc',
            'n_procs': 6  # Adjust based on your system
        },

        # Template files (FSL standard templates)
        'templates': {
            'mni152_t1_2mm': '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz',
            'mni152_t1_1mm': '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz',
            'fmrib58_fa': '/usr/local/fsl/data/standard/FMRIB58_FA_1mm.nii.gz'
        },

        # Anatomical preprocessing
        'anatomical': {
            'bet': {
                'frac': 0.5,
                'reduce_bias': True,
                'robust': True
            },
            'bias_correction': {
                'n_iterations': [50, 50, 30, 20],  # N4 multi-resolution iterations
                'shrink_factor': 3,  # Downsampling for speed
                'convergence_threshold': 0.001,
                'bspline_fitting_distance': 300  # B-spline mesh resolution
            },
            'segmentation_method': 'ants',  # 'ants' (Atropos) or 'fsl' (FAST)
            'atropos': {
                'number_of_tissue_classes': 3,  # CSF, GM, WM
                'initialization': 'KMeans',  # KMeans, Otsu, or PriorProbabilityImages
                'n_iterations': 5,
                'convergence_threshold': 0.001,
                'mrf_smoothing_factor': 0.1,
                'mrf_radius': [1, 1, 1]  # Markov Random Field spatial smoothing radius
            },
            'registration_method': 'fsl',  # or 'ants'
            'run_qc': True
        },

        # Diffusion preprocessing
        'diffusion': {
            'denoise_method': 'dwidenoise',
            'bet': {
                'frac': 0.3  # Lower than anatomical (0.5) - DWI has lower contrast
            },
            'topup': {
                'enabled': 'auto',  # 'auto', True, or False
                'readout_time': 0.05  # Check your protocol!
            },
            'eddy_config': {
                'flm': 'linear',
                'slm': 'linear',
                'use_cuda': True  # Set to False if no GPU
            },
            'advanced_models': {
                'enabled': 'auto',  # Auto-detect multi-shell
                'fit_dki': True,
                'fit_noddi': True,
                'fit_sandi': False,  # Requires ≥3 shells
                'fit_activeax': False,  # Requires specific protocol
                'use_amico': True  # Recommended: 100x faster
            },
            'run_qc': True
        },

        # Functional preprocessing
        'functional': {
            'tr': 1.029,  # Check your protocol!
            'te': [10.0, 30.0, 50.0],  # For multi-echo, check your protocol!
            'bet': {
                'frac': 0.3  # Lower than anatomical - functional has lower contrast
            },
            'highpass': 0.001,
            'lowpass': 0.08,
            'fwhm': 6,
            'normalize_to_mni': True,
            'tedana': {
                'enabled': True,  # For multi-echo
                'tedpca': 225,  # num_volumes / 2 for 450 volumes (improves ICA convergence)
                'tree': 'kundu'
            },
            'aroma': {
                'enabled': 'auto'  # Auto-enables for single-echo
            },
            'acompcor': {
                'enabled': True,
                'num_components': 5,
                'variance_threshold': 0.5
            },
            'run_qc': True
        },

        # ASL (Arterial Spin Labeling) preprocessing
        'asl': {
            'labeling_type': 'pcasl',
            'labeling_duration': 1.932,  # τ (tau) - auto-extracted from DICOM
            'post_labeling_delay': 2.031,  # PLD - auto-extracted from DICOM
            'bet': {
                'frac': 0.3  # Lower than anatomical - ASL has lower contrast
            },
            'labeling_efficiency': 0.85,
            't1_blood': 1.65,  # T1 at 3T
            'blood_brain_partition': 0.9,
            'label_control_order': 'control_first',  # Auto-detected
            'background_suppression_pulses': 1,  # Auto-detected
            'apply_m0_calibration': True,
            'wm_cbf_reference': 25.0,  # ml/100g/min
            'apply_pvc': False,
            'normalize_to_mni': False,
            'run_qc': True
        },

        # FreeSurfer (EXPERIMENTAL - NOT PRODUCTION READY)
        'freesurfer': {
            'enabled': False,
            'subjects_dir': f'{study_root}/freesurfer',
            'use_for_tractography': False,
            'use_for_masks': False
        }
    }

    # Write YAML with comments
    header = f"""# MRI Preprocessing Pipeline Configuration
# Generated for: {study_root}
#
# IMPORTANT: Review and customize before using!
# - Verify all paths are correct
# - Check sequence-specific parameters (TR, TE, readout_time)
# - Adjust processing parameters based on your needs
#
# Directory Structure:
#   {study_root}/
#   ├── raw/dicom/           # Raw DICOM files (input)
#   ├── bids/                # Converted NIfTI files (created by pipeline)
#   ├── derivatives/         # Preprocessed outputs
#   ├── work/                # Temporary Nipype files (can be deleted)
#   ├── qc/                  # Quality control reports
#   ├── logs/                # Log files
#   └── transforms/          # Spatial transformation matrices
#
# Usage:
#   # Single subject
#   uv run python run_simple_pipeline.py --subject SUB_ID --dicom-dir {study_root}/raw/dicom/SUB_ID --config {study_root}/config.yaml
#
#   # Batch processing
#   uv run python run_batch_simple.py --config {study_root}/config.yaml
#

"""

    # Write file
    with open(output_file, 'w') as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)

    print(f"✓ Created configuration: {output_file}")
    print()
    print("Next steps:")
    print(f"  1. Review and edit: {output_file}")
    print(f"  2. Check sequence parameters (TR, TE, readout_time)")
    print(f"  3. Verify FSL template paths exist")
    print(f"  4. Validate: uv run python verify_environment.py")
    print()
    print(f"Then run pipeline:")
    print(f"  uv run python run_simple_pipeline.py --subject SUBJECT_ID --dicom-dir {study_root}/raw/dicom/SUBJECT_ID --config {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Create config.yaml for MRI preprocessing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create config for IRC805 study (creates /mnt/bytopia/IRC805/config.yaml)
    python create_config.py --study-root /mnt/bytopia/IRC805

    # Create with custom location
    python create_config.py --study-root /mnt/bytopia/IRC805 --output /path/to/my_config.yaml

    # Interactive mode
    python create_config.py
        """
    )

    parser.add_argument(
        '--study-root',
        type=Path,
        help='Study root directory (e.g., /mnt/bytopia/IRC805)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output config file (default: {study_root}/config.yaml)'
    )

    args = parser.parse_args()

    # Interactive mode if no study root provided
    if not args.study_root:
        print("="*70)
        print("MRI Preprocessing Pipeline - Config Generator")
        print("="*70)
        print()

        study_root = input("Enter study root directory (e.g., /mnt/bytopia/IRC805): ").strip()

        if not study_root:
            print("Error: Study root is required")
            sys.exit(1)

        args.study_root = Path(study_root)

    # Verify study root exists or ask to create
    if not args.study_root.exists():
        print(f"Warning: {args.study_root} does not exist")
        create = input("Create directory structure? [y/N]: ").strip().lower()

        if create == 'y':
            args.study_root.mkdir(parents=True, exist_ok=True)
            (args.study_root / 'raw' / 'dicom').mkdir(parents=True, exist_ok=True)
            (args.study_root / 'logs').mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory structure")
        else:
            print("Config will be created but directories don't exist yet")

    # Check for DICOM directory
    dicom_dir = args.study_root / 'raw' / 'dicom'
    if not dicom_dir.exists():
        print(f"Warning: DICOM directory not found: {dicom_dir}")
        print("Make sure to place DICOM files in this location before running pipeline")

    # Create config
    create_config_template(args.study_root, args.output)


if __name__ == '__main__':
    main()
