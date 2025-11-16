#!/usr/bin/env python3
"""
Production MRI Preprocessing Runner

Runs anatomical, diffusion, and functional preprocessing workflows
on a single subject using the validated production workflows.

Usage:
    # Run anatomical preprocessing
    python run_preprocessing.py --subject IRC805-0580101 --modality anat

    # Run diffusion preprocessing
    python run_preprocessing.py --subject IRC805-0580101 --modality dwi

    # Run functional preprocessing (requires anatomical to be done first)
    python run_preprocessing.py --subject IRC805-0580101 --modality func

    # Run all modalities
    python run_preprocessing.py --subject IRC805-0580101 --modality all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from mri_preprocess.config import load_config

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


def get_base_config(study_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get base configuration for all workflows from loaded config."""
    return {
        'paths': {
            'logs': str(study_root / 'logs')
        },
        'templates': config.get('templates', {
            'mni152_t1_2mm': '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
        }),
        'execution': config.get('execution', {
            'plugin': 'MultiProc',
            'plugin_args': {'n_procs': 6}
        }),
        'n_procs': config.get('execution', {}).get('n_procs', 6)
    }


def run_anatomical(subject: str, study_root: Path, yaml_config: Dict[str, Any]) -> bool:
    """Run anatomical preprocessing."""
    logger.info("="*70)
    logger.info(f"ANATOMICAL PREPROCESSING: {subject}")
    logger.info("="*70)

    from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing

    # Build config from YAML
    config = get_base_config(study_root, yaml_config)
    anat_params = yaml_config.get('anatomical', {})
    config.update({
        'bet': anat_params.get('bet', {
            'frac': 0.5,
            'reduce_bias': True,
            'robust': True
        }),
        'fast': anat_params.get('fast', {
            'bias_iters': 4,
            'bias_lowpass': 10
        }),
        'run_qc': anat_params.get('run_qc', True)
    })

    # Find T1w file
    anat_dir = study_root / f'subjects/{subject}/nifti/anat'
    t1w_files = list(anat_dir.glob('*T1*.nii.gz'))

    if not t1w_files:
        logger.error(f"No T1w file found in {anat_dir}")
        return False

    t1w_file = t1w_files[0]
    logger.info(f"Input T1w: {t1w_file}")

    # Use standardized output directory: {study_root}/derivatives
    outdir = study_root / 'derivatives'

    try:
        results = run_anat_preprocessing(
            config=config,
            subject=subject,
            t1w_file=t1w_file,
            output_dir=outdir,  # Pass derivatives directory
            run_qc=True
        )

        logger.info("✓ Anatomical preprocessing completed")
        logger.info(f"  Output: {outdir / subject / 'anat'}")
        logger.info(f"  Brain: {results.get('brain')}")
        logger.info(f"  CSF segmentation: {results.get('csf_prob')}")
        logger.info(f"  GM segmentation: {results.get('gm_prob')}")
        logger.info(f"  WM segmentation: {results.get('wm_prob')}")
        return True

    except Exception as e:
        logger.error(f"Anatomical preprocessing failed: {e}", exc_info=True)
        return False


def run_diffusion(subject: str, study_root: Path, yaml_config: Dict[str, Any]) -> bool:
    """Run diffusion preprocessing."""
    logger.info("="*70)
    logger.info(f"DIFFUSION PREPROCESSING: {subject}")
    logger.info("="*70)

    from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing

    # Build config from YAML
    config = get_base_config(study_root, yaml_config)
    dwi_params = yaml_config.get('diffusion', {})
    config.update({
        'topup': dwi_params.get('topup', {
            'readout_time': 0.05
        }),
        'eddy': dwi_params.get('eddy', {
            'use_cuda': True
        }),
        'run_qc': dwi_params.get('run_qc', True)
    })

    # Find DWI files
    dwi_dir = study_root / f'subjects/{subject}/nifti/dwi'

    # Find multi-shell files
    dwi_files = sorted(dwi_dir.glob('*DTI*.nii.gz'))
    bval_files = sorted(dwi_dir.glob('*DTI*.bval'))
    bvec_files = sorted(dwi_dir.glob('*DTI*.bvec'))

    # Find reverse phase encoding files
    rev_phase_files = sorted(dwi_dir.glob('*SE_EPI*.nii.gz'))

    if not dwi_files:
        logger.error(f"No DWI files found in {dwi_dir}")
        return False

    logger.info(f"Found {len(dwi_files)} DWI files")
    logger.info(f"Found {len(rev_phase_files)} reverse phase files")

    # Use standardized output directory: {study_root}/derivatives
    outdir = study_root / 'derivatives'

    try:
        results = run_dwi_multishell_topup_preprocessing(
            config=config,
            subject=subject,
            dwi_files=dwi_files,
            bval_files=bval_files,
            bvec_files=bvec_files,
            rev_phase_files=rev_phase_files,
            output_dir=outdir,  # Pass derivatives directory
            run_bedpostx=False
        )

        logger.info("✓ Diffusion preprocessing completed")
        logger.info(f"  Output: {outdir / subject / 'dwi'}")
        logger.info(f"  FA map: {results.get('fa')}")
        logger.info(f"  MD map: {results.get('md')}")
        return True

    except Exception as e:
        logger.error(f"Diffusion preprocessing failed: {e}", exc_info=True)
        return False


def run_functional(subject: str, study_root: Path, yaml_config: Dict[str, Any]) -> bool:
    """Run functional preprocessing."""
    logger.info("="*70)
    logger.info(f"FUNCTIONAL PREPROCESSING: {subject}")
    logger.info("="*70)

    # Use standardized output directory: {study_root}/derivatives
    outdir = study_root / 'derivatives'

    # Check if anatomical preprocessing is done
    anat_derivatives = outdir / subject / 'anat'
    if not anat_derivatives.exists():
        logger.error(f"Anatomical preprocessing required first!")
        logger.error(f"Run: python run_preprocessing.py --subject {subject} --modality anat")
        return False

    from mri_preprocess.workflows.func_preprocess import run_func_preprocessing

    # Build config from YAML
    config = get_base_config(study_root, yaml_config)
    func_params = yaml_config.get('functional', {})
    config.update({
        'tr': func_params.get('tr', 1.029),
        'te': func_params.get('te', [10.0, 30.0, 50.0]),
        'highpass': func_params.get('highpass', 0.001),
        'lowpass': func_params.get('lowpass', 0.08),
        'fwhm': func_params.get('fwhm', 6),
        'tedana': func_params.get('tedana', {
            'enabled': True,
            'tedpca': 'kundu',
            'tree': 'kundu'
        }),
        'aroma': func_params.get('aroma', {
            'enabled': False  # Redundant with TEDANA
        }),
        'acompcor': func_params.get('acompcor', {
            'enabled': True,
            'num_components': 5,
            'variance_threshold': 0.5
        }),
        'run_qc': func_params.get('run_qc', True)
    })

    # Find functional files (multi-echo)
    rest_dir = study_root / f'subjects/{subject}/nifti/rest'
    func_files = sorted(rest_dir.glob('*RESTING*_e*.nii.gz'))

    if not func_files:
        logger.error(f"No functional files found in {rest_dir}")
        return False

    logger.info(f"Found {len(func_files)} echo files")

    try:
        results = run_func_preprocessing(
            config=config,
            subject=subject,
            func_file=func_files,
            output_dir=outdir,  # Pass derivatives directory
            anat_derivatives=anat_derivatives
        )

        logger.info("✓ Functional preprocessing completed")
        logger.info(f"  Output: {outdir / subject / 'func'}")
        logger.info(f"  Preprocessed BOLD: {results.get('preprocessed_bold')}")
        return True

    except Exception as e:
        logger.error(f"Functional preprocessing failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run MRI preprocessing workflows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--subject',
        required=True,
        help='Subject ID (e.g., IRC805-0580101)'
    )
    parser.add_argument(
        '--modality',
        required=True,
        choices=['anat', 'dwi', 'func', 'all'],
        help='Modality to process'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Path to config YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--study-root',
        type=Path,
        default=None,
        help='Study root directory (overrides config.yaml if provided)'
    )

    args = parser.parse_args()

    # Load config from YAML (this uses mri_preprocess.config.load_config with variable substitution)
    yaml_config = load_config(args.config, validate=False)

    # Use study_root from CLI if provided, otherwise extract from config
    # Config can have either 'project_dir' (README.md format) or 'study_root'
    if args.study_root:
        study_root = args.study_root
    elif 'project_dir' in yaml_config:
        study_root = Path(yaml_config['project_dir'])
    elif 'study_root' in yaml_config:
        study_root = Path(yaml_config['study_root'])
    else:
        raise ValueError("study_root must be provided via --study-root or in config.yaml as 'project_dir' or 'study_root'")

    logger.info("="*70)
    logger.info("MRI PREPROCESSING PIPELINE")
    logger.info("="*70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Modality: {args.modality}")
    logger.info(f"Study root: {study_root}")
    logger.info("")

    success = True

    if args.modality == 'anat' or args.modality == 'all':
        success = run_anatomical(args.subject, study_root, yaml_config) and success

    if args.modality == 'dwi' or args.modality == 'all':
        success = run_diffusion(args.subject, study_root, yaml_config) and success

    if args.modality == 'func' or args.modality == 'all':
        success = run_functional(args.subject, study_root, yaml_config) and success

    if success:
        logger.info("="*70)
        logger.info("✓ ALL PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        return 0
    else:
        logger.error("="*70)
        logger.error("✗ PREPROCESSING FAILED")
        logger.error("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
