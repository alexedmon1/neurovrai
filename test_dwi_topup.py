#!/usr/bin/env python3
"""
Test script for DWI preprocessing with TOPUP.
"""

import sys
from pathlib import Path
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing
from mri_preprocess.utils.topup_helper import create_topup_files_for_multishell

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test DWI preprocessing with TOPUP on IRC805-0580101."""

    # Subject info
    subject = 'IRC805-0580101'
    subject_dir = Path('/mnt/bytopia/IRC805/subjects') / subject
    dwi_dir = subject_dir / 'nifti' / 'dwi'

    logger.info(f"Testing DWI preprocessing for {subject}")
    logger.info(f"DWI directory: {dwi_dir}")

    # Find DWI files
    dwi_b1000_b2000 = list(dwi_dir.glob('*DTI_2shell_b1000_b2000_MB4*.nii.gz'))[0]
    dwi_b3000 = list(dwi_dir.glob('*DTI_1shell_b3000_MB4*.nii.gz'))[0]

    bval_b1000_b2000 = dwi_b1000_b2000.with_suffix('').with_suffix('.bval')
    bval_b3000 = dwi_b3000.with_suffix('').with_suffix('.bval')

    bvec_b1000_b2000 = dwi_b1000_b2000.with_suffix('').with_suffix('.bvec')
    bvec_b3000 = dwi_b3000.with_suffix('').with_suffix('.bvec')

    # Find reverse PE files
    rev_pe_files = sorted(dwi_dir.glob('*SE_EPI_Posterior*.nii.gz'))

    logger.info(f"\nFound files:")
    logger.info(f"  DWI shell 1: {dwi_b1000_b2000.name}")
    logger.info(f"  DWI shell 2: {dwi_b3000.name}")
    logger.info(f"  Reverse PE files: {len(rev_pe_files)}")
    for rev in rev_pe_files:
        logger.info(f"    - {rev.name}")

    # Organize files
    dwi_files = [dwi_b1000_b2000, dwi_b3000]
    bval_files = [bval_b1000_b2000, bval_b3000]
    bvec_files = [bvec_b1000_b2000, bvec_b3000]

    # Verify all files exist
    all_files = dwi_files + bval_files + bvec_files + rev_pe_files
    for f in all_files:
        if not f.exists():
            logger.error(f"File not found: {f}")
            return 1

    logger.info("\n" + "="*60)
    logger.info("STEP 1: Creating TOPUP acquisition parameter files")
    logger.info("="*60)

    # Create parameter files
    params_dir = Path('/mnt/bytopia/development/IRC805/dwi_params')

    try:
        acqparams_file, index_file = create_topup_files_for_multishell(
            dwi_files=dwi_files,
            pe_direction='AP',  # Assuming AP based on typical acquisition
            readout_time=0.05,   # Typical value, may need adjustment
            output_dir=params_dir
        )
        logger.info(f"Created: {acqparams_file}")
        logger.info(f"Created: {index_file}")
    except Exception as e:
        logger.error(f"Error creating parameter files: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logger.info("\n" + "="*60)
    logger.info("STEP 2: Creating configuration")
    logger.info("="*60)

    # Create config
    config = {
        'paths': {
            'logs': str(Path('/mnt/bytopia/IRC805/logs')),
        },
        'diffusion': {
            'topup': {
                'encoding_file': str(acqparams_file)
            },
            'eddy': {
                'acqp_file': str(acqparams_file),
                'index_file': str(index_file),
                'method': 'jac',
                'repol': True,
                'use_cuda': True
            },
            'bedpostx': {
                'n_fibres': 2,
                'n_jumps': 1250,
                'burn_in': 1000,
                'use_gpu': True
            }
        },
        'execution': {
            'plugin': 'MultiProc',
            'plugin_args': {
                'n_procs': 4
            }
        }
    }

    logger.info("Configuration created")

    logger.info("\n" + "="*60)
    logger.info("STEP 3: Running DWI preprocessing with TOPUP")
    logger.info("="*60)

    # Setup directories - using new structure
    # study_root is the base directory containing dicoms/, nifti/, derivatives/, work/
    study_root = Path('/mnt/bytopia/development/IRC805')
    logger.info(f"Study root: {study_root}")
    logger.info(f"  Derivatives will be in: {study_root}/derivatives/dwi_topup/{subject}/")
    logger.info(f"  Working directory: {study_root}/work/{subject}/dwi_topup/")

    try:
        results = run_dwi_multishell_topup_preprocessing(
            config=config,
            subject=subject,
            dwi_files=dwi_files,
            bval_files=bval_files,
            bvec_files=bvec_files,
            rev_phase_files=rev_pe_files,
            output_dir=study_root,  # Pass study root, function will create derivatives/work structure
            work_dir=None,  # Let function create work_dir automatically
            run_bedpostx=False  # Start without BEDPOSTX for faster testing
        )

        logger.info("\n" + "="*60)
        logger.info("SUCCESS! DWI preprocessing completed")
        logger.info("="*60)
        logger.info("\nOutputs:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")

        return 0

    except Exception as e:
        logger.error(f"\nERROR during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
