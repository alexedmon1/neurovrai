#!/usr/bin/env python3
"""
Re-run Functional/Anatomical Statistical Analyses with 5000 Permutations

This script re-runs FSL randomise for ASL, ReHo, fALFF, and VBM with:
- Full 6-contrast design (sex, age, group effects)
- 5000 permutations for robust inference
- TFCE correction

Usage:
    python rerun_functional_anatomical_stats.py
"""

import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Setup logging
log_file = Path('logs') / f'rerun_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Configuration
ANALYSIS_ROOT = Path('/mnt/bytopia/IRC805/analysis')
DESIGN_ROOT = Path('/mnt/bytopia/IRC805/data/designs')
N_PERM = 5000

# Analysis configurations
ANALYSES = {
    'ASL': {
        'input_4d': ANALYSIS_ROOT / 'asl' / 'asl_analysis' / 'all_cbf_4D.nii.gz',
        'mask': ANALYSIS_ROOT / 'asl' / 'asl_analysis' / 'group_mask.nii.gz',
        'design_dir': DESIGN_ROOT / 'asl',
        'output_dir': ANALYSIS_ROOT / 'asl' / 'asl_analysis' / 'randomise_output_5000perm',
        'prefix': 'randomise'
    },
    'ReHo': {
        'input_4d': ANALYSIS_ROOT / 'func' / 'reho' / 'all_reho_normalized_z.nii.gz',
        'mask': ANALYSIS_ROOT / 'func' / 'reho' / 'mock_study' / 'group_mask.nii.gz',
        'design_dir': DESIGN_ROOT / 'func_reho',
        'output_dir': ANALYSIS_ROOT / 'func' / 'reho' / 'randomise_5000perm',
        'prefix': 'reho_randomise'
    },
    'fALFF': {
        'input_4d': ANALYSIS_ROOT / 'func' / 'falff' / 'all_falff_normalized_z.nii.gz',
        'mask': None,  # Will create from data
        'design_dir': DESIGN_ROOT / 'func_falff',
        'output_dir': ANALYSIS_ROOT / 'func' / 'falff' / 'randomise_5000perm',
        'prefix': 'falff_randomise'
    },
    'VBM': {
        'input_4d': ANALYSIS_ROOT / 'anat' / 'vbm' / 'vbm_analysis' / 'merged_GM.nii.gz',
        'mask': ANALYSIS_ROOT / 'anat' / 'vbm' / 'vbm_analysis' / 'group_mask.nii.gz',
        'design_dir': DESIGN_ROOT / 'vbm',
        'output_dir': ANALYSIS_ROOT / 'anat' / 'vbm' / 'vbm_analysis' / 'stats' / 'randomise_output_5000perm',
        'prefix': 'randomise'
    }
}


def create_mask_from_4d(input_4d: Path, output_mask: Path):
    """Create binary mask from 4D data (non-zero voxels in all volumes)"""
    logger.info(f"Creating mask from {input_4d.name}")

    # Use fslmaths to create mask
    # 1. Take temporal mean
    # 2. Binarize
    # 3. Dilate slightly and erode to fill small holes
    cmd = [
        'fslmaths',
        str(input_4d),
        '-Tmean',
        '-bin',
        str(output_mask)
    ]

    subprocess.run(cmd, check=True)
    logger.info(f"  Created mask: {output_mask}")


def run_randomise(config: dict, analysis_name: str):
    """
    Run FSL randomise for a single analysis

    Parameters
    ----------
    config : dict
        Analysis configuration
    analysis_name : str
        Name of analysis (for logging)
    """

    logger.info("")
    logger.info("="*80)
    logger.info(f"Running {analysis_name}")
    logger.info("="*80)

    # Check input files
    input_4d = config['input_4d']
    if not input_4d.exists():
        logger.error(f"Input file not found: {input_4d}")
        return False

    # Get design files
    design_mat = config['design_dir'] / 'design.mat'
    design_con = config['design_dir'] / 'design.con'

    if not design_mat.exists() or not design_con.exists():
        logger.error(f"Design files not found in {config['design_dir']}")
        return False

    # Handle mask
    mask = config['mask']
    if mask is None:
        # Create mask from data
        mask = config['output_dir'] / 'group_mask.nii.gz'
        mask.parent.mkdir(parents=True, exist_ok=True)
        create_mask_from_4d(input_4d, mask)

    if not mask.exists():
        logger.error(f"Mask file not found: {mask}")
        return False

    # Create output directory
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log analysis info
    logger.info(f"Input: {input_4d}")
    logger.info(f"Mask: {mask}")
    logger.info(f"Design: {design_mat}")
    logger.info(f"Contrasts: {design_con}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Permutations: {N_PERM}")

    # Build randomise command
    output_prefix = output_dir / config['prefix']

    cmd = [
        'randomise',
        '-i', str(input_4d),
        '-o', str(output_prefix),
        '-d', str(design_mat),
        '-t', str(design_con),
        '-m', str(mask),
        '-n', str(N_PERM),
        '-T',  # TFCE correction
        '-x',  # Output corrected p-values only (saves space)
        '--uncorrp'  # Also save uncorrected p-values
    ]

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("")
    logger.info("Starting randomise (this may take 30-60 minutes)...")

    try:
        # Run randomise
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=output_dir
        )

        # Save log
        log_file = output_dir / 'randomise.log'
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        logger.info(f"✓ {analysis_name} completed successfully")
        logger.info(f"  Log: {log_file}")

        # Count output files
        n_tstat = len(list(output_dir.glob(f'{config["prefix"]}_tstat*.nii.gz')))
        n_corrp = len(list(output_dir.glob(f'{config["prefix"]}_tfce_corrp_tstat*.nii.gz')))
        logger.info(f"  Generated {n_tstat} t-stat maps, {n_corrp} corrp maps")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {analysis_name} failed")
        logger.error(f"  Error: {e}")
        if e.stderr:
            logger.error(f"  Stderr: {e.stderr}")
        return False

    except Exception as e:
        logger.error(f"✗ {analysis_name} failed with unexpected error")
        logger.error(f"  Error: {e}")
        return False


def main():
    """Run all analyses"""

    logger.info("="*80)
    logger.info("Re-running Functional/Anatomical Statistical Analyses")
    logger.info("="*80)
    logger.info(f"Number of permutations: {N_PERM}")
    logger.info(f"Analyses: {', '.join(ANALYSES.keys())}")
    logger.info("")

    # Track results
    results = {}

    # Run each analysis
    for analysis_name, config in ANALYSES.items():
        success = run_randomise(config, analysis_name)
        results[analysis_name] = success

    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    n_success = sum(results.values())
    n_failed = len(results) - n_success

    logger.info(f"Successful: {n_success}/{len(results)}")
    logger.info(f"Failed: {n_failed}/{len(results)}")
    logger.info("")

    for analysis_name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {analysis_name}")

    logger.info("")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    if n_failed > 0:
        logger.info("Some analyses failed. Check the log file for details.")
        sys.exit(1)
    else:
        logger.info("All analyses completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run cluster analysis: python batch_functional_cluster_analysis.py")
        logger.info("  2. View reports in /mnt/bytopia/IRC805/analysis/cluster_reports/")


if __name__ == '__main__':
    main()
