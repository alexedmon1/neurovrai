#!/usr/bin/env python3
"""
Run Functional/Anatomical Statistical Analyses in PARALLEL with 5000 Permutations

Launches all 4 analyses (ASL, ReHo, fALFF, VBM) simultaneously as background processes.

Usage:
    python run_parallel_stats.py
"""

import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time

# Setup logging
log_file = Path('logs') / f'parallel_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        'mask': None,  # Will create
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
    """Create binary mask from 4D data"""
    logger.info(f"  Creating mask from {input_4d.name}")
    cmd = ['fslmaths', str(input_4d), '-Tmean', '-bin', str(output_mask)]
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info(f"  ✓ Created mask: {output_mask}")


def launch_randomise(config: dict, analysis_name: str):
    """
    Launch FSL randomise as background process

    Returns:
        tuple: (process, output_dir, analysis_name)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Launching {analysis_name}")
    logger.info(f"{'='*80}")

    # Check input files
    input_4d = config['input_4d']
    if not input_4d.exists():
        logger.error(f"Input file not found: {input_4d}")
        return None

    # Get design files
    design_mat = config['design_dir'] / 'design.mat'
    design_con = config['design_dir'] / 'design.con'

    if not design_mat.exists() or not design_con.exists():
        logger.error(f"Design files not found in {config['design_dir']}")
        return None

    # Handle mask
    mask = config['mask']
    if mask is None:
        mask = config['output_dir'] / 'group_mask.nii.gz'
        mask.parent.mkdir(parents=True, exist_ok=True)
        create_mask_from_4d(input_4d, mask)

    if not mask.exists():
        logger.error(f"Mask file not found: {mask}")
        return None

    # Create output directory
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log info
    logger.info(f"  Input: {input_4d}")
    logger.info(f"  Mask: {mask}")
    logger.info(f"  Design: {design_mat}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Permutations: {N_PERM}")

    # Build randomise command
    output_prefix = output_dir / config['prefix']
    log_file = output_dir / 'randomise.log'

    cmd = [
        'randomise',
        '-i', str(input_4d),
        '-o', str(output_prefix),
        '-d', str(design_mat),
        '-t', str(design_con),
        '-m', str(mask),
        '-n', str(N_PERM),
        '-T',  # TFCE
        '-x',  # Output corrected p only
        '--uncorrp'  # Also uncorrected
    ]

    logger.info(f"  Command: {' '.join(cmd)}")

    # Launch process in background
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=output_dir
        )

    logger.info(f"  ✓ Launched {analysis_name} (PID: {process.pid})")
    logger.info(f"  Log: {log_file}")

    return (process, output_dir, analysis_name, config['prefix'])


def monitor_processes(processes):
    """Monitor running processes and report when they complete"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Monitoring {len(processes)} parallel analyses")
    logger.info(f"{'='*80}\n")

    completed = []

    while len(completed) < len(processes):
        for i, (proc, out_dir, name, prefix) in enumerate(processes):
            if i in completed:
                continue

            # Check if process finished
            if proc.poll() is not None:
                completed.append(i)

                # Check success
                returncode = proc.returncode
                if returncode == 0:
                    # Count outputs
                    n_tstat = len(list(out_dir.glob(f'{prefix}_tstat*.nii.gz')))
                    n_corrp = len(list(out_dir.glob(f'{prefix}_tfce_corrp_tstat*.nii.gz')))

                    logger.info(f"✅ {name} COMPLETED")
                    logger.info(f"   Generated {n_tstat} t-stat maps, {n_corrp} corrp maps")
                    logger.info(f"   Output: {out_dir}")
                else:
                    logger.error(f"❌ {name} FAILED (exit code {returncode})")
                    logger.error(f"   Check log: {out_dir / 'randomise.log'}")

        if len(completed) < len(processes):
            time.sleep(30)  # Check every 30 seconds

    logger.info(f"\n{'='*80}")
    logger.info("ALL ANALYSES COMPLETE")
    logger.info(f"{'='*80}\n")


def main():
    """Launch all analyses in parallel"""

    logger.info(f"{'='*80}")
    logger.info("Parallel Functional/Anatomical Statistical Analyses")
    logger.info(f"{'='*80}")
    logger.info(f"Number of permutations: {N_PERM}")
    logger.info(f"Analyses: {', '.join(ANALYSES.keys())}")
    logger.info(f"Mode: PARALLEL (all analyses run simultaneously)")
    logger.info("")

    # Launch all processes
    processes = []

    for analysis_name, config in ANALYSES.items():
        result = launch_randomise(config, analysis_name)
        if result:
            processes.append(result)
        else:
            logger.error(f"Failed to launch {analysis_name}")

    if not processes:
        logger.error("No analyses launched successfully")
        sys.exit(1)

    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Launched {len(processes)} analyses in parallel")
    logger.info(f"{'='*80}")
    logger.info("")
    logger.info("PIDs:")
    for proc, _, name, _ in processes:
        logger.info(f"  {name}: {proc.pid}")
    logger.info("")
    logger.info("To monitor progress manually:")
    logger.info("  ps aux | grep randomise | grep -v grep")
    logger.info("")
    logger.info("Starting automated monitoring...")

    # Monitor until all complete
    try:
        monitor_processes(processes)

        logger.info("\n✅ All analyses completed successfully!")
        logger.info(f"\nNext steps:")
        logger.info("  1. Run cluster analysis: python batch_functional_cluster_analysis.py")
        logger.info("  2. View reports in /mnt/bytopia/IRC805/analysis/cluster_reports/")

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        logger.info("Processes are still running in background. PIDs:")
        for proc, _, name, _ in processes:
            if proc.poll() is None:
                logger.info(f"  {name}: {proc.pid}")
        logger.info("\nTo kill all: pkill randomise")


if __name__ == '__main__':
    main()
