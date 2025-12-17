#!/usr/bin/env python3
"""Run structural connectivity on all subjects with BEDPOSTX and FreeSurfer.

Usage:
    uv run python scripts/batch/run_structural_connectivity.py

Prerequisites:
    - BEDPOSTX outputs in {derivatives}/{subject}/dwi/bedpostx/
    - FreeSurfer outputs in {freesurfer_dir}/{subject}/
    - Study config.yaml with freesurfer.subjects_dir configured
"""

import sys
sys.path.insert(0, '/home/edm9fd/sandbox/neurovrai')

from pathlib import Path
import logging
from datetime import datetime
import yaml

# Set up logging
log_dir = Path('/home/edm9fd/sandbox/neurovrai/logs')
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f'structural_connectivity_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from neurovrai.connectome.structural_connectivity import compute_structural_connectivity
from neurovrai.connectome.atlas_dwi_transform import prepare_atlas_for_tractography

def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_ready_subjects(derivatives_dir: Path, freesurfer_dir: Path) -> list:
    """Find subjects that have both BEDPOSTX and FreeSurfer outputs."""
    # Get subjects with BEDPOSTX
    bedpostx_subjects = set()
    for bedpostx_path in derivatives_dir.glob('*/dwi/bedpostx'):
        if bedpostx_path.is_dir():
            subject = bedpostx_path.parent.parent.name
            bedpostx_subjects.add(subject)

    # Get subjects with FreeSurfer
    freesurfer_subjects = set()
    for fs_path in freesurfer_dir.glob('IRC805-*'):
        if fs_path.is_dir():
            freesurfer_subjects.add(fs_path.name)

    # Return intersection
    ready = sorted(bedpostx_subjects & freesurfer_subjects)
    return ready

def main():
    # Load study config
    config_path = Path('/mnt/bytopia/IRC805/config.yaml')
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Study paths
    study_root = Path('/mnt/bytopia/IRC805')
    derivatives_dir = study_root / 'derivatives'
    freesurfer_dir = Path(config.get('freesurfer', {}).get('subjects_dir', study_root / 'freesurfer'))
    connectome_dir = study_root / 'connectome' / 'structural'

    # Find ready subjects
    subjects = find_ready_subjects(derivatives_dir, freesurfer_dir)
    logger.info(f"Found {len(subjects)} subjects ready for structural connectivity")

    if not subjects:
        logger.error("No subjects found with both BEDPOSTX and FreeSurfer")
        sys.exit(1)

    # Track results
    results = {'completed': [], 'failed': [], 'skipped': []}

    for i, subject in enumerate(subjects, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {subject} ({i}/{len(subjects)})")
        logger.info(f"{'='*60}")

        # Check for existing output
        output_dir = connectome_dir / subject
        matrix_file = output_dir / 'connectivity_matrix.npy'

        if matrix_file.exists():
            logger.info(f"  Skipping {subject} - connectivity matrix already exists")
            results['skipped'].append(subject)
            continue

        # Set up paths
        bedpostx_dir = derivatives_dir / subject / 'dwi' / 'bedpostx'
        fs_subject_dir = freesurfer_dir / subject

        # Verify paths exist
        if not bedpostx_dir.exists():
            logger.error(f"  BEDPOSTX not found: {bedpostx_dir}")
            results['failed'].append((subject, "BEDPOSTX missing"))
            continue

        if not fs_subject_dir.exists():
            logger.error(f"  FreeSurfer not found: {fs_subject_dir}")
            results['failed'].append((subject, "FreeSurfer missing"))
            continue

        try:
            # Step 1: Prepare atlas in DWI space
            logger.info(f"  Preparing Desikan-Killiany atlas in DWI space...")
            atlas_dir = output_dir / 'atlas'
            atlas_dwi, roi_names, atlas_metadata = prepare_atlas_for_tractography(
                subject=subject,
                atlas_name='desikan_killiany',
                derivatives_dir=derivatives_dir,
                output_dir=atlas_dir,
                fs_subjects_dir=freesurfer_dir,
                config=config
            )
            logger.info(f"  Atlas prepared with {len(roi_names)} ROIs")

            # Step 2: Run structural connectivity
            logger.info(f"  Running probtrackx2 tractography...")
            sc_results = compute_structural_connectivity(
                bedpostx_dir=bedpostx_dir,
                atlas_file=atlas_dwi,
                output_dir=output_dir,
                subject=subject,
                derivatives_dir=derivatives_dir,
                fs_subjects_dir=freesurfer_dir,
                config=config,
                n_samples=5000,
                step_length=0.5,
                curvature_threshold=0.2,
                avoid_ventricles=True,
                use_gmwmi_seeding=True,
                batch_mode=True,
                use_gpu=True
            )

            logger.info(f"  Completed {subject}")
            logger.info(f"  Matrix shape: {sc_results['connectivity_matrix'].shape}")
            logger.info(f"  Total time: {sc_results.get('total_time', 'N/A')}")
            results['completed'].append(subject)

        except Exception as e:
            logger.error(f"  Failed {subject}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            results['failed'].append((subject, str(e)))

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Completed: {len(results['completed'])}")
    for s in results['completed']:
        logger.info(f"  - {s}")

    logger.info(f"Skipped (already done): {len(results['skipped'])}")
    for s in results['skipped']:
        logger.info(f"  - {s}")

    logger.info(f"Failed: {len(results['failed'])}")
    for s, reason in results['failed']:
        logger.info(f"  - {s}: {reason}")

    logger.info(f"\nTotal: {len(subjects)} subjects")
    logger.info(f"Log file: {log_file}")

if __name__ == '__main__':
    main()
