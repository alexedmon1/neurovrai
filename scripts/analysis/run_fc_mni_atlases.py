#!/usr/bin/env python3
"""
Batch Functional Connectivity Analysis using MNI Atlases

Computes FC matrices for all subjects using MNI-normalized BOLD data
and multiple standard MNI-space atlases.

Atlases:
- Schaefer 200, 400 parcels (Kong2022 17-network order)
- AAL3 (Automated Anatomical Labeling v3)

Usage:
    python run_fc_mni_atlases.py --study-root /mnt/bytopia/IRC805

Output:
    {study_root}/connectome/functional/{subject}/{atlas}/
        - fc_matrix.npy
        - fc_matrix.csv
        - fc_roi_names.txt
        - fc_summary.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurovrai.connectome.roi_extraction import extract_roi_timeseries, load_atlas
from neurovrai.connectome.functional_connectivity import compute_functional_connectivity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Atlas definitions - all in 2mm MNI space (91x109x91)
ATLASES = {
    'schaefer200': {
        'path': '/mnt/arborea/atlases/Schaefer2018/Parcellations_Kong2022_17network_order/MNI/Schaefer2018_200Parcels_Kong2022_17Networks_order_FSLMNI152_2mm.nii.gz',
        'labels': None,  # ROI names from atlas indices
        'description': 'Schaefer 200 parcels (Kong2022 17-network)'
    },
    'schaefer400': {
        'path': '/mnt/arborea/atlases/Schaefer2018/Parcellations_Kong2022_17network_order/MNI/Schaefer2018_400Parcels_Kong2022_17Networks_order_FSLMNI152_2mm.nii.gz',
        'labels': None,
        'description': 'Schaefer 400 parcels (Kong2022 17-network)'
    },
    'aal3': {
        'path': '/mnt/arborea/atlases/AAL3/AAL3.nii.gz',
        'labels': '/mnt/arborea/atlases/AAL3/AAL3.nii.txt',
        'description': 'AAL3 (Automated Anatomical Labeling v3)'
    }
}


def load_aal3_labels(labels_file: Path) -> Dict[int, str]:
    """Load AAL3 region labels from the text file."""
    labels = {}
    if labels_file.exists():
        with open(labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        # Format: index name [other info]
                        idx = int(parts[0])
                        name = parts[1]
                        labels[idx] = name
    return labels


def get_subjects(study_root: Path) -> List[str]:
    """Get list of subjects with MNI-normalized BOLD data."""
    subjects = []
    derivatives = study_root / 'derivatives'

    for subj_dir in sorted(derivatives.glob('IRC805-*')):
        bold_mni = subj_dir / 'func' / f'{subj_dir.name}_bold_mni.nii.gz'
        if bold_mni.exists():
            subjects.append(subj_dir.name)

    return subjects


def run_fc_for_subject(
    subject: str,
    bold_file: Path,
    atlas_name: str,
    atlas_path: Path,
    output_dir: Path,
    labels_file: Optional[Path] = None
) -> Dict:
    """
    Run functional connectivity analysis for a single subject and atlas.

    Returns:
        Dictionary with results and status
    """
    result = {
        'subject': subject,
        'atlas': atlas_name,
        'status': 'pending',
        'output_dir': str(output_dir),
        'error': None
    }

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already completed
        fc_matrix_file = output_dir / 'fc_matrix.npy'
        if fc_matrix_file.exists():
            logger.info(f"  {subject}/{atlas_name}: Already exists, skipping")
            result['status'] = 'skipped'
            return result

        logger.info(f"  {subject}/{atlas_name}: Extracting timeseries...")

        # Load atlas
        atlas = load_atlas(atlas_path, labels_file=labels_file)

        # Extract ROI timeseries
        timeseries, roi_names = extract_roi_timeseries(
            data_file=bold_file,
            atlas=atlas,
            min_voxels=5,  # Lower threshold for finer parcellations
            statistic='mean'
        )

        logger.info(f"  {subject}/{atlas_name}: Computing FC matrix ({len(roi_names)} ROIs)...")

        # Compute functional connectivity
        fc_results = compute_functional_connectivity(
            timeseries=timeseries,
            roi_names=roi_names,
            method='pearson',
            fisher_z=True,
            partial=False,
            threshold=None,
            output_dir=output_dir,
            output_prefix='fc'
        )

        result['status'] = 'success'
        result['n_rois'] = len(roi_names)
        result['n_timepoints'] = timeseries.shape[0]
        result['mean_fc'] = fc_results['summary']['mean_connectivity']

        logger.info(f"  {subject}/{atlas_name}: Done ({len(roi_names)} ROIs, mean FC={result['mean_fc']:.3f})")

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"  {subject}/{atlas_name}: ERROR - {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Batch FC analysis using MNI atlases'
    )
    parser.add_argument(
        '--study-root',
        type=Path,
        default=Path('/mnt/bytopia/IRC805'),
        help='Study root directory'
    )
    parser.add_argument(
        '--atlases',
        nargs='+',
        choices=list(ATLASES.keys()) + ['all'],
        default=['all'],
        help='Atlases to use (default: all)'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        default=None,
        help='Specific subjects to process (default: all with BOLD MNI data)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing outputs'
    )

    args = parser.parse_args()

    study_root = args.study_root

    # Determine atlases to use
    if 'all' in args.atlases:
        atlases_to_use = list(ATLASES.keys())
    else:
        atlases_to_use = args.atlases

    # Get subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = get_subjects(study_root)

    logger.info("=" * 70)
    logger.info("FUNCTIONAL CONNECTIVITY ANALYSIS - MNI ATLASES")
    logger.info("=" * 70)
    logger.info(f"Study root: {study_root}")
    logger.info(f"Subjects: {len(subjects)}")
    logger.info(f"Atlases: {', '.join(atlases_to_use)}")
    logger.info("=" * 70)

    if not subjects:
        logger.error("No subjects found with MNI-normalized BOLD data!")
        return 1

    # Track results
    all_results = []

    # Process each subject
    for i, subject in enumerate(subjects, 1):
        logger.info(f"\n[{i}/{len(subjects)}] {subject}")

        bold_file = study_root / 'derivatives' / subject / 'func' / f'{subject}_bold_mni.nii.gz'

        if not bold_file.exists():
            logger.warning(f"  BOLD MNI file not found: {bold_file}")
            continue

        # Process each atlas
        for atlas_name in atlases_to_use:
            atlas_info = ATLASES[atlas_name]
            atlas_path = Path(atlas_info['path'])
            labels_file = Path(atlas_info['labels']) if atlas_info['labels'] else None

            output_dir = study_root / 'connectome' / 'functional' / subject / atlas_name

            # Check if force overwrite
            if args.force and (output_dir / 'fc_matrix.npy').exists():
                import shutil
                shutil.rmtree(output_dir)

            result = run_fc_for_subject(
                subject=subject,
                bold_file=bold_file,
                atlas_name=atlas_name,
                atlas_path=atlas_path,
                output_dir=output_dir,
                labels_file=labels_file
            )

            all_results.append(result)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    for atlas_name in atlases_to_use:
        atlas_results = [r for r in all_results if r['atlas'] == atlas_name]
        success = sum(1 for r in atlas_results if r['status'] == 'success')
        skipped = sum(1 for r in atlas_results if r['status'] == 'skipped')
        errors = sum(1 for r in atlas_results if r['status'] == 'error')

        logger.info(f"{atlas_name}: {success} success, {skipped} skipped, {errors} errors")

    # Save log
    log_file = study_root / 'logs' / f'fc_mni_atlases_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'study_root': str(study_root),
            'atlases': atlases_to_use,
            'subjects': subjects,
            'results': all_results
        }, f, indent=2)

    logger.info(f"\nLog saved: {log_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
