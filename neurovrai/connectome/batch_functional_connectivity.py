#!/usr/bin/env python3
"""
Batch Functional Connectivity Analysis with Multiple Atlases

Process multiple subjects and atlases to generate functional connectomes.

Usage:
    uv run python neurovrai/connectome/batch_functional_connectivity.py \
        --study-root /mnt/bytopia/IRC805 \
        --atlases schaefer200 schaefer400 aal harvardoxford \
        --output-dir /mnt/bytopia/IRC805/analysis/func/connectivity
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

from neurovrai.connectome.roi_extraction import extract_roi_timeseries, load_atlas
from neurovrai.connectome.functional_connectivity import compute_functional_connectivity


# Atlas definitions (MNI space, 2mm resolution for functional data)
ATLAS_DEFINITIONS = {
    'harvardoxford_cort': {
        'file': '/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz',
        'labels': None,  # XML format not supported by roi_extraction
        'description': 'Harvard-Oxford Cortical (48 regions)',
        'type': 'discrete'
    },
    'harvardoxford_sub': {
        'file': '/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz',
        'labels': None,  # XML format not supported by roi_extraction
        'description': 'Harvard-Oxford Subcortical (21 regions)',
        'type': 'discrete'
    },
    'juelich': {
        'file': '/usr/local/fsl/data/atlases/Juelich/Juelich-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'description': 'Juelich Histological Atlas',
        'type': 'discrete'
    },
    'talairach': {
        'file': '/usr/local/fsl/data/atlases/Talairach/Talairach-labels-2mm.nii.gz',
        'labels': None,
        'description': 'Talairach Atlas',
        'type': 'discrete'
    },
    'cerebellum_mniflirt': {
        'file': '/usr/local/fsl/data/atlases/Cerebellum/Cerebellum-MNIflirt-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'description': 'Cerebellum MNI FLIRT',
        'type': 'discrete'
    }
}


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_fc_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")

    return logger


def find_subjects(study_root: Path) -> List[str]:
    """Find all subjects with preprocessed functional data"""
    derivatives_dir = study_root / "derivatives"

    subjects = []
    for func_file in derivatives_dir.glob("*/func/*_bold_preprocessed.nii.gz"):
        subject = func_file.parts[-3]
        subjects.append(subject)

    subjects = sorted(list(set(subjects)))
    return subjects


def get_subject_files(study_root: Path, subject: str) -> Dict[str, Path]:
    """Get paths to subject's functional data files"""
    deriv_dir = study_root / "derivatives" / subject
    func_dir = deriv_dir / "func"

    files = {
        'func': func_dir / f"{subject}_bold_preprocessed.nii.gz",
        'mask': func_dir / "func_mask.nii.gz",
        'brain': func_dir / "func_brain.nii.gz",
    }

    # Validate files exist
    for key, filepath in files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"{key} file not found: {filepath}")

    return files


def process_subject_atlas(
    subject: str,
    files: Dict[str, Path],
    atlas_name: str,
    atlas_config: Dict,
    output_dir: Path,
    method: str = 'pearson',
    fisher_z: bool = True,
    logger: logging.Logger = None
) -> Dict:
    """
    Process one subject with one atlas

    Returns:
        Dictionary with results and status
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"\n{'='*80}")
    logger.info(f"Subject: {subject} | Atlas: {atlas_name}")
    logger.info(f"{'='*80}")

    try:
        # Create output directory
        subject_atlas_dir = output_dir / subject / atlas_name
        subject_atlas_dir.mkdir(parents=True, exist_ok=True)

        # Load functional data to get dimensions
        func_img = nib.load(files['func'])
        func_shape = func_img.shape[:3]
        logger.info(f"Functional data shape: {func_shape}")

        # Load and resample atlas to match functional data
        atlas_file = Path(atlas_config['file'])
        logger.info(f"Loading atlas: {atlas_file.name}")

        atlas_img = nib.load(atlas_file)
        atlas_shape = atlas_img.shape[:3]
        logger.info(f"Atlas original shape: {atlas_shape}")

        # Resample atlas to functional space
        if atlas_shape != func_shape:
            logger.info(f"Resampling atlas from {atlas_shape} to {func_shape}")
            from nilearn.image import resample_to_img

            # Resample atlas to match functional data (nearest neighbor for discrete labels)
            atlas_resampled = resample_to_img(
                atlas_img,
                func_img,
                interpolation='nearest'
            )

            # Save resampled atlas for reference
            resampled_file = subject_atlas_dir / f"atlas_{atlas_name}_resampled.nii.gz"
            nib.save(atlas_resampled, resampled_file)
            logger.info(f"Saved resampled atlas: {resampled_file.name}")

            # Load atlas from resampled version
            atlas = load_atlas(
                atlas_file=resampled_file,
                labels_file=atlas_config.get('labels')
            )
        else:
            atlas = load_atlas(
                atlas_file=atlas_file,
                labels_file=atlas_config.get('labels')
            )

        logger.info(f"Atlas: {atlas_config['description']}")
        logger.info(f"Number of ROIs: {atlas.n_rois}")

        # Extract ROI timeseries
        logger.info(f"Extracting ROI timeseries from: {files['func'].name}")

        timeseries, roi_names = extract_roi_timeseries(
            data_file=files['func'],
            atlas=atlas,
            mask_file=files['mask'],
            min_voxels=10,
            statistic='mean'
        )

        logger.info(f"Extracted timeseries shape: {timeseries.shape}")
        logger.info(f"Number of valid ROIs: {len(roi_names)}")

        # Compute functional connectivity
        logger.info(f"Computing functional connectivity (method={method})")

        fc_results = compute_functional_connectivity(
            timeseries=timeseries,
            roi_names=roi_names,
            method=method,
            fisher_z=fisher_z,
            partial=False,
            threshold=None,
            output_dir=subject_atlas_dir,
            output_prefix='fc'
        )

        # Save analysis metadata
        metadata = {
            'subject': subject,
            'atlas': atlas_name,
            'atlas_description': atlas_config['description'],
            'atlas_file': str(atlas_config['file']),
            'func_file': str(files['func']),
            'n_rois': len(roi_names),
            'n_timepoints': timeseries.shape[0],
            'method': method,
            'fisher_z': fisher_z,
            'analysis_date': datetime.now().isoformat(),
            'summary': fc_results['summary']
        }

        metadata_file = subject_atlas_dir / 'analysis_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Success! Results saved to: {subject_atlas_dir}")
        logger.info(f"  Mean connectivity: {fc_results['summary']['mean_connectivity']:.4f}")
        logger.info(f"  Non-zero edges: {fc_results['summary']['n_edges_nonzero']}")

        return {
            'subject': subject,
            'atlas': atlas_name,
            'status': 'success',
            'output_dir': str(subject_atlas_dir),
            'n_rois': len(roi_names),
            'mean_connectivity': fc_results['summary']['mean_connectivity']
        }

    except Exception as e:
        logger.error(f"✗ Failed: {str(e)}", exc_info=True)
        return {
            'subject': subject,
            'atlas': atlas_name,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Batch functional connectivity analysis with multiple atlases",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Path to study root directory'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (default: {study-root}/analysis/func/connectivity)'
    )

    parser.add_argument(
        '--atlases',
        nargs='+',
        choices=list(ATLAS_DEFINITIONS.keys()) + ['all'],
        default=['all'],
        help='Atlases to use (default: all)'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to process (default: all with preprocessed data)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='pearson',
        choices=['pearson', 'spearman'],
        help='Correlation method (default: pearson)'
    )

    parser.add_argument(
        '--no-fisher-z',
        action='store_true',
        help='Disable Fisher z-transformation'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.study_root / "analysis" / "func" / "connectivity"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.output_dir, verbose=args.verbose)

    logger.info("="*80)
    logger.info("BATCH FUNCTIONAL CONNECTIVITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Study root: {args.study_root}")
    logger.info(f"Output directory: {args.output_dir}")

    # Find subjects
    if args.subjects:
        subjects = args.subjects
        logger.info(f"Processing {len(subjects)} specified subjects")
    else:
        subjects = find_subjects(args.study_root)
        logger.info(f"Found {len(subjects)} subjects with preprocessed functional data")

    logger.info(f"Subjects: {', '.join(subjects)}")

    # Select atlases
    if 'all' in args.atlases:
        atlases = list(ATLAS_DEFINITIONS.keys())
    else:
        atlases = args.atlases

    logger.info(f"\nProcessing {len(atlases)} atlases:")
    for atlas_name in atlases:
        logger.info(f"  - {atlas_name}: {ATLAS_DEFINITIONS[atlas_name]['description']}")

    # Process all combinations
    total = len(subjects) * len(atlases)
    logger.info(f"\nTotal analyses to run: {total}")
    logger.info("="*80)

    results = []
    successful = 0
    failed = 0

    for i, subject in enumerate(subjects, 1):
        logger.info(f"\n[Subject {i}/{len(subjects)}] {subject}")

        try:
            # Get subject files
            files = get_subject_files(args.study_root, subject)
            logger.info(f"  Functional data: {files['func'].name}")

            # Process each atlas
            for j, atlas_name in enumerate(atlases, 1):
                logger.info(f"\n  [Atlas {j}/{len(atlases)}] {atlas_name}")

                result = process_subject_atlas(
                    subject=subject,
                    files=files,
                    atlas_name=atlas_name,
                    atlas_config=ATLAS_DEFINITIONS[atlas_name],
                    output_dir=args.output_dir,
                    method=args.method,
                    fisher_z=not args.no_fisher_z,
                    logger=logger
                )

                results.append(result)

                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1

        except Exception as e:
            logger.error(f"✗ Subject {subject} failed: {str(e)}", exc_info=True)
            for atlas_name in atlases:
                results.append({
                    'subject': subject,
                    'atlas': atlas_name,
                    'status': 'failed',
                    'error': str(e)
                })
                failed += 1

    # Save summary results
    summary_file = args.output_dir / 'batch_processing_summary.json'
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'study_root': str(args.study_root),
        'total_analyses': total,
        'successful': successful,
        'failed': failed,
        'subjects': subjects,
        'atlases': atlases,
        'method': args.method,
        'fisher_z': not args.no_fisher_z,
        'results': results
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total analyses: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/total*100:.1f}%")
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info(f"Results directory: {args.output_dir}")
    logger.info("="*80)

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
