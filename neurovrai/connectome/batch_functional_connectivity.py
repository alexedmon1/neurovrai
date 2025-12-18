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
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

from neurovrai.connectome.roi_extraction import extract_roi_timeseries, load_atlas
from neurovrai.connectome.functional_connectivity import compute_functional_connectivity
from neurovrai.connectome.atlas_func_transform import (
    FuncAtlasTransformer,
    FUNC_ATLAS_CONFIGS,
    prepare_atlas_for_fc
)


# Atlas definitions (MNI space, 2mm resolution for functional data)
# Also includes FreeSurfer atlases that require transformation
ATLAS_DEFINITIONS = {
    # FSL MNI atlases (simple resampling)
    'harvardoxford_cort': {
        'file': '/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'description': 'Harvard-Oxford Cortical (48 regions)',
        'type': 'discrete',
        'space': 'MNI152'
    },
    'harvardoxford_sub': {
        'file': '/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'description': 'Harvard-Oxford Subcortical (21 regions)',
        'type': 'discrete',
        'space': 'MNI152'
    },
    'juelich': {
        'file': '/usr/local/fsl/data/atlases/Juelich/Juelich-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'description': 'Juelich Histological Atlas',
        'type': 'discrete',
        'space': 'MNI152'
    },
    'talairach': {
        'file': '/usr/local/fsl/data/atlases/Talairach/Talairach-labels-2mm.nii.gz',
        'labels': None,
        'description': 'Talairach Atlas',
        'type': 'discrete',
        'space': 'MNI152'
    },
    'cerebellum_mniflirt': {
        'file': '/usr/local/fsl/data/atlases/Cerebellum/Cerebellum-MNIflirt-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'description': 'Cerebellum MNI FLIRT',
        'type': 'discrete',
        'space': 'MNI152'
    },
    # Schaefer parcellations (MNI space)
    'schaefer_100': {
        'file': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz',
        'labels': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_100Parcels_7Networks_order.txt',
        'description': 'Schaefer 100 parcels (7 Networks)',
        'type': 'discrete',
        'space': 'MNI152'
    },
    'schaefer_200': {
        'file': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz',
        'labels': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_200Parcels_7Networks_order.txt',
        'description': 'Schaefer 200 parcels (7 Networks)',
        'type': 'discrete',
        'space': 'MNI152'
    },
    'schaefer_400': {
        'file': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz',
        'labels': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_400Parcels_7Networks_order.txt',
        'description': 'Schaefer 400 parcels (7 Networks)',
        'type': 'discrete',
        'space': 'MNI152'
    },
    # FreeSurfer atlases (require transformation from FS space)
    'desikan_killiany': {
        'source': 'freesurfer',
        'atlas_type': 'aparc+aseg',
        'description': 'Desikan-Killiany cortical (68) + subcortical',
        'type': 'discrete',
        'space': 'freesurfer'
    },
    'destrieux': {
        'source': 'freesurfer',
        'atlas_type': 'aparc.a2009s+aseg',
        'description': 'Destrieux cortical (148) + subcortical',
        'type': 'discrete',
        'space': 'freesurfer'
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


def generate_atlas_registration_qc(
    subject: str,
    atlas_file: Path,
    func_ref: Path,
    qc_dir: Path,
    atlas_name: str = 'atlas',
    logger: Optional[logging.Logger] = None
) -> Optional[Path]:
    """
    Generate QC visualization for atlas registration on functional data.

    Creates a 3x3 grid showing atlas overlay on functional data in
    axial, coronal, and sagittal views.

    Args:
        subject: Subject ID
        atlas_file: Path to atlas transformed to functional space
        func_ref: Path to functional reference (mean or brain)
        qc_dir: Base QC directory (will create {qc_dir}/{subject}/fc/)
        atlas_name: Name of atlas for output filename
        logger: Optional logger

    Returns:
        Path to saved QC image, or None if failed
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Create subject QC directory
        subject_qc_dir = qc_dir / subject / 'fc'
        subject_qc_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        atlas_img = nib.load(atlas_file)
        func_img = nib.load(func_ref)
        atlas_data = atlas_img.get_fdata()
        func_data = func_img.get_fdata()

        # Handle 4D functional data
        if func_data.ndim == 4:
            func_data = func_data.mean(axis=3)

        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f'Atlas Registration QC: {subject}\nAtlas: {atlas_name}', fontsize=14)

        # Axial slices
        z_slices = [atlas_data.shape[2]//4, atlas_data.shape[2]//2, 3*atlas_data.shape[2]//4]
        for i, z in enumerate(z_slices):
            ax = axes[0, i]
            ax.imshow(func_data[:,:,z].T, cmap='gray', origin='lower')
            masked = np.ma.masked_where(atlas_data[:,:,z] == 0, atlas_data[:,:,z])
            ax.imshow(masked.T, cmap='nipy_spectral', alpha=0.5, origin='lower')
            ax.set_title(f'Axial z={z}')
            ax.axis('off')

        # Coronal slices
        y_slices = [atlas_data.shape[1]//4, atlas_data.shape[1]//2, 3*atlas_data.shape[1]//4]
        for i, y in enumerate(y_slices):
            ax = axes[1, i]
            ax.imshow(func_data[:,y,:].T, cmap='gray', origin='lower')
            masked = np.ma.masked_where(atlas_data[:,y,:] == 0, atlas_data[:,y,:])
            ax.imshow(masked.T, cmap='nipy_spectral', alpha=0.5, origin='lower')
            ax.set_title(f'Coronal y={y}')
            ax.axis('off')

        # Sagittal slices
        x_slices = [atlas_data.shape[0]//4, atlas_data.shape[0]//2, 3*atlas_data.shape[0]//4]
        for i, x in enumerate(x_slices):
            ax = axes[2, i]
            ax.imshow(func_data[x,:,:].T, cmap='gray', origin='lower')
            masked = np.ma.masked_where(atlas_data[x,:,:] == 0, atlas_data[x,:,:])
            ax.imshow(masked.T, cmap='nipy_spectral', alpha=0.5, origin='lower')
            ax.set_title(f'Sagittal x={x}')
            ax.axis('off')

        plt.tight_layout()

        # Save QC image
        qc_file = subject_qc_dir / f'{subject}_{atlas_name}_registration.png'
        plt.savefig(qc_file, dpi=100, bbox_inches='tight')
        plt.close()

        logger.info(f"  QC image saved: {qc_file}")
        return qc_file

    except Exception as e:
        logger.warning(f"  QC generation failed: {e}")
        return None


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

    # Find functional file (try preprocessed, then bandpass filtered)
    func_file = func_dir / f"{subject}_bold_preprocessed.nii.gz"
    if not func_file.exists():
        func_file = func_dir / f"{subject}_bold_bandpass_filtered.nii.gz"

    # Find brain mask (variable naming convention)
    brain_dir = func_dir / "brain"
    mask_file = None
    if brain_dir.exists():
        mask_files = list(brain_dir.glob('*brain_mask.nii.gz'))
        if mask_files:
            mask_file = mask_files[0]

    # Fallback to standard names
    if mask_file is None:
        mask_file = func_dir / "func_mask.nii.gz"

    # Find brain file
    brain_file = func_dir / "func_brain.nii.gz"
    if not brain_file.exists() and brain_dir.exists():
        brain_files = list(brain_dir.glob('*brain.nii.gz'))
        if brain_files:
            brain_file = brain_files[0]

    files = {
        'func': func_file,
        'mask': mask_file,
        'brain': brain_file,
    }

    # Validate critical files exist
    if not files['func'].exists():
        raise FileNotFoundError(f"func file not found: {files['func']}")
    if not files['mask'].exists():
        raise FileNotFoundError(f"mask file not found: {files['mask']}")

    return files


def process_subject_atlas(
    subject: str,
    files: Dict[str, Path],
    atlas_name: str,
    atlas_config: Dict,
    output_dir: Path,
    derivatives_dir: Path,
    fs_subjects_dir: Optional[Path] = None,
    method: str = 'pearson',
    fisher_z: bool = True,
    qc_dir: Optional[Path] = None,
    run_qc: bool = True,
    logger: logging.Logger = None
) -> Dict:
    """
    Process one subject with one atlas

    Args:
        subject: Subject ID
        files: Dictionary with 'func', 'mask', 'brain' paths
        atlas_name: Name of atlas
        atlas_config: Atlas configuration dictionary
        output_dir: Output directory for results
        derivatives_dir: Path to derivatives directory (for transforms)
        fs_subjects_dir: FreeSurfer subjects directory (for FS atlases)
        method: Correlation method ('pearson' or 'spearman')
        fisher_z: Apply Fisher z-transformation
        qc_dir: QC output directory (default: None - skips QC)
        run_qc: Whether to generate QC images
        logger: Logger instance

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

        # Check if this is a FreeSurfer atlas
        is_freesurfer = atlas_config.get('space') == 'freesurfer' or atlas_config.get('source') == 'freesurfer'

        if is_freesurfer:
            # FreeSurfer atlas - requires transformation
            if fs_subjects_dir is None:
                raise ValueError(
                    f"Atlas '{atlas_name}' requires FreeSurfer. "
                    f"Provide --fs-subjects-dir argument."
                )

            logger.info(f"FreeSurfer atlas: transforming from FS space to functional space")

            # Use FuncAtlasTransformer for proper transformation
            transformer = FuncAtlasTransformer(
                subject=subject,
                derivatives_dir=derivatives_dir,
                fs_subjects_dir=fs_subjects_dir
            )

            # Check if transforms are available
            available = transformer.get_available_transforms()
            if not available['fs_to_func']:
                raise RuntimeError(
                    f"FreeSurfer→Func transform not available for {subject}. "
                    "Missing FreeSurfer outputs or functional registration transforms."
                )

            # Transform atlas to functional space
            resampled_file = subject_atlas_dir / f"atlas_{atlas_name}_in_func.nii.gz"
            intermediate_dir = subject_atlas_dir / 'intermediate'
            intermediate_dir.mkdir(parents=True, exist_ok=True)

            atlas_func, roi_names = transformer.transform_atlas_to_func(
                atlas_name=atlas_name,
                output_file=resampled_file,
                intermediate_dir=intermediate_dir
            )

            logger.info(f"Transformed atlas: {resampled_file.name}")
            logger.info(f"ROIs from FreeSurfer: {len(roi_names)}")

            # Load atlas using the transformed file
            atlas = load_atlas(
                atlas_file=resampled_file,
                labels_file=None  # ROI names from FreeSurfer labels
            )
            atlas_in_func_file = resampled_file

        else:
            # MNI atlas - simple resampling
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
                atlas_in_func_file = resampled_file
            else:
                atlas = load_atlas(
                    atlas_file=atlas_file,
                    labels_file=atlas_config.get('labels')
                )
                atlas_in_func_file = atlas_file  # Original atlas (already in right space)

            roi_names = None  # Will be extracted from atlas

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
            'atlas_file': str(atlas_config.get('file', 'freesurfer')),
            'atlas_source': atlas_config.get('source', 'unknown'),
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

        # Generate QC visualization
        qc_file = None
        if run_qc and qc_dir is not None:
            # Find functional reference for QC
            func_ref = files.get('brain') or files.get('mask')
            if func_ref and func_ref.exists():
                qc_file = generate_atlas_registration_qc(
                    subject=subject,
                    atlas_file=atlas_in_func_file,
                    func_ref=func_ref,
                    qc_dir=qc_dir,
                    atlas_name=atlas_name,
                    logger=logger
                )

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
        '--fs-subjects-dir',
        type=Path,
        help='FreeSurfer subjects directory (required for FreeSurfer atlases like desikan_killiany)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--qc-dir',
        type=Path,
        help='QC output directory (default: {study-root}/qc)'
    )

    parser.add_argument(
        '--no-qc',
        action='store_true',
        help='Skip QC image generation'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.study_root / "analysis" / "func" / "connectivity"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set QC directory
    if args.qc_dir is None:
        args.qc_dir = args.study_root / "qc"

    args.qc_dir.mkdir(parents=True, exist_ok=True)

    # Derivatives directory
    derivatives_dir = args.study_root / "derivatives"

    # Setup logging
    logger = setup_logging(args.output_dir, verbose=args.verbose)

    logger.info("="*80)
    logger.info("BATCH FUNCTIONAL CONNECTIVITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Study root: {args.study_root}")
    logger.info(f"Output directory: {args.output_dir}")
    if args.fs_subjects_dir:
        logger.info(f"FreeSurfer directory: {args.fs_subjects_dir}")

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

    # Filter out FreeSurfer atlases if fs_subjects_dir not provided
    freesurfer_atlases = [a for a in atlases if ATLAS_DEFINITIONS[a].get('space') == 'freesurfer']
    if freesurfer_atlases and not args.fs_subjects_dir:
        logger.warning(f"FreeSurfer atlases requested but --fs-subjects-dir not provided.")
        logger.warning(f"Skipping: {freesurfer_atlases}")
        atlases = [a for a in atlases if a not in freesurfer_atlases]

    if not atlases:
        logger.error("No valid atlases to process!")
        sys.exit(1)

    logger.info(f"\nProcessing {len(atlases)} atlases:")
    for atlas_name in atlases:
        atlas_type = "FreeSurfer" if ATLAS_DEFINITIONS[atlas_name].get('space') == 'freesurfer' else "MNI"
        logger.info(f"  - {atlas_name}: {ATLAS_DEFINITIONS[atlas_name]['description']} [{atlas_type}]")

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
                    derivatives_dir=derivatives_dir,
                    fs_subjects_dir=args.fs_subjects_dir,
                    method=args.method,
                    fisher_z=not args.no_fisher_z,
                    qc_dir=args.qc_dir,
                    run_qc=not args.no_qc,
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
