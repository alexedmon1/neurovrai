#!/usr/bin/env python3
"""
Batch Structural Connectivity Analysis with Multiple Atlases

Process multiple subjects and atlases to generate structural connectomes
using probabilistic tractography (FSL probtrackx2).

Usage:
    uv run python neurovrai/connectome/batch_structural_connectivity.py \
        --study-root /mnt/bytopia/IRC805 \
        --atlases schaefer_200 desikan_killiany \
        --output-dir /mnt/bytopia/IRC805/connectome/structural
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurovrai.connectome.structural_connectivity import (
    StructuralConnectivityError,
    check_fsl_installation,
    run_bedpostx,
    validate_bedpostx_outputs,
    compute_structural_connectivity,
)
from neurovrai.connectome.atlas_dwi_transform import (
    ATLAS_CONFIGS,
    DWIAtlasTransformer,
    prepare_atlas_for_tractography,
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_sc_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    root_logger.handlers = []

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

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    root_logger.info(f"Log file: {log_file}")

    return root_logger


def find_subjects_with_dwi(study_root: Path) -> List[str]:
    """
    Find all subjects with preprocessed DWI data

    Args:
        study_root: Path to study root directory

    Returns:
        List of subject IDs with DWI preprocessing complete
    """
    derivatives_dir = study_root / "derivatives"

    subjects = []
    for subject_dir in sorted(derivatives_dir.glob("*")):
        if not subject_dir.is_dir():
            continue

        dwi_dir = subject_dir / "dwi"
        if not dwi_dir.exists():
            continue

        # Check for required DWI files
        eddy_file = dwi_dir / "dwi_eddy_corrected.nii.gz"
        mask_file = dwi_dir / "dwi_brain_mask.nii.gz"
        bval_file = dwi_dir / "dwi.bval"
        bvec_file = dwi_dir / "dwi_eddy_rotated.bvec"

        if eddy_file.exists() and mask_file.exists() and bval_file.exists() and bvec_file.exists():
            subjects.append(subject_dir.name)

    return subjects


def find_subjects_with_bedpostx(study_root: Path) -> List[str]:
    """
    Find all subjects with completed BEDPOSTX output

    Args:
        study_root: Path to study root directory

    Returns:
        List of subject IDs with BEDPOSTX complete
    """
    derivatives_dir = study_root / "derivatives"

    subjects = []
    for subject_dir in sorted(derivatives_dir.glob("*")):
        if not subject_dir.is_dir():
            continue

        # Check for BEDPOSTX output directory
        bedpostx_dir = subject_dir / "dwi.bedpostX"
        if not bedpostx_dir.exists():
            bedpostx_dir = subject_dir / "dwi" / "bedpostx"

        if not bedpostx_dir.exists():
            continue

        # Verify BEDPOSTX is complete
        dyads1 = bedpostx_dir / "dyads1.nii.gz"
        mean_f1 = bedpostx_dir / "mean_f1samples.nii.gz"

        if dyads1.exists() and mean_f1.exists():
            subjects.append(subject_dir.name)

    return subjects


def find_subjects_ready_for_tractography(study_root: Path) -> Tuple[List[str], List[str]]:
    """
    Categorize subjects by tractography readiness

    Args:
        study_root: Path to study root directory

    Returns:
        Tuple of (ready_for_tractography, need_bedpostx)
    """
    subjects_with_dwi = find_subjects_with_dwi(study_root)
    subjects_with_bedpostx = find_subjects_with_bedpostx(study_root)

    ready = subjects_with_bedpostx
    need_bedpostx = [s for s in subjects_with_dwi if s not in subjects_with_bedpostx]

    return ready, need_bedpostx


def prepare_bedpostx_input(
    subject: str,
    derivatives_dir: Path
) -> Path:
    """
    Prepare input directory for BEDPOSTX with required file names

    BEDPOSTX expects specific file names:
    - data.nii.gz (DWI data)
    - nodif_brain_mask.nii.gz (brain mask)
    - bvals (b-values)
    - bvecs (b-vectors)

    Args:
        subject: Subject ID
        derivatives_dir: Path to derivatives directory

    Returns:
        Path to prepared BEDPOSTX input directory
    """
    dwi_dir = derivatives_dir / subject / "dwi"

    # Check if BEDPOSTX input already prepared
    bedpostx_input = dwi_dir / "bedpostx_input"

    if bedpostx_input.exists():
        # Verify all files exist
        required = ["data.nii.gz", "nodif_brain_mask.nii.gz", "bvals", "bvecs"]
        all_exist = all((bedpostx_input / f).exists() for f in required)
        if all_exist:
            logger.info(f"  BEDPOSTX input already prepared: {bedpostx_input}")
            return bedpostx_input

    bedpostx_input.mkdir(parents=True, exist_ok=True)

    # Create symlinks with required names
    import os

    links = {
        "data.nii.gz": dwi_dir / "dwi_eddy_corrected.nii.gz",
        "nodif_brain_mask.nii.gz": dwi_dir / "dwi_brain_mask.nii.gz",
        "bvals": dwi_dir / "dwi.bval",
        "bvecs": dwi_dir / "dwi_eddy_rotated.bvec",
    }

    for link_name, source in links.items():
        link_path = bedpostx_input / link_name
        if link_path.exists():
            link_path.unlink()

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        os.symlink(source.resolve(), link_path)
        logger.debug(f"  Created symlink: {link_name} -> {source.name}")

    logger.info(f"  Prepared BEDPOSTX input: {bedpostx_input}")
    return bedpostx_input


def run_bedpostx_batch(
    subjects: List[str],
    study_root: Path,
    use_gpu: bool = True,
    n_fibers: int = 2,
    force: bool = False
) -> Dict[str, str]:
    """
    Run BEDPOSTX on multiple subjects

    Args:
        subjects: List of subject IDs
        study_root: Path to study root directory
        use_gpu: Use GPU acceleration (default: True)
        n_fibers: Number of fiber orientations to model (default: 2)
        force: Force rerun even if outputs exist (default: False)

    Returns:
        Dictionary mapping subject ID to status ('success', 'failed', 'skipped')
    """
    bedpostx_available, _, _ = check_fsl_installation()
    if not bedpostx_available:
        raise StructuralConnectivityError(
            "BEDPOSTX not found. Ensure FSL is installed and $FSLDIR is set."
        )

    derivatives_dir = study_root / "derivatives"
    results = {}

    for i, subject in enumerate(subjects, 1):
        logger.info(f"\n[{i}/{len(subjects)}] BEDPOSTX: {subject}")

        try:
            # Check if already completed
            bedpostx_output = derivatives_dir / subject / "dwi" / "bedpostx_input.bedpostX"
            if bedpostx_output.exists() and not force:
                dyads1 = bedpostx_output / "dyads1.nii.gz"
                if dyads1.exists():
                    logger.info(f"  Already completed, skipping (use --force to rerun)")
                    results[subject] = 'skipped'
                    continue

            # Prepare input directory
            bedpostx_input = prepare_bedpostx_input(subject, derivatives_dir)

            # Run BEDPOSTX
            output_dir = run_bedpostx(
                dwi_dir=bedpostx_input,
                n_fibers=n_fibers,
                use_gpu=use_gpu,
                force=force
            )

            logger.info(f"  BEDPOSTX completed: {output_dir}")
            results[subject] = 'success'

        except Exception as e:
            logger.error(f"  BEDPOSTX failed: {e}")
            results[subject] = 'failed'

    return results


def get_bedpostx_dir(subject: str, derivatives_dir: Path) -> Optional[Path]:
    """
    Find BEDPOSTX output directory for a subject

    Args:
        subject: Subject ID
        derivatives_dir: Path to derivatives directory

    Returns:
        Path to BEDPOSTX output directory, or None if not found
    """
    # Check common locations
    possible_paths = [
        derivatives_dir / subject / "dwi" / "bedpostx_input.bedpostX",
        derivatives_dir / subject / "dwi.bedpostX",
        derivatives_dir / subject / "dwi" / "bedpostx",
    ]

    for path in possible_paths:
        if path.exists():
            # Verify it's complete
            dyads1 = path / "dyads1.nii.gz"
            if dyads1.exists():
                return path

    return None


def process_subject_atlas_structural(
    subject: str,
    atlas_name: str,
    study_root: Path,
    output_dir: Path,
    fs_subjects_dir: Optional[Path] = None,
    n_samples: int = 5000,
    step_length: float = 0.5,
    curvature_threshold: float = 0.2,
    normalize: bool = True,
    threshold: Optional[float] = None,
    avoid_ventricles: bool = True,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Process structural connectivity for one subject with one atlas

    Args:
        subject: Subject ID
        atlas_name: Atlas name (from ATLAS_CONFIGS)
        study_root: Path to study root directory
        output_dir: Base output directory
        fs_subjects_dir: FreeSurfer subjects directory (for FS atlases)
        n_samples: Number of tractography samples per voxel
        step_length: Tractography step length in mm
        curvature_threshold: Curvature threshold for tractography
        normalize: Normalize connectivity by waytotal
        threshold: Optional threshold for weak connections
        avoid_ventricles: Exclude streamlines through ventricles (default: True)
        config: Optional config dictionary for all tractography settings

    Returns:
        Dictionary with results and status
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Subject: {subject} | Atlas: {atlas_name}")
    logger.info(f"{'='*80}")

    derivatives_dir = study_root / "derivatives"
    subject_atlas_dir = output_dir / atlas_name / subject

    try:
        # Create output directory
        subject_atlas_dir.mkdir(parents=True, exist_ok=True)

        # Check if already completed
        sc_matrix_file = subject_atlas_dir / "sc_matrix.npy"
        if sc_matrix_file.exists():
            logger.info(f"  Already completed: {sc_matrix_file}")
            # Load existing results
            sc_matrix = np.load(sc_matrix_file)
            metadata_file = subject_atlas_dir / "analysis_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                return {
                    'subject': subject,
                    'atlas': atlas_name,
                    'status': 'skipped',
                    'output_dir': str(subject_atlas_dir),
                    'n_rois': metadata.get('n_rois', sc_matrix.shape[0]),
                    'n_connections': metadata.get('n_connections', int(np.sum(sc_matrix > 0))),
                }
            return {
                'subject': subject,
                'atlas': atlas_name,
                'status': 'skipped',
                'output_dir': str(subject_atlas_dir),
            }

        # Step 1: Find BEDPOSTX directory
        bedpostx_dir = get_bedpostx_dir(subject, derivatives_dir)
        if bedpostx_dir is None:
            raise StructuralConnectivityError(
                f"BEDPOSTX not found for {subject}. Run BEDPOSTX first."
            )
        logger.info(f"  BEDPOSTX directory: {bedpostx_dir}")

        # Step 2: Transform atlas to DWI space
        logger.info(f"  Transforming atlas to DWI space...")

        atlas_dwi_file, roi_names, transform_info = prepare_atlas_for_tractography(
            subject=subject,
            atlas_name=atlas_name,
            derivatives_dir=derivatives_dir,
            output_dir=subject_atlas_dir,
            fs_subjects_dir=fs_subjects_dir,
        )

        logger.info(f"  Atlas in DWI space: {atlas_dwi_file.name}")
        logger.info(f"  Number of ROIs: {len(roi_names)}")

        # Step 3: Compute structural connectivity
        logger.info(f"  Running tractography (n_samples={n_samples})...")

        sc_results = compute_structural_connectivity(
            bedpostx_dir=bedpostx_dir,
            atlas_file=atlas_dwi_file,
            output_dir=subject_atlas_dir,
            n_samples=n_samples,
            step_length=step_length,
            curvature_threshold=curvature_threshold,
            normalize=normalize,
            threshold=threshold,
            avoid_ventricles=avoid_ventricles,
            subject=subject,
            derivatives_dir=derivatives_dir,
            fs_subjects_dir=fs_subjects_dir,
            config=config,
        )

        # Save connectivity matrix with standard name
        np.save(
            subject_atlas_dir / "sc_matrix.npy",
            sc_results['connectivity_matrix']
        )

        # Save ROI names
        with open(subject_atlas_dir / "sc_roi_names.txt", 'w') as f:
            for name in roi_names:
                f.write(f"{name}\n")

        # Save analysis metadata
        metadata = {
            'subject': subject,
            'atlas': atlas_name,
            'atlas_config': ATLAS_CONFIGS.get(atlas_name, {}),
            'bedpostx_dir': str(bedpostx_dir),
            'n_rois': len(roi_names),
            'n_samples': n_samples,
            'step_length': step_length,
            'curvature_threshold': curvature_threshold,
            'normalize': normalize,
            'threshold': threshold,
            'avoid_ventricles': avoid_ventricles,
            'n_connections': int(np.sum(sc_results['connectivity_matrix'] > 0)),
            'connection_density': float(
                np.sum(sc_results['connectivity_matrix'] > 0) /
                (len(roi_names) * (len(roi_names) - 1))
            ) if len(roi_names) > 1 else 0,
            'transform_info': transform_info,
            'analysis_date': datetime.now().isoformat(),
        }

        with open(subject_atlas_dir / 'analysis_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Success! Results saved to: {subject_atlas_dir}")
        logger.info(f"  Connections: {metadata['n_connections']}")
        logger.info(f"  Density: {metadata['connection_density']:.3f}")

        return {
            'subject': subject,
            'atlas': atlas_name,
            'status': 'success',
            'output_dir': str(subject_atlas_dir),
            'n_rois': len(roi_names),
            'n_connections': metadata['n_connections'],
            'connection_density': metadata['connection_density'],
        }

    except Exception as e:
        logger.error(f"  Failed: {str(e)}", exc_info=True)
        return {
            'subject': subject,
            'atlas': atlas_name,
            'status': 'failed',
            'error': str(e),
        }


def run_structural_connectivity_batch(
    study_root: Path,
    atlases: List[str],
    output_dir: Path,
    subjects: Optional[List[str]] = None,
    fs_subjects_dir: Optional[Path] = None,
    n_samples: int = 5000,
    step_length: float = 0.5,
    curvature_threshold: float = 0.2,
    normalize: bool = True,
    threshold: Optional[float] = None,
    skip_bedpostx_check: bool = False,
    avoid_ventricles: bool = True,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Run structural connectivity analysis for multiple subjects and atlases

    Args:
        study_root: Path to study root directory
        atlases: List of atlas names to process
        output_dir: Base output directory
        subjects: Optional list of specific subjects (default: all with BEDPOSTX)
        fs_subjects_dir: FreeSurfer subjects directory
        n_samples: Number of tractography samples per voxel
        step_length: Tractography step length in mm
        curvature_threshold: Curvature threshold for tractography
        normalize: Normalize connectivity by waytotal
        threshold: Optional threshold for weak connections
        skip_bedpostx_check: Skip BEDPOSTX requirement check
        avoid_ventricles: Exclude streamlines through ventricles (default: True)
        config: Optional config dictionary for all tractography settings

    Returns:
        Dictionary with batch results summary
    """
    # Find subjects
    if subjects is None:
        if skip_bedpostx_check:
            subjects = find_subjects_with_dwi(study_root)
        else:
            subjects = find_subjects_with_bedpostx(study_root)

    if not subjects:
        logger.warning("No subjects found for processing")
        return {'status': 'no_subjects', 'results': []}

    logger.info(f"\nProcessing {len(subjects)} subjects with {len(atlases)} atlases")
    logger.info(f"Subjects: {', '.join(subjects)}")
    logger.info(f"Atlases: {', '.join(atlases)}")

    total = len(subjects) * len(atlases)
    logger.info(f"Total analyses: {total}")

    results = []
    successful = 0
    failed = 0
    skipped = 0

    for i, subject in enumerate(subjects, 1):
        logger.info(f"\n[Subject {i}/{len(subjects)}] {subject}")

        for j, atlas_name in enumerate(atlases, 1):
            logger.info(f"  [Atlas {j}/{len(atlases)}] {atlas_name}")

            result = process_subject_atlas_structural(
                subject=subject,
                atlas_name=atlas_name,
                study_root=study_root,
                output_dir=output_dir,
                fs_subjects_dir=fs_subjects_dir,
                n_samples=n_samples,
                step_length=step_length,
                curvature_threshold=curvature_threshold,
                normalize=normalize,
                threshold=threshold,
                avoid_ventricles=avoid_ventricles,
                config=config,
            )

            results.append(result)

            if result['status'] == 'success':
                successful += 1
            elif result['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1

    summary = {
        'analysis_date': datetime.now().isoformat(),
        'study_root': str(study_root),
        'output_dir': str(output_dir),
        'total_analyses': total,
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'subjects': subjects,
        'atlases': atlases,
        'parameters': {
            'n_samples': n_samples,
            'step_length': step_length,
            'curvature_threshold': curvature_threshold,
            'normalize': normalize,
            'threshold': threshold,
            'avoid_ventricles': avoid_ventricles,
        },
        'results': results,
    }

    # Save summary
    summary_file = output_dir / 'batch_sc_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved: {summary_file}")

    return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Batch structural connectivity analysis with multiple atlases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all subjects with BEDPOSTX using Schaefer 200 and Desikan-Killiany
    uv run python neurovrai/connectome/batch_structural_connectivity.py \\
        --study-root /mnt/bytopia/IRC805 \\
        --atlases schaefer_200 desikan_killiany

    # Run BEDPOSTX first for subjects without it
    uv run python neurovrai/connectome/batch_structural_connectivity.py \\
        --study-root /mnt/bytopia/IRC805 \\
        --run-bedpostx --use-gpu

    # Process specific subjects
    uv run python neurovrai/connectome/batch_structural_connectivity.py \\
        --study-root /mnt/bytopia/IRC805 \\
        --subjects IRC805-0580101 IRC805-1720201 \\
        --atlases schaefer_200
        """
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
        help='Output directory (default: {study-root}/connectome/structural)'
    )

    parser.add_argument(
        '--atlases',
        nargs='+',
        choices=list(ATLAS_CONFIGS.keys()) + ['all'],
        default=['schaefer_200'],
        help='Atlases to use (default: schaefer_200)'
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help='Specific subjects to process (default: all with BEDPOSTX)'
    )

    parser.add_argument(
        '--fs-subjects-dir',
        type=Path,
        help='FreeSurfer subjects directory (required for FreeSurfer atlases)'
    )

    # BEDPOSTX options
    bedpostx_group = parser.add_argument_group('BEDPOSTX options')
    bedpostx_group.add_argument(
        '--run-bedpostx',
        action='store_true',
        help='Run BEDPOSTX on subjects without it'
    )
    bedpostx_group.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration for BEDPOSTX'
    )
    bedpostx_group.add_argument(
        '--n-fibers',
        type=int,
        default=2,
        help='Number of fiber orientations for BEDPOSTX (default: 2)'
    )

    # Tractography options
    tract_group = parser.add_argument_group('Tractography options')
    tract_group.add_argument(
        '--n-samples',
        type=int,
        default=5000,
        help='Number of samples per voxel (default: 5000)'
    )
    tract_group.add_argument(
        '--step-length',
        type=float,
        default=0.5,
        help='Step length in mm (default: 0.5)'
    )
    tract_group.add_argument(
        '--curvature-threshold',
        type=float,
        default=0.2,
        help='Curvature threshold (default: 0.2)'
    )
    tract_group.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable waytotal normalization'
    )
    tract_group.add_argument(
        '--threshold',
        type=float,
        help='Threshold for weak connections'
    )
    tract_group.add_argument(
        '--no-avoid-ventricles',
        action='store_true',
        help='Disable ventricle avoidance (NOT recommended - ventricles contain CSF, not white matter)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml file for tractography settings'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--status-only',
        action='store_true',
        help='Only show status, do not run processing'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.study_root / "connectome" / "structural"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log = setup_logging(args.output_dir, verbose=args.verbose)

    log.info("=" * 80)
    log.info("BATCH STRUCTURAL CONNECTIVITY ANALYSIS")
    log.info("=" * 80)
    log.info(f"Study root: {args.study_root}")
    log.info(f"Output directory: {args.output_dir}")

    # Load config file if provided
    config = None
    if args.config is not None and args.config.exists():
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        log.info(f"Loaded config from: {args.config}")

    # Check FSL installation
    bedpostx_available, probtrackx_available, probtrackx_gpu_available = check_fsl_installation()
    log.info(f"\nFSL Tools:")
    log.info(f"  BEDPOSTX: {'Available' if bedpostx_available else 'NOT FOUND'}")
    log.info(f"  Probtrackx2: {'Available' if probtrackx_available else 'NOT FOUND'}")
    log.info(f"  Probtrackx2 GPU: {'Available' if probtrackx_gpu_available else 'NOT FOUND'}")

    if not probtrackx_available:
        log.error("Probtrackx2 is required. Ensure FSL is installed.")
        sys.exit(1)

    # Check subject status
    ready, need_bedpostx = find_subjects_ready_for_tractography(args.study_root)
    subjects_with_dwi = find_subjects_with_dwi(args.study_root)

    log.info(f"\nSubject Status:")
    log.info(f"  Subjects with DWI: {len(subjects_with_dwi)}")
    log.info(f"  Ready for tractography (have BEDPOSTX): {len(ready)}")
    log.info(f"  Need BEDPOSTX: {len(need_bedpostx)}")

    if ready:
        log.info(f"\n  Ready subjects: {', '.join(ready)}")
    if need_bedpostx:
        log.info(f"\n  Subjects needing BEDPOSTX: {', '.join(need_bedpostx)}")

    if args.status_only:
        log.info("\n[Status only mode, exiting]")
        sys.exit(0)

    # Run BEDPOSTX if requested
    if args.run_bedpostx and need_bedpostx:
        log.info(f"\n{'='*80}")
        log.info("Running BEDPOSTX")
        log.info(f"{'='*80}")

        subjects_to_run = args.subjects if args.subjects else need_bedpostx
        subjects_to_run = [s for s in subjects_to_run if s in need_bedpostx]

        if subjects_to_run:
            bedpostx_results = run_bedpostx_batch(
                subjects=subjects_to_run,
                study_root=args.study_root,
                use_gpu=args.use_gpu,
                n_fibers=args.n_fibers,
            )

            # Update ready list
            ready, need_bedpostx = find_subjects_ready_for_tractography(args.study_root)
            log.info(f"\nAfter BEDPOSTX: {len(ready)} subjects ready")

    # Select atlases
    if 'all' in args.atlases:
        atlases = list(ATLAS_CONFIGS.keys())
    else:
        atlases = args.atlases

    # Filter atlases that require FreeSurfer
    if args.fs_subjects_dir is None:
        fs_atlases = [a for a in atlases if ATLAS_CONFIGS.get(a, {}).get('source') == 'freesurfer']
        if fs_atlases:
            log.warning(f"\nFreeSurfer atlases require --fs-subjects-dir: {fs_atlases}")
            atlases = [a for a in atlases if a not in fs_atlases]
            if not atlases:
                log.error("No valid atlases remaining after removing FreeSurfer atlases")
                sys.exit(1)

    log.info(f"\nAtlases to process: {', '.join(atlases)}")

    # Select subjects
    if args.subjects:
        subjects = [s for s in args.subjects if s in ready]
        if len(subjects) < len(args.subjects):
            missing = set(args.subjects) - set(subjects)
            log.warning(f"Subjects not ready (missing BEDPOSTX): {missing}")
    else:
        subjects = ready

    if not subjects:
        log.error("No subjects ready for processing. Run BEDPOSTX first.")
        sys.exit(1)

    # Run batch processing
    log.info(f"\n{'='*80}")
    log.info("Starting Batch Structural Connectivity")
    log.info(f"{'='*80}")

    summary = run_structural_connectivity_batch(
        study_root=args.study_root,
        atlases=atlases,
        output_dir=args.output_dir,
        subjects=subjects,
        fs_subjects_dir=args.fs_subjects_dir,
        n_samples=args.n_samples,
        step_length=args.step_length,
        curvature_threshold=args.curvature_threshold,
        normalize=not args.no_normalize,
        threshold=args.threshold,
        avoid_ventricles=not args.no_avoid_ventricles,
        config=config,
    )

    # Final summary
    log.info(f"\n{'='*80}")
    log.info("BATCH PROCESSING COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"Total analyses: {summary['total_analyses']}")
    log.info(f"Successful: {summary['successful']}")
    log.info(f"Skipped (already done): {summary['skipped']}")
    log.info(f"Failed: {summary['failed']}")

    if summary['total_analyses'] > 0:
        success_rate = (summary['successful'] + summary['skipped']) / summary['total_analyses'] * 100
        log.info(f"Success rate: {success_rate:.1f}%")

    log.info(f"\nResults directory: {args.output_dir}")
    log.info(f"Summary file: {args.output_dir / 'batch_sc_summary.json'}")

    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
