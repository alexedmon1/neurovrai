#!/usr/bin/env python3
"""
T2-weighted (T2w) preprocessing workflow.

Workflow steps:
1. Reorientation to standard orientation (fslreorient2std)
2. Bias field correction (N4)
3. Skull stripping (BET)
4. Registration to T1w space (FLIRT)

The T2w→T1w transform is saved to the centralized transforms directory
for reuse by other workflows (T1-T2-ratio analysis, WMH analysis).

This workflow requires T1w preprocessing to be completed first, as it
needs the T1w brain image as the registration reference.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

from nipype import Workflow, Node
from nipype.interfaces import fsl, ants
from nipype.interfaces.io import DataSink
import nibabel as nib

from neurovrai.utils.workflow import (
    setup_logging,
    get_fsl_config,
    get_node_config,
    get_execution_config,
    validate_inputs
)
from neurovrai.utils.transforms import save_transform

logger = logging.getLogger(__name__)


# =============================================================================
# T2w File Discovery
# =============================================================================

def find_t2w_image(bids_dir: Path, subject: str) -> Optional[Path]:
    """
    Find the best T2w image for a subject from BIDS directory.

    Selection criteria (in order of preference):
    1. 3D T2W CS5 sequence (highest quality)
    2. T2W Sagittal (non-reformat)
    3. Any other T2W file

    Parameters
    ----------
    bids_dir : Path
        BIDS directory root
    subject : str
        Subject identifier

    Returns
    -------
    Path or None
        Path to T2w image, or None if not found
    """
    anat_dir = bids_dir / subject / 'anat'
    if not anat_dir.exists():
        logger.warning(f"BIDS anat directory not found: {anat_dir}")
        return None

    # Look for T2w files (case insensitive)
    t2w_files = list(anat_dir.glob('*T2W*.nii.gz')) + list(anat_dir.glob('*T2w*.nii.gz'))

    if not t2w_files:
        logger.warning(f"No T2w files found in {anat_dir}")
        return None

    # Prefer 3D T2W CS5 (highest quality)
    for f in t2w_files:
        if 'CS5' in f.name and 'Reformat' not in f.name:
            logger.info(f"Selected T2w (3D CS5): {f.name}")
            return f

    # Fall back to first non-reformat T2w
    for f in t2w_files:
        if 'Reformat' not in f.name:
            logger.info(f"Selected T2w: {f.name}")
            return f

    # Last resort: any T2w
    logger.info(f"Selected T2w (fallback): {t2w_files[0].name}")
    return t2w_files[0]


def find_t1w_brain(derivatives_dir: Path, subject: str) -> Optional[Path]:
    """
    Find the T1w brain image from preprocessing derivatives.

    Parameters
    ----------
    derivatives_dir : Path
        Derivatives directory root
    subject : str
        Subject identifier

    Returns
    -------
    Path or None
        Path to T1w brain image, or None if not found
    """
    anat_dir = derivatives_dir / subject / 'anat'

    # Check brain/ subdirectory first (Nipype DataSink structure)
    brain_dir = anat_dir / 'brain'
    if brain_dir.exists():
        brain_files = list(brain_dir.glob('*brain.nii.gz'))
        if brain_files:
            logger.info(f"Found T1w brain: {brain_files[0].name}")
            return brain_files[0]

    # Check for direct brain file
    for pattern in ['*_brain.nii.gz', 'brain.nii.gz', 'T1w_brain.nii.gz']:
        candidates = list(anat_dir.glob(pattern))
        if candidates:
            logger.info(f"Found T1w brain: {candidates[0].name}")
            return candidates[0]

    logger.warning(f"No T1w brain found in {anat_dir}")
    return None


# =============================================================================
# Preprocessing Node Creators
# =============================================================================

def create_reorient_node(name: str = 'reorient') -> Node:
    """Create a reorientation node using FSL fslreorient2std."""
    reorient = Node(
        fsl.Reorient2Std(output_type='NIFTI_GZ'),
        name=name
    )
    return reorient


def create_bias_correction_node(name: str = 'n4_bias_correction') -> Node:
    """
    Create bias correction node using ANTs N4BiasFieldCorrection.

    N4 is preferred over FAST for T2w images as it works better
    with different tissue contrasts.
    """
    n4 = Node(
        ants.N4BiasFieldCorrection(),
        name=name
    )
    n4.inputs.dimension = 3
    n4.inputs.n_iterations = [50, 50, 30, 20]
    n4.inputs.convergence_threshold = 1e-6
    n4.inputs.bspline_fitting_distance = 200
    n4.inputs.shrink_factor = 3
    return n4


def create_skull_strip_node(
    frac: float = 0.5,
    name: str = 'skull_strip'
) -> Node:
    """
    Create skull stripping node using FSL BET.

    Parameters
    ----------
    frac : float
        Fractional intensity threshold (0.3-0.7 typical for T2w).
        Higher values = more aggressive skull stripping.
    name : str
        Node name

    Notes
    -----
    T2w images may need different BET parameters than T1w.
    Using robust brain center estimation (-R) helps with T2w.
    """
    bet = Node(
        fsl.BET(
            frac=frac,
            robust=True,
            mask=True,
            output_type='NIFTI_GZ'
        ),
        name=name
    )
    return bet


# =============================================================================
# Registration Functions
# =============================================================================

def register_t2w_to_t1w(
    t2w_file: Path,
    t1w_brain: Path,
    output_file: Path,
    output_transform: Path,
    dof: int = 6
) -> Tuple[Path, Path]:
    """
    Co-register T2w image to T1w space using FSL FLIRT.

    Uses correlation ratio cost function (optimized for cross-modality).

    Parameters
    ----------
    t2w_file : Path
        Input T2w image (skull-stripped recommended)
    t1w_brain : Path
        T1w brain (skull-stripped) as reference
    output_file : Path
        Registered T2w in T1w space
    output_transform : Path
        Affine transformation matrix (.mat)
    dof : int
        Degrees of freedom (6 = rigid body, 12 = affine)

    Returns
    -------
    tuple
        (registered_t2w, transform_matrix)
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'flirt',
        '-in', str(t2w_file),
        '-ref', str(t1w_brain),
        '-out', str(output_file),
        '-omat', str(output_transform),
        '-dof', str(dof),
        '-cost', 'corratio',
        '-searchrx', '-90', '90',
        '-searchry', '-90', '90',
        '-searchrz', '-90', '90'
    ]

    logger.info("Running FLIRT: T2w -> T1w registration")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FLIRT failed: {result.stderr}")
        raise RuntimeError(f"FLIRT registration failed: {result.stderr}")

    logger.info(f"T2w registered to T1w: {output_file}")
    return output_file, output_transform


# =============================================================================
# Main Preprocessing Functions
# =============================================================================

def run_t2w_preprocessing(
    config: Dict[str, Any],
    subject: str,
    t2w_file: Path,
    t1w_brain: Path,
    output_dir: Path,
    work_dir: Optional[Path] = None,
    save_transforms: bool = True,
    bet_frac: float = 0.5,
    run_bias_correction: bool = True
) -> Dict[str, Path]:
    """
    Run T2w preprocessing pipeline.

    This is the main entry point for T2w preprocessing.
    It performs bias correction, skull stripping, and registration to T1w.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    t2w_file : Path
        Input T2w file
    t1w_brain : Path
        T1w brain-extracted image (reference for registration)
    output_dir : Path
        Study root directory (e.g., /mnt/bytopia/IRC805/)
        Derivatives will be saved to: {output_dir}/derivatives/{subject}/t2w/
    work_dir : Path, optional
        Working directory for temporary files
        Default: {output_dir}/work/{subject}/t2w_preprocess/
    save_transforms : bool
        Save transforms to centralized location (default: True)
    bet_frac : float
        BET fractional intensity threshold (default: 0.5)
    run_bias_correction : bool
        Run N4 bias correction (default: True)

    Returns
    -------
    dict
        Dictionary with output file paths:
        - 'reoriented': Reoriented T2w
        - 'bias_corrected': N4 bias-corrected T2w (if run_bias_correction=True)
        - 'brain': Skull-stripped T2w brain
        - 'brain_mask': Brain mask
        - 't2w_to_t1w': T2w registered to T1w space
        - 't2w_to_t1w_mat': Registration matrix

    Examples
    --------
    >>> from neurovrai.config import load_config
    >>> config = load_config("study.yaml")
    >>> results = run_t2w_preprocessing(
    ...     config=config,
    ...     subject="sub-001",
    ...     t2w_file=Path("/data/bids/sub-001/anat/sub-001_T2w.nii.gz"),
    ...     t1w_brain=Path("/data/derivatives/sub-001/anat/brain/T1w_brain.nii.gz"),
    ...     output_dir=Path("/data/study")
    ... )
    >>> print(results['t2w_to_t1w'])
    """
    # Setup directories first (needed for logging)
    study_root = Path(output_dir)
    derivatives_dir = study_root / 'derivatives' / subject / 't2w'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    if work_dir is None:
        work_dir = study_root / 'work' / subject / 't2w_preprocess'
    work_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_dir = study_root / 'logs' / subject
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, subject, 't2w_preprocess')

    logger.info("=" * 70)
    logger.info(f"T2w Preprocessing: {subject}")
    logger.info("=" * 70)

    # Validate inputs
    validate_inputs(t2w_file)
    validate_inputs(t1w_brain)

    outputs = {}

    # Step 1: Reorient to standard orientation
    logger.info("Step 1: Reorienting T2w to standard orientation...")
    reoriented_file = derivatives_dir / 'reoriented' / f'{subject}_t2w_reoriented.nii.gz'
    reoriented_file.parent.mkdir(parents=True, exist_ok=True)

    reorient_cmd = ['fslreorient2std', str(t2w_file), str(reoriented_file)]
    result = subprocess.run(reorient_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"fslreorient2std failed: {result.stderr}")

    outputs['reoriented'] = reoriented_file
    logger.info(f"  Reoriented: {reoriented_file}")

    # Step 2: N4 Bias Correction (optional)
    current_input = reoriented_file
    if run_bias_correction:
        logger.info("Step 2: Running N4 bias correction...")
        bias_corrected_file = derivatives_dir / 'bias_corrected' / f'{subject}_t2w_n4.nii.gz'
        bias_corrected_file.parent.mkdir(parents=True, exist_ok=True)

        n4_cmd = [
            'N4BiasFieldCorrection',
            '-d', '3',
            '-i', str(current_input),
            '-o', str(bias_corrected_file),
            '-s', '3',  # shrink factor
            '-c', '[50x50x30x20,1e-6]',  # convergence
            '-b', '[200]'  # bspline fitting distance
        ]
        result = subprocess.run(n4_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"N4 failed, continuing without bias correction: {result.stderr}")
        else:
            outputs['bias_corrected'] = bias_corrected_file
            current_input = bias_corrected_file
            logger.info(f"  Bias corrected: {bias_corrected_file}")

    # Step 3: Skull stripping
    logger.info("Step 3: Skull stripping with BET...")
    brain_file = derivatives_dir / 'brain' / f'{subject}_t2w_brain.nii.gz'
    brain_file.parent.mkdir(parents=True, exist_ok=True)
    mask_file = derivatives_dir / 'mask' / f'{subject}_t2w_brain_mask.nii.gz'
    mask_file.parent.mkdir(parents=True, exist_ok=True)

    bet_cmd = [
        'bet',
        str(current_input),
        str(brain_file.with_suffix('').with_suffix('')),  # Remove .nii.gz for BET output naming
        '-f', str(bet_frac),
        '-R',  # Robust brain center estimation
        '-m'   # Generate mask
    ]
    result = subprocess.run(bet_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"BET failed: {result.stderr}")

    # BET creates files with _mask suffix, move to proper location
    bet_output = brain_file.with_suffix('').with_suffix('.nii.gz')
    bet_mask = brain_file.with_suffix('').with_suffix('').with_name(
        brain_file.stem.replace('.nii', '') + '_mask.nii.gz'
    )

    # Handle BET output naming
    if bet_output.exists():
        outputs['brain'] = bet_output
    else:
        # BET may have created with different naming
        potential_brain = list(brain_file.parent.glob('*brain*.nii.gz'))
        if potential_brain:
            outputs['brain'] = potential_brain[0]

    potential_mask = list(brain_file.parent.glob('*mask*.nii.gz'))
    if potential_mask:
        import shutil
        shutil.move(str(potential_mask[0]), str(mask_file))
        outputs['brain_mask'] = mask_file

    logger.info(f"  Brain: {outputs.get('brain')}")
    logger.info(f"  Mask: {outputs.get('brain_mask')}")

    # Step 4: Register to T1w
    logger.info("Step 4: Registering T2w to T1w space...")
    registered_dir = derivatives_dir / 'registered'
    registered_dir.mkdir(parents=True, exist_ok=True)

    t2w_to_t1w = registered_dir / 't2w_to_t1w.nii.gz'
    t2w_to_t1w_mat = registered_dir / 't2w_to_t1w.mat'

    # Use the skull-stripped brain for registration
    t2w_brain_for_reg = outputs.get('brain', current_input)

    register_t2w_to_t1w(
        t2w_file=t2w_brain_for_reg,
        t1w_brain=t1w_brain,
        output_file=t2w_to_t1w,
        output_transform=t2w_to_t1w_mat
    )

    outputs['t2w_to_t1w'] = t2w_to_t1w
    outputs['t2w_to_t1w_mat'] = t2w_to_t1w_mat

    # Step 5: Save transform to centralized location
    if save_transforms:
        logger.info("Step 5: Saving transforms to centralized location...")
        save_transform(
            t2w_to_t1w_mat,
            study_root, subject, 't2w', 't1w', 'affine'
        )
        logger.info(f"  Saved: {study_root / 'transforms' / subject / 't2w-t1w-affine.mat'}")

    logger.info("=" * 70)
    logger.info("T2w Preprocessing Complete")
    logger.info("=" * 70)

    return outputs


def run_t2w_preprocessing_batch(
    config: Dict[str, Any],
    study_root: Path,
    subjects: Optional[List[str]] = None,
    n_jobs: int = 1,
    **kwargs
) -> Dict[str, Dict]:
    """
    Run T2w preprocessing on multiple subjects.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    study_root : Path
        Study root directory
    subjects : list, optional
        List of subject IDs. If None, auto-discovers subjects.
    n_jobs : int
        Number of parallel jobs (default: 1, sequential)
    **kwargs
        Additional arguments passed to run_t2w_preprocessing

    Returns
    -------
    dict
        Results dictionary keyed by subject ID
    """
    from joblib import Parallel, delayed

    bids_dir = study_root / 'bids'
    derivatives_dir = study_root / 'derivatives'

    # Discover subjects if not provided
    if subjects is None:
        subjects = discover_subjects_for_t2w(study_root)

    logger.info(f"Processing {len(subjects)} subjects for T2w preprocessing")

    results = {}
    failed = []

    def process_subject(subject: str) -> Tuple[str, Dict]:
        try:
            # Find T2w
            t2w_file = find_t2w_image(bids_dir, subject)
            if t2w_file is None:
                return subject, {'error': 'No T2w found'}

            # Find T1w brain
            t1w_brain = find_t1w_brain(derivatives_dir, subject)
            if t1w_brain is None:
                return subject, {'error': 'No T1w brain found - run T1w preprocessing first'}

            # Run preprocessing
            result = run_t2w_preprocessing(
                config=config,
                subject=subject,
                t2w_file=t2w_file,
                t1w_brain=t1w_brain,
                output_dir=study_root,
                **kwargs
            )
            return subject, result

        except Exception as e:
            logger.error(f"Failed to process {subject}: {e}")
            return subject, {'error': str(e)}

    if n_jobs == 1:
        # Sequential processing
        for subject in subjects:
            subj, result = process_subject(subject)
            results[subj] = result
            if 'error' in result:
                failed.append(subj)
    else:
        # Parallel processing
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(process_subject)(s) for s in subjects
        )
        for subj, result in parallel_results:
            results[subj] = result
            if 'error' in result:
                failed.append(subj)

    logger.info(f"Completed: {len(subjects) - len(failed)}/{len(subjects)} subjects")
    if failed:
        logger.warning(f"Failed subjects: {failed}")

    return results


def discover_subjects_for_t2w(study_root: Path) -> List[str]:
    """
    Discover subjects that have both T2w data and T1w preprocessing completed.

    Parameters
    ----------
    study_root : Path
        Study root directory

    Returns
    -------
    list
        List of subject IDs ready for T2w preprocessing
    """
    bids_dir = study_root / 'bids'
    derivatives_dir = study_root / 'derivatives'

    subjects = []

    # Find all subjects in BIDS
    if not bids_dir.exists():
        logger.warning(f"BIDS directory not found: {bids_dir}")
        return subjects

    for subject_dir in sorted(bids_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        # Check for T2w
        t2w = find_t2w_image(bids_dir, subject)
        if t2w is None:
            continue

        # Check for T1w brain (prerequisite)
        t1w_brain = find_t1w_brain(derivatives_dir, subject)
        if t1w_brain is None:
            logger.debug(f"Skipping {subject}: T1w preprocessing not complete")
            continue

        subjects.append(subject)

    logger.info(f"Found {len(subjects)} subjects ready for T2w preprocessing")
    return subjects


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == '__main__':
    import argparse
    from neurovrai.config import load_config

    parser = argparse.ArgumentParser(
        description='T2w Preprocessing Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Single subject
    single_parser = subparsers.add_parser('single', help='Process single subject')
    single_parser.add_argument('--subject', required=True, help='Subject ID')
    single_parser.add_argument('--config', required=True, help='Config file')
    single_parser.add_argument('--study-root', required=True, help='Study root directory')
    single_parser.add_argument('--bet-frac', type=float, default=0.5, help='BET fractional intensity')
    single_parser.add_argument('--no-bias-correction', action='store_true', help='Skip N4 bias correction')

    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple subjects')
    batch_parser.add_argument('--config', required=True, help='Config file')
    batch_parser.add_argument('--study-root', required=True, help='Study root directory')
    batch_parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    batch_parser.add_argument('--bet-frac', type=float, default=0.5, help='BET fractional intensity')

    args = parser.parse_args()

    if args.command == 'single':
        config = load_config(args.config)
        study_root = Path(args.study_root)

        # Find inputs
        t2w_file = find_t2w_image(study_root / 'bids', args.subject)
        t1w_brain = find_t1w_brain(study_root / 'derivatives', args.subject)

        if t2w_file is None:
            print(f"ERROR: No T2w file found for {args.subject}")
            exit(1)
        if t1w_brain is None:
            print(f"ERROR: No T1w brain found for {args.subject}. Run T1w preprocessing first.")
            exit(1)

        results = run_t2w_preprocessing(
            config=config,
            subject=args.subject,
            t2w_file=t2w_file,
            t1w_brain=t1w_brain,
            output_dir=study_root,
            bet_frac=args.bet_frac,
            run_bias_correction=not args.no_bias_correction
        )

        print("\nOutputs:")
        for key, path in results.items():
            print(f"  {key}: {path}")

    elif args.command == 'batch':
        config = load_config(args.config)
        study_root = Path(args.study_root)

        results = run_t2w_preprocessing_batch(
            config=config,
            study_root=study_root,
            n_jobs=args.n_jobs,
            bet_frac=args.bet_frac
        )

        print(f"\nProcessed {len(results)} subjects")
        for subject, result in results.items():
            status = "✓" if 'error' not in result else f"✗ {result['error']}"
            print(f"  {subject}: {status}")

    else:
        parser.print_help()
