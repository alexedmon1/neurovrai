#!/usr/bin/env python3
"""
Probabilistic tractography workflows using FSL probtrackx2.

This module provides functions to run tractography analysis using BEDPOSTX
output and atlas-based or FreeSurfer-based ROIs.

FreeSurfer Integration Status:
    **NOT PRODUCTION READY** - Detection hooks only, transform pipeline not implemented

    Current implementation:
    - Detects FreeSurfer outputs in SUBJECTS_DIR
    - Extracts ROIs from aparc+aseg.mgz parcellation
    - Falls back to atlas ROIs if FreeSurfer not available

    CRITICAL MISSING COMPONENTS:
    - ROIs extracted in FreeSurfer native space (NOT DWI space)
    - No anatomical→DWI transform pipeline implemented
    - Results will be INCORRECT if FreeSurfer integration is enabled

    DO NOT USE until transform pipeline is complete.

    Future work needed:
    1. Implement anatomical→DWI registration and transform extraction
    2. Warp FreeSurfer ROIs from anatomical to DWI space
    3. Validate transform accuracy with QC metrics
    4. Handle FreeSurfer native space vs. preprocessing T1 space
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from nipype import Workflow, Node
from nipype.interfaces import fsl

from mri_preprocess.utils.workflow import (
    setup_logging,
    get_node_config,
    get_execution_config
)
from mri_preprocess.utils.atlas_rois import prepare_probtrackx_rois
from mri_preprocess.utils.freesurfer_utils import (
    detect_freesurfer_subject,
    get_freesurfer_rois_for_tractography,
    check_freesurfer_availability
)


def run_probtrackx2_connectivity(
    bedpostx_dir: Path,
    seed_mask: Path,
    target_masks: Optional[List[Path]] = None,
    output_dir: Path = Path('./probtrackx_output'),
    brain_mask: Optional[Path] = None,
    n_samples: int = 5000,
    n_steps: int = 2000,
    step_length: float = 0.5,
    curvature_threshold: float = 0.2,
    waypoint_masks: Optional[List[Path]] = None,
    exclusion_mask: Optional[Path] = None,
    use_gpu: bool = True
) -> Dict[str, Path]:
    """
    Run FSL probtrackx2 for probabilistic tractography.

    This function performs probabilistic streamline tractography from seed
    regions to optional target regions using BEDPOSTX fiber orientation
    distributions.

    Parameters
    ----------
    bedpostx_dir : Path
        Directory containing BEDPOSTX outputs (.bedpostX directory)
    seed_mask : Path
        Binary mask defining seed region(s)
    target_masks : list of Path, optional
        List of binary target masks for connectivity analysis
    output_dir : Path
        Output directory for tractography results
    brain_mask : Path, optional
        Brain mask (if not provided, uses mask from BEDPOSTX)
    n_samples : int
        Number of streamlines per seed voxel (default: 5000)
    n_steps : int
        Maximum number of steps per streamline (default: 2000)
    step_length : float
        Step length in mm (default: 0.5)
    curvature_threshold : float
        Curvature threshold (default: 0.2, range: 0-1)
    waypoint_masks : list of Path, optional
        Waypoint masks that streamlines must pass through
    exclusion_mask : Path, optional
        Exclusion mask (terminate streamlines if entered)
    use_gpu : bool
        Use GPU-accelerated probtrackx2_gpu (default: True, highly recommended)

    Returns
    -------
    dict
        Dictionary with output file paths:
        - 'fdt_paths': Main tractography output (connectivity distribution)
        - 'waytotal': Total streamlines generated
        - 'matrix_seeds_to_all_targets': Connectivity matrix (if targets provided)

    Notes
    -----
    GPU acceleration provides dramatic speedups (10-50x faster) compared to CPU version.
    If use_gpu=True but probtrackx2_gpu is not available, will fall back to CPU version.

    Examples
    --------
    >>> results = run_probtrackx2_connectivity(
    ...     bedpostx_dir=Path('bedpostx_output.bedpostX'),
    ...     seed_mask=Path('hippocampus_l.nii.gz'),
    ...     target_masks=[Path('thalamus_l.nii.gz')],
    ...     output_dir=Path('tractography'),
    ...     use_gpu=True
    ... )
    """
    logger = logging.getLogger(__name__)

    # Determine which probtrackx to use
    probtrackx_cmd = 'probtrackx2_gpu' if use_gpu else 'probtrackx2'

    # Check if GPU version is available
    if use_gpu:
        import shutil
        if shutil.which('probtrackx2_gpu') is None:
            logger.warning("probtrackx2_gpu not found, falling back to CPU version")
            logger.warning("  For GPU support, ensure FSL GPU tools are installed")
            probtrackx_cmd = 'probtrackx2'
        else:
            logger.info("Running probtrackx2 with GPU acceleration")

    logger.info(f"Running {probtrackx_cmd} probabilistic tractography")

    bedpostx_dir = Path(bedpostx_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify BEDPOSTX directory
    if not bedpostx_dir.exists():
        raise FileNotFoundError(f"BEDPOSTX directory not found: {bedpostx_dir}")

    # Find merged_th1samples, merged_ph1samples, merged_f1samples
    required_files = ['merged_th1samples', 'merged_ph1samples', 'merged_f1samples']
    for fname in required_files:
        if not (bedpostx_dir / (fname + '.nii.gz')).exists():
            raise FileNotFoundError(
                f"Required BEDPOSTX file not found: {bedpostx_dir / fname}.nii.gz"
            )

    # Get brain mask
    if brain_mask is None:
        brain_mask = bedpostx_dir / 'nodif_brain_mask.nii.gz'
        if not brain_mask.exists():
            raise FileNotFoundError(
                f"Brain mask not found: {brain_mask}. "
                "Please provide brain_mask parameter."
            )

    # Build probtrackx2 command
    cmd = [
        probtrackx_cmd,
        '--samples=' + str(bedpostx_dir / 'merged'),
        '--mask=' + str(brain_mask),
        '--seed=' + str(seed_mask),
        '--dir=' + str(output_dir),
        '--nsamples=' + str(n_samples),
        '--nsteps=' + str(n_steps),
        '--steplength=' + str(step_length),
        '--cthr=' + str(curvature_threshold),
        '--forcedir',
        '--opd',  # Output path distribution
        '--os2t'  # Output seed-to-target pathways
    ]

    # Add target masks if provided
    if target_masks:
        # Create targets file
        targets_file = output_dir / 'targets.txt'
        with open(targets_file, 'w') as f:
            for target in target_masks:
                f.write(str(target) + '\n')
        cmd.extend([
            '--targetmasks=' + str(targets_file),
            '--omatrix1'  # Output connectivity matrix
        ])

    # Add waypoint masks if provided
    if waypoint_masks:
        waypoints_file = output_dir / 'waypoints.txt'
        with open(waypoints_file, 'w') as f:
            for waypoint in waypoint_masks:
                f.write(str(waypoint) + '\n')
        cmd.append('--waypoints=' + str(waypoints_file))

    # Add exclusion mask if provided
    if exclusion_mask:
        cmd.append('--avoid=' + str(exclusion_mask))

    # Run probtrackx2
    logger.info(f"  Seed: {seed_mask.name}")
    if target_masks:
        logger.info(f"  Targets: {len(target_masks)}")
    logger.info(f"  Samples per voxel: {n_samples}")
    logger.info(f"  Output: {output_dir}")

    import subprocess
    subprocess.run(cmd, check=True)

    # Collect outputs
    outputs = {
        'output_dir': output_dir,
        'fdt_paths': output_dir / 'fdt_paths.nii.gz',
        'waytotal': output_dir / 'waytotal',
    }

    if target_masks:
        outputs['matrix_seeds_to_all_targets'] = output_dir / 'matrix_seeds_to_all_targets'

    logger.info("Tractography complete!")
    if outputs['waytotal'].exists():
        with open(outputs['waytotal']) as f:
            waytotal = f.read().strip()
        logger.info(f"  Total streamlines: {waytotal}")

    return outputs


def run_seed_to_target_connectivity(
    bedpostx_dir: Path,
    seed_regions: Dict[str, Path],
    target_regions: Dict[str, Path],
    output_dir: Path,
    brain_mask: Optional[Path] = None,
    n_samples: int = 5000,
    use_gpu: bool = True
) -> Dict[str, Dict[str, Path]]:
    """
    Run connectivity analysis from multiple seeds to multiple targets.

    This is a convenience function that runs probtrackx2 for each seed-target
    pair and organizes results.

    Parameters
    ----------
    bedpostx_dir : Path
        BEDPOSTX output directory
    seed_regions : dict
        Dictionary of {region_name: mask_path} for seeds
    target_regions : dict
        Dictionary of {region_name: mask_path} for targets
    output_dir : Path
        Output directory
    brain_mask : Path, optional
        Brain mask
    n_samples : int
        Samples per voxel
    use_gpu : bool
        Use GPU acceleration (default: True)

    Returns
    -------
    dict
        Nested dictionary: {seed_name: {target_name: results_dict}}

    Examples
    --------
    >>> results = run_seed_to_target_connectivity(
    ...     bedpostx_dir=Path('bedpostx.bedpostX'),
    ...     seed_regions={'hippocampus_l': Path('hipp_l.nii.gz')},
    ...     target_regions={'thalamus_l': Path('thal_l.nii.gz')},
    ...     output_dir=Path('connectivity')
    ... )
    """
    logger = logging.getLogger(__name__)
    logger.info("Running seed-to-target connectivity analysis")
    logger.info(f"  Seeds: {len(seed_regions)}")
    logger.info(f"  Targets: {len(target_regions)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for seed_name, seed_mask in seed_regions.items():
        logger.info(f"\nProcessing seed: {seed_name}")
        results[seed_name] = {}

        # Run tractography for this seed to all targets
        seed_output_dir = output_dir / seed_name
        tract_results = run_probtrackx2_connectivity(
            bedpostx_dir=bedpostx_dir,
            seed_mask=seed_mask,
            target_masks=list(target_regions.values()),
            output_dir=seed_output_dir,
            brain_mask=brain_mask,
            n_samples=n_samples,
            use_gpu=use_gpu
        )

        results[seed_name] = tract_results

        # Parse connectivity matrix if available
        if 'matrix_seeds_to_all_targets' in tract_results:
            matrix_file = tract_results['matrix_seeds_to_all_targets']
            if matrix_file.exists():
                import numpy as np
                matrix = np.loadtxt(matrix_file)
                logger.info(f"  Connectivity matrix shape: {matrix.shape}")

                # Map to target names
                target_names = list(target_regions.keys())
                for i, target_name in enumerate(target_names):
                    if i < len(matrix):
                        results[seed_name][target_name] = float(matrix[i])
                        logger.info(f"    {seed_name} -> {target_name}: {matrix[i]:.0f} streamlines")

    logger.info("\nConnectivity analysis complete!")
    return results


def run_atlas_based_tractography(
    config: Dict[str, Any],
    subject: str,
    bedpostx_dir: Path,
    dwi_reference: Path,
    output_dir: Path,
    seed_regions: List[str],
    target_regions: Optional[List[str]] = None,
    atlas: str = 'HarvardOxford-subcortical',
    mni_to_dwi_warp: Optional[Path] = None,
    n_samples: int = 5000,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Complete atlas-based tractography workflow.

    This function:
    1. Warps atlas to DWI space
    2. Extracts seed and target ROIs
    3. Runs probtrackx2 connectivity analysis
    4. Organizes results

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    bedpostx_dir : Path
        BEDPOSTX output directory
    dwi_reference : Path
        Reference DWI volume (FA or b0)
    output_dir : Path
        Output directory
    seed_regions : list of str
        Seed region names from atlas
    target_regions : list of str, optional
        Target region names from atlas
    atlas : str
        Atlas name (default: 'HarvardOxford-subcortical')
    mni_to_dwi_warp : Path, optional
        MNI to DWI warp field
    n_samples : int
        Streamlines per voxel
    use_gpu : bool
        Use GPU acceleration (default: True)

    Returns
    -------
    dict
        Results dictionary with connectivity metrics

    Examples
    --------
    >>> results = run_atlas_based_tractography(
    ...     config=config,
    ...     subject='sub-001',
    ...     bedpostx_dir=Path('bedpostx.bedpostX'),
    ...     dwi_reference=Path('FA.nii.gz'),
    ...     output_dir=Path('tractography'),
    ...     seed_regions=['hippocampus_l', 'hippocampus_r'],
    ...     target_regions=['thalamus_l', 'thalamus_r']
    ... )
    """
    logger = logging.getLogger(__name__)
    log_dir = Path(config['paths']['logs'])
    logger = setup_logging(log_dir, subject, 'tractography')

    logger.info(f"Atlas-based tractography for {subject}")

    output_dir = Path(output_dir)
    roi_dir = output_dir / 'rois'

    # Check if FreeSurfer integration is enabled and available
    use_freesurfer = False
    if check_freesurfer_availability(config):
        fs_config = config.get('freesurfer', {})
        if fs_config.get('use_for_tractography', False):
            fs_dir = detect_freesurfer_subject(subject, config=config)
            if fs_dir:
                use_freesurfer = True
                logger.info("✓ FreeSurfer integration enabled - using FreeSurfer parcellations for ROIs")
            else:
                logger.info("FreeSurfer enabled but no outputs found for subject - using atlas ROIs")

    # Step 1: Prepare ROIs
    if use_freesurfer:
        logger.info("Step 1: Extracting FreeSurfer-based ROIs")
        # Get FreeSurfer subject directory
        fs_dir = detect_freesurfer_subject(subject, config=config)

        # Extract FreeSurfer ROIs
        all_regions = seed_regions + (target_regions if target_regions else [])
        fs_rois = get_freesurfer_rois_for_tractography(
            fs_subject_dir=fs_dir,
            roi_names=all_regions,
            output_dir=roi_dir
        )

        # TODO: Warp FreeSurfer ROIs from native anatomical space to DWI space
        # This requires anatomical→DWI transform from registration
        logger.warning("FreeSurfer ROI warping to DWI space not yet implemented")
        logger.warning("ROIs are in FreeSurfer native space - results may be inaccurate")

        # Split into seeds and targets
        rois = {
            'seeds': {k: v for k, v in fs_rois.items() if k in seed_regions},
            'targets': {k: v for k, v in fs_rois.items() if k in (target_regions or [])}
        }
    else:
        logger.info("Step 1: Preparing atlas-based ROIs")
        rois = prepare_probtrackx_rois(
            dwi_reference=dwi_reference,
            seed_regions=seed_regions,
            target_regions=target_regions,
            atlas=atlas,
            output_dir=roi_dir,
            mni_to_dwi_warp=mni_to_dwi_warp
        )

    # Step 2: Run connectivity analysis
    logger.info("Step 2: Running tractography")
    tract_dir = output_dir / 'tractography'
    connectivity_results = run_seed_to_target_connectivity(
        bedpostx_dir=bedpostx_dir,
        seed_regions=rois['seeds'],
        target_regions=rois['targets'] if rois['targets'] else {},
        output_dir=tract_dir,
        n_samples=n_samples,
        use_gpu=use_gpu
    )

    logger.info("Atlas-based tractography complete!")

    return {
        'rois': rois,
        'connectivity': connectivity_results,
        'output_dir': output_dir
    }
