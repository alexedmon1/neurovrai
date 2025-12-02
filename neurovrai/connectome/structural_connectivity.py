#!/usr/bin/env python3
"""
Structural Connectivity Module

Probabilistic tractography-based structural connectivity analysis using FSL probtrackx2.

This module provides:
- BEDPOSTX integration for fiber orientation modeling
- Probtrackx2 wrapper for probabilistic tractography
- ROI-to-ROI structural connectivity matrices
- Network construction from tractography results
- Tractography quality control metrics

Requirements:
- Completed DWI preprocessing (eddy correction, DTI fitting)
- Atlas parcellation in DWI space (or transformation to apply)
- FSL installed with probtrackx2 and bedpostx available

Workflow:
    1. Run BEDPOSTX on preprocessed DWI data (fiber orientation modeling)
    2. Prepare atlas/ROI masks in DWI space
    3. Run probtrackx2 in network mode (ROI-to-ROI tractography)
    4. Construct connectivity matrix from tractography outputs
    5. Threshold and analyze resulting structural network

Usage:
    # Step 1: Run BEDPOSTX
    bedpostx_dir = run_bedpostx(
        dwi_dir='derivatives/subject/dwi/',
        n_fibers=2,
        n_jumps=1250
    )

    # Step 2: Compute structural connectivity matrix
    sc_results = compute_structural_connectivity(
        bedpostx_dir=bedpostx_dir,
        atlas_file='parcellations/schaefer_400_dwi.nii.gz',
        output_dir='connectome/structural/',
        n_samples=5000
    )

    # Access connectivity matrix
    sc_matrix = sc_results['connectivity_matrix']
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StructuralConnectivityError(Exception):
    """Raised when structural connectivity analysis fails"""
    pass


def check_fsl_installation() -> Tuple[bool, bool]:
    """
    Check if FSL tools are available

    Returns:
        Tuple of (bedpostx_available, probtrackx2_available)
    """
    try:
        bedpostx_result = subprocess.run(
            ['which', 'bedpostx'],
            capture_output=True,
            text=True
        )
        bedpostx_available = bedpostx_result.returncode == 0
    except Exception:
        bedpostx_available = False

    try:
        probtrackx_result = subprocess.run(
            ['which', 'probtrackx2'],
            capture_output=True,
            text=True
        )
        probtrackx_available = probtrackx_result.returncode == 0
    except Exception:
        probtrackx_available = False

    return bedpostx_available, probtrackx_available


def validate_bedpostx_inputs(
    dwi_dir: Path
) -> Dict[str, Path]:
    """
    Validate that required BEDPOSTX input files exist

    Args:
        dwi_dir: Directory containing preprocessed DWI data

    Returns:
        Dictionary with paths to required files

    Raises:
        StructuralConnectivityError: If required files are missing
    """
    dwi_dir = Path(dwi_dir)

    if not dwi_dir.exists():
        raise StructuralConnectivityError(f"DWI directory not found: {dwi_dir}")

    # Required files for BEDPOSTX
    required_files = {
        'data': 'data.nii.gz',
        'nodif_brain_mask': 'nodif_brain_mask.nii.gz',
        'bvals': 'bvals',
        'bvecs': 'bvecs'
    }

    files = {}
    missing = []

    for key, filename in required_files.items():
        filepath = dwi_dir / filename
        if filepath.exists():
            files[key] = filepath
        else:
            missing.append(filename)

    if missing:
        raise StructuralConnectivityError(
            f"Missing required BEDPOSTX files in {dwi_dir}:\n"
            f"  {', '.join(missing)}\n\n"
            f"Required files: {', '.join(required_files.values())}"
        )

    return files


def run_bedpostx(
    dwi_dir: Path,
    output_dir: Optional[Path] = None,
    n_fibers: int = 2,
    n_jumps: int = 1250,
    burn_in: int = 1000,
    sample_every: int = 25,
    use_gpu: bool = False,
    force: bool = False
) -> Path:
    """
    Run BEDPOSTX for fiber orientation modeling

    BEDPOSTX performs Bayesian Estimation of Diffusion Parameters Obtained
    using Sampling Techniques, modeling crossing fibers in each voxel.

    Args:
        dwi_dir: Directory containing preprocessed DWI data with files:
            - data.nii.gz: Eddy-corrected DWI data
            - nodif_brain_mask.nii.gz: Brain mask
            - bvals: b-values
            - bvecs: b-vectors (eddy-rotated)
        output_dir: Output directory (default: {dwi_dir}.bedpostX)
        n_fibers: Number of fibers to model per voxel (default: 2)
        n_jumps: Number of MCMC jumps (default: 1250)
        burn_in: Burn-in period (default: 1000)
        sample_every: Sample every N iterations (default: 25)
        use_gpu: Use GPU acceleration if available (default: False)
        force: Overwrite existing BEDPOSTX output (default: False)

    Returns:
        Path to BEDPOSTX output directory

    Raises:
        StructuralConnectivityError: If BEDPOSTX fails or inputs invalid

    Note:
        BEDPOSTX can take several hours to complete. Monitor progress in
        {output_dir}/logs/. Use use_gpu=True for significant speedup if
        CUDA-enabled GPU is available.
    """
    bedpostx_available, _ = check_fsl_installation()
    if not bedpostx_available:
        raise StructuralConnectivityError(
            "bedpostx not found. Ensure FSL is installed and $FSLDIR is set."
        )

    dwi_dir = Path(dwi_dir)

    # Validate inputs
    input_files = validate_bedpostx_inputs(dwi_dir)

    # Determine output directory
    if output_dir is None:
        output_dir = dwi_dir.parent / f"{dwi_dir.name}.bedpostX"
    else:
        output_dir = Path(output_dir)

    # Check if already completed
    if output_dir.exists() and not force:
        # Check for completion marker or key output files
        dyads_file = output_dir / "dyads1.nii.gz"
        mean_f1_file = output_dir / "mean_f1samples.nii.gz"

        if dyads_file.exists() and mean_f1_file.exists():
            logger.info(f"BEDPOSTX output already exists: {output_dir}")
            logger.info("Use force=True to rerun")
            return output_dir
        else:
            logger.warning(f"Incomplete BEDPOSTX output found, will rerun")

    logger.info("=" * 80)
    logger.info("Running BEDPOSTX")
    logger.info("=" * 80)
    logger.info(f"Input directory: {dwi_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of fibers: {n_fibers}")
    logger.info(f"MCMC jumps: {n_jumps}")
    logger.info(f"Burn-in: {burn_in}")
    logger.info(f"GPU acceleration: {use_gpu}")

    # Build bedpostx command
    if use_gpu:
        cmd = ['bedpostx_gpu', str(dwi_dir)]
    else:
        cmd = ['bedpostx', str(dwi_dir)]

    # BEDPOSTX reads parameters from environment or uses defaults
    # We'll create a temporary options file
    env = {}
    if output_dir != dwi_dir.parent / f"{dwi_dir.name}.bedpostX":
        logger.warning(
            "Custom output_dir requires manual setup. "
            "BEDPOSTX creates {input_dir}.bedpostX by default."
        )

    # Execute BEDPOSTX
    start_time = time.time()
    log_file = dwi_dir.parent / "bedpostx.log"

    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info("This may take several hours. Monitor progress in logs/")

        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600

        logger.info(f"✓ BEDPOSTX completed in {elapsed_hours:.1f} hours")

    except subprocess.CalledProcessError as e:
        logger.error(f"BEDPOSTX failed with exit code {e.returncode}")
        logger.error(f"Check log file: {log_file}")
        raise StructuralConnectivityError(f"BEDPOSTX execution failed: {e}")

    # Verify outputs
    expected_output_dir = dwi_dir.parent / f"{dwi_dir.name}.bedpostX"
    if not expected_output_dir.exists():
        raise StructuralConnectivityError(
            f"BEDPOSTX output directory not created: {expected_output_dir}"
        )

    # Check key output files
    dyads_file = expected_output_dir / "dyads1.nii.gz"
    if not dyads_file.exists():
        raise StructuralConnectivityError(
            f"BEDPOSTX output incomplete. Missing: {dyads_file}"
        )

    logger.info(f"✓ BEDPOSTX outputs validated: {expected_output_dir}")

    return expected_output_dir


def validate_bedpostx_outputs(
    bedpostx_dir: Path,
    n_fibers: int = 2
) -> Dict[str, Path]:
    """
    Validate BEDPOSTX outputs exist and are complete

    Args:
        bedpostx_dir: Path to BEDPOSTX output directory
        n_fibers: Expected number of fiber compartments

    Returns:
        Dictionary with paths to key output files

    Raises:
        StructuralConnectivityError: If outputs are incomplete
    """
    bedpostx_dir = Path(bedpostx_dir)

    if not bedpostx_dir.exists():
        raise StructuralConnectivityError(
            f"BEDPOSTX directory not found: {bedpostx_dir}"
        )

    # Check for required files
    outputs = {
        'merged': bedpostx_dir / 'merged',
        'nodif_brain_mask': bedpostx_dir / 'nodif_brain_mask.nii.gz'
    }

    # Check fiber orientation files for each compartment
    for i in range(1, n_fibers + 1):
        outputs[f'dyads{i}'] = bedpostx_dir / f'dyads{i}.nii.gz'
        outputs[f'mean_f{i}samples'] = bedpostx_dir / f'mean_f{i}samples.nii.gz'
        outputs[f'mean_th{i}samples'] = bedpostx_dir / f'mean_th{i}samples.nii.gz'
        outputs[f'mean_ph{i}samples'] = bedpostx_dir / f'mean_ph{i}samples.nii.gz'

    missing = []
    for key, filepath in outputs.items():
        if not filepath.exists():
            missing.append(filepath.name)

    if missing:
        raise StructuralConnectivityError(
            f"Incomplete BEDPOSTX outputs in {bedpostx_dir}:\n"
            f"  Missing: {', '.join(missing)}\n\n"
            f"BEDPOSTX may still be running. Check logs/ directory."
        )

    return outputs


def prepare_atlas_for_probtrackx(
    atlas_file: Path,
    output_dir: Path,
    min_voxels_per_roi: int = 10
) -> Tuple[Path, List[str]]:
    """
    Prepare atlas for probtrackx2 network mode

    Creates individual ROI masks and coordinate list required by probtrackx2.

    Args:
        atlas_file: Path to atlas parcellation (integer labels)
        output_dir: Output directory for ROI masks
        min_voxels_per_roi: Minimum voxels per ROI (exclude smaller ROIs)

    Returns:
        Tuple of (seeds_list_file, roi_names)
        - seeds_list_file: Text file with paths to individual ROI masks
        - roi_names: List of ROI names/labels

    Raises:
        StructuralConnectivityError: If atlas is invalid
    """
    atlas_file = Path(atlas_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not atlas_file.exists():
        raise StructuralConnectivityError(f"Atlas file not found: {atlas_file}")

    logger.info(f"Preparing atlas for probtrackx2: {atlas_file.name}")

    # Load atlas
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata().astype(int)

    # Get unique ROI labels (exclude 0 = background)
    roi_labels = np.unique(atlas_data)
    roi_labels = roi_labels[roi_labels > 0]

    logger.info(f"  Found {len(roi_labels)} ROIs in atlas")

    # Create individual ROI masks
    roi_masks = []
    roi_names = []
    valid_labels = []

    for label in roi_labels:
        roi_mask = (atlas_data == label).astype(np.uint8)
        n_voxels = np.sum(roi_mask)

        if n_voxels < min_voxels_per_roi:
            logger.warning(f"  ROI {label}: only {n_voxels} voxels, excluding")
            continue

        # Save individual ROI mask
        roi_file = output_dir / f"roi_{label:03d}.nii.gz"
        roi_img = nib.Nifti1Image(roi_mask, atlas_img.affine, atlas_img.header)
        nib.save(roi_img, roi_file)

        roi_masks.append(roi_file)
        roi_names.append(f"ROI_{label:03d}")
        valid_labels.append(label)

        logger.info(f"  ROI {label}: {n_voxels} voxels -> {roi_file.name}")

    if len(roi_masks) == 0:
        raise StructuralConnectivityError(
            f"No valid ROIs found in atlas. Check min_voxels_per_roi parameter."
        )

    # Create seeds list file (required by probtrackx2 --network option)
    seeds_list_file = output_dir / "seeds.txt"
    with open(seeds_list_file, 'w') as f:
        for roi_file in roi_masks:
            f.write(f"{roi_file}\n")

    logger.info(f"✓ Created {len(roi_masks)} ROI masks")
    logger.info(f"✓ Seeds list: {seeds_list_file}")

    # Save ROI names for later use
    roi_names_file = output_dir / "roi_names.txt"
    with open(roi_names_file, 'w') as f:
        for name in roi_names:
            f.write(f"{name}\n")

    return seeds_list_file, roi_names


def run_probtrackx2_network(
    bedpostx_dir: Path,
    seeds_list: Path,
    output_dir: Path,
    n_samples: int = 5000,
    step_length: float = 0.5,
    curvature_threshold: float = 0.2,
    loop_check: bool = True,
    distance_correct: bool = True,
    waypoint_mask: Optional[Path] = None,
    exclusion_mask: Optional[Path] = None,
    termination_mask: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Run probtrackx2 in network mode for ROI-to-ROI tractography

    Args:
        bedpostx_dir: Path to BEDPOSTX output directory
        seeds_list: Path to seeds list file (from prepare_atlas_for_probtrackx)
        output_dir: Output directory for tractography results
        n_samples: Number of samples per seed voxel (default: 5000)
        step_length: Step length in mm (default: 0.5)
        curvature_threshold: Curvature threshold, 0-1 (default: 0.2)
        loop_check: Discard looping paths (default: True)
        distance_correct: Apply distance correction (default: True)
        waypoint_mask: Optional waypoint mask (e.g., white matter)
        exclusion_mask: Optional exclusion mask (e.g., CSF)
        termination_mask: Optional termination mask (e.g., grey matter)

    Returns:
        Dictionary with paths to output files

    Raises:
        StructuralConnectivityError: If probtrackx2 fails

    Note:
        Network mode runs tractography from each seed ROI to all target ROIs,
        creating an NxN connectivity matrix. This can take several hours for
        large atlases (e.g., Schaefer 400 parcellation).
    """
    _, probtrackx_available = check_fsl_installation()
    if not probtrackx_available:
        raise StructuralConnectivityError(
            "probtrackx2 not found. Ensure FSL is installed and $FSLDIR is set."
        )

    bedpostx_dir = Path(bedpostx_dir)
    seeds_list = Path(seeds_list)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate BEDPOSTX outputs
    bedpostx_outputs = validate_bedpostx_outputs(bedpostx_dir)

    # Validate seeds list
    if not seeds_list.exists():
        raise StructuralConnectivityError(f"Seeds list not found: {seeds_list}")

    # Count number of ROIs
    with open(seeds_list, 'r') as f:
        n_rois = len([line for line in f if line.strip()])

    logger.info("=" * 80)
    logger.info("Running Probtrackx2 (Network Mode)")
    logger.info("=" * 80)
    logger.info(f"BEDPOSTX directory: {bedpostx_dir}")
    logger.info(f"Seeds list: {seeds_list}")
    logger.info(f"Number of ROIs: {n_rois}")
    logger.info(f"Samples per voxel: {n_samples}")
    logger.info(f"Output directory: {output_dir}")

    # Build probtrackx2 command
    cmd = [
        'probtrackx2',
        '--samples', str(bedpostx_outputs['merged']),
        '--mask', str(bedpostx_outputs['nodif_brain_mask']),
        '--seed', str(seeds_list),
        '--network',  # Enable network mode
        '--dir', str(output_dir),
        '--nsamples', str(n_samples),
        '--steplength', str(step_length),
        '--curvthresh', str(curvature_threshold),
        '--opd',  # Output path distribution
        '--forcedir',  # Overwrite output directory
        '--os2t',  # Output seeds to targets
    ]

    # Optional parameters
    if loop_check:
        cmd.append('--loopcheck')

    if distance_correct:
        cmd.append('--distthresh=0.0')  # Enable distance correction

    if waypoint_mask is not None:
        if not Path(waypoint_mask).exists():
            raise StructuralConnectivityError(f"Waypoint mask not found: {waypoint_mask}")
        cmd.extend(['--waypoints', str(waypoint_mask)])

    if exclusion_mask is not None:
        if not Path(exclusion_mask).exists():
            raise StructuralConnectivityError(f"Exclusion mask not found: {exclusion_mask}")
        cmd.extend(['--avoid', str(exclusion_mask)])

    if termination_mask is not None:
        if not Path(termination_mask).exists():
            raise StructuralConnectivityError(f"Termination mask not found: {termination_mask}")
        cmd.extend(['--stop', str(termination_mask)])

    # Execute probtrackx2
    start_time = time.time()
    log_file = output_dir / "probtrackx2.log"

    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info(f"This may take several hours for {n_rois} ROIs...")

        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        elapsed = time.time() - start_time
        elapsed_hours = elapsed / 3600

        logger.info(f"✓ Probtrackx2 completed in {elapsed_hours:.2f} hours")

    except subprocess.CalledProcessError as e:
        logger.error(f"Probtrackx2 failed with exit code {e.returncode}")
        logger.error(f"Check log file: {log_file}")
        raise StructuralConnectivityError(f"Probtrackx2 execution failed: {e}")

    # Collect output files
    output_files = {
        'log': log_file,
        'fdt_network_matrix': output_dir / 'fdt_network_matrix',
        'waytotal': output_dir / 'waytotal',
        'fdt_paths': []
    }

    # Check for network matrix (key output)
    if not output_files['fdt_network_matrix'].exists():
        raise StructuralConnectivityError(
            f"Probtrackx2 output incomplete. Missing: fdt_network_matrix"
        )

    # Find individual seed-to-target path files
    for seed_dir in sorted(output_dir.glob('seeds_to_*')):
        output_files['fdt_paths'].append(seed_dir)

    logger.info(f"✓ Network matrix: {output_files['fdt_network_matrix']}")
    logger.info(f"✓ Found {len(output_files['fdt_paths'])} seed directories")

    return output_files


def construct_connectivity_matrix(
    probtrackx_output_dir: Path,
    roi_names: List[str],
    normalize: bool = True,
    threshold: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Construct structural connectivity matrix from probtrackx2 outputs

    Args:
        probtrackx_output_dir: Path to probtrackx2 output directory
        roi_names: List of ROI names (must match order in seeds list)
        normalize: Normalize by waytotal (default: True)
        threshold: Optional threshold for weak connections (0-1)

    Returns:
        Dictionary containing:
            - connectivity_matrix: Structural connectivity matrix (n_rois, n_rois)
            - waytotal: Number of successful samples per seed
            - connectivity_matrix_raw: Unnormalized matrix

    Raises:
        StructuralConnectivityError: If outputs are invalid
    """
    probtrackx_output_dir = Path(probtrackx_output_dir)

    # Load fdt_network_matrix (FSL format: space-separated)
    matrix_file = probtrackx_output_dir / 'fdt_network_matrix'
    if not matrix_file.exists():
        raise StructuralConnectivityError(
            f"Network matrix not found: {matrix_file}"
        )

    logger.info(f"Loading connectivity matrix: {matrix_file}")

    # Read matrix
    try:
        connectivity_raw = np.loadtxt(matrix_file)
    except Exception as e:
        raise StructuralConnectivityError(f"Failed to load network matrix: {e}")

    n_rois = len(roi_names)
    if connectivity_raw.shape != (n_rois, n_rois):
        raise StructuralConnectivityError(
            f"Matrix shape {connectivity_raw.shape} doesn't match "
            f"number of ROIs {n_rois}"
        )

    logger.info(f"  Matrix shape: {connectivity_raw.shape}")
    logger.info(f"  Total connections: {np.sum(connectivity_raw > 0)}")

    # Load waytotal (number of successful samples from each seed)
    waytotal_file = probtrackx_output_dir / 'waytotal'
    if waytotal_file.exists():
        waytotal = np.loadtxt(waytotal_file)
        logger.info(f"  Loaded waytotal: {len(waytotal)} seeds")
    else:
        logger.warning("waytotal file not found, normalization unavailable")
        waytotal = None
        normalize = False

    # Normalize by waytotal if requested
    if normalize and waytotal is not None:
        connectivity_norm = connectivity_raw.copy()
        for i in range(n_rois):
            if waytotal[i] > 0:
                connectivity_norm[i, :] /= waytotal[i]
            else:
                logger.warning(f"ROI {roi_names[i]}: waytotal = 0")

        logger.info("  Normalized by waytotal")
    else:
        connectivity_norm = connectivity_raw.copy()

    # Apply threshold if specified
    if threshold is not None:
        connections_before = np.sum(connectivity_norm > 0)
        connectivity_norm[connectivity_norm < threshold] = 0
        connections_after = np.sum(connectivity_norm > 0)

        logger.info(f"  Threshold: {threshold}")
        logger.info(f"  Connections: {connections_before} → {connections_after}")

    # Make symmetric (average i->j and j->i)
    connectivity_symmetric = (connectivity_norm + connectivity_norm.T) / 2

    logger.info(f"  Final connections: {np.sum(connectivity_symmetric > 0)}")
    logger.info(f"  Connection density: {np.sum(connectivity_symmetric > 0) / (n_rois * (n_rois - 1)):.3f}")

    return {
        'connectivity_matrix': connectivity_symmetric,
        'connectivity_matrix_raw': connectivity_raw,
        'connectivity_matrix_normalized': connectivity_norm,
        'waytotal': waytotal,
        'roi_names': roi_names
    }


def compute_structural_connectivity(
    bedpostx_dir: Path,
    atlas_file: Path,
    output_dir: Path,
    n_samples: int = 5000,
    step_length: float = 0.5,
    curvature_threshold: float = 0.2,
    normalize: bool = True,
    threshold: Optional[float] = None,
    min_voxels_per_roi: int = 10,
    waypoint_mask: Optional[Path] = None,
    exclusion_mask: Optional[Path] = None
) -> Dict:
    """
    Complete workflow: Compute structural connectivity matrix using probtrackx2

    This is the main function that orchestrates the full structural connectivity
    analysis pipeline.

    Args:
        bedpostx_dir: Path to BEDPOSTX output directory
        atlas_file: Path to atlas parcellation in DWI space
        output_dir: Output directory for all results
        n_samples: Number of samples per seed voxel (default: 5000)
        step_length: Tractography step length in mm (default: 0.5)
        curvature_threshold: Curvature threshold 0-1 (default: 0.2)
        normalize: Normalize by waytotal (default: True)
        threshold: Optional threshold for weak connections (default: None)
        min_voxels_per_roi: Minimum voxels per ROI (default: 10)
        waypoint_mask: Optional waypoint mask (e.g., white matter)
        exclusion_mask: Optional exclusion mask (e.g., CSF)

    Returns:
        Dictionary containing:
            - connectivity_matrix: Structural connectivity matrix
            - roi_names: List of ROI names
            - output_dir: Path to output directory
            - probtrackx_outputs: Dictionary of probtrackx output file paths
            - metadata: Analysis metadata

    Raises:
        StructuralConnectivityError: If any step fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STRUCTURAL CONNECTIVITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"BEDPOSTX directory: {bedpostx_dir}")
    logger.info(f"Atlas: {atlas_file}")
    logger.info(f"Output: {output_dir}")

    # Step 1: Validate BEDPOSTX outputs
    logger.info("\n[Step 1] Validating BEDPOSTX outputs...")
    bedpostx_outputs = validate_bedpostx_outputs(bedpostx_dir)
    logger.info("✓ BEDPOSTX outputs validated")

    # Step 2: Prepare atlas for probtrackx2
    logger.info("\n[Step 2] Preparing atlas for probtrackx2...")
    roi_dir = output_dir / 'roi_masks'
    seeds_list, roi_names = prepare_atlas_for_probtrackx(
        atlas_file=atlas_file,
        output_dir=roi_dir,
        min_voxels_per_roi=min_voxels_per_roi
    )
    logger.info(f"✓ Created {len(roi_names)} ROI masks")

    # Step 3: Run probtrackx2 in network mode
    logger.info("\n[Step 3] Running probtrackx2 network mode...")
    probtrackx_dir = output_dir / 'probtrackx_output'
    probtrackx_outputs = run_probtrackx2_network(
        bedpostx_dir=bedpostx_dir,
        seeds_list=seeds_list,
        output_dir=probtrackx_dir,
        n_samples=n_samples,
        step_length=step_length,
        curvature_threshold=curvature_threshold,
        waypoint_mask=waypoint_mask,
        exclusion_mask=exclusion_mask
    )
    logger.info("✓ Probtrackx2 completed")

    # Step 4: Construct connectivity matrix
    logger.info("\n[Step 4] Constructing connectivity matrix...")
    sc_results = construct_connectivity_matrix(
        probtrackx_output_dir=probtrackx_dir,
        roi_names=roi_names,
        normalize=normalize,
        threshold=threshold
    )
    logger.info("✓ Connectivity matrix constructed")

    # Save connectivity matrix and metadata
    logger.info("\n[Step 5] Saving results...")

    # Save connectivity matrix (NumPy format)
    np.save(
        output_dir / 'structural_connectivity_matrix.npy',
        sc_results['connectivity_matrix']
    )

    # Save as CSV for easy inspection
    sc_df = pd.DataFrame(
        sc_results['connectivity_matrix'],
        index=roi_names,
        columns=roi_names
    )
    sc_df.to_csv(output_dir / 'structural_connectivity_matrix.csv')

    # Save metadata
    metadata = {
        'n_rois': len(roi_names),
        'atlas_file': str(atlas_file),
        'bedpostx_dir': str(bedpostx_dir),
        'n_samples': n_samples,
        'step_length': step_length,
        'curvature_threshold': curvature_threshold,
        'normalized': normalize,
        'threshold': threshold,
        'n_connections': int(np.sum(sc_results['connectivity_matrix'] > 0)),
        'connection_density': float(np.sum(sc_results['connectivity_matrix'] > 0) / (len(roi_names) * (len(roi_names) - 1))),
        'roi_names': roi_names
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Connectivity matrix: {output_dir / 'structural_connectivity_matrix.npy'}")
    logger.info(f"✓ CSV export: {output_dir / 'structural_connectivity_matrix.csv'}")
    logger.info(f"✓ Metadata: {output_dir / 'metadata.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("STRUCTURAL CONNECTIVITY ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"ROIs: {len(roi_names)}")
    logger.info(f"Connections: {metadata['n_connections']}")
    logger.info(f"Density: {metadata['connection_density']:.3f}")

    return {
        'connectivity_matrix': sc_results['connectivity_matrix'],
        'roi_names': roi_names,
        'output_dir': str(output_dir),
        'probtrackx_outputs': {k: str(v) if not isinstance(v, list) else [str(x) for x in v]
                                for k, v in probtrackx_outputs.items()},
        'metadata': metadata
    }
