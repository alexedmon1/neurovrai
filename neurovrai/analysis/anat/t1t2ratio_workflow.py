#!/usr/bin/env python3
"""
T1w/T2w Ratio Analysis Workflow.

Computes T1w/T2w ratio maps for myelin content estimation in white matter,
following methodology from Du et al. 2019 (PMID: 30408230).

The T1w/T2w ratio is used as a parsimonious marker for tissue microstructure,
particularly myelin content in white matter.

Workflow:
1. Load preprocessed T1w and T2w images
2. Compute T1w/T2w ratio in native space
3. Normalize ratio map to MNI space
4. Create white matter mask
5. Apply WM mask and smooth for group analysis
6. Run group statistics (FSL randomise or nilearn GLM)

Prerequisites:
- T1w preprocessing completed (t1w_preprocess.py)
- T2w preprocessing completed (t2w_preprocess.py)
- T1w→MNI transform available

Reference:
    Du G, Lewis MM, Sica C, Kong L, Huang X. Magnetic resonance T1w/T2w ratio:
    A parsimonious marker for Parkinson disease. Ann Neurol. 2019 Jan;85(1):96-104.
    doi: 10.1002/ana.25376. PMID: 30408230.
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from neurovrai.utils.transforms import find_transform

logger = logging.getLogger(__name__)


# =============================================================================
# Input Discovery Functions
# =============================================================================

def find_t1w_bias_corrected(derivatives_dir: Path, subject: str) -> Optional[Path]:
    """Find T1w bias-corrected image from preprocessing derivatives."""
    anat_dir = derivatives_dir / subject / 'anat'

    # Check bias_corrected/ subdirectory
    bc_dir = anat_dir / 'bias_corrected'
    if bc_dir.exists():
        bc_files = list(bc_dir.glob('*.nii.gz'))
        if bc_files:
            return bc_files[0]

    # Fallback patterns
    for pattern in ['*_n4.nii.gz', '*bias_corrected*.nii.gz']:
        candidates = list(anat_dir.glob(pattern))
        if candidates:
            return candidates[0]

    # If no bias-corrected, use brain
    brain_dir = anat_dir / 'brain'
    if brain_dir.exists():
        brain_files = list(brain_dir.glob('*brain.nii.gz'))
        if brain_files:
            logger.warning(f"No bias-corrected T1w found, using brain: {brain_files[0]}")
            return brain_files[0]

    return None


def find_t2w_registered(derivatives_dir: Path, subject: str) -> Optional[Path]:
    """
    Find T2w image registered to T1w space.

    Checks current location ({derivatives}/{subject}/anat/t2w/registered/)
    and falls back to legacy location ({derivatives}/{subject}/t2w/registered/).
    """
    # Check current location: anat/t2w/registered/
    t2w_dir = derivatives_dir / subject / 'anat' / 't2w'
    reg_dir = t2w_dir / 'registered'
    if reg_dir.exists():
        reg_files = list(reg_dir.glob('t2w_to_t1w.nii.gz'))
        if reg_files:
            return reg_files[0]
        # Try other patterns
        reg_files = list(reg_dir.glob('*.nii.gz'))
        if reg_files:
            return reg_files[0]

    # Fallback: check legacy location {derivatives}/{subject}/t2w/registered/
    legacy_t2w_dir = derivatives_dir / subject / 't2w'
    legacy_reg_dir = legacy_t2w_dir / 'registered'
    if legacy_reg_dir.exists():
        reg_files = list(legacy_reg_dir.glob('t2w_to_t1w.nii.gz'))
        if reg_files:
            return reg_files[0]
        reg_files = list(legacy_reg_dir.glob('*.nii.gz'))
        if reg_files:
            return reg_files[0]

    return None


def find_wm_probability(derivatives_dir: Path, subject: str) -> Optional[Path]:
    """Find white matter probability map from segmentation."""
    seg_dir = derivatives_dir / subject / 'anat' / 'segmentation'

    if not seg_dir.exists():
        return None

    # Check for standardized naming first
    wm_file = seg_dir / 'wm.nii.gz'
    if wm_file.exists():
        return wm_file

    # Check for POSTERIOR_03 (typically WM in Atropos)
    posterior_wm = seg_dir / 'POSTERIOR_03.nii.gz'
    if posterior_wm.exists():
        return posterior_wm

    # Check for pve_2 (FSL FAST WM)
    pve_wm = seg_dir / 'pve_2.nii.gz'
    if pve_wm.exists():
        return pve_wm

    return None


def discover_subjects(study_root: Path) -> List[str]:
    """
    Discover subjects ready for T1-T2-ratio analysis.

    Requires:
    - T1w bias-corrected image
    - T2w registered to T1w
    - WM probability map
    - T1w→MNI transform
    """
    derivatives_dir = study_root / 'derivatives'
    transforms_dir = study_root / 'transforms'

    subjects = []

    if not derivatives_dir.exists():
        logger.warning(f"Derivatives directory not found: {derivatives_dir}")
        return subjects

    for subject_dir in sorted(derivatives_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name

        # Check all requirements
        t1w_bc = find_t1w_bias_corrected(derivatives_dir, subject)
        t2w_reg = find_t2w_registered(derivatives_dir, subject)
        wm_prob = find_wm_probability(derivatives_dir, subject)
        transform = find_transform(study_root, subject, 't1w', 'mni')

        if all([t1w_bc, t2w_reg, wm_prob, transform]):
            subjects.append(subject)
        else:
            missing = []
            if not t1w_bc:
                missing.append('T1w bias-corrected')
            if not t2w_reg:
                missing.append('T2w registered')
            if not wm_prob:
                missing.append('WM probability')
            if not transform:
                missing.append('T1w→MNI transform')
            logger.debug(f"Skipping {subject}: missing {', '.join(missing)}")

    logger.info(f"Found {len(subjects)} subjects ready for T1-T2-ratio analysis")
    return subjects


# =============================================================================
# Ratio Computation Functions
# =============================================================================

def compute_t1t2_ratio(
    t1w_file: Path,
    t2w_file: Path,
    output_file: Path,
    mask_file: Optional[Path] = None,
    min_t2w_threshold: float = 10.0
) -> Path:
    """
    Compute T1w/T2w ratio map.

    Parameters
    ----------
    t1w_file : Path
        T1w image (bias-corrected recommended)
    t2w_file : Path
        T2w image (registered to T1w space)
    output_file : Path
        Output ratio map
    mask_file : Path, optional
        Brain mask to apply
    min_t2w_threshold : float
        Minimum T2w value to avoid division by zero

    Returns
    -------
    Path
        Path to output ratio map
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load images
    t1w_img = nib.load(str(t1w_file))
    t2w_img = nib.load(str(t2w_file))

    t1w_data = t1w_img.get_fdata()
    t2w_data = t2w_img.get_fdata()

    # Handle shape mismatch
    if t1w_data.shape != t2w_data.shape:
        logger.warning(f"Shape mismatch: T1w {t1w_data.shape} vs T2w {t2w_data.shape}")
        # Resample T2w to T1w space if needed
        raise ValueError(f"T1w and T2w shapes don't match. Ensure T2w is registered to T1w space.")

    # Create mask for valid voxels (avoid division by zero)
    valid_mask = t2w_data > min_t2w_threshold

    # Apply brain mask if provided
    if mask_file is not None:
        brain_mask = nib.load(str(mask_file)).get_fdata() > 0
        valid_mask = valid_mask & brain_mask

    # Compute ratio
    ratio_data = np.zeros_like(t1w_data)
    ratio_data[valid_mask] = t1w_data[valid_mask] / t2w_data[valid_mask]

    # Save ratio map
    ratio_img = nib.Nifti1Image(ratio_data.astype(np.float32), t1w_img.affine, t1w_img.header)
    nib.save(ratio_img, str(output_file))

    logger.info(f"Computed T1w/T2w ratio: {output_file}")
    logger.info(f"  Ratio range: {ratio_data[valid_mask].min():.3f} - {ratio_data[valid_mask].max():.3f}")
    logger.info(f"  Ratio mean: {ratio_data[valid_mask].mean():.3f}")

    return output_file


def normalize_to_mni(
    input_file: Path,
    transform_file: Path,
    output_file: Path,
    reference: Optional[Path] = None,
    interpolation: str = 'BSpline'
) -> Path:
    """
    Normalize image to MNI space using existing transform.

    Parameters
    ----------
    input_file : Path
        Input image in native space
    transform_file : Path
        ANTs composite transform (.h5) or FSL warp field
    output_file : Path
        Output image in MNI space
    reference : Path, optional
        MNI reference image. Default: FSL MNI152 2mm template
    interpolation : str
        Interpolation method (BSpline, Linear, NearestNeighbor)

    Returns
    -------
    Path
        Path to normalized image
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Default reference
    if reference is None:
        fsl_dir = Path(subprocess.run(['echo', '$FSLDIR'], capture_output=True, text=True, shell=True).stdout.strip())
        if not fsl_dir.exists():
            fsl_dir = Path('/usr/local/fsl')
        reference = fsl_dir / 'data' / 'standard' / 'MNI152_T1_2mm_brain.nii.gz'

    # Detect transform type
    if str(transform_file).endswith('.h5'):
        # ANTs composite transform
        cmd = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(input_file),
            '-r', str(reference),
            '-o', str(output_file),
            '-n', interpolation,
            '-t', str(transform_file)
        ]
    else:
        # FSL warp field
        cmd = [
            'applywarp',
            '--in=' + str(input_file),
            '--ref=' + str(reference),
            '--warp=' + str(transform_file),
            '--out=' + str(output_file),
            '--interp=spline'
        ]

    logger.debug(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Normalization failed: {result.stderr}")

    logger.info(f"Normalized to MNI: {output_file}")
    return output_file


def create_wm_mask(
    wm_prob_file: Path,
    output_file: Path,
    threshold: float = 0.5
) -> Path:
    """
    Create binary WM mask from probability map.

    Parameters
    ----------
    wm_prob_file : Path
        WM probability map
    output_file : Path
        Output binary mask
    threshold : float
        Probability threshold (default: 0.5)

    Returns
    -------
    Path
        Path to binary mask
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    wm_img = nib.load(str(wm_prob_file))
    wm_data = wm_img.get_fdata()

    mask_data = (wm_data >= threshold).astype(np.int16)

    mask_img = nib.Nifti1Image(mask_data, wm_img.affine, wm_img.header)
    nib.save(mask_img, str(output_file))

    n_voxels = mask_data.sum()
    logger.info(f"Created WM mask: {output_file} ({n_voxels} voxels)")

    return output_file


def apply_mask_and_smooth(
    input_file: Path,
    mask_file: Path,
    output_file: Path,
    smooth_fwhm: float = 4.0
) -> Path:
    """
    Apply mask and smooth image.

    Parameters
    ----------
    input_file : Path
        Input image
    mask_file : Path
        Binary mask
    output_file : Path
        Output masked and smoothed image
    smooth_fwhm : float
        Smoothing kernel FWHM in mm (0 = no smoothing)

    Returns
    -------
    Path
        Path to output image
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Apply mask
    masked_file = output_file.with_name(output_file.stem + '_masked.nii.gz')

    cmd_mask = ['fslmaths', str(input_file), '-mas', str(mask_file), str(masked_file)]
    result = subprocess.run(cmd_mask, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Masking failed: {result.stderr}")

    # Smooth if requested
    if smooth_fwhm > 0:
        # Convert FWHM to sigma
        sigma = smooth_fwhm / (2 * np.sqrt(2 * np.log(2)))
        cmd_smooth = ['fslmaths', str(masked_file), '-s', str(sigma), str(output_file)]
        result = subprocess.run(cmd_smooth, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Smoothing failed: {result.stderr}")
        # Remove intermediate file
        masked_file.unlink()
    else:
        import shutil
        shutil.move(str(masked_file), str(output_file))

    logger.info(f"Applied mask and smoothing ({smooth_fwhm}mm): {output_file}")
    return output_file


# =============================================================================
# Main Workflow Functions
# =============================================================================

def prepare_t1t2ratio_single(
    subject: str,
    study_root: Path,
    output_dir: Path,
    wm_threshold: float = 0.5,
    smooth_fwhm: float = 4.0,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Prepare T1w/T2w ratio map for a single subject.

    Parameters
    ----------
    subject : str
        Subject identifier
    study_root : Path
        Study root directory
    output_dir : Path
        Output directory for ratio analysis
    wm_threshold : float
        WM probability threshold for mask (default: 0.5)
    smooth_fwhm : float
        Smoothing FWHM in mm (default: 4.0)
    overwrite : bool
        Overwrite existing outputs (default: False)

    Returns
    -------
    dict
        Results dictionary with output paths and metadata
    """
    logger.info(f"Processing T1-T2-ratio for {subject}")

    derivatives_dir = study_root / 'derivatives'
    subj_output = output_dir / subject
    subj_output.mkdir(parents=True, exist_ok=True)

    # Check for existing output
    final_output = subj_output / 't1t2_ratio_mni_wm_smooth.nii.gz'
    if final_output.exists() and not overwrite:
        logger.info(f"  Output exists, skipping: {final_output}")
        return {'subject': subject, 'status': 'skipped', 'output': final_output}

    # Find inputs
    t1w_bc = find_t1w_bias_corrected(derivatives_dir, subject)
    t2w_reg = find_t2w_registered(derivatives_dir, subject)
    wm_prob = find_wm_probability(derivatives_dir, subject)
    transform = find_transform(study_root, subject, 't1w', 'mni')

    if not all([t1w_bc, t2w_reg, wm_prob, transform]):
        missing = []
        if not t1w_bc:
            missing.append('T1w bias-corrected')
        if not t2w_reg:
            missing.append('T2w registered')
        if not wm_prob:
            missing.append('WM probability')
        if not transform:
            missing.append('T1w→MNI transform')
        error_msg = f"Missing inputs: {', '.join(missing)}"
        logger.error(f"  {error_msg}")
        return {'subject': subject, 'status': 'error', 'error': error_msg}

    logger.info(f"  T1w: {t1w_bc.name}")
    logger.info(f"  T2w: {t2w_reg.name}")
    logger.info(f"  WM prob: {wm_prob.name}")
    logger.info(f"  Transform: {transform.name}")

    try:
        # Step 1: Compute ratio in native space
        ratio_native = subj_output / 't1t2_ratio.nii.gz'
        compute_t1t2_ratio(t1w_bc, t2w_reg, ratio_native)

        # Step 2: Normalize ratio to MNI
        ratio_mni = subj_output / 't1t2_ratio_mni.nii.gz'
        normalize_to_mni(ratio_native, transform, ratio_mni)

        # Step 3: Normalize WM probability to MNI
        wm_prob_mni = subj_output / 'wm_prob_mni.nii.gz'
        normalize_to_mni(wm_prob, transform, wm_prob_mni)

        # Step 4: Create WM mask in MNI
        wm_mask_mni = subj_output / 'wm_mask_mni.nii.gz'
        create_wm_mask(wm_prob_mni, wm_mask_mni, threshold=wm_threshold)

        # Step 5: Mask and smooth ratio
        ratio_wm_smooth = subj_output / 't1t2_ratio_mni_wm_smooth.nii.gz'
        apply_mask_and_smooth(ratio_mni, wm_mask_mni, ratio_wm_smooth, smooth_fwhm=smooth_fwhm)

        # Save metrics
        ratio_img = nib.load(str(ratio_wm_smooth))
        ratio_data = ratio_img.get_fdata()
        valid_data = ratio_data[ratio_data > 0]

        metrics = {
            'subject': subject,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'wm_threshold': wm_threshold,
                'smooth_fwhm': smooth_fwhm
            },
            'inputs': {
                't1w_bc': str(t1w_bc),
                't2w_reg': str(t2w_reg),
                'wm_prob': str(wm_prob),
                'transform': str(transform)
            },
            'outputs': {
                'ratio_native': str(ratio_native),
                'ratio_mni': str(ratio_mni),
                'wm_mask_mni': str(wm_mask_mni),
                'ratio_wm_smooth': str(ratio_wm_smooth)
            },
            'statistics': {
                'n_wm_voxels': int((ratio_data > 0).sum()),
                'ratio_mean': float(valid_data.mean()) if len(valid_data) > 0 else None,
                'ratio_std': float(valid_data.std()) if len(valid_data) > 0 else None,
                'ratio_median': float(np.median(valid_data)) if len(valid_data) > 0 else None,
                'ratio_min': float(valid_data.min()) if len(valid_data) > 0 else None,
                'ratio_max': float(valid_data.max()) if len(valid_data) > 0 else None
            }
        }

        # Save metrics JSON
        metrics_file = subj_output / 't1t2ratio_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"  T1w/T2w ratio complete: mean={metrics['statistics']['ratio_mean']:.3f}")

        return metrics

    except Exception as e:
        logger.error(f"  Failed: {e}")
        return {'subject': subject, 'status': 'error', 'error': str(e)}


def prepare_t1t2ratio_batch(
    study_root: Path,
    output_dir: Path,
    subjects: Optional[List[str]] = None,
    n_jobs: int = 4,
    wm_threshold: float = 0.5,
    smooth_fwhm: float = 4.0,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Prepare T1w/T2w ratio maps for multiple subjects.

    Parameters
    ----------
    study_root : Path
        Study root directory
    output_dir : Path
        Output directory for ratio analysis
    subjects : list, optional
        List of subject IDs. If None, auto-discovers subjects.
    n_jobs : int
        Number of parallel jobs (default: 4)
    wm_threshold : float
        WM probability threshold for mask (default: 0.5)
    smooth_fwhm : float
        Smoothing FWHM in mm (default: 4.0)
    overwrite : bool
        Overwrite existing outputs (default: False)

    Returns
    -------
    dict
        Batch results summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects if not provided
    if subjects is None:
        subjects = discover_subjects(study_root)

    if not subjects:
        logger.warning("No subjects found for T1-T2-ratio analysis")
        return {'status': 'error', 'error': 'No subjects found'}

    logger.info(f"Processing {len(subjects)} subjects with {n_jobs} parallel jobs")

    # Process subjects
    results = Parallel(n_jobs=n_jobs)(
        delayed(prepare_t1t2ratio_single)(
            subject=s,
            study_root=study_root,
            output_dir=output_dir,
            wm_threshold=wm_threshold,
            smooth_fwhm=smooth_fwhm,
            overwrite=overwrite
        )
        for s in subjects
    )

    # Compile summary
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    skipped = [r for r in results if r.get('status') == 'skipped']

    summary = {
        'n_total': len(subjects),
        'n_successful': len(successful),
        'n_failed': len(failed),
        'n_skipped': len(skipped),
        'successful_subjects': [r['subject'] for r in successful],
        'failed_subjects': [(r['subject'], r.get('error', 'unknown')) for r in failed],
        'parameters': {
            'wm_threshold': wm_threshold,
            'smooth_fwhm': smooth_fwhm
        }
    }

    # Save batch summary
    summary_file = output_dir / 'batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Batch complete: {len(successful)}/{len(subjects)} successful")

    # Create group outputs if we have successful subjects
    if successful:
        generate_group_outputs(output_dir, [r['subject'] for r in successful])

    return summary


def generate_group_outputs(output_dir: Path, subjects: List[str]) -> Dict[str, Path]:
    """
    Generate group-level outputs for T1-T2-ratio analysis.

    Creates:
    - Merged 4D volume of all subjects
    - Group WM mask (intersection)
    - Summary CSV

    Parameters
    ----------
    output_dir : Path
        Output directory containing individual subject results
    subjects : list
        List of successful subject IDs

    Returns
    -------
    dict
        Paths to group outputs
    """
    group_dir = output_dir / 'group'
    group_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating group outputs...")

    # Collect subject files
    subject_files = []
    for subject in subjects:
        ratio_file = output_dir / subject / 't1t2_ratio_mni_wm_smooth.nii.gz'
        if ratio_file.exists():
            subject_files.append(str(ratio_file))

    if not subject_files:
        logger.warning("No subject files found for group outputs")
        return {}

    # Merge into 4D
    merged_file = group_dir / 'merged_t1t2ratio.nii.gz'
    cmd = ['fslmerge', '-t', str(merged_file)] + subject_files
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to merge: {result.stderr}")
    else:
        logger.info(f"Created merged 4D: {merged_file}")

    # Create group mask (mean > 0)
    group_mask = group_dir / 'group_wm_mask.nii.gz'
    cmd = ['fslmaths', str(merged_file), '-Tmean', '-thr', '0.01', '-bin', str(group_mask)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"Created group mask: {group_mask}")

    # Create summary CSV
    summary_data = []
    for subject in subjects:
        metrics_file = output_dir / subject / 't1t2ratio_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                stats = metrics.get('statistics', {})
                summary_data.append({
                    'subject': subject,
                    'n_wm_voxels': stats.get('n_wm_voxels'),
                    'ratio_mean': stats.get('ratio_mean'),
                    'ratio_std': stats.get('ratio_std'),
                    'ratio_median': stats.get('ratio_median'),
                    'ratio_min': stats.get('ratio_min'),
                    'ratio_max': stats.get('ratio_max')
                })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = group_dir / 't1t2ratio_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Created summary CSV: {summary_csv}")

    # Save subject list
    subject_list = group_dir / 'subject_list.txt'
    with open(subject_list, 'w') as f:
        for s in subjects:
            f.write(f"{s}\n")

    return {
        'merged': merged_file,
        'group_mask': group_mask,
        'summary_csv': group_dir / 't1t2ratio_summary.csv',
        'subject_list': subject_list
    }


def run_t1t2ratio_analysis(
    t1t2ratio_dir: Path,
    design_dir: Path,
    method: str = 'randomise',
    n_permutations: int = 5000,
    tfce: bool = True,
    cluster_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Run group-level statistical analysis on T1w/T2w ratio maps.

    Parameters
    ----------
    t1t2ratio_dir : Path
        Directory containing prepared ratio maps
    design_dir : Path
        Directory containing design matrix files
    method : str
        Statistical method: 'randomise', 'glm', or 'both'
    n_permutations : int
        Number of permutations for randomise (default: 5000)
    tfce : bool
        Use TFCE correction (default: True)
    cluster_threshold : float
        Cluster-forming threshold (default: 0.95 = p < 0.05)

    Returns
    -------
    dict
        Analysis results and output paths
    """
    from neurovrai.analysis.stats.randomise_wrapper import run_randomise
    from neurovrai.analysis.stats.enhanced_cluster_report import create_enhanced_cluster_report

    t1t2ratio_dir = Path(t1t2ratio_dir)
    design_dir = Path(design_dir)

    group_dir = t1t2ratio_dir / 'group'
    stats_dir = t1t2ratio_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Find merged file
    merged_file = group_dir / 'merged_t1t2ratio.nii.gz'
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_file}")

    # Find design files
    design_mat = design_dir / 'design.mat'
    design_con = design_dir / 'design.con'
    if not design_mat.exists() or not design_con.exists():
        raise FileNotFoundError(f"Design files not found in {design_dir}")

    # Find mask
    group_mask = group_dir / 'group_wm_mask.nii.gz'

    logger.info("Running T1-T2-ratio group analysis")
    logger.info(f"  Input: {merged_file}")
    logger.info(f"  Design: {design_dir}")
    logger.info(f"  Method: {method}")

    results = {}

    if method in ['randomise', 'both']:
        randomise_dir = stats_dir / 'randomise'
        randomise_dir.mkdir(parents=True, exist_ok=True)

        randomise_result = run_randomise(
            input_file=merged_file,
            design_mat=design_mat,
            contrast_con=design_con,
            output_dir=randomise_dir,
            mask=group_mask if group_mask.exists() else None,
            n_permutations=n_permutations,
            tfce=tfce
        )
        results['randomise'] = randomise_result

        # Generate cluster reports
        cluster_dir = stats_dir / 'cluster_reports'
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Find significant results
        for corrp_file in randomise_dir.glob('*_tfce_corrp_*.nii.gz'):
            contrast_name = corrp_file.stem.replace('_tfce_corrp_', '_')
            tstat_file = corrp_file.with_name(corrp_file.name.replace('_tfce_corrp_', '_'))

            if tstat_file.exists():
                try:
                    create_enhanced_cluster_report(
                        stat_map=tstat_file,
                        corrp_map=corrp_file,
                        threshold=cluster_threshold,
                        output_dir=cluster_dir,
                        contrast_name=contrast_name,
                        atlas_type='jhu'  # Use JHU atlas for white matter
                    )
                except Exception as e:
                    logger.warning(f"Failed to create cluster report: {e}")

    if method in ['glm', 'both']:
        from neurovrai.analysis.stats.nilearn_glm import run_second_level_glm

        glm_dir = stats_dir / 'glm'
        glm_dir.mkdir(parents=True, exist_ok=True)

        # Load design matrix
        participants_file = design_dir / 'participants_matched.tsv'
        if participants_file.exists():
            design_df = pd.read_csv(participants_file, sep='\t')
        else:
            logger.warning("participants_matched.tsv not found, GLM may fail")
            design_df = None

        if design_df is not None:
            # Get subject files
            subject_files = []
            for subject in design_df['participant_id']:
                ratio_file = t1t2ratio_dir / subject / 't1t2_ratio_mni_wm_smooth.nii.gz'
                if ratio_file.exists():
                    subject_files.append(ratio_file)

            if subject_files:
                glm_result = run_second_level_glm(
                    input_files=subject_files,
                    design_matrix=design_df,
                    contrasts={},  # Use default contrasts
                    output_dir=glm_dir,
                    mask=group_mask if group_mask.exists() else None
                )
                results['glm'] = glm_result

    logger.info("T1-T2-ratio analysis complete")
    return results


def generate_group_summary(t1t2ratio_dir: Path) -> pd.DataFrame:
    """
    Generate summary DataFrame from individual subject metrics.

    Parameters
    ----------
    t1t2ratio_dir : Path
        Directory containing T1-T2-ratio results

    Returns
    -------
    pd.DataFrame
        Summary statistics for all subjects
    """
    t1t2ratio_dir = Path(t1t2ratio_dir)
    summary_data = []

    for subject_dir in sorted(t1t2ratio_dir.iterdir()):
        if not subject_dir.is_dir() or subject_dir.name in ['group', 'stats']:
            continue

        metrics_file = subject_dir / 't1t2ratio_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                stats = metrics.get('statistics', {})
                summary_data.append({
                    'subject': subject_dir.name,
                    'status': metrics.get('status', 'unknown'),
                    'n_wm_voxels': stats.get('n_wm_voxels'),
                    'ratio_mean': stats.get('ratio_mean'),
                    'ratio_std': stats.get('ratio_std'),
                    'ratio_median': stats.get('ratio_median'),
                    'ratio_min': stats.get('ratio_min'),
                    'ratio_max': stats.get('ratio_max')
                })

    return pd.DataFrame(summary_data)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='T1w/T2w Ratio Analysis Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare single subject
  python -m neurovrai.analysis.anat.t1t2ratio_workflow prepare \\
      --subject sub-001 \\
      --study-root /data/study \\
      --output-dir /data/study/analysis/t1t2ratio

  # Batch preparation
  python -m neurovrai.analysis.anat.t1t2ratio_workflow batch \\
      --study-root /data/study \\
      --output-dir /data/study/analysis/t1t2ratio \\
      --n-jobs 4

  # Run group statistics
  python -m neurovrai.analysis.anat.t1t2ratio_workflow analyze \\
      --t1t2ratio-dir /data/study/analysis/t1t2ratio \\
      --design-dir /data/study/designs/t1t2ratio \\
      --method randomise
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Prepare single subject
    prep_parser = subparsers.add_parser('prepare', help='Prepare single subject')
    prep_parser.add_argument('--subject', required=True, help='Subject ID')
    prep_parser.add_argument('--study-root', required=True, help='Study root directory')
    prep_parser.add_argument('--output-dir', required=True, help='Output directory')
    prep_parser.add_argument('--wm-threshold', type=float, default=0.5, help='WM threshold')
    prep_parser.add_argument('--smooth-fwhm', type=float, default=4.0, help='Smoothing FWHM')
    prep_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing')

    # Batch preparation
    batch_parser = subparsers.add_parser('batch', help='Batch preparation')
    batch_parser.add_argument('--study-root', required=True, help='Study root directory')
    batch_parser.add_argument('--output-dir', required=True, help='Output directory')
    batch_parser.add_argument('--n-jobs', type=int, default=4, help='Parallel jobs')
    batch_parser.add_argument('--wm-threshold', type=float, default=0.5, help='WM threshold')
    batch_parser.add_argument('--smooth-fwhm', type=float, default=4.0, help='Smoothing FWHM')
    batch_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing')

    # Group analysis
    analyze_parser = subparsers.add_parser('analyze', help='Run group analysis')
    analyze_parser.add_argument('--t1t2ratio-dir', required=True, help='T1T2 ratio directory')
    analyze_parser.add_argument('--design-dir', required=True, help='Design files directory')
    analyze_parser.add_argument('--method', default='randomise', choices=['randomise', 'glm', 'both'])
    analyze_parser.add_argument('--n-permutations', type=int, default=5000, help='Permutations')
    analyze_parser.add_argument('--no-tfce', action='store_true', help='Disable TFCE')

    # Generate report
    report_parser = subparsers.add_parser('report', help='Generate summary report')
    report_parser.add_argument('--t1t2ratio-dir', required=True, help='T1T2 ratio directory')

    args = parser.parse_args()

    if args.command == 'prepare':
        result = prepare_t1t2ratio_single(
            subject=args.subject,
            study_root=Path(args.study_root),
            output_dir=Path(args.output_dir),
            wm_threshold=args.wm_threshold,
            smooth_fwhm=args.smooth_fwhm,
            overwrite=args.overwrite
        )
        print(f"\nResult: {result['status']}")
        if 'outputs' in result:
            for key, path in result['outputs'].items():
                print(f"  {key}: {path}")

    elif args.command == 'batch':
        result = prepare_t1t2ratio_batch(
            study_root=Path(args.study_root),
            output_dir=Path(args.output_dir),
            n_jobs=args.n_jobs,
            wm_threshold=args.wm_threshold,
            smooth_fwhm=args.smooth_fwhm,
            overwrite=args.overwrite
        )
        print(f"\nBatch complete: {result['n_successful']}/{result['n_total']} successful")

    elif args.command == 'analyze':
        result = run_t1t2ratio_analysis(
            t1t2ratio_dir=Path(args.t1t2ratio_dir),
            design_dir=Path(args.design_dir),
            method=args.method,
            n_permutations=args.n_permutations,
            tfce=not args.no_tfce
        )
        print("\nAnalysis complete")

    elif args.command == 'report':
        from neurovrai.analysis.anat.t1t2ratio_reporting import generate_t1t2ratio_html_report
        report_path = generate_t1t2ratio_html_report(Path(args.t1t2ratio_dir))
        print(f"\nReport generated: {report_path}")

    else:
        parser.print_help()
