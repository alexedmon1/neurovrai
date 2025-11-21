#!/usr/bin/env python3
"""
FSL Randomise Wrapper with TFCE

Executes FSL's randomise for nonparametric permutation testing with:
- Threshold-Free Cluster Enhancement (TFCE)
- Multiple comparison correction
- T-tests and F-tests
- Comprehensive logging and error handling

FSL randomise performs permutation-based inference to identify significant
clusters while controlling family-wise error rate.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np


class RandomiseError(Exception):
    """Raised when randomise execution fails"""
    pass


def check_fsl_installation() -> bool:
    """
    Check if FSL is installed and accessible

    Returns:
        True if FSL is available, False otherwise
    """
    try:
        result = subprocess.run(
            ['which', 'randomise'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def validate_inputs(
    input_file: Path,
    design_mat: Path,
    contrast_con: Path,
    mask: Optional[Path] = None
):
    """
    Validate that all required input files exist

    Args:
        input_file: 4D volume for analysis
        design_mat: FSL design matrix
        contrast_con: FSL contrast matrix
        mask: Optional binary mask

    Raises:
        RandomiseError: If any required files are missing
    """
    if not input_file.exists():
        raise RandomiseError(f"Input file not found: {input_file}")

    if not design_mat.exists():
        raise RandomiseError(f"Design matrix not found: {design_mat}")

    if not contrast_con.exists():
        raise RandomiseError(f"Contrast file not found: {contrast_con}")

    if mask is not None and not mask.exists():
        raise RandomiseError(f"Mask file not found: {mask}")

    # Validate NIfTI files
    try:
        img = nib.load(input_file)
        if len(img.shape) != 4:
            raise RandomiseError(
                f"Input file must be 4D volume, got shape {img.shape}"
            )
    except Exception as e:
        raise RandomiseError(f"Failed to load input file: {e}")

    if mask is not None:
        try:
            mask_img = nib.load(mask)
            if len(mask_img.shape) != 3:
                raise RandomiseError(
                    f"Mask must be 3D volume, got shape {mask_img.shape}"
                )
        except Exception as e:
            raise RandomiseError(f"Failed to load mask: {e}")


def run_randomise(
    input_file: Path,
    design_mat: Path,
    contrast_con: Path,
    output_dir: Path,
    mask: Optional[Path] = None,
    n_permutations: int = 5000,
    tfce: bool = True,
    voxel_threshold: Optional[float] = None,
    demean: bool = False,
    variance_smoothing: Optional[float] = None,
    seed: Optional[int] = None,
    n_threads: int = 1
) -> Dict:
    """
    Execute FSL randomise with specified parameters

    Args:
        input_file: 4D input volume (e.g., all_FA_skeletonised.nii.gz)
        design_mat: FSL design matrix (.mat file)
        contrast_con: FSL contrast matrix (.con file)
        output_dir: Output directory for results
        mask: Optional binary mask (3D volume)
        n_permutations: Number of permutations (default: 5000)
        tfce: Use Threshold-Free Cluster Enhancement (default: True)
        voxel_threshold: Cluster-forming threshold (only if tfce=False)
        demean: Demean data temporally (for within-subject designs)
        variance_smoothing: Variance smoothing (mm, for heterogeneous variance)
        seed: Random seed for reproducibility
        n_threads: Number of parallel threads (default: 1)

    Returns:
        Dictionary with execution results and output file paths

    Raises:
        RandomiseError: If randomise execution fails
    """
    # Check FSL
    if not check_fsl_installation():
        raise RandomiseError(
            "FSL randomise not found. Ensure FSL is installed and $FSLDIR is set."
        )

    # Validate inputs
    validate_inputs(input_file, design_mat, contrast_con, mask)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build randomise command
    output_basename = output_dir / "randomise"

    cmd = [
        'randomise',
        '-i', str(input_file),
        '-o', str(output_basename),
        '-d', str(design_mat),
        '-t', str(contrast_con),
        '-n', str(n_permutations),
    ]

    # Add mask if provided
    if mask is not None:
        cmd.extend(['-m', str(mask)])

    # TFCE or voxel-wise thresholding
    if tfce:
        cmd.append('--T2')  # TFCE for T-statistics
        logging.info("Using Threshold-Free Cluster Enhancement (TFCE)")
    elif voxel_threshold is not None:
        cmd.extend(['-c', str(voxel_threshold)])
        logging.info(f"Using cluster threshold: {voxel_threshold}")

    # Optional parameters
    if demean:
        cmd.append('-D')
        logging.info("Demeaning data temporally")

    if variance_smoothing is not None:
        cmd.extend(['-v', str(variance_smoothing)])
        logging.info(f"Variance smoothing: {variance_smoothing}mm")

    if seed is not None:
        cmd.extend(['--seed', str(seed)])
        logging.info(f"Random seed: {seed}")

    # Parallel execution
    if n_threads > 1:
        # Note: FSL randomise doesn't have built-in threading,
        # but you can use GNU parallel or similar
        logging.warning(
            "FSL randomise doesn't natively support threading. "
            "Consider using parallel randomise or PALM for faster execution."
        )

    # Log command
    logging.info("=" * 80)
    logging.info("Executing FSL Randomise")
    logging.info("=" * 80)
    logging.info(f"Command: {' '.join(cmd)}")
    logging.info(f"Input: {input_file}")
    logging.info(f"Design: {design_mat}")
    logging.info(f"Contrasts: {contrast_con}")
    logging.info(f"Permutations: {n_permutations}")
    logging.info(f"Output: {output_dir}")

    # Execute
    start_time = time.time()
    log_file = output_dir / "randomise.log"

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        elapsed = time.time() - start_time
        logging.info(f"✓ Randomise completed in {elapsed:.1f} seconds")

    except subprocess.CalledProcessError as e:
        logging.error(f"Randomise failed with exit code {e.returncode}")
        logging.error(f"Check log file: {log_file}")
        raise RandomiseError(f"Randomise execution failed: {e}")

    # Collect output files
    output_files = {
        'log': log_file,
        'tstat': [],
        'tstat_tfce': [],
        'tstat_corrp': [],
        'fstat': [],
        'fstat_tfce': [],
        'fstat_corrp': []
    }

    for f in output_dir.glob('randomise_*.nii.gz'):
        name = f.name
        if 'tstat' in name and 'tfce' in name and 'corrp' in name:
            output_files['tstat_corrp'].append(f)
        elif 'tstat' in name and 'tfce' in name:
            output_files['tstat_tfce'].append(f)
        elif 'tstat' in name:
            output_files['tstat'].append(f)
        elif 'fstat' in name and 'tfce' in name and 'corrp' in name:
            output_files['fstat_corrp'].append(f)
        elif 'fstat' in name and 'tfce' in name:
            output_files['fstat_tfce'].append(f)
        elif 'fstat' in name:
            output_files['fstat'].append(f)

    # Sort files
    for key in output_files:
        if isinstance(output_files[key], list):
            output_files[key].sort()

    logging.info("\nOutput files:")
    for key, files in output_files.items():
        if isinstance(files, list) and files:
            logging.info(f"  {key}: {len(files)} files")
        elif not isinstance(files, list):
            logging.info(f"  {key}: {files}")

    return {
        'success': True,
        'elapsed_time': elapsed,
        'n_permutations': n_permutations,
        'output_dir': str(output_dir),
        'output_files': {k: [str(f) for f in v] if isinstance(v, list) else str(v)
                         for k, v in output_files.items()}
    }


def get_significant_voxels(
    corrp_file: Path,
    threshold: float = 0.95
) -> Dict:
    """
    Extract significant voxels from corrected p-value map

    FSL randomise outputs 1-p values, so threshold at 0.95 = p < 0.05

    Args:
        corrp_file: Path to randomise corrected p-value file
        threshold: Threshold for significance (default: 0.95 for p<0.05)

    Returns:
        Dictionary with significant voxel statistics
    """
    if not corrp_file.exists():
        raise RandomiseError(f"Corrected p-value file not found: {corrp_file}")

    img = nib.load(corrp_file)
    data = img.get_fdata()

    # Count significant voxels
    sig_mask = data >= threshold
    n_significant = np.sum(sig_mask)

    if n_significant > 0:
        sig_values = data[sig_mask]
        max_val = np.max(sig_values)
        mean_val = np.mean(sig_values)
    else:
        max_val = 0
        mean_val = 0

    return {
        'n_significant_voxels': int(n_significant),
        'max_corrp': float(max_val),
        'mean_corrp': float(mean_val),
        'threshold': threshold
    }


def summarize_results(
    output_dir: Path,
    threshold: float = 0.95
) -> Dict:
    """
    Summarize randomise results across all contrasts

    Args:
        output_dir: Directory containing randomise outputs
        threshold: Significance threshold (default: 0.95 for p<0.05)

    Returns:
        Summary dictionary with significant findings
    """
    output_dir = Path(output_dir)
    summary = {
        'contrasts': []
    }

    # Find all corrected p-value maps
    corrp_files = sorted(output_dir.glob('*_corrp_*.nii.gz'))

    for corrp_file in corrp_files:
        # Extract contrast info from filename
        # Expected: randomise_tstat1_corrp_tfce.nii.gz
        name = corrp_file.stem.replace('.nii', '')
        parts = name.split('_')

        contrast_type = None
        contrast_num = None
        for part in parts:
            if 'tstat' in part:
                contrast_type = 'tstat'
                contrast_num = part.replace('tstat', '')
            elif 'fstat' in part:
                contrast_type = 'fstat'
                contrast_num = part.replace('fstat', '')

        # Get significant voxels
        sig_info = get_significant_voxels(corrp_file, threshold)

        summary['contrasts'].append({
            'file': str(corrp_file),
            'type': contrast_type,
            'contrast_number': contrast_num,
            'n_significant_voxels': sig_info['n_significant_voxels'],
            'max_corrp': sig_info['max_corrp'],
            'significant': sig_info['n_significant_voxels'] > 0
        })

    logging.info("\n" + "=" * 80)
    logging.info("RANDOMISE RESULTS SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Threshold: {threshold} (p < {1-threshold})")

    for contrast in summary['contrasts']:
        status = "✓ SIGNIFICANT" if contrast['significant'] else "✗ Not significant"
        logging.info(
            f"{contrast['type']}{contrast['contrast_number']}: "
            f"{contrast['n_significant_voxels']} voxels {status}"
        )

    return summary


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # This would be run after prepare_tbss.py creates the skeleton data
    result = run_randomise(
        input_file=Path('/study/analysis/tbss_FA/FA/all_FA_skeletonised.nii.gz'),
        design_mat=Path('/study/analysis/tbss_FA/model1/design.mat'),
        contrast_con=Path('/study/analysis/tbss_FA/model1/design.con'),
        output_dir=Path('/study/analysis/tbss_FA/model1/randomise_output/'),
        mask=Path('/study/analysis/tbss_FA/FA/mean_FA_skeleton_mask.nii.gz'),
        n_permutations=5000,
        tfce=True,
        seed=42
    )

    print(f"\nResults saved to: {result['output_dir']}")
    print(f"Execution time: {result['elapsed_time']:.1f} seconds")
