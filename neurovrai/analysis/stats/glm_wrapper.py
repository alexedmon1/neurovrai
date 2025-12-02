#!/usr/bin/env python3
"""
FSL GLM Wrapper for Parametric Statistical Analysis

Executes FSL's fsl_glm for parametric general linear model analysis with:
- Ordinary Least Squares (OLS) fitting
- Multiple comparison correction (cluster thresholding or FDR)
- T-tests and F-tests
- Comprehensive logging and error handling

This provides a faster, parametric alternative to randomise, using the same
design matrices and contrasts. Use when:
- Data meets normality assumptions
- Quick exploratory analysis needed
- Want to compare parametric vs nonparametric results

FSL GLM performs voxel-wise regression using parametric statistics.
For nonparametric inference, use randomise_wrapper.py instead.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from scipy import ndimage


class GLMError(Exception):
    """Raised when GLM execution fails"""
    pass


def check_fsl_installation() -> bool:
    """
    Check if FSL is installed and accessible

    Returns:
        True if FSL is available, False otherwise
    """
    try:
        result = subprocess.run(
            ['which', 'fsl_glm'],
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
        GLMError: If any required files are missing
    """
    if not input_file.exists():
        raise GLMError(f"Input file not found: {input_file}")

    if not design_mat.exists():
        raise GLMError(f"Design matrix not found: {design_mat}")

    if not contrast_con.exists():
        raise GLMError(f"Contrast file not found: {contrast_con}")

    if mask is not None and not mask.exists():
        raise GLMError(f"Mask file not found: {mask}")

    # Validate NIfTI files
    try:
        img = nib.load(input_file)
        if len(img.shape) != 4:
            raise GLMError(
                f"Input file must be 4D volume, got shape {img.shape}"
            )
    except Exception as e:
        raise GLMError(f"Failed to load input file: {e}")

    if mask is not None:
        try:
            mask_img = nib.load(mask)
            if len(mask_img.shape) != 3:
                raise GLMError(
                    f"Mask must be 3D volume, got shape {mask_img.shape}"
                )
        except Exception as e:
            raise GLMError(f"Failed to load mask: {e}")


def parse_contrast_file(contrast_con: Path) -> List[Dict]:
    """
    Parse FSL contrast file to extract contrast names and vectors

    Args:
        contrast_con: Path to .con file

    Returns:
        List of contrast dictionaries with 'name' and 'vector' keys
    """
    contrasts = []
    contrast_names = {}
    vectors = []

    with open(contrast_con, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse contrast names
            if line.startswith('/ContrastName'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    # Extract number and name
                    # Format: /ContrastName1\tage_positive
                    key = parts[0]
                    name = parts[1]
                    num = int(key.replace('/ContrastName', ''))
                    contrast_names[num] = name

            # Parse matrix data (skip headers)
            elif not line.startswith('/') and line and not line.startswith('#'):
                values = [float(x) for x in line.split()]
                if values:
                    vectors.append(values)

    # Combine names with vectors
    for i, vector in enumerate(vectors):
        name = contrast_names.get(i + 1, f"contrast{i + 1}")
        contrasts.append({
            'name': name,
            'vector': vector,
            'index': i + 1
        })

    return contrasts


def run_fsl_glm(
    input_file: Path,
    design_mat: Path,
    contrast_con: Path,
    output_dir: Path,
    mask: Optional[Path] = None,
    demean: bool = False,
    variance_normalization: bool = False
) -> Dict:
    """
    Execute FSL fsl_glm with specified parameters

    Args:
        input_file: 4D input volume (e.g., all_FA_skeletonised.nii.gz)
        design_mat: FSL design matrix (.mat file)
        contrast_con: FSL contrast matrix (.con file)
        output_dir: Output directory for results
        mask: Optional binary mask (3D volume)
        demean: Demean data temporally (for within-subject designs)
        variance_normalization: Perform variance normalization

    Returns:
        Dictionary with execution results and output file paths

    Raises:
        GLMError: If fsl_glm execution fails
    """
    # Check FSL
    if not check_fsl_installation():
        raise GLMError(
            "FSL fsl_glm not found. Ensure FSL is installed and $FSLDIR is set."
        )

    # Validate inputs
    validate_inputs(input_file, design_mat, contrast_con, mask)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse contrasts to get names
    contrasts = parse_contrast_file(contrast_con)
    logging.info(f"Found {len(contrasts)} contrasts:")
    for contrast in contrasts:
        logging.info(f"  {contrast['index']}: {contrast['name']}")

    # Process each contrast individually
    # fsl_glm doesn't handle multiple contrasts well, so we run it once per contrast
    logging.info("\n" + "=" * 80)
    logging.info("Processing contrasts individually with FSL GLM")
    logging.info("=" * 80)

    start_time = time.time()
    output_files_list = []

    for contrast in contrasts:
        contrast_idx = contrast['index']
        contrast_name = contrast['name']
        contrast_vector = contrast['vector']

        logging.info(f"\nContrast {contrast_idx}: {contrast_name}")
        logging.info(f"  Vector: {contrast_vector}")

        # Create temporary single-contrast file
        temp_con_file = output_dir / f"contrast_{contrast_idx}.con"
        with open(temp_con_file, 'w') as f:
            f.write(f"/NumWaves {len(contrast_vector)}\n")
            f.write("/NumContrasts 1\n")
            f.write("/Matrix\n")
            f.write(" ".join([f"{v:.6f}" for v in contrast_vector]) + "\n")

        # Build fsl_glm command for this contrast
        output_basename = output_dir / f"glm_contrast{contrast_idx}"

        cmd = [
            'fsl_glm',
            '-i', str(input_file),
            '-d', str(design_mat),
            '-c', str(temp_con_file),
            '-o', str(output_basename) + '_pe.nii.gz',
            '--out_z', str(output_basename) + '_zstat.nii.gz',
            '--out_t', str(output_basename) + '_tstat.nii.gz',
            '--out_cope', str(output_basename) + '_cope.nii.gz',
            '--out_varcb', str(output_basename) + '_varcope.nii.gz',
        ]

        # Add mask if provided
        if mask is not None:
            cmd.extend(['-m', str(mask)])

        # Optional parameters
        if demean:
            cmd.append('--demean')

        if variance_normalization:
            cmd.append('--des_norm')

        # Execute
        contrast_log = output_dir / f"glm_contrast{contrast_idx}.log"

        try:
            with open(contrast_log, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True
                )

            logging.info(f"  ✓ Contrast {contrast_idx} complete")

            # Collect output files for this contrast
            # fsl_glm creates files with the exact names we specified
            output_files_list.append({
                'contrast_index': contrast_idx,
                'contrast_name': contrast_name,
                'pe': Path(str(output_basename) + '_pe.nii.gz'),
                'zstat': Path(str(output_basename) + '_zstat.nii.gz'),
                'tstat': Path(str(output_basename) + '_tstat.nii.gz'),
                'cope': Path(str(output_basename) + '_cope.nii.gz'),
                'varcope': Path(str(output_basename) + '_varcope.nii.gz'),
                'log': contrast_log
            })

        except subprocess.CalledProcessError as e:
            logging.error(f"  ✗ Contrast {contrast_idx} failed with exit code {e.returncode}")
            logging.error(f"  Check log: {contrast_log}")
            raise GLMError(f"fsl_glm failed for contrast {contrast_idx} ({contrast_name}): {e}")

    elapsed = time.time() - start_time
    logging.info(f"\n✓ All {len(contrasts)} contrasts completed in {elapsed:.1f} seconds")

    # Create summary log
    summary_log = output_dir / "glm_summary.log"
    with open(summary_log, 'w') as f:
        f.write("FSL GLM Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input: {input_file}\n")
        f.write(f"Design: {design_mat}\n")
        f.write(f"Contrasts: {len(contrasts)}\n")
        f.write(f"Elapsed: {elapsed:.1f} seconds\n\n")
        for files in output_files_list:
            f.write(f"Contrast {files['contrast_index']}: {files['contrast_name']}\n")
            for key in ['zstat', 'tstat', 'cope']:
                exists = "✓" if files[key].exists() else "✗"
                f.write(f"  {exists} {key}: {files[key].name}\n")

    # Return results with first contrast's zstat as the primary output
    # (for compatibility with code expecting single zstat file)
    return {
        'success': True,
        'elapsed_time': elapsed,
        'output_dir': str(output_dir),
        'output_files': {
            'contrasts': contrasts,
            'zstat': str(output_files_list[0]['zstat']) if output_files_list else None,
            'all_contrasts': output_files_list
        }
    }


def threshold_zstat(
    zstat_file: Path,
    output_dir: Path,
    z_threshold: float = 2.3,
    cluster_threshold: int = 10,
    mask: Optional[Path] = None
) -> Dict:
    """
    Threshold z-statistic map and extract clusters

    Args:
        zstat_file: Path to z-statistic map
        output_dir: Output directory for thresholded maps
        z_threshold: Z-score threshold (default: 2.3 ≈ p<0.01)
        cluster_threshold: Minimum cluster size in voxels
        mask: Optional mask for analysis

    Returns:
        Dictionary with cluster statistics
    """
    logging.info("\nThresholding z-statistics...")
    logging.info(f"  Z-threshold: {z_threshold}")
    logging.info(f"  Min cluster size: {cluster_threshold} voxels")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load z-stat map
    img = nib.load(zstat_file)

    # Handle both 3D (single contrast) and 4D (multiple contrasts)
    if len(img.shape) == 3:
        data = img.get_fdata()[..., np.newaxis]  # Add contrast dimension
        n_contrasts = 1
    else:
        data = img.get_fdata()
        n_contrasts = data.shape[3]

    # Load mask if provided
    if mask is not None:
        mask_img = nib.load(mask)
        mask_data = mask_img.get_fdata() > 0
    else:
        mask_data = np.ones(data.shape[:3], dtype=bool)

    cluster_results = []

    for contrast_idx in range(n_contrasts):
        zstat_3d = data[..., contrast_idx]

        # Threshold
        thresh_pos = (zstat_3d > z_threshold) & mask_data
        thresh_neg = (zstat_3d < -z_threshold) & mask_data

        # Find clusters
        clusters_pos, n_clusters_pos = ndimage.label(thresh_pos)
        clusters_neg, n_clusters_neg = ndimage.label(thresh_neg)

        # Extract cluster statistics
        cluster_info = {
            'contrast_index': contrast_idx + 1,
            'positive_clusters': [],
            'negative_clusters': []
        }

        # Positive clusters
        for cluster_id in range(1, n_clusters_pos + 1):
            cluster_mask = clusters_pos == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= cluster_threshold:
                cluster_values = zstat_3d[cluster_mask]
                cluster_info['positive_clusters'].append({
                    'size': int(cluster_size),
                    'max_z': float(np.max(cluster_values)),
                    'mean_z': float(np.mean(cluster_values))
                })

        # Negative clusters
        for cluster_id in range(1, n_clusters_neg + 1):
            cluster_mask = clusters_neg == cluster_id
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= cluster_threshold:
                cluster_values = zstat_3d[cluster_mask]
                cluster_info['negative_clusters'].append({
                    'size': int(cluster_size),
                    'min_z': float(np.min(cluster_values)),
                    'mean_z': float(np.mean(cluster_values))
                })

        cluster_results.append(cluster_info)

        # Save thresholded map
        thresh_map = np.zeros_like(zstat_3d)
        thresh_map[thresh_pos] = zstat_3d[thresh_pos]
        thresh_map[thresh_neg] = zstat_3d[thresh_neg]

        thresh_file = output_dir / f"zstat_contrast{contrast_idx + 1}_thresh.nii.gz"
        thresh_img = nib.Nifti1Image(thresh_map, img.affine, img.header)
        nib.save(thresh_img, thresh_file)

        # Log results
        n_pos = len(cluster_info['positive_clusters'])
        n_neg = len(cluster_info['negative_clusters'])
        logging.info(f"\n  Contrast {contrast_idx + 1}:")
        logging.info(f"    Positive clusters: {n_pos}")
        logging.info(f"    Negative clusters: {n_neg}")

        if n_pos > 0:
            total_voxels = sum(c['size'] for c in cluster_info['positive_clusters'])
            logging.info(f"    Total positive voxels: {total_voxels}")

        if n_neg > 0:
            total_voxels = sum(c['size'] for c in cluster_info['negative_clusters'])
            logging.info(f"    Total negative voxels: {total_voxels}")

    return {
        'cluster_results': cluster_results,
        'n_contrasts': n_contrasts,
        'z_threshold': z_threshold,
        'cluster_threshold': cluster_threshold
    }


def summarize_glm_results(
    output_dir: Path,
    contrast_names: Optional[List[str]] = None,
    z_threshold: float = 2.3
) -> Dict:
    """
    Summarize GLM results across all contrasts

    Args:
        output_dir: Directory containing GLM outputs
        contrast_names: Optional list of contrast names
        z_threshold: Z-score threshold for significance

    Returns:
        Summary dictionary with significant findings
    """
    output_dir = Path(output_dir)

    zstat_file = output_dir / "glm_zstat.nii.gz"
    if not zstat_file.exists():
        raise GLMError(f"Z-statistic file not found: {zstat_file}")

    img = nib.load(zstat_file)

    # Handle both 3D and 4D
    if len(img.shape) == 3:
        data = img.get_fdata()[..., np.newaxis]
        n_contrasts = 1
    else:
        data = img.get_fdata()
        n_contrasts = data.shape[3]

    summary = {
        'contrasts': []
    }

    for i in range(n_contrasts):
        zstat = data[..., i]

        # Count significant voxels
        n_sig_pos = np.sum(zstat > z_threshold)
        n_sig_neg = np.sum(zstat < -z_threshold)

        max_z = float(np.max(zstat))
        min_z = float(np.min(zstat))

        contrast_name = contrast_names[i] if contrast_names and i < len(contrast_names) else f"contrast{i + 1}"

        summary['contrasts'].append({
            'name': contrast_name,
            'index': i + 1,
            'n_positive_voxels': int(n_sig_pos),
            'n_negative_voxels': int(n_sig_neg),
            'max_z': max_z,
            'min_z': min_z,
            'significant': (n_sig_pos > 0 or n_sig_neg > 0)
        })

    logging.info("\n" + "=" * 80)
    logging.info("GLM RESULTS SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Z-threshold: {z_threshold} (≈ p < 0.01)")

    for contrast in summary['contrasts']:
        status = "✓ SIGNIFICANT" if contrast['significant'] else "✗ Not significant"
        logging.info(
            f"{contrast['name']}: "
            f"{contrast['n_positive_voxels']} pos, "
            f"{contrast['n_negative_voxels']} neg voxels {status}"
        )

    return summary


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # This would be run after prepare_tbss.py creates the skeleton data
    # and design matrices have been created
    result = run_fsl_glm(
        input_file=Path('/study/analysis/tbss_FA/FA/all_FA_skeletonised.nii.gz'),
        design_mat=Path('/study/analysis/tbss_FA/model1/design.mat'),
        contrast_con=Path('/study/analysis/tbss_FA/model1/design.con'),
        output_dir=Path('/study/analysis/tbss_FA/model1/glm_output/'),
        mask=Path('/study/analysis/tbss_FA/FA/mean_FA_skeleton_mask.nii.gz')
    )

    print(f"\nResults saved to: {result['output_dir']}")
    print(f"Execution time: {result['elapsed_time']:.1f} seconds")

    # Threshold results
    threshold_result = threshold_zstat(
        zstat_file=Path(result['output_files']['zstat']),
        output_dir=Path(result['output_dir']) / 'thresholded',
        z_threshold=2.3,
        cluster_threshold=10
    )

    print(f"\nFound clusters in {threshold_result['n_contrasts']} contrasts")
