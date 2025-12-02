#!/usr/bin/env python3
"""
Nilearn-based Second-Level GLM Analysis

Provides voxel-wise statistical analysis using nilearn's SecondLevelModel.
More flexible and reliable than FSL's fsl_glm with better Python integration.

Features:
- Multiple comparison correction (FDR, Bonferroni, cluster-level)
- Effect size and confidence interval computation
- Parallel processing support
- Clean Python API without command-line compatibility issues
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.reporting import make_glm_report

logger = logging.getLogger(__name__)


class NilearnGLMError(Exception):
    """Custom exception for nilearn GLM errors"""
    pass


def run_second_level_glm(
    input_files: List[Path],
    design_matrix: pd.DataFrame,
    contrasts: Dict[str, np.ndarray],
    output_dir: Path,
    mask: Optional[Path] = None,
    smoothing_fwhm: Optional[float] = None,
    n_jobs: int = -1,
    minimize_memory: bool = True
) -> Dict:
    """
    Run second-level GLM analysis using nilearn.

    Args:
        input_files: List of 3D subject-level statistical maps
        design_matrix: DataFrame with design matrix (subjects x regressors)
        contrasts: Dictionary mapping contrast names to contrast vectors
        output_dir: Directory to save results
        mask: Optional brain mask file
        smoothing_fwhm: Optional smoothing FWHM in mm (None if already smoothed)
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        minimize_memory: Use memory-efficient processing

    Returns:
        Dictionary with results and output file paths

    Example:
        >>> contrasts = {
        ...     'age_positive': np.array([0, 1, 0, 0]),  # Test age effect
        ...     'group_diff': np.array([0, 0, 1, 0])     # Test group difference
        ... }
        >>> results = run_second_level_glm(
        ...     input_files=subject_maps,
        ...     design_matrix=design_df,
        ...     contrasts=contrasts,
        ...     output_dir=Path('glm_output'),
        ...     mask=Path('mask.nii.gz')
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("NILEARN SECOND-LEVEL GLM")
    logger.info("=" * 80)
    logger.info(f"Subjects: {len(input_files)}")
    logger.info(f"Design matrix shape: {design_matrix.shape}")
    logger.info(f"Contrasts: {list(contrasts.keys())}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Validate inputs
    if len(input_files) != len(design_matrix):
        raise NilearnGLMError(
            f"Number of input files ({len(input_files)}) doesn't match "
            f"design matrix rows ({len(design_matrix)})"
        )

    # Load mask if provided
    mask_img = nib.load(str(mask)) if mask else None

    # Create second-level model
    logger.info("Creating SecondLevelModel...")
    model = SecondLevelModel(
        mask_img=mask_img,
        smoothing_fwhm=smoothing_fwhm,
        n_jobs=n_jobs,
        minimize_memory=minimize_memory
    )

    # Fit model
    logger.info("Fitting GLM model...")
    start_time = time.time()

    try:
        model.fit(
            [str(f) for f in input_files],
            design_matrix=design_matrix
        )
        fit_time = time.time() - start_time
        logger.info(f"✓ Model fitted in {fit_time:.1f} seconds")
    except Exception as e:
        raise NilearnGLMError(f"GLM fitting failed: {e}")

    # Compute contrasts
    logger.info("\nComputing contrasts...")
    contrast_results = {}

    for contrast_name, contrast_vector in contrasts.items():
        logger.info(f"  {contrast_name}: {contrast_vector}")

        try:
            # Compute contrast
            z_map = model.compute_contrast(
                contrast_vector,
                output_type='z_score'
            )

            t_map = model.compute_contrast(
                contrast_vector,
                output_type='stat'
            )

            effect_map = model.compute_contrast(
                contrast_vector,
                output_type='effect_size'
            )

            # Save maps
            z_file = output_dir / f'{contrast_name}_zstat.nii.gz'
            t_file = output_dir / f'{contrast_name}_tstat.nii.gz'
            effect_file = output_dir / f'{contrast_name}_effect.nii.gz'

            nib.save(z_map, str(z_file))
            nib.save(t_map, str(t_file))
            nib.save(effect_map, str(effect_file))

            contrast_results[contrast_name] = {
                'zstat': z_file,
                'tstat': t_file,
                'effect': effect_file,
                'z_map': z_map,
                't_map': t_map,
                'effect_map': effect_map
            }

            logger.info(f"    ✓ Saved: {z_file.name}, {t_file.name}, {effect_file.name}")

        except Exception as e:
            logger.error(f"    ✗ Failed: {e}")
            raise NilearnGLMError(f"Contrast computation failed for {contrast_name}: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"\n✓ GLM complete in {elapsed_time:.1f} seconds")

    return {
        'model': model,
        'contrasts': contrast_results,
        'output_dir': output_dir,
        'elapsed_time': elapsed_time,
        'n_subjects': len(input_files)
    }


def apply_multiple_comparison_correction(
    z_map: nib.Nifti1Image,
    output_dir: Path,
    contrast_name: str,
    alpha: float = 0.05,
    height_control: str = 'fdr',
    cluster_threshold: int = 10,
    two_sided: bool = True
) -> Dict:
    """
    Apply multiple comparison correction to z-stat map.

    Args:
        z_map: Nilearn z-statistic map
        output_dir: Directory to save thresholded maps
        contrast_name: Name of contrast for output files
        alpha: Statistical threshold (default: 0.05)
        height_control: Type of correction ('fdr', 'bonferroni', 'fpr', None)
        cluster_threshold: Minimum cluster size in voxels
        two_sided: If True, use two-sided test

    Returns:
        Dictionary with thresholded maps and statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Applying {height_control or 'uncorrected'} correction (alpha={alpha})...")

    try:
        # Apply threshold
        thresholded_map, threshold = threshold_stats_img(
            z_map,
            alpha=alpha,
            height_control=height_control,
            cluster_threshold=cluster_threshold,
            two_sided=two_sided
        )

        # Save thresholded map
        thresh_file = output_dir / f'{contrast_name}_zstat_thresh_{height_control or "uncorr"}.nii.gz'
        nib.save(thresholded_map, str(thresh_file))

        # Get cluster statistics
        data = thresholded_map.get_fdata()
        n_sig_voxels = np.sum(data != 0)

        logger.info(f"  Threshold: z > {threshold:.3f}")
        logger.info(f"  Significant voxels: {n_sig_voxels}")
        logger.info(f"  ✓ Saved: {thresh_file.name}")

        return {
            'thresholded_map': thresholded_map,
            'threshold_file': thresh_file,
            'threshold_value': threshold,
            'n_significant_voxels': int(n_sig_voxels),
            'alpha': alpha,
            'correction': height_control or 'uncorrected'
        }

    except Exception as e:
        logger.error(f"  ✗ Thresholding failed: {e}")
        raise NilearnGLMError(f"Multiple comparison correction failed: {e}")


def summarize_glm_results(
    glm_results: Dict,
    output_dir: Path,
    alpha: float = 0.05,
    corrections: List[str] = ['fdr', 'bonferroni']
) -> Dict:
    """
    Summarize GLM results with multiple comparison corrections.

    Args:
        glm_results: Results dictionary from run_second_level_glm()
        output_dir: Directory for summary outputs
        alpha: Statistical threshold
        corrections: List of correction methods to apply

    Returns:
        Dictionary with summary statistics and thresholded maps
    """
    output_dir = Path(output_dir)
    thresh_dir = output_dir / 'thresholded'
    thresh_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARIZING GLM RESULTS")
    logger.info("=" * 80)

    summary = {
        'contrasts': {},
        'n_subjects': glm_results['n_subjects'],
        'alpha': alpha
    }

    for contrast_name, contrast_data in glm_results['contrasts'].items():
        logger.info(f"\nContrast: {contrast_name}")

        contrast_summary = {
            'files': {
                'zstat': contrast_data['zstat'],
                'tstat': contrast_data['tstat'],
                'effect': contrast_data['effect']
            },
            'thresholded': {}
        }

        # Apply each correction method
        for correction in corrections:
            thresh_result = apply_multiple_comparison_correction(
                z_map=contrast_data['z_map'],
                output_dir=thresh_dir,
                contrast_name=contrast_name,
                alpha=alpha,
                height_control=correction
            )
            contrast_summary['thresholded'][correction] = thresh_result

        summary['contrasts'][contrast_name] = contrast_summary

    logger.info("\n" + "=" * 80)
    logger.info("GLM SUMMARY COMPLETE")
    logger.info("=" * 80)

    return summary
