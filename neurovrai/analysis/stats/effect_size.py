#!/usr/bin/env python3
"""
Effect Size Calculation from FSL Randomise Results

This module computes Cohen's d and other effect size measures from
FSL randomise t-statistic maps. Cohen's d provides a standardized
measure of effect magnitude that is independent of sample size.

Key features:
- Convert t-statistics to Cohen's d
- Support for different design types (one-sample, two-sample, paired)
- Generate both uncorrected and thresholded effect size maps
- Calculate confidence intervals for effect sizes
- Create publication-ready visualizations
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from nilearn import plotting, masking

logger = logging.getLogger(__name__)


class EffectSizeError(Exception):
    """Raised when effect size calculation fails"""
    pass


def t_to_cohens_d(
    t_stat: np.ndarray,
    n1: int,
    n2: Optional[int] = None,
    design_type: str = "two_sample"
) -> np.ndarray:
    """
    Convert t-statistic to Cohen's d effect size.

    Cohen's d represents the standardized mean difference between groups,
    where:
    - d = 0.2 is considered a small effect
    - d = 0.5 is considered a medium effect
    - d = 0.8 is considered a large effect

    Parameters
    ----------
    t_stat : np.ndarray
        T-statistic values (can be 3D or 4D array)
    n1 : int
        Sample size for group 1 (or total N for one-sample)
    n2 : int, optional
        Sample size for group 2 (only for two-sample designs)
    design_type : str
        Type of design: "one_sample", "two_sample", or "paired"

    Returns
    -------
    np.ndarray
        Cohen's d effect size values (same shape as input)

    References
    ----------
    Rosenthal, R. (1991). Meta-analytic procedures for social research.
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    if design_type == "one_sample" or design_type == "paired":
        # For one-sample or paired t-test: d = t / sqrt(n)
        d = t_stat / np.sqrt(n1)

    elif design_type == "two_sample":
        if n2 is None:
            raise EffectSizeError("n2 required for two-sample design")

        # For independent samples t-test: d = t * sqrt(1/n1 + 1/n2)
        # This accounts for unequal sample sizes
        d = t_stat * np.sqrt(1/n1 + 1/n2)

    else:
        raise EffectSizeError(f"Unknown design type: {design_type}")

    return d


def hedges_g(d: np.ndarray, n1: int, n2: Optional[int] = None) -> np.ndarray:
    """
    Convert Cohen's d to Hedges' g (bias-corrected effect size).

    Hedges' g corrects for small sample bias in Cohen's d.
    Recommended when total sample size < 20.

    Parameters
    ----------
    d : np.ndarray
        Cohen's d values
    n1 : int
        Sample size for group 1
    n2 : int, optional
        Sample size for group 2

    Returns
    -------
    np.ndarray
        Hedges' g values
    """
    # Calculate degrees of freedom
    if n2 is None:
        df = n1 - 1  # One-sample or paired
    else:
        df = n1 + n2 - 2  # Two-sample

    # Correction factor (J)
    # Approximation from Hedges (1981)
    j = 1 - (3 / (4 * df - 1))

    return d * j


def calculate_effect_size_ci(
    d: np.ndarray,
    n1: int,
    n2: Optional[int] = None,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for Cohen's d.

    Uses the non-central t-distribution method.

    Parameters
    ----------
    d : np.ndarray
        Cohen's d values
    n1 : int
        Sample size for group 1
    n2 : int, optional
        Sample size for group 2
    confidence : float
        Confidence level (default: 0.95 for 95% CI)

    Returns
    -------
    tuple
        (lower_ci, upper_ci) arrays
    """
    # Calculate standard error of d
    if n2 is None:
        # One-sample or paired
        n_eff = n1
        se_d = np.sqrt(1/n1 + d**2/(2*n1))
    else:
        # Two-sample
        n_eff = (n1 * n2) / (n1 + n2)  # Harmonic mean
        se_d = np.sqrt((n1 + n2)/(n1 * n2) + d**2/(2*(n1 + n2)))

    # Calculate CI using normal approximation
    # (More complex methods exist using non-central t-distribution)
    z_crit = stats.norm.ppf((1 + confidence) / 2)

    lower_ci = d - z_crit * se_d
    upper_ci = d + z_crit * se_d

    return lower_ci, upper_ci


def create_effect_size_maps(
    tstat_file: Path,
    output_dir: Path,
    design_info: Dict,
    corrp_file: Optional[Path] = None,
    p_threshold: float = 0.05,
    calculate_hedges: bool = True,
    calculate_ci: bool = True
) -> Dict:
    """
    Create effect size maps from randomise t-statistic outputs.

    Parameters
    ----------
    tstat_file : Path
        Path to randomise t-statistic file (e.g., randomise_tstat1.nii.gz)
    output_dir : Path
        Directory for output files
    design_info : dict
        Dictionary containing:
        - 'n1': Sample size for group 1
        - 'n2': Sample size for group 2 (optional)
        - 'design_type': "one_sample", "two_sample", or "paired"
        - 'contrast_name': Name of contrast (optional)
    corrp_file : Path, optional
        Path to corrected p-value file for thresholding
    p_threshold : float
        P-value threshold for corrected maps (default: 0.05)
    calculate_hedges : bool
        Also calculate Hedges' g (default: True)
    calculate_ci : bool
        Calculate confidence intervals (default: True)

    Returns
    -------
    dict
        Dictionary with paths to output files and statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not tstat_file.exists():
        raise EffectSizeError(f"T-statistic file not found: {tstat_file}")

    # Extract design information
    n1 = design_info.get('n1')
    n2 = design_info.get('n2', None)
    design_type = design_info.get('design_type', 'two_sample')
    contrast_name = design_info.get('contrast_name', 'contrast')

    if n1 is None:
        raise EffectSizeError("n1 (sample size) must be specified in design_info")

    logger.info(f"Creating effect size maps for {contrast_name}")
    logger.info(f"  Design type: {design_type}")
    logger.info(f"  Sample sizes: n1={n1}, n2={n2}")

    # Load t-statistic map
    tstat_img = nib.load(tstat_file)
    tstat_data = tstat_img.get_fdata()

    # Calculate Cohen's d
    logger.info("Calculating Cohen's d...")
    d_data = t_to_cohens_d(tstat_data, n1, n2, design_type)

    # Save uncorrected Cohen's d
    d_uncorrected_file = output_dir / f"{contrast_name}_cohens_d_uncorrected.nii.gz"
    d_img = nib.Nifti1Image(d_data, tstat_img.affine, tstat_img.header)
    nib.save(d_img, d_uncorrected_file)
    logger.info(f"  Saved: {d_uncorrected_file}")

    # Calculate statistics
    mask = np.abs(tstat_data) > 0  # Non-zero voxels
    d_masked = d_data[mask]

    stats_dict = {
        'mean_d': float(np.mean(d_masked)),
        'median_d': float(np.median(d_masked)),
        'std_d': float(np.std(d_masked)),
        'min_d': float(np.min(d_masked)),
        'max_d': float(np.max(d_masked)),
        'n_voxels': int(np.sum(mask))
    }

    # Calculate percentage of voxels in each effect size category
    small_effect = np.sum((np.abs(d_masked) >= 0.2) & (np.abs(d_masked) < 0.5))
    medium_effect = np.sum((np.abs(d_masked) >= 0.5) & (np.abs(d_masked) < 0.8))
    large_effect = np.sum(np.abs(d_masked) >= 0.8)

    stats_dict['percent_small'] = 100 * small_effect / len(d_masked)
    stats_dict['percent_medium'] = 100 * medium_effect / len(d_masked)
    stats_dict['percent_large'] = 100 * large_effect / len(d_masked)

    logger.info(f"  Mean Cohen's d: {stats_dict['mean_d']:.3f}")
    logger.info(f"  Effect sizes: {stats_dict['percent_small']:.1f}% small, "
                f"{stats_dict['percent_medium']:.1f}% medium, "
                f"{stats_dict['percent_large']:.1f}% large")

    output_files = {'d_uncorrected': str(d_uncorrected_file)}

    # Calculate Hedges' g if requested
    if calculate_hedges:
        logger.info("Calculating Hedges' g...")
        g_data = hedges_g(d_data, n1, n2)
        g_file = output_dir / f"{contrast_name}_hedges_g_uncorrected.nii.gz"
        g_img = nib.Nifti1Image(g_data, tstat_img.affine, tstat_img.header)
        nib.save(g_img, g_file)
        output_files['g_uncorrected'] = str(g_file)
        logger.info(f"  Saved: {g_file}")

    # Calculate confidence intervals if requested
    if calculate_ci:
        logger.info("Calculating 95% confidence intervals...")
        lower_ci, upper_ci = calculate_effect_size_ci(d_data, n1, n2)

        # Save CI maps
        lower_file = output_dir / f"{contrast_name}_cohens_d_ci_lower.nii.gz"
        upper_file = output_dir / f"{contrast_name}_cohens_d_ci_upper.nii.gz"

        lower_img = nib.Nifti1Image(lower_ci, tstat_img.affine, tstat_img.header)
        upper_img = nib.Nifti1Image(upper_ci, tstat_img.affine, tstat_img.header)

        nib.save(lower_img, lower_file)
        nib.save(upper_img, upper_file)

        output_files['ci_lower'] = str(lower_file)
        output_files['ci_upper'] = str(upper_file)

        logger.info(f"  Saved CI maps: {lower_file.name}, {upper_file.name}")

    # Create thresholded maps if corrected p-values provided
    if corrp_file and corrp_file.exists():
        logger.info(f"Creating thresholded maps (p < {p_threshold})...")

        corrp_img = nib.load(corrp_file)
        corrp_data = corrp_img.get_fdata()

        # FSL randomise outputs 1-p, so threshold at 1-p_threshold
        sig_mask = corrp_data >= (1 - p_threshold)

        # Apply threshold to Cohen's d
        d_thresholded = np.where(sig_mask, d_data, 0)
        d_thresh_file = output_dir / f"{contrast_name}_cohens_d_p{p_threshold:.2f}.nii.gz"
        d_thresh_img = nib.Nifti1Image(d_thresholded, tstat_img.affine, tstat_img.header)
        nib.save(d_thresh_img, d_thresh_file)

        output_files['d_thresholded'] = str(d_thresh_file)

        # Calculate statistics for significant voxels
        sig_voxels = d_thresholded[sig_mask]
        if len(sig_voxels) > 0:
            stats_dict['sig_mean_d'] = float(np.mean(sig_voxels))
            stats_dict['sig_n_voxels'] = int(np.sum(sig_mask))
            logger.info(f"  Significant voxels: {stats_dict['sig_n_voxels']}")
            logger.info(f"  Mean d (significant): {stats_dict['sig_mean_d']:.3f}")
        else:
            logger.info("  No significant voxels found")

    # Save statistics
    stats_file = output_dir / f"{contrast_name}_effect_size_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    output_files['stats'] = str(stats_file)

    # Create visualization
    logger.info("Creating effect size visualizations...")
    create_effect_size_plot(
        d_uncorrected_file,
        output_dir / f"{contrast_name}_cohens_d_plot.png",
        title=f"Cohen's d: {contrast_name}",
        threshold=0.2  # Show effects |d| > 0.2
    )

    return {
        'output_files': output_files,
        'statistics': stats_dict,
        'design_info': design_info
    }


def create_effect_size_plot(
    effect_size_file: Path,
    output_file: Path,
    title: str = "Effect Size (Cohen's d)",
    threshold: float = 0.2,
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None
):
    """
    Create a visualization of effect size map.

    Parameters
    ----------
    effect_size_file : Path
        Path to effect size NIfTI file
    output_file : Path
        Output path for plot
    title : str
        Plot title
    threshold : float
        Minimum absolute value to display (default: 0.2)
    cmap : str
        Colormap (default: RdBu_r for diverging)
    vmax : float, optional
        Maximum value for colormap scaling
    """
    import matplotlib.pyplot as plt

    # Load effect size map
    img = nib.load(effect_size_file)

    # Determine display threshold
    if vmax is None:
        data = img.get_fdata()
        vmax = np.percentile(np.abs(data[data != 0]), 95)
        vmax = max(vmax, 1.0)  # At least show up to d=1

    # Create the plot
    fig = plt.figure(figsize=(12, 8))

    # Use nilearn for brain visualization
    display = plotting.plot_stat_map(
        img,
        threshold=threshold,
        display_mode='ortho',
        cut_coords=[0, 0, 0],
        title=title,
        colorbar=True,
        cmap=cmap,
        vmax=vmax,
        symmetric_cbar=True,
        figure=fig
    )

    # Add effect size interpretation
    fig.text(0.5, 0.02,
             'Effect Size: Small (|d|=0.2), Medium (|d|=0.5), Large (|d|=0.8)',
             ha='center', fontsize=10, style='italic')

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot: {output_file}")


def batch_effect_size_calculation(
    randomise_dir: Path,
    design_file: Path,
    output_dir: Path,
    contrast_names: Optional[Dict[int, str]] = None
) -> Dict:
    """
    Calculate effect sizes for all contrasts from a randomise analysis.

    Parameters
    ----------
    randomise_dir : Path
        Directory containing randomise outputs
    design_file : Path
        Path to design matrix file (for extracting sample sizes)
    output_dir : Path
        Output directory for effect size maps
    contrast_names : dict, optional
        Dictionary mapping contrast numbers to names

    Returns
    -------
    dict
        Results for all contrasts
    """
    randomise_dir = Path(randomise_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse design matrix to get sample sizes
    design_info = parse_design_matrix(design_file)

    # Find all t-statistic files
    tstat_files = sorted(randomise_dir.glob("randomise_tstat*.nii.gz"))

    if not tstat_files:
        raise EffectSizeError(f"No t-statistic files found in {randomise_dir}")

    logger.info(f"Found {len(tstat_files)} t-statistic files")

    results = {}

    for tstat_file in tstat_files:
        # Extract contrast number
        contrast_num = int(tstat_file.stem.split('tstat')[-1])

        # Get contrast name
        if contrast_names and contrast_num in contrast_names:
            contrast_name = contrast_names[contrast_num]
        else:
            contrast_name = f"contrast{contrast_num}"

        logger.info(f"\nProcessing contrast {contrast_num}: {contrast_name}")

        # Look for corresponding corrected p-value file
        corrp_file = randomise_dir / f"randomise_tfce_corrp_tstat{contrast_num}.nii.gz"
        if not corrp_file.exists():
            corrp_file = None
            logger.info("  No corrected p-value file found")

        # Create effect size maps
        contrast_output_dir = output_dir / contrast_name

        result = create_effect_size_maps(
            tstat_file=tstat_file,
            output_dir=contrast_output_dir,
            design_info={**design_info, 'contrast_name': contrast_name},
            corrp_file=corrp_file
        )

        results[contrast_name] = result

    # Save summary
    summary_file = output_dir / "effect_size_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nEffect size calculation complete. Summary: {summary_file}")

    return results


def parse_design_matrix(design_file: Path) -> Dict:
    """
    Parse FSL design matrix to extract sample sizes.

    Parameters
    ----------
    design_file : Path
        Path to .mat or .csv file

    Returns
    -------
    dict
        Design information including sample sizes
    """
    design_file = Path(design_file)

    if design_file.suffix == '.csv':
        # CSV format
        df = pd.read_csv(design_file)
        n_subjects = len(df)

        # Try to determine design type from columns
        if 'group' in df.columns:
            groups = df['group'].unique()
            if len(groups) == 2:
                n1 = len(df[df['group'] == groups[0]])
                n2 = len(df[df['group'] == groups[1]])
                design_type = "two_sample"
            else:
                n1 = n_subjects
                n2 = None
                design_type = "one_sample"
        else:
            n1 = n_subjects
            n2 = None
            design_type = "one_sample"

    elif design_file.suffix == '.mat':
        # FSL .mat format
        with open(design_file, 'r') as f:
            lines = f.readlines()

        # Extract NumPoints
        for line in lines:
            if '/NumPoints' in line:
                n_subjects = int(line.split()[-1])
                break

        # For now, assume one-sample unless we can determine otherwise
        # (Would need to parse the actual matrix to determine design type)
        n1 = n_subjects
        n2 = None
        design_type = "one_sample"

    else:
        raise EffectSizeError(f"Unsupported design file format: {design_file.suffix}")

    return {
        'n1': n1,
        'n2': n2,
        'design_type': design_type,
        'n_subjects': n_subjects
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example: Calculate effect sizes from randomise output
    randomise_dir = Path("/path/to/randomise/output")
    design_file = Path("/path/to/design.csv")
    output_dir = Path("/path/to/effect_sizes")

    # Define contrast names
    contrasts = {
        1: "patients_vs_controls",
        2: "controls_vs_patients",
        3: "treatment_effect"
    }

    # Run batch calculation
    results = batch_effect_size_calculation(
        randomise_dir=randomise_dir,
        design_file=design_file,
        output_dir=output_dir,
        contrast_names=contrasts
    )

    print("\nEffect Size Summary:")
    for contrast, result in results.items():
        stats = result['statistics']
        print(f"\n{contrast}:")
        print(f"  Mean |d|: {abs(stats['mean_d']):.3f}")
        print(f"  Effect sizes: {stats['percent_small']:.1f}% small, "
              f"{stats['percent_medium']:.1f}% medium, "
              f"{stats['percent_large']:.1f}% large")
        if 'sig_n_voxels' in stats:
            print(f"  Significant voxels: {stats['sig_n_voxels']}")