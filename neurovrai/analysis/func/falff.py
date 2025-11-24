#!/usr/bin/env python3
"""
fALFF (fractional Amplitude of Low-Frequency Fluctuations) Analysis

ALFF measures the amplitude of low-frequency fluctuations in the BOLD signal.
fALFF is the ratio of ALFF to total amplitude across all frequencies, which
makes it more robust to physiological noise.

References:
- Zou et al. (2008). An improved approach to detection of amplitude of
  low-frequency fluctuation (ALFF) for resting-state fMRI. Journal of
  Neuroscience Methods, 172(1), 137-141.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from scipy import signal
import logging


def compute_power_spectrum(timeseries: np.ndarray,
                          tr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum of a time series using FFT

    Args:
        timeseries: 1D array of BOLD signal values
        tr: Repetition time in seconds

    Returns:
        Tuple of (frequencies, power_spectrum)
    """
    n = len(timeseries)

    # Remove mean (detrend constant)
    timeseries = timeseries - np.mean(timeseries)

    # Compute FFT
    fft_vals = np.fft.fft(timeseries)

    # Compute power spectrum (magnitude squared)
    power = np.abs(fft_vals) ** 2

    # Get frequencies
    freqs = np.fft.fftfreq(n, d=tr)

    # Only keep positive frequencies
    positive_freq_idx = freqs > 0
    freqs = freqs[positive_freq_idx]
    power = power[positive_freq_idx]

    return freqs, power


def compute_alff(timeseries: np.ndarray,
                tr: float,
                low_freq: float = 0.01,
                high_freq: float = 0.08) -> Tuple[float, float]:
    """
    Compute ALFF and fALFF for a single voxel time series

    Args:
        timeseries: 1D array of BOLD signal values
        tr: Repetition time in seconds
        low_freq: Lower bound of frequency range (Hz)
        high_freq: Upper bound of frequency range (Hz)

    Returns:
        Tuple of (ALFF, fALFF)
        - ALFF: Amplitude of low-frequency fluctuations
        - fALFF: fractional ALFF (ratio to total amplitude)
    """
    # Compute power spectrum
    freqs, power = compute_power_spectrum(timeseries, tr)

    # Find indices for low-frequency range
    lf_idx = (freqs >= low_freq) & (freqs <= high_freq)

    if not np.any(lf_idx):
        return 0.0, 0.0

    # Compute ALFF (square root of sum of power in LF range)
    alff = np.sqrt(np.sum(power[lf_idx]))

    # Compute total amplitude (square root of sum of all power)
    total_amplitude = np.sqrt(np.sum(power))

    # Compute fALFF
    if total_amplitude > 0:
        falff = alff / total_amplitude
    else:
        falff = 0.0

    return float(alff), float(falff)


def compute_falff_map(func_file: Path,
                     mask_file: Optional[Path] = None,
                     tr: Optional[float] = None,
                     low_freq: float = 0.01,
                     high_freq: float = 0.08,
                     output_dir: Optional[Path] = None) -> dict:
    """
    Compute ALFF and fALFF maps for whole brain

    Args:
        func_file: Path to preprocessed 4D functional image
                  Should be in native space, after nuisance regression
        mask_file: Optional brain mask
        tr: Repetition time in seconds (will try to read from header if not provided)
        low_freq: Lower bound of frequency range (default: 0.01 Hz)
        high_freq: Upper bound of frequency range (default: 0.08 Hz)
        output_dir: Optional directory to save output maps

    Returns:
        Dictionary containing:
        - alff_data: 3D ALFF map
        - falff_data: 3D fALFF map
        - alff_img: ALFF NIfTI image
        - falff_img: fALFF NIfTI image
        - statistics: Dictionary of summary statistics
    """
    logging.info("=" * 80)
    logging.info("Computing ALFF and fALFF")
    logging.info("=" * 80)
    logging.info(f"Input: {func_file}")
    logging.info(f"Frequency range: {low_freq} - {high_freq} Hz")

    # Load functional data
    func_img = nib.load(func_file)
    func_data = func_img.get_fdata()

    if func_data.ndim != 4:
        raise ValueError(f"Expected 4D functional data, got {func_data.ndim}D")

    nx, ny, nz, nt = func_data.shape
    logging.info(f"Dimensions: {nx} x {ny} x {nz} x {nt} timepoints")

    # Get TR from header or use provided value
    if tr is None:
        try:
            # Try to get TR from NIfTI header
            tr = float(func_img.header.get_zooms()[3])
            logging.info(f"TR from header: {tr} seconds")
        except:
            raise ValueError("TR not found in header and not provided")
    else:
        logging.info(f"Using provided TR: {tr} seconds")

    # Calculate frequency resolution
    freq_resolution = 1.0 / (nt * tr)
    nyquist_freq = 1.0 / (2 * tr)
    logging.info(f"Frequency resolution: {freq_resolution:.4f} Hz")
    logging.info(f"Nyquist frequency: {nyquist_freq:.4f} Hz")

    if high_freq > nyquist_freq:
        logging.warning(f"High frequency ({high_freq} Hz) exceeds Nyquist ({nyquist_freq:.4f} Hz)")
        logging.warning(f"Setting high frequency to Nyquist")
        high_freq = nyquist_freq

    # Load mask
    if mask_file is not None:
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata().astype(bool)
        logging.info(f"Using brain mask: {mask_file}")
        logging.info(f"  Brain voxels: {np.sum(mask_data)}")
    else:
        # Create mask from non-zero voxels
        mask_data = np.any(func_data != 0, axis=-1)
        logging.info("No mask provided, using non-zero voxels")
        logging.info(f"  Brain voxels: {np.sum(mask_data)}")

    # Initialize ALFF and fALFF maps
    alff_data = np.zeros((nx, ny, nz), dtype=np.float32)
    falff_data = np.zeros((nx, ny, nz), dtype=np.float32)

    # Compute ALFF/fALFF for each voxel
    brain_voxels = np.where(mask_data)
    n_voxels = len(brain_voxels[0])

    logging.info(f"Computing ALFF/fALFF for {n_voxels} voxels...")

    for i, (x, y, z) in enumerate(zip(*brain_voxels)):
        if (i + 1) % 10000 == 0:
            logging.info(f"  Progress: {i+1}/{n_voxels} voxels ({100*(i+1)/n_voxels:.1f}%)")

        # Extract time series
        timeseries = func_data[x, y, z, :]

        # Skip if all zeros or constant
        if np.std(timeseries) == 0:
            continue

        # Compute ALFF and fALFF
        alff, falff = compute_alff(timeseries, tr, low_freq, high_freq)

        alff_data[x, y, z] = alff
        falff_data[x, y, z] = falff

    logging.info("  ✓ ALFF/fALFF computation complete")

    # Create output images
    alff_img = nib.Nifti1Image(alff_data, func_img.affine, func_img.header)
    falff_img = nib.Nifti1Image(falff_data, func_img.affine, func_img.header)

    # Save if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        alff_file = output_dir / 'alff.nii.gz'
        falff_file = output_dir / 'falff.nii.gz'

        nib.save(alff_img, alff_file)
        nib.save(falff_img, falff_file)

        logging.info(f"  Saved ALFF: {alff_file}")
        logging.info(f"  Saved fALFF: {falff_file}")

    # Compute statistics
    brain_alff = alff_data[mask_data]
    brain_falff = falff_data[mask_data]

    statistics = {
        'alff': {
            'mean': float(np.mean(brain_alff)),
            'std': float(np.std(brain_alff)),
            'min': float(np.min(brain_alff)),
            'max': float(np.max(brain_alff))
        },
        'falff': {
            'mean': float(np.mean(brain_falff)),
            'std': float(np.std(brain_falff)),
            'min': float(np.min(brain_falff)),
            'max': float(np.max(brain_falff))
        }
    }

    logging.info("\nALFF Statistics:")
    logging.info(f"  Mean: {statistics['alff']['mean']:.4f}")
    logging.info(f"  Std:  {statistics['alff']['std']:.4f}")
    logging.info(f"  Min:  {statistics['alff']['min']:.4f}")
    logging.info(f"  Max:  {statistics['alff']['max']:.4f}")

    logging.info("\nfALFF Statistics:")
    logging.info(f"  Mean: {statistics['falff']['mean']:.4f}")
    logging.info(f"  Std:  {statistics['falff']['std']:.4f}")
    logging.info(f"  Min:  {statistics['falff']['min']:.4f}")
    logging.info(f"  Max:  {statistics['falff']['max']:.4f}")
    logging.info("=" * 80)

    return {
        'alff_data': alff_data,
        'falff_data': falff_data,
        'alff_img': alff_img,
        'falff_img': falff_img,
        'statistics': statistics
    }


def compute_falff_zscore(alff_file: Path,
                        falff_file: Path,
                        mask_file: Optional[Path] = None,
                        output_dir: Optional[Path] = None) -> dict:
    """
    Standardize ALFF and fALFF maps to z-scores

    Z-score normalization makes values comparable across subjects

    Args:
        alff_file: Path to ALFF map
        falff_file: Path to fALFF map
        mask_file: Optional brain mask
        output_dir: Optional directory to save z-scored maps

    Returns:
        Dictionary containing z-scored data and images
    """
    logging.info("Standardizing ALFF/fALFF to z-scores...")

    # Load maps
    alff_img = nib.load(alff_file)
    falff_img = nib.load(falff_file)

    alff_data = alff_img.get_fdata()
    falff_data = falff_img.get_fdata()

    # Load mask
    if mask_file is not None:
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata().astype(bool)
    else:
        mask_data = (alff_data > 0) & (falff_data > 0)

    # Z-score ALFF
    brain_alff = alff_data[mask_data]
    mean_alff = np.mean(brain_alff)
    std_alff = np.std(brain_alff)

    alff_zscore = np.zeros_like(alff_data)
    alff_zscore[mask_data] = (brain_alff - mean_alff) / std_alff

    # Z-score fALFF
    brain_falff = falff_data[mask_data]
    mean_falff = np.mean(brain_falff)
    std_falff = np.std(brain_falff)

    falff_zscore = np.zeros_like(falff_data)
    falff_zscore[mask_data] = (brain_falff - mean_falff) / std_falff

    # Create images
    alff_zscore_img = nib.Nifti1Image(alff_zscore, alff_img.affine, alff_img.header)
    falff_zscore_img = nib.Nifti1Image(falff_zscore, falff_img.affine, falff_img.header)

    # Save if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        alff_zscore_file = output_dir / 'alff_zscore.nii.gz'
        falff_zscore_file = output_dir / 'falff_zscore.nii.gz'

        nib.save(alff_zscore_img, alff_zscore_file)
        nib.save(falff_zscore_img, falff_zscore_file)

        logging.info(f"  Saved z-scored ALFF: {alff_zscore_file}")
        logging.info(f"  Saved z-scored fALFF: {falff_zscore_file}")

    return {
        'alff_zscore': alff_zscore,
        'falff_zscore': falff_zscore,
        'alff_zscore_img': alff_zscore_img,
        'falff_zscore_img': falff_zscore_img
    }


if __name__ == '__main__':
    import argparse

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Compute ALFF and fALFF for resting-state fMRI"
    )
    parser.add_argument(
        '--func',
        type=Path,
        required=True,
        help='Preprocessed 4D functional image'
    )
    parser.add_argument(
        '--mask',
        type=Path,
        help='Brain mask'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--tr',
        type=float,
        help='Repetition time in seconds (will try to read from header if not provided)'
    )
    parser.add_argument(
        '--low-freq',
        type=float,
        default=0.01,
        help='Lower frequency bound (default: 0.01 Hz)'
    )
    parser.add_argument(
        '--high-freq',
        type=float,
        default=0.08,
        help='Upper frequency bound (default: 0.08 Hz)'
    )
    parser.add_argument(
        '--zscore',
        action='store_true',
        help='Also save z-scored maps'
    )

    args = parser.parse_args()

    # Compute ALFF/fALFF
    result = compute_falff_map(
        func_file=args.func,
        mask_file=args.mask,
        tr=args.tr,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        output_dir=args.output_dir
    )

    # Compute z-scored versions
    if args.zscore:
        alff_file = args.output_dir / 'alff.nii.gz'
        falff_file = args.output_dir / 'falff.nii.gz'

        compute_falff_zscore(
            alff_file=alff_file,
            falff_file=falff_file,
            mask_file=args.mask,
            output_dir=args.output_dir
        )
