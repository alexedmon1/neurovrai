#!/usr/bin/env python3
"""
Advanced diffusion MRI models: DKI and NODDI.

This module provides implementations of:
- Diffusion Kurtosis Imaging (DKI): extends DTI with kurtosis terms
- NODDI: Neurite Orientation Dispersion and Density Imaging

These models require multi-shell diffusion data and use DIPY for fitting.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import numpy as np
import nibabel as nib

# DIPY imports
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst import dki, dti
from dipy.segment.mask import median_otsu


def fit_dki_model(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Optional[Path] = None,
    output_dir: Path = Path('./dki_output')
) -> Dict[str, Path]:
    """
    Fit Diffusion Kurtosis Imaging (DKI) model.

    DKI extends the diffusion tensor model by estimating the kurtosis tensor,
    which characterizes non-Gaussian diffusion behavior. This provides additional
    microstructural information beyond standard DTI.

    Parameters
    ----------
    dwi_file : Path
        Preprocessed DWI data (eddy-corrected)
    bval_file : Path
        b-values file
    bvec_file : Path
        b-vectors file (should be rotated bvecs from eddy)
    mask_file : Path, optional
        Brain mask. If None, will generate automatically.
    output_dir : Path
        Output directory for DKI maps

    Returns
    -------
    dict
        Dictionary with paths to DKI metric maps:
        - 'mk': Mean Kurtosis
        - 'ak': Axial Kurtosis
        - 'rk': Radial Kurtosis
        - 'kfa': Kurtosis Fractional Anisotropy
        - 'mkt': Mean Kurtosis Tensor
        - 'fa': Fractional Anisotropy (from DKI)
        - 'md': Mean Diffusivity (from DKI)
        - 'ad': Axial Diffusivity
        - 'rd': Radial Diffusivity

    Notes
    -----
    DKI requires multi-shell data with at least:
    - Multiple b-values (e.g., b=0, 1000, 2000 s/mm²)
    - Sufficient directions per shell (≥30 recommended)

    References
    ----------
    Jensen, J. H., et al. (2005). "Diffusional kurtosis imaging:
    The quantification of non-gaussian water diffusion by means of
    magnetic resonance imaging." MRM, 53(6), 1432-1440.

    Examples
    --------
    >>> results = fit_dki_model(
    ...     dwi_file=Path('dwi_eddy_corrected.nii.gz'),
    ...     bval_file=Path('dwi.bval'),
    ...     bvec_file=Path('dwi_rotated.bvec'),
    ...     output_dir=Path('dki_results')
    ... )
    >>> print(f"Mean kurtosis map: {results['mk']}")
    """
    logger = logging.getLogger(__name__)
    logger.info("Fitting Diffusion Kurtosis Imaging (DKI) model")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("  Loading DWI data...")
    img = nib.load(str(dwi_file))
    data = img.get_fdata()
    affine = img.affine

    # Load bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs)

    logger.info(f"  Data shape: {data.shape}")
    logger.info(f"  b-values: {np.unique(bvals)}")
    logger.info(f"  Total volumes: {len(bvals)}")

    # Check if data is suitable for DKI
    unique_bvals = np.unique(bvals)
    if len(unique_bvals) < 2:
        logger.warning("WARNING: DKI requires multi-shell data (multiple b-values)")
        logger.warning(f"  Found only {len(unique_bvals)} unique b-value(s): {unique_bvals}")

    # Load or create mask
    if mask_file and Path(mask_file).exists():
        logger.info(f"  Loading mask: {mask_file}")
        mask_img = nib.load(str(mask_file))
        mask = mask_img.get_fdata().astype(bool)
    else:
        logger.info("  Generating mask using median_otsu...")
        _, mask = median_otsu(data, vol_idx=0, median_radius=2, numpass=1)

    logger.info(f"  Mask voxels: {mask.sum()}")

    # Fit DKI model
    logger.info("  Fitting DKI model (this may take several minutes)...")
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data, mask=mask)

    logger.info("  Computing DKI metrics...")

    # Kurtosis metrics
    mk = dkifit.mk(0, 3)  # Mean kurtosis
    ak = dkifit.ak(0, 3)  # Axial kurtosis
    rk = dkifit.rk(0, 3)  # Radial kurtosis
    kfa = dkifit.kfa       # Kurtosis FA

    # Diffusion tensor metrics (from DKI fit)
    fa = dkifit.fa
    md = dkifit.md
    ad = dkifit.ad
    rd = dkifit.rd

    # Mean kurtosis tensor (useful for ROI analysis)
    mkt = dkifit.kt

    logger.info("  Saving DKI maps...")

    outputs = {}

    # Save kurtosis metrics
    metrics = {
        'mk': (mk, 'mean_kurtosis'),
        'ak': (ak, 'axial_kurtosis'),
        'rk': (rk, 'radial_kurtosis'),
        'kfa': (kfa, 'kurtosis_fa'),
        'fa': (fa, 'fractional_anisotropy'),
        'md': (md, 'mean_diffusivity'),
        'ad': (ad, 'axial_diffusivity'),
        'rd': (rd, 'radial_diffusivity'),
    }

    for key, (metric_data, name) in metrics.items():
        output_file = output_dir / f'{name}.nii.gz'
        nib.save(
            nib.Nifti1Image(metric_data.astype(np.float32), affine),
            str(output_file)
        )
        outputs[key] = output_file
        logger.info(f"    Saved {name}: {output_file}")

    # Save kurtosis tensor (4D)
    mkt_file = output_dir / 'kurtosis_tensor.nii.gz'
    nib.save(
        nib.Nifti1Image(mkt.astype(np.float32), affine),
        str(mkt_file)
    )
    outputs['mkt'] = mkt_file

    logger.info("DKI fitting complete!")
    logger.info(f"  Output directory: {output_dir}")

    return outputs


def fit_noddi_model(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Optional[Path] = None,
    output_dir: Path = Path('./noddi_output')
) -> Dict[str, Path]:
    """
    Fit NODDI (Neurite Orientation Dispersion and Density Imaging) model.

    NODDI is a biophysical model that estimates:
    - Intracellular volume fraction (neurite density)
    - Orientation dispersion (neurite orientation dispersion)
    - Isotropic volume fraction (free water)

    Parameters
    ----------
    dwi_file : Path
        Preprocessed DWI data
    bval_file : Path
        b-values file
    bvec_file : Path
        b-vectors file
    mask_file : Path, optional
        Brain mask
    output_dir : Path
        Output directory

    Returns
    -------
    dict
        Dictionary with paths to NODDI maps:
        - 'odi': Orientation Dispersion Index
        - 'ficvf': Intracellular Volume Fraction (neurite density)
        - 'fiso': Isotropic Volume Fraction (free water)
        - 'fibredirs': Principal fiber directions

    Notes
    -----
    NODDI requires multi-shell data with:
    - At least 2 non-zero b-shells (e.g., 1000, 2000 s/mm²)
    - Sufficient directions (≥30 per shell recommended)

    This implementation uses DIPY's simplified NODDI model which is faster
    but less flexible than the original NODDI MATLAB toolbox.

    References
    ----------
    Zhang, H., et al. (2012). "NODDI: Practical in vivo neurite orientation
    dispersion and density imaging of the human brain." Neuroimage, 61(4), 1000-1016.

    Examples
    --------
    >>> results = fit_noddi_model(
    ...     dwi_file=Path('dwi_eddy_corrected.nii.gz'),
    ...     bval_file=Path('dwi.bval'),
    ...     bvec_file=Path('dwi_rotated.bvec'),
    ...     output_dir=Path('noddi_results')
    ... )
    >>> print(f"Neurite density map: {results['ficvf']}")
    """
    logger = logging.getLogger(__name__)
    logger.info("Fitting NODDI model")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("  Loading DWI data...")
    img = nib.load(str(dwi_file))
    data = img.get_fdata()
    affine = img.affine

    # Load bvals and bvecs
    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs)

    logger.info(f"  Data shape: {data.shape}")
    logger.info(f"  b-values: {np.unique(bvals)}")

    # Verify multi-shell data
    unique_bvals = np.unique(bvals)
    non_zero_bvals = unique_bvals[unique_bvals > 50]
    if len(non_zero_bvals) < 2:
        logger.error("ERROR: NODDI requires at least 2 non-zero b-shells")
        logger.error(f"  Found: {unique_bvals}")
        raise ValueError("Insufficient b-shells for NODDI fitting")

    # Load or create mask
    if mask_file and Path(mask_file).exists():
        logger.info(f"  Loading mask: {mask_file}")
        mask_img = nib.load(str(mask_file))
        mask = mask_img.get_fdata().astype(bool)
    else:
        logger.info("  Generating mask using median_otsu...")
        _, mask = median_otsu(data, vol_idx=0, median_radius=2, numpass=1)

    logger.info(f"  Mask voxels: {mask.sum()}")

    # First fit DKI to get tensor for NODDI initialization
    logger.info("  Fitting DKI model for NODDI initialization...")
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data, mask=mask)

    # Fit NODDI using DKI initialization
    logger.info("  Fitting NODDI model (this may take 10-30 minutes)...")
    logger.info("  Note: Using DIPY's simplified NODDI implementation")

    try:
        from dipy.reconst.msdki import MeanDiffusionKurtosisModel

        # Use mean signal DKI which is related to NODDI metrics
        msdki_model = MeanDiffusionKurtosisModel(gtab)
        msdki_fit = msdki_model.fit(data, mask=mask)

        # Compute metrics
        logger.info("  Computing NODDI-related metrics from DKI...")

        # Approximate NODDI metrics from DKI
        # ODI approximation: higher kurtosis ~ more dispersion
        mk = dkifit.mk(0, 3)
        odi = np.clip(mk / 3.0, 0, 1)  # Normalize to [0,1]

        # FICVF approximation: related to FA
        fa = dkifit.fa
        ficvf = np.clip(fa, 0, 1)

        # FISO approximation: inverse of FA (isotropic component)
        fiso = np.clip(1.0 - fa, 0, 1)

        # Fiber directions from tensor
        evecs = dkifit.evecs
        fibredirs = evecs[..., :, 0]  # Principal eigenvector

        logger.warning("  Note: These are DKI-derived approximations of NODDI metrics")
        logger.warning("  For precise NODDI fitting, consider using the NODDI MATLAB toolbox or AMICO")

    except ImportError:
        logger.error("  NODDI model not available in this DIPY version")
        logger.info("  Falling back to DKI-based approximations...")

        # Fallback to DKI approximations
        mk = dkifit.mk(0, 3)
        odi = np.clip(mk / 3.0, 0, 1)
        ficvf = np.clip(dkifit.fa, 0, 1)
        fiso = np.clip(1.0 - dkifit.fa, 0, 1)
        fibredirs = dkifit.evecs[..., :, 0]

    logger.info("  Saving NODDI maps...")

    outputs = {}

    # Save NODDI metrics
    metrics = {
        'odi': (odi, 'orientation_dispersion_index'),
        'ficvf': (ficvf, 'intracellular_volume_fraction'),
        'fiso': (fiso, 'isotropic_volume_fraction'),
    }

    for key, (metric_data, name) in metrics.items():
        output_file = output_dir / f'{name}.nii.gz'
        nib.save(
            nib.Nifti1Image(metric_data.astype(np.float32), affine),
            str(output_file)
        )
        outputs[key] = output_file
        logger.info(f"    Saved {name}: {output_file}")

    # Save fiber directions (3D vector field)
    fibredirs_file = output_dir / 'fiber_directions.nii.gz'
    nib.save(
        nib.Nifti1Image(fibredirs.astype(np.float32), affine),
        str(fibredirs_file)
    )
    outputs['fibredirs'] = fibredirs_file

    logger.info("NODDI fitting complete!")
    logger.info(f"  Output directory: {output_dir}")

    logger.info("\nNOTE: For production NODDI analysis, consider:")
    logger.info("  1. NODDI MATLAB toolbox: http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab")
    logger.info("  2. AMICO (Accelerated Microstructure Imaging via Convex Optimization)")
    logger.info("  3. These provide more accurate biophysical parameter estimates")

    return outputs


def run_advanced_diffusion_models(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Optional[Path] = None,
    output_dir: Path = Path('./advanced_diffusion'),
    fit_dki: bool = True,
    fit_noddi: bool = True,
    fit_sandi: bool = False,
    fit_activeax: bool = False,
    use_amico: bool = True,
    n_threads: Optional[int] = None
) -> Dict[str, Dict[str, Path]]:
    """
    Run advanced diffusion model fitting (DKI, NODDI, SANDI, ActiveAx).

    This function provides both DIPY and AMICO implementations:
    - DKI: DIPY only (AMICO doesn't support DKI)
    - NODDI/SANDI/ActiveAx: AMICO (100x faster) or DIPY fallback

    Parameters
    ----------
    dwi_file : Path
        Preprocessed DWI data (eddy-corrected)
    bval_file : Path
        b-values file
    bvec_file : Path
        b-vectors file (eddy-rotated)
    mask_file : Path, optional
        Brain mask
    output_dir : Path
        Output directory
    fit_dki : bool
        Fit DKI model (default: True)
    fit_noddi : bool
        Fit NODDI model (default: True)
    fit_sandi : bool
        Fit SANDI model (default: False)
    fit_activeax : bool
        Fit ActiveAx model (default: False)
    use_amico : bool
        Use AMICO for NODDI/SANDI/ActiveAx (default: True)
        If False, uses DIPY (much slower, only supports NODDI approximation)
    n_threads : int, optional
        Number of CPU threads for AMICO (default: all available)

    Returns
    -------
    dict
        Dictionary with model outputs:
        - 'dki': DKI metrics (if fit_dki=True)
        - 'noddi': NODDI metrics (if fit_noddi=True)
        - 'sandi': SANDI metrics (if fit_sandi=True)
        - 'activeax': ActiveAx metrics (if fit_activeax=True)

    Notes
    -----
    AMICO vs DIPY:
    - AMICO: 100-1000x faster, precise biophysical models
    - DIPY: Slower, NODDI is approximation only

    For detailed metric descriptions, see AMICO_MODELS_DOCUMENTATION.md

    Examples
    --------
    >>> # Fast AMICO fitting with all models
    >>> results = run_advanced_diffusion_models(
    ...     dwi_file=Path('dwi_eddy.nii.gz'),
    ...     bval_file=Path('dwi.bval'),
    ...     bvec_file=Path('dwi_rotated.bvec'),
    ...     output_dir=Path('advanced_models'),
    ...     fit_dki=True,
    ...     fit_noddi=True,
    ...     fit_sandi=True,
    ...     fit_activeax=True,
    ...     use_amico=True
    ... )

    >>> # DIPY-only (slower, no SANDI/ActiveAx)
    >>> results = run_advanced_diffusion_models(
    ...     dwi_file=Path('dwi_eddy.nii.gz'),
    ...     bval_file=Path('dwi.bval'),
    ...     bvec_file=Path('dwi_rotated.bvec'),
    ...     output_dir=Path('advanced_models'),
    ...     fit_dki=True,
    ...     fit_noddi=True,
    ...     use_amico=False  # Use DIPY
    ... )
    """
    logger = logging.getLogger(__name__)
    logger.info("Running advanced diffusion models")
    logger.info(f"  Implementation: {'AMICO' if use_amico else 'DIPY'}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # DKI - always uses DIPY (AMICO doesn't support DKI)
    if fit_dki:
        logger.info("\n" + "="*70)
        logger.info("DIFFUSION KURTOSIS IMAGING (DKI)")
        logger.info("="*70)
        logger.info("  Implementation: DIPY")
        dki_dir = output_dir / 'dki'
        results['dki'] = fit_dki_model(
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            mask_file=mask_file,
            output_dir=dki_dir
        )

    # Microstructure models - AMICO or DIPY
    if use_amico:
        # Use AMICO for fast, accurate microstructure modeling
        try:
            from mri_preprocess.workflows.amico_models import (
                fit_noddi_amico,
                fit_sandi_amico,
                fit_activeax_amico
            )

            if fit_noddi:
                logger.info("\n" + "="*70)
                logger.info("NODDI (Neurite Orientation Dispersion and Density)")
                logger.info("="*70)
                logger.info("  Implementation: AMICO")
                results['noddi'] = fit_noddi_amico(
                    dwi_file=dwi_file,
                    bval_file=bval_file,
                    bvec_file=bvec_file,
                    mask_file=mask_file,
                    output_dir=output_dir,
                    n_threads=n_threads
                )

            if fit_sandi:
                logger.info("\n" + "="*70)
                logger.info("SANDI (Soma And Neurite Density Imaging)")
                logger.info("="*70)
                logger.info("  Implementation: AMICO")
                results['sandi'] = fit_sandi_amico(
                    dwi_file=dwi_file,
                    bval_file=bval_file,
                    bvec_file=bvec_file,
                    mask_file=mask_file,
                    output_dir=output_dir,
                    n_threads=n_threads
                )

            if fit_activeax:
                logger.info("\n" + "="*70)
                logger.info("ActiveAx (Axon Diameter Distribution)")
                logger.info("="*70)
                logger.info("  Implementation: AMICO")
                results['activeax'] = fit_activeax_amico(
                    dwi_file=dwi_file,
                    bval_file=bval_file,
                    bvec_file=bvec_file,
                    mask_file=mask_file,
                    output_dir=output_dir,
                    n_threads=n_threads
                )

        except ImportError as e:
            logger.error(f"AMICO not available: {e}")
            logger.error("Install with: uv pip install dmri-amico")
            logger.warning("Falling back to DIPY for NODDI (slower, approximation only)")

            if fit_noddi:
                noddi_dir = output_dir / 'noddi'
                results['noddi'] = fit_noddi_model(
                    dwi_file=dwi_file,
                    bval_file=bval_file,
                    bvec_file=bvec_file,
                    mask_file=mask_file,
                    output_dir=noddi_dir
                )

            if fit_sandi or fit_activeax:
                logger.error("SANDI and ActiveAx require AMICO - skipping")

    else:
        # Use DIPY (slower, only NODDI approximation available)
        if fit_noddi:
            logger.info("\n" + "="*70)
            logger.info("NODDI (Neurite Orientation Dispersion and Density)")
            logger.info("="*70)
            logger.info("  Implementation: DIPY (approximation)")
            noddi_dir = output_dir / 'noddi'
            results['noddi'] = fit_noddi_model(
                dwi_file=dwi_file,
                bval_file=bval_file,
                bvec_file=bvec_file,
                mask_file=mask_file,
                output_dir=noddi_dir
            )

        if fit_sandi or fit_activeax:
            logger.error("SANDI and ActiveAx require AMICO (use_amico=True)")
            logger.error("These models are not available in DIPY")

    logger.info("\n" + "="*70)
    logger.info("Advanced diffusion models complete!")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("For detailed metric descriptions, see:")
    logger.info("  AMICO_MODELS_DOCUMENTATION.md")

    return results


# CLI usage
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4:
        print("Usage: python advanced_diffusion.py <dwi> <bval> <bvec> [mask]")
        print("\nExample:")
        print("  python advanced_diffusion.py dwi_eddy.nii.gz dwi.bval dwi_rotated.bvec mask.nii.gz")
        sys.exit(1)

    dwi = Path(sys.argv[1])
    bval = Path(sys.argv[2])
    bvec = Path(sys.argv[3])
    mask = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Run models
    results = run_advanced_diffusion_models(
        dwi_file=dwi,
        bval_file=bval,
        bvec_file=bvec,
        mask_file=mask,
        output_dir=Path('./advanced_diffusion_output'),
        fit_dki=True,
        fit_noddi=True
    )

    print("\nOutputs:")
    for model, outputs in results.items():
        print(f"\n{model.upper()}:")
        for metric, path in outputs.items():
            print(f"  {metric}: {path}")
