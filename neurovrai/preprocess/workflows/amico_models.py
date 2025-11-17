#!/usr/bin/env python3
"""
AMICO-based microstructure modeling for diffusion MRI.

This module provides fast implementations of advanced diffusion models using
AMICO (Accelerated Microstructure Imaging via Convex Optimization):

- NODDI: Neurite Orientation Dispersion and Density Imaging
- SANDI: Soma And Neurite Density Imaging
- ActiveAx: Axon diameter distribution modeling

AMICO is 100-1000x faster than traditional optimization approaches by reformulating
microstructure models as linear inverse problems with pre-computed dictionaries.

References:
    Daducci et al. (2015) "Accelerated Microstructure Imaging via Convex Optimization (AMICO)
    from diffusion MRI data" NeuroImage 105:32-44
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)

# Note: AMICO uses pkg_resources which is deprecated. The warning is from AMICO's code,
# not ours. AMICO should update to use importlib.metadata instead.


def prepare_amico_data(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Prepare DWI data for AMICO processing.

    AMICO requires data organized in a specific directory structure:
    - dwi.nii.gz (4D diffusion data)
    - dwi.bval (b-values)
    - dwi.bvec (b-vectors)
    - mask.nii.gz (brain mask)
    - AMICO/ (output directory)

    Args:
        dwi_file: Path to preprocessed DWI data (eddy-corrected)
        bval_file: Path to b-values file
        bvec_file: Path to b-vectors file (eddy-rotated)
        mask_file: Path to brain mask
        output_dir: Directory for AMICO outputs

    Returns:
        Tuple of (study_dir, subject_id) for AMICO processing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create AMICO directory structure
    study_dir = output_dir / 'amico_workspace'
    subject_id = 'subject'
    subject_dir = study_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing AMICO data structure in {study_dir}")

    # Copy/link input files to AMICO structure
    import shutil

    dwi_link = subject_dir / 'dwi.nii.gz'
    bval_link = subject_dir / 'dwi.bval'
    bvec_link = subject_dir / 'dwi.bvec'
    mask_link = subject_dir / 'mask.nii.gz'

    # Create symbolic links or copies
    for src, dst in [
        (dwi_file, dwi_link),
        (bval_file, bval_link),
        (bvec_file, bvec_link),
        (mask_file, mask_link)
    ]:
        if dst.exists():
            dst.unlink()

        try:
            dst.symlink_to(src.resolve())
            logger.debug(f"Linked {src.name} -> {dst}")
        except OSError:
            # Fallback to copy if symlink fails
            shutil.copy2(src, dst)
            logger.debug(f"Copied {src.name} -> {dst}")

    return study_dir, subject_id


def fit_noddi_amico(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path,
    parallel_diffusivity: float = 1.7e-3,  # mm²/s
    isotropic_diffusivity: float = 3.0e-3,  # mm²/s
    n_threads: Optional[int] = None
) -> Dict[str, Path]:
    """
    Fit NODDI (Neurite Orientation Dispersion and Density) model using AMICO.

    NODDI models brain tissue as three compartments:
    1. Intracellular (restricted): Neurites (axons + dendrites)
    2. Extracellular (hindered): Isotropic Gaussian diffusion
    3. CSF (free water): Fast isotropic diffusion

    Output Metrics:
        - FICVF (Intracellular Volume Fraction): Neurite density [0-1]
          Higher values = more densely packed neurites
        - ODI (Orientation Dispersion Index): Neurite dispersion [0-1]
          0 = perfectly aligned, 1 = maximally dispersed
        - FISO (Isotropic/CSF Volume Fraction): Free water [0-1]
          Higher in ventricles, lower in white matter
        - FIT_ICVF/FIT_OD/FIT_ISOVF: Same as above (AMICO naming)
        - DIR (Principal fiber direction): 3D vector for main orientation

    Requirements:
        - Multi-shell data (≥2 non-zero b-values)
        - Recommended: b ≤ 3000 s/mm² (higher b-values don't improve NODDI)
        - Sufficient angular resolution (≥30 directions per shell)

    Args:
        dwi_file: Path to preprocessed DWI data (eddy-corrected)
        bval_file: Path to b-values file
        bvec_file: Path to b-vectors file (eddy-rotated)
        mask_file: Path to brain mask
        output_dir: Directory for outputs
        parallel_diffusivity: Intra-axonal diffusivity (default: 1.7e-3 mm²/s)
        isotropic_diffusivity: CSF diffusivity (default: 3.0e-3 mm²/s)
        n_threads: Number of CPU threads (default: all available)

    Returns:
        Dictionary with paths to output metric files:
            - 'ficvf': Neurite density map
            - 'odi': Orientation dispersion map
            - 'fiso': Free water fraction map
            - 'dir': Fiber direction map

    Runtime: ~2-5 minutes (100x faster than traditional NODDI)
    """
    import amico

    logger.info("="*70)
    logger.info("AMICO NODDI Fitting")
    logger.info("="*70)
    logger.info(f"DWI data: {dwi_file.name}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Prepare data
    study_dir, subject_id = prepare_amico_data(
        dwi_file, bval_file, bvec_file, mask_file, output_dir
    )

    # Setup AMICO
    logger.info("Initializing AMICO...")
    amico.core.setup()

    # Load evaluation object
    ae = amico.Evaluation(str(study_dir), subject_id)

    # Convert FSL bval/bvec to AMICO scheme format
    logger.info("Converting bval/bvec to AMICO scheme format...")
    scheme_file = study_dir / subject_id / 'dwi.scheme'
    amico.util.fsl2scheme(
        bvalsFilename=str(study_dir / subject_id / 'dwi.bval'),
        bvecsFilename=str(study_dir / subject_id / 'dwi.bvec'),
        schemeFilename=str(scheme_file)
    )

    logger.info("Loading DWI data...")
    ae.load_data(
        dwi_filename='dwi.nii.gz',
        scheme_filename='dwi.scheme',
        mask_filename='mask.nii.gz',
        b0_thr=10  # Treat b < 10 as b=0
    )

    # Set NODDI model
    logger.info("Setting up NODDI model...")
    logger.info(f"  Parallel diffusivity: {parallel_diffusivity:.3e} mm²/s")
    logger.info(f"  Isotropic diffusivity: {isotropic_diffusivity:.3e} mm²/s")

    ae.set_model("NODDI")
    ae.model.set(
        dPar=parallel_diffusivity,
        dIso=isotropic_diffusivity
    )

    # Generate response functions (dictionary)
    logger.info("Generating response function dictionary...")
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()

    # Fit model
    logger.info("Fitting NODDI model (this may take a few minutes)...")
    if n_threads is not None:
        ae.fit(n_threads=n_threads)
    else:
        ae.fit()

    # Save results
    logger.info("Saving NODDI outputs...")
    ae.save_results()

    # Copy outputs to final location
    amico_out = study_dir / subject_id / 'AMICO' / 'NODDI'
    final_out = output_dir / 'noddi'
    final_out.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Map AMICO output names to standard names
    # Note: AMICO uses lowercase 'fit_' prefix
    metric_map = {
        'fit_NDI.nii.gz': 'ficvf.nii.gz',   # Neurite Density Index (intracellular volume fraction)
        'fit_ODI.nii.gz': 'odi.nii.gz',     # Orientation Dispersion Index
        'fit_FWF.nii.gz': 'fiso.nii.gz',    # Free Water Fraction (isotropic volume fraction)
        'fit_dir.nii.gz': 'dir.nii.gz'      # Fiber direction
    }

    import shutil
    for amico_name, final_name in metric_map.items():
        src = amico_out / amico_name
        dst = final_out / final_name

        if src.exists():
            shutil.copy2(src, dst)
            output_files[final_name.replace('.nii.gz', '')] = dst
            logger.info(f"  ✓ {final_name}")
        else:
            logger.warning(f"  ✗ {amico_name} not found")

    logger.info("")
    logger.info("NODDI fitting completed successfully!")
    logger.info(f"Output directory: {final_out}")
    logger.info("")

    return output_files


def fit_sandi_amico(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path,
    soma_radius_range: Tuple[float, float] = (1.0, 12.0),  # μm
    n_threads: Optional[int] = None
) -> Dict[str, Path]:
    """
    Fit SANDI (Soma And Neurite Density Imaging) model using AMICO.

    SANDI extends NODDI by explicitly modeling neuronal cell bodies (somas)
    as restricted spheres, separating soma contributions from neurites.

    Model Compartments:
    1. Soma (restricted sphere): Cell bodies with variable radius
    2. Neurite (stick): Axons and dendrites (thin cylinders)
    3. Extra-cellular (hindered): Extracellular space
    4. CSF (free water): Fast isotropic diffusion

    Output Metrics:
        - FSOMA (Soma Volume Fraction): Neuronal cell body density [0-1]
          Higher in gray matter (cortex, nuclei)
        - FNEURITE (Neurite Volume Fraction): Axon/dendrite density [0-1]
          Higher in white matter
        - FEC (Extra-cellular Volume Fraction): Extracellular space [0-1]
        - FCSF (CSF Volume Fraction): Free water [0-1]
        - RSOMA (Soma Radius): Mean soma radius [μm]
          Typical range: 5-12 μm
        - DIR (Principal neurite direction): 3D orientation vector

    Requirements:
        - High b-values (≥3000 s/mm²) for soma sensitivity
        - Multi-shell data (≥3 non-zero b-values recommended)
        - Good SNR (soma signal is weak)

    Best for: Gray matter analysis, soma size changes in disease

    Args:
        dwi_file: Path to preprocessed DWI data (eddy-corrected)
        bval_file: Path to b-values file
        bvec_file: Path to b-vectors file (eddy-rotated)
        mask_file: Path to brain mask
        output_dir: Directory for outputs
        soma_radius_range: Min and max soma radius in μm (default: 1-12 μm)
        n_threads: Number of CPU threads (default: all available)

    Returns:
        Dictionary with paths to output metric files:
            - 'fsoma': Soma volume fraction
            - 'fneurite': Neurite volume fraction
            - 'fec': Extra-cellular volume fraction
            - 'fcsf': CSF volume fraction
            - 'rsoma': Soma radius
            - 'dir': Neurite direction

    Runtime: ~3-6 minutes

    References:
        Palombo et al. (2020) "SANDI: A compartment-based model for non-invasive
        apparent soma and neurite imaging by diffusion MRI" NeuroImage 215:116835
    """
    import amico

    logger.info("="*70)
    logger.info("AMICO SANDI Fitting")
    logger.info("="*70)
    logger.info(f"DWI data: {dwi_file.name}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Prepare data
    study_dir, subject_id = prepare_amico_data(
        dwi_file, bval_file, bvec_file, mask_file, output_dir
    )

    # Setup AMICO
    logger.info("Initializing AMICO...")
    amico.core.setup()

    # Load evaluation object
    ae = amico.Evaluation(str(study_dir), subject_id)

    # Create STEJSKALTANNER scheme with gradient timing for SANDI
    logger.info("Creating STEJSKALTANNER scheme with gradient timing...")
    from ..utils.gradient_timing import get_gradient_timing, create_amico_scheme_with_timing

    # Try to get gradient timing from BIDS JSON or estimate
    bids_json = None
    # Look for BIDS JSON in common locations
    for search_path in [dwi_file.parent, dwi_file.parent.parent]:
        json_candidates = list(search_path.glob('*.json'))
        if json_candidates:
            bids_json = json_candidates[0]
            break

    try:
        TE, delta, Delta = get_gradient_timing(
            bids_json=bids_json,
            TE=None,  # Will be extracted from JSON
            manufacturer_model="Philips",  # Default, will be read from JSON if available
            allow_estimation=True
        )
    except Exception as e:
        logger.warning(f"Could not determine gradient timing: {e}")
        logger.info("Using default estimates: TE=127ms, δ=20ms, Δ=63.5ms")
        TE, delta, Delta = 0.127, 0.020, 0.0635

    scheme_file = create_amico_scheme_with_timing(
        bval_file=study_dir / subject_id / 'dwi.bval',
        bvec_file=study_dir / subject_id / 'dwi.bvec',
        output_scheme=study_dir / subject_id / 'dwi.scheme',
        TE=TE,
        delta=delta,
        Delta=Delta
    )

    logger.info("Loading DWI data...")
    ae.load_data(
        dwi_filename='dwi.nii.gz',
        scheme_filename='dwi.scheme',
        mask_filename='mask.nii.gz',
        b0_thr=10
    )

    # BUGFIX: AMICO's STEJSKALTANNER parser computes incorrect b-values
    # Manually set b-values from the original bval file
    logger.info("Correcting b-values from STEJSKALTANNER scheme...")
    bvals_correct = np.loadtxt(study_dir / subject_id / 'dwi.bval')
    logger.info(f"  Original b-values (from AMICO): min={ae.scheme.b.min():.1f}, max={ae.scheme.b.max():.1f}, unique={np.unique(ae.scheme.b)}")
    ae.scheme.b = bvals_correct
    logger.info(f"  Corrected b-values (from file): min={ae.scheme.b.min():.1f}, max={ae.scheme.b.max():.1f}, unique={np.unique(ae.scheme.b)}")

    # Recompute b0_idx/dwi_idx with corrected b-values
    ae.scheme.b0_idx = np.where(ae.scheme.b <= ae.scheme.b0_thr)[0]
    ae.scheme.dwi_idx = np.where(ae.scheme.b > ae.scheme.b0_thr)[0]
    ae.scheme.b0_count = len(ae.scheme.b0_idx)
    ae.scheme.dwi_count = len(ae.scheme.dwi_idx)
    logger.info(f"  Updated counts: b0={ae.scheme.b0_count}, DWI={ae.scheme.dwi_count}")

    # Recompute shells with correct b-values (following AMICO's internal logic)
    ae.scheme.shells = []
    tmp = np.ascontiguousarray(ae.scheme.raw[:,3:])  # G, Delta, delta, TE
    schemeUnique, schemeUniqueInd = np.unique(tmp.view([('', tmp.dtype)]*tmp.shape[1]), return_index=True)
    schemeUnique = schemeUnique.view(tmp.dtype).reshape((schemeUnique.shape[0], tmp.shape[1]))
    schemeUnique = [tmp[index] for index in sorted(schemeUniqueInd)]
    bUnique = [ae.scheme.b[index] for index in sorted(schemeUniqueInd)]

    for i in range(len(schemeUnique)):
        if bUnique[i] <= ae.scheme.b0_thr:
            continue
        shell = {}
        shell['b'] = bUnique[i]
        shell['G'] = schemeUnique[i][0]
        shell['Delta'] = schemeUnique[i][1]
        shell['delta'] = schemeUnique[i][2]
        shell['TE'] = schemeUnique[i][3]
        shell['idx'] = np.where((tmp == schemeUnique[i]).all(axis=1))[0]
        shell['grad'] = ae.scheme.raw[shell['idx'],0:3]
        ae.scheme.shells.append(shell)

    logger.info(f"  Detected {len(ae.scheme.shells)} shells:")
    for shell in ae.scheme.shells:
        logger.info(f"    b={shell['b']:.0f} s/mm² ({len(shell['idx'])} volumes)")

    # Set SANDI model
    logger.info("Setting up SANDI model...")
    logger.info(f"  Soma radius range: {soma_radius_range[0]}-{soma_radius_range[1]} μm")

    # Create soma radius array (AMICO expects array in meters, not min/max/step)
    # Convert from micrometers to meters
    Rs_um = np.arange(soma_radius_range[0], soma_radius_range[1] + 0.5, 0.5)  # in μm
    Rs_m = Rs_um * 1e-6  # convert to meters
    logger.info(f"  Soma radii: {len(Rs_m)} values from {Rs_um[0]:.1f} to {Rs_um[-1]:.1f} μm")

    ae.set_model("SANDI")
    ae.model.set(
        d_is=0.003,    # Intra-soma diffusivity (mm²/s)
        Rs=Rs_m,       # Soma radii array (meters)
        d_in=np.linspace(0.00025, 0.003, 5),  # Intra-neurite diffusivities (mm²/s)
        d_isos=np.linspace(0.00025, 0.003, 5)  # Extra-cellular diffusivities (mm²/s)
    )

    # Generate response functions
    logger.info("Generating SANDI dictionary...")
    logger.info("  (This may take longer than NODDI due to soma compartment)")
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()

    # Fit model
    logger.info("Fitting SANDI model...")
    if n_threads is not None:
        ae.fit(n_threads=n_threads)
    else:
        ae.fit()

    # Save results
    logger.info("Saving SANDI outputs...")
    ae.save_results()

    # Copy outputs to final location
    amico_out = study_dir / subject_id / 'AMICO' / 'SANDI'
    final_out = output_dir / 'sandi'
    final_out.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Map AMICO output names to standard names
    # Note: AMICO uses lowercase "fit_" prefix
    metric_map = {
        'fit_fsoma.nii.gz': 'fsoma.nii.gz',      # Soma volume fraction
        'fit_fneurite.nii.gz': 'fneurite.nii.gz', # Neurite volume fraction
        'fit_fextra.nii.gz': 'fec.nii.gz',       # Extra-cellular fraction
        'fit_Rsoma.nii.gz': 'rsoma.nii.gz',      # Soma radius (μm)
        'fit_dir.nii.gz': 'dir.nii.gz'           # Neurite direction
    }

    import shutil
    for amico_name, final_name in metric_map.items():
        src = amico_out / amico_name
        dst = final_out / final_name

        if src.exists():
            shutil.copy2(src, dst)
            output_files[final_name.replace('.nii.gz', '')] = dst
            logger.info(f"  ✓ {final_name}")
        else:
            logger.warning(f"  ✗ {amico_name} not found")

    logger.info("")
    logger.info("SANDI fitting completed successfully!")
    logger.info(f"Output directory: {final_out}")
    logger.info("")

    return output_files


def fit_activeax_amico(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path,
    axon_diameter_range: Tuple[float, float] = (0.1, 10.0),  # μm
    n_threads: Optional[int] = None
) -> Dict[str, Path]:
    """
    Fit ActiveAx (Axon Diameter Distribution) model using AMICO.

    ActiveAx specifically models axon diameter distributions in white matter
    by distinguishing between restricted (intra-axonal) and hindered
    (extra-axonal) diffusion components.

    Model Compartments:
    1. Intra-axonal (restricted cylinder): Inside axons with varying diameter
    2. Extra-axonal (hindered): Extracellular space around axons

    Output Metrics:
        - FICVF (Intra-axonal Volume Fraction): Axon density [0-1]
          Higher = more densely packed axons
        - DIAM (Mean Axon Diameter): Average axon diameter [μm]
          Typical range: 0.5-5 μm in human brain
          Larger in motor pathways, smaller in sensory
        - DIR (Principal fiber direction): 3D orientation vector
        - FVF_TOT (Total fiber volume fraction): FICVF across all diameters

    Requirements:
        - Multi-shell data with high b-values (ideally ≥3000 s/mm²)
        - Strong diffusion gradients (high gradient strength)
        - Good angular sampling (≥60 directions recommended)
        - Long diffusion times for diameter sensitivity

    Best for: White matter microstructure, axon diameter mapping

    Note:
        Axon diameter estimation is challenging and requires very strong
        gradients (>300 mT/m clinical, achievable on Connectom scanner).
        Standard clinical scanners may not provide reliable diameter estimates,
        but can still provide useful FICVF measurements.

    Args:
        dwi_file: Path to preprocessed DWI data (eddy-corrected)
        bval_file: Path to b-values file
        bvec_file: Path to b-vectors file (eddy-rotated)
        mask_file: Path to brain mask
        output_dir: Directory for outputs
        axon_diameter_range: Min and max axon diameter in μm (default: 0.1-10 μm)
        n_threads: Number of CPU threads (default: all available)

    Returns:
        Dictionary with paths to output metric files:
            - 'ficvf': Intra-axonal volume fraction (axon density)
            - 'diam': Mean axon diameter
            - 'dir': Fiber direction
            - 'fvf_tot': Total fiber volume fraction

    Runtime: ~3-6 minutes

    References:
        Alexander et al. (2010) "Orientationally invariant indices of axon
        diameter and density from diffusion MRI" NeuroImage 52(4):1374-1389
    """
    import amico

    logger.info("="*70)
    logger.info("AMICO ActiveAx Fitting")
    logger.info("="*70)
    logger.info(f"DWI data: {dwi_file.name}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    logger.info("NOTE: Reliable axon diameter estimation requires strong")
    logger.info("      diffusion gradients (>300 mT/m). Standard clinical")
    logger.info("      scanners may provide limited diameter sensitivity.")
    logger.info("")

    # Prepare data
    study_dir, subject_id = prepare_amico_data(
        dwi_file, bval_file, bvec_file, mask_file, output_dir
    )

    # Setup AMICO
    logger.info("Initializing AMICO...")
    amico.core.setup()

    # Load evaluation object
    ae = amico.Evaluation(str(study_dir), subject_id)

    # Create STEJSKALTANNER scheme with gradient timing for CylinderZeppelinBall
    logger.info("Creating STEJSKALTANNER scheme with gradient timing...")
    from ..utils.gradient_timing import get_gradient_timing, create_amico_scheme_with_timing

    # Try to get gradient timing from BIDS JSON or estimate
    bids_json = None
    # Look for BIDS JSON in common locations
    for search_path in [dwi_file.parent, dwi_file.parent.parent]:
        json_candidates = list(search_path.glob('*.json'))
        if json_candidates:
            bids_json = json_candidates[0]
            break

    try:
        TE, delta, Delta = get_gradient_timing(
            bids_json=bids_json,
            TE=None,  # Will be extracted from JSON
            manufacturer_model="Philips",  # Default, will be read from JSON if available
            allow_estimation=True
        )
    except Exception as e:
        logger.warning(f"Could not determine gradient timing: {e}")
        logger.info("Using default estimates: TE=127ms, δ=20ms, Δ=63.5ms")
        TE, delta, Delta = 0.127, 0.020, 0.0635

    scheme_file = create_amico_scheme_with_timing(
        bval_file=study_dir / subject_id / 'dwi.bval',
        bvec_file=study_dir / subject_id / 'dwi.bvec',
        output_scheme=study_dir / subject_id / 'dwi.scheme',
        TE=TE,
        delta=delta,
        Delta=Delta
    )

    logger.info("Loading DWI data...")
    ae.load_data(
        dwi_filename='dwi.nii.gz',
        scheme_filename='dwi.scheme',
        mask_filename='mask.nii.gz',
        b0_thr=10
    )

    # BUGFIX: AMICO's STEJSKALTANNER parser computes incorrect b-values
    # Manually set b-values from the original bval file
    logger.info("Correcting b-values from STEJSKALTANNER scheme...")
    bvals_correct = np.loadtxt(study_dir / subject_id / 'dwi.bval')
    logger.info(f"  Original b-values (from AMICO): min={ae.scheme.b.min():.1f}, max={ae.scheme.b.max():.1f}, unique={np.unique(ae.scheme.b)}")
    ae.scheme.b = bvals_correct
    logger.info(f"  Corrected b-values (from file): min={ae.scheme.b.min():.1f}, max={ae.scheme.b.max():.1f}, unique={np.unique(ae.scheme.b)}")

    # Recompute b0_idx/dwi_idx with corrected b-values
    ae.scheme.b0_idx = np.where(ae.scheme.b <= ae.scheme.b0_thr)[0]
    ae.scheme.dwi_idx = np.where(ae.scheme.b > ae.scheme.b0_thr)[0]
    ae.scheme.b0_count = len(ae.scheme.b0_idx)
    ae.scheme.dwi_count = len(ae.scheme.dwi_idx)
    logger.info(f"  Updated counts: b0={ae.scheme.b0_count}, DWI={ae.scheme.dwi_count}")

    # Recompute shells with correct b-values (following AMICO's internal logic)
    ae.scheme.shells = []
    tmp = np.ascontiguousarray(ae.scheme.raw[:,3:])  # G, Delta, delta, TE
    schemeUnique, schemeUniqueInd = np.unique(tmp.view([('', tmp.dtype)]*tmp.shape[1]), return_index=True)
    schemeUnique = schemeUnique.view(tmp.dtype).reshape((schemeUnique.shape[0], tmp.shape[1]))
    schemeUnique = [tmp[index] for index in sorted(schemeUniqueInd)]
    bUnique = [ae.scheme.b[index] for index in sorted(schemeUniqueInd)]

    for i in range(len(schemeUnique)):
        if bUnique[i] <= ae.scheme.b0_thr:
            continue
        shell = {}
        shell['b'] = bUnique[i]
        shell['G'] = schemeUnique[i][0]
        shell['Delta'] = schemeUnique[i][1]
        shell['delta'] = schemeUnique[i][2]
        shell['TE'] = schemeUnique[i][3]
        shell['idx'] = np.where((tmp == schemeUnique[i]).all(axis=1))[0]
        shell['grad'] = ae.scheme.raw[shell['idx'],0:3]
        ae.scheme.shells.append(shell)

    logger.info(f"  Detected {len(ae.scheme.shells)} shells:")
    for shell in ae.scheme.shells:
        logger.info(f"    b={shell['b']:.0f} s/mm² ({len(shell['idx'])} volumes)")

    # Set CylinderZeppelinBall model (ActiveAx equivalent in AMICO)
    logger.info("Setting up CylinderZeppelinBall model (ActiveAx)...")
    logger.info(f"  Axon diameter range: {axon_diameter_range[0]}-{axon_diameter_range[1]} μm")

    # Create axon radius array (convert from diameter in μm to radius in meters)
    # ActiveAx uses diameter, but AMICO's Rs parameter is radius in meters
    diam_um = np.arange(axon_diameter_range[0], axon_diameter_range[1] + 0.5, 0.5)  # in μm
    Rs_m = (diam_um / 2) * 1e-6  # convert diameter to radius, then to meters
    logger.info(f"  Axon radii: {len(Rs_m)} values from {diam_um[0]/2:.2f} to {diam_um[-1]/2:.2f} μm")

    ae.set_model("CylinderZeppelinBall")

    # Set isExvivo flag for in vivo data (must be set before model.set())
    ae.model.isExvivo = False

    ae.model.set(
        d_par=1.7e-3,     # Parallel diffusivity (mm²/s)
        Rs=Rs_m,          # Axon radii array (meters)
        d_perps=np.array([1.19e-3, 0.85e-3, 0.51e-3, 0.17e-3]),  # Perpendicular diffusivities (mm²/s)
        d_isos=np.array([3.0e-3])  # Isotropic diffusivity (CSF, mm²/s)
    )

    # Generate response functions
    logger.info("Generating ActiveAx dictionary...")
    logger.info("  (Computing response functions for axon diameter distribution)")
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()

    # Fit model
    logger.info("Fitting ActiveAx model...")
    # Note: CylinderZeppelinBall model's fit() method doesn't accept n_threads parameter
    # Threading is controlled internally by the model
    ae.fit()

    # Save results
    logger.info("Saving ActiveAx outputs...")
    ae.save_results()

    # Copy outputs to final location
    amico_out = study_dir / subject_id / 'AMICO' / 'CylinderZeppelinBall'
    final_out = output_dir / 'activeax'
    final_out.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Map AMICO output names to standard names
    # Note: AMICO uses lowercase "fit_" prefix
    # CylinderZeppelinBall outputs: fit_v (volume fraction), fit_a, fit_d, fit_dir
    metric_map = {
        'fit_v.nii.gz': 'ficvf.nii.gz',     # Intra-axonal volume fraction
        'fit_d.nii.gz': 'diam.nii.gz',      # Mean axon diameter (weighted by fit_a)
        'fit_dir.nii.gz': 'dir.nii.gz'      # Fiber direction
    }

    import shutil
    for amico_name, final_name in metric_map.items():
        src = amico_out / amico_name
        dst = final_out / final_name

        if src.exists():
            shutil.copy2(src, dst)
            output_files[final_name.replace('.nii.gz', '')] = dst
            logger.info(f"  ✓ {final_name}")
        else:
            logger.warning(f"  ✗ {amico_name} not found")

    logger.info("")
    logger.info("ActiveAx fitting completed successfully!")
    logger.info(f"Output directory: {final_out}")
    logger.info("")

    return output_files


def run_all_amico_models(
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    mask_file: Path,
    output_dir: Path,
    fit_noddi: bool = True,
    fit_sandi: bool = True,
    fit_activeax: bool = True,
    n_threads: Optional[int] = None
) -> Dict[str, Dict[str, Path]]:
    """
    Run all AMICO microstructure models on DWI data.

    This convenience function runs NODDI, SANDI, and ActiveAx fitting
    in sequence on the same input data.

    Args:
        dwi_file: Path to preprocessed DWI data (eddy-corrected)
        bval_file: Path to b-values file
        bvec_file: Path to b-vectors file (eddy-rotated)
        mask_file: Path to brain mask
        output_dir: Directory for all outputs
        fit_noddi: Whether to fit NODDI model (default: True)
        fit_sandi: Whether to fit SANDI model (default: True)
        fit_activeax: Whether to fit ActiveAx model (default: True)
        n_threads: Number of CPU threads (default: all available)

    Returns:
        Dictionary with results from each model:
            {
                'noddi': {...},
                'sandi': {...},
                'activeax': {...}
            }

    Total Runtime: ~8-15 minutes for all three models
    """
    results = {}

    if fit_noddi:
        logger.info("Running NODDI model...")
        results['noddi'] = fit_noddi_amico(
            dwi_file, bval_file, bvec_file, mask_file,
            output_dir, n_threads=n_threads
        )

    if fit_sandi:
        logger.info("Running SANDI model...")
        results['sandi'] = fit_sandi_amico(
            dwi_file, bval_file, bvec_file, mask_file,
            output_dir, n_threads=n_threads
        )

    if fit_activeax:
        logger.info("Running ActiveAx model...")
        results['activeax'] = fit_activeax_amico(
            dwi_file, bval_file, bvec_file, mask_file,
            output_dir, n_threads=n_threads
        )

    logger.info("="*70)
    logger.info("All AMICO models completed!")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    return results
