#!/usr/bin/env python3
"""
Functional (rs-fMRI) preprocessing workflow with multi-echo support.

Workflow features:
1. Multi-echo denoising with TEDANA (if multi-echo data)
2. Motion correction (MCFLIRT)
3. ICA-AROMA for motion artifact removal (auto-enabled for single-echo)
4. Nuisance regression (ACompCor) using tissue masks from anatomical workflow
5. Temporal filtering (bandpass)
6. Spatial smoothing
7. Registration to MNI space using T1w→MNI transforms

Key integrations:
- Uses CSF/WM masks from anatomical FAST segmentation for ACompCor
- Reuses T1w→MNI transformations for efficient registration
- TEDANA enabled by default for multi-echo fMRI
- ICA-AROMA auto-enabled for single-echo data (primary denoising method)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
import subprocess

from nipype import Workflow, Node, MapNode
from nipype.interfaces import fsl, utility as niu, afni
from nipype.interfaces.io import DataSink
from nipype.algorithms import confounds
import nibabel as nib
import numpy as np

from neurovrai.utils.workflow import (
    setup_logging,
    get_node_config,
    get_execution_config,
    validate_inputs
)
from neurovrai.preprocess.qc.func_qc import (
    compute_motion_qc,
    compute_tsnr,
    compute_dvars,
    create_carpet_plot,
    generate_func_qc_report
)
from neurovrai.preprocess.utils.acompcor_helper import (
    run_fast_segmentation,
    register_masks_to_functional,
    prepare_acompcor_masks,
    extract_acompcor_components,
    regress_out_components
)
from neurovrai.preprocess.utils.func_normalization import normalize_func_to_mni152
from neurovrai.preprocess.utils.func_registration import (
    compute_func_mean,
    create_func_to_mni_transforms,
    apply_inverse_transform_to_masks
)
from neurovrai.preprocess.qc.func_registration_qc import generate_registration_qc_report
from neurovrai.utils.transforms import create_transform_registry

logger = logging.getLogger(__name__)


def run_tedana(
    echo_files: List[Path],
    echo_times: List[float],
    output_dir: Path,
    mask_file: Optional[Path] = None,
    tedpca: any = 0.95,  # Can be 'kundu', 'aic', 'kic', 'mdl', float (variance), or int (n_components)
    tree: str = 'kundu'
) -> Dict[str, Path]:
    """
    Run TEDANA multi-echo denoising.

    Parameters
    ----------
    echo_files : list of Path
        List of echo files (e1, e2, e3)
    echo_times : list of float
        Echo times in milliseconds
    output_dir : Path
        Output directory for TEDANA results
    mask_file : Path, optional
        Brain mask file
    tedpca : str, int, or float, optional
        PCA component selection method:
        - 'kundu': Kundu decision tree (auto, may be unstable)
        - 'aic', 'kic', 'mdl': Information criteria methods
        - float (0-1): Variance explained (e.g., 0.95 = 95%)
        - int: Specific number of components (e.g., 225)
        Default: 0.95 (95% variance)
    tree : str, optional
        Decision tree for component classification
        Default: 'kundu'

    Returns
    -------
    dict
        Paths to TEDANA outputs:
        - optcom: Optimally combined data
        - denoised: Denoised data
        - metrics: Component metrics
        - report: HTML report
    """
    from tedana import workflows

    logger.info("=" * 70)
    logger.info("TEDANA Multi-Echo Denoising")
    logger.info("=" * 70)
    logger.info(f"Echo files: {len(echo_files)}")
    for i, (efile, te) in enumerate(zip(echo_files, echo_times), 1):
        logger.info(f"  Echo {i}: TE={te}ms, {efile.name}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if TEDANA outputs already exist
    expected_outputs = {
        'optcom': output_dir / 'tedana_desc-optcom_bold.nii.gz',
        'denoised': output_dir / 'tedana_desc-denoised_bold.nii.gz',
        'metrics': output_dir / 'tedana_desc-tedana_metrics.tsv',
        'report': output_dir / 'tedana_tedana_report.html'
    }

    # Create parameter hash to detect if parameters changed
    import hashlib
    import json
    param_dict = {
        'echo_files': [str(f) for f in echo_files],
        'echo_times': echo_times,
        'tedpca': tedpca,
        'tree': tree,
        'mask': str(mask_file) if mask_file else None
    }
    param_hash = hashlib.md5(json.dumps(param_dict, sort_keys=True).encode()).hexdigest()
    hash_file = output_dir / '.tedana_params.md5'

    # Check if outputs exist AND parameters haven't changed
    all_exist = all(p.exists() for p in expected_outputs.values())
    params_unchanged = hash_file.exists() and hash_file.read_text() == param_hash

    if all_exist and params_unchanged:
        logger.info("TEDANA outputs already exist - using cached results")
        logger.info(f"  Cached denoised output: {expected_outputs['denoised']}")
        logger.info(f"  Cached HTML report: {expected_outputs['report']}")
        logger.info("TEDANA completed successfully")
    elif all_exist and not hash_file.exists():
        # Outputs exist but no hash file (first run with new caching code)
        logger.info("TEDANA outputs exist - using cached results (no hash file yet)")
        logger.info(f"  Cached denoised output: {expected_outputs['denoised']}")
        logger.info("TEDANA completed successfully")
        # Save hash for future runs
        hash_file.write_text(param_hash)
    else:
        if all_exist and hash_file.exists() and not params_unchanged:
            logger.info("TEDANA parameters changed - re-running...")
        elif not all_exist:
            logger.info("TEDANA outputs incomplete - running...")
        # Convert echo times to seconds (TEDANA expects seconds)
        tes_sec = [te / 1000.0 for te in echo_times]

        # Run TEDANA workflow
        logger.info("Running TEDANA workflow...")
        logger.info(f"  TE values (sec): {tes_sec}")

        workflows.tedana_workflow(
            data=[str(f) for f in echo_files],
            tes=tes_sec,
            out_dir=str(output_dir),
            mask=str(mask_file) if mask_file else None,
            tedpca=tedpca,  # Configurable: can be 'kundu', 'aic', 'kic', 'mdl', float (0-1), or int (num components)
            tree=tree,  # Configurable: 'kundu' (default) or 'kundu_tedort'
            verbose=True,
            prefix='tedana',
            overwrite=True  # Allow overwriting partial outputs from failed runs
        )

        logger.info("TEDANA completed successfully")

        # Save parameter hash for future cache validation
        hash_file.write_text(param_hash)

    # Return paths to key outputs
    return {
        'optcom': output_dir / 'tedana_desc-optcom_bold.nii.gz',
        'denoised': output_dir / 'tedana_desc-denoised_bold.nii.gz',
        'metrics': output_dir / 'tedana_desc-tedana_metrics.tsv',
        'report': output_dir / 'tedana_report.html'
    }


def create_func_preprocessing_workflow(
    name: str,
    config: Dict[str, Any],
    work_dir: Path,
    use_aroma: bool = False,
    is_multiecho: bool = False
) -> Workflow:
    """
    Create functional preprocessing workflow.

    Parameters
    ----------
    name : str
        Workflow name
    config : dict
        Configuration dictionary with parameters
    work_dir : Path
        Working directory for Nipype
    use_aroma : bool
        DEPRECATED - not used (kept for compatibility)
    is_multiecho : bool
        DEPRECATED - not used (both paths now identical)

    Returns
    -------
    Workflow
        Nipype workflow for functional preprocessing

    Notes
    -----
    This workflow performs temporal filtering and spatial smoothing only.
    Motion correction and registration must be performed BEFORE this workflow.

    Workflow: bandpass filtering → spatial smoothing

    Motion correction is now performed outside this workflow to enable:
    - ANTs-based registration on raw motion-corrected data
    - Proper fMRI → T1w → MNI transform chain
    - Inverse transforms for atlas transformation to functional space
    """
    wf = Workflow(name=name, base_dir=str(work_dir))

    # Extract configuration
    tr = config.get('tr', 1.029)
    highpass = config.get('highpass', 0.001)
    lowpass = config.get('lowpass', 0.08)
    fwhm = config.get('fwhm', 6)
    bet_frac = config.get('bet', {}).get('frac', 0.3)  # Default: 0.3 for functional

    # Temporal filtering (bandpass)
    bandpass = Node(afni.Bandpass(
        highpass=highpass,
        lowpass=lowpass,
        tr=tr,
        outputtype='NIFTI_GZ'
    ), name='bandpass_filter')

    # Spatial smoothing
    # Note: ACompCor would go between bandpass and smooth if implemented
    smooth = Node(fsl.Smooth(
        fwhm=fwhm,
        output_type='NIFTI_GZ'
    ), name='spatial_smooth')

    # Both multi-echo and single-echo are now motion-corrected before this workflow
    # So the workflow is simply: bandpass → smooth for both
    wf.connect([
        (bandpass, smooth, [('out_file', 'in_file')])
    ])

    # Note: ICA-AROMA support removed - motion correction now happens
    # before this workflow with proper registration. TEDANA (multi-echo) or
    # alternative motion correction methods should be used instead.

    return wf


def run_func_preprocessing(
    config: Dict[str, Any],
    subject: str,
    func_file: Union[Path, List[Path]],
    output_dir: Path,
    work_dir: Optional[Path] = None,
    anat_derivatives: Optional[Path] = None,
    session: Optional[str] = None
) -> Dict[str, Path]:
    """
    Run functional preprocessing with TEDANA and ACompCor.

    Parameters
    ----------
    config : dict
        Configuration dictionary with preprocessing parameters:
        - tr: Repetition time in seconds
        - te: List of echo times in milliseconds (for multi-echo)
        - highpass: Highpass filter frequency (Hz)
        - lowpass: Lowpass filter frequency (Hz)
        - fwhm: Smoothing kernel FWHM (mm)
        - n_procs: Number of parallel processes (default: 4)
        - tedana: dict with 'enabled', 'tedpca', 'tree'
        - aroma: dict with 'enabled' (default: False - redundant with TEDANA), 'denoise_type'
        - acompcor: dict with 'enabled', 'num_components'
        - run_qc: Run quality control (default: True)
        - fd_threshold: Framewise displacement threshold in mm (default: 0.5)
    subject : str
        Subject identifier
    func_file : Path or list of Path
        Input functional file(s). If list, treated as multi-echo data
    output_dir : Path
        Study root directory (e.g., /mnt/bytopia/IRC805/)
        Derivatives will be saved to: {output_dir}/derivatives/func_preproc/{subject}/
    work_dir : Path, optional
        Working directory for temporary Nipype files
        Default: {output_dir}/work/{subject}/func_preproc/
    anat_derivatives : Path, optional
        Path to anatomical derivatives directory (e.g., {study_root}/derivatives/{subject}/anat/)
        Used to locate tissue segmentations (CSF, GM, WM) for ACompCor
    session : str, optional
        Session identifier

    Returns
    -------
    dict
        Output file paths:
        - preprocessed: Final preprocessed functional data
        - optcom: TEDANA optimally combined data (if multi-echo)
        - motion_params: Motion parameters
        - motion_rms: RMS motion plots
        - acompcor_components: ACompCor nuisance regressors
        - tedana_report: TEDANA HTML report (if multi-echo)
        - motion_qc: Motion QC metrics (if run_qc=True)
        - tsnr_qc: tSNR metrics (if run_qc=True)
        - qc_report: HTML QC report (if run_qc=True)
        - derivatives_dir: Output directory path
        - work_dir: Working directory path

    Notes
    -----
    Pipeline (multi-echo): TEDANA → MCFLIRT → Bandpass → Smooth → QC
    Pipeline (single-echo): MCFLIRT → (optional: ICA-AROMA) → Bandpass → Smooth → QC

    For multi-echo data:
    1. TEDANA removes thermal noise and identifies BOLD signal using TE-dependence
    2. Use optimally combined data for further preprocessing
    3. ICA-AROMA is redundant and disabled by default (TEDANA is superior)

    For single-echo data:
    1. Skip TEDANA
    2. ICA-AROMA can be enabled for motion artifact removal
    3. Proceed with standard preprocessing
    """
    logger.info("=" * 70)
    logger.info(f"Functional Preprocessing: {subject}")
    logger.info("=" * 70)
    logger.info(f"Input data: {func_file if not isinstance(func_file, list) else f'{len(func_file)} echoes'}")
    logger.info(f"Anatomical derivatives: {anat_derivatives}")
    logger.info("")

    # Setup directory structure
    # output_dir is the derivatives base (e.g., /mnt/bytopia/IRC805/derivatives)
    # Use standardized hierarchy: {outdir}/{subject}/{modality}/
    outdir = Path(output_dir)

    # Create simple, standardized hierarchy
    derivatives_dir = outdir / subject / 'func'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    # Derive study root from output_dir (derivatives directory)
    # output_dir is derivatives, so study_root is one level up
    study_root = outdir.parent

    # Work directory: {study_root}/work/{subject}/
    # Nipype will add workflow name as subdirectory
    if work_dir is None:
        work_dir = study_root / 'work' / subject
    else:
        work_dir = Path(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Study root: {study_root}")
    logger.info(f"Derivatives: {derivatives_dir}")
    logger.info(f"Working dir: {work_dir}")
    logger.info("")

    results = {
        'derivatives_dir': derivatives_dir,
        'work_dir': work_dir
    }

    # Detect multi-echo vs single-echo
    is_multiecho = isinstance(func_file, list) and len(func_file) > 1

    # Extract BET parameters from config
    bet_frac = config.get('bet', {}).get('frac', 0.3)  # Default: 0.3 for functional

    # Step 1: Multi-echo motion correction and TEDANA
    if is_multiecho and config.get('tedana', {}).get('enabled', True):
        echo_files = [Path(f) for f in func_file]
        echo_times = config.get('te', [10.0, 30.0, 50.0])  # milliseconds
        n_echoes = len(echo_files)

        logger.info("=" * 70)
        logger.info("Multi-Echo Motion Correction")
        logger.info("=" * 70)
        logger.info(f"Number of echoes: {n_echoes}")
        logger.info("")

        # Identify middle echo (typically has best contrast)
        middle_echo_idx = n_echoes // 2
        middle_echo = echo_files[middle_echo_idx]
        logger.info(f"Middle echo (reference): Echo {middle_echo_idx + 1} - {middle_echo.name}")
        logger.info("")

        # Step 1a: Run MCFLIRT on middle echo to get transformation matrices
        mcflirt_dir = derivatives_dir / 'mcflirt_echo'
        mcflirt_dir.mkdir(parents=True, exist_ok=True)

        mcflirt_out = mcflirt_dir / f'echo{middle_echo_idx + 1}_mcf.nii.gz'
        # MCFLIRT creates .nii.mat directory when we remove .nii.gz from output name
        # e.g., -out echo2_mcf.nii creates echo2_mcf.nii.mat/ directory
        mcflirt_base = str(mcflirt_out).replace('.nii.gz', '.nii')
        mcflirt_mat_dir = mcflirt_base + '.mat'

        # Check if motion correction already completed with same parameters
        import hashlib
        import json
        mcflirt_params = {
            'input_file': str(middle_echo),
            'output_base': mcflirt_base,
            'options': ['-plots', '-mats', '-rmsrel', '-rmsabs']
        }
        mcflirt_hash = hashlib.md5(json.dumps(mcflirt_params, sort_keys=True).encode()).hexdigest()
        mcflirt_hash_file = mcflirt_dir / '.mcflirt_params.md5'

        outputs_exist = mcflirt_out.exists() and Path(mcflirt_mat_dir).exists()
        params_unchanged = mcflirt_hash_file.exists() and mcflirt_hash_file.read_text() == mcflirt_hash

        if outputs_exist and params_unchanged:
            logger.info("Motion correction already completed - using cached results")
            logger.info(f"  Cached output: {mcflirt_out}")
        elif outputs_exist and not mcflirt_hash_file.exists():
            # Outputs exist but no hash file (first run with new caching code)
            logger.info("Motion correction already completed - using cached results (no hash file yet)")
            logger.info(f"  Cached output: {mcflirt_out}")
            # Save hash for future runs
            mcflirt_hash_file.write_text(mcflirt_hash)
        else:
            if outputs_exist and mcflirt_hash_file.exists() and not params_unchanged:
                logger.info("Motion correction parameters changed - re-running...")
            logger.info("Running motion correction on middle echo...")
            mcflirt_cmd = [
                'mcflirt',
                '-in', str(middle_echo),
                '-out', mcflirt_base,  # Output base name (e.g., echo2_mcf.nii)
                '-plots',
                '-mats',
                '-rmsrel',
                '-rmsabs'
            ]
            subprocess.run(mcflirt_cmd, check=True, capture_output=True)

            # Save parameter hash for future cache validation
            mcflirt_hash_file.write_text(mcflirt_hash)

        # Store motion parameters from middle echo
        # MCFLIRT creates .nii.par when output is echo2_mcf.nii
        motion_params = mcflirt_dir / f'echo{middle_echo_idx + 1}_mcf.nii.par'
        if motion_params.exists():
            results['motion_params'] = motion_params
        logger.info(f"  Motion parameters: {motion_params}")
        logger.info("")

        # Step 1b: Apply transformations to all echoes
        logger.info("Applying motion correction to all echoes...")
        motion_corrected_echoes = []

        for i, echo_file in enumerate(echo_files):
            logger.info(f"  Echo {i + 1}: {echo_file.name}")

            if i == middle_echo_idx:
                # Middle echo is already motion corrected
                motion_corrected_echoes.append(mcflirt_out)
                logger.info(f"    Already corrected (reference)")
            else:
                # Apply transformation matrices from middle echo to other echoes
                output_echo = mcflirt_dir / f'echo{i + 1}_mcf.nii.gz'

                # Check if already processed
                if output_echo.exists():
                    logger.info(f"    Using cached motion-corrected echo")
                    motion_corrected_echoes.append(output_echo)
                else:
                    # Use applyxfm4D to apply the transformation matrices
                    applyxfm_cmd = [
                        'applyxfm4D',
                        str(echo_file),           # Input echo
                        str(middle_echo),         # Reference (middle echo)
                        str(output_echo),         # Output
                        str(mcflirt_mat_dir),     # Transformation matrices directory
                        '-fourdigit'              # MAT file naming convention
                    ]
                    subprocess.run(applyxfm_cmd, check=True, capture_output=True)
                    motion_corrected_echoes.append(output_echo)
                    logger.info(f"    Corrected using middle echo transforms")

        logger.info("")
        logger.info("All echoes motion-corrected and aligned")
        logger.info("")

        # Step 1c: Create brain mask from motion-corrected middle echo
        mask_file = derivatives_dir / 'func_mask.nii.gz'

        if mask_file.exists():
            logger.info("Brain mask already exists - using cached version")
            logger.info(f"  Cached mask: {mask_file}")
        else:
            logger.info("Creating brain mask from motion-corrected middle echo...")
            bet_cmd = [
                'bet',
                str(mcflirt_out),
                str(derivatives_dir / 'func_brain'),
                '-f', str(bet_frac),
                '-m',
                '-R'
            ]
            subprocess.run(bet_cmd, check=True, capture_output=True)

            # Move mask to expected location
            mask_output = derivatives_dir / 'func_brain_mask.nii.gz'
            if mask_output.exists():
                mask_output.rename(mask_file)
            logger.info(f"  Brain mask: {mask_file}")
        logger.info("")

        # Step 1d: Compute functional mean for registration
        logger.info("Computing functional mean for registration...")
        func_mean_file = mcflirt_dir / 'func_mean.nii.gz'
        if not func_mean_file.exists():
            compute_func_mean(mcflirt_out, func_mean_file)
        else:
            logger.info(f"  Using cached func mean: {func_mean_file}")
        logger.info("")

        # Step 1e: Register functional → T1w → MNI (ANTs-based)
        if anat_derivatives:
            registration_dir = derivatives_dir / 'registration'
            registration_dir.mkdir(parents=True, exist_ok=True)

            # Locate anatomical preprocessing outputs
            anat_dir = Path(anat_derivatives)
            brain_files = list(anat_dir.glob('brain/*brain.nii.gz'))
            t1w_to_mni_transform = anat_dir / 'transforms' / 'ants_Composite.h5'
            mni_template = Path(config.get('paths', {}).get('mni152_t1_2mm',
                                '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'))

            if brain_files and t1w_to_mni_transform.exists():
                brain_file = brain_files[0]

                logger.info("Creating functional → MNI transform pipeline (ANTs)...")
                logger.info(f"  Functional mean: {func_mean_file.name}")
                logger.info(f"  T1w brain: {brain_file.name}")
                logger.info(f"  T1w→MNI transform: {t1w_to_mni_transform.name}")
                logger.info("")

                try:
                    registration_results = create_func_to_mni_transforms(
                        func_mean=func_mean_file,
                        t1w_brain=brain_file,
                        t1w_to_mni_transform=t1w_to_mni_transform,
                        mni_template=mni_template,
                        output_dir=registration_dir
                    )

                    # Store registration results
                    results['func_to_t1w_transform'] = registration_results['func_to_t1w']
                    results['func_to_mni_transform'] = registration_results['func_to_mni']
                    results['func_warped_to_t1w'] = registration_results['func_warped_to_t1w']

                    logger.info("Registration pipeline completed successfully")
                    logger.info("")

                    # Generate registration QC visualizations
                    if config.get('run_qc', True):
                        logger.info("Generating registration QC visualizations...")
                        # QC goes in study-wide QC directory
                        study_root = output_dir.parent if output_dir.name == subject else output_dir.parent.parent
                        qc_dir = study_root / 'qc' / subject / 'func' / 'registration'
                        try:
                            qc_results = generate_registration_qc_report(
                                func_mean=func_mean_file,
                                t1w_brain=brain_file,
                                mni_template=mni_template,
                                func_to_t1w_transform=registration_results['func_to_t1w'],
                                func_to_mni_transform=registration_results['func_to_mni'],
                                output_dir=qc_dir
                            )
                            results['registration_qc'] = qc_results
                            logger.info(f"  QC visualizations saved to: {qc_dir}")
                            logger.info("")
                        except Exception as e:
                            logger.warning(f"Registration QC generation failed: {e}")
                            logger.warning("Continuing without QC visualizations")
                            logger.warning("")

                except Exception as e:
                    logger.error(f"Registration failed: {e}")
                    logger.error("Continuing without registration transforms")
                    logger.error("")
            else:
                logger.warning("Anatomical transforms not found - skipping functional registration")
                logger.warning(f"  Searched for: {t1w_to_mni_transform}")
                logger.warning("")
        else:
            logger.warning("No anatomical derivatives provided - skipping functional registration")
            logger.warning("")

        # Step 1f: Run TEDANA on motion-corrected echoes
        tedana_dir = derivatives_dir / 'tedana'
        # Extract TEDANA config (with defaults)
        tedana_config = config.get('tedana', {})
        tedpca = tedana_config.get('tedpca', 0.95)  # Default: 95% variance (auto)
        tree = tedana_config.get('tree', 'kundu')   # Default: kundu decision tree

        tedana_results = run_tedana(
            echo_files=motion_corrected_echoes,
            echo_times=echo_times,
            output_dir=tedana_dir,
            mask_file=mask_file,
            tedpca=tedpca,
            tree=tree
        )

        # Use optimally combined data for further processing
        func_input = tedana_results['optcom']
        results['tedana_optcom'] = tedana_results['optcom']
        results['tedana_denoised'] = tedana_results['denoised']
        results['tedana_report'] = tedana_results['report']
        results['motion_corrected_echoes'] = motion_corrected_echoes

        logger.info(f"TEDANA output: {func_input}")
        logger.info("")
    else:
        # Single echo - run motion correction and registration
        func_input = Path(func_file) if not isinstance(func_file, list) else Path(func_file[0])
        logger.info("Single-echo data - skipping TEDANA")
        logger.info("")

        # Step 1a: Run motion correction on single-echo data
        logger.info("=" * 70)
        logger.info("Motion Correction (MCFLIRT)")
        logger.info("=" * 70)
        logger.info("")

        mcflirt_dir = derivatives_dir / 'motion_correction'
        mcflirt_dir.mkdir(parents=True, exist_ok=True)

        mcflirt_out = mcflirt_dir / f'{subject}_mcf.nii.gz'
        motion_params = mcflirt_dir / f'{subject}_mcf.nii.par'

        if mcflirt_out.exists() and motion_params.exists():
            logger.info("Motion correction already completed - using cached results")
            logger.info(f"  Cached output: {mcflirt_out}")
        else:
            logger.info("Running motion correction...")
            mcflirt_cmd = [
                'mcflirt',
                '-in', str(func_input),
                '-out', str(mcflirt_dir / f'{subject}_mcf.nii'),
                '-plots',
                '-mats',
                '-rmsrel',
                '-rmsabs'
            ]
            subprocess.run(mcflirt_cmd, check=True, capture_output=True)

        results['motion_params'] = motion_params
        logger.info(f"  Motion-corrected: {mcflirt_out}")
        logger.info(f"  Motion parameters: {motion_params}")
        logger.info("")

        # Step 1b: Create brain mask
        mask_file = derivatives_dir / 'func_mask.nii.gz'
        if not mask_file.exists():
            logger.info("Creating brain mask...")
            bet_frac = config.get('bet', {}).get('frac', 0.3)
            bet_cmd = [
                'bet',
                str(mcflirt_out),
                str(derivatives_dir / 'func_brain'),
                '-f', str(bet_frac),
                '-m',
                '-R'
            ]
            subprocess.run(bet_cmd, check=True, capture_output=True)

            mask_output = derivatives_dir / 'func_brain_mask.nii.gz'
            if mask_output.exists():
                mask_output.rename(mask_file)
            logger.info(f"  Brain mask: {mask_file}")
        else:
            logger.info(f"Brain mask already exists: {mask_file}")
        logger.info("")

        # Step 1c: Compute functional mean for registration
        logger.info("Computing functional mean for registration...")
        func_mean_file = mcflirt_dir / 'func_mean.nii.gz'
        if not func_mean_file.exists():
            compute_func_mean(mcflirt_out, func_mean_file)
        else:
            logger.info(f"  Using cached func mean: {func_mean_file}")
        logger.info("")

        # Step 1d: Register functional → T1w → MNI (ANTs-based)
        if anat_derivatives:
            registration_dir = derivatives_dir / 'registration'
            registration_dir.mkdir(parents=True, exist_ok=True)

            # Locate anatomical preprocessing outputs
            anat_dir = Path(anat_derivatives)
            brain_files = list(anat_dir.glob('brain/*brain.nii.gz'))
            t1w_to_mni_transform = anat_dir / 'transforms' / 'ants_Composite.h5'
            mni_template = Path(config.get('paths', {}).get('mni152_t1_2mm',
                                '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'))

            if brain_files and t1w_to_mni_transform.exists():
                brain_file = brain_files[0]

                logger.info("Creating functional → MNI transform pipeline (ANTs)...")
                logger.info(f"  Functional mean: {func_mean_file.name}")
                logger.info(f"  T1w brain: {brain_file.name}")
                logger.info(f"  T1w→MNI transform: {t1w_to_mni_transform.name}")
                logger.info("")

                try:
                    registration_results = create_func_to_mni_transforms(
                        func_mean=func_mean_file,
                        t1w_brain=brain_file,
                        t1w_to_mni_transform=t1w_to_mni_transform,
                        mni_template=mni_template,
                        output_dir=registration_dir
                    )

                    # Store registration results
                    results['func_to_t1w_transform'] = registration_results['func_to_t1w']
                    results['func_to_mni_transform'] = registration_results['func_to_mni']
                    results['func_warped_to_t1w'] = registration_results['func_warped_to_t1w']

                    logger.info("Registration pipeline completed successfully")
                    logger.info("")

                    # Generate registration QC visualizations
                    if config.get('run_qc', True):
                        logger.info("Generating registration QC visualizations...")
                        # QC goes in study-wide QC directory
                        study_root = output_dir.parent if output_dir.name == subject else output_dir.parent.parent
                        qc_dir = study_root / 'qc' / subject / 'func' / 'registration'
                        try:
                            qc_results = generate_registration_qc_report(
                                func_mean=func_mean_file,
                                t1w_brain=brain_file,
                                mni_template=mni_template,
                                func_to_t1w_transform=registration_results['func_to_t1w'],
                                func_to_mni_transform=registration_results['func_to_mni'],
                                output_dir=qc_dir
                            )
                            results['registration_qc'] = qc_results
                            logger.info(f"  QC visualizations saved to: {qc_dir}")
                            logger.info("")
                        except Exception as e:
                            logger.warning(f"Registration QC generation failed: {e}")
                            logger.warning("Continuing without QC visualizations")
                            logger.warning("")

                except Exception as e:
                    logger.error(f"Registration failed: {e}")
                    logger.error("Continuing without registration transforms")
                    logger.error("")
            else:
                logger.warning("Anatomical transforms not found - skipping functional registration")
                logger.warning(f"  Searched for: {t1w_to_mni_transform}")
                logger.warning("")
        else:
            logger.warning("No anatomical derivatives provided - skipping functional registration")
            logger.warning("")

        # Update func_input to use motion-corrected data
        func_input = mcflirt_out

    # Step 2: Locate tissue masks from anatomical preprocessing for ACompCor
    acompcor_enabled = config.get('acompcor', {}).get('enabled', True)
    csf_mask = None
    wm_mask = None
    brain_file = None

    if acompcor_enabled and anat_derivatives:
        logger.info("=" * 70)
        logger.info("Loading tissue masks from anatomical preprocessing")
        logger.info("=" * 70)

        anat_dir = Path(anat_derivatives)
        seg_dir = anat_dir / 'segmentation'

        # Locate tissue probability maps from Atropos
        # POSTERIOR_01.nii.gz = CSF, POSTERIOR_02.nii.gz = GM, POSTERIOR_03.nii.gz = WM
        csf_files = list(seg_dir.glob('*POSTERIOR_01.nii.gz'))
        wm_files = list(seg_dir.glob('*POSTERIOR_03.nii.gz'))
        brain_files = list(anat_dir.glob('brain/*brain.nii.gz'))

        if csf_files and wm_files and brain_files:
            csf_mask = csf_files[0]
            wm_mask = wm_files[0]
            brain_file = brain_files[0]

            logger.info(f"  CSF mask: {csf_mask.name}")
            logger.info(f"  WM mask: {wm_mask.name}")
            logger.info(f"  T1w brain: {brain_file.name}")
            logger.info("")
        else:
            logger.warning("Tissue masks not found in anatomical derivatives")
            logger.warning(f"  Searched in: {seg_dir}")
            logger.warning("  Disabling ACompCor")
            acompcor_enabled = False
            logger.info("")

    # Step 3: Create and run preprocessing workflow (bandpass filtering)
    logger.info("Creating preprocessing workflow...")

    # Determine if AROMA should be used
    # Auto-enable for single-echo (primary denoising method)
    # Disabled for multi-echo (TEDANA handles denoising)
    aroma_config = config.get('aroma', {}).get('enabled', 'auto')

    if aroma_config == 'auto':
        # Auto: Use AROMA for single-echo, skip for multi-echo
        use_aroma = not is_multiecho
        if use_aroma:
            logger.info("ICA-AROMA enabled for single-echo data (auto-detected)")
    elif aroma_config is True:
        # Explicitly enabled
        use_aroma = True
        if is_multiecho:
            logger.warning("ICA-AROMA enabled for multi-echo data - this is redundant with TEDANA")
            logger.warning("Consider using 'auto' setting to skip AROMA for multi-echo")
    else:
        # Explicitly disabled
        use_aroma = False
        if not is_multiecho:
            logger.warning("ICA-AROMA disabled for single-echo data - no motion artifact removal")

    # Get execution configuration and set Nipype config BEFORE creating workflow
    # This ensures hash_method is set when nodes are instantiated
    exec_config = get_execution_config(config)

    wf = create_func_preprocessing_workflow(
        name='func_preproc',
        config=config,
        work_dir=work_dir,
        use_aroma=use_aroma,
        is_multiecho=is_multiecho
    )

    # Set input for the first node in the workflow
    # Both multi-echo and single-echo are now motion-corrected before this point
    # So bandpass is the first node for both
    wf.get_node('bandpass_filter').inputs.in_file = str(func_input)

    # Write workflow graph
    graph_file = derivatives_dir / 'workflow_graph.png'
    wf.write_graph(graph2use='flat', format='png', simple_form=True)
    logger.info(f"Workflow graph: {graph_file}")
    logger.info("")

    # Run workflow
    logger.info("Running bandpass filtering and smoothing...")
    wf_result = wf.run(**exec_config)

    # Get outputs from work directory (result pickle may be in temp location)
    bandpass_dir = work_dir / 'func_preproc' / 'bandpass_filter'
    smooth_dir = work_dir / 'func_preproc' / 'spatial_smooth'

    bandpass_files = list(bandpass_dir.glob('*_bp.nii.gz'))
    smooth_files = list(smooth_dir.glob('*_smooth.nii.gz'))

    if not bandpass_files:
        raise FileNotFoundError(f"Bandpass output not found in {bandpass_dir}")
    if not smooth_files:
        raise FileNotFoundError(f"Smoothed output not found in {smooth_dir}")

    bandpass_output = bandpass_files[0]
    smooth_output = smooth_files[0]

    logger.info(f"Bandpass filtered: {bandpass_output}")
    logger.info(f"Smoothed: {smooth_output}")

    # Step 4: Apply ACompCor (if enabled)
    if acompcor_enabled and csf_mask and wm_mask and brain_file:
        logger.info("=" * 70)
        logger.info("ACompCor Nuisance Regression")
        logger.info("=" * 70)
        logger.info("")

        acompcor_dir = derivatives_dir / 'acompcor'
        acompcor_dir.mkdir(parents=True, exist_ok=True)

        # Step 4a: Transform tissue masks to functional space using ANTs
        logger.info("Step 1: Transforming tissue masks to functional space...")

        # Check if we have the ANTs registration transforms
        if 'func_to_t1w_transform' in results:
            # Use ANTs inverse transform to bring masks to functional space
            func_mean_file = derivatives_dir / 'motion_correction' / 'func_mean.nii.gz'
            if not func_mean_file.exists():
                # Try multi-echo location
                func_mean_file = derivatives_dir / 'mcflirt' / 'func_mean.nii.gz'

            csf_func, wm_func = apply_inverse_transform_to_masks(
                csf_mask=csf_mask,
                wm_mask=wm_mask,
                t1w_to_func_transform=results['func_to_t1w_transform'],  # This gets inverted automatically
                reference_image=func_mean_file,
                output_dir=acompcor_dir
            )
            results['csf_func_mask'] = csf_func
            results['wm_func_mask'] = wm_func
            logger.info("  ANTs-based mask transformation complete")
            logger.info("")

            # Generate tissue mask QC visualizations
            if config.get('run_qc', True) and 'registration_qc' in results:
                logger.info("Generating tissue mask QC visualizations...")
                # QC goes in study-wide QC directory
                study_root = output_dir.parent if output_dir.name == subject else output_dir.parent.parent
                qc_dir = study_root / 'qc' / subject / 'func' / 'registration'
                try:
                    from neurovrai.preprocess.qc.func_registration_qc import qc_tissue_masks_in_func
                    tissue_qc = qc_tissue_masks_in_func(
                        func_mean=func_mean_file,
                        csf_mask_func=csf_func,
                        wm_mask_func=wm_func,
                        output_dir=qc_dir / 'tissue_masks'
                    )
                    results['registration_qc']['tissue_masks'] = tissue_qc
                    logger.info("  Tissue mask QC complete")
                    logger.info("")
                except Exception as e:
                    logger.warning(f"Tissue mask QC generation failed: {e}")
                    logger.warning("")

        else:
            # Fallback to old FSL-based registration (should not happen with new pipeline)
            logger.warning("ANTs transforms not available - falling back to FSL registration")
            logger.warning("This is not recommended and may produce poor alignment!")
            csf_func, wm_func, func_to_anat_bbr = register_masks_to_functional(
                t1w_brain=brain_file,
                func_ref=bandpass_output,
                csf_mask=csf_mask,
                wm_mask=wm_mask,
                output_dir=acompcor_dir
            )
            results['csf_func_mask'] = csf_func
            results['wm_func_mask'] = wm_func

            # Save BBR transform to TransformRegistry for reuse in normalization
            registry = create_transform_registry(config, subject)
            bbr_transform = registry.save_linear_transform(
                transform_file=func_to_anat_bbr,
                source_space='func',
                target_space='T1w',
                source_image=func_input,
                reference=brain_file
            )
            results['func_to_anat_bbr'] = bbr_transform
            logger.info(f"  BBR transform saved to registry: {bbr_transform}")
            logger.info("")

        # Step 4b: Prepare masks (threshold and erode)
        logger.info("Step 2: Preparing ACompCor masks...")
        csf_eroded, wm_eroded = prepare_acompcor_masks(
            csf_mask=csf_func,
            wm_mask=wm_func,
            output_dir=acompcor_dir,
            csf_threshold=config.get('acompcor', {}).get('csf_threshold', 0.9),
            wm_threshold=config.get('acompcor', {}).get('wm_threshold', 0.9)
        )
        results['csf_acompcor_mask'] = csf_eroded
        results['wm_acompcor_mask'] = wm_eroded
        logger.info("")

        # Step 4c: Extract ACompCor components
        logger.info("Step 3: Extracting ACompCor components...")
        acompcor_result = extract_acompcor_components(
            func_file=bandpass_output,
            csf_mask=csf_eroded,
            wm_mask=wm_eroded,
            output_dir=acompcor_dir,
            num_components=config.get('acompcor', {}).get('num_components', 5),
            variance_threshold=config.get('acompcor', {}).get('variance_threshold', 0.5)
        )
        results['acompcor_components'] = acompcor_result['components_file']
        results['acompcor_variance'] = acompcor_result['variance_explained']
        logger.info("")

        # Step 4d: Regress out components
        logger.info("Step 4: Regressing out ACompCor components...")
        acompcor_cleaned = derivatives_dir / f'{subject}_bold_acompcor_cleaned.nii.gz'
        regress_out_components(
            func_file=bandpass_output,
            components_file=acompcor_result['components_file'],
            output_file=acompcor_cleaned
        )
        logger.info("")

        # Use ACompCor-cleaned data for smoothing
        smooth_input = acompcor_cleaned
    else:
        # No ACompCor - use bandpass output directly
        smooth_input = bandpass_output

    # Step 5: Apply spatial smoothing to ACompCor-cleaned (or bandpass-filtered) data
    logger.info("Applying spatial smoothing...")
    fwhm = config.get('fwhm', 6)
    smoothed_file = derivatives_dir / f'{subject}_bold_preprocessed.nii.gz'

    smooth_cmd = [
        'fslmaths',
        str(smooth_input),
        '-s', str(fwhm / 2.355),  # Convert FWHM to sigma
        str(smoothed_file)
    ]
    subprocess.run(smooth_cmd, check=True, capture_output=True)

    results['preprocessed'] = smoothed_file
    logger.info(f"Smoothed ({fwhm}mm FWHM): {smoothed_file}")
    logger.info("")

    # Step 6: Get motion parameters and mask based on workflow type
    if is_multiecho:
        # Motion params already extracted during multi-echo preprocessing
        # Mask already created before TEDANA
        func_mask = derivatives_dir / 'func_mask.nii.gz'
    else:
        # Single-echo: get from workflow nodes
        mcflirt_node = wf.get_node('motion_correction')
        if hasattr(mcflirt_node, 'result') and mcflirt_node.result:
            results['motion_params'] = Path(mcflirt_node.result.outputs.par_file)
            results['motion_rms'] = Path(mcflirt_node.result.outputs.rms_files)

        bet_node = wf.get_node('brain_extraction')
        if hasattr(bet_node, 'result') and bet_node.result:
            func_mask = Path(bet_node.result.outputs.mask_file)
        else:
            # If no BET in workflow, use functional mask from TEDANA
            func_mask = derivatives_dir / 'func_mask.nii.gz'

    # Step 7: Quality Control
    if config.get('run_qc', True):
        logger.info("=" * 70)
        logger.info("Running Quality Control")
        logger.info("=" * 70)
        logger.info("")

        qc_dir = study_root / 'qc' / subject / 'func'
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Motion QC (if motion params available)
        if 'motion_params' in results and results['motion_params'].exists():
            logger.info("Computing motion QC metrics...")
            motion_metrics = compute_motion_qc(
                motion_file=results['motion_params'],
                tr=config.get('tr', 1.029),
                output_dir=qc_dir,
                fd_threshold=config.get('fd_threshold', 0.5)
            )
            results['motion_qc'] = motion_metrics
            logger.info("")
        else:
            logger.warning("Motion parameters not found - skipping motion QC")
            motion_metrics = None

        # tSNR calculation
        logger.info("Computing temporal SNR...")
        tsnr_metrics = compute_tsnr(
            func_file=results['preprocessed'],
            mask_file=func_mask,
            output_dir=qc_dir
        )
        results['tsnr_qc'] = tsnr_metrics
        logger.info("")

        # DVARS calculation (artifact detection)
        logger.info("Computing DVARS (artifact detection)...")
        dvars_metrics = compute_dvars(
            func_file=results['preprocessed'],
            mask_file=func_mask,
            output_dir=qc_dir,
            dvars_threshold=config.get('dvars_threshold', 1.5)
        )
        results['dvars_qc'] = dvars_metrics
        logger.info("")

        # Carpet plot generation (voxel intensity visualization)
        logger.info("Creating carpet plot...")
        carpet_metrics = create_carpet_plot(
            func_file=results['preprocessed'],
            mask_file=func_mask,
            motion_file=results.get('motion_params'),
            output_dir=qc_dir,
            tr=config.get('tr', 1.029)
        )
        results['carpet_qc'] = carpet_metrics
        logger.info("")

        # Generate QC report (only if we have motion metrics)
        if motion_metrics:
            logger.info("Generating QC report...")
            qc_report = generate_func_qc_report(
                subject=subject,
                motion_metrics=motion_metrics,
                tsnr_metrics=tsnr_metrics,
                dvars_metrics=dvars_metrics,
                carpet_metrics=carpet_metrics,
                tedana_report=results.get('tedana_report'),
                output_file=qc_dir / f'{subject}_func_qc_report.html'
            )
            results['qc_report'] = qc_report
            logger.info("")
        else:
            logger.warning("Skipping QC report generation (no motion metrics)")

    # Step 8: Spatial normalization to MNI152 (optional, reuses anatomical transforms)
    if config.get('normalize_to_mni', False):
        logger.info("")
        logger.info("=" * 70)
        logger.info("Step 8: Normalizing functional data to MNI152")
        logger.info("=" * 70)
        logger.info("")

        # Check for required transforms from TransformRegistry
        bbr_transform = results.get('func_to_anat_bbr')

        # Get anatomical transforms from TransformRegistry
        registry = create_transform_registry(config, subject)
        anat_transforms = registry.get_nonlinear_transform('T1w', 'MNI152')

        # Verify all transforms exist
        if bbr_transform and bbr_transform.exists() and anat_transforms:
            t1w_to_mni_warp, t1w_to_mni_affine = anat_transforms

            # Get registration method from registry
            registration_method = registry.get_transform_method('T1w', 'MNI152') or 'fsl'

            logger.info("All required transforms found - proceeding with normalization")
            logger.info(f"  BBR (func→anat): {bbr_transform}")
            logger.info(f"  Affine (anat→MNI): {t1w_to_mni_affine}")
            logger.info(f"  Warp (anat→MNI): {t1w_to_mni_warp}")
            logger.info(f"  Registration method: {registration_method}")
            logger.info("")

            try:
                norm_results = normalize_func_to_mni152(
                    func_file=results['preprocessed'],
                    func_to_anat_bbr=bbr_transform,
                    t1w_to_mni_affine=t1w_to_mni_affine,
                    t1w_to_mni_warp=t1w_to_mni_warp,
                    output_dir=derivatives_dir,
                    mni152_template=None,  # Uses $FSLDIR default
                    interpolation='spline',
                    registration_method=registration_method
                )

                results['func_to_mni_warp'] = norm_results['func_to_mni_warp']
                results['func_normalized'] = norm_results['func_normalized']

                logger.info("Functional normalization complete!")
                logger.info(f"  Concatenated warp: {results['func_to_mni_warp']}")
                logger.info(f"  Normalized functional data: {results['func_normalized']}")
                logger.info("")
            except Exception as e:
                logger.error(f"Functional normalization failed: {e}")
                logger.warning("Continuing without normalization...")
        else:
            logger.warning("Required transforms not found - skipping normalization")
            if not bbr_transform or not bbr_transform.exists():
                logger.warning(f"  Missing BBR transform: {bbr_transform}")
            if not t1w_to_mni_affine.exists():
                logger.warning(f"  Missing anatomical affine: {t1w_to_mni_affine}")
            if not t1w_to_mni_warp.exists():
                logger.warning(f"  Missing anatomical warp: {t1w_to_mni_warp}")
            logger.warning("  Tip: Run anatomical preprocessing first to generate transforms")
            logger.info("")

    logger.info("=" * 70)
    logger.info("Preprocessing Complete")
    logger.info("=" * 70)
    logger.info(f"Preprocessed data: {results['preprocessed']}")
    logger.info(f"Motion parameters: {results['motion_params']}")
    if 'tedana_report' in results:
        logger.info(f"TEDANA report: {results['tedana_report']}")
    if 'qc_report' in results:
        logger.info(f"QC report: {results['qc_report']}")
    logger.info("")

    return results


def get_execution_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get Nipype execution configuration."""
    return {
        'plugin': 'MultiProc',
        'plugin_args': {
            'n_procs': config.get('n_procs', 4)
        }
    }
