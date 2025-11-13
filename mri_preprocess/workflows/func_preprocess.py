#!/usr/bin/env python3
"""
Functional (rs-fMRI) preprocessing workflow with multi-echo support.

Workflow features:
1. Multi-echo denoising with TEDANA (if multi-echo data)
2. Motion correction (MCFLIRT)
3. ICA-AROMA for motion artifact removal
4. Nuisance regression (ACompCor) using tissue masks from anatomical workflow
5. Temporal filtering (bandpass)
6. Spatial smoothing
7. Registration to MNI space using T1w→MNI transforms

Key integrations:
- Uses CSF/WM masks from anatomical FAST segmentation for ACompCor
- Reuses T1w→MNI transformations for efficient registration
- TEDANA enabled by default for multi-echo fMRI
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

from mri_preprocess.utils.workflow import (
    setup_logging,
    get_node_config,
    get_execution_config,
    validate_inputs
)
from mri_preprocess.qc.func_qc import (
    compute_motion_qc,
    compute_tsnr,
    generate_func_qc_report
)
from mri_preprocess.utils.acompcor_helper import (
    run_fast_segmentation,
    register_masks_to_functional,
    prepare_acompcor_masks,
    extract_acompcor_components,
    regress_out_components
)

logger = logging.getLogger(__name__)


def run_tedana(
    echo_files: List[Path],
    echo_times: List[float],
    output_dir: Path,
    mask_file: Optional[Path] = None
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
        tedpca='kundu',
        tree='kundu',
        verbose=True,
        prefix='tedana'
    )

    logger.info("TEDANA completed successfully")

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
    base_dir: Path,
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
    base_dir : Path
        Working directory for Nipype
    use_aroma : bool
        Whether to include ICA-AROMA (default: False, redundant with TEDANA)
    is_multiecho : bool
        Whether input is from multi-echo TEDANA (already motion-corrected)

    Returns
    -------
    Workflow
        Nipype workflow for functional preprocessing

    Notes
    -----
    For multi-echo data:
    - Motion correction is done BEFORE TEDANA (outside this workflow)
    - Input to this workflow is already motion-corrected TEDANA output
    - Only bandpass filtering and smoothing are applied

    For single-echo data:
    - Motion correction is done within this workflow
    - Optional ICA-AROMA can be added for motion artifact removal
    """
    wf = Workflow(name=name, base_dir=str(base_dir))

    # Extract configuration
    tr = config.get('tr', 1.029)
    highpass = config.get('highpass', 0.001)
    lowpass = config.get('lowpass', 0.08)
    fwhm = config.get('fwhm', 6)

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

    if is_multiecho:
        # Multi-echo: Input is already motion-corrected TEDANA output
        # Just do bandpass → smooth
        wf.connect([
            (bandpass, smooth, [('out_file', 'in_file')])
        ])
    else:
        # Single-echo: Need motion correction
        mcflirt = Node(fsl.MCFLIRT(
            cost='leastsquares',
            save_plots=True,
            save_mats=True,
            save_rms=True,
            output_type='NIFTI_GZ'
        ), name='motion_correction')

        bet = Node(fsl.BET(
            frac=0.3,
            functional=True,
            mask=True,
            output_type='NIFTI_GZ'
        ), name='brain_extraction')

        if use_aroma:
            # Single-echo with ICA-AROMA
            aroma = Node(fsl.ICA_AROMA(
                denoise_type='both',
                TR=tr
            ), name='ica_aroma')

            wf.connect([
                (mcflirt, bet, [('out_file', 'in_file')]),
                (bet, aroma, [('out_file', 'in_file'),
                              ('mask_file', 'mask')]),
                (mcflirt, aroma, [('par_file', 'motion_parameters')]),
                (aroma, bandpass, [('aggr_denoised_file', 'in_file')]),
                (bandpass, smooth, [('out_file', 'in_file')])
            ])
        else:
            # Single-echo without AROMA
            wf.connect([
                (mcflirt, bet, [('out_file', 'in_file')]),
                (bet, bandpass, [('out_file', 'in_file')]),
                (bandpass, smooth, [('out_file', 'in_file')])
            ])

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

    # Work directory: {study_root}/work/{subject}/
    # Nipype will add workflow name as subdirectory
    if work_dir is None:
        study_root = outdir.parent
        work_dir = study_root / 'work' / subject
    else:
        work_dir = Path(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Study root: {study_root if work_dir is None else work_dir.parent.parent}")
    logger.info(f"Derivatives: {derivatives_dir}")
    logger.info(f"Working dir: {work_dir}")
    logger.info("")

    results = {
        'derivatives_dir': derivatives_dir,
        'work_dir': work_dir
    }

    # Detect multi-echo vs single-echo
    is_multiecho = isinstance(func_file, list) and len(func_file) > 1

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
        logger.info("Running motion correction on middle echo...")
        mcflirt_dir = derivatives_dir / 'mcflirt_echo'
        mcflirt_dir.mkdir(parents=True, exist_ok=True)

        mcflirt_out = mcflirt_dir / f'echo{middle_echo_idx + 1}_mcf.nii.gz'
        # MCFLIRT creates .nii.mat directory when we remove .nii.gz from output name
        # e.g., -out echo2_mcf.nii creates echo2_mcf.nii.mat/ directory
        mcflirt_base = str(mcflirt_out).replace('.nii.gz', '.nii')
        mcflirt_mat_dir = mcflirt_base + '.mat'

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
        logger.info("Creating brain mask from motion-corrected middle echo...")
        mask_file = derivatives_dir / 'func_mask.nii.gz'

        bet_cmd = [
            'bet',
            str(mcflirt_out),
            str(derivatives_dir / 'func_brain'),
            '-f', '0.3',
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

        # Step 1d: Run TEDANA on motion-corrected echoes
        tedana_dir = derivatives_dir / 'tedana'
        tedana_results = run_tedana(
            echo_files=motion_corrected_echoes,
            echo_times=echo_times,
            output_dir=tedana_dir,
            mask_file=mask_file
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
        # Single echo - use input directly
        func_input = Path(func_file) if not isinstance(func_file, list) else Path(func_file[0])
        logger.info("Single-echo data - skipping TEDANA")
        logger.info("")

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
        brain_files = list(anat_dir.glob('*brain.nii.gz'))

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
    # Default: False for multi-echo (TEDANA handles denoising)
    # Can be explicitly enabled via config
    use_aroma = config.get('aroma', {}).get('enabled', False)
    if is_multiecho and use_aroma:
        logger.warning("ICA-AROMA is redundant with TEDANA for multi-echo data")
        logger.warning("Consider disabling AROMA to reduce processing time")

    wf = create_func_preprocessing_workflow(
        name='func_preproc',
        config=config,
        base_dir=work_dir,
        use_aroma=use_aroma,
        is_multiecho=is_multiecho
    )

    # Set input for bandpass filter (first node in the workflow)
    wf.get_node('bandpass_filter').inputs.in_file = str(func_input)

    # Write workflow graph
    graph_file = derivatives_dir / 'workflow_graph.png'
    wf.write_graph(graph_png=str(graph_file), format='png', simple_form=True)
    logger.info(f"Workflow graph: {graph_file}")
    logger.info("")

    # Run workflow
    logger.info("Running bandpass filtering and smoothing...")
    exec_config = get_execution_config(config)
    wf_result = wf.run(**exec_config)

    # Get bandpass-filtered output
    bandpass_node = wf.get_node('bandpass_filter')
    bandpass_output = Path(bandpass_node.result.outputs.out_file)
    logger.info(f"Bandpass filtered: {bandpass_output}")

    # Step 4: Apply ACompCor (if enabled)
    if acompcor_enabled and csf_mask and wm_mask and brain_file:
        logger.info("=" * 70)
        logger.info("ACompCor Nuisance Regression")
        logger.info("=" * 70)
        logger.info("")

        acompcor_dir = derivatives_dir / 'acompcor'
        acompcor_dir.mkdir(parents=True, exist_ok=True)

        # Step 4a: Register tissue masks to functional space
        logger.info("Step 1: Registering tissue masks to functional space...")
        csf_func, wm_func = register_masks_to_functional(
            t1w_brain=brain_file,
            func_ref=bandpass_output,  # Use bandpass output as reference
            csf_mask=csf_mask,
            wm_mask=wm_mask,
            output_dir=acompcor_dir
        )
        results['csf_func_mask'] = csf_func
        results['wm_func_mask'] = wm_func
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

        qc_dir = derivatives_dir / 'qc'
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

        # Generate QC report (only if we have motion metrics)
        if motion_metrics:
            logger.info("Generating QC report...")
            qc_report = generate_func_qc_report(
                subject=subject,
                motion_metrics=motion_metrics,
                tsnr_metrics=tsnr_metrics,
                tedana_report=results.get('tedana_report'),
                output_file=qc_dir / f'{subject}_func_qc_report.html'
            )
            results['qc_report'] = qc_report
            logger.info("")
        else:
            logger.warning("Skipping QC report generation (no motion metrics)")

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
