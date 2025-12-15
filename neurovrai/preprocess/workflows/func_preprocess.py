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
from nipype.interfaces.base import File, Directory, traits, TraitedSpec
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
    compute_skull_strip_qc,
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


# Custom Nipype interface for FSL applyxfm4D
class ApplyXFM4DInputSpec(fsl.base.FSLCommandInputSpec):
    """Input specification for applyxfm4D."""
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0,
                   desc='4D input file')
    ref_vol = File(exists=True, mandatory=True, argstr='%s', position=1,
                   desc='Reference volume')
    out_file = File(argstr='%s', position=2, name_source=['in_file'],
                    name_template='%s_warp', desc='Registered output file')
    trans_dir = Directory(exists=True, mandatory=True, argstr='%s', position=3,
                          desc='Directory containing transformation matrices')
    single_matrix = traits.Str(argstr='%s', position=4,
                                desc='Single matrix filename')
    four_digit = traits.Bool(argstr='-fourdigit', desc='Use 4-digit MAT file naming')


class ApplyXFM4DOutputSpec(TraitedSpec):
    """Output specification for applyxfm4D."""
    out_file = File(exists=True, desc='Registered output file')


class ApplyXFM4D(fsl.base.FSLCommand):
    """
    Use FSL applyxfm4D to apply 4D transformation matrices.

    Applies transformation matrices from mcflirt to register 4D data.

    Examples
    --------
    >>> from nipype.interfaces import fsl
    >>> applyxfm = ApplyXFM4D()
    >>> applyxfm.inputs.in_file = 'functional.nii.gz'
    >>> applyxfm.inputs.ref_vol = 'reference.nii.gz'
    >>> applyxfm.inputs.trans_dir = 'transforms.mat'
    >>> applyxfm.inputs.four_digit = True
    >>> applyxfm.cmdline
    'applyxfm4D functional.nii.gz reference.nii.gz functional_warp.nii.gz transforms.mat -fourdigit'
    """
    _cmd = 'applyxfm4D'
    input_spec = ApplyXFM4DInputSpec
    output_spec = ApplyXFM4DOutputSpec


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


def create_mcflirt_node(config: Dict[str, Any], name: str = 'motion_correction') -> Node:
    """
    Create MCFLIRT motion correction node.

    Parameters
    ----------
    config : dict
        Configuration dictionary with MCFLIRT parameters
    name : str, optional
        Node name (default: 'motion_correction')

    Returns
    -------
    Node
        Configured MCFLIRT node
    """
    mcflirt = Node(
        fsl.MCFLIRT(
            cost='normcorr',
            save_plots=True,
            save_mats=True,
            save_rms=True,
            output_type='NIFTI_GZ'
        ),
        name=name
    )
    return mcflirt


def create_bet_node(config: Dict[str, Any], name: str = 'brain_extraction') -> Node:
    """
    Create BET brain extraction node for functional data.

    Parameters
    ----------
    config : dict
        Configuration dictionary with BET parameters
    name : str, optional
        Node name (default: 'brain_extraction')

    Returns
    -------
    Node
        Configured BET node
    """
    bet_config = config.get('bet', {})

    bet = Node(
        fsl.BET(
            frac=bet_config.get('frac', 0.3),  # Default 0.3 for functional
            robust=True,
            mask=True,
            output_type='NIFTI_GZ'
        ),
        name=name
    )
    return bet


def create_bandpass_node(config: Dict[str, Any], name: str = 'bandpass_filter') -> Node:
    """
    Create AFNI bandpass filtering node.

    Parameters
    ----------
    config : dict
        Configuration dictionary with filtering parameters:
        - highpass: High-pass filter frequency (Hz)
        - lowpass: Low-pass filter frequency (Hz)
        - tr: Repetition time (seconds)
    name : str, optional
        Node name (default: 'bandpass_filter')

    Returns
    -------
    Node
        Configured Bandpass node
    """
    bandpass = Node(
        afni.Bandpass(
            highpass=config.get('highpass', 0.001),
            lowpass=config.get('lowpass', 0.08),
            tr=config.get('tr', 1.029),
            outputtype='NIFTI_GZ'
        ),
        name=name
    )
    return bandpass


def create_smooth_node(config: Dict[str, Any], name: str = 'spatial_smooth') -> Node:
    """
    Create FSL smoothing node.

    Parameters
    ----------
    config : dict
        Configuration dictionary with smoothing parameters:
        - fwhm: Full-width half-maximum kernel size (mm)
    name : str, optional
        Node name (default: 'spatial_smooth')

    Returns
    -------
    Node
        Configured Smooth node
    """
    smooth = Node(
        fsl.Smooth(
            fwhm=config.get('fwhm', 6),
            output_type='NIFTI_GZ'
        ),
        name=name
    )
    return smooth


def create_bet_4d_workflow(config: Dict[str, Any], name: str = 'bet_4d') -> Workflow:
    """
    Create workflow for brain extraction on 4D functional data.

    This workflow:
    1. Computes temporal mean (4D → 3D)
    2. Runs BET on the 3D mean
    3. Applies mask to the full 4D data

    Parameters
    ----------
    config : dict
        Configuration dictionary with BET parameters
    name : str, optional
        Workflow name

    Returns
    -------
    Workflow
        BET workflow for 4D data with inputs: in_file
        and outputs: masked_file, mask_file
    """
    from nipype import Workflow, Node
    from nipype.interfaces import fsl, utility as niu

    workflow = Workflow(name=name)

    # Input node
    inputnode = Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')

    # Output node
    outputnode = Node(niu.IdentityInterface(fields=['masked_file', 'mask_file']), name='outputnode')

    # Compute temporal mean (4D → 3D)
    mean = Node(
        fsl.MeanImage(dimension='T', output_type='NIFTI_GZ'),
        name='temporal_mean'
    )

    # Brain extraction on 3D mean
    bet_frac = config.get('functional', {}).get('bet', {}).get('frac', 0.3)
    bet = Node(
        fsl.BET(
            frac=bet_frac,
            mask=True,
            robust=True,
            output_type='NIFTI_GZ'
        ),
        name='bet_3d'
    )

    # Apply mask to 4D data
    apply_mask = Node(
        fsl.ApplyMask(output_type='NIFTI_GZ'),
        name='apply_mask'
    )

    # Connect nodes
    workflow.connect([
        (inputnode, mean, [('in_file', 'in_file')]),
        (mean, bet, [('out_file', 'in_file')]),
        (inputnode, apply_mask, [('in_file', 'in_file')]),
        (bet, apply_mask, [('mask_file', 'mask_file')]),
        (apply_mask, outputnode, [('out_file', 'masked_file')]),
        (bet, outputnode, [('mask_file', 'mask_file')])
    ])

    return workflow


def create_aroma_node(config: Dict[str, Any], name: str = 'ica_aroma') -> Node:
    """
    Create ICA-AROMA node for motion artifact removal.

    Parameters
    ----------
    config : dict
        Configuration dictionary with AROMA parameters:
        - aroma.denoise_type: 'nonaggr' (non-aggressive, default) or 'aggr' (aggressive)
    name : str, optional
        Node name (default: 'ica_aroma')

    Returns
    -------
    Node
        Configured ICA-AROMA node

    Notes
    -----
    ICA-AROMA extracts TR from the NIfTI header automatically.
    """
    from nipype.interfaces.fsl import ICA_AROMA

    denoise_type = config.get('aroma', {}).get('denoise_type', 'nonaggr')

    aroma = Node(
        ICA_AROMA(
            denoise_type=denoise_type,
            out_dir='aroma_output'
        ),
        name=name
    )
    return aroma


def create_multiecho_motion_correction_workflow(
    name: str,
    config: Dict[str, Any],
    work_dir: Path,
    output_dir: Path
) -> Workflow:
    """
    Phase 1: Multi-echo motion correction and brain extraction workflow.

    Performs motion correction on multi-echo fMRI data:
    1. Run MCFLIRT on echo 2 (middle echo as reference)
    2. Apply transforms to echo 1 and echo 3 in parallel using applyxfm4D
    3. Brain extraction (BET) on motion-corrected echo 2 to create mask for TEDANA

    Parameters
    ----------
    name : str
        Workflow name
    config : dict
        Configuration dictionary with BET parameters
    work_dir : Path
        Working directory for Nipype intermediate outputs
    output_dir : Path
        Output directory for final derivatives (via DataSink)

    Returns
    -------
    Workflow
        Nipype workflow for multi-echo motion correction
    """
    wf = Workflow(name=name)
    wf.base_dir = str(work_dir)

    logger.info(f"Creating {name} workflow (Phase 1: Multi-echo motion correction)")
    logger.info(f"  Work directory: {work_dir}")
    logger.info(f"  Output directory: {output_dir}")

    # INPUT NODE - echo files
    inputnode = Node(
        niu.IdentityInterface(fields=[
            'echo1',
            'echo2',
            'echo3'
        ]),
        name='inputnode'
    )

    # MCFLIRT on echo 2 (reference/middle echo)
    mcflirt_echo2 = Node(
        fsl.MCFLIRT(
            cost='normcorr',
            save_plots=True,
            save_mats=True,
            save_rms=True,
            output_type='NIFTI_GZ'
        ),
        name='mcflirt_echo2'
    )

    # Function to extract .mat directory from MCFLIRT mat_file output
    # mat_file is a list like ['/path/file.mat/MAT_0000', '/path/file.mat/MAT_0001', ...]
    # We need to extract the directory: '/path/file.mat'
    def get_mat_directory(mat_file_list):
        """Extract .mat directory path from MCFLIRT output."""
        from pathlib import Path
        if isinstance(mat_file_list, list) and len(mat_file_list) > 0:
            # Get parent directory of first MAT file
            return str(Path(mat_file_list[0]).parent)
        elif isinstance(mat_file_list, str):
            return str(Path(mat_file_list).parent)
        else:
            raise ValueError(f"Unexpected mat_file format: {mat_file_list}")

    get_mat_dir = Node(
        niu.Function(
            input_names=['mat_file_list'],
            output_names=['mat_dir'],
            function=get_mat_directory
        ),
        name='get_mat_dir'
    )

    # Apply transforms to echo 1
    applyxfm_echo1 = Node(
        ApplyXFM4D(
            four_digit=True
        ),
        name='applyxfm_echo1'
    )

    # Apply transforms to echo 3
    applyxfm_echo3 = Node(
        ApplyXFM4D(
            four_digit=True
        ),
        name='applyxfm_echo3'
    )

    # Compute mean of motion-corrected echo 2 (4D → 3D)
    mean_func = Node(
        fsl.MeanImage(
            dimension='T',
            output_type='NIFTI_GZ'
        ),
        name='mean_func'
    )

    # Brain extraction on mean functional image
    bet_config = config.get('functional', {}).get('bet', {})
    bet = Node(
        fsl.BET(
            frac=bet_config.get('frac', 0.3),  # Default 0.3 for functional
            robust=True,
            mask=True,
            output_type='NIFTI_GZ'
        ),
        name='brain_extraction'
    )

    # OUTPUT NODE - motion corrected echoes + brain mask
    outputnode = Node(
        niu.IdentityInterface(fields=[
            'echo1_corrected',
            'echo2_corrected',
            'echo3_corrected',
            'motion_params',
            'motion_plots',
            'brain_mask'
        ]),
        name='outputnode'
    )

    # DATASINK - save motion correction outputs
    datasink = Node(DataSink(), name='datasink')
    datasink.inputs.base_directory = str(output_dir)
    datasink.inputs.container = ''

    # WORKFLOW CONNECTIONS
    wf.connect([
        # Run MCFLIRT on echo 2
        (inputnode, mcflirt_echo2, [('echo2', 'in_file')]),

        # Extract .mat directory from MCFLIRT output
        (mcflirt_echo2, get_mat_dir, [('mat_file', 'mat_file_list')]),

        # Compute mean and run brain extraction
        (mcflirt_echo2, mean_func, [('out_file', 'in_file')]),
        (mean_func, bet, [('out_file', 'in_file')]),

        # Apply echo 2 transforms to echo 1
        (inputnode, applyxfm_echo1, [('echo1', 'in_file')]),
        (mcflirt_echo2, applyxfm_echo1, [('out_file', 'ref_vol')]),
        (get_mat_dir, applyxfm_echo1, [('mat_dir', 'trans_dir')]),

        # Apply echo 2 transforms to echo 3
        (inputnode, applyxfm_echo3, [('echo3', 'in_file')]),
        (mcflirt_echo2, applyxfm_echo3, [('out_file', 'ref_vol')]),
        (get_mat_dir, applyxfm_echo3, [('mat_dir', 'trans_dir')]),

        # Collect outputs
        (applyxfm_echo1, outputnode, [('out_file', 'echo1_corrected')]),
        (mcflirt_echo2, outputnode, [
            ('out_file', 'echo2_corrected'),
            ('par_file', 'motion_params'),
            ('rms_files', 'motion_plots')
        ]),
        (applyxfm_echo3, outputnode, [('out_file', 'echo3_corrected')]),
        (bet, outputnode, [('mask_file', 'brain_mask')]),

        # Save via DataSink
        (outputnode, datasink, [
            ('echo1_corrected', 'motion_correction.@echo1'),
            ('echo2_corrected', 'motion_correction.@echo2'),
            ('echo3_corrected', 'motion_correction.@echo3'),
            ('motion_params', 'motion_correction.@params'),
            ('motion_plots', 'motion_correction.@plots'),
            ('brain_mask', 'brain.@mask')
        ])
    ])

    return wf


def create_func_preprocessing_workflow(
    name: str,
    config: Dict[str, Any],
    work_dir: Path,
    output_dir: Path,
    is_multiecho: bool = False
) -> Workflow:
    """
    Phase 2: Functional preprocessing workflow (motion correction and denoising).

    For multi-echo: Processes TEDANA output (motion correction done in Phase 1)
    For single-echo: Includes motion correction and ICA-AROMA

    Parameters
    ----------
    name : str
        Workflow name
    config : dict
        Configuration dictionary with preprocessing parameters
    work_dir : Path
        Working directory for Nipype intermediate outputs
    output_dir : Path
        Output directory for final derivatives (via DataSink)
    is_multiecho : bool, optional
        If True, skips motion correction (input is TEDANA output from Phase 1)
        If False, runs motion correction on single-echo data
        Default: False

    Returns
    -------
    Workflow
        Nipype workflow for functional preprocessing

    Notes
    -----
    CRITICAL: Workflow outputs motion-corrected/denoised data.
    ACompCor, bandpass filtering, and smoothing are applied AFTER this workflow.

    Correct pipeline order (HCP-compliant):
    1. Motion correction (MCFLIRT)
    2. Brain extraction (BET)
    3. ICA-AROMA (motion artifact removal, single-echo only)
    4. ACompCor (nuisance regression) - OUTSIDE workflow
    5. Bandpass filtering - OUTSIDE workflow
    6. Spatial smoothing - OUTSIDE workflow

    All intermediate outputs stored in work_dir.
    Final outputs saved to output_dir via DataSink.
    """
    wf = Workflow(name=name)
    wf.base_dir = str(work_dir)

    logger.info(f"Creating {name} workflow")
    logger.info(f"  Work directory: {work_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Multi-echo mode: {is_multiecho}")

    # Extract functional config subsection (consistent with main function)
    func_config = config.get('functional', {})

    # INPUT NODE - Define all workflow inputs
    inputnode = Node(
        niu.IdentityInterface(fields=[
            'func_file',        # Input functional data
            'csf_mask',         # CSF mask from anatomical (for ACompCor)
            'wm_mask',          # WM mask from anatomical (for ACompCor)
            't1w_brain'         # T1w brain for registration (optional)
        ]),
        name='inputnode'
    )

    # PROCESSING NODES
    # Only create MCFLIRT and BET for single-echo
    # Multi-echo: Phase 1 already did motion correction and brain extraction
    if not is_multiecho:
        mcflirt = create_mcflirt_node(func_config, name='motion_correction')
        bet_4d = create_bet_4d_workflow(func_config, name='brain_extraction_4d')

        # ICA-AROMA for single-echo (auto-enabled unless explicitly disabled)
        aroma_config = func_config.get('aroma', {}).get('enabled', 'auto')
        use_aroma = (aroma_config == 'auto') or (aroma_config is True)
        if use_aroma:
            aroma = create_aroma_node(func_config, name='ica_aroma')
        else:
            aroma = None
    else:
        aroma = None

    # NOTE: Bandpass and smoothing removed from workflow
    # They will be applied AFTER ACompCor (outside this workflow)

    # OUTPUT NODE - Aggregate all outputs
    outputnode = Node(
        niu.IdentityInterface(fields=[
            'motion_corrected',
            'motion_params',
            'brain_mask',
            'denoised'  # Motion-corrected + AROMA-denoised (if applicable)
        ]),
        name='outputnode'
    )

    # DATASINK - Save results to derivatives
    datasink = Node(DataSink(), name='datasink')
    datasink.inputs.base_directory = str(output_dir)
    datasink.inputs.container = ''  # Empty - output_dir is already final location

    # WORKFLOW CONNECTIONS
    connections = []

    if not is_multiecho:
        # Single-echo: Run motion correction and brain extraction
        logger.info("  Including motion correction and brain extraction for single-echo data")
        connections.extend([
            # Motion correction
            (inputnode, mcflirt, [('func_file', 'in_file')]),
            # Brain extraction on 4D data (mean → BET → apply mask)
            (mcflirt, bet_4d, [('out_file', 'inputnode.in_file')]),
            (mcflirt, outputnode, [
                ('out_file', 'motion_corrected'),
                ('par_file', 'motion_params')
            ]),
            # Brain mask output from BET 4D workflow
            (bet_4d, outputnode, [('outputnode.mask_file', 'brain_mask')]),
        ])

        # Add ICA-AROMA if enabled (motion artifact removal)
        if aroma is not None:
            logger.info("  Including ICA-AROMA for motion artifact removal")
            denoise_type = func_config.get('aroma', {}).get('denoise_type', 'nonaggr')
            aroma_output_field = 'nonaggr_denoised_file' if denoise_type == 'nonaggr' else 'aggr_denoised_file'

            connections.extend([
                # AROMA inputs: motion-corrected data, motion params, mask
                (mcflirt, aroma, [
                    ('out_file', 'in_file'),
                    ('par_file', 'motion_parameters')
                ]),
                (bet_4d, aroma, [('outputnode.mask_file', 'mask')]),
                # AROMA output → outputnode (for ACompCor)
                (aroma, outputnode, [(aroma_output_field, 'denoised')]),
            ])
        else:
            logger.info("  ICA-AROMA disabled - using motion-corrected data directly")
            connections.extend([
                # Motion-corrected output → outputnode (for ACompCor)
                (bet_4d, outputnode, [('outputnode.masked_file', 'denoised')]),
            ])
    else:
        # Multi-echo: Input is TEDANA output, brain extraction done in Phase 1
        logger.info("  Skipping motion correction and brain extraction (multi-echo - done in Phase 1)")
        connections.extend([
            # TEDANA output → outputnode (for ACompCor)
            (inputnode, outputnode, [('func_file', 'denoised')]),
        ])

    # Save outputs via DataSink
    connections.extend([
        (outputnode, datasink, [
            ('denoised', 'denoised.@data')  # Motion-corrected + AROMA-denoised
        ])
    ])

    # Add single-echo specific outputs (motion correction and brain mask)
    if not is_multiecho:
        connections.extend([
            (outputnode, datasink, [
                ('motion_corrected', 'motion_correction.@corrected'),
                ('motion_params', 'motion_correction.@params'),
                ('brain_mask', 'brain.@mask')  # Brain mask from Phase 2 BET
            ])
        ])

    wf.connect(connections)

    logger.info(f"Workflow created with {len(wf._graph.nodes())} nodes")

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
        Full configuration dictionary (same as DWI preprocessing).
        The function will access the 'functional' subsection internally.

        Required config structure:
        - functional: dict with preprocessing parameters
          - tr: Repetition time in seconds
          - te: List of echo times in milliseconds (for multi-echo)
          - highpass: Highpass filter frequency (Hz)
          - lowpass: Lowpass filter frequency (Hz)
          - fwhm: Smoothing kernel FWHM (mm)
          - tedana: dict with 'enabled', 'tedpca', 'tree'
          - aroma: dict with 'enabled' (default: auto), 'denoise_type'
          - acompcor: dict with 'enabled', 'num_components'
          - run_qc: Run quality control (default: True)
          - fd_threshold: Framewise displacement threshold in mm (default: 0.5)
        - templates: dict with MNI template paths
        - paths: dict with 'transforms' directory (for TransformRegistry)
        - execution: dict with 'n_procs', 'plugin' (optional)
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
    CORRECTED HCP-COMPLIANT PIPELINE ORDER:

    Multi-echo:
    1. Motion correction (MCFLIRT on middle echo)
    2. Brain extraction (BET)
    3. TEDANA denoising (optimal combination + ICA-based denoising)
    4. ACompCor (nuisance regression using CSF/WM masks)
    5. Bandpass filtering
    6. Spatial smoothing
    7. QC

    Single-echo:
    1. Motion correction (MCFLIRT)
    2. Brain extraction (BET)
    3. ICA-AROMA (motion artifact removal)
    4. ACompCor (nuisance regression using CSF/WM masks)
    5. Bandpass filtering
    6. Spatial smoothing
    7. QC

    KEY INSIGHT: Both pipelines converge after step 3 (TEDANA or AROMA).
    Steps 4-7 are IDENTICAL for both multi-echo and single-echo.

    CRITICAL: ACompCor is applied BEFORE bandpass filtering and smoothing.
    This is the correct order per HCP best practices and neuroimaging literature.
    Linear filtering operations (bandpass, smooth) are non-commutative with nuisance
    regression, and the order matters for data quality.
    """
    logger.info("=" * 70)
    logger.info(f"Functional Preprocessing: {subject}")
    logger.info("=" * 70)
    logger.info(f"Input data: {func_file if not isinstance(func_file, list) else f'{len(func_file)} echoes'}")
    logger.info(f"Anatomical derivatives: {anat_derivatives}")
    logger.info("")

    # Extract functional config subsection (consistent with DWI preprocessing pattern)
    func_config = config.get('functional', {})
    logger.info(f"Config structure: {'functional' in config and 'full config' or 'functional subsection only (legacy)'}")

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

    # Extract BET parameters from functional config
    bet_frac = func_config.get('bet', {}).get('frac', 0.3)  # Default: 0.3 for functional

    # Step 1: Multi-echo motion correction and TEDANA
    if is_multiecho and func_config.get('tedana', {}).get('enabled', True):
        echo_files = [Path(f) for f in func_file]
        echo_times = func_config.get('te', [10.0, 30.0, 50.0])  # milliseconds
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

        # Step 1a: Run Phase 1 Nipype Workflow - Multi-echo motion correction
        logger.info("=" * 70)
        logger.info("PHASE 1: Multi-echo Motion Correction (Nipype Workflow)")
        logger.info("=" * 70)
        logger.info("")

        # Get execution configuration
        exec_config = get_execution_config(config)

        # Create Phase 1 workflow
        wf_phase1 = create_multiecho_motion_correction_workflow(
            name='func_phase1_motion',
            config=config,
            work_dir=work_dir.parent,  # Nipype adds workflow name subdirectory
            output_dir=derivatives_dir
        )

        # Set workflow inputs
        wf_phase1.get_node('inputnode').inputs.echo1 = str(echo_files[0])
        wf_phase1.get_node('inputnode').inputs.echo2 = str(echo_files[1])  # Middle echo (reference)
        wf_phase1.get_node('inputnode').inputs.echo3 = str(echo_files[2])

        # Run Phase 1 workflow
        logger.info("Running Phase 1 workflow...")
        wf_phase1_result = wf_phase1.run(**exec_config)

        logger.info("")
        logger.info("Phase 1 Complete - Motion correction outputs in work directory")
        logger.info("")

        # Find motion-corrected echoes in work directory
        phase1_work_dir = work_dir.parent / 'func_phase1_motion'
        mcflirt_echo2_dir = phase1_work_dir / 'mcflirt_echo2'
        applyxfm_echo1_dir = phase1_work_dir / 'applyxfm_echo1'
        applyxfm_echo3_dir = phase1_work_dir / 'applyxfm_echo3'
        bet_dir = phase1_work_dir / 'brain_extraction'

        # Find the output files
        echo1_corrected = list(applyxfm_echo1_dir.glob('*_warp.nii.gz'))[0]
        echo2_corrected = list(mcflirt_echo2_dir.glob('*_mcf.nii.gz'))[0]
        echo3_corrected = list(applyxfm_echo3_dir.glob('*_warp.nii.gz'))[0]

        motion_corrected_echoes = [echo1_corrected, echo2_corrected, echo3_corrected]

        # Find motion parameters and brain mask
        motion_params = list(mcflirt_echo2_dir.glob('*_mcf.nii.gz.par'))[0]
        brain_mask = list(bet_dir.glob('*_mask.nii.gz'))[0]

        results['motion_params'] = motion_params

        logger.info(f"Motion-corrected echoes:")
        for i, echo in enumerate(motion_corrected_echoes):
            logger.info(f"  Echo {i + 1}: {echo}")
        logger.info(f"Motion parameters: {motion_params}")
        logger.info(f"Brain mask (from Phase 1): {brain_mask}")
        logger.info("")

        # Use middle echo for registration reference
        mcflirt_out = echo2_corrected
        mask_file = brain_mask  # Use mask from Phase 1

        # Step 1c: Compute functional mean for registration
        logger.info("Computing functional mean for registration...")
        func_mean_file = mcflirt_echo2_dir / 'func_mean.nii.gz'
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
            mni_template = Path(config.get('templates', {}).get('mni152_t1_2mm',
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
                    if func_config.get('run_qc', True):
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
        # NOTE: TEDANA outputs go to work_dir (temporary), not derivatives
        tedana_dir = work_dir / 'tedana'
        # Extract TEDANA config (with defaults)
        # Check both functional.tedana and top-level tedana for backwards compatibility
        # func_config already extracted at top of function
        tedana_config = func_config.get('tedana', {})
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
        # Single echo - compute func_mean and register before workflow
        func_input = Path(func_file) if not isinstance(func_file, list) else Path(func_file[0])
        logger.info("Single-echo data - skipping TEDANA")
        logger.info("")

        # Step 1: Compute functional mean for registration (before any processing)
        logger.info("=" * 70)
        logger.info("Single-Echo Preprocessing")
        logger.info("=" * 70)

        # Step 2: Register functional → T1w → MNI (ANTs-based)
        if anat_derivatives:
            registration_dir = derivatives_dir / 'registration'
            registration_dir.mkdir(parents=True, exist_ok=True)

            # Compute mean from raw functional data (before motion correction)
            # Save to registration directory for consistency with multi-echo
            logger.info("Computing functional mean for registration...")
            func_mean_file = registration_dir / 'func_mean.nii.gz'
            if not func_mean_file.exists():
                compute_func_mean(func_input, func_mean_file)
            else:
                logger.info(f"  Using cached func mean: {func_mean_file}")
            logger.info("")

            # Locate anatomical preprocessing outputs
            anat_dir = Path(anat_derivatives)
            brain_files = list(anat_dir.glob('brain/*brain.nii.gz'))
            t1w_to_mni_transform = anat_dir / 'transforms' / 'ants_Composite.h5'
            mni_template = Path(config.get('templates', {}).get('mni152_t1_2mm',
                                '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'))

            if brain_files and t1w_to_mni_transform.exists():
                brain_file = brain_files[0]

                logger.info("Creating functional → MNI transform pipeline (ANTs)...")
                logger.info(f"  Functional mean: {func_mean_file.name}")
                logger.info(f"  T1w brain: {brain_file.name}")
                logger.info(f"  T1w→MNI transform: {t1w_to_mni_transform.name}")
                logger.info("  Note: ANTs registration typically takes 2-5 minutes...")
                logger.info("")

                try:
                    registration_results = create_func_to_mni_transforms(
                        func_mean=func_mean_file,
                        t1w_brain=brain_file,
                        t1w_to_mni_transform=t1w_to_mni_transform,
                        mni_template=mni_template,
                        output_dir=registration_dir
                    )

                    # Store registration results (needed for ACompCor mask transformation)
                    results['func_to_t1w_transform'] = registration_results['func_to_t1w']
                    results['func_to_mni_transform'] = registration_results['func_to_mni']
                    results['func_warped_to_t1w'] = registration_results['func_warped_to_t1w']

                    logger.info("Registration pipeline completed successfully")
                    logger.info("")

                    # Generate registration QC visualizations
                    if func_config.get('run_qc', True):
                        logger.info("Generating registration QC visualizations...")
                        # QC goes in study-wide QC directory
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

        logger.info("Motion correction, brain extraction, and AROMA will be handled by Nipype workflow")
        logger.info("")

    # Step 2: Locate tissue masks from anatomical preprocessing for ACompCor
    acompcor_enabled = func_config.get('acompcor', {}).get('enabled', True)
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

    # Step 3: Create and run Phase 2 preprocessing workflow (filtering and smoothing)
    if is_multiecho:
        logger.info("=" * 70)
        logger.info("PHASE 2: Filtering and Smoothing (Nipype Workflow)")
        logger.info("=" * 70)
        logger.info("")
    else:
        logger.info("Creating preprocessing workflow...")

    # Determine if AROMA should be used
    # Auto-enable for single-echo (primary denoising method)
    # Disabled for multi-echo (TEDANA handles denoising)
    aroma_config = func_config.get('aroma', {}).get('enabled', 'auto')

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

    # Create preprocessing workflow with proper Nipype architecture
    wf = create_func_preprocessing_workflow(
        name='func_preproc',
        config=config,
        work_dir=work_dir.parent,  # Nipype adds workflow name subdirectory
        output_dir=derivatives_dir,
        is_multiecho=is_multiecho
    )

    # Set workflow inputs via inputnode
    wf.get_node('inputnode').inputs.func_file = str(func_input)
    if csf_mask and wm_mask:
        wf.get_node('inputnode').inputs.csf_mask = str(csf_mask)
        wf.get_node('inputnode').inputs.wm_mask = str(wm_mask)

    # Write workflow graph
    graph_file = derivatives_dir / 'workflow_graph.png'
    wf.write_graph(graph2use='flat', format='png', simple_form=True)
    logger.info(f"Workflow graph: {graph_file}")
    logger.info("")

    # Run workflow
    if is_multiecho:
        logger.info("Running Phase 2 workflow...")
    else:
        logger.info("Running preprocessing workflow (motion correction, brain extraction, filtering, smoothing)...")
    wf_result = wf.run(**exec_config)

    logger.info("")
    logger.info("=" * 70)
    if is_multiecho:
        logger.info("Phase 2 Complete")
    else:
        logger.info("Workflow Complete")
    logger.info("=" * 70)
    logger.info("")

    # Get denoised output from work directory
    # This is motion-corrected + AROMA-denoised (if applicable), BEFORE filtering
    denoised_dir = work_dir / 'func_preproc' / 'denoised'
    denoised_files = list(denoised_dir.glob('*.nii.gz'))

    if not denoised_files:
        # Try derivatives directory (where DataSink saves it)
        derivatives_denoised_dir = derivatives_dir / 'denoised'
        if derivatives_denoised_dir.exists():
            denoised_files = list(derivatives_denoised_dir.glob('*.nii.gz'))

    if not denoised_files:
        # Try alternative location for AROMA output in work dir
        aroma_dir = work_dir / 'func_preproc' / 'ica_aroma' / 'aroma_output'
        if aroma_dir.exists():
            denoise_type = func_config.get('aroma', {}).get('denoise_type', 'nonaggr')
            denoised_files = list(aroma_dir.glob(f'denoised_func_data_{denoise_type}.nii.gz'))

    if not denoised_files:
        # Try BET masked output (if AROMA disabled)
        bet_dir = work_dir / 'func_preproc' / 'brain_extraction_4d' / 'apply_mask'
        if bet_dir.exists():
            denoised_files = list(bet_dir.glob('*.nii.gz'))

    if not denoised_files:
        raise FileNotFoundError(
            f"Denoised output not found. Searched:\n"
            f"  1. {denoised_dir}\n"
            f"  2. {derivatives_dir / 'denoised'}\n"
            f"  3. {work_dir / 'func_preproc' / 'ica_aroma' / 'aroma_output'}\n"
            f"  4. {work_dir / 'func_preproc' / 'brain_extraction_4d' / 'apply_mask'}"
        )

    denoised_output = denoised_files[0]
    logger.info(f"Denoised output (motion-corrected + AROMA): {denoised_output}")
    logger.info("")

    # Step 4: Apply ACompCor (if enabled)
    # CRITICAL: ACompCor is applied to denoised data (post-AROMA/TEDANA, pre-filtering)
    # This is the correct HCP-compliant order
    if acompcor_enabled and csf_mask and wm_mask and brain_file:
        logger.info("=" * 70)
        logger.info("STEP 4: ACompCor Nuisance Regression")
        logger.info("=" * 70)
        logger.info("Applying to denoised data (post-AROMA/TEDANA, pre-filtering)")
        logger.info("")

        # ACompCor intermediates go to work_dir (temporary)
        acompcor_dir = work_dir / 'acompcor'
        acompcor_dir.mkdir(parents=True, exist_ok=True)

        # Step 4a: Transform tissue masks to functional space using ANTs
        logger.info("Step 4a: Transforming tissue masks to functional space...")

        # Check if we have the ANTs registration transforms
        if 'func_to_t1w_transform' in results:
            # Use ANTs inverse transform to bring masks to functional space
            # Look for func_mean in registration directory (consistent location for both pipelines)
            func_mean_file = derivatives_dir / 'registration' / 'func_mean.nii.gz'

            # Fallback locations for backward compatibility
            if not func_mean_file.exists():
                func_mean_file = work_dir / 'func_mean_raw.nii.gz'  # Old single-echo
            if not func_mean_file.exists():
                func_mean_file = work_dir / 'mcflirt_echo2' / 'func_mean.nii.gz'  # Old multi-echo

            if not func_mean_file.exists():
                raise FileNotFoundError(
                    f"func_mean not found. Searched:\n"
                    f"  1. {derivatives_dir / 'registration' / 'func_mean.nii.gz'} (standard location)\n"
                    f"  2. {work_dir / 'func_mean_raw.nii.gz'} (old single-echo)\n"
                    f"  3. {work_dir / 'mcflirt_echo2' / 'func_mean.nii.gz'} (old multi-echo)"
                )

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

            # CRITICAL: Use motion-corrected mean (NOT bandpass_output)
            # Registration must use raw structural information before filtering
            func_mean_file = derivatives_dir / 'motion_correction' / 'func_mean.nii.gz'
            if not func_mean_file.exists():
                # Try multi-echo location
                func_mean_file = derivatives_dir / 'mcflirt_echo' / 'func_mean.nii.gz'

            if not func_mean_file.exists():
                # Create mean from motion-corrected data (NOT bandpass filtered)
                logger.info("  Creating functional mean from motion-corrected data...")
                func_mean_file = acompcor_dir / 'func_mean.nii.gz'
                compute_func_mean(func_input, func_mean_file)

            csf_func, wm_func, func_to_anat_bbr = register_masks_to_functional(
                t1w_brain=brain_file,
                func_ref=func_mean_file,  # Use motion-corrected mean (unfiltered)
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
        logger.info("Step 4b: Preparing ACompCor masks...")
        csf_eroded, wm_eroded = prepare_acompcor_masks(
            csf_mask=csf_func,
            wm_mask=wm_func,
            output_dir=acompcor_dir,
            csf_threshold=func_config.get('acompcor', {}).get('csf_threshold', 0.9),
            wm_threshold=func_config.get('acompcor', {}).get('wm_threshold', 0.9)
        )
        results['csf_acompcor_mask'] = csf_eroded
        results['wm_acompcor_mask'] = wm_eroded
        logger.info("")

        # Step 4c: Extract ACompCor components from denoised data
        logger.info("Step 4c: Extracting ACompCor components from denoised data...")
        acompcor_result = extract_acompcor_components(
            func_file=denoised_output,  # Use denoised, not bandpass!
            csf_mask=csf_eroded,
            wm_mask=wm_eroded,
            output_dir=acompcor_dir,
            num_components=func_config.get('acompcor', {}).get('num_components', 5),
            variance_threshold=func_config.get('acompcor', {}).get('variance_threshold', 0.5)
        )
        results['acompcor_components'] = acompcor_result['components_file']
        results['acompcor_variance'] = acompcor_result['variance_explained']
        logger.info("")

        # Step 4d: Regress out components from denoised data
        logger.info("Step 4d: Regressing out ACompCor components...")
        acompcor_cleaned = acompcor_dir / f'{subject}_bold_acompcor_cleaned.nii.gz'
        regress_out_components(
            func_file=denoised_output,  # Use denoised, not bandpass!
            components_file=acompcor_result['components_file'],
            output_file=acompcor_cleaned
        )
        logger.info(f"  ACompCor-cleaned output: {acompcor_cleaned}")
        logger.info("")

        # Use ACompCor-cleaned data for filtering
        filtering_input = acompcor_cleaned
    else:
        logger.info("ACompCor disabled - using denoised data directly for filtering")
        # No ACompCor - use denoised output directly
        filtering_input = denoised_output

    # Step 5: Apply bandpass filtering (HCP-compliant order: AFTER ACompCor)
    logger.info("=" * 70)
    logger.info("STEP 5: Bandpass Filtering")
    logger.info("=" * 70)
    logger.info("Applying to ACompCor-cleaned data (or denoised if ACompCor disabled)")
    logger.info("")

    bandpass_output = derivatives_dir / f'{subject}_bold_bandpass_filtered.nii.gz'
    highpass = func_config.get('highpass', 0.001)
    lowpass = func_config.get('lowpass', 0.08)
    tr = func_config.get('tr', 1.029)

    logger.info(f"Bandpass parameters: {highpass} - {lowpass} Hz, TR={tr}s")
    bandpass_cmd = [
        '3dBandpass',
        '-prefix', str(bandpass_output),
        '-input', str(filtering_input),
        str(highpass), str(lowpass), str(filtering_input)
    ]
    subprocess.run(bandpass_cmd, check=True, capture_output=True)
    logger.info(f"  Bandpass output: {bandpass_output}")
    logger.info("")

    # Step 6: Apply spatial smoothing (HCP-compliant order: AFTER bandpass)
    logger.info("=" * 70)
    logger.info("STEP 6: Spatial Smoothing")
    logger.info("=" * 70)
    logger.info("Applying to bandpass-filtered data")
    logger.info("")

    fwhm = func_config.get('fwhm', 6)
    smoothed_file = derivatives_dir / f'{subject}_bold_preprocessed.nii.gz'

    logger.info(f"Smoothing kernel: {fwhm}mm FWHM")
    smooth_cmd = [
        'fslmaths',
        str(bandpass_output),
        '-s', str(fwhm / 2.355),  # Convert FWHM to sigma
        str(smoothed_file)
    ]
    subprocess.run(smooth_cmd, check=True, capture_output=True)

    results['preprocessed'] = smoothed_file
    logger.info(f"  Final preprocessed output: {smoothed_file}")
    logger.info("")

    # Step 7: Get motion parameters and mask from derivatives (saved by DataSink)
    if is_multiecho:
        # Motion params already extracted during multi-echo preprocessing
        # Mask already created before TEDANA
        func_mask = derivatives_dir / 'func_mask.nii.gz'
    else:
        # Single-echo: get from derivatives directory where DataSink saved them
        motion_dir = derivatives_dir / 'motion_correction'
        brain_dir = derivatives_dir / 'brain'

        # Find motion parameters
        par_files = list(motion_dir.glob('*.par'))
        if par_files:
            results['motion_params'] = par_files[0]
            logger.info(f"Found motion parameters: {results['motion_params']}")

        # Find brain mask
        mask_files = list(brain_dir.glob('*mask.nii.gz'))
        if mask_files:
            func_mask = mask_files[0]
            logger.info(f"Found brain mask: {func_mask}")
        else:
            # Fallback to denoised directory
            func_mask = derivatives_dir / 'func_mask.nii.gz'
            logger.warning(f"Using fallback mask location: {func_mask}")

    # Step 8: Quality Control
    if func_config.get('run_qc', True):
        logger.info("=" * 70)
        logger.info("STEP 8: Quality Control")
        logger.info("=" * 70)
        logger.info("")

        qc_dir = study_root / 'qc' / subject / 'func'
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Motion QC (if motion params available)
        if 'motion_params' in results and results['motion_params'].exists():
            logger.info("Computing motion QC metrics...")
            motion_metrics = compute_motion_qc(
                motion_file=results['motion_params'],
                tr=func_config.get('tr', 1.029),
                output_dir=qc_dir,
                fd_threshold=func_config.get('fd_threshold', 0.5)
            )
            results['motion_qc'] = motion_metrics
            logger.info("")
        else:
            logger.warning("Motion parameters not found - skipping motion QC")
            motion_metrics = None

        # Brain mask QC
        logger.info("Computing brain mask QC...")
        # Find or compute functional mean image
        func_mean_file = None

        # Check work directory for pre-computed mean
        if is_multiecho:
            phase1_work_dir = work_dir.parent / 'func_phase1_motion'
            potential_mean = phase1_work_dir / 'mcflirt_echo2' / 'func_mean.nii.gz'
        else:
            potential_mean = work_dir / 'motion_correction' / 'func_mean.nii.gz'

        if potential_mean.exists():
            func_mean_file = potential_mean
            logger.info(f"  Using pre-computed mean: {func_mean_file}")
        else:
            # Compute mean from preprocessed data
            func_mean_file = qc_dir / 'func_mean.nii.gz'
            logger.info(f"  Computing mean from preprocessed data...")
            subprocess.run([
                'fslmaths', str(results['preprocessed']), '-Tmean', str(func_mean_file)
            ], check=True, capture_output=True)
            logger.info(f"  Saved: {func_mean_file}")

        # Run brain mask QC
        skull_strip_metrics = compute_skull_strip_qc(
            func_mean_file=func_mean_file,
            mask_file=func_mask,
            output_dir=qc_dir,
            subject=subject
        )
        results['skull_strip_qc'] = skull_strip_metrics
        logger.info("")

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
            dvars_threshold=func_config.get('dvars_threshold', 1.5)
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
            tr=func_config.get('tr', 1.029)
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
                skull_strip_metrics=skull_strip_metrics,
                tedana_report=results.get('tedana_report'),
                output_file=qc_dir / f'{subject}_func_qc_report.html'
            )
            results['qc_report'] = qc_report
            logger.info("")
        else:
            logger.warning("Skipping QC report generation (no motion metrics)")

    # Step 9: Spatial normalization to MNI152 (optional, reuses anatomical transforms)
    if func_config.get('normalize_to_mni', False):
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 9: Normalizing functional data to MNI152")
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
            if not anat_transforms:
                logger.warning("  Missing anatomical transforms (T1w→MNI)")
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


# get_execution_config is imported from neurovrai.utils.workflow
# It properly accesses config['execution']['n_procs']
