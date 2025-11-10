#!/usr/bin/env python3
"""
Anatomical (T1w) preprocessing workflow using Nipype and FSL.

Workflow steps:
1. Reorientation to standard orientation (fslreorient2std)
2. Skull stripping (BET)
3. Bias field correction (FAST)
4. Linear registration to MNI152 (FLIRT)
5. Nonlinear registration to MNI152 (FNIRT)
6. Save transformations to TransformRegistry

The computed transformations are saved and reused by other workflows
(diffusion, functional, myelin) to avoid duplicate computation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any

from nipype import Workflow, Node, MapNode
from nipype.interfaces import fsl, utility as niu
from nipype.interfaces.io import DataSink

from mri_preprocess.utils.workflow import (
    setup_logging,
    get_fsl_config,
    get_node_config,
    get_reference_template,
    get_execution_config,
    validate_inputs
)
from mri_preprocess.utils.transforms import TransformRegistry
from mri_preprocess.utils.bids import get_derivatives_dir


def create_reorient_node(name: str = 'reorient') -> Node:
    """
    Create reorientation node.

    Reorients image to standard (RAS) orientation.

    Parameters
    ----------
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for reorientation

    Examples
    --------
    >>> reorient = create_reorient_node()
    >>> reorient.inputs.in_file = "T1w.nii.gz"
    """
    reorient = Node(
        fsl.Reorient2Std(),
        name=name
    )
    return reorient


def create_skull_strip_node(
    config: Dict[str, Any],
    name: str = 'skull_strip'
) -> Node:
    """
    Create skull stripping node using BET.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for skull stripping

    Examples
    --------
    >>> bet = create_skull_strip_node(config)
    >>> bet.inputs.in_file = "T1w_reoriented.nii.gz"
    """
    # Get BET configuration
    bet_config = get_node_config('bet', config)

    bet = Node(
        fsl.BET(),
        name=name
    )

    # Set parameters from config
    bet.inputs.frac = bet_config.get('frac', 0.5)
    bet.inputs.robust = bet_config.get('robust', True)
    bet.inputs.mask = True  # Generate brain mask
    bet.inputs.output_type = 'NIFTI_GZ'

    return bet


def create_bias_correction_node(
    config: Dict[str, Any],
    name: str = 'bias_correction'
) -> Node:
    """
    Create bias field correction node using FAST.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for bias correction

    Examples
    --------
    >>> fast = create_bias_correction_node(config)
    >>> fast.inputs.in_files = "T1w_brain.nii.gz"
    """
    # Get FAST configuration
    fast_config = get_node_config('fast', config)

    fast = Node(
        fsl.FAST(),
        name=name
    )

    # Set parameters
    fast.inputs.img_type = 1  # T1-weighted
    fast.inputs.bias_iters = fast_config.get('bias_iters', 4)
    fast.inputs.bias_lowpass = fast_config.get('bias_lowpass', 10)
    fast.inputs.segments = True
    fast.inputs.output_biascorrected = True
    fast.inputs.output_type = 'NIFTI_GZ'

    return fast


def create_linear_registration_node(
    config: Dict[str, Any],
    name: str = 'linear_reg'
) -> Node:
    """
    Create linear (affine) registration node using FLIRT.

    Registers T1w brain to MNI152 template.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for linear registration

    Examples
    --------
    >>> flirt = create_linear_registration_node(config)
    >>> flirt.inputs.in_file = "T1w_brain.nii.gz"
    >>> flirt.inputs.reference = "/path/to/MNI152_T1_2mm_brain.nii.gz"
    """
    # Get FLIRT configuration
    flirt_config = get_node_config('flirt', config)

    flirt = Node(
        fsl.FLIRT(),
        name=name
    )

    # Set parameters
    flirt.inputs.dof = flirt_config.get('dof', 12)  # 12 DOF affine
    flirt.inputs.cost = flirt_config.get('cost', 'corratio')
    flirt.inputs.searchr_x = flirt_config.get('searchr_x', [-90, 90])
    flirt.inputs.searchr_y = flirt_config.get('searchr_y', [-90, 90])
    flirt.inputs.searchr_z = flirt_config.get('searchr_z', [-90, 90])
    flirt.inputs.interp = 'trilinear'
    flirt.inputs.output_type = 'NIFTI_GZ'

    return flirt


def create_nonlinear_registration_node(
    config: Dict[str, Any],
    name: str = 'nonlinear_reg'
) -> Node:
    """
    Create nonlinear registration node using FNIRT.

    Performs nonlinear registration to MNI152 template.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for nonlinear registration

    Examples
    --------
    >>> fnirt = create_nonlinear_registration_node(config)
    >>> fnirt.inputs.in_file = "T1w_brain.nii.gz"
    >>> fnirt.inputs.ref_file = "/path/to/MNI152_T1_2mm.nii.gz"
    >>> fnirt.inputs.affine_file = "T1w_to_MNI_affine.mat"
    """
    # Get FNIRT configuration
    fnirt_config = get_node_config('fnirt', config)

    fnirt = Node(
        fsl.FNIRT(),
        name=name
    )

    # Set parameters
    fnirt.inputs.fieldcoeff_file = True  # Output warp coefficients
    fnirt.inputs.warped_file = True  # Output warped image
    fnirt.inputs.output_type = 'NIFTI_GZ'

    # Optional config parameters
    if 'warp_resolution' in fnirt_config:
        fnirt.inputs.warp_resolution = fnirt_config['warp_resolution']

    return fnirt


def create_anat_preprocessing_workflow(
    config: Dict[str, Any],
    subject: str,
    t1w_file: Path,
    output_dir: Path,
    work_dir: Path,
    session: Optional[str] = None,
    name: str = 'anat_preprocess'
) -> Workflow:
    """
    Create complete anatomical preprocessing workflow.

    This workflow:
    1. Reorients T1w to standard orientation
    2. Performs skull stripping
    3. Applies bias field correction
    4. Registers to MNI152 space (linear + nonlinear)
    5. Saves transformations to TransformRegistry for reuse

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    t1w_file : Path
        Input T1w NIfTI file
    output_dir : Path
        Output directory for derivatives
    work_dir : Path
        Working directory for intermediate files
    session : str, optional
        Session identifier
    name : str
        Workflow name

    Returns
    -------
    Workflow
        Nipype workflow ready to run

    Examples
    --------
    >>> from mri_preprocess.config import load_config
    >>> config = load_config("study.yaml")
    >>> wf = create_anat_preprocessing_workflow(
    ...     config=config,
    ...     subject="sub-001",
    ...     t1w_file=Path("/data/rawdata/sub-001/anat/sub-001_T1w.nii.gz"),
    ...     output_dir=Path("/data/derivatives/mri-preprocess"),
    ...     work_dir=Path("/tmp/work")
    ... )
    >>> wf.run()
    """
    # Validate inputs
    validate_inputs(t1w_file)

    # Setup logging
    log_dir = Path(config['paths']['logs'])
    logger = setup_logging(log_dir, subject, name)
    logger.info(f"Creating anatomical preprocessing workflow for {subject}")

    # Create workflow
    wf = Workflow(name=name)
    wf.base_dir = str(work_dir)

    # Get reference template
    mni_template = get_reference_template('mni152_t1_2mm', config)
    mni_brain_template = get_reference_template('mni152_t1_2mm', config)

    logger.info(f"Using MNI template: {mni_template}")

    # === Create nodes ===

    # Input node
    inputnode = Node(
        niu.IdentityInterface(fields=['t1w']),
        name='inputnode'
    )
    inputnode.inputs.t1w = str(t1w_file)

    # 1. Reorientation
    reorient = create_reorient_node()

    # 2. Skull stripping
    skull_strip = create_skull_strip_node(config)

    # 3. Bias correction
    bias_correct = create_bias_correction_node(config)

    # 4. Linear registration
    linear_reg = create_linear_registration_node(config)
    linear_reg.inputs.reference = str(mni_brain_template)

    # 5. Nonlinear registration
    nonlinear_reg = create_nonlinear_registration_node(config)
    nonlinear_reg.inputs.ref_file = str(mni_template)

    # Output node
    outputnode = Node(
        niu.IdentityInterface(
            fields=[
                'reoriented_t1w',
                'brain',
                'brain_mask',
                'bias_corrected',
                'tissue_class_map',
                'csf_prob',
                'gm_prob',
                'wm_prob',
                'mni_affine_mat',
                'mni_warp',
                'mni_warped'
            ]
        ),
        name='outputnode'
    )

    # DataSink for outputs
    datasink = Node(
        DataSink(),
        name='datasink'
    )
    datasink.inputs.base_directory = str(
        get_derivatives_dir(output_dir, 'mri-preprocess', subject, session, create=True)
    )
    datasink.inputs.container = 'anat'

    # === Connect workflow ===

    # Reorientation
    wf.connect([
        (inputnode, reorient, [('t1w', 'in_file')]),
        (reorient, outputnode, [('out_file', 'reoriented_t1w')])
    ])

    # Skull stripping
    wf.connect([
        (reorient, skull_strip, [('out_file', 'in_file')]),
        (skull_strip, outputnode, [
            ('out_file', 'brain'),
            ('mask_file', 'brain_mask')
        ])
    ])

    # Bias correction and tissue segmentation
    wf.connect([
        (skull_strip, bias_correct, [('out_file', 'in_files')]),
        (bias_correct, outputnode, [
            ('restored_image', 'bias_corrected'),
            ('tissue_class_map', 'tissue_class_map'),
            ('probability_maps', 'csf_prob', lambda x: x[0]),  # CSF = index 0
            ('probability_maps', 'gm_prob', lambda x: x[1]),   # GM = index 1
            ('probability_maps', 'wm_prob', lambda x: x[2])    # WM = index 2
        ])
    ])

    # Linear registration
    wf.connect([
        (bias_correct, linear_reg, [('restored_image', 'in_file')]),
        (linear_reg, outputnode, [('out_matrix_file', 'mni_affine_mat')])
    ])

    # Nonlinear registration
    wf.connect([
        (reorient, nonlinear_reg, [('out_file', 'in_file')]),
        (linear_reg, nonlinear_reg, [('out_matrix_file', 'affine_file')]),
        (nonlinear_reg, outputnode, [
            ('fieldcoeff_file', 'mni_warp'),
            ('warped_file', 'mni_warped')
        ])
    ])

    # Save outputs
    wf.connect([
        (outputnode, datasink, [
            ('reoriented_t1w', 'reoriented'),
            ('brain', 'brain'),
            ('brain_mask', 'mask'),
            ('bias_corrected', 'bias_corrected'),
            ('tissue_class_map', 'segmentation.@tissue_class'),
            ('csf_prob', 'segmentation.@csf'),
            ('gm_prob', 'segmentation.@gm'),
            ('wm_prob', 'segmentation.@wm'),
            ('mni_affine_mat', 'transforms.@affine'),
            ('mni_warp', 'transforms.@warp'),
            ('mni_warped', 'mni_space')
        ])
    ])

    logger.info("Anatomical preprocessing workflow created")

    return wf


def run_anat_preprocessing(
    config: Dict[str, Any],
    subject: str,
    t1w_file: Path,
    output_dir: Path,
    work_dir: Path,
    session: Optional[str] = None,
    save_transforms: bool = True
) -> Dict[str, Path]:
    """
    Run anatomical preprocessing and save transformations.

    This is the main entry point for anatomical preprocessing.
    It creates and runs the workflow, then saves transformations
    to the TransformRegistry.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    t1w_file : Path
        Input T1w file
    output_dir : Path
        Output directory
    work_dir : Path
        Working directory
    session : str, optional
        Session identifier
    save_transforms : bool
        Save transforms to registry (default: True)

    Returns
    -------
    dict
        Dictionary with output file paths

    Examples
    --------
    >>> from mri_preprocess.config import load_config
    >>> config = load_config("study.yaml")
    >>> results = run_anat_preprocessing(
    ...     config=config,
    ...     subject="sub-001",
    ...     t1w_file=Path("/data/rawdata/sub-001/anat/sub-001_T1w.nii.gz"),
    ...     output_dir=Path("/data/derivatives/mri-preprocess"),
    ...     work_dir=Path("/tmp/work")
    ... )
    >>> print(results['brain'])
    """
    # Create workflow
    wf = create_anat_preprocessing_workflow(
        config=config,
        subject=subject,
        t1w_file=t1w_file,
        output_dir=output_dir,
        work_dir=work_dir,
        session=session
    )

    # Get execution configuration
    exec_config = get_execution_config(config)

    # Run workflow
    wf.run(**exec_config)

    # Collect outputs
    derivatives_dir = get_derivatives_dir(
        output_dir, 'mri-preprocess', subject, session
    )
    anat_dir = derivatives_dir / 'anat'

    segmentation_dir = anat_dir / 'segmentation'

    outputs = {
        'brain': list(anat_dir.glob('*brain.nii.gz'))[0] if anat_dir.exists() else None,
        'brain_mask': list(anat_dir.glob('*mask.nii.gz'))[0] if anat_dir.exists() else None,
        'bias_corrected': list(anat_dir.glob('*bias_corrected.nii.gz'))[0] if anat_dir.exists() else None,
        'csf_prob': list(segmentation_dir.glob('*pve_0*.nii.gz'))[0] if segmentation_dir.exists() else None,
        'gm_prob': list(segmentation_dir.glob('*pve_1*.nii.gz'))[0] if segmentation_dir.exists() else None,
        'wm_prob': list(segmentation_dir.glob('*pve_2*.nii.gz'))[0] if segmentation_dir.exists() else None,
        'tissue_class_map': list(segmentation_dir.glob('*seg*.nii.gz'))[0] if segmentation_dir.exists() else None,
        'mni_affine': list((anat_dir / 'transforms').glob('*.mat'))[0] if (anat_dir / 'transforms').exists() else None,
        'mni_warp': list((anat_dir / 'transforms').glob('*warp.nii.gz'))[0] if (anat_dir / 'transforms').exists() else None,
        'mni_warped': list((anat_dir / 'mni_space').glob('*.nii.gz'))[0] if (anat_dir / 'mni_space').exists() else None
    }

    # Save transformations to registry
    if save_transforms and outputs['mni_affine'] and outputs['mni_warp']:
        from mri_preprocess.utils.transforms import create_transform_registry

        registry = create_transform_registry(config, subject, session)

        # Save nonlinear transformation
        registry.save_nonlinear_transform(
            warp_file=outputs['mni_warp'],
            affine_file=outputs['mni_affine'],
            source_space='T1w',
            target_space='MNI152',
            reference=get_reference_template('mni152_t1_2mm', config),
            source_image=t1w_file
        )

        logging.info(f"Saved T1w�MNI152 transformation to registry")

    return outputs
