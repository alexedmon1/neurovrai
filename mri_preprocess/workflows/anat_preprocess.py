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
    # fieldcoeff_file and warped_file will be auto-generated by Nipype
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
    # output_dir is now the full derivatives directory path
    datasink.inputs.base_directory = str(output_dir)
    datasink.inputs.container = 'anat'

    # === Connect workflow ===

    # Reorientation
    wf.connect(inputnode, 't1w', reorient, 'in_file')
    wf.connect(reorient, 'out_file', outputnode, 'reoriented_t1w')

    # Skull stripping
    wf.connect(reorient, 'out_file', skull_strip, 'in_file')
    wf.connect(skull_strip, 'out_file', outputnode, 'brain')
    wf.connect(skull_strip, 'mask_file', outputnode, 'brain_mask')

    # Function to extract tissue maps from probability_maps list
    def extract_tissue_map(probability_maps, index):
        """Extract a single tissue map from FAST probability maps."""
        return probability_maps[index]

    extract_csf = Node(
        niu.Function(
            input_names=['probability_maps', 'index'],
            output_names=['tissue_map'],
            function=extract_tissue_map
        ),
        name='extract_csf'
    )
    extract_csf.inputs.index = 0  # CSF

    extract_gm = Node(
        niu.Function(
            input_names=['probability_maps', 'index'],
            output_names=['tissue_map'],
            function=extract_tissue_map
        ),
        name='extract_gm'
    )
    extract_gm.inputs.index = 1  # GM

    extract_wm = Node(
        niu.Function(
            input_names=['probability_maps', 'index'],
            output_names=['tissue_map'],
            function=extract_tissue_map
        ),
        name='extract_wm'
    )
    extract_wm.inputs.index = 2  # WM

    # Bias correction and tissue segmentation
    wf.connect(skull_strip, 'out_file', bias_correct, 'in_files')
    wf.connect(bias_correct, 'restored_image', outputnode, 'bias_corrected')
    wf.connect(bias_correct, 'tissue_class_map', outputnode, 'tissue_class_map')
    wf.connect(bias_correct, 'probability_maps', extract_csf, 'probability_maps')
    wf.connect(bias_correct, 'probability_maps', extract_gm, 'probability_maps')
    wf.connect(bias_correct, 'probability_maps', extract_wm, 'probability_maps')
    wf.connect(extract_csf, 'tissue_map', outputnode, 'csf_prob')
    wf.connect(extract_gm, 'tissue_map', outputnode, 'gm_prob')
    wf.connect(extract_wm, 'tissue_map', outputnode, 'wm_prob')

    # Linear registration
    wf.connect(bias_correct, 'restored_image', linear_reg, 'in_file')
    wf.connect(linear_reg, 'out_matrix_file', outputnode, 'mni_affine_mat')

    # Nonlinear registration
    wf.connect(reorient, 'out_file', nonlinear_reg, 'in_file')
    wf.connect(linear_reg, 'out_matrix_file', nonlinear_reg, 'affine_file')
    wf.connect(nonlinear_reg, 'fieldcoeff_file', outputnode, 'mni_warp')
    wf.connect(nonlinear_reg, 'warped_file', outputnode, 'mni_warped')

    # Save outputs
    wf.connect(outputnode, 'reoriented_t1w', datasink, 'reoriented')
    wf.connect(outputnode, 'brain', datasink, 'brain')
    wf.connect(outputnode, 'brain_mask', datasink, 'mask')
    wf.connect(outputnode, 'bias_corrected', datasink, 'bias_corrected')
    wf.connect(outputnode, 'tissue_class_map', datasink, 'segmentation.@tissue_class')
    wf.connect(outputnode, 'csf_prob', datasink, 'segmentation.@csf')
    wf.connect(outputnode, 'gm_prob', datasink, 'segmentation.@gm')
    wf.connect(outputnode, 'wm_prob', datasink, 'segmentation.@wm')
    wf.connect(outputnode, 'mni_affine_mat', datasink, 'transforms.@affine')
    wf.connect(outputnode, 'mni_warp', datasink, 'transforms.@warp')
    wf.connect(outputnode, 'mni_warped', datasink, 'mni_space')

    logger.info("Anatomical preprocessing workflow created")

    return wf


def run_anat_preprocessing(
    config: Dict[str, Any],
    subject: str,
    t1w_file: Path,
    output_dir: Path,
    work_dir: Optional[Path] = None,
    session: Optional[str] = None,
    save_transforms: bool = True,
    run_qc: bool = True
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
        Study root directory (e.g., /mnt/bytopia/development/IRC805/)
        Derivatives will be saved to: {output_dir}/derivatives/anat_preproc/{subject}/
    work_dir : Path, optional
        Working directory for temporary Nipype files
        Default: {output_dir}/work/{subject}/anat_preproc/
    session : str, optional
        Session identifier
    save_transforms : bool
        Save transforms to registry (default: True)
    run_qc : bool
        Run quality control after preprocessing (default: True)

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
    # Setup directory structure
    # output_dir is the study root (e.g., /mnt/bytopia/development/IRC805/)
    study_root = Path(output_dir)

    # Create directory hierarchy
    derivatives_dir = study_root / 'derivatives' / 'anat_preproc' / subject
    if work_dir is None:
        work_dir = study_root / 'work' / subject / 'anat_preproc'
    else:
        work_dir = Path(work_dir)

    derivatives_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create workflow
    wf = create_anat_preprocessing_workflow(
        config=config,
        subject=subject,
        t1w_file=t1w_file,
        output_dir=derivatives_dir,  # Pass derivatives directory directly
        work_dir=work_dir,
        session=session
    )

    # Get execution configuration
    exec_config = get_execution_config(config)

    # Run workflow
    wf.run(**exec_config)

    # Collect outputs
    # derivatives_dir is already set at the top of the function
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

    # Run Quality Control
    if run_qc:
        logging.info("="*70)
        logging.info("Running Anatomical QC")
        logging.info("="*70)

        qc_dir = study_root / 'qc' / 'anat' / subject
        qc_results = {}

        try:
            # 1. Skull Strip QC
            if outputs.get('brain_mask'):
                from mri_preprocess.qc.anat.skull_strip_qc import SkullStripQualityControl

                logging.info("Running Skull Strip QC...")
                skull_qc = SkullStripQualityControl(
                    subject=subject,
                    anat_dir=derivatives_dir / 'anat',
                    qc_dir=qc_dir / 'skull_strip'
                )
                qc_results['skull_strip'] = skull_qc.run_qc(
                    t1w_file=t1w_file,
                    brain_file=outputs.get('brain'),
                    mask_file=outputs.get('brain_mask')
                )
                logging.info("  ✓ Skull Strip QC completed")

            # 2. Segmentation QC
            if outputs.get('csf_prob') or outputs.get('gm_prob') or outputs.get('wm_prob'):
                from mri_preprocess.qc.anat.segmentation_qc import SegmentationQualityControl

                logging.info("Running Segmentation QC...")
                seg_qc = SegmentationQualityControl(
                    subject=subject,
                    anat_dir=segmentation_dir,
                    qc_dir=qc_dir / 'segmentation'
                )
                qc_results['segmentation'] = seg_qc.run_qc(
                    csf_file=outputs.get('csf_prob'),
                    gm_file=outputs.get('gm_prob'),
                    wm_file=outputs.get('wm_prob')
                )
                logging.info("  ✓ Segmentation QC completed")

            # 3. Registration QC
            if outputs.get('mni_warped'):
                from mri_preprocess.qc.anat.registration_qc import RegistrationQualityControl

                logging.info("Running Registration QC...")
                reg_qc = RegistrationQualityControl(
                    subject=subject,
                    anat_dir=anat_dir / 'mni_space',
                    qc_dir=qc_dir / 'registration'
                )
                qc_results['registration'] = reg_qc.run_qc(
                    registered_file=outputs.get('mni_warped'),
                    template_file=None,  # Use FSL MNI152 template
                    registered_mask=None,  # Will be auto-detected or generated
                    template_mask=None  # Use FSL MNI152 mask
                )
                logging.info("  ✓ Registration QC completed")

            # Save combined QC results
            import json
            combined_qc_file = qc_dir / 'combined_qc_results.json'
            combined_qc_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert Path objects to strings for JSON serialization
            qc_results_serializable = json.loads(json.dumps(qc_results, default=str))

            with open(combined_qc_file, 'w') as f:
                json.dump(qc_results_serializable, f, indent=2)

            logging.info(f"Saved combined QC results: {combined_qc_file}")
            logging.info("="*70)
            logging.info("Anatomical QC Complete")
            logging.info("="*70)

            outputs['qc_results'] = qc_results
            outputs['qc_dir'] = qc_dir

        except Exception as e:
            logging.error(f"QC failed with error: {e}")
            logging.warning("Continuing without QC...")

    return outputs
