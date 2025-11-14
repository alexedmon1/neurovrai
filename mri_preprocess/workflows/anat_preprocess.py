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
from nipype.interfaces import fsl, ants, utility as niu
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
    Create bias correction node using ANTs N4BiasFieldCorrection.

    Fast bias correction (~2.5 min on high-res data).

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for N4 bias correction

    Examples
    --------
    >>> n4 = create_bias_correction_node(config)
    >>> n4.inputs.input_image = "T1w_brain.nii.gz"
    """
    n4 = Node(
        ants.N4BiasFieldCorrection(),
        name=name
    )

    # Set parameters for fast processing
    n4.inputs.dimension = 3
    n4.inputs.n_iterations = [50, 50, 30, 20]  # Reduced iterations for speed
    n4.inputs.shrink_factor = 3  # Downsample for speed
    n4.inputs.convergence_threshold = 0.001
    n4.inputs.bspline_fitting_distance = 300

    return n4


def create_segmentation_node(
    config: Dict[str, Any],
    name: str = 'segmentation'
) -> Node:
    """
    Create tissue segmentation node using ANTs Atropos.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for tissue segmentation

    Examples
    --------
    >>> atropos = create_segmentation_node(config)
    >>> atropos.inputs.intensity_images = "T1w_brain.nii.gz"
    >>> atropos.inputs.mask_image = "T1w_brain_mask.nii.gz"
    """
    atropos = Node(
        ants.Atropos(),
        name=name
    )

    # Set parameters for 3-class segmentation
    # Input will be the N4-bias-corrected image
    atropos.inputs.dimension = 3
    atropos.inputs.number_of_tissue_classes = 3  # CSF, GM, WM
    atropos.inputs.initialization = 'KMeans'  # Nipype expects just the method name
    atropos.inputs.likelihood_model = 'Gaussian'
    atropos.inputs.mrf_smoothing_factor = 0.1
    atropos.inputs.mrf_radius = [1, 1, 1]
    atropos.inputs.convergence_threshold = 0.001
    atropos.inputs.n_iterations = 5
    atropos.inputs.save_posteriors = True  # Save probability maps
    atropos.inputs.output_posteriors_name_template = 'POSTERIOR_%02d.nii.gz'

    return atropos


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
    fnirt.inputs.output_type = 'NIFTI_GZ'
    # Enable warp coefficient file output (required for normalization)
    fnirt.inputs.fieldcoeff_file = True  # Auto-generates filename
    # warped_file is auto-generated by default

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

    # 3. Bias correction (ANTs N4, fast ~2.5min)
    bias_correct = create_bias_correction_node(config)

    # 4. Tissue segmentation (ANTs Atropos)
    segment = create_segmentation_node(config)

    # 5. Linear registration
    linear_reg = create_linear_registration_node(config)
    linear_reg.inputs.reference = str(mni_brain_template)

    # 6. Nonlinear registration
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
    # Set container to empty string to avoid redundant /anat/anat/ hierarchy
    # (base_directory is already {study_root}/derivatives/{subject}/anat/)
    datasink.inputs.container = ''

    # === Connect workflow ===

    # Reorientation
    wf.connect(inputnode, 't1w', reorient, 'in_file')
    wf.connect(reorient, 'out_file', outputnode, 'reoriented_t1w')

    # Skull stripping
    wf.connect(reorient, 'out_file', skull_strip, 'in_file')
    wf.connect(skull_strip, 'out_file', outputnode, 'brain')
    wf.connect(skull_strip, 'mask_file', outputnode, 'brain_mask')

    # Bias correction (ANTs N4)
    wf.connect(skull_strip, 'out_file', bias_correct, 'input_image')
    wf.connect(skull_strip, 'mask_file', bias_correct, 'mask_image')
    wf.connect(bias_correct, 'output_image', outputnode, 'bias_corrected')

    # Function to extract tissue maps from probability_maps list
    def extract_tissue_map(probability_maps, index):
        """Extract a single tissue map from posteriors list."""
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

    # Tissue segmentation using ANTs Atropos (on N4-bias-corrected image)
    # Atropos requires both intensity image and mask
    wf.connect(bias_correct, 'output_image', segment, 'intensity_images')
    wf.connect(skull_strip, 'mask_file', segment, 'mask_image')
    wf.connect(segment, 'classified_image', outputnode, 'tissue_class_map')
    wf.connect(segment, 'posteriors', extract_csf, 'probability_maps')
    wf.connect(segment, 'posteriors', extract_gm, 'probability_maps')
    wf.connect(segment, 'posteriors', extract_wm, 'probability_maps')
    wf.connect(extract_csf, 'tissue_map', outputnode, 'csf_prob')
    wf.connect(extract_gm, 'tissue_map', outputnode, 'gm_prob')
    wf.connect(extract_wm, 'tissue_map', outputnode, 'wm_prob')

    # Linear registration (use bias-corrected brain)
    wf.connect(bias_correct, 'output_image', linear_reg, 'in_file')
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
    # output_dir is the derivatives base (e.g., /mnt/bytopia/IRC805/derivatives)
    # Use standardized hierarchy: {outdir}/{subject}/{modality}/
    outdir = Path(output_dir)

    # Create simple, standardized hierarchy
    derivatives_dir = outdir / subject / 'anat'
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
    # Find output files in derivatives directory
    # Handle both old (anat/anat/) and new (anat/) hierarchy
    anat_dir_old = derivatives_dir / 'anat'  # Old double-anat hierarchy
    anat_dir_new = derivatives_dir          # New flat hierarchy

    # Check which structure exists
    if (anat_dir_old / 'brain').exists():
        anat_dir = anat_dir_old  # Old structure
    else:
        anat_dir = anat_dir_new  # New structure

    # Find files using recursive globbing to handle both structures
    outputs = {
        'brain': list(derivatives_dir.glob('**/brain/*.nii.gz'))[0] if list(derivatives_dir.glob('**/brain/*.nii.gz')) else None,
        'brain_mask': list(derivatives_dir.glob('**/mask/*.nii.gz'))[0] if list(derivatives_dir.glob('**/mask/*.nii.gz')) else None,
        'bias_corrected': list(derivatives_dir.glob('**/bias_corrected/*.nii.gz'))[0] if list(derivatives_dir.glob('**/bias_corrected/*.nii.gz')) else None,
        # Atropos outputs: POSTERIOR_01.nii.gz (CSF), POSTERIOR_02.nii.gz (GM), POSTERIOR_03.nii.gz (WM)
        'csf_prob': list(derivatives_dir.glob('**/segmentation/POSTERIOR_01.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/POSTERIOR_01.nii.gz')) else None,
        'gm_prob': list(derivatives_dir.glob('**/segmentation/POSTERIOR_02.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/POSTERIOR_02.nii.gz')) else None,
        'wm_prob': list(derivatives_dir.glob('**/segmentation/POSTERIOR_03.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/POSTERIOR_03.nii.gz')) else None,
        'tissue_class_map': list(derivatives_dir.glob('**/segmentation/*labeled.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/*labeled.nii.gz')) else None,
        'mni_affine': list(derivatives_dir.glob('**/transforms/*.mat'))[0] if list(derivatives_dir.glob('**/transforms/*.mat')) else None,
        'mni_warp': list(derivatives_dir.glob('**/transforms/*warp.nii.gz'))[0] if list(derivatives_dir.glob('**/transforms/*warp.nii.gz')) else None,
        'mni_warped': list(derivatives_dir.glob('**/mni_space/*.nii.gz'))[0] if list(derivatives_dir.glob('**/mni_space/*.nii.gz')) else None
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

        qc_dir = study_root / 'qc' / subject / 'anat'
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
                    anat_dir=derivatives_dir,
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
