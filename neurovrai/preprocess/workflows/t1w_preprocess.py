#!/usr/bin/env python3
"""
T1-weighted (T1w) anatomical preprocessing workflow using Nipype and FSL.

Workflow steps:
1. Reorientation to standard orientation (fslreorient2std)
2. Skull stripping (BET)
3. Bias field correction (N4)
4. Tissue segmentation (Atropos)
5. Linear registration to MNI152 (FLIRT)
6. Nonlinear registration to MNI152 (FNIRT/ANTs)
7. Save transformations to TransformRegistry

The computed transformations are saved and reused by other workflows
(diffusion, functional, T2w, T1-T2-ratio) to avoid duplicate computation.

Note: This module was renamed from anat_preprocess.py. Backward compatibility
aliases (run_anat_preprocessing, create_anat_preprocessing_workflow) are provided.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import shutil

from nipype import Workflow, Node, MapNode
from nipype.interfaces import fsl, ants, utility as niu
from nipype.interfaces.io import DataSink
import nibabel as nib
import numpy as np

from neurovrai.utils.workflow import (
    setup_logging,
    get_fsl_config,
    get_node_config,
    get_reference_template,
    get_execution_config,
    validate_inputs
)
from neurovrai.utils.transforms import TransformRegistry


def standardize_atropos_tissues(
    derivatives_dir: Path,
    use_symlinks: bool = True
) -> bool:
    """
    Standardize Atropos tissue posterior names after segmentation.

    Atropos with K-means initialization produces POSTERIOR files in arbitrary order.
    This function identifies tissues based on mean T1w intensity and creates
    standardized symlinks: csf.nii.gz, gm.nii.gz, wm.nii.gz

    Parameters
    ----------
    derivatives_dir : Path
        Subject derivatives directory (contains anat/segmentation/)
    use_symlinks : bool
        If True, create symlinks; if False, copy files

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        seg_dir = derivatives_dir / 'segmentation'
        if not seg_dir.exists():
            logger.warning(f"Segmentation directory not found: {seg_dir}")
            return False

        # Check if already standardized
        existing = [seg_dir / f'{tissue}.nii.gz' for tissue in ['csf', 'gm', 'wm']]
        if all(f.exists() for f in existing):
            logger.info("Tissues already standardized")
            return True

        # Find T1w brain for intensity-based identification
        t1w_brain_candidates = [
            derivatives_dir / 'brain.nii.gz',
            derivatives_dir / 'T1w_brain.nii.gz',
        ]

        # Also check in brain/ subdirectory (Nipype DataSink structure)
        brain_dir = derivatives_dir / 'brain'
        if brain_dir.exists():
            brain_files = list(brain_dir.glob('*brain.nii.gz'))
            t1w_brain_candidates.extend(brain_files)

        # Use the first one that exists
        t1w_brain = None
        for candidate in t1w_brain_candidates:
            if candidate and candidate.exists():
                t1w_brain = candidate
                break

        if t1w_brain is None:
            logger.warning("T1w brain not found for tissue identification")
            return False

        # Load T1w image
        logger.info(f"Identifying Atropos tissues using {t1w_brain.name}...")
        t1w_img = nib.load(str(t1w_brain))
        t1w_data = t1w_img.get_fdata()

        # Load all posteriors and calculate mean T1w intensity
        posteriors = {}
        for post_file in sorted(seg_dir.glob('POSTERIOR_*.nii.gz')):
            post_img = nib.load(str(post_file))
            post_data = post_img.get_fdata()

            # Calculate weighted mean intensity
            masked_intensity = t1w_data * post_data
            mean_intensity = np.sum(masked_intensity) / (np.sum(post_data) + 1e-10)
            posteriors[post_file] = mean_intensity

        if len(posteriors) < 3:
            logger.warning(f"Found only {len(posteriors)} posteriors, need 3")
            return False

        # Sort by intensity (CSF=lowest, GM=middle, WM=highest)
        sorted_posteriors = sorted(posteriors.items(), key=lambda x: x[1])

        tissue_map = {
            'csf': sorted_posteriors[0][0],  # Lowest intensity
            'gm': sorted_posteriors[1][0],   # Middle intensity
            'wm': sorted_posteriors[2][0]    # Highest intensity
        }

        # Log identification
        for tissue, post_file in tissue_map.items():
            intensity = posteriors[post_file]
            logger.info(f"  {tissue.upper():3s} -> {post_file.name} (intensity: {intensity:.2f})")

        # Create standardized files
        for tissue, source_file in tissue_map.items():
            target_file = seg_dir / f'{tissue}.nii.gz'

            if use_symlinks:
                # Create relative symlink
                rel_source = source_file.relative_to(seg_dir)
                target_file.symlink_to(rel_source)
                logger.info(f"  Created symlink: {target_file.name} -> {source_file.name}")
            else:
                # Copy file
                shutil.copy2(str(source_file), str(target_file))
                logger.info(f"  Copied: {source_file.name} -> {target_file.name}")

        return True

    except Exception as e:
        logger.error(f"Failed to standardize Atropos tissues: {e}")
        return False


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

    # Extract bias correction parameters from config
    bias_config = config.get('anatomical', {}).get('bias_correction', {})

    # Set parameters from config
    n4.inputs.dimension = 3
    n4.inputs.n_iterations = bias_config.get('n_iterations', [50, 50, 30, 20])
    n4.inputs.shrink_factor = bias_config.get('shrink_factor', 3)
    n4.inputs.convergence_threshold = bias_config.get('convergence_threshold', 0.001)
    n4.inputs.bspline_fitting_distance = bias_config.get('bspline_fitting_distance', 300)

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

    # Extract Atropos parameters from config
    atropos_config = config.get('anatomical', {}).get('atropos', {})

    # Set parameters from config
    atropos.inputs.dimension = 3
    atropos.inputs.number_of_tissue_classes = atropos_config.get('number_of_tissue_classes', 3)
    atropos.inputs.initialization = atropos_config.get('initialization', 'KMeans')
    atropos.inputs.likelihood_model = 'Gaussian'  # Standard choice
    atropos.inputs.mrf_smoothing_factor = atropos_config.get('mrf_smoothing_factor', 0.1)
    atropos.inputs.mrf_radius = atropos_config.get('mrf_radius', [1, 1, 1])
    atropos.inputs.convergence_threshold = atropos_config.get('convergence_threshold', 0.001)
    atropos.inputs.n_iterations = atropos_config.get('n_iterations', 5)
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


def create_ants_registration_node(
    config: Dict[str, Any],
    name: str = 'ants_reg'
) -> Node:
    """
    Create ANTs registration node using antsRegistration (SyN).

    Performs robust nonlinear registration to MNI152 template using
    Symmetric Normalization (SyN), which is often more robust than FNIRT.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for ANTs registration

    Examples
    --------
    >>> ants_reg = create_ants_registration_node(config)
    >>> ants_reg.inputs.moving_image = "T1w_brain.nii.gz"
    >>> ants_reg.inputs.fixed_image = "/path/to/MNI152_T1_2mm.nii.gz"
    """
    # Get ANTs configuration
    ants_config = get_node_config('ants_registration', config)

    ants_reg = Node(
        ants.Registration(),
        name=name
    )

    # ANTs SyN registration parameters
    # These are optimized for T1w -> MNI152 registration
    ants_reg.inputs.dimension = 3
    ants_reg.inputs.float = True  # Use float precision for speed
    ants_reg.inputs.output_transform_prefix = 'ants_'
    ants_reg.inputs.output_warped_image = True
    ants_reg.inputs.interpolation = 'Linear'
    ants_reg.inputs.use_histogram_matching = True
    ants_reg.inputs.winsorize_lower_quantile = 0.005
    ants_reg.inputs.winsorize_upper_quantile = 0.995

    # Multi-stage registration: Rigid → Affine → SyN
    ants_reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    ants_reg.inputs.transform_parameters = [
        (0.1,),  # Rigid: gradient step
        (0.1,),  # Affine: gradient step
        (0.1, 3.0, 0.0)  # SyN: gradient step, flow sigma, total sigma
    ]

    # Similarity metrics for each stage
    ants_reg.inputs.metric = ['MI', 'MI', 'CC']  # Mutual Info for linear, CC for nonlinear
    ants_reg.inputs.metric_weight = [1.0, 1.0, 1.0]
    ants_reg.inputs.radius_or_number_of_bins = [32, 32, 4]  # MI bins for linear, CC radius for SyN

    # Convergence criteria for each stage
    ants_reg.inputs.number_of_iterations = [
        [1000, 500, 250, 100],  # Rigid
        [1000, 500, 250, 100],  # Affine
        [100, 70, 50, 20]  # SyN
    ]
    ants_reg.inputs.convergence_threshold = [1e-6, 1e-6, 1e-6]
    ants_reg.inputs.convergence_window_size = [10, 10, 10]

    # Smoothing sigmas (in voxels) for multi-resolution
    ants_reg.inputs.smoothing_sigmas = [
        [3, 2, 1, 0],  # Rigid
        [3, 2, 1, 0],  # Affine
        [3, 2, 1, 0]  # SyN
    ]

    # Shrink factors for multi-resolution
    ants_reg.inputs.shrink_factors = [
        [8, 4, 2, 1],  # Rigid
        [8, 4, 2, 1],  # Affine
        [8, 4, 2, 1]  # SyN
    ]

    # Output settings
    ants_reg.inputs.write_composite_transform = True

    return ants_reg


def create_t1w_preprocessing_workflow(
    config: Dict[str, Any],
    subject: str,
    t1w_file: Path,
    output_dir: Path,
    work_dir: Path,
    session: Optional[str] = None,
    name: str = 't1w_preprocess'
) -> Workflow:
    """
    Create complete T1w anatomical preprocessing workflow.

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
    >>> from neurovrai.config import load_config
    >>> config = load_config("study.yaml")
    >>> wf = create_t1w_preprocessing_workflow(
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

    # 5. Registration to MNI152 (ANTs or FSL)
    registration_method = config.get('anatomical', {}).get('registration_method', 'ants')

    if registration_method == 'ants':
        # ANTs SyN registration (single node: rigid + affine + SyN)
        registration = create_ants_registration_node(config)
        registration.inputs.fixed_image = str(mni_brain_template)
    else:
        # FSL registration (two nodes: FLIRT + FNIRT)
        linear_reg = create_linear_registration_node(config)
        linear_reg.inputs.reference = str(mni_brain_template)

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

    # Registration connections (method-specific)
    if registration_method == 'ants':
        # ANTs registration (use bias-corrected brain as moving image)
        wf.connect(bias_correct, 'output_image', registration, 'moving_image')
        wf.connect(registration, 'composite_transform', outputnode, 'mni_warp')
        wf.connect(registration, 'warped_image', outputnode, 'mni_warped')
        # ANTs doesn't have separate affine matrix, composite_transform includes all stages
        # For compatibility, we'll save the composite transform as both outputs
        wf.connect(registration, 'composite_transform', outputnode, 'mni_affine_mat')
    else:
        # FSL registration (FLIRT + FNIRT)
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


def run_t1w_preprocessing(
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
    Run T1w anatomical preprocessing and save transformations.

    This is the main entry point for T1w preprocessing.
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
    >>> from neurovrai.config import load_config
    >>> config = load_config("study.yaml")
    >>> results = run_t1w_preprocessing(
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

    # Get execution configuration and set Nipype config BEFORE creating workflow
    # This ensures hash_method is set when nodes are instantiated
    exec_config = get_execution_config(config)

    # Create workflow
    wf = create_t1w_preprocessing_workflow(
        config=config,
        subject=subject,
        t1w_file=t1w_file,
        output_dir=derivatives_dir,  # Pass derivatives directory directly
        work_dir=work_dir,
        session=session
    )

    # Run workflow
    wf.run(**exec_config)

    # Standardize Atropos tissue names (if using Atropos)
    # This creates csf.nii.gz, gm.nii.gz, wm.nii.gz symlinks based on intensity
    seg_dir = derivatives_dir / 'segmentation'
    if seg_dir.exists() and list(seg_dir.glob('POSTERIOR_*.nii.gz')):
        logger = logging.getLogger(__name__)
        logger.info("Standardizing Atropos tissue names...")
        standardize_atropos_tissues(derivatives_dir, use_symlinks=True)

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
        # Tissue segmentation: Prefer standardized names (csf.nii.gz, gm.nii.gz, wm.nii.gz)
        # Fall back to POSTERIOR_*.nii.gz if standardized names don't exist
        # NOTE: Atropos with K-means produces POSTERIOR files in RANDOM order - don't assume ordering!
        'csf_prob': (list(derivatives_dir.glob('**/segmentation/csf.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/csf.nii.gz')) else
                     list(derivatives_dir.glob('**/segmentation/POSTERIOR_01.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/POSTERIOR_01.nii.gz')) else None),
        'gm_prob': (list(derivatives_dir.glob('**/segmentation/gm.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/gm.nii.gz')) else
                    list(derivatives_dir.glob('**/segmentation/POSTERIOR_02.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/POSTERIOR_02.nii.gz')) else None),
        'wm_prob': (list(derivatives_dir.glob('**/segmentation/wm.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/wm.nii.gz')) else
                    list(derivatives_dir.glob('**/segmentation/POSTERIOR_03.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/POSTERIOR_03.nii.gz')) else None),
        'tissue_class_map': list(derivatives_dir.glob('**/segmentation/*labeled.nii.gz'))[0] if list(derivatives_dir.glob('**/segmentation/*labeled.nii.gz')) else None,
        # For FSL: look for .mat and warp.nii.gz
        # For ANTs: look for .h5 composite transform
        'mni_affine': (list(derivatives_dir.glob('**/transforms/*.mat'))[0] if list(derivatives_dir.glob('**/transforms/*.mat')) else
                       list(derivatives_dir.glob('**/transforms/*.h5'))[0] if list(derivatives_dir.glob('**/transforms/*.h5')) else None),
        'mni_warp': (list(derivatives_dir.glob('**/transforms/*warp.nii.gz'))[0] if list(derivatives_dir.glob('**/transforms/*warp.nii.gz')) else
                     list(derivatives_dir.glob('**/transforms/*.h5'))[0] if list(derivatives_dir.glob('**/transforms/*.h5')) else None),
        'mni_warped': list(derivatives_dir.glob('**/mni_space/*.nii.gz'))[0] if list(derivatives_dir.glob('**/mni_space/*.nii.gz')) else None
    }

    # Save transformations to standardized location: {study_root}/transforms/{subject}/
    if save_transforms and outputs['mni_affine'] and outputs['mni_warp']:
        from neurovrai.utils.transforms import save_transform

        # Get registration method
        registration_method = config.get('anatomical', {}).get('registration_method', 'ants')

        # Save transform to standardized location
        transforms_dir = study_root / 'transforms' / subject
        transforms_dir.mkdir(parents=True, exist_ok=True)

        if registration_method == 'ants':
            # ANTs: save composite transform as t1w-mni-composite.h5
            save_transform(
                outputs['mni_warp'],  # For ANTs, this is the composite .h5
                study_root, subject, 't1w', 'mni', 'composite'
            )
            logging.info(f"Saved ANTs T1w->MNI composite to: {transforms_dir / 't1w-mni-composite.h5'}")
        else:
            # FSL: save separate warp and affine
            save_transform(outputs['mni_warp'], study_root, subject, 't1w', 'mni', 'warp')
            save_transform(outputs['mni_affine'], study_root, subject, 't1w', 'mni', 'affine')
            logging.info(f"Saved FSL T1w->MNI transforms to: {transforms_dir}")

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
                from neurovrai.preprocess.qc.anat.skull_strip_qc import SkullStripQualityControl

                logging.info("Running Skull Strip QC...")
                skull_qc = SkullStripQualityControl(
                    subject=subject,
                    anat_dir=derivatives_dir,
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
                from neurovrai.preprocess.qc.anat.segmentation_qc import SegmentationQualityControl

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
                from neurovrai.preprocess.qc.anat.registration_qc import RegistrationQualityControl

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


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================
# These aliases are provided for backward compatibility with code that imports
# from the old anat_preprocess module name.

run_anat_preprocessing = run_t1w_preprocessing
"""Alias for run_t1w_preprocessing (backward compatibility)."""

create_anat_preprocessing_workflow = create_t1w_preprocessing_workflow
"""Alias for create_t1w_preprocessing_workflow (backward compatibility)."""

# Also provide run_anatomical_preprocessing as an alias
run_anatomical_preprocessing = run_t1w_preprocessing
"""Alias for run_t1w_preprocessing (backward compatibility)."""
