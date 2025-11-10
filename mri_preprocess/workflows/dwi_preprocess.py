#!/usr/bin/env python3
"""
Diffusion MRI preprocessing workflow using Nipype and FSL.

Workflow steps:
1. Eddy current and motion correction (eddy)
2. Brain extraction using T1w mask
3. DTI tensor fitting (dtifit)
4. BEDPOSTX probabilistic modeling (optional)
5. Registration to MNI space using T1w’MNI transforms from TransformRegistry

Key feature: Reuses T1w’MNI transformations computed during anatomical
preprocessing, avoiding duplicate computation.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

from nipype import Workflow, Node, MapNode
from nipype.interfaces import fsl, utility as niu
from nipype.interfaces.io import DataSink

from mri_preprocess.utils.workflow import (
    setup_logging,
    get_node_config,
    get_execution_config,
    validate_inputs
)
from mri_preprocess.utils.transforms import TransformRegistry, create_transform_registry
from mri_preprocess.utils.bids import get_derivatives_dir


def create_eddy_node(
    config: Dict[str, Any],
    name: str = 'eddy'
) -> Node:
    """
    Create eddy current and motion correction node.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for eddy correction

    Examples
    --------
    >>> eddy = create_eddy_node(config)
    >>> eddy.inputs.in_file = "dwi.nii.gz"
    >>> eddy.inputs.in_bval = "dwi.bval"
    >>> eddy.inputs.in_bvec = "dwi.bvec"
    """
    eddy_config = get_node_config('eddy', config)

    eddy = Node(
        fsl.Eddy(),
        name=name
    )

    # Set parameters from config
    if 'acqp_file' in eddy_config:
        eddy.inputs.in_acqp = eddy_config['acqp_file']
    if 'index_file' in eddy_config:
        eddy.inputs.in_index = eddy_config['index_file']

    eddy.inputs.method = eddy_config.get('method', 'jac')
    eddy.inputs.repol = eddy_config.get('repol', True)
    eddy.inputs.output_type = 'NIFTI_GZ'

    return eddy


def create_dtifit_node(
    name: str = 'dtifit'
) -> Node:
    """
    Create DTI tensor fitting node.

    Computes diffusion tensor and derived metrics (FA, MD, etc.).

    Parameters
    ----------
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for dtifit

    Examples
    --------
    >>> dtifit = create_dtifit_node()
    >>> dtifit.inputs.dwi = "dwi_eddy.nii.gz"
    >>> dtifit.inputs.mask = "dwi_mask.nii.gz"
    """
    dtifit = Node(
        fsl.DTIFit(),
        name=name
    )

    dtifit.inputs.output_type = 'NIFTI_GZ'

    return dtifit


def create_bedpostx_node(
    config: Dict[str, Any],
    name: str = 'bedpostx'
) -> Node:
    """
    Create BEDPOSTX node for probabilistic tractography modeling.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for BEDPOSTX

    Examples
    --------
    >>> bedpostx = create_bedpostx_node(config)
    >>> bedpostx.inputs.dwi = "dwi_eddy.nii.gz"
    >>> bedpostx.inputs.mask = "dwi_mask.nii.gz"
    """
    bedpostx_config = get_node_config('bedpostx', config)

    bedpostx = Node(
        fsl.BEDPOSTX(),
        name=name
    )

    # Set parameters
    bedpostx.inputs.n_fibres = bedpostx_config.get('n_fibres', 2)
    bedpostx.inputs.n_jumps = bedpostx_config.get('n_jumps', 1250)
    bedpostx.inputs.burn_in = bedpostx_config.get('burn_in', 1000)

    return bedpostx


def create_apply_warp_node(
    name: str = 'apply_warp'
) -> Node:
    """
    Create node to apply nonlinear warp to diffusion maps.

    Uses FNIRT warp from TransformRegistry to transform FA/MD maps to MNI space.

    Parameters
    ----------
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for applying warp

    Examples
    --------
    >>> applywarp = create_apply_warp_node()
    >>> applywarp.inputs.in_file = "FA.nii.gz"
    >>> applywarp.inputs.field_file = "t1w_to_mni_warp.nii.gz"
    >>> applywarp.inputs.ref_file = "MNI152_T1_2mm.nii.gz"
    """
    applywarp = Node(
        fsl.ApplyWarp(),
        name=name
    )

    applywarp.inputs.interp = 'trilinear'
    applywarp.inputs.output_type = 'NIFTI_GZ'

    return applywarp


def create_dwi_preprocessing_workflow(
    config: Dict[str, Any],
    subject: str,
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_dir: Path,
    work_dir: Path,
    t1w_brain_mask: Optional[Path] = None,
    session: Optional[str] = None,
    run_bedpostx: bool = False,
    name: str = 'dwi_preprocess'
) -> Workflow:
    """
    Create complete diffusion preprocessing workflow.

    This workflow:
    1. Runs eddy for motion/distortion correction
    2. Extracts brain using T1w mask
    3. Fits diffusion tensor (DTIFit)
    4. Optionally runs BEDPOSTX for tractography
    5. Transforms outputs to MNI space using T1w’MNI transforms from TransformRegistry

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    dwi_file : Path
        Input DWI NIfTI file
    bval_file : Path
        b-value file
    bvec_file : Path
        b-vector file
    output_dir : Path
        Output directory for derivatives
    work_dir : Path
        Working directory
    t1w_brain_mask : Path, optional
        T1w brain mask (for registration)
    session : str, optional
        Session identifier
    run_bedpostx : bool
        Run BEDPOSTX (default: False, computationally expensive)
    name : str
        Workflow name

    Returns
    -------
    Workflow
        Nipype workflow ready to run

    Examples
    --------
    >>> wf = create_dwi_preprocessing_workflow(
    ...     config=config,
    ...     subject="sub-001",
    ...     dwi_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.nii.gz"),
    ...     bval_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.bval"),
    ...     bvec_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.bvec"),
    ...     output_dir=Path("/data/derivatives/mri-preprocess"),
    ...     work_dir=Path("/tmp/work")
    ... )
    >>> wf.run()
    """
    # Validate inputs
    validate_inputs(dwi_file, bval_file, bvec_file)

    # Setup logging
    log_dir = Path(config['paths']['logs'])
    logger = setup_logging(log_dir, subject, name)
    logger.info(f"Creating diffusion preprocessing workflow for {subject}")

    # Create workflow
    wf = Workflow(name=name)
    wf.base_dir = str(work_dir)

    # === Create nodes ===

    # Input node
    inputnode = Node(
        niu.IdentityInterface(fields=['dwi', 'bval', 'bvec', 't1w_mask']),
        name='inputnode'
    )
    inputnode.inputs.dwi = str(dwi_file)
    inputnode.inputs.bval = str(bval_file)
    inputnode.inputs.bvec = str(bvec_file)
    if t1w_brain_mask:
        inputnode.inputs.t1w_mask = str(t1w_brain_mask)

    # 1. Eddy correction
    eddy = create_eddy_node(config)

    # 2. Brain extraction
    # Generate mask from b0
    extract_b0 = Node(
        fsl.ExtractROI(t_min=0, t_size=1),
        name='extract_b0'
    )

    bet_dwi = Node(
        fsl.BET(frac=0.3, mask=True, robust=True),
        name='bet_dwi'
    )

    # 3. DTI fitting
    dtifit = create_dtifit_node()

    # 4. BEDPOSTX (optional)
    if run_bedpostx:
        bedpostx = create_bedpostx_node(config)

    # Output node
    outputnode = Node(
        niu.IdentityInterface(
            fields=[
                'eddy_corrected',
                'rotated_bvec',
                'dwi_mask',
                'fa',
                'md',
                'l1',
                'l2',
                'l3',
                'tensor',
                'fa_mni',
                'md_mni'
            ]
        ),
        name='outputnode'
    )

    # DataSink
    datasink = Node(
        DataSink(),
        name='datasink'
    )
    datasink.inputs.base_directory = str(
        get_derivatives_dir(output_dir, 'mri-preprocess', subject, session, create=True)
    )
    datasink.inputs.container = 'dwi'

    # === Connect workflow ===

    # Eddy correction
    wf.connect([
        (inputnode, eddy, [
            ('dwi', 'in_file'),
            ('bval', 'in_bval'),
            ('bvec', 'in_bvec')
        ]),
        (eddy, outputnode, [
            ('out_corrected', 'eddy_corrected'),
            ('out_rotated_bvecs', 'rotated_bvec')
        ])
    ])

    # Brain extraction
    wf.connect([
        (eddy, extract_b0, [('out_corrected', 'in_file')]),
        (extract_b0, bet_dwi, [('roi_file', 'in_file')]),
        (bet_dwi, outputnode, [('mask_file', 'dwi_mask')])
    ])

    # DTI fitting
    wf.connect([
        (eddy, dtifit, [
            ('out_corrected', 'dwi'),
            ('out_rotated_bvecs', 'bvecs')
        ]),
        (inputnode, dtifit, [('bval', 'bvals')]),
        (bet_dwi, dtifit, [('mask_file', 'mask')]),
        (dtifit, outputnode, [
            ('FA', 'fa'),
            ('MD', 'md'),
            ('L1', 'l1'),
            ('L2', 'l2'),
            ('L3', 'l3'),
            ('tensor', 'tensor')
        ])
    ])

    # BEDPOSTX (if requested)
    if run_bedpostx:
        wf.connect([
            (eddy, bedpostx, [
                ('out_corrected', 'dwi'),
                ('out_rotated_bvecs', 'bvecs')
            ]),
            (inputnode, bedpostx, [('bval', 'bvals')]),
            (bet_dwi, bedpostx, [('mask_file', 'mask')])
        ])

    # Save outputs
    wf.connect([
        (outputnode, datasink, [
            ('eddy_corrected', 'eddy_corrected'),
            ('rotated_bvec', 'rotated_bvec'),
            ('dwi_mask', 'mask'),
            ('fa', 'dti.@fa'),
            ('md', 'dti.@md'),
            ('l1', 'dti.@l1'),
            ('l2', 'dti.@l2'),
            ('l3', 'dti.@l3'),
            ('tensor', 'dti.@tensor')
        ])
    ])

    logger.info("Diffusion preprocessing workflow created")

    return wf


def run_dwi_preprocessing(
    config: Dict[str, Any],
    subject: str,
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_dir: Path,
    work_dir: Path,
    session: Optional[str] = None,
    run_bedpostx: bool = False,
    warp_to_mni: bool = True
) -> Dict[str, Path]:
    """
    Run diffusion preprocessing with TransformRegistry integration.

    This is the main entry point for diffusion preprocessing.
    It creates and runs the workflow, then optionally warps outputs
    to MNI space using transforms from the TransformRegistry.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    dwi_file : Path
        Input DWI file
    bval_file : Path
        b-value file
    bvec_file : Path
        b-vector file
    output_dir : Path
        Output directory
    work_dir : Path
        Working directory
    session : str, optional
        Session identifier
    run_bedpostx : bool
        Run BEDPOSTX (default: False)
    warp_to_mni : bool
        Warp outputs to MNI using TransformRegistry (default: True)

    Returns
    -------
    dict
        Dictionary with output file paths

    Examples
    --------
    >>> results = run_dwi_preprocessing(
    ...     config=config,
    ...     subject="sub-001",
    ...     dwi_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.nii.gz"),
    ...     bval_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.bval"),
    ...     bvec_file=Path("/data/rawdata/sub-001/dwi/sub-001_dwi.bvec"),
    ...     output_dir=Path("/data/derivatives/mri-preprocess"),
    ...     work_dir=Path("/tmp/work")
    ... )
    >>> print(results['fa'])
    """
    logger = logging.getLogger(__name__)

    # Create workflow
    wf = create_dwi_preprocessing_workflow(
        config=config,
        subject=subject,
        dwi_file=dwi_file,
        bval_file=bval_file,
        bvec_file=bvec_file,
        output_dir=output_dir,
        work_dir=work_dir,
        session=session,
        run_bedpostx=run_bedpostx
    )

    # Get execution configuration
    exec_config = get_execution_config(config)

    # Run workflow
    wf.run(**exec_config)

    # Collect outputs
    derivatives_dir = get_derivatives_dir(
        output_dir, 'mri-preprocess', subject, session
    )
    dwi_dir = derivatives_dir / 'dwi'
    dti_dir = dwi_dir / 'dti'

    outputs = {
        'eddy_corrected': list(dwi_dir.glob('*eddy_corrected*.nii.gz'))[0] if dwi_dir.exists() else None,
        'fa': list(dti_dir.glob('*FA.nii.gz'))[0] if dti_dir.exists() else None,
        'md': list(dti_dir.glob('*MD.nii.gz'))[0] if dti_dir.exists() else None,
        'mask': list(dwi_dir.glob('*mask.nii.gz'))[0] if dwi_dir.exists() else None,
    }

    # === CRITICAL: Use TransformRegistry for MNI warping ===
    # This demonstrates the compute-once-reuse-everywhere pattern
    if warp_to_mni and outputs['fa'] and outputs['md']:
        logger.info("Warping DTI metrics to MNI space using TransformRegistry...")

        # Load transforms from registry
        registry = create_transform_registry(config, subject, session)

        # Check if T1w’MNI transform exists
        if registry.has_transform('T1w', 'MNI152', 'nonlinear'):
            logger.info(" Found T1w’MNI152 transform in registry - reusing!")

            warp_file, affine_file = registry.get_nonlinear_transform('T1w', 'MNI152')

            from mri_preprocess.utils.workflow import get_reference_template
            mni_ref = get_reference_template('mni152_t1_2mm', config)

            # Warp FA to MNI
            import subprocess
            fa_mni = dti_dir / (outputs['fa'].stem.replace('.nii', '_mni.nii') + '.gz')
            cmd = [
                'applywarp',
                '--in=' + str(outputs['fa']),
                '--ref=' + str(mni_ref),
                '--warp=' + str(warp_file),
                '--out=' + str(fa_mni)
            ]
            subprocess.run(cmd, check=True)
            outputs['fa_mni'] = fa_mni
            logger.info(f"Created FA in MNI space: {fa_mni}")

            # Warp MD to MNI
            md_mni = dti_dir / (outputs['md'].stem.replace('.nii', '_mni.nii') + '.gz')
            cmd = [
                'applywarp',
                '--in=' + str(outputs['md']),
                '--ref=' + str(mni_ref),
                '--warp=' + str(warp_file),
                '--out=' + str(md_mni)
            ]
            subprocess.run(cmd, check=True)
            outputs['md_mni'] = md_mni
            logger.info(f"Created MD in MNI space: {md_mni}")

        else:
            logger.warning("  T1w’MNI152 transform not found in registry")
            logger.warning("  Run anatomical preprocessing first to compute transforms")

    return outputs
