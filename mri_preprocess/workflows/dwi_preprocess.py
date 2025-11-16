#!/usr/bin/env python3
"""
Diffusion MRI preprocessing workflow using Nipype and FSL.

Workflow steps:
1. TOPUP distortion correction using reverse phase-encoding acquisition (per shell)
2. Merge multiple DWI acquisitions (bval, bvec, nifti) after TOPUP correction
3. Eddy current and motion correction (eddy with TOPUP integration)
4. Brain extraction
5. DTI tensor fitting (dtifit)
6. BEDPOSTX probabilistic modeling (optional)
7. Probabilistic tractography (probtrackx2, optional)
8. Registration to MNI space using T1w->MNI transforms from TransformRegistry

Key features:
- TOPUP susceptibility distortion correction applied BEFORE merging shells
- Handles multiple DWI shells with automatic concatenation
- Reuses T1w->MNI transformations computed during anatomical preprocessing
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import numpy as np

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
from mri_preprocess.utils.topup_helper import create_topup_files_for_multishell
from mri_preprocess.utils.dwi_normalization import (
    normalize_dwi_to_fmrib58,
    apply_warp_to_metrics
)
from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models


def merge_dwi_files(
    dwi_files: List[Path],
    bval_files: List[Path],
    bvec_files: List[Path],
    output_dir: Path
) -> Tuple[Path, Path, Path]:
    """
    Merge multiple DWI acquisitions into single files.

    Concatenates bval, bvec, and NIfTI files from multiple shells/acquisitions.
    This should be called AFTER TOPUP correction has been applied to each shell.

    Parameters
    ----------
    dwi_files : list of Path
        DWI NIfTI files to merge
    bval_files : list of Path
        b-value files to merge
    bvec_files : list of Path
        b-vector files to merge
    output_dir : Path
        Output directory for merged files

    Returns
    -------
    tuple
        (merged_dwi, merged_bval, merged_bvec) paths

    Examples
    --------
    >>> merged_dwi, merged_bval, merged_bvec = merge_dwi_files(
    ...     [Path("dwi_b1000.nii.gz"), Path("dwi_b2000.nii.gz")],
    ...     [Path("dwi_b1000.bval"), Path("dwi_b2000.bval")],
    ...     [Path("dwi_b1000.bvec"), Path("dwi_b2000.bvec")],
    ...     Path("/output")
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge bval files
    bvals = []
    for bval_file in bval_files:
        bval = np.loadtxt(bval_file)
        bvals.append(bval)
    merged_bvals = np.concatenate(bvals)
    merged_bval_file = output_dir / 'dwi_merged.bval'
    np.savetxt(merged_bval_file, merged_bvals, delimiter='\t', newline='\t', fmt='%f')

    # Merge bvec files (paste side-by-side)
    merged_bvec_file = output_dir / 'dwi_merged.bvec'
    with open(merged_bvec_file, 'w') as f:
        subprocess.call(['paste'] + [str(f) for f in bvec_files], stdout=f)

    # Merge NIfTI files
    merge = fsl.Merge()
    merge.inputs.in_files = [str(f) for f in dwi_files]
    merge.inputs.dimension = 't'
    merge.inputs.output_type = 'NIFTI_GZ'
    merge.inputs.merged_file = str(output_dir / 'dwi_merged.nii.gz')
    merge.run()

    merged_dwi_file = output_dir / 'dwi_merged.nii.gz'

    return merged_dwi_file, merged_bval_file, merged_bvec_file


def create_topup_node(
    config: Dict[str, Any],
    name: str = 'topup'
) -> Node:
    """
    Create TOPUP node for susceptibility distortion correction.

    TOPUP estimates the susceptibility-induced off-resonance field using
    pairs of images acquired with opposite phase-encoding directions.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for TOPUP

    Examples
    --------
    >>> topup = create_topup_node(config)
    >>> topup.inputs.in_file = "b0_PA_AP.nii.gz"
    >>> topup.inputs.encoding_file = "acqparams.txt"
    """
    topup_config = get_node_config('topup', config)

    topup = Node(
        fsl.TOPUP(),
        name=name
    )

    # Set parameters from config
    if 'encoding_file' in topup_config:
        topup.inputs.encoding_file = topup_config['encoding_file']

    # Use default configuration file (b02b0.cnf) unless specified
    if 'config' in topup_config:
        topup.inputs.config = topup_config['config']

    topup.inputs.output_type = 'NIFTI_GZ'

    return topup


def create_applytopup_node(
    name: str = 'applytopup'
) -> Node:
    """
    Create ApplyTOPUP node to apply distortion correction.

    Applies the field estimated by TOPUP to correct images.

    Parameters
    ----------
    name : str
        Node name

    Returns
    -------
    Node
        Nipype Node for ApplyTOPUP

    Examples
    --------
    >>> applytopup = create_applytopup_node()
    >>> applytopup.inputs.in_files = ["dwi.nii.gz"]
    >>> applytopup.inputs.in_topup_fieldcoef = "topup_fieldcoef.nii.gz"
    >>> applytopup.inputs.in_topup_movpar = "topup_movpar.txt"
    """
    applytopup = Node(
        fsl.ApplyTOPUP(),
        name=name
    )

    applytopup.inputs.method = 'jac'
    applytopup.inputs.output_type = 'NIFTI_GZ'

    return applytopup


def create_eddy_node(
    config: Dict[str, Any],
    name: str = 'eddy',
    use_topup: bool = True
) -> Node:
    """
    Create eddy current and motion correction node.

    When use_topup=True, eddy will use TOPUP outputs for improved correction.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    name : str
        Node name
    use_topup : bool
        Use TOPUP outputs for correction (default: True)

    Returns
    -------
    Node
        Nipype Node for eddy correction

    Examples
    --------
    >>> eddy = create_eddy_node(config, use_topup=True)
    >>> eddy.inputs.in_file = "dwi.nii.gz"
    >>> eddy.inputs.in_bval = "dwi.bval"
    >>> eddy.inputs.in_bvec = "dwi.bvec"
    >>> eddy.inputs.in_topup_fieldcoef = "topup_fieldcoef.nii.gz"
    >>> eddy.inputs.in_topup_movpar = "topup_movpar.txt"
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

    # Use CUDA if available
    eddy.inputs.use_cuda = eddy_config.get('use_cuda', True)

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


def run_dwi_multishell_topup_preprocessing(
    config: Dict[str, Any],
    subject: str,
    dwi_files: List[Path],
    bval_files: List[Path],
    bvec_files: List[Path],
    rev_phase_files: Optional[List[Path]] = None,
    output_dir: Path = None,
    work_dir: Optional[Path] = None,
    session: Optional[str] = None,
    run_bedpostx: bool = False
) -> Dict[str, Path]:
    """
    Run complete multi-shell DWI preprocessing with TOPUP correction.

    This function implements the standard FSL multi-shell preprocessing pipeline:
    1. Merge all DWI shells (bval, bvec, nifti) BEFORE correction
    2. Merge all reverse phase-encoding images
    3. Extract b0 volumes from merged DWI and reverse PE
    4. Run TOPUP once on merged b0 volumes
    5. Run eddy with TOPUP outputs on merged DWI (homogeneous correction)
    6. Perform brain extraction, DTI fitting, and optional BEDPOSTX

    This approach ensures homogeneous correction across all shells and follows
    FSL best practices.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    dwi_files : list of Path
        DWI NIfTI files (one per shell, in order)
    bval_files : list of Path
        b-value files (corresponding to dwi_files)
    bvec_files : list of Path
        b-vector files (corresponding to dwi_files)
    rev_phase_files : list of Path, optional
        Reverse phase-encoding SE_EPI files (one per shell, in same order)
        If not provided or empty, TOPUP will be skipped and eddy will run without it
    output_dir : Path
        Study root directory (e.g., /mnt/bytopia/development/IRC805/)
        Derivatives will be saved to: {output_dir}/derivatives/dwi_topup/{subject}/
    work_dir : Path
        Working directory for temporary Nipype files
        Default: {output_dir}/work/{subject}/dwi_topup/
    session : str, optional
        Session identifier
    run_bedpostx : bool
        Run BEDPOSTX (computationally expensive)

    Returns
    -------
    dict
        Dictionary with output file paths

    Notes
    -----
    This function requires:
    - acqparams.txt: Phase encoding parameters for each acquisition
    - index.txt: Maps each volume to corresponding line in acqparams.txt

    Examples
    --------
    >>> results = run_dwi_multishell_topup_preprocessing(
    ...     config=config,
    ...     subject="sub-001",
    ...     dwi_files=[Path("b1000.nii.gz"), Path("b2000.nii.gz")],
    ...     bval_files=[Path("b1000.bval"), Path("b2000.bval")],
    ...     bvec_files=[Path("b1000.bvec"), Path("b2000.bvec")],
    ...     rev_phase_files=[Path("b1000_PA.nii.gz"), Path("b2000_PA.nii.gz")],
    ...     output_dir=Path("/data/derivatives"),
    ...     work_dir=Path("/tmp/work")
    ... )
    """
    logger = logging.getLogger(__name__)

    # Determine if TOPUP should be used
    topup_config = config.get('diffusion', {}).get('topup', {})
    topup_enabled = topup_config.get('enabled', 'auto')  # auto, true, false

    # Auto-detect: use TOPUP if reverse PE files are available
    has_reverse_pe = rev_phase_files is not None and len(rev_phase_files) > 0

    if topup_enabled == 'auto':
        use_topup = has_reverse_pe
    elif topup_enabled is True:
        use_topup = True
        if not has_reverse_pe:
            logger.warning("TOPUP enabled in config but no reverse PE files provided")
            logger.warning("Proceeding without TOPUP")
            use_topup = False
    else:  # topup_enabled is False
        use_topup = False
        if has_reverse_pe:
            logger.info("Reverse PE files available but TOPUP disabled in config")

    logger.info("="*70)
    logger.info(f"STARTING MULTI-SHELL DWI PREPROCESSING")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Number of shells: {len(dwi_files)}")
    logger.info(f"TOPUP distortion correction: {'ENABLED' if use_topup else 'DISABLED'}")
    if not use_topup and has_reverse_pe:
        logger.info("  (Reverse PE files available but TOPUP disabled in config)")
    elif not use_topup:
        logger.info("  (No reverse PE files available)")
    logger.info("")
    logger.info("Pipeline steps:")
    logger.info("  1. Merge DWI shells")
    if use_topup:
        logger.info("  2. Merge reverse phase-encoding images")
        logger.info("  3-5. Extract and merge b0 volumes")
        logger.info("  6. Run TOPUP distortion correction (~5-10 min)")
        logger.info("  7. Run eddy correction with TOPUP (~10-30 min)")
    else:
        logger.info("  2. Extract b0 volumes")
        logger.info("  3. Run eddy correction (~10-30 min)")
    logger.info("  " + ("8" if use_topup else "4") + ". Brain extraction and DTI fitting (~5 min)")
    if run_bedpostx:
        logger.info("  " + ("9" if use_topup else "5") + ". Run BEDPOSTX (~1-4 hours)")
    logger.info("")
    logger.info("Estimated total time: " + ("2-5 hours" if run_bedpostx else "20-45 minutes"))
    logger.info("="*70)
    logger.info("")

    # Setup directory structure
    # output_dir is the derivatives base (e.g., /mnt/bytopia/IRC805/derivatives)
    # Use standardized hierarchy: {outdir}/{subject}/{modality}/
    outdir = Path(output_dir)

    # Create simple, standardized hierarchy
    derivatives_dir = outdir / subject / 'dwi'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    # Derive study root from output_dir (derivatives directory)
    # output_dir is derivatives, so study_root is one level up
    study_root = outdir.parent

    # Work directory for preprocessing
    # TOPUP files go in dwi_preprocess/, Nipype workflow creates subdirectory
    if work_dir is None:
        work_dir = study_root / 'work' / subject / 'dwi_preprocess'
    else:
        work_dir = Path(work_dir) / 'dwi_preprocess'  # Append workflow name to provided work_dir

    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Study root: {study_root}")
    logger.info(f"Derivatives output: {derivatives_dir}")
    logger.info(f"Working directory: {work_dir}")
    logger.info("")

    # Step 1: Merge DWI shells FIRST (before any correction)
    logger.info("Step 1: Merging DWI shells")
    merged_dwi, merged_bval, merged_bvec = merge_dwi_files(
        dwi_files,
        bval_files,
        bvec_files,
        work_dir
    )
    logger.info(f"  Merged DWI: {merged_dwi}")

    # Step 2: Extract b0 from merged DWI (always needed for brain mask)
    logger.info("Step 2: Extracting b0 from merged DWI")
    b0_dwi = work_dir / 'b0_dwi.nii.gz'
    extract_b0_dwi = fsl.ExtractROI(
        in_file=str(merged_dwi),
        t_min=0,
        t_size=1,
        roi_file=str(b0_dwi)
    )
    extract_b0_dwi.run()
    logger.info(f"  b0 extracted: {b0_dwi}")

    # Initialize TOPUP outputs
    topup_fieldcoef = None
    topup_movpar = None
    b0_merged = None

    # Conditional TOPUP steps
    if use_topup:
        # Step 3: Merge reverse phase-encoding images
        logger.info("Step 3: Merging reverse phase-encoding images")
        merged_rev = work_dir / 'rev_phase_merged.nii.gz'
        merge_rev = fsl.Merge(
            in_files=[str(f) for f in rev_phase_files],
            dimension='t',
            merged_file=str(merged_rev)
        )
        merge_rev.run()
        logger.info(f"  Merged reverse PE: {merged_rev}")

        # Step 4: Extract b0 from merged reverse PE
        logger.info("Step 4: Extracting b0 from merged reverse PE")
        b0_rev = work_dir / 'b0_rev.nii.gz'
        extract_b0_rev = fsl.ExtractROI(
            in_file=str(merged_rev),
            t_min=0,
            t_size=1,
            roi_file=str(b0_rev)
        )
        extract_b0_rev.run()

        # Step 5: Merge b0 volumes for TOPUP (DWI b0 first, then reverse PE b0)
        logger.info("Step 5: Merging b0 volumes for TOPUP")
        b0_merged = work_dir / 'b0_merged.nii.gz'
        merge_b0 = fsl.Merge(
            in_files=[str(b0_dwi), str(b0_rev)],
            dimension='t',
            merged_file=str(b0_merged)
        )
        merge_b0.run()
        logger.info(f"  Merged b0 for TOPUP: {b0_merged}")

        # Step 6: Run TOPUP
        logger.info("Step 6: Running TOPUP")
        logger.info("  This may take 5-10 minutes...")
        topup_config = get_node_config('topup', config)
        encoding_file = topup_config.get('encoding_file')

        # Auto-generate acqparams.txt if not provided
        if not encoding_file or not Path(encoding_file).exists():
            logger.info("  Auto-generating TOPUP acquisition parameters...")

        # Get phase encoding direction and readout time from config
        pe_direction = topup_config.get('pe_direction', 'AP')  # Default: AP
        readout_time = topup_config.get('readout_time', 0.05)  # Default: 0.05s

        # Create param files directory
        param_dir = work_dir / 'topup_params'
        param_dir.mkdir(parents=True, exist_ok=True)

        # Generate acqparams.txt and index.txt
        acqparams_file, index_file = create_topup_files_for_multishell(
            dwi_files=dwi_files,
            pe_direction=pe_direction,
            readout_time=readout_time,
            output_dir=param_dir
        )

        encoding_file = acqparams_file
        logger.info(f"  Generated: {encoding_file}")
        logger.info(f"  Generated: {index_file}")
        logger.info(f"  PE direction: {pe_direction}, Readout time: {readout_time}s")

        # Update config to include auto-generated files for eddy
        # Note: get_node_config('eddy', config) looks for config['diffusion']['eddy']
        if 'diffusion' not in config:
            config['diffusion'] = {}
        if 'eddy' not in config['diffusion']:
            config['diffusion']['eddy'] = {}
        config['diffusion']['eddy']['acqp_file'] = str(acqparams_file)
        config['diffusion']['eddy']['index_file'] = str(index_file)

        topup_out = work_dir / 'topup_results'

        # Run TOPUP with progress logging
        topup_cmd = [
            'topup',
            '--imain=' + str(b0_merged),
            '--datain=' + str(encoding_file),
            '--out=' + str(topup_out),
            '--fout=' + str(topup_out) + '_field',
            '--iout=' + str(topup_out) + '_corrected',
            '--verbose'
        ]

        logger.info("  Running: " + ' '.join(topup_cmd))
        logger.info("  Progress (SSD = sum of squared differences, lower is better):")

        import sys
        process = subprocess.Popen(
            topup_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output and show progress
        iteration_count = 0
        for line in process.stdout:
            line = line.strip()
            if line:
                # Show SSD lines (these indicate progress)
                if 'SSD' in line:
                    iteration_count += 1
                    logger.info(f"    Iteration {iteration_count}: {line}")
                # Show important messages
                elif any(keyword in line.lower() for keyword in ['error', 'warning', 'finished', 'done']):
                    logger.info(f"    {line}")

        process.wait()

        if process.returncode != 0:
            logger.error(f"TOPUP failed with return code {process.returncode}")
            raise RuntimeError(f"TOPUP failed")

        logger.info(f"  TOPUP completed successfully!")
        logger.info(f"  Field coefficient: {topup_out}_fieldcoef.nii.gz")
        logger.info(f"  Corrected image: {topup_out}_corrected.nii.gz")

        # Set TOPUP outputs for eddy
        topup_fieldcoef = Path(str(topup_out) + '_fieldcoef.nii.gz')
        topup_movpar = Path(str(topup_out) + '_movpar.txt')
    else:
        # No TOPUP - eddy will run without distortion correction
        logger.info("Skipping TOPUP - running eddy without distortion correction")
        topup_fieldcoef = None
        topup_movpar = None

    # Step N: Run eddy correction
    step_num = 7 if use_topup else 3
    logger.info(f"Step {step_num}: Running eddy correction{' with TOPUP integration' if use_topup else ''}")
    # Pass parent directory so Nipype workflow files go directly in dwi_preprocess/
    # (Nipype would add workflow name as subdirectory, creating dwi_preprocess/dwi_eddy_dtifit/)
    wf = create_dwi_preprocessing_workflow(
        config=config,
        subject=subject,
        dwi_file=merged_dwi,
        bval_file=merged_bval,
        bvec_file=merged_bvec,
        output_dir=derivatives_dir,  # Pass derivatives directory directly
        work_dir=work_dir.parent,  # Pass {study_root}/work/{subject}/ so workflow creates dwi_eddy_dtifit/ subdirectory
        topup_fieldcoef=topup_fieldcoef,
        topup_movpar=topup_movpar,
        session=session,
        run_bedpostx=run_bedpostx,
        name='dwi_preprocess'  # Changed from 'dwi_eddy_dtifit' to match expected directory
    )

    # Get execution configuration
    exec_config = get_execution_config(config)

    # Run workflow
    logger.info("  Running Nipype workflow (eddy, brain extraction, DTI fitting)...")
    wf.run(**exec_config)

    # Collect outputs
    # derivatives_dir is already set at the top of the function
    # Files are saved directly to derivatives_dir by DataSink

    # Find output files (DataSink puts them in subdirectories)
    eddy_files = list(derivatives_dir.glob('**/eddy_corrected*.nii.gz')) if derivatives_dir.exists() else []
    fa_files = list(derivatives_dir.glob('**/*FA.nii.gz')) if derivatives_dir.exists() else []
    md_files = list(derivatives_dir.glob('**/*MD.nii.gz')) if derivatives_dir.exists() else []
    mask_files = list(derivatives_dir.glob('**/*mask*.nii.gz')) if derivatives_dir.exists() else []
    bvec_files = list(derivatives_dir.glob('**/*rotated_bvecs*')) if derivatives_dir.exists() else []

    outputs = {
        'merged_dwi': merged_dwi,
        'merged_bval': merged_bval,
        'merged_bvec': merged_bvec,
        'b0_merged': b0_merged,
        'topup_fieldcoef': topup_fieldcoef,  # None if TOPUP was skipped
        'topup_movpar': topup_movpar,  # None if TOPUP was skipped
        'eddy_corrected': eddy_files[0] if eddy_files else None,
        'fa': fa_files[0] if fa_files else None,
        'md': md_files[0] if md_files else None,
        'mask': mask_files[0] if mask_files else None,
        'rotated_bvec': bvec_files[0] if bvec_files else None,
    }

    # Step 7.5: Advanced diffusion models (DKI, NODDI) if multi-shell
    adv_config = config.get('diffusion', {}).get('advanced_models', {})
    adv_enabled = adv_config.get('enabled', 'auto')

    # Auto-detect multi-shell: check if ≥2 non-zero b-values
    is_multishell = False
    if merged_bval and merged_bval.exists():
        bvals = np.loadtxt(merged_bval)
        unique_bvals = np.unique(bvals[bvals > 50])  # Ignore b=0 and noise
        is_multishell = len(unique_bvals) >= 2
        logger.info(f"  Detected {len(unique_bvals)} unique b-values: {unique_bvals}")
        logger.info(f"  Multi-shell data: {is_multishell}")

    # Determine if we should run advanced models
    run_advanced = False
    if adv_enabled == 'auto':
        run_advanced = is_multishell
    elif adv_enabled is True:
        run_advanced = True

    if run_advanced and outputs['eddy_corrected'] and outputs['mask']:
        logger.info("")
        logger.info("="*70)
        logger.info("Step 7.5: Advanced Diffusion Models (DKI/NODDI)")
        logger.info("="*70)
        logger.info("")

        try:
            # Run advanced diffusion models
            adv_results = run_advanced_diffusion_models(
                dwi_file=outputs['eddy_corrected'],
                bval_file=merged_bval,
                bvec_file=outputs['rotated_bvec'],
                mask_file=outputs['mask'],
                output_dir=derivatives_dir,  # Will create dki/ and noddi/ subdirs
                fit_dki=adv_config.get('fit_dki', True),
                fit_noddi=adv_config.get('fit_noddi', True),
                fit_sandi=adv_config.get('fit_sandi', False),
                fit_activeax=adv_config.get('fit_activeax', False),
                use_amico=adv_config.get('use_amico', True)
            )

            # Add advanced model outputs to results
            outputs['advanced_models'] = adv_results
            logger.info("  ✓ Advanced diffusion models completed")

        except Exception as e:
            logger.warning(f"  Advanced diffusion models failed: {e}")
            logger.warning("  Continuing with standard DTI metrics only")
    elif adv_enabled is not False:
        if not is_multishell:
            logger.info("  Skipping advanced models: single-shell data (requires ≥2 b-values)")
        elif not outputs['eddy_corrected']:
            logger.warning("  Skipping advanced models: eddy correction output not found")

    # Step 8: Spatial normalization to FMRIB58_FA template
    if outputs['fa'] and outputs['fa'].exists():
        logger.info("")
        logger.info("="*70)
        logger.info("Step 8: Normalizing DWI metrics to FMRIB58_FA template")
        logger.info("="*70)
        logger.info("")

        # Run normalization
        norm_results = normalize_dwi_to_fmrib58(
            fa_file=outputs['fa'],
            output_dir=derivatives_dir,
            fmrib58_template=None  # Uses $FSLDIR default
        )

        # Add transform outputs
        outputs['fa_to_fmrib58_affine'] = norm_results['affine_mat']
        outputs['fa_to_fmrib58_warp'] = norm_results['forward_warp']
        outputs['fmrib58_to_fa_warp'] = norm_results['inverse_warp']
        outputs['fa_normalized'] = norm_results['fa_normalized']

        # Collect all DWI metrics for normalization
        dti_dir = derivatives_dir / 'dti'
        metric_files = []

        # Standard DTI metrics
        if dti_dir.exists():
            for metric in ['MD', 'AD', 'RD', 'L1', 'L2', 'L3']:
                metric_file = list(dti_dir.glob(f'*{metric}.nii.gz'))
                if metric_file:
                    metric_files.append(metric_file[0])

        # Check for DKI metrics (if advanced_diffusion was run)
        dki_dir = derivatives_dir / 'dki'
        if dki_dir.exists():
            logger.info("  Found DKI metrics, including in normalization...")
            for metric in ['mean_kurtosis', 'axial_kurtosis', 'radial_kurtosis', 'kurtosis_fa']:
                metric_file = list(dki_dir.glob(f'*{metric}.nii.gz'))
                if metric_file:
                    metric_files.append(metric_file[0])

        # Check for NODDI metrics (if advanced_diffusion was run)
        noddi_dir = derivatives_dir / 'noddi'
        if noddi_dir.exists():
            logger.info("  Found NODDI metrics, including in normalization...")
            for metric in ['ficvf', 'odi', 'fiso']:
                metric_file = list(noddi_dir.glob(f'*{metric}.nii.gz'))
                if metric_file:
                    metric_files.append(metric_file[0])

        # Apply warp to all metrics
        if metric_files:
            logger.info(f"  Normalizing {len(metric_files)} DWI metric maps...")
            normalized_files = apply_warp_to_metrics(
                metric_files=metric_files,
                forward_warp=norm_results['forward_warp'],
                fmrib58_template=None,
                output_dir=derivatives_dir,
                interpolation='spline'
            )
            outputs['normalized_metrics'] = normalized_files
            logger.info(f"  Successfully normalized {len(normalized_files)} metrics")

        logger.info("")
        logger.info("Normalization complete!")
        logger.info(f"  Forward warp (for group analyses): {outputs['fa_to_fmrib58_warp']}")
        logger.info(f"  Inverse warp (for tractography ROIs): {outputs['fmrib58_to_fa_warp']}")
        logger.info(f"  Normalized metrics saved to: {derivatives_dir / 'normalized'}")
        logger.info("")

    # Step 9: Quality Control
    run_qc = config.get('diffusion', {}).get('run_qc', True)
    if run_qc:
        logger.info("")
        logger.info("="*70)
        logger.info("Step 9: Quality Control")
        logger.info("="*70)
        logger.info("")

        # Setup QC directory (study-level)
        qc_dir = study_root / 'qc' / subject / 'dwi'
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Import QC modules
        from mri_preprocess.qc.dwi import TOPUPQualityControl, MotionQualityControl, DTIQualityControl

        # 1. TOPUP QC (if TOPUP was used)
        if use_topup and outputs.get('topup_fieldcoef'):
            logger.info("Running TOPUP QC...")
            topup_qc_dir = qc_dir / 'topup'
            topup_qc_dir.mkdir(parents=True, exist_ok=True)

            topup_qc = TOPUPQualityControl(
                subject=subject,
                work_dir=work_dir / 'dwi_preprocess',
                qc_dir=topup_qc_dir
            )

            try:
                topup_results = topup_qc.run_qc(
                    topup_log=None,  # Will auto-detect
                    fieldcoef_file=outputs.get('topup_fieldcoef')
                )
                outputs['topup_qc'] = topup_results
                logger.info(f"  ✓ TOPUP QC complete: {topup_qc_dir}")
            except Exception as e:
                logger.warning(f"  TOPUP QC failed: {e}")

        # 2. Motion QC (eddy parameters)
        if outputs.get('eddy_corrected'):
            logger.info("Running Motion QC...")
            motion_qc_dir = qc_dir / 'motion'
            motion_qc_dir.mkdir(parents=True, exist_ok=True)

            motion_qc = MotionQualityControl(
                subject=subject,
                work_dir=work_dir / 'dwi_preprocess',
                qc_dir=motion_qc_dir
            )

            try:
                motion_results = motion_qc.run_qc(
                    eddy_params_file=None,  # Will auto-detect
                    fd_threshold=config.get('diffusion', {}).get('fd_threshold', 1.0)
                )
                outputs['motion_qc'] = motion_results
                logger.info(f"  ✓ Motion QC complete: {motion_qc_dir}")
            except Exception as e:
                logger.warning(f"  Motion QC failed: {e}")

        # 3. DTI QC (metrics validation)
        if outputs.get('fa') and outputs.get('md'):
            logger.info("Running DTI QC...")
            dti_qc_dir = qc_dir / 'dti'
            dti_qc_dir.mkdir(parents=True, exist_ok=True)

            dti_qc = DTIQualityControl(
                subject=subject,
                metrics_dir=derivatives_dir / 'dti',
                qc_dir=dti_qc_dir
            )

            try:
                dti_results = dti_qc.run_qc(
                    metrics=['FA', 'MD', 'AD', 'RD'],
                    mask_file=outputs.get('mask')
                )
                outputs['dti_qc'] = dti_results
                logger.info(f"  ✓ DTI QC complete: {dti_qc_dir}")
            except Exception as e:
                logger.warning(f"  DTI QC failed: {e}")

        logger.info("")
        logger.info(f"QC reports saved to: {qc_dir}")
        logger.info("")

    logger.info("")
    logger.info("="*70)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info("")
    logger.info("Key outputs:")
    logger.info(f"  Eddy-corrected DWI: {outputs['eddy_corrected']}")
    logger.info(f"  FA map: {outputs['fa']}")
    logger.info(f"  MD map: {outputs['md']}")
    logger.info(f"  Brain mask: {outputs['mask']}")
    logger.info("")
    logger.info("TOPUP outputs:")
    logger.info(f"  Field coefficient: {outputs['topup_fieldcoef']}")
    logger.info(f"  Movement parameters: {outputs['topup_movpar']}")
    logger.info("")
    logger.info("Intermediate files:")
    logger.info(f"  Merged DWI: {outputs['merged_dwi']}")
    logger.info(f"  Merged bval: {outputs['merged_bval']}")
    logger.info(f"  Merged bvec: {outputs['merged_bvec']}")
    logger.info("")
    logger.info(f"Derivatives directory: {derivatives_dir}")
    logger.info("="*70)

    return outputs


def create_dwi_preprocessing_workflow(
    config: Dict[str, Any],
    subject: str,
    dwi_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_dir: Path,
    work_dir: Path,
    t1w_brain_mask: Optional[Path] = None,
    topup_fieldcoef: Optional[Path] = None,
    topup_movpar: Optional[Path] = None,
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
    5. Transforms outputs to MNI space using T1w�MNI transforms from TransformRegistry

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
    topup_fieldcoef : Path, optional
        TOPUP field coefficient file from previous TOPUP run
    topup_movpar : Path, optional
        TOPUP movement parameter file from previous TOPUP run
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

    # 1. Brain extraction (BEFORE eddy - eddy needs a mask!)
    # Generate mask from b0 of input DWI
    extract_b0 = Node(
        fsl.ExtractROI(t_min=0, t_size=1),
        name='extract_b0'
    )

    # Get BET parameters from config
    bet_config = config.get('diffusion', {}).get('bet', {})
    bet_frac = bet_config.get('frac', 0.3)  # Default: 0.3 for DWI

    bet_dwi = Node(
        fsl.BET(frac=bet_frac, mask=True, robust=True),
        name='bet_dwi'
    )

    # 2. Eddy correction (uses mask from above)
    eddy = create_eddy_node(config)

    # Set TOPUP outputs if provided
    if topup_fieldcoef and topup_movpar:
        eddy.inputs.in_topup_fieldcoef = str(topup_fieldcoef)
        eddy.inputs.in_topup_movpar = str(topup_movpar)

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
    # output_dir is now the full derivatives directory path (e.g., /study/derivatives/dwi_topup/subject/)
    datasink = Node(
        DataSink(),
        name='datasink'
    )
    datasink.inputs.base_directory = str(output_dir)
    datasink.inputs.container = ''  # No container, already in subject directory

    # === Connect workflow ===

    # Brain extraction (from input DWI, before eddy)
    wf.connect([
        (inputnode, extract_b0, [('dwi', 'in_file')]),
        (extract_b0, bet_dwi, [('roi_file', 'in_file')]),
        (bet_dwi, outputnode, [('mask_file', 'dwi_mask')])
    ])

    # Eddy correction (using mask from above)
    wf.connect([
        (inputnode, eddy, [
            ('dwi', 'in_file'),
            ('bval', 'in_bval'),
            ('bvec', 'in_bvec')
        ]),
        (bet_dwi, eddy, [('mask_file', 'in_mask')]),  # Connect mask to eddy
        (eddy, outputnode, [
            ('out_corrected', 'eddy_corrected'),
            ('out_rotated_bvecs', 'rotated_bvec')
        ])
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

        # Check if T1w�MNI transform exists
        if registry.has_transform('T1w', 'MNI152', 'nonlinear'):
            logger.info(" Found T1w�MNI152 transform in registry - reusing!")

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
            logger.warning("� T1w�MNI152 transform not found in registry")
            logger.warning("  Run anatomical preprocessing first to compute transforms")

    return outputs
