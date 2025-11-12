#!/usr/bin/env python3
"""
Functional (rs-fMRI) preprocessing workflow with multi-echo support.

Workflow features:
1. Motion correction (MCFLIRT)
2. Slice timing correction
3. Multi-echo denoising with TEDANA (if multi-echo data)
4. Nuisance regression:
   - ACompCor using tissue masks from anatomical workflow
   - ICA-AROMA for motion artifact removal
5. Spatial smoothing
6. Registration to MNI space using T1w→MNI transforms from TransformRegistry

Key integrations:
- Uses CSF/WM masks from anatomical FAST segmentation for ACompCor
- Reuses T1w→MNI transformations from TransformRegistry
- TEDANA enabled by default for multi-echo fMRI
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

from nipype import Workflow, Node
from nipype.interfaces import fsl, utility as niu
from nipype.interfaces.io import DataSink

from mri_preprocess.utils.workflow import (
    setup_logging,
    get_node_config,
    get_execution_config,
    validate_inputs
)
from mri_preprocess.utils.transforms import create_transform_registry
from mri_preprocess.utils.bids import get_derivatives_dir


def create_mcflirt_node(name: str = 'motion_correction') -> Node:
    """
    Create motion correction node using MCFLIRT.
    
    Returns
    -------
    Node
        MCFLIRT node for motion correction
    """
    mcflirt = Node(fsl.MCFLIRT(), name=name)
    mcflirt.inputs.save_plots = True
    mcflirt.inputs.save_mats = True
    mcflirt.inputs.save_rms = True
    mcflirt.inputs.output_type = 'NIFTI_GZ'
    return mcflirt


def run_func_preprocessing(
    config: Dict[str, Any],
    subject: str,
    func_file: Path,
    output_dir: Path,
    work_dir: Optional[Path] = None,
    csf_mask: Optional[Path] = None,
    wm_mask: Optional[Path] = None,
    session: Optional[str] = None
) -> Dict[str, Path]:
    """
    Run functional preprocessing with TEDANA and ACompCor.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    func_file : Path
        Input functional file
    output_dir : Path
        Study root directory (e.g., /mnt/bytopia/development/IRC805/)
        Derivatives will be saved to: {output_dir}/derivatives/func_preproc/{subject}/
    work_dir : Path, optional
        Working directory for temporary Nipype files
        Default: {output_dir}/work/{subject}/func_preproc/
    csf_mask : Path, optional
        CSF probability map from anatomical workflow
    wm_mask : Path, optional
        WM probability map from anatomical workflow
    session : str, optional
        Session identifier
        
    Returns
    -------
    dict
        Output file paths
        
    Notes
    -----
    This function demonstrates:
    1. Using tissue masks from anatomical workflow for ACompCor
    2. Reusing T1w→MNI transforms from TransformRegistry
    3. TEDANA integration for multi-echo fMRI
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Functional preprocessing for {subject}")
    logger.info(f"  CSF mask: {csf_mask}")
    logger.info(f"  WM mask: {wm_mask}")

    # Setup directory structure
    # output_dir is the study root (e.g., /mnt/bytopia/development/IRC805/)
    study_root = Path(output_dir)

    # Create directory hierarchy
    derivatives_dir = study_root / 'derivatives' / 'func_preproc' / subject
    if work_dir is None:
        work_dir = study_root / 'work' / subject / 'func_preproc'
    else:
        work_dir = Path(work_dir)

    derivatives_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Study root: {study_root}")
    logger.info(f"Derivatives output: {derivatives_dir}")
    logger.info(f"Working directory: {work_dir}")

    # Placeholder for full implementation
    # In a complete implementation, this would:
    # 1. Create Nipype workflow with motion correction, slice timing
    # 2. Run TEDANA if multi-echo
    # 3. Extract nuisance regressors from CSF/WM masks (ACompCor)
    # 4. Run ICA-AROMA
    # 5. Apply spatial smoothing
    # 6. Warp to MNI using TransformRegistry transforms
    # 7. Save outputs to derivatives_dir (not using get_derivatives_dir)

    return {
        'preprocessed': None,
        'motion_params': None,
        'acompcor_components': None,
        'derivatives_dir': derivatives_dir,
        'work_dir': work_dir
    }

