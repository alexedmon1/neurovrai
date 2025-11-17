#!/usr/bin/env python3
"""
Workflow helper utilities for Nipype.

Provides common node configurations and workflow setup functions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os


def setup_logging(
    log_dir: Path,
    subject: str,
    workflow_name: str,
    level: str = 'INFO'
) -> logging.Logger:
    """
    Set up logging for a preprocessing workflow.

    Parameters
    ----------
    log_dir : Path
        Directory for log files
    subject : str
        Subject ID
    workflow_name : str
        Name of workflow
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns
    -------
    logging.Logger
        Configured logger

    Examples
    --------
    >>> logger = setup_logging(Path("/data/logs"), "sub-001", "anat-prep")
    >>> logger.info("Starting preprocessing")
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger_name = f"{subject}_{workflow_name}"
    logger = logging.getLogger(logger_name)

    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    log_file = log_dir / f"{subject}_{workflow_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_fsl_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract FSL configuration from config dict.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        FSL configuration with keys: fsldir, output_type

    Examples
    --------
    >>> fsl_config = get_fsl_config(config)
    >>> print(fsl_config['fsldir'])
    '/usr/local/fsl'
    """
    fsl_config = {}

    if 'fsl' in config:
        fsl_config['fsldir'] = config['fsl'].get('fsldir', os.getenv('FSLDIR'))
        fsl_config['output_type'] = config['fsl'].get('output_type', 'NIFTI_GZ')
    else:
        fsl_config['fsldir'] = os.getenv('FSLDIR', '/usr/local/fsl')
        fsl_config['output_type'] = 'NIFTI_GZ'

    return fsl_config


def set_fsl_output_type(output_type: str = 'NIFTI_GZ'):
    """
    Set FSL output type environment variable.

    Parameters
    ----------
    output_type : str
        FSL output type (NIFTI, NIFTI_GZ, NIFTI_PAIR, etc.)

    Examples
    --------
    >>> set_fsl_output_type('NIFTI_GZ')
    """
    os.environ['FSLOUTPUTTYPE'] = output_type


def get_node_config(node_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get configuration for a specific node type.

    Parameters
    ----------
    node_type : str
        Type of node (e.g., 'bet', 'fast', 'flirt')
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Node-specific configuration

    Examples
    --------
    >>> bet_config = get_node_config('bet', config)
    >>> print(bet_config['frac'])
    0.5
    """
    # Map node types to config sections
    node_map = {
        # Anatomical
        'bet': 'anatomical.skull_strip',
        'fast': 'anatomical.bias_correction',
        'flirt': 'anatomical.registration',
        'fnirt': 'anatomical.registration',

        # Diffusion
        'topup': 'diffusion.topup',
        'eddy': 'diffusion.eddy',
        'dtifit': 'diffusion.dtifit',
        'bedpostx': 'diffusion.bedpostx',

        # Functional
        'mcflirt': 'functional.motion_correction',
        'susan': 'functional.smoothing',
    }

    if node_type not in node_map:
        return {}

    # Navigate to config section
    config_path = node_map[node_type].split('.')
    node_config = config
    for key in config_path:
        if key in node_config:
            node_config = node_config[key]
        else:
            return {}

    return node_config if isinstance(node_config, dict) else {}


def get_reference_template(template_name: str, config: Dict[str, Any]) -> Path:
    """
    Get path to reference template.

    Parameters
    ----------
    template_name : str
        Name of template (e.g., 'mni152_t1_2mm')
    config : dict
        Configuration dictionary

    Returns
    -------
    Path
        Path to template file

    Examples
    --------
    >>> mni = get_reference_template('mni152_t1_2mm', config)
    >>> print(mni)
    Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz')
    """
    if 'templates' in config and template_name in config['templates']:
        template_path = config['templates'][template_name]
    else:
        # Default FSL templates
        fsldir = get_fsl_config(config)['fsldir']
        template_map = {
            'mni152_t1_2mm': f'{fsldir}/data/standard/MNI152_T1_2mm_brain.nii.gz',
            'mni152_t1_1mm': f'{fsldir}/data/standard/MNI152_T1_1mm_brain.nii.gz',
            'mni152_mask_2mm': f'{fsldir}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz',
            'mni152_mask_1mm': f'{fsldir}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz',
        }
        template_path = template_map.get(template_name, '')

    return Path(template_path) if template_path else None


def get_execution_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get workflow execution configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Execution configuration for Nipype

    Examples
    --------
    >>> exec_config = get_execution_config(config)
    >>> print(exec_config['plugin'])
    'MultiProc'
    >>> print(exec_config['plugin_args'])
    {'n_procs': 4}
    """
    from nipype import config as nipype_config

    if 'execution' not in config:
        # Defaults
        nipype_config.set('execution', 'hash_method', 'content')
        return {
            'plugin': 'MultiProc',
            'plugin_args': {'n_procs': 2}
        }

    exec_section = config['execution']

    plugin = exec_section.get('plugin', 'MultiProc')
    n_procs = exec_section.get('n_procs', 2)

    # Set Nipype execution options for caching
    hash_method = exec_section.get('hash_method', 'content')
    stop_on_first_crash = exec_section.get('stop_on_first_crash', False)
    keep_inputs = exec_section.get('keep_inputs', False)
    remove_unnecessary_outputs = exec_section.get('remove_unnecessary_outputs', True)

    nipype_config.set('execution', 'hash_method', hash_method)
    nipype_config.set('execution', 'stop_on_first_crash', str(stop_on_first_crash).lower())
    nipype_config.set('execution', 'keep_inputs', str(keep_inputs).lower())
    nipype_config.set('execution', 'remove_unnecessary_outputs', str(remove_unnecessary_outputs).lower())

    return {
        'plugin': plugin,
        'plugin_args': {'n_procs': n_procs}
    }


def get_workflow_graph_path(workflow_dir: Path, workflow_name: str) -> Path:
    """
    Get path for workflow graph visualization.

    Parameters
    ----------
    workflow_dir : Path
        Workflow working directory
    workflow_name : str
        Name of workflow

    Returns
    -------
    Path
        Path for graph PNG file

    Examples
    --------
    >>> graph_path = get_workflow_graph_path(Path("/tmp/work"), "anat-prep")
    >>> print(graph_path)
    Path('/tmp/work/workflow_graphs/anat-prep_graph.png')
    """
    graph_dir = workflow_dir / 'workflow_graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_file = graph_dir / f'{workflow_name}_graph.png'
    return graph_file


def check_dependencies():
    """
    Check that required external dependencies are available.

    Raises
    ------
    EnvironmentError
        If required dependencies are missing

    Examples
    --------
    >>> check_dependencies()  # Raises if FSL not found
    """
    # Check FSL
    fsldir = os.getenv('FSLDIR')
    if not fsldir:
        raise EnvironmentError(
            "FSLDIR environment variable not set. "
            "Please install FSL and set FSLDIR."
        )

    fsl_bin = Path(fsldir) / 'bin'
    if not fsl_bin.exists():
        raise EnvironmentError(
            f"FSL bin directory not found: {fsl_bin}. "
            f"Please check FSL installation."
        )

    # Check for key FSL tools
    required_tools = ['bet', 'fast', 'flirt', 'fnirt', 'eddy', 'dtifit']
    missing_tools = []

    for tool in required_tools:
        tool_path = fsl_bin / tool
        if not tool_path.exists():
            missing_tools.append(tool)

    if missing_tools:
        print(f"Warning: Some FSL tools not found: {missing_tools}")
        print("Some workflows may not work correctly.")


def create_datasink_substitutions(subject: str) -> list:
    """
    Create filename substitutions for Nipype DataSink.

    Cleans up Nipype's default naming conventions.

    Parameters
    ----------
    subject : str
        Subject ID

    Returns
    -------
    list
        List of (pattern, replacement) tuples

    Examples
    --------
    >>> subs = create_datasink_substitutions("sub-001")
    """
    substitutions = [
        # Remove leading underscores
        ('/_', '/'),
        # Remove trailing underscores
        ('_/', '/'),
        # Clean up node names
        ('_reorient', ''),
        ('_bet', ''),
        ('_fast', ''),
        ('_flirt', ''),
        ('_fnirt', ''),
    ]

    return substitutions


def validate_inputs(*file_paths: Path) -> bool:
    """
    Validate that input files exist.

    Parameters
    ----------
    *file_paths : Path
        Paths to check

    Returns
    -------
    bool
        True if all files exist

    Raises
    ------
    FileNotFoundError
        If any file doesn't exist

    Examples
    --------
    >>> validate_inputs(
    ...     Path("/data/sub-001_T1w.nii.gz"),
    ...     Path("/data/sub-001_T2w.nii.gz")
    ... )
    True
    """
    for file_path in file_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

    return True
