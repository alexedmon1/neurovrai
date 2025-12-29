"""
Backward compatibility stub for anat_preprocess module.

This module has been renamed to t1w_preprocess.py.
All imports are re-exported from the new module for backward compatibility.

Deprecation Warning:
    This module name is deprecated. Please update your imports to use:

    from neurovrai.preprocess.workflows.t1w_preprocess import (
        run_t1w_preprocessing,
        create_t1w_preprocessing_workflow,
    )

    The old function names (run_anat_preprocessing, etc.) are still available
    as aliases in the new module.
"""

import warnings

# Issue a deprecation warning when this module is imported
warnings.warn(
    "The 'anat_preprocess' module has been renamed to 't1w_preprocess'. "
    "Please update your imports to use 'from neurovrai.preprocess.workflows.t1w_preprocess import ...'",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new module
from neurovrai.preprocess.workflows.t1w_preprocess import (
    # New function names
    run_t1w_preprocessing,
    create_t1w_preprocessing_workflow,

    # Backward compatibility aliases
    run_anat_preprocessing,
    create_anat_preprocessing_workflow,
    run_anatomical_preprocessing,

    # Helper functions
    standardize_atropos_tissues,
    create_reorient_node,
    create_skull_strip_node,
    create_bias_correction_node,
    create_segmentation_node,
    create_linear_registration_node,
    create_nonlinear_registration_node,
    create_ants_registration_node,
)

__all__ = [
    'run_t1w_preprocessing',
    'create_t1w_preprocessing_workflow',
    'run_anat_preprocessing',
    'create_anat_preprocessing_workflow',
    'run_anatomical_preprocessing',
    'standardize_atropos_tissues',
    'create_reorient_node',
    'create_skull_strip_node',
    'create_bias_correction_node',
    'create_segmentation_node',
    'create_linear_registration_node',
    'create_nonlinear_registration_node',
    'create_ants_registration_node',
]
