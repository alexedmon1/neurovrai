"""
Shared utilities for neurovrai package.

These utilities are used across multiple modules (preprocess, analysis, connectome).
Module-specific utilities should remain in their respective utils/ directories.

Available utilities
-------------------
workflow : Common Nipype workflow helpers
transforms : Spatial transformation registry
"""

from neurovrai.utils.workflow import (
    setup_logging,
    get_node_config,
    get_execution_config,
    validate_inputs
)

from neurovrai.utils.transforms import create_transform_registry

__all__ = [
    'setup_logging',
    'get_node_config',
    'get_execution_config',
    'validate_inputs',
    'create_transform_registry'
]
