"""
Statistical Analysis Module

Provides tools for group-level statistical analysis including:
- Design matrix generation from participant data
- FSL randomise wrapper (nonparametric permutation testing)
- FSL GLM wrapper (parametric analysis)
- Cluster extraction and reporting
"""

from neurovrai.analysis.stats.design_matrix import (
    generate_design_files,
    create_design_matrix,
    create_contrast_matrix,
    load_participants
)

from neurovrai.analysis.stats.randomise_wrapper import (
    run_randomise,
    summarize_results,
    get_significant_voxels
)

from neurovrai.analysis.stats.glm_wrapper import (
    run_fsl_glm,
    threshold_zstat,
    summarize_glm_results
)

__all__ = [
    # Design matrix functions
    'generate_design_files',
    'create_design_matrix',
    'create_contrast_matrix',
    'load_participants',
    # Randomise functions
    'run_randomise',
    'summarize_results',
    'get_significant_voxels',
    # GLM functions
    'run_fsl_glm',
    'threshold_zstat',
    'summarize_glm_results',
]
