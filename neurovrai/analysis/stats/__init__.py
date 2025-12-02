"""
Statistical Analysis Module

Provides tools for group-level statistical analysis including:
- Design matrix generation from participant data
- Design matrix matching with available MRI data
- FSL randomise wrapper (nonparametric permutation testing)
- Nilearn GLM wrapper (parametric analysis with Python)
- Cluster extraction and reporting
"""

from neurovrai.analysis.stats.design_matrix import (
    generate_design_files,
    create_design_matrix,
    create_contrast_matrix,
    load_participants
)

from neurovrai.analysis.stats.design_matrix_matching import (
    filter_design_matrix_by_subjects,
    discover_subjects_for_analysis,
    validate_design_data_match,
    create_matched_design_for_analysis
)

from neurovrai.analysis.stats.randomise_wrapper import (
    run_randomise,
    summarize_results,
    get_significant_voxels
)

from neurovrai.analysis.stats.nilearn_glm import (
    run_second_level_glm,
    apply_multiple_comparison_correction,
    summarize_glm_results
)

__all__ = [
    # Design matrix functions
    'generate_design_files',
    'create_design_matrix',
    'create_contrast_matrix',
    'load_participants',
    # Design matrix matching functions
    'filter_design_matrix_by_subjects',
    'discover_subjects_for_analysis',
    'validate_design_data_match',
    'create_matched_design_for_analysis',
    # Randomise functions
    'run_randomise',
    'summarize_results',
    'get_significant_voxels',
    # Nilearn GLM functions
    'run_second_level_glm',
    'apply_multiple_comparison_correction',
    'summarize_glm_results',
]
