"""
Resting-state fMRI analysis modules

Modules:
- reho: Regional Homogeneity (ReHo) analysis
- falff: Amplitude of Low-Frequency Fluctuations (ALFF/fALFF) analysis
- resting_workflow: Integrated workflow for resting-state analysis (ReHo + fALFF)
- melodic: Group ICA analysis using FSL MELODIC
- dual_regression: Dual regression for subject-specific ICA maps from group ICA
"""

from .reho import compute_reho_map, compute_reho_zscore
from .falff import compute_falff_map, compute_falff_zscore
from .resting_workflow import run_resting_state_analysis
from .melodic import run_melodic_group_ica, prepare_subjects_for_melodic
from .dual_regression import run_dual_regression, validate_dual_regression_inputs

__all__ = [
    'compute_reho_map',
    'compute_reho_zscore',
    'compute_falff_map',
    'compute_falff_zscore',
    'run_resting_state_analysis',
    'run_melodic_group_ica',
    'prepare_subjects_for_melodic',
    'run_dual_regression',
    'validate_dual_regression_inputs',
]
