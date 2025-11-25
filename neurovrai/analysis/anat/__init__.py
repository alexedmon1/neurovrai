"""
Anatomical Analysis Module

VBM (Voxel-Based Morphometry) and structural analysis tools.
"""

from .vbm_workflow import prepare_vbm_data, run_vbm_analysis

__all__ = [
    'prepare_vbm_data',
    'run_vbm_analysis'
]
