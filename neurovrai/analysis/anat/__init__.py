"""
Anatomical Analysis Module

VBM (Voxel-Based Morphometry), WMH (White Matter Hyperintensity),
and structural analysis tools.
"""

from .vbm_workflow import prepare_vbm_data, run_vbm_analysis
from .wmh_detection import detect_wmh, compute_lesion_metrics, get_lesion_size_distribution
from .wmh_workflow import (
    run_wmh_analysis_single,
    run_wmh_analysis_batch,
    generate_group_summary,
    discover_subjects
)
from .wmh_reporting import generate_wmh_html_report

__all__ = [
    # VBM
    'prepare_vbm_data',
    'run_vbm_analysis',
    # WMH Detection
    'detect_wmh',
    'compute_lesion_metrics',
    'get_lesion_size_distribution',
    # WMH Workflow
    'run_wmh_analysis_single',
    'run_wmh_analysis_batch',
    'generate_group_summary',
    'discover_subjects',
    # WMH Reporting
    'generate_wmh_html_report'
]
