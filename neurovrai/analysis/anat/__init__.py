"""
Anatomical Analysis Module

VBM (Voxel-Based Morphometry), WMH (White Matter Hyperintensity),
T1-T2-ratio, and structural analysis tools.
"""

from .vbm_workflow import prepare_vbm_data, run_vbm_analysis
from .wmh_detection import detect_wmh, compute_lesion_metrics, get_lesion_size_distribution
from .wmh_workflow import (
    run_wmh_analysis_single,
    run_wmh_analysis_batch,
    generate_group_summary as wmh_generate_group_summary,
    discover_subjects as wmh_discover_subjects
)
from .wmh_reporting import generate_wmh_html_report
from .t1t2ratio_workflow import (
    prepare_t1t2ratio_single,
    prepare_t1t2ratio_batch,
    run_t1t2ratio_analysis,
    generate_group_summary as t1t2ratio_generate_group_summary,
    discover_subjects as t1t2ratio_discover_subjects
)
from .t1t2ratio_reporting import generate_t1t2ratio_html_report

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
    'wmh_generate_group_summary',
    'wmh_discover_subjects',
    # WMH Reporting
    'generate_wmh_html_report',
    # T1-T2-Ratio Workflow
    'prepare_t1t2ratio_single',
    'prepare_t1t2ratio_batch',
    'run_t1t2ratio_analysis',
    't1t2ratio_generate_group_summary',
    't1t2ratio_discover_subjects',
    # T1-T2-Ratio Reporting
    'generate_t1t2ratio_html_report'
]
