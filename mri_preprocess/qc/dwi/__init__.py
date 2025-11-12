#!/usr/bin/env python3
"""
DWI Quality Control Module.

Provides QC metrics, visualizations, and reports for DWI preprocessing:
- TOPUP distortion correction QC
- Motion parameters QC
- DTI metrics validation
- SNR analysis
- Visual inspection tools
"""

from .topup_qc import TOPUPQualityControl
from .motion_qc import MotionQualityControl
from .dti_qc import DTIQualityControl
# TODO: Implement remaining module incrementally
# from .dwi_qc_report import DWIQCReport

__all__ = [
    'TOPUPQualityControl',
    'MotionQualityControl',
    'DTIQualityControl',
    # 'DWIQCReport',
]
