#!/usr/bin/env python3
"""
Quality Control (QC) Module for MRI Preprocessing.

This module provides tools for generating quality control metrics,
visualizations, and reports for preprocessed MRI data.

Submodules:
    dwi: DWI-specific QC (TOPUP, eddy, DTI metrics)
    anat: Anatomical QC (segmentation, registration)
    rest: Resting-state fMRI QC (motion, carpet plots, ICA-AROMA)
"""

__version__ = '0.1.0'
