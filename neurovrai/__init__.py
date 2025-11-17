"""
neurovrai: Comprehensive MRI Preprocessing and Analysis Pipeline

A unified package for neuroimaging data processing, from raw DICOM to
connectivity matrices and group statistics.

Modules
-------
preprocess : Subject-level preprocessing for all MRI modalities
    - Anatomical (T1w/T2w)
    - Diffusion (DWI/DTI/DKI/NODDI)
    - Functional (rs-fMRI with TEDANA/ICA-AROMA)
    - ASL (perfusion with CBF quantification)

analysis : Group-level statistical analyses (planned)
    - VBM, TBSS, MELODIC, ReHo, fALFF

connectome : Connectivity and network analysis (planned)
    - Structural/functional connectivity matrices
    - Graph theory metrics
    - Network visualization

Usage
-----
>>> from neurovrai.config import load_config
>>> from neurovrai.preprocess.workflows import anat_preprocess
>>> config = load_config('config.yaml')
"""

__version__ = "2.0.0-alpha"
__author__ = "Alexandre Edmond"
__all__ = ['preprocess', 'analysis', 'connectome', 'config', 'utils']

# Make config loader easily accessible
from neurovrai.config import load_config

__all__.append('load_config')
