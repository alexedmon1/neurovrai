#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 8 12:27:00 2022

@author: edm9fd
"""

import os

def create_key(template, outtype=('nii.gz'), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return (template, outtype, annotation_classes)


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    allowed template fields - follow python string module:
    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    rest = create_key('func/sub-{subject}_task-rest_bold')
    rest_correction = create_key('func/sub-{subject}_task-rest_correction_bold')
    old_rest = create_key('func/sub-{subject}_task-rest_bold_single_run{item:02d}')


    info = {rest:[], rest_correction:[],old_rest:[]}

    for idx, s in enumerate(seqinfo):
        if (s.dim3 == 56700) and ('RESTING ME3 MB3 SENSE3' in s.series_description):
            info[rest] = [s.series_id]
        if (s.dim3 == 630) and ('fMRI_CORRECTION_MB3_ME3_SENSE3_3mm_TR1104_TE15_DTE20' in s.series_description):
            info[rest_correction] = [s.series_id]
        if (s.dim3 == 5775) and ('RESTING STATE' in s.series_description):
            info[old_rest].append({'item': s.series_id})

            
            
    return info