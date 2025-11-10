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
    t1w = create_key('anat/sub-{subject}_t1w')
    t2w = create_key('anat/sub-{subject}_t2w')
    rest = create_key('func/sub-{subject}_task-rest_bold')
    rest_correction = create_key('func/sub-{subject}_task-rest_correction_bold')
    old_rest = create_key('func/sub-{subject}_task-rest_bold_single_run{item:02d}')
    dwi_posterior = create_key('struc/sub-{subject}_dwi_posterior-{item:02d}')
    dwi1shell = create_key('dwi/sub-{subject}_dwi_b3000')
    dwi2shell = create_key('dwi/sub-{subject}_dwi_b1000_b2000')
    asl_source = create_key('perf/sub-{subject}_source_asl')
    old_dwi = create_key('dwi/sub-{subject}_dwi_32dir')
    asl_pre = create_key('perf/sub-{subject}_pre_asl')
    asl = create_key('perf/sub-{subject}_asl')

    info = {t1w:[], t2w:[], rest:[], rest_correction:[], dwi1shell:[], dwi2shell:[], dwi_posterior:[], 
            asl:[], asl_source:[], asl_pre:[], old_rest:[], old_dwi:[]}

    for idx, s in enumerate(seqinfo):
        if (s.dim3 == 400) and ('3D_T1_TFE_SAG_CS3' in s.series_description):
            info[t1w]  = [s.series_id]
        if (s.dim3 == 160) and ('T2W' in s.series_description):
            info[t2w] = [s.series_id]
        if (s.dim3 >= 9000) and ('DelRec - DTI_1shell_b3000_MB4' in s.series_description):
            info[dwi1shell] = [s.series_id]
        if (s.dim3 >= 6840) and ('DelRec - DTI_2shell_b1000_b2000_MB4' in s.series_description):
            info[dwi2shell] = [s.series_id]
        if (s.dim3 == 56700) and ('RESTING ME3 MB3 SENSE3' in s.series_description):
            info[rest] = [s.series_id]
        if (s.dim3 == 144) and ('SE_EPI Posterior' in s.series_description):
            info[dwi_posterior].append({'item': s.series_id})
        if (s.dim3 == 630) and ('fMRI_CORRECTION_MB3_ME3_SENSE3_3mm_TR1104_TE15_DTE20' in s.series_description):
            info[rest_correction] = [s.series_id]
        if (s.dim3 == 1452) and ('SOURCE - DelRec - pCASL1' in s.series_description):
            info[asl_source] = [s.series_id]
        if (s.dim3 == 4) and ('PCA_PRE_INT' in s.series_description):
            info[asl_pre] = [s.series_id]
        if (s.dim3 == 22) and ('DelRec - pCASL1' in s.series_description):
            info[asl] = [s.series_id]
        if (s.dim3 == 5775) and ('RESTING STATE' in s.series_description):
            info[old_rest].append({'item': s.series_id})
        if ('WIP DTI_32_2.37mm' in s.series_description):
            info[old_dwi] = [s.series_id]
            
            
    return info