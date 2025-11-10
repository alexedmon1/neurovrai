#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:07:57 2022

@author: edm9fd
"""

from nipype.interfaces import fsl
from glob import glob
import os

def average_multiecho(img1, img2, img3):
    maths = fsl.MultiImageMaths()
    maths.inputs.in_file = img1
    maths.inputs.op_string = '-add %s -add %s -div 3'
    maths.inputs.operand_files = [img2, img3]
    maths.inputs.output_type = "NIFTI_GZ"
    maths.inputs.out_file = 'mean_merest.nii.gz'
    maths.run()

subject_folders = glob('/mnt/elysium/IRC805/heudiconv/*/')

for i in subject_folders:
    try:
        print('Averaging ' + i)
        os.chdir(i + '/func')
        files = glob('*-rest_bold_heudiconv*.nii.gz')
        average_multiecho(files[0], files[1], files[2])
    except:
        None