#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:14:41 2023

@author: edm9fd
"""

from glob import glob
import shutil
import os

#%% Organize Files

files = glob('/mnt/elysium/IRC805/myelin/*/T1w_T2w_ratio/*.nii.gz')

#os.makedirs('/mnt/elysium/IRC805/myelin/myelin_maps/')
for i in files:
    subject = i.split('/')[-3].split('_')[-1]
    shutil.copy(i, '/mnt/elysium/IRC805/myelin/myelin_maps/'+str(subject)+'_myelin.nii.gz')
    
#%% Smooth Files
from nipype.interfaces import fsl

def smooth_image(img, fwhm):
    subject = img.split('/')[-1].split('_')[0]
    smooth = fsl.Smooth()
    smooth.inputs.in_file = img
    smooth.inputs.fwhm = fwhm
    smooth.inputs.output_type = 'NIFTI_GZ'
    smooth.inputs.smoothed_file = '/mnt/elysium/IRC805/myelin/myelin_maps/smooth/'+str(subject)+'_'+str(fwhm)+'mm.nii.gz'
    smooth.run()

files = glob('/mnt/elysium/IRC805/myelin/myelin_maps/*_myelin.nii.gz')
for i in files:
    smooth_image(i, 2)
    smooth_image(i, 3)
    smooth_image(i, 4)
    


