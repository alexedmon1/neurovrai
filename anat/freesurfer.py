#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 07:49:15 2020

Wrapper for Freesurfer

@author: edm9fd
"""
from glob import glob
from nipype.interfaces.freesurfer import ReconAll
import os




def recon_all(subject_folder, sub_dir):
    sub = subject_folder.split('/')[-1]
    
    reconall = ReconAll()
    reconall.inputs.subject_id = sub
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = sub_dir
    reconall.inputs.T1_files = glob(i+'/nifti/anat/*_T1_*.nii.gz')[0]  #CUSTOM
    reconall.inputs.T2_file = glob(i+'/nifti/anat/*_T2W_Sagittal_*.nii.gz')[0] #CUSTOM
    reconall.inputs.use_T2 = True
    reconall.inputs.parallel = True
    reconall.inputs.hires = True
    reconall.inputs.openmp = 10
    reconall.run()    
    
SUBJECTS_DIR = '/mnt/bytopia/IRC805/freesurfer/' #INSERT SUBJECTS_DIR
subjects = glob('/mnt/bytopia/IRC805/subjects/*') #INSERT FOLDER WITH SUBJECTS NIFTI DATA

completed = []
for i in subjects:
    s = i.split('/')[-1]
    if os.path.isdir(SUBJECTS_DIR + '/' + str(s)):
        print('Skipping '+str(s))
        next
    else:
        print(str(s))
        recon_all(i, SUBJECTS_DIR)
        outfile = open('/mnt/bytopia/IRC805/status/freesurfer_complete.txt', 'a') #CUSTOM
        outfile.write(str(s)+' \n')
        outfile.close()
        
