#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:14:26 2023

@author: edm9fd
"""
#%%
from nipype.interfaces import fsl
from glob import glob
import os
import subprocess


def subject_no(f):
    f1 = f.split('/')[-1].split('_')[0]
    return f1


def create_RD(L2_file, L3_file, output):
    command_list = ["fslmaths"] + [L2_file] + ["-add"] + [L3_file] + ["-div","2"] + [output]
    subprocess.run(command_list)
   
    
def warp_nonFA_file(f, w, m):
    subj = subject_no(f)
    newfilename = str(subj)+'_'+m+'_target.nii.gz'
    aw = fsl.ApplyWarp()
    aw.inputs.in_file = f
    aw.inputs.field_file = w
    aw.inputs.ref_file = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
    aw.inputs.output_type = 'NIFTI_GZ'
    aw.inputs.out_file = newfilename
    aw.run()

#%%
MD_files = glob('/mnt/elysium/IRC805/dti/dtifit/data/MD/*.nii.gz')
os.chdir('/mnt/elysium/IRC805/dti/dtifit/data/MD')
for i in MD_files:
    subj = subject_no(i)
    warp_file = '/mnt/elysium/IRC805/dti/dtifit/data/FA/'+str(subj)+'_FA_FA_to_target_warp.nii.gz'
    warp_nonFA_file(i, warp_file, 'MD')
    
#%%  
AD_files = glob('/mnt/elysium/IRC805/dti/dtifit/data/AD/*.nii.gz')

os.chdir('/mnt/elysium/IRC805/dti/dtifit/data/AD')
for i in AD_files:
    subj = subject_no(i)
    warp_file = '/mnt/elysium/IRC805/dti/dtifit/data/FA/'+str(subj)+'_FA_FA_to_target_warp.nii.gz'
    warp_nonFA_file(i, warp_file, 'AD')
        
    
#%%

os.makedirs('/mnt/elysium/IRC805/dti/dtifit/data/RD', exist_ok=True)
L2_files = glob('/mnt/elysium/IRC805/dti/dtifit/L2/*.nii.gz')
                
for f in L2_files:
    subject = f.split('/')[-1].split('_')[0]
    l2 = '/mnt/elysium/IRC805/dti/dtifit/L2/'+subject+'_L2.nii.gz'
    l3 = '/mnt/elysium/IRC805/dti/dtifit/L3/'+subject+'_L3.nii.gz'
    out = '/mnt/elysium/IRC805/dti/dtifit/data/RD/'+subject+'_RD.nii.gz'
    create_RD(l2, l3, out)
    
RD_files = glob('/mnt/elysium/IRC805/dti/dtifit/data/RD/*.nii.gz')
os.chdir('/mnt/elysium/IRC805/dti/dtifit/data/RD')
for i in RD_files:
    subj = subject_no(i)
    warp_file = '/mnt/elysium/IRC805/dti/dtifit/data/FA/'+str(subj)+'_FA_FA_to_target_warp.nii.gz'
    warp_nonFA_file(i, warp_file, 'RD')
        
    