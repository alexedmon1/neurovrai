#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:56:17 2023

@author: edm9fd
"""

from glob import glob
import shutil
import os

FA_files = glob('/mnt/valkyrie/IRC805/dti/*/dti_preprocess/dtifit/*FA.nii.gz')
MD_files = glob('/mnt/valkyrie/IRC805/dti/*/dti_preprocess/dtifit/*MD.nii.gz')
AD_files = glob('/mnt/valkyrie/IRC805/dti/*/dti_preprocess/dtifit/*L1.nii.gz')
L2_files = glob('/mnt/valkyrie/IRC805/dti/*/dti_preprocess/dtifit/*L2.nii.gz')
L3_files = glob('/mnt/valkyrie/IRC805/dti/*/dti_preprocess/dtifit/*L3.nii.gz')


FA_files = glob('/mnt/elysium/IRC805/dti/*preprocess/*/dti_preprocess/dtifit/*FA.nii.gz')
MD_files = glob('/mnt/elysium/IRC805/dti/*preprocess/*/dti_preprocess/dtifit/*MD.nii.gz')
AD_files = glob('/mnt/elysium/IRC805/dti/*preprocess/*/dti_preprocess/dtifit/*L1.nii.gz')
L2_files = glob('/mnt/elysium/IRC805/dti/*preprocess/*/dti_preprocess/dtifit/*L2.nii.gz')
L3_files = glob('/mnt/elysium/IRC805/dti/*preprocess/*/dti_preprocess/dtifit/*L3.nii.gz')




for i in FA_files:
    subject=i.split('/')[-4]
    shutil.copy(i,'/mnt/elysium/IRC805/dti/dtifit/FA/'+subject+'_FA.nii.gz')
    
for i in MD_files:
    subject=i.split('/')[-4]
    shutil.copy(i,'/mnt/elysium/IRC805/dti/dtifit/MD/'+subject+'_FA.nii.gz')

for i in AD_files:
    subject=i.split('/')[-4]
    shutil.copy(i,'/mnt/elysium/IRC805/dti/dtifit/L1/'+subject+'_FA.nii.gz')

for i in L2_files:
    subject=i.split('/')[-4]
    shutil.copy(i,'/mnt/elysium/IRC805/dti/dtifit/L2/'+subject+'_L2.nii.gz')

for i in L3_files:
    subject=i.split('/')[-4]
    shutil.copy(i,'/mnt/elysium/IRC805/dti/dtifit/L3/'+subject+'_L3.nii.gz')
