#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:02:51 2022

@author: edm9fd
"""

#%%
import subprocess
from glob import glob
import os
import shutil

#%% Definitions

def create_RD(L2_file, L3_file, output):
    command_list = ["fslmaths"] + [L2_file] + ["-add"] + [L3_file] + ["-div","2"] + [output]
    subprocess.run(command_list)

#%%
exposure = 'meana13'
with open('/mnt/elysium/IRC805/dti/tbss/'+exposure+'/filelist.txt') as t:
    FA_files = t.read().splitlines()

#%% MD

os.makedirs('/mnt/elysium/IRC805/dti/tbss/'+exposure+'/MD', exist_ok=True)

for f in FA_files:
    subject = f.split('/')[-1].split('_')[0]
    shutil.copy('/mnt/elysium/IRC805/dti/dtifit/MD/'+subject+'_FA.nii.gz', 
                '/mnt/elysium/IRC805/dti/tbss/'+exposure+'/MD/'+subject+'_FA.nii.gz')

#%% AD

os.makedirs('/mnt/elysium/IRC805/dti/tbss/'+exposure+'/AD', exist_ok=True)

for f in FA_files:
    subject = f.split('/')[-1].split('_')[0]
    shutil.copy('/mnt/elysium/IRC805/dti/dtifit/L1/'+subject+'_FA.nii.gz',
                '/mnt/elysium/IRC805/dti/tbss/'+exposure+'/AD/'+subject+'_FA.nii.gz')

#%% RD

os.makedirs('/mnt/elysium/IRC805/dti/tbss/'+exposure+'/RD', exist_ok=True)

for f in FA_files:
    subject = f.split('/')[-1].split('_')[0]
    l2 = '/mnt/elysium/IRC805/dti/dtifit/L2/'+subject+'_L2.nii.gz'
    l3 = '/mnt/elysium/IRC805/dti/dtifit/L3/'+subject+'_L3.nii.gz'
    out = '/mnt/elysium/IRC805/dti/tbss/'+exposure+'/RD/'+subject+'_FA.nii.gz'
    create_RD(l2, l3, out)

#%%

files = glob('/mnt/elysium/IRC805/dti/dtifit/FA/*.nii.gz')
for i in files:
    subject = i.split('/')[-1].split('_')[0]
    l2 = '/mnt/elysium/IRC805/dti/dtifit/L2/'+subject+'_L2.nii.gz'
    l3 = '/mnt/elysium/IRC805/dti/dtifit/L3/'+subject+'_L3.nii.gz'
    out = '/mnt/elysium/IRC805/dti/dtifit/RD/'+subject+'_FA.nii.gz'
    create_RD(l2,l3,out)
