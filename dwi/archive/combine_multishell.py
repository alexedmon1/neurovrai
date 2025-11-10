#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:43:12 2022

@author: edm9fd
"""

from nipype.interfaces import fsl
import numpy as np
import os
from glob import glob
import shutil
import subprocess

def combine_bval(file1, file2):
    x = np.loadtxt(file1)
    y = np.loadtxt(file2)
    xy = np.concatenate((x, y), axis=0)
    return xy



def combine_bvec(file1, file2):
    x = np.loadtxt(file1)
    y = np.loadtxt(file2)
    xy = np.concatenate((x, y), axis=1)
    return xy


def combine_nifti(nii1000, nii3000):
    merge = fsl.Merge()
    merge.inputs.in_files = [nii1000, nii3000]
    merge.inputs.dimension = 't'
    merge.inputs.output_type = 'NIFTI_GZ'
    merge.run()


def combined_bvec(subject, b1000_file, b3000_file):
    destination = '/mnt/elysium/IRC805/dti/preprocess/' + subject + '/sub-'+subject+'_dwi_combined.bvec'
    with open(destination, 'w') as f:
        subprocess.call(["paste", b1000_file, b3000_file], stdout=f)

    


dwi_folders = ['/mnt/elysium/IRC805/heudiconv/2930302/dwi',
               '/mnt/elysium/IRC805/heudiconv/3330101/dwi']#glob('/mnt/elysium/IRC805/heudiconv/*/dwi')
print('Combining multishell data...')
for i in dwi_folders:
    os.chdir(i)
    subject = str(i.split('/')[-2])
    print(subject + ' . . .')
    bval_1000 = 'sub-'+subject+'_dwi_b1000_b2000.bval'
    bval_3000 = 'sub-'+subject+'_dwi_b3000.bval'
    bvec_1000 = 'sub-'+subject+'_dwi_b1000_b2000.bvec'
    bvec_3000 = 'sub-'+subject+'_dwi_b3000.bvec'
    nifti_1000 = 'sub-'+subject+'_dwi_b1000_b2000.nii.gz'
    nifti_3000 = 'sub-'+subject+'_dwi_b3000.nii.gz'

    bvals = combine_bval(bval_1000, bval_3000)
    bvecs = combine_bvec(bvec_1000, bvec_3000)
    np.savetxt('/mnt/elysium/IRC805/dti/preprocess/'+subject+'/sub-'+subject+'_dwi_combined.bval', bvals, delimiter='\t', newline="\t", fmt='%f', encoding='utf-8')
    
    combined_bvec(subject, bvec_1000, bvec_3000)
    combine_nifti(nifti_1000, nifti_3000)
    print('. . . Done.')
    
    
    
"""
    with open('/mnt/valkyrie/IRC805/dti/'+subject+'/sub-'+subject+'_dwi_combined.bvec','wb') as wfd:
        for f in [bvec_1000, bvec_3000]:
            with open(f,'rb') as fd:
                print(fd)
                shutil.copyfileobj(fd, wfd)
                

    with open('/mnt/valkyrie/IRC805/dti/'+subject+'/sub-'+subject+'_dwi_combined.bval','wb') as wfd:
        for f in [bval_1000, bval_3000]:
            with open(f,'rb') as fd:
                print(fd)
                shutil.copyfileobj(fd, wfd)
                                
"""           
                

#

def combined_bvec(subject, b1000_file, b3000_file):
    destination = '/mnt/valkyrie/IRC805/dti/' + subject + '/sub-'+subject+'_dwi_combined.bvec'
    command_list = ["paste"] + ['-d" "'] + b1000_file + b3000_file + [">>"] + destination
    subprocess.run(command_list)
    


## Index file

z = [1 for x in range(220)]

np.savetxt('/mnt/valkyrie/IRC805/dti/index.txt', z, delimiter=" ", newline=" ", fmt='%d' )





