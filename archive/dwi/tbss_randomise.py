#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:12:14 2022

@author: edm9fd


"""
import nipype.interfaces.fsl as fsl
from glob import glob
from joblib import Parallel, delayed
import os



def randomise(exposure):
    #os.chdir('/mnt/elysium/IRC805/dti/tbss/'+exposure)
    files = glob('/mnt/elysium/IRC805/dti/tbss/'+exposure+'/stats/all_*_skeletonised.nii.gz')
    for i in files:
        name = i.split('/')[-1].split('.')[0].split('_')[1]
        rand = fsl.Randomise()
        rand.inputs.in_file = i
        rand.inputs.design_mat = '/mnt/elysium/IRC805/dti/tbss/'+exposure+'/design.mat'
        rand.inputs.tcon = '/mnt/elysium/IRC805/dti/tbss/'+exposure+'/contrast.con'
        rand.inputs.mask = '/mnt/elysium/IRC805/dti/tbss/'+exposure+'/stats/mean_FA_mask.nii.gz'
        rand.inputs.output_type = 'NIFTI_GZ'
        rand.inputs.tfce2D = True
        rand.inputs.base_name = exposure+'_'+name
        rand.run()

exposure = ['meana11', 'meana12', 'meana13']
#files = glob('stats/all_*_skeletonised.nii.gz')
os.chdir('/mnt/elysium/IRC805/dti/tbss')
results = Parallel(n_jobs=3)(delayed(randomise)(i) for i in exposure)

