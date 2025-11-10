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
    rand = fsl.Randomise()
    rand.inputs.in_file = '/mnt/elysium/IRC805/morph/vbm/mriglu/stats/GM_mod_merg_s2.nii.gz'
    rand.inputs.design_mat = '/mnt/elysium/IRC805/morph/vbm/mriglu/design.mat'
    rand.inputs.tcon = '/mnt/elysium/IRC805/morph/vbm/mriglu/contrast.con'
    rand.inputs.mask = '/mnt/elysium/IRC805/morph/vbm/mriglu/stats/GM_mask.nii.gz'
    rand.inputs.tfce = True
    rand.inputs.output_type = 'NIFTI_GZ'
    rand.inputs.base_name = exposure +'_2mm_cont'
    rand.run()

exposures = ['mriglu']

results = Parallel(n_jobs=2)(delayed(randomise)(i) for i in exposures)

