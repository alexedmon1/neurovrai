"""
Performs DualRegression to create Z-maps for each mask
"""

from nipype.interfaces import fsl
import pandas as pd
import os
from joblib import Parallel, delayed

study_list = ['mriglu', 'meana11', 'meana12', 'meana13']

def dualregression(i):
    print('Performing DualRegression on ' + i)
    #os.chdir('/mnt/elysium/IRC805/rest/'+i)
    filelist = pd.read_csv('/mnt/elysium/IRC805/rest/'+i+'/filelist.txt', header=None)
    bold_files = filelist[0].tolist()

    dr = fsl.model.DualRegression()
    dr.inputs.in_files = bold_files
    dr.inputs.group_IC_maps_4D = '/mnt/elysium/IRC805/rest/MelodicICA/melodic_IC.nii.gz'
    dr.inputs.design_file = '/mnt/elysium/IRC805/rest/'+i+'/design.mat'
    dr.inputs.con_file = '/mnt/elysium/IRC805/rest/'+i+'/contrast.con'
    dr.inputs.n_perm = 5000
    dr.inputs.out_dir = '/mnt/elysium/IRC805/rest/'+i+'/results'
    dr.inputs.output_type = 'NIFTI_GZ'
    dr.run()


#dualregression('t_scastot')
results = Parallel(n_jobs=1)(delayed(dualregression)(i) for i in study_list)
