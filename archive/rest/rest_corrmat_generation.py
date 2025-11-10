#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:45:47 2020

@author: edm9fd
"""

import os
import glob
import pandas as pd
from nipype.interfaces import fsl
from scipy.stats import pearsonr

os.chdir('/mnt/elysium/IRC805/rest/proc/')
filenames = glob.glob('*_rest.nii.gz')
filenames

atlas = '/mnt/share/Atlases/AAL2/aal2.nii.gz'
atlas_labels = '/mnt/share/Atlases/AAL2/aal2.nii.txt'



def parse_filename(f):
    f1 = f.split('.')
    f2 = f1[0].split('_')
    return f2


def AAL_extract_mean_timeseries(f, label_file):
    subject_number = parse_filename(f)[0]
    meants = fsl.utils.ImageMeants()
    meants.inputs.in_file = f
    meants.inputs.args = "--label="+label_file
    meants.inputs.output_type = 'NIFTI_GZ'
    meants.inputs.out_file = subject_number + '_AAL2_fslmeants.txt'
    meants.run()


def subject_corrmatrix(filename):
    tcf = pd.read_csv(filename, header=None, sep='  ')
    cols = len(tcf.columns)
    df = pd.DataFrame()
    for row in range(cols):
        z = list()
        for col in range(cols):
            corr, pvalue = pearsonr(tcf[row].values, tcf[col].values)
            z.append(corr)
        dfz = pd.DataFrame(z).transpose()
        df = pd.concat([df, dfz], axis=0)
    df.reset_index(inplace=True, drop=True)
    return df


for i in filenames:
    AAL_extract_mean_timeseries(i, atlas)
#%%
rois = pd.read_csv(atlas_labels, header=None, sep = ' ')
for i in glob.glob('*_AAL2_fslmeants.txt'):
    subject_number = parse_filename(i)[0]
    df = subject_corrmatrix(i)
    df.columns = rois[1]
    df.to_csv(subject_number + '_pearsonr_matrix.csv')

#%%
import seaborn as sns
df = pd.read_csv('0580101_pearsonr_matrix.csv')
df = df.drop('Unnamed: 0', axis=1)
sns.heatmap(df, vmin=-1, vmax=1)
rois
