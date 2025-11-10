#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:53:11 2023

@author: edm9fd

After running fsl-cluster.sh in every folder...
Combine all cluster results files into a single dataframe
Use this dataframe to inquire whether any results are > 0.95 (p < 0.05)
"""

import pandas as pd
from glob import glob
import os

study_folder = '/mnt/bytopia/IRC805/dti/tbss/mriglu/'
analyses = [i.split('/')[-1] for i in glob(study_folder+'*')]
results_folder_name = 'stats'

def split_results_filename(f):
    f1 = f.split('/')
    f2 = f1[-1].split('_')
    analysis, mask, test = f1[-3], f2[0], f2[-2]
    return analysis, mask, test

cluster_df = pd.DataFrame(columns=['Analysis', 'Region', 'Test', 'Cluster Index', 'Voxels', 'MAX', 'MAX X (vox)', 'MAX Y (vox)',
       'MAX Z (vox)', 'COG X (vox)', 'COG Y (vox)', 'COG Z (vox)'])

"""
for i in analyses:
    files = glob(study_folder + i +'/' + results_folder_name +'/*_clusters.txt')
    for j in files:
        a, m, t = split_results_filename(j)
        df = pd.read_table(j)
        df['Analysis'] = a
        df['Region'] = m
        df['Test'] = t
        cluster_df = pd.concat([cluster_df, df], axis=0)

"""
files = glob(study_folder +'/' + '*_clusters.txt')
for j in files:
        a, m, t = split_results_filename(j)
        df = pd.read_table(j)
        df['Analysis'] = a
        df['Region'] = m
        df['Test'] = t
        cluster_df = pd.concat([cluster_df, df], axis=0)



#cluster_df.to_csv(study_folder+'/phthbpa_sex_interaction_cluster_results.csv', index=False)
