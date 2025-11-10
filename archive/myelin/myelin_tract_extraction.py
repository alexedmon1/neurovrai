#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:11:55 2023

@author: edm9fd
"""

import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image
import numpy as np
from glob import glob

subject_list = glob('/mnt/elysium/IRC805/myelin/myelin_maps/smooth/*_3mm.nii.gz')
JHU_label_map = '/usr/local/fsl/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz'
JHU_labels = '/mnt/elysium/IRC805/myelin/myelin_maps/jhu_labels.txt'
labels = pd.read_csv(JHU_labels, index_col='index')
img = image.get_data(JHU_label_map)
labels2 = labels.drop(0, axis=0)


def extract_data_from_labels(f, strat='mean'):
    masker = NiftiLabelsMasker(labels_img=JHU_label_map, labels=labels, strategy=strat)
    masker.fit(f)
    res = masker.transform(f)
    return res


#%% JHU Data Extraction -> Dictionary
d1 = {}
for i in subject_list:
    mu = extract_data_from_labels(i, 'mean')
    med = extract_data_from_labels(i, 'median')
    var = extract_data_from_labels(i, 'variance')
    skew = 3 * (mu - med) / np.sqrt(var)
    
    subject = i.split('/')[-1].split('_')[0]
    d1[subject] = {'mean':mu,
                   'median':med,
                   'variance':var,
                   'skew':skew}    

#%% Create Dataframe

stat = 'variance'
n = 1

for i in d1.keys():
    if n == 1:
        df = pd.DataFrame.from_dict(d1[i][stat])  
        df.columns = labels2['label']
        df['subject'] = i
        df.set_index('subject', inplace=True)
        n += 1
    elif n > 1:
        df2 = pd.DataFrame.from_dict(d1[i][stat])  
        df2.columns = labels2['label']
        df2['subject'] = i
        df2.set_index('subject', inplace=True)
        df = pd.concat([df, df2], axis=0)
        n += 1
        
df.to_csv('/mnt/elysium/IRC805/data/JHU_WM_tracts_myelin_'+stat+'.csv')
