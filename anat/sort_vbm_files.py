#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:23:51 2022

@author: edm9fd
"""

#%%
import pandas as pd
import shutil
import os

exposure = 'mriglu'

#%%
filelist1 = pd.read_table('/mnt/elysium/IRC805/morph/vbm/'+exposure+'/filelist.txt', header=None)

#%%
for i in filelist1.iterrows():
    filename = i[1][0].split('/')[-1]
    os.makedirs('/mnt/elysium/IRC805/morph/vbm/'+exposure, exist_ok=True)
    shutil.copy(i[1][0], '/mnt/elysium/IRC805/morph/vbm/'+exposure+'/'+filename)