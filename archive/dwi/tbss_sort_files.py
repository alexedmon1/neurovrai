#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:33:14 2022

@author: edm9fd
"""
#%%
import pandas as pd
import shutil
import os

exposure = 'meana12'

#%%
filelist = pd.read_table('/mnt/elysium/IRC805/dti/tbss/'+exposure+'/filelist.txt', header=None)
for i in filelist.iterrows():
    filename = i[1][0].split('/')[-1]
    shutil.copy(i[1][0], '/mnt/elysium/IRC805/dti/tbss/'+exposure+'/'+filename)
    
