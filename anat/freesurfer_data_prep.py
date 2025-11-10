#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:47:32 2025

@author: edm9fd
"""

from glob import glob
import subprocess
import sys
import os
import pandas as pd


def create_subject_list(subdir, outdir):
    subject_files = glob(subdir+'/*')
    subject_files.remove(subdir+'/fsaverage')
    with open(outdir+'/subjects.txt', 'w') as f:
        for item in subject_files:
            f.write("%s\n" % item)
            

class fs_options:
    
    def __init__(self, subfile, outdir):
# Setup
        self.hemi_list = ['lh', 'rh']
        self.aparc_measure_list = ['area', 'volume', 'thickness', 'thicknessstd', 
                      'meancurv', 'gauscurv', 'foldind', 'curvind']
        self.parc_list = ['aparc', 'aparc.a2009s', 'aparc.DKTatlas']
        self.aseg_measure_list = ['volume']
        self.subject_list = subfile
        self.tablefile_dir = outdir


    def aparc_cmd(sfile, tdir, hemi, parc = 'aparc', meas = 'area', delim = 'comma'):
        """
        cmd = aparcstats2table --subjectsfile= --tablefile= --hemi= --parc= --measure= --skip --delimiter=
        [aparcstats2table, s, t, h, p, m, d, --skip]
        """
        s = '--subjectsfile='+sfile
        t = '--tablefile='+tdir+parc+'_'+hemi+'_'+meas+'.csv'
        h = '--hemi='+hemi
        p = '--parc='+parc
        m = '--meas='+meas
        d = '--delimiter='+delim
        return ['aparcstats2table', s, t, h, p, m, d, '--skip']


    def aseg_cmd(sfile, tdir, meas='volume', delim='comma'):
        """
        cmd = aparcstats2table --subjectsfile= --tablefile= --measure= --delimiter= --skip 
        [aparcstats2table, s, t, m, d, --skip]
        """
        s = '--subjectsfile='+sfile
        t = '--tablefile='+tdir+'asegstats.csv'
        m = '--meas='+meas
        d = '--delimiter='+delim
        return ['asegstats2table', s, t, m, d, '--skip']
    
    
    def subjects_first_column(x):
        v = []
        for i in range(len(df.iloc[:,0])):
            sub = x.iloc[:,0][i].split('/')[-1]
            sub = sub.replace('_', '-')
            v.append(sub)
        x.iloc[:,0] = v
        return x


for i in aseg_measure_list:
    cmd = aseg_cmd(subject_list, tablefile_dir, meas=i)
    subprocess.run(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    
parc = parc_list[2]
for i in hemi_list:
    for j in aparc_measure_list:
        cmd = aparc_cmd(subject_list, tablefile_dir, hemi=i, parc=parc, meas=j)
        subprocess.run(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        files = glob(tablefile_dir+'*')


for i in files:
    df = pd.read_csv(i)
    df = subjects_first_column(df)
    df.to_csv(i, index=False)
    
    
    files = glob(tablefile_dir+'aparc*')
files

for i in files:
    df = pd.read_csv(i)
    df.drop(['BrainSegVolNotVent', 'eTIV'], axis=1, inplace=True)
    df.to_csv(i, index=False) 