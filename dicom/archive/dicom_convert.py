#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:31:47 2021

@author: AE
"""

from glob import glob
import subprocess
import sys

subject_folders = glob('/mnt/bytopia/IRC805/dicom/*')#glob(sys.argv[1]+'*')

def subject_number(f):
    f1 = f.split('/')
    return f1[-1]

def heudiconv(sub):
    """
    heudiconv 
    -d 
    -s sub
    -f 
    -o 

    """
    subject = subject_number(sub)
    dicom_dir = ['-d', i+'/*/*']
    subject = ['-s', sub]
    heuristic = ['-f', 'heuristic.py']
    converter = ['-c', 'dcm2niix']
    out_dir = ['-o', 'bids']
    other = ['--overwrite']
    command_list = ['uvx'] + ['heudiconv'] + dicom_dir + subject + heuristic + converter + out_dir + other
    subprocess.run(command_list, stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

for i in subject_folders:
    print('Processing '+i+ '. . .')
    heudiconv(i)

