#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 08:56:17 2025

@author: edm9fd
"""

import os
import subprocess
import shutil
from glob import glob
import pydicom

class support:
    def __init__(self):
        self=self
        
        
    def subject_number(dcm_folder):
        f1 = dcm_folder.split('/')
        return f1[-3]
    
    
    def dicom_info(dicom):
        dcm = pydicom.dcmread(dicom)
        desc = dcm.SeriesDescription
        #subj = dcm. #Move subject no. to here. 
        return desc
    
    
    def set_modality(descriptor):
        if any(descriptor in sub for sub in ['T2W CS5 OF1 TR2500', 'T2W Sagittal Reformat', '3D_T1_TFE_SAG_CS3', 'AX T1 MPR', 'COR T1 MPR']):
            return 'anat'
        if any(descriptor in sub for sub in ['fMRI_CORRECTION_MB3_ME3_SENSE3_3mm_TR1104_TE15_DTE20', 'RESTING ME3 MB3 SENSE3', 'RESTING STATE']):
            return 'rest'
        if any(descriptor in sub for sub in ['SE_EPI Posterior', 'DelRec - DTI_1shell_b3000_MB4', 'DelRec - DTI_2shell_b1000_b2000_MB4', 'dWIP DTI_32_2.37mm CLEAR', 'facWIP DTI_32_2.37mm CLEAR', 'DTI_32_2.37mm', 'isoWIP DTI_32_2.37mm CLEAR']):
            return 'dwi'
        if any(descriptor in sub for sub in ['PCA_PRE_INT', 'DelRec - pCASL1', 'DelRec - pCASL1','WIP SOURCE - DelRec - pCASL1']):
            return 'asl'
        if any(descriptor in sub for sub in ['Survey']):
            return 'others'

class dcm2niix(support):
    def __init__(self, dcm_folder, outdir):
        self.subject = support.subject_number(dcm_folder)    
        self.outdir = outdir
        self.dcm_folder = dcm_folder
        self.dcm_descriptor = support.dicom_info(glob(dcm_folder+'/*.dcm')[0])
        self.modality = support.set_modality(self.dcm_descriptor)
        

    def run_dcm2niix(self):
        options = ['-z', 'y']
        filename = ['-f', '%s_%i_%z_%p']
        command_list = ['dcm2niix'] + options + filename + [self.dcm_folder]
        subprocess.run(command_list, stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE)


    def external_file_lists(self):
        nii_files = glob(self.dcm_folder+'/*.nii.gz')
        json_files = glob(self.dcm_folder+'/*.json')
        bval_files = glob(self.dcm_folder+'/*.bval')
        bvec_files = glob(self.dcm_folder+'/*.bvec')
        self.nii_files = nii_files
        self.json_files = json_files
        self.bval_files = bval_files
        self.bvec_files = bvec_files


    def transfer_files(self):
        for i in self.nii_files:
            filename = i.split('/')[-1]
            os.makedirs(self.outdir+'/'+self.subject+'/'+self.modality+'/', exist_ok=True)
            shutil.copy(i, self.outdir+'/'+self.subject+'/'+self.modality+'/'+filename)
        for i in self.json_files:
            filename = i.split('/')[-1]
            os.makedirs(self.outdir+'/'+self.subject+'/'+self.modality+'/', exist_ok=True)
            shutil.copy(i, self.outdir+'/'+self.subject+'/'+self.modality+'/'+filename)
        if self.modality == 'dwi':
            for i in self.bval_files:
                filename = i.split('/')[-1]
                os.makedirs(self.outdir+'/'+self.subject+'/'+self.modality+'/', exist_ok=True)
                shutil.copy(i, self.outdir+'/'+self.subject+'/'+self.modality+'/'+filename)
            for i in self.bvec_files:
                filename = i.split('/')[-1]
                os.makedirs(self.outdir+'/'+self.subject+'/'+self.modality+'/', exist_ok=True)
                shutil.copy(i, self.outdir+'/'+self.subject+'/'+self.modality+'/'+filename)
        else:
            next
            
    def unique_images(self, dcm_folders):
        d = []
        for i in dcm_folders:
            os.chdir(i)
            dcm1 = glob('*.dcm')[0]
            dcm2 = pydicom.dcmread(dcm1)
            d.append(dcm2.SeriesDescription)
        self.res = list(set(d))

#%%

dicom_folders = glob('/mnt/bytopia/IRC805/dicom/*/*/*')
for i in dicom_folders:
    dcm = dcm2niix(i,'/mnt/bytopia/IRC805/bids/')
    dcm.run_dcm2niix()
    dcm.external_file_lists()
    dcm.transfer_files()



#%%
d = []
for i in dicom_folders:
    os.chdir(i)
    dcm1 = glob('*.dcm')[0]
    dcm2 = pydicom.dcmread(dcm1)
    d.append(dcm2.SeriesDescription)
res = list(set(d))