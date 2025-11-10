#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:07:41 2025

@author: edm9fd
"""
from nipype.interfaces import fsl
from nipype import Node, Workflow, SelectFiles
import os
from glob import glob

class anat:
    def __init__(self, folder, t1wfile, outdir):
        os.chdir(folder)
        self = self
        self.folder = folder
        self.outdir = outdir
        self.t1w = glob('anat/*'+t1wfile+'.nii.gz')[0]
        self.fsldir = os.getenv('FSLDIR')
    
    def preproc_anat(self):
        templates = {'t1w': self.t1w}
        selectfiles = Node(SelectFiles(templates, base_directory=self.folder), 
                           name="selectfiles")
        t1w_reorient = Node(fsl.Reorient2Std(), name='t1w_reorient')
        t1w_fast = Node(fsl.FAST(img_type=1, output_type='NIFTI_GZ', output_biascorrected=True), 
                        name='t1w_bias_correction')
        t1w_bet = Node(fsl.BET(output_type='NIFTI_GZ', reduce_bias=True, frac=0.5), 
                       name='t1w_skullstrip')
        t1w_flirt = Node(fsl.FLIRT(reference=self.fsldir+'/data/standard/MNI152_T1_2mm_brain.nii.gz',
                                   output_type='NIFTI_GZ', 
                                   dof=12, 
                                   cost_func='bbr', 
                                   out_file='t1w_flirt.nii.gz',
                                   out_matrix_file = 't1w_flirt.mat'), 
                         name='t1w_MNI152_affine')
        t1w_fnirt = Node(fsl.FNIRT(ref_file=self.fsldir+'/data/standard/MNI152_T1_2mm_brain.nii.gz',
                                   output_type='NIFTI_GZ',
                                   warped_file = 't1w_fnirt.nii.gz',
                                   field_file = 't1w_fnirt_warp_field.nii.gz',
                                   fieldcoeff_file = 't1w_fnirt_coefficients.nii.gz'), name='t1w_MNI152_warp')
        
        
        wf = Workflow(name='anat-prep', base_dir=self.outdir)
        wf.connect([
                    (selectfiles, t1w_reorient, [('t1w','in_file')]),
                    (t1w_reorient, t1w_fast, [('out_file', 'in_files')]),
                    (t1w_fast, t1w_bet, [('restored_image','in_file')]),
                    (t1w_bet, t1w_flirt, [('out_file','in_file')]),
                    (t1w_flirt, t1w_fnirt, [('out_matrix_file', 'affine_file')]),
                    (t1w_bet, t1w_fnirt, [('out_file', 'in_file')])
                    ])

        wf.write_graph(dotfilename='workflow_graph.dot', format='png')
        wf.run('MultiProc', plugin_args={'n_procs': 2, 'n_gpu_procs':1})
    
    
#%%
subject_list = glob('/mnt/bytopia/IRC805/subjects/*/nifti/anat')
for sub_folder in subject_list:
    path = subject_list[0].split('/')
    nifti_folder = os.path.join('/',path[1], path[2], path[3], path[4], path[5], path[6])
    subject = path[5]
    out = os.path.join('/',path[1], path[2], path[3], path[4], path[5])
    proc = anat(nifti_folder, 'WIP_3D_T1_TFE_SAG_CS3', out)
    proc.preproc_anat()
