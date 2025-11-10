#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 12:44:05 2025

@author: edm9fd
"""

from nipype.interfaces import fsl
from nipype import Node, Workflow, SelectFiles
import os
from glob import glob
import numpy as np
import subprocess


class dti:
    def __init__(self, subject, subfolder, basename1, basename2, outdir):
        os.chdir(subfolder+'/dwi')
        self.basename1 = basename1
        self.basename2 = basename2
        self.bval1 = glob('*'+basename1+'.bval')[0]
        self.bval2 = glob('*'+basename2+'.bval')[0]
        self.bvec1 = glob('*'+basename1+'.bvec')[0]
        self.bvec2 = glob('*'+basename2+'.bvec')[0]
        self.dwi1 = glob('*'+basename1+'.nii.gz')[0]
        self.dwi2 = glob('*'+basename2+'.nii.gz')[0]
        self.outdir = outdir
        self.subfolder = subfolder
        self.subject = subject

    def combine_bval(self):
        x = np.loadtxt(self.bval1)
        y = np.loadtxt(self.bval2)
        xy = np.concatenate((x, y), axis=0)
        np.savetxt('dti_combined.bval', xy, delimiter='\t', newline="\t", fmt='%f', encoding='utf-8')


    def combine_nifti(self):
        merge = fsl.Merge()
        merge.inputs.in_files = [self.dwi1, self.dwi2]
        merge.inputs.dimension = 't'
        merge.inputs.output_type = 'NIFTI_GZ'
        merge.run()
        os.rename(glob('*_merged.nii.gz')[0], 'dti_merged.nii.gz')
        
        
    def combine_bvec(self):
        destination = 'dti_combined.bvec' #self.subfolder + 'dwi/dti_combined.bvec'
        with open(destination, 'w') as f:
            subprocess.call(["paste", self.bvec1, self.bvec2], stdout=f)

    def multishell(self):
        self.combine_bval()
        self.combine_bvec()
        self.combine_nifti()
        self._combined_bvec = 'dti_combined.bvec'
        self._combined_bval = 'dti_combined.bval'
        self._combined_nifti = 'dti_merged.nii.gz'

        
    def preproc_multishell(self, acqp, index):
        templates = {'dwi': 'dwi/dti_merged.nii.gz',
                 'bvecs': 'dwi/dti_combined.bvec',
                 'bvals': 'dwi/dti_combined.bval',
                 't1w': 'anat/'+self.t1w}
        selectfiles = Node(SelectFiles(templates, base_directory=self.subfolder), 
                       name="selectfiles")
        dti_b0 = Node(fsl.ExtractROI(t_min=0, t_size=1), 
                  name='remove_B0')
        dti_bet = Node(fsl.BET(frac=0.5, output_type='NIFTI_GZ', mask=True), 
                   name='bet_B0')
        dti_eddy = Node(fsl.Eddy(in_acqp=acqp, 
                             in_index=index, use_cuda=True, 
                             output_type='NIFTI_GZ', num_threads=1), 
                    name='eddy')
        dti_fit = Node(fsl.DTIFit(output_type='NIFTI_GZ', sse=True, save_tensor=True), 
                   name='dtifit')
        dti_bedpostx = Node(fsl.BEDPOSTX5(burn_in=200, args='-NJOBS 1', 
                                      sample_every=25, n_jumps=5000, n_fibres=3, 
                                      output_type='NIFTI_GZ'), 
                        name='bedpostX')

    
        wf = Workflow(name='dti-prep', base_dir=self.outdir)
        wf.connect([(selectfiles, dti_b0, [('dwi', 'in_file')]),
                    (selectfiles, dti_eddy, [('bvecs', 'in_bvec')]),
                    (selectfiles, dti_eddy, [('bvals', 'in_bval')]),
                    (dti_b0, dti_bet, [('roi_file', 'in_file')]),
                    (dti_bet, dti_eddy, [('mask_file', 'in_mask')]),
                    (selectfiles, dti_eddy, [('dwi', 'in_file')]),
                    
                    (dti_bet, dti_fit, [('mask_file', 'mask')]),
                    (selectfiles, dti_fit, [('bvecs','bvecs')]),
                    (selectfiles, dti_fit, [('bvals','bvals')]),
                    (dti_eddy, dti_fit, [('out_corrected','dwi')]),
                    
                    (selectfiles, dti_bedpostx, [('bvecs','bvecs')]),
                    (selectfiles, dti_bedpostx, [('bvals','bvals')]),
                    (dti_bet, dti_bedpostx, [('mask_file','mask')]),
                    (dti_eddy, dti_bedpostx, [('out_corrected','dwi')])
                    ])
        
        dti_bedpostx.inputs.use_gpu = True
        wf.write_graph(dotfilename='workflow_graph.dot', format='png')
        wf.run('MultiProc', plugin_args={'n_procs': 2, 'n_gpu_procs':1})
        
        
#%% Example usage

subject_list = glob('/mnt/bytopia/IRC805/subjects/*/nifti/dwi')

mnt, bytopia, IRC805, subjects, sub, nifti, dwi = subject_list[0].split('/')

init = dti( 'IRC805-0580101',
            '/mnt/bytopia/IRC805/subjects/IRC805-0580101/nifti',
            'DTI_2shell_b1000_b2000_MB4',
            'DTI_1shell_b3000_MB4',
            '/mnt/bytopia/IRC805/subjects/IRC805-0580101/')
init.multishell()
init.preproc_multishell('/mnt/bytopia/IRC805/dti/acqp.txt', '/mnt/bytopia/IRC805/dti/index.txt')


