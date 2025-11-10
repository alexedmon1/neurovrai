#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:24:24 2021

@author: edm9fd

DTI Preprocessing using nipype:
    1. eddy correction - use CUDA
    2. BET
    3. dtifit - save file 
    4. bedpostx - save file - use CUDA
    5. probtrackx? - perhaps perform separately for GPU usage
    
    
module load cuda/9.1
"""

from nipype.interfaces import fsl
from nipype import Node, Workflow, SelectFiles
import os
from glob import glob

def dti_preproc(subject_no, basedir):

    templates = {'dwi': str(subject_no)+'_dti.nii',
                 'bvecs': str(subject_no)+'_dti.bvec',
                 'bvals': str(subject_no)+'_dti.bval',
                 't1w': str(subject_no)+'_t1w.nii.gz'}
    selectfiles = Node(SelectFiles(templates, base_directory=basedir), 
                       name="selectfiles")
    dti_b0 = Node(fsl.ExtractROI(t_min=0, t_size=1), 
                  name='remove_B0')
    dti_bet = Node(fsl.BET(frac=0.5, output_type='NIFTI_GZ', mask=True), 
                   name='bet_B0')
    dti_eddy = Node(fsl.Eddy(in_acqp='/mnt/elysium/IRC805/dti/acqp.txt', 
                             in_index='/mnt/elysium/IRC805/dti/index3.txt',
                             use_cuda=True, output_type='NIFTI_GZ', num_threads=10), 
                    name='eddy')
    dti_fit = Node(fsl.DTIFit(output_type='NIFTI_GZ', sse=True, save_tensor=True), 
                   name='dtifit')
    dti_bedpostx = Node(fsl.BEDPOSTX5(burn_in=200, args='-NJOBS 10', 
                                      sample_every=25, n_jumps=5000, n_fibres=3, 
                                      output_type='NIFTI_GZ'), 
                        name='bedpostX')
    
    t1w_reorient = Node(fsl.Reorient2Std(), name='t1w_reorient')
    t1w_fast = Node(fsl.FAST(img_type=1, output_type='NIFTI_GZ', output_biascorrected=True), 
                    name='t1w_fast')
    t1w_bet = Node(fsl.BET(output_type='NIFTI_GZ', reduce_bias=True, frac=0.5), 
                   name='t1w_bet')
    t1w_flirt = Node(fsl.FLIRT(reference='/home/edm9fd/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz',
                               output_type='NIFTI_GZ', dof=12, cost_func='bbr', out_file='t1w_flirt_output.nii.gz'), 
                     name='t1w_flirt')
    t1w_fnirt = Node(fsl.FNIRT(ref_file='/home/edm9fd/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz',
                               output_type='NIFTI_GZ'), name='t1w_fnirt')
    dti_2t1w = Node(fsl.FLIRT(output_type='NIFTI_GZ', dof=12, cost_func='mutualinfo'), name='dti_flirt')
    dti_xfm = Node(fsl.ConvertXFM(output_type='NIFTI_GZ', concat_xfm=True),
                   name='dti_xfm')
    dti_warp = Node(fsl.ConvertWarp(reference='/home/edm9fd/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz',
                                    out_file='dti_probtrackx_invxfm.nii.gz', output_type='NIFTI_GZ'), name='dti_warp') #combine dti_2t1w, t1w_flirt, t1w_fnirt
    dti_apply_warp = Node(fsl.ApplyWarp(ref_file='/home/edm9fd/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz', 
                                       output_type='NIFTI_GZ', out_file='warped_dti_b0.nii.gz'), name='dti_apply_warp')
    
    
    wf = Workflow(name='dti_preprocess', base_dir=i)
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
                (dti_eddy, dti_bedpostx, [('out_corrected','dwi')]),
                
                (selectfiles, t1w_reorient, [('t1w','in_file')]),
                (t1w_reorient, t1w_fast, [('out_file', 'in_files')]),
                (t1w_fast, t1w_bet, [('restored_image','in_file')]),
                (t1w_bet, t1w_flirt, [('out_file','in_file')]),
                (t1w_flirt, t1w_fnirt, [('out_file', 'in_file')]),
                (dti_bet, dti_2t1w, [('out_file','in_file')]),
                (t1w_bet, dti_2t1w, [('out_file', 'reference')]),
                (dti_2t1w, dti_xfm, [('out_matrix_file', 'in_file')]),
                (t1w_flirt, dti_xfm, [('out_matrix_file', 'in_file2')]),
                (dti_xfm, dti_warp, [('out_file','premat')]),
                (t1w_fnirt, dti_warp, [('field_file', 'warp1')]),
                (dti_bet, dti_apply_warp, [('out_file', 'in_file')]),
                (dti_warp, dti_apply_warp, [('out_file', 'field_file')])
                ])
    
    dti_bedpostx.inputs.use_gpu = True
    wf.write_graph(dotfilename='workflow_graph.dot', format='png')
    wf.run('MultiProc', plugin_args={'n_procs': 10})
    
    
subject_list = glob('/mnt/elysium/IRC805/dti/single_preprocess/*/')
with open('/mnt/elysium/IRC805/status/dti_preproc_status.txt') as f:
    done_list = [line.rstrip('\n').strip() for line in f]

for sub in done_list:
    subject_list.remove(sub)


for i in subject_list:
    subject = i.split('/')[-2]
    print(subject)
    dti_preproc(subject, i)
    file1 = open('/mnt/elysium/IRC805/status/dti_preproc_status.txt', 'a')
    file1.write(i +' \n')
    file1.close()
