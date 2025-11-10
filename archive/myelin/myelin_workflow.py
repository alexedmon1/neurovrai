#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:11:56 2023

Should be able to transform this into a standardized file

@author: edm9fd
"""
from nipype.interfaces import fsl
from glob import glob
from nipype.interfaces import freesurfer
from nipype import Node, Workflow, IdentityInterface, SelectFiles
import os
from joblib import Parallel, delayed


def myelin_workflow(subject):
    subject = str(subject)
    infosource = Node(IdentityInterface(fields=['subject_id'], subject_id=subject),
                      name='infosource')
    templates = {'t1w':'/mnt/elysium/IRC805/morph/t1w/{subject_id}/{subject_id}_t1w.nii.gz', 
                 't2w':'/mnt/elysium/IRC805/morph/t2w/{subject_id}/{subject_id}_t2w.nii.gz'}
    
    selectfiles = Node(SelectFiles(templates), name="selectfiles")

    t1w_bet = Node(fsl.BET(frac=0.5,output_type='NIFTI_GZ', reduce_bias=True), name='skullstrip_t1w')
    t2w_bet = Node(fsl.BET(frac=0.7, robust=True, output_type='NIFTI_GZ'), name='skullstrip_t2w')
    
    #t2w_fast = Node(fsl.FAST(img_type=2, number_classes=4, output_type='NIFTI_GZ', segments=True), name='segment_t2w')
    #merge_segments = Node(fsl.Merge(dimension='t', output_type='NIFTI_GZ'), name='merge_segments')
    #extract_wm = Node(fsl.ExtractROI(t_min=1, t_size=1, output_type='NIFTI_GZ'), name='Extract_WM')
    #extract_gm = Node(fsl.ExtractROI(t_min=2, t_size=1, output_type='NIFTI_GZ'), name='Extract_GM')
    #add_mask = Node(fsl.BinaryMaths(operation='add', output_type='NIFTI_GZ'),'combined_GM_WM_mask')
    #mask_t2w_GWM = Node(fsl.ApplyMask(), name='mask_t2w_GWM') 
    
    
    t1w_mni_flirt = Node(fsl.FLIRT(dof=12, reference='/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz', 
                                  cost='bbr',
                                  output_type='NIFTI_GZ'), name='T1wtoMNI_flirt')
    t1w_mni_fnirt = Node(fsl.FNIRT(ref_file='/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz', 
                                  refmask_file='/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz',
                                  output_type='NIFTI_GZ',jacobian_file=True), name='T1wtoMNI_fnirt')

    
    t2w_t1w_flirt = Node(fsl.FLIRT(dof=12, cost='bbr', output_type='NIFTI_GZ'), name='T2wtoT1w_flirt')
    apply_warp = Node(fsl.ApplyWarp(ref_file='/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz'), name='T2w_to_MNI')
    
    
    apply_t1w_mask = Node(fsl.ApplyMask(mask_file='/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz'), name='masked_t1w')
    apply_t2w_mask = Node(fsl.ApplyMask(mask_file='/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz'), name='masked_t2w')
    
    t1w_t2w_map = Node(fsl.BinaryMaths(operation='div', output_type='NIFTI_GZ'), name='T1w_T2w_ratio')
    #erode_map = Node(fsl.ErodeImage(kernel_size=0.5 , kernel_shape='sphere'), name='erode_ratio_map')
    #mask_map = Node(fsl.ApplyMask(mask_file='/mnt/elysium/IRC805/mni_segment/test.nii.gz'), name='Masked_Ratio')


    wf = Workflow(name='myelin_wf_'+subject, base_dir='/mnt/elysium/IRC805/myelin')
    wf.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                (selectfiles, t1w_bet, [('t1w', 'in_file')]),
                (selectfiles, t2w_bet, [('t2w', 'in_file')]),
                
                (t1w_bet, t1w_mni_flirt, [('out_file', 'in_file')]),
                (t1w_bet, t1w_mni_fnirt, [('out_file', 'in_file')]),
                (t1w_mni_flirt, t1w_mni_fnirt, [('out_matrix_file', 'affine_file')]),


                #(t2w_bet, t2w_fast, [('out_file', 'in_files')]),
                #(t2w_fast, merge_segments, [('tissue_class_files', 'in_files')]),
                #(merge_segments, extract_gm, [('merged_file', 'in_file')]),
                #(merge_segments, extract_wm, [('merged_file', 'in_file')]),
                #(extract_gm, add_mask, [('roi_file', 'in_file')]),
                #(extract_wm, add_mask, [('roi_file', 'operand_file')]),
                #(t2w_bet, mask_t2w_GWM, [('out_file', 'in_file')]),
                #(add_mask, mask_t2w_GWM, [('out_file', 'mask_file')]),
                #(mask_t2w_GWM, 
                
                (t2w_bet, t2w_t1w_flirt, [('out_file', 'in_file')]),
                (t1w_mni_flirt, t2w_t1w_flirt, [('out_file', 'reference')]),
                
                (t2w_t1w_flirt, apply_warp, [('out_matrix_file', 'premat')]),
                (t1w_mni_fnirt, apply_warp, [('field_file', 'field_file')]),
                #(mask_t2w_GWM, apply_warp, [('out_file', 'in_file')]),
                (t2w_bet, apply_warp, [('out_file', 'in_file')]),
                (apply_warp, apply_t2w_mask, [('out_file', 'in_file')]),
                (t1w_mni_fnirt, apply_t1w_mask, [('warped_file', 'in_file')]),
                
                (apply_t2w_mask, t1w_t2w_map, [('out_file', 'operand_file')]),
                (apply_t1w_mask, t1w_t2w_map, [('out_file', 'in_file')])
                ])

    t1w_bet.inputs.in_file = '/mnt/elysium/IRC805/morph/t1w/'+subject+'/'+subject+'_t1w.nii.gz'
    t2w_bet.inputs.in_file = '/mnt/elysium/IRC805/morph/t2w/'+subject+'/'+subject+'_t2w.nii.gz'
    wf.write_graph(dotfilename='workflow_graph.dot', format='png')
    wf.run('MultiProc', plugin_args={'n_procs': 1})
    file1 = open('/mnt/elysium/IRC805/status/myelin_workflow_status.txt', 'a')
    file1.write(subject +' \n')
    file1.close()

os.chdir('/mnt/elysium/IRC805/morph/t2w/')
subject_folders = glob('*')
#subject_folders = [ 580101 ]#glob('*')

with open('/mnt/elysium/IRC805/status/myelin_workflow_status.txt') as f:
   remove_list = [line.rstrip('/n').strip() for line in f]
for item in remove_list:
    subject_folders.remove(item)

results = Parallel(n_jobs=3)(delayed(myelin_workflow)(i) for i in subject_folders)
