#!/usr/bin/env python3
"""
Structural -> Functional coregistration
Merge functional files
Functional preprocessing
"""

import os
from nipype import Workflow, Node, MapNode, JoinNode, SelectFiles, IdentityInterface
from nipype.interfaces import fsl
from nipype.interfaces.utility.base import Merge
from nipype.interfaces import io
from nipype.algorithms import confounds
from nipype.interfaces import afni
from glob import glob
import yaml
import sys
from joblib import Parallel, delayed


tr = 2.00001
vols = 165
lowpass = 0.08
highpass = 0.001
fwhm = 6
output_type ='NIFTI_GZ'
reference_file = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'

def rest_preproc(t1w_image, rest_image, base_dir):


    #   Structural Nodes
    t1w_reorient = Node(fsl.Reorient2Std(in_file=t1w_image, output_type=output_type), name="str_reorient")
    t1w_fast = Node(fsl.FAST(output_biascorrected=True, img_type=1,
                             output_type=output_type), name='str_fast')
    t1w_bet = Node(fsl.BET(frac=0.5, reduce_bias=True, mask=True, output_type=output_type), name='str_bet')
    t1w_segment = Node(fsl.FAST(number_classes=3, segments=True, img_type=1, no_bias=True, output_type=output_type),
                       name='str_segment')
    t1w_rigid = Node(fsl.FLIRT(dof=6, cost_func='bbr', reference=reference_file,
                               output_type=output_type), name='str_rigid')
    t1w_trans = Node(fsl.FLIRT(dof=12, cost_func='bbr', reference=reference_file,
                               output_type=output_type), name='str_trans')
    merge_segments = Node(fsl.Merge(dimension='t', tr=tr, output_type=output_type), name='merge_segments')
    extract_wm = Node(fsl.ExtractROI(t_min=2, t_size=1, output_type=output_type), name='Extract_WM')
    extract_csf = Node(fsl.ExtractROI(t_min=0, t_size=1, output_type=output_type), name='Extract_CSF')
    erode_mask = Node(fsl.ErodeImage(output_type=output_type), name='Binary_Erosion')
    create_motion_mask = Node(fsl.ImageMaths(op_string='-add', output_type=output_type), name='Create_Motion_Mask')

    #   Functional Nodes
    rs_reorient = Node(fsl.Reorient2Std(in_file=rest_image, output_type=output_type), name='func_reorient')
    rs_bet = Node(fsl.BET(frac=0.3, functional=True, output_type=output_type, mask=True), name='func_bet')
    rs_moco = Node(fsl.MCFLIRT(cost='leastsquares', save_plots=True, dof=6, stages=4, interpolation='sinc', output_type=output_type),
                   name='func_moco')
    rs_1000 = Node(fsl.maths.MathsCommand(args='-ing 1000', output_type=output_type), name='normalize_1000')
    rs_extract_tr = Node(fsl.ExtractROI(t_min=0, t_size=1, output_type=output_type), name='extract_single_tr_image')
    rs_func2subj = Node(fsl.FLIRT(dof=6, cost_func='mutualinfo', output_type=output_type), name='func2subj')
    rs_combine1 = Node(fsl.ConvertXFM(concat_xfm=True, output_type=output_type), name='Combine_Matrix1')
    rs_combine2 = Node(fsl.ConvertXFM(concat_xfm=True, output_type=output_type), name='Combine_Matrix2')
    rs_split = Node(fsl.utils.Split(dimension='t', output_type=output_type, out_base_name='a'), name='func_split')
    rs_trans = MapNode(fsl.ApplyXFM(reference=reference_file, apply_xfm=True, output_type=output_type), name='func_applyXFM', iterfield=['in_file'])
    merge_files = Node(Merge(vols), iterfield=['in1'], name='merge_filenames')
    rs_merge = Node(interface=fsl.Merge(dimension='t', tr=tr, output_type=output_type), name='func_merge')
    rs_aroma = Node(fsl.ICA_AROMA(denoise_type='both', TR=tr), name='func_aroma')
    rs_smooth = Node(fsl.utils.Smooth(fwhm=fwhm, output_type=output_type), name='func_smooth')
    rs_mask = Node(fsl.ApplyMask(mask_file='/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz', output_type=output_type),
                   name="func_mask")
    rs_bandpass = Node(afni.Bandpass(highpass=highpass, lowpass=lowpass, tr=tr, outputtype=output_type), name="func_bandpass")
    acompcor = Node(confounds.ACompCor(num_components=6, repetition_time=tr), name='ACompCor')
    acompcor_glm = Node(fsl.GLM(output_type=output_type, out_res_name='acompcor.nii.gz'), name='ACompCor_GLM')

    #   Workflow
    wf = Workflow(name='rest_preprocess', base_dir=base_dir)
    wf.connect([(t1w_reorient, t1w_fast, [('out_file', 'in_files')]),
                (t1w_fast, t1w_bet, [('restored_image', 'in_file')]),
                (t1w_bet, t1w_rigid, [('out_file', 'in_file')]),
                (t1w_rigid, t1w_trans, [('out_file', 'in_file')]),
                (t1w_trans, t1w_segment, [('out_file', 'in_files')]),
                (t1w_segment, merge_segments, [('tissue_class_files', 'in_files')]),
                (merge_segments, extract_csf, [('merged_file', 'in_file')]),
                (merge_segments, extract_wm, [('merged_file', 'in_file')]),
                (extract_wm, create_motion_mask, [('roi_file', 'in_file')]),
                (extract_csf, create_motion_mask, [('roi_file', 'in_file2')]),
                (create_motion_mask, erode_mask, [('out_file', 'in_file')]),

                
                (rs_reorient, rs_bet, [('out_file', 'in_file')]),
                (rs_bet, rs_moco, [('out_file', 'in_file')]),
                (rs_moco, rs_smooth, [('out_file', 'in_file')]),
                (rs_smooth, rs_1000, [('smoothed_file', 'in_file')]),
                (rs_1000, rs_aroma, [('out_file', 'in_file')]),
                (rs_moco, rs_aroma, [('par_file', 'motion_parameters')]),
                (rs_bet, rs_aroma, [('mask_file', 'mask')]),
                (rs_aroma, rs_extract_tr, [('aggr_denoised_file', 'in_file')]),
                (rs_extract_tr, rs_func2subj, [('roi_file', 'in_file')]),
                (t1w_bet, rs_func2subj, [('out_file', 'reference')]),
                (rs_func2subj, rs_combine1, [('out_matrix_file', 'in_file')]),
                (t1w_rigid, rs_combine1, [('out_matrix_file', 'in_file2')]),
                (rs_combine1, rs_combine2, [('out_file', 'in_file')]),
                (t1w_trans, rs_combine2, [('out_matrix_file', 'in_file2')]),
                (rs_aroma, rs_split, [('aggr_denoised_file', 'in_file')]),
                (rs_split, rs_trans, [('out_files', 'in_file')]),
                (rs_combine2, rs_trans, [('out_file', 'in_matrix_file')]),
                (rs_trans, merge_files, [('out_file', 'in1')]),
                (merge_files, rs_merge, [('out', 'in_files')]),
                (rs_merge, acompcor, [('merged_file', 'realigned_file')]),
                (erode_mask, acompcor, [('out_file', 'mask_files')]),
                (rs_merge, rs_mask, [('merged_file', 'in_file')]),
                (rs_mask, rs_bandpass, [('out_file', 'in_file')]),
                (rs_bandpass, acompcor_glm, [('out_file', 'in_file')]),
                (acompcor, acompcor_glm, [('components_file', 'design')])])

    wf.write_graph(dotfilename='workflow_graph.dot', format='png')
    wf.run('MultiProc', plugin_args={'n_procs': 6})


def subject_name(z):
    fs = z.split('/')
    return fs[-2]


"""
Run from Project Folder
Code for running pipeline
Loops through subjects in one folder
"""


sub_directory = '/mnt/elysium/IRC805/heudiconv/'
basedir = '/mnt/elysium/IRC805/rest/'
subject_list = glob(sub_directory + '*/')

with open('/mnt/elysium/IRC805/status/irc805_rest_status.txt') as f:
    remove_list = [line.rstrip('/n').strip() for line in f]
    
for item in remove_list:
    subject_list.remove(item)

def preprocessing(x):
    sub = subject_name(x)
    rest = '/mnt/elysium/IRC805/heudiconv/'+str(sub)+'/func/sub-'+str(sub)+'_task-rest_bold_single_run01.nii.gz'
    t1w_file = '/mnt/elysium/IRC805/heudiconv/'+str(sub)+'/anat/sub-'+str(sub)+'_t1w.nii.gz'
    print('Processing ' + sub + ' . . .')
    if os.path.exists(rest):
        rest_preproc(t1w_file, rest, basedir+str(sub))
        print('Done!')
        with open('/mnt/elysium/IRC805/status/irc805_rest_status.txt', 'a') as file1:
            file1.write(x +' \n')
    else:
        with open('/mnt/elysium/IRC805/status/irc805_rest_status.txt', 'a') as file1:
            file1.write(x +' \n')

    
    
results = Parallel(n_jobs=2)(delayed(preprocessing)(i) for i in subject_list)



