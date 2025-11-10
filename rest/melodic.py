from nipype.interfaces import fsl
import os
from glob import glob

os.chdir('/mnt/elysium/IRC805/rest')

rsfiles = glob('/mnt/elysium/IRC805/rest/*/rest_preprocess/func_bandpass/a0000_flirt_merged_masked_bp.nii.gz')
#bad_list = ['/mnt/arcadia/CARES/processed_data/resting_state/CARES070/rest_preproc/final_merge/acompcor_smooth_masked_filt_merged.nii.gz']
#for i in bad_list:
#    rsfiles.remove(i)
#rsfiles.sort()

with open('/mnt/elysium/IRC805/rest/melodic_list.txt', 'w') as fileopen:
    for subject in rsfiles:
        fileopen.write('%s\n' % subject)


mel = fsl.MELODIC()
mel.inputs.approach = 'concat'
mel.inputs.in_files = rsfiles
mel.inputs.no_bet = True
mel.inputs.tr_sec = 1.029047
mel.inputs.report = True
mel.inputs.out_dir = 'MelodicICA'
mel.inputs.out_all = True
mel.inputs.dim = 20
mel.inputs.sep_vn = True
mel.inputs.mask = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
mel.inputs.bg_image = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
mel.inputs.args = '--verbose'
#--migpN=400 --migp_factor=2 --sep_vn
mel.run()
