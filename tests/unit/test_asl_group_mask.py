"""The ASL group mask is brain-limited and ignores interpolation residue."""

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np

_spec = importlib.util.spec_from_file_location(
    'run_asl_group_analysis',
    Path(__file__).resolve().parents[2] / 'scripts' / 'analysis' / 'run_asl_group_analysis.py')
group = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(group)


def test_mask_drops_residue_and_non_brain(tmp_path):
    shape = (10, 10, 10)
    brain = np.zeros(shape, np.uint8)
    brain[2:8, 2:8, 2:8] = 1
    data = np.full(shape + (4,), 1e-3, np.float32)        # residue everywhere
    data[3:7, 3:7, 3:7, :] = 50.0                           # real CBF inside the brain
    data[0, 0, 0, :] = 50.0                                 # outside the brain
    nib.save(nib.Nifti1Image(brain, np.eye(4)), str(tmp_path / 'brain.nii.gz'))
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(tmp_path / 'all.nii.gz'))
    mask = nib.load(group.create_mean_mask(tmp_path / 'all.nii.gz', tmp_path,
                                           brain_mask=tmp_path / 'brain.nii.gz')).get_fdata()
    assert mask.sum() == 4 ** 3
    assert mask[0, 0, 0] == 0
