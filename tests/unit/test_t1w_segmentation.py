"""Unit tests for T1w segmentation method choice, FAST naming, superseding and the record.

Hermetic: no FSL or ANTs is run; outputs are stand-in files with the names each tool writes.
"""
import json

import nibabel as nib
import numpy as np
import pytest

from neurovrai.preprocess.utils.t1w_segmentation import (
    detect_segmentation_method,
    fast_settings,
    name_fast_tissues,
    prepare_segmentation_dir,
    segmentation_method,
    supersede_segmentation,
    write_segmentation_record,
)
from neurovrai.utils.tissue_maps import find_tissue_map

STEM = '201_T1_reoriented_brain_corrected'


def _img(path, value=1.0):
    nib.save(nib.Nifti1Image(np.full((3, 3, 3), value, np.float32), np.eye(4)), str(path))
    return path


def _fast_outputs(seg_dir):
    seg_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        _img(seg_dir / f'{STEM}_pve_{i}.nii.gz', value=i)
    _img(seg_dir / f'{STEM}_seg.nii.gz')


def _atropos_outputs(seg_dir):
    seg_dir.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):
        _img(seg_dir / f'POSTERIOR_{i:02d}.nii.gz', value=10 + i)
    _img(seg_dir / f'{STEM}_labeled.nii.gz')
    # the arbitrary K-means order: gm points at the wrong posterior in a failed run
    (seg_dir / 'gm.nii.gz').symlink_to('POSTERIOR_03.nii.gz')
    (seg_dir / 'wm.nii.gz').symlink_to('POSTERIOR_02.nii.gz')
    (seg_dir / 'csf.nii.gz').symlink_to('POSTERIOR_01.nii.gz')


@pytest.mark.parametrize('raw,expected', [
    (None, 'ants'), ('ants', 'ants'), ('Atropos', 'ants'), ('fsl', 'fsl'), ('FAST', 'fsl'),
])
def test_segmentation_method(raw, expected):
    cfg = {} if raw is None else {'anatomical': {'segmentation_method': raw}}
    assert segmentation_method(cfg) == expected


def test_unknown_method_is_an_error():
    with pytest.raises(ValueError):
        segmentation_method({'anatomical': {'segmentation_method': 'spm'}})


def test_fast_settings_defaults():
    assert fast_settings({}) == {'img_type': 1, 'number_classes': 3}


def test_name_fast_tissues_fixed_order(tmp_path):
    seg = tmp_path / 'segmentation'
    _fast_outputs(seg)
    named = name_fast_tissues(seg)
    assert {t: p.resolve().name for t, p in named.items()} == {
        'csf': f'{STEM}_pve_0.nii.gz', 'gm': f'{STEM}_pve_1.nii.gz', 'wm': f'{STEM}_pve_2.nii.gz'}
    assert find_tissue_map(seg, 'gm') == seg / 'gm.nii.gz'
    name_fast_tissues(seg)  # again: replaces the links, no error
    assert (seg / 'wm.nii.gz').resolve().name == f'{STEM}_pve_2.nii.gz'


def test_detect_method(tmp_path):
    a, f, e = tmp_path / 'a', tmp_path / 'f', tmp_path / 'e'
    _atropos_outputs(a)
    _fast_outputs(f)
    e.mkdir()
    assert detect_segmentation_method(a) == 'ants'
    assert detect_segmentation_method(f) == 'fsl'
    assert detect_segmentation_method(e) is None


def test_method_change_supersedes_old_outputs(tmp_path):
    seg = tmp_path / 'segmentation'
    _atropos_outputs(seg)
    dest = prepare_segmentation_dir(seg, 'fsl')
    assert dest == seg / 'superseded_ants'
    assert not [p for p in seg.iterdir() if p.is_file() or p.is_symlink()]  # nothing stale on top
    assert (dest / 'POSTERIOR_03.nii.gz').exists() and (dest / 'SUPERSEDED.txt').exists()
    assert (dest / 'gm.nii.gz').is_symlink()  # links kept as links, still resolving
    assert (dest / 'gm.nii.gz').resolve().name == 'POSTERIOR_03.nii.gz'
    assert find_tissue_map(seg, 'gm') is None  # the superseded gm is never found


def test_same_method_keeps_outputs(tmp_path):
    seg = tmp_path / 'segmentation'
    _fast_outputs(seg)
    assert prepare_segmentation_dir(seg, 'fsl') is None
    assert len(list(seg.glob('*pve_*.nii.gz'))) == 3


def test_second_supersede_gets_its_own_folder(tmp_path):
    seg = tmp_path / 'segmentation'
    _fast_outputs(seg)
    first = supersede_segmentation(seg, 'one')
    _fast_outputs(seg)
    second = supersede_segmentation(seg, 'two')
    assert first != second and first.exists() and second.name.startswith('superseded_fsl_')


def test_record(tmp_path):
    seg = tmp_path / 'segmentation'
    _fast_outputs(seg)
    name_fast_tissues(seg)
    cfg = {'anatomical': {'segmentation_method': 'fsl'}}
    rec = json.loads(write_segmentation_record(seg, 'fsl', cfg, tmp_path / 'in.nii.gz').read_text())
    assert rec['method'] == 'fsl' and rec['tool'] == 'FSL FAST'
    assert rec['settings'] == {'img_type': 1, 'number_classes': 3}
    assert rec['tissues']['gm'] == f'{STEM}_pve_1.nii.gz'
    assert detect_segmentation_method(seg) == 'fsl'
