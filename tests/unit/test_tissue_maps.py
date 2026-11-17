"""Tissue maps are found by name, never by Atropos output position."""

import logging

import pytest

from neurovrai.utils.tissue_maps import find_tissue_map, find_tissue_maps


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b'')


def test_named_maps_win_over_posteriors(tmp_path):
    _touch(tmp_path, 'POSTERIOR_01.nii.gz', 'POSTERIOR_02.nii.gz', 'POSTERIOR_03.nii.gz',
           'csf.nii.gz', 'gm.nii.gz', 'wm.nii.gz')
    found = find_tissue_maps(tmp_path)
    assert {t: p.name for t, p in found.items()} == {
        'csf': 'csf.nii.gz', 'gm': 'gm.nii.gz', 'wm': 'wm.nii.gz'}


def test_fast_order_is_fixed(tmp_path):
    _touch(tmp_path, 't1w_brain_pve_0.nii.gz', 't1w_brain_pve_1.nii.gz', 't1w_brain_pve_2.nii.gz')
    assert find_tissue_map(tmp_path, 'csf').name == 't1w_brain_pve_0.nii.gz'
    assert find_tissue_map(tmp_path, 'GM').name == 't1w_brain_pve_1.nii.gz'
    assert find_tissue_map(tmp_path, 'wm').name == 't1w_brain_pve_2.nii.gz'


def test_unnamed_posteriors_are_never_used(tmp_path, caplog):
    _touch(tmp_path, 'POSTERIOR_01.nii.gz', 'POSTERIOR_02.nii.gz', 'POSTERIOR_03.nii.gz')
    with caplog.at_level(logging.WARNING):
        assert find_tissue_map(tmp_path, 'wm') is None
    assert 'standardize_atropos_tissues' in caplog.text


def test_missing_dir_and_bad_tissue(tmp_path):
    assert find_tissue_map(tmp_path / 'nope', 'wm') is None
    with pytest.raises(ValueError):
        find_tissue_map(tmp_path, 'bone')
