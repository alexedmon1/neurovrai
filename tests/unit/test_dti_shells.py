"""The tensor is fitted on b=0 plus shells up to diffusion.dti.max_bval only."""

import nibabel as nib
import numpy as np
import pytest

from neurovrai.preprocess.utils.dti_shells import (
    dti_shell_settings, dti_volume_index, select_dti_volumes)


def _series(tmp_path, bvals):
    bvals = np.asarray(bvals, dtype=float)
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(3, bvals.size))
    vecs /= np.linalg.norm(vecs, axis=0)
    data = np.arange(2 * 2 * 2 * bvals.size, dtype=np.float32).reshape(2, 2, 2, bvals.size)
    dwi = tmp_path / 'dwi.nii.gz'
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(dwi))
    np.savetxt(tmp_path / 'dwi.bval', bvals[None, :], fmt='%g')
    np.savetxt(tmp_path / 'dwi.bvec', vecs, fmt='%.6f')
    return dwi, tmp_path / 'dwi.bval', tmp_path / 'dwi.bvec', data, vecs


def test_index_keeps_b0_and_b1000_only():
    b = [0.0001] * 5 + [1000] * 30 + [2000] * 60 + [0] * 5 + [3000] * 120
    keep = dti_volume_index(b)
    assert keep.size == 40
    assert set(np.asarray(b)[keep].round()) == {0, 1000}


def test_tolerance_and_too_few_directions():
    assert dti_volume_index([0, 1005, 995, 1000, 1000, 1000, 1000]).size == 7
    with pytest.raises(ValueError, match='at least 6'):
        dti_volume_index([0] * 5 + [3000] * 120)
    assert dti_volume_index([0] * 5 + [3000] * 120, max_bval=3000).size == 125


def test_select_writes_subset_and_records_bvals(tmp_path):
    bvals = [0] + [1000] * 6 + [2000] * 4 + [800] * 2
    dwi, bval, bvec, data, vecs = _series(tmp_path, bvals)
    out = tmp_path / 'out'
    d, bv, bc = select_dti_volumes(dwi, bval, bvec, out)
    keep = [0, 1, 2, 3, 4, 5, 6, 11, 12]
    assert np.array_equal(nib.load(d).get_fdata(), data[..., keep])
    assert np.array_equal(np.loadtxt(bv), np.asarray(bvals, float)[keep])
    assert np.allclose(np.loadtxt(bc), vecs[:, keep], atol=1e-6)


def test_single_shell_b800_passes_through(tmp_path):
    dwi, bval, bvec, _, _ = _series(tmp_path, [0] + [800] * 32)
    d, bv, _ = select_dti_volumes(dwi, bval, bvec, tmp_path / 'out')
    assert d == str(dwi)
    assert np.loadtxt(bv).size == 33


def test_settings_from_config():
    assert dti_shell_settings({}) == (1000.0, 50.0)
    assert dti_shell_settings({'diffusion': {'dti': {'max_bval': 1500}}}) == (1500.0, 50.0)


def test_workflow_fits_tensor_on_selected_shells(tmp_path):
    from neurovrai.preprocess.workflows.dwi_preprocess import create_dwi_preprocessing_workflow
    for f in ('d.nii.gz', 'd.bval', 'd.bvec'):
        (tmp_path / f).write_bytes(b'')
    wf = create_dwi_preprocessing_workflow(
        config={'paths': {'logs': str(tmp_path / 'logs')}}, subject='sub-x', dwi_file=tmp_path / 'd.nii.gz', bval_file=tmp_path / 'd.bval',
        bvec_file=tmp_path / 'd.bvec', output_dir=tmp_path / 'out', work_dir=tmp_path / 'work')
    g = wf._graph
    names = {n.name: n for n in g.nodes()}
    dtifit, select = names['dtifit'], names['select_dti_shells']
    fed = {c[1] for u, _v, e in g.in_edges(dtifit, data=True) if u.name == 'select_dti_shells'
           for c in e['connect']}
    assert fed == {'dwi', 'bvals', 'bvecs'}
    assert not any(u.name == 'inputnode' for u, _ in g.in_edges(dtifit))
    assert select.inputs.max_bval == 1000.0
