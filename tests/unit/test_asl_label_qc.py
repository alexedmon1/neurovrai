"""An ASL run whose labelling had no effect is caught, and run choice is deterministic."""

import nibabel as nib
import numpy as np

from neurovrai.preprocess.utils.asl_cbf import choose_asl_run, label_signal_qc


def _pairs(dm_fraction, n_pairs=33, seed=0, noise=2.0):
    rng = np.random.default_rng(seed)
    tissue = np.full((6, 6, 4), 1000.0)
    tissue[:2] = 50.0  # dark background
    control = tissue[..., None] + rng.normal(0, noise, (6, 6, 4, n_pairs))
    label = control - dm_fraction * tissue[..., None] + rng.normal(0, noise, (6, 6, 4, n_pairs))
    return control, label


def test_normal_run_passes():
    qc = label_signal_qc(*_pairs(0.008))
    assert qc['passed'] and 0.007 < qc['label_signal'] < 0.009 and qc['positive_pairs'] == 1.0


def test_null_label_fails():
    qc = label_signal_qc(*_pairs(0.0, noise=20.0))
    assert not qc['passed'] and 'little or no effect' in qc['reason']


def test_reversed_order_fails():
    qc = label_signal_qc(*_pairs(-0.008))
    assert not qc['passed'] and 'reversed' in qc['reason']


def _write_run(path, dm_fraction, seed):
    control, label = _pairs(dm_fraction, seed=seed, noise=20.0)
    series = np.empty(control.shape[:3] + (2 * control.shape[3],), np.float32)
    series[..., 0::2], series[..., 1::2] = control, label   # control first
    nib.save(nib.Nifti1Image(series, np.eye(4)), str(path))


def test_choose_skips_failed_first_run(tmp_path):
    _write_run(tmp_path / '1204_SOURCE_pCASL.nii.gz', 0.005, 1)
    _write_run(tmp_path / '1004_SOURCE_pCASL.nii.gz', 0.0, 2)
    chosen, report = choose_asl_run([tmp_path / '1204_SOURCE_pCASL.nii.gz',
                                     tmp_path / '1004_SOURCE_pCASL.nii.gz'])
    assert chosen.name == '1204_SOURCE_pCASL.nii.gz'
    assert list(report) == ['1004_SOURCE_pCASL.nii.gz', '1204_SOURCE_pCASL.nii.gz']


def test_choose_returns_none_when_all_fail(tmp_path):
    _write_run(tmp_path / '1004_SOURCE_pCASL.nii.gz', 0.0, 3)
    assert choose_asl_run([tmp_path / '1004_SOURCE_pCASL.nii.gz'])[0] is None
