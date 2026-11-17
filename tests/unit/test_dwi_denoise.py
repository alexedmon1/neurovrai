"""
Unit tests for pre-eddy DWI denoising + Gibbs removal (dipy wrappers, ported from neurofaune).

Hermetic: small synthetic 4-D volumes through the real dipy routines (fast at this size). The
first three tests are neurofaune's; the rest cover neurovrai's per-series wrapper.
"""
import json

import nibabel as nib
import numpy as np
import pytest

from neurovrai.preprocess.utils.dwi_denoise import (
    degibbs_dwi,
    denoise_dwi_mppca,
    denoise_dwi_series,
    denoise_settings,
    _min_radius,
    _sigma_summary,
)


def _save(tmp_path, data, name="dwi.nii.gz", affine=None):
    affine = np.eye(4) if affine is None else affine
    f = tmp_path / name
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(f))
    return f


def _noisy_block(seed=0, shape=(20, 20, 10, 30), sd=15.0):
    rng = np.random.default_rng(seed)
    base = np.zeros(shape, dtype=np.float32)
    base[5:15, 5:15, 2:8, :] = 100.0
    return base, base + rng.normal(0, sd, base.shape).astype(np.float32)


def test_mppca_preserves_geometry_and_denoises(tmp_path):
    base, noisy = _noisy_block()
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    out = denoise_dwi_mppca(_save(tmp_path, noisy, affine=affine), tmp_path / "den.nii.gz")
    den = nib.load(str(out))
    d = den.get_fdata()
    assert d.shape == noisy.shape
    np.testing.assert_allclose(den.affine, affine)
    assert np.isfinite(d).all() and (d >= 0).all()
    roi = (slice(6, 14), slice(6, 14), slice(3, 7), slice(None))
    assert np.abs(d[roi] - base[roi]).mean() < np.abs(noisy[roi] - base[roi]).mean()


def test_degibbs_preserves_geometry(tmp_path):
    rng = np.random.default_rng(1)
    data = np.zeros((24, 24, 6, 5), dtype=np.float32)
    data[6:18, 6:18, :, :] = 80.0  # sharp box -> Gibbs-prone edges
    data += rng.normal(0, 2, data.shape).astype(np.float32)
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    out = degibbs_dwi(_save(tmp_path, data, affine=affine), tmp_path / "dg.nii.gz")
    dg = nib.load(str(out))
    g = dg.get_fdata()
    assert g.shape == data.shape
    np.testing.assert_allclose(dg.affine, affine)
    assert np.isfinite(g).all() and (g >= 0).all()


def test_degibbs_does_not_mutate_input_array(tmp_path):
    data = np.zeros((16, 16, 4, 3), dtype=np.float32)
    data[4:12, 4:12, :, :] = 50.0
    f = _save(tmp_path, data)
    before = nib.load(str(f)).get_fdata().copy()
    degibbs_dwi(f, tmp_path / "dg.nii.gz")
    np.testing.assert_array_equal(before, nib.load(str(f)).get_fdata())


def test_scaled_int16_input_keeps_its_values(tmp_path):
    # dcm2niix writes int16 with scl_slope; the float output must carry the scaled values, once
    data = np.zeros((16, 16, 4, 3), dtype=np.int16)
    data[4:12, 4:12, :, :] = 1000
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_slope_inter(0.5, 0.0)
    f = tmp_path / "scaled.nii.gz"
    nib.save(img, str(f))
    out = nib.load(str(degibbs_dwi(f, tmp_path / "dg.nii.gz")))
    assert out.get_data_dtype() == np.float32
    assert abs(out.get_fdata()[8, 8, 2, 0] - 500.0) < 5.0


def test_settings_defaults():
    assert denoise_settings({}) == {'denoise': True, 'patch_radius': 'auto',
                                    'degibbs': True, 'num_processes': 1}
    assert denoise_settings({'diffusion': {'denoise': {'patch_radius': 3}}})['patch_radius'] == 3
    assert denoise_settings({'diffusion': {'denoise': {'patch_radius': 'AUTO'}}})['patch_radius'] == 'auto'
    with pytest.raises(ValueError):
        denoise_settings({'diffusion': {'denoise': {'patch_radius': 0}}})
    s = denoise_settings({'diffusion': {'denoise': {'enabled': False}, 'degibbs': {'num_processes': 4}}})
    assert s['denoise'] is False and s['degibbs'] is True and s['num_processes'] == 4


def test_series_denoised_separately_with_record(tmp_path):
    _, a = _noisy_block(seed=2)
    _, b = _noisy_block(seed=3, shape=(20, 20, 10, 20), sd=5.0)
    fa, fb = _save(tmp_path, a, "a.nii.gz"), _save(tmp_path, b, "b.nii.gz")
    outs, rec = denoise_dwi_series([fa, fb], {}, tmp_path / "work", record_dir=tmp_path / "rec")
    assert len(outs) == 2 and all(o.exists() for o in outs)
    assert nib.load(str(outs[0])).shape == a.shape and nib.load(str(outs[1])).shape == b.shape
    saved = json.loads((tmp_path / "rec" / "denoise.json").read_text())
    assert [s['n_volumes'] for s in saved['series']] == [30, 20]
    # each series gets its own noise estimate: the second was made with a third of the noise
    sig_a, sig_b = (s['noise_sigma_median'] for s in saved['series'])
    assert sig_b < 0.6 * sig_a
    assert len(list((tmp_path / "rec").glob("*_noise_sigma.nii.gz"))) == 2
    assert not list((tmp_path / "work").glob("*_denoised.nii.gz"))  # intermediate removed
    assert rec['series'][0]['input'] == str(fa)


def test_series_reused_when_unchanged_and_redone_when_settings_change(tmp_path):
    _, a = _noisy_block(seed=4)
    fa = _save(tmp_path, a, "a.nii.gz")
    work = tmp_path / "work"
    (out,), _ = denoise_dwi_series([fa], {}, work)
    mtime = out.stat().st_mtime_ns
    (again,), _ = denoise_dwi_series([fa], {}, work)
    assert again == out and out.stat().st_mtime_ns == mtime
    cfg = {'diffusion': {'denoise': {'patch_radius': 1}}}
    (redone,), rec = denoise_dwi_series([fa], cfg, work)
    assert redone.stat().st_mtime_ns != mtime and rec['denoise']['patch_radius'] == 1


def test_disabled_passes_inputs_through(tmp_path):
    f = _save(tmp_path, np.ones((4, 4, 4, 2)))
    cfg = {'diffusion': {'denoise': {'enabled': False}, 'degibbs': {'enabled': False}}}
    outs, rec = denoise_dwi_series([f], cfg, tmp_path / "work")
    assert outs == [f] and not (tmp_path / "work").exists()
    assert rec['denoise']['enabled'] is False


def test_auto_radius_follows_dipys_sample_rule():
    # dipy needs (2r+1)^3 - 1 >= volumes; IRC805's b3000 series has 125 volumes -> radius 3
    assert _min_radius(26) == 1 and _min_radius(27) == 2
    assert _min_radius(95) == 2 and _min_radius(124) == 2 and _min_radius(125) == 3
    assert _min_radius(342) == 3 and _min_radius(343) == 4


def test_auto_radius_recorded_per_series(tmp_path):
    _, a = _noisy_block(seed=5, shape=(20, 20, 10, 20))
    _, b = _noisy_block(seed=6, shape=(20, 20, 10, 30))
    outs, rec = denoise_dwi_series([_save(tmp_path, a, "a.nii.gz"), _save(tmp_path, b, "b.nii.gz")],
                                   {'diffusion': {'degibbs': {'enabled': False}}}, tmp_path / "work")
    assert rec['denoise']['patch_radius'] == 'auto'
    assert [s['patch_radius'] for s in rec['series']] == [1, 2]
    assert all(s['noise_sigma_missing_fraction'] < 0.01 for s in rec['series'])


def test_sigma_summary_ignores_missing_estimates(tmp_path):
    dwi = np.zeros((10, 10, 4, 3), dtype=np.float32)
    dwi[2:8, 2:8, :, :] = 100.0
    sigma = np.full((10, 10, 4), np.nan, dtype=np.float32)
    sigma[2:8, 2:8, :] = 7.0
    sigma[2, 2:8, :] = np.nan  # 1 of 6 brain rows without an estimate
    out = _sigma_summary(_save(tmp_path, sigma, "sigma.nii.gz"), _save(tmp_path, dwi, "dwi.nii.gz"))
    assert out['noise_sigma_median'] == pytest.approx(7.0)
    assert out['noise_sigma_missing_fraction'] == pytest.approx(1 / 6, abs=1e-3)
