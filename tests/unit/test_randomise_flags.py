"""The randomise command each analysis builds: TFCE type, variance smoothing, --uncorrp.

FSL is not needed: subprocess.run is replaced by a recorder.
"""
import subprocess
from pathlib import Path

import pytest

from neurovrai.analysis.func import batch_reho_falff_analysis as rf
from neurovrai.analysis.stats import randomise_wrapper as rw


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(rw, "check_fsl_installation", lambda: True)
    monkeypatch.setattr(rw, "validate_inputs", lambda *a, **k: None)
    monkeypatch.setattr(rw.subprocess, "run", fake_run)
    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    return calls


def _wrapper(tmp_path, **kwargs):
    return rw.run_randomise(
        input_file=tmp_path / "all.nii.gz", design_mat=tmp_path / "design.mat",
        contrast_con=tmp_path / "design.con", output_dir=tmp_path / "out", **kwargs)


def test_wrapper_defaults_to_3d_tfce_with_uncorrected_p(tmp_path, recorded):
    result = _wrapper(tmp_path)
    cmd = recorded[0]
    assert "-T" in cmd and "--T2" not in cmd
    assert "--uncorrp" in cmd
    assert "-v" not in cmd
    assert result["tfce_2d"] is False and result["command"] == cmd


def test_wrapper_2d_tfce_for_skeletons(tmp_path, recorded):
    _wrapper(tmp_path, tfce_2d=True)
    assert "--T2" in recorded[0] and "-T" not in recorded[0]


def test_wrapper_without_uncorrected_p(tmp_path, recorded):
    _wrapper(tmp_path, uncorrected_p=False)
    assert "--uncorrp" not in recorded[0]


def _reho(tmp_path, **kwargs):
    return rf.run_randomise(
        input_4d=tmp_path / "all.nii.gz", output_prefix=tmp_path / "out" / "reho",
        design_mat=tmp_path / "design.mat", design_con=tmp_path / "design.con",
        mask=tmp_path / "mask.nii.gz", **kwargs)


def test_reho_falff_keeps_5mm_variance_smoothing_by_default(tmp_path, recorded):
    assert _reho(tmp_path)
    cmd = recorded[0]
    assert cmd[cmd.index("-v") + 1] == "5.0"
    assert "-T" in cmd and "--T2" not in cmd and "--uncorrp" in cmd


def test_reho_falff_plain_t_when_smoothing_is_none(tmp_path, recorded):
    _reho(tmp_path, variance_smoothing=None)
    assert "-v" not in recorded[0]
