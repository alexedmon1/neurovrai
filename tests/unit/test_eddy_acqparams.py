"""eddy gets --acqp and --index whether or not TOPUP ran.

The multi-shell workflow only wrote these files on the TOPUP path, so the
no-TOPUP branch handed eddy a config with neither key set and the flags were
silently omitted. Documented as an open bug since 2025-11-24.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurovrai.preprocess.utils.topup_helper import ensure_eddy_acqparams

pytestmark = pytest.mark.unit


def _dwi(tmp_path: Path, n_volumes: int = 7) -> Path:
    path = tmp_path / "dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, n_volumes), dtype=np.float32), np.eye(4)), path)
    return path


def test_writes_one_acqparams_line_and_an_index_per_volume(tmp_path):
    config = {}
    acqparams, index = ensure_eddy_acqparams(config, _dwi(tmp_path, 7), tmp_path / "work")

    assert acqparams.read_text().split() == ["0", "-1", "0", "0.050000"], "AP at 0.05s by default"
    assert index.read_text().split() == ["1"] * 7, "every volume maps to the single acquisition"


def test_the_config_carries_the_paths_create_eddy_node_reads(tmp_path):
    config = {}
    acqparams, index = ensure_eddy_acqparams(config, _dwi(tmp_path), tmp_path / "work")

    assert config["diffusion"]["eddy"]["acqp_file"] == str(acqparams)
    assert config["diffusion"]["eddy"]["index_file"] == str(index)


def test_direction_and_readout_come_from_the_study_config(tmp_path):
    config = {"diffusion": {"eddy": {"phase_encoding_direction": "PA", "readout_time": 0.062}}}
    acqparams, _ = ensure_eddy_acqparams(config, _dwi(tmp_path), tmp_path / "work")

    assert acqparams.read_text().split() == ["0", "1", "0", "0.062000"]


def test_readout_time_falls_back_to_the_topup_setting(tmp_path):
    config = {"diffusion": {"topup": {"readout_time": 0.037}}}
    acqparams, _ = ensure_eddy_acqparams(config, _dwi(tmp_path), tmp_path / "work")

    assert acqparams.read_text().split()[-1] == "0.037000"


def test_a_null_acqp_file_is_not_a_path(tmp_path):
    """configs/default.yaml ships acqp_file: null, and `in eddy_config` was true
    for it -- so eddy was handed in_acqp=None rather than nothing at all."""
    from neurovrai.preprocess.workflows.dwi_preprocess import create_eddy_node

    node = create_eddy_node({"diffusion": {"eddy": {"acqp_file": None, "index_file": None}}})

    assert not isinstance(node.inputs.in_acqp, str)
    assert not isinstance(node.inputs.in_index, str)
