"""Minimum detectable effect: the helper, and the two tabular outputs that carry it.

The reference values are shared with neurofaune and source-analytics, so the
three tools agree on one number for one design.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from neurovrai.analysis.stats.power import (
    MDE_ALPHA,
    MDE_POWER,
    NBS_NOTE,
    independent_se_df,
    mde_fields,
    mde_percent,
    minimum_detectable_effect,
    paired_se_df,
    welch_se_df,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "se, df, expected",
    [
        (1.0, 10, 3.107197),
        (1.0, 26, 2.911195),
        (2.09, 24, 6.104376),
        (1.0, np.inf, 2.801585),
        (0.5, 4, 1.858705),
    ],
)
def test_reference_values(se, df, expected):
    assert minimum_detectable_effect(se, df) == pytest.approx(expected, abs=1e-6)


def test_missing_df_uses_normal_quantiles():
    assert minimum_detectable_effect(1.0, None) == pytest.approx(2.801585, abs=1e-6)


def test_elementwise_on_arrays_with_nan_and_inf():
    out = minimum_detectable_effect(np.array([1.0, np.nan, 1.0]), np.array([10.0, 10.0, np.inf]))
    assert out[0] == pytest.approx(3.107197, abs=1e-6)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.801585, abs=1e-6)


def test_welch_se_and_df():
    se, df = welch_se_df(2.0, 10, 3.0, 12)
    va, vb = 4.0 / 10, 9.0 / 12
    assert se == pytest.approx(np.sqrt(va + vb))
    assert df == pytest.approx((va + vb) ** 2 / (va**2 / 9 + vb**2 / 11))


def test_independent_se_matches_the_t_test_it_belongs_to():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 9), rng.normal(8, 3, 14)
    for equal_var in (True, False):
        se, df = independent_se_df(a, b, equal_var=equal_var)
        t = stats.ttest_ind(a, b, equal_var=equal_var).statistic
        assert (a.mean() - b.mean()) / se == pytest.approx(t)
    assert independent_se_df(a, b, equal_var=True)[1] == 21.0


def test_paired_se_matches_ttest_rel():
    rng = np.random.default_rng(1)
    a = rng.normal(5, 1, 12)
    b = a + rng.normal(0.3, 0.5, 12)
    se, df = paired_se_df(a, b)
    assert df == 11.0
    assert (a.mean() - b.mean()) / se == pytest.approx(stats.ttest_rel(a, b).statistic)


def test_mde_percent_blank_without_a_usable_reference():
    assert mde_percent(2.0, 50.0) == pytest.approx(4.0)
    assert mde_percent(2.0, -50.0) == pytest.approx(4.0)
    assert np.isnan(mde_percent(2.0, None))
    assert np.isnan(mde_percent(2.0, 0.0))


def test_mde_percent_is_blank_within_one_sd_of_zero():
    """The rule shared with neurofaune and source-analytics: |mean| < sd."""
    assert np.isnan(mde_percent(2.0, 4.9, sd=5.0))
    assert mde_percent(2.0, 5.0, sd=5.0) == pytest.approx(40.0)
    assert mde_percent(2.0, -5.1, sd=5.0) == pytest.approx(2.0 / 5.1 * 100)
    assert np.isnan(mde_fields(1.0, 26, reference_mean=0.5, sd=1.0)["mde_percent"])


def test_mde_fields_carry_the_contract():
    row = mde_fields(1.0, 10, reference_mean=31.07197)
    assert set(row) == {"se", "df", "mde", "mde_percent", "mde_power", "mde_alpha", "mde_note"}
    assert row["mde"] == pytest.approx(3.107197, abs=1e-6)
    assert row["mde_percent"] == pytest.approx(10.0, abs=1e-4)
    assert (row["mde_power"], row["mde_alpha"], row["mde_note"]) == (0.8, 0.05, "")


@pytest.mark.parametrize("se, df", [(np.nan, 10), (None, None), (0.0, 10)])
def test_mde_is_never_blank_without_a_note(se, df):
    row = mde_fields(se, df, reference_mean=5.0)
    assert np.isnan(row["mde"])
    assert row["mde_note"].startswith("not computed: ")


def test_a_single_subject_group_gets_a_note_not_a_number():
    se, df = independent_se_df(np.array([1.0]), np.array([2.0, 3.0, 4.0]))
    assert mde_fields(se, df)["mde_note"].startswith("not computed: ")


# --------------------------------------------------------------------------
# WMH group comparison CSV
# --------------------------------------------------------------------------


def _load_wmh_script():
    spec = importlib.util.spec_from_file_location(
        "wmh_group_comparison", REPO / "scripts" / "wmh_group_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wmh_comparison_writes_the_mde_columns(tmp_path):
    wmh = _load_wmh_script()
    rng = np.random.default_rng(2)
    n = 20
    wmh_df = pd.DataFrame(
        {
            "subject_id": range(n),
            "subject": [f"STUDY-{i}" for i in range(n)],
            "n_lesions": rng.integers(0, 10, n),
            "total_volume_mm3": rng.normal(500, 100, n),
            "mean_lesion_volume": rng.normal(50, 10, n),
            "max_lesion_volume": np.zeros(n),  # no variance: must be noted, not numbered
        }
    )
    glu_df = pd.DataFrame({"Subject": range(n), "mriglu": [1] * 8 + [2] * 12})

    results, _ = wmh.compare_groups(wmh_df, glu_df, tmp_path)
    written = pd.read_csv(tmp_path / "wmh_group_comparison.csv", keep_default_na=False)

    for column in ("mean_difference", "t_stat", "se", "df", "mde", "mde_percent",
                   "mde_power", "mde_alpha", "mde_note"):
        assert column in written.columns

    row = results.set_index("metric").loc["total_volume_mm3"]
    poor = wmh_df["total_volume_mm3"][:8]
    good = wmh_df["total_volume_mm3"][8:]
    t = stats.ttest_ind(poor, good)
    assert row["df"] == 18
    assert row["t_stat"] == pytest.approx(t.statistic)
    assert row["mean_difference"] / row["se"] == pytest.approx(t.statistic)
    assert row["mde"] == pytest.approx(minimum_detectable_effect(row["se"], 18))
    assert row["mde_percent"] == pytest.approx(100 * row["mde"] / abs(good.mean()))
    assert row["mde_note"] == ""

    flat = written.set_index("metric").loc["max_lesion_volume"]
    assert flat["mde"] == ""
    assert flat["mde_note"].startswith("not computed: ")
    assert set(written["mde_power"]) == {MDE_POWER}
    assert set(written["mde_alpha"]) == {MDE_ALPHA}


# --------------------------------------------------------------------------
# Connectome edgewise group difference
# --------------------------------------------------------------------------


def _matrices(rng, n, n_rois=5, shift=0.0):
    m = rng.normal(0.3 + shift, 0.1, (n, n_rois, n_rois))
    return (m + m.transpose(0, 2, 1)) / 2


@pytest.mark.parametrize("paired", [False, True])
def test_group_difference_writes_se_and_mde_matrices(tmp_path, paired):
    pytest.importorskip("statsmodels")  # FDR step; not a declared dependency
    from neurovrai.connectome.group_analysis import compute_group_difference

    rng = np.random.default_rng(3)
    g1 = _matrices(rng, 10)
    g2 = g1 + rng.normal(0.05, 0.05, g1.shape) if paired else _matrices(rng, 13, shift=0.05)

    res = compute_group_difference(g1, g2, paired=paired, output_dir=tmp_path, output_prefix="cmp")

    se = np.load(tmp_path / "cmp_se_matrix.npy")
    mde = np.load(tmp_path / "cmp_mde_matrix.npy")
    iu = np.triu_indices(5, k=1)
    df = 9 if paired else 21
    assert res["mde_df"] == df
    assert np.allclose(res["difference_matrix"][iu] / se[iu], res["t_matrix"][iu])
    assert np.allclose(mde[iu], minimum_detectable_effect(se[iu], df))
    assert np.isnan(np.diag(mde)).all()

    summary = json.loads((tmp_path / "cmp_summary.json").read_text())
    assert (summary["mde_power"], summary["mde_alpha"], summary["mde_note"]) == (0.8, 0.05, "")
    assert summary["mde_df"] == df


def test_nbs_says_why_it_has_no_mde(tmp_path):
    pytest.importorskip("networkx")  # not a declared dependency
    from neurovrai.connectome.group_analysis import compute_network_based_statistic

    rng = np.random.default_rng(4)
    res = compute_network_based_statistic(
        _matrices(rng, 6), _matrices(rng, 6, shift=0.5),
        threshold=2.0, n_permutations=5, output_dir=tmp_path,
    )
    assert res["mde_note"] == NBS_NOTE
    if (tmp_path / "nbs_components.json").exists():
        assert json.loads((tmp_path / "nbs_components.json").read_text())["mde_note"] == NBS_NOTE
