"""Cohen's d from a t statistic (stats/effect_size.py), against d computed from the data.

Each test simulates subjects, computes t the way randomise would for that design
(scipy's t-test is the same GLM t for these designs), and checks that the conversion
gives back the d computed directly from the groups' means and SDs. The `.mat` tests
read an FSL design file as randomise is given one (TODO_EFFECT_SIZE_TESTING.md;
lumiere docs/RESULTS_PRODUCERS.md §4.1: "assumes a one-sample design (d = t / √N)
for any .mat").
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from neurovrai.analysis.stats.effect_size import (
    EffectSizeError,
    batch_effect_size_calculation,
    calculate_effect_size_ci,
    hedges_g,
    parse_design_matrix,
    t_to_cohens_d,
)

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(20261113)


def _pooled_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled


def _write_vest(path: Path, matrix: np.ndarray) -> Path:
    """An FSL VEST file (design.mat / design.con), as Glm or text2vest writes it."""
    rows, cols = matrix.shape
    lines = [f"/NumWaves\t{cols}", f"/NumPoints\t{rows}",
             "/PPheights\t" + "\t".join(["1"] * cols), "", "/Matrix"]
    lines += ["\t".join(f"{v:g}" for v in row) for row in matrix]
    path.write_text("\n".join(lines) + "\n")
    return path


def _two_group_design(n1: int, n2: int) -> np.ndarray:
    """Cell-means coding, the usual randomise two-group design: one indicator per group."""
    return np.vstack([np.tile([1, 0], (n1, 1)), np.tile([0, 1], (n2, 1))])


# --- the formulas, given the design ------------------------------------------------

def test_one_sample_d_is_the_mean_over_the_sd():
    x = RNG.normal(0.5, 1.0, 15)
    t = stats.ttest_1samp(x, 0).statistic
    assert t_to_cohens_d(np.array(t), n1=15, design_type="one_sample") == pytest.approx(
        x.mean() / x.std(ddof=1))


def test_paired_d_is_the_mean_difference_over_its_sd():
    pre = RNG.normal(0, 1, 18)
    post = pre + RNG.normal(0.4, 0.8, 18)
    t = stats.ttest_rel(post, pre).statistic
    diff = post - pre
    assert t_to_cohens_d(np.array(t), n1=18, design_type="paired") == pytest.approx(
        diff.mean() / diff.std(ddof=1))


@pytest.mark.parametrize("n1,n2", [(10, 10), (12, 20), (6, 25)])
def test_two_sample_d_is_the_mean_difference_over_the_pooled_sd(n1, n2):
    a, b = RNG.normal(0.6, 1.0, n1), RNG.normal(0.0, 1.0, n2)
    t = stats.ttest_ind(a, b, equal_var=True).statistic
    assert t_to_cohens_d(np.array(t), n1=n1, n2=n2, design_type="two_sample") == pytest.approx(
        _pooled_d(a, b))


def test_the_conversion_works_voxelwise_and_keeps_the_sign():
    t = np.array([[-3.0, 0.0], [2.0, 5.0]])
    d = t_to_cohens_d(t, n1=8, n2=8, design_type="two_sample")
    np.testing.assert_allclose(d, t * np.sqrt(1 / 8 + 1 / 8))


# --- the design, read from an FSL .mat ---------------------------------------------

def test_a_one_sample_mat_is_read_as_one_sample(tmp_path):
    mat = _write_vest(tmp_path / "design.mat", np.ones((14, 1)))
    info = parse_design_matrix(mat)
    assert (info["design_type"], info["n1"], info["n2"], info["n_subjects"]) == (
        "one_sample", 14, None, 14)


def test_a_two_group_mat_is_read_as_two_groups(tmp_path):
    mat = _write_vest(tmp_path / "design.mat", _two_group_design(10, 13))
    _write_vest(tmp_path / "design.con", np.array([[1, -1], [-1, 1]]))
    info = parse_design_matrix(mat)
    assert (info["design_type"], info["n1"], info["n2"], info["df"]) == ("two_sample", 10, 13, 21)
    assert info["variance_factors"] == pytest.approx({1: 1 / 10 + 1 / 13, 2: 1 / 10 + 1 / 13})


def test_a_two_group_mat_without_its_contrasts_is_refused(tmp_path):
    """Which groups a d compares is the contrast's; it is never guessed."""
    mat = _write_vest(tmp_path / "design.mat", _two_group_design(10, 13))
    with pytest.raises(EffectSizeError, match="design.con"):
        parse_design_matrix(mat)


def test_the_contrast_file_can_be_named(tmp_path):
    mat = _write_vest(tmp_path / "design.mat", _two_group_design(8, 9))
    con = _write_vest(tmp_path / "group.con", np.array([[1, -1]]))
    assert parse_design_matrix(mat, con)["variance_factors"] == pytest.approx({1: 1 / 8 + 1 / 9})


def test_d_from_a_two_group_mat_is_the_true_d(tmp_path):
    """End to end, as calculate_effect_sizes.py --design-file does it."""
    n1, n2 = 10, 10
    a, b = RNG.normal(0.8, 1.0, n1), RNG.normal(0.0, 1.0, n2)
    t = stats.ttest_ind(a, b, equal_var=True).statistic
    _write_vest(tmp_path / "design.con", np.array([[1, -1]]))
    info = parse_design_matrix(_write_vest(tmp_path / "design.mat", _two_group_design(n1, n2)))
    d = t * np.sqrt(info["variance_factors"][1])
    assert d == pytest.approx(_pooled_d(a, b))


def test_with_a_covariate_d_is_the_adjusted_difference_over_the_residual_sd(tmp_path):
    """Group (cell means) and a centred age: d = the contrast's estimate / residual SD."""
    n1, n2 = 14, 11
    design = np.column_stack([_two_group_design(n1, n2), RNG.normal(0, 1, n1 + n2)])
    y = design @ np.array([1.0, 0.4, 0.7]) + RNG.normal(0, 1, n1 + n2)
    c = np.array([1.0, -1.0, 0.0])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    df = n1 + n2 - 3
    sigma = np.sqrt(np.sum((y - design @ beta) ** 2) / df)
    t = c @ beta / (sigma * np.sqrt(c @ np.linalg.inv(design.T @ design) @ c))

    _write_vest(tmp_path / "design.con", c[None, :])
    info = parse_design_matrix(_write_vest(tmp_path / "design.mat", design))
    assert (info["design_type"], info["df"]) == ("glm", df)
    assert t * np.sqrt(info["variance_factors"][1]) == pytest.approx(c @ beta / sigma)


def test_the_glm_route_reduces_to_the_two_sample_hedges_g_and_ci():
    d, n1, n2 = np.array([0.5, -1.2]), 12, 20
    np.testing.assert_allclose(hedges_g(d, n1 + n2, df=n1 + n2 - 2), hedges_g(d, n1, n2))
    np.testing.assert_allclose(
        calculate_effect_size_ci(d, n1 + n2, variance_factor=1 / n1 + 1 / n2),
        calculate_effect_size_ci(d, n1, n2))


def test_the_batch_gives_each_contrast_its_own_d(tmp_path):
    """randomise's folder, design.mat and design.con: the --design-file route, end to end."""
    nib = pytest.importorskip("nibabel")
    n1, n2 = 9, 12
    tstat = RNG.normal(0, 2, (4, 4, 3)).astype(np.float32)
    rdir = tmp_path / "randomise"
    rdir.mkdir()
    for k, sign in ((1, 1), (2, -1)):
        nib.save(nib.Nifti1Image(sign * tstat, np.eye(4)), rdir / f"randomise_tstat{k}.nii.gz")
    mat = _write_vest(tmp_path / "design.mat", _two_group_design(n1, n2))
    _write_vest(tmp_path / "design.con", np.array([[1, -1], [-1, 1]]))

    results = batch_effect_size_calculation(rdir, mat, tmp_path / "out",
                                            contrast_names={1: "a_gt_b", 2: "b_gt_a"})
    for name, sign in (("a_gt_b", 1), ("b_gt_a", -1)):
        d = nib.load(results[name]["output_files"]["d_uncorrected"]).get_fdata()
        np.testing.assert_allclose(d, sign * tstat * np.sqrt(1 / n1 + 1 / n2), rtol=1e-6)
