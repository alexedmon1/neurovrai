"""
Minimum detectable effect (MDE) for tabular tests.

Every tabular inferential result neurovrai writes carries the difference its
design could have detected: the minimum detectable effect at 80% power and
two-sided alpha 0.05. A null then says what the study could have seen, in the
estimate's own units, rather than only that p was large.

    mde = (t.ppf(1 - alpha/2, df) + t.ppf(power, df)) * se

with normal quantiles when df is missing or infinite.

The contract (shared, by name, with neurofaune and source-analytics) is the
set of fields `mde_fields` returns for one row:

    se, df, mde, mde_percent, mde_power, mde_alpha, mde_note

`mde_note` is empty when the MDE was computed and "not computed: <reason>"
otherwise; an MDE is never left blank without one.

What this module deliberately does not do:

- No multiple-comparison correction. The alpha is nominal (0.05) even where the
  test that produced the p-value was FDR- or permutation-corrected; the
  corrected design's MDE would be larger.
- No voxelwise maps. randomise/TFCE and nilearn second-level maps are corrected
  by their own procedures, and a per-voxel MDE would not describe them. They get
  `VOXELWISE_NOTE`, never a number. Network-based statistic components likewise
  get `NBS_NOTE`.
- It computes the SE of the test that was actually run. A Student (pooled)
  t-test gets the pooled SE and n1 + n2 - 2 df, a Welch test the Welch SE and
  Welch-Satterthwaite df, a paired test the SD of the differences over sqrt(n)
  and n - 1 df. Choosing a different SE than the one behind the p-value would
  make the MDE describe a test nobody ran.
"""

import numpy as np
from scipy import stats

MDE_POWER = 0.80
MDE_ALPHA = 0.05

VOXELWISE_NOTE = "not computed: voxelwise map"
NBS_NOTE = "not computed: network-based statistic component"

# With no SD to judge by, only a reference mean this close to zero is refused a percentage.
_NEAR_ZERO = 1e-12

ArrayLike = float | np.ndarray


def _quantile(q: float, df: ArrayLike) -> ArrayLike:
    """Student-t quantile, or the normal quantile where df is missing or infinite."""
    if df is None:
        return stats.norm.ppf(q)
    df_arr = np.asarray(df, dtype=float)
    finite = np.isfinite(df_arr)
    out = np.where(finite, stats.t.ppf(q, np.where(finite, df_arr, 1.0)), stats.norm.ppf(q))
    return out if out.ndim else float(out)


def minimum_detectable_effect(
    se: ArrayLike,
    df: ArrayLike | None = None,
    alpha: float = MDE_ALPHA,
    power: float = MDE_POWER,
) -> ArrayLike:
    """
    Difference detectable with `power` at two-sided `alpha`, in the units of `se`.

    Works elementwise on arrays (e.g. an edge matrix). NaN where se is NaN.
    """
    se_arr = np.asarray(se, dtype=float)
    mde = (_quantile(1 - alpha / 2, df) + _quantile(power, df)) * se_arr
    return mde if np.ndim(mde) else float(mde)


def independent_se_df(
    a: np.ndarray, b: np.ndarray, equal_var: bool = True, axis: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    SE and df of mean(a) - mean(b) for an independent two-sample t-test.

    equal_var=True matches scipy.stats.ttest_ind's default (Student, pooled
    variance, df = n_a + n_b - 2); equal_var=False is Welch with
    Welch-Satterthwaite df. Works along `axis`, so a stack of matrices gives an
    SE and df per edge.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a = a.shape[axis]
    n_b = b.shape[axis]
    with np.errstate(divide="ignore", invalid="ignore"):
        # ddof=1 with a single subject gives NaN, which mde_fields reports.
        var_a = np.var(a, axis=axis, ddof=1) if n_a > 1 else np.full(np.var(a, axis=axis).shape, np.nan)
        var_b = np.var(b, axis=axis, ddof=1) if n_b > 1 else np.full(np.var(b, axis=axis).shape, np.nan)
    if not equal_var:
        return welch_se_df(np.sqrt(var_a), n_a, np.sqrt(var_b), n_b)
    df = float(n_a + n_b - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / df
        se = np.sqrt(pooled * (1.0 / n_a + 1.0 / n_b))
    if np.ndim(se):
        return se, np.full(np.shape(se), df)
    return float(se), df


def welch_se_df(sd_a: ArrayLike, n_a: int, sd_b: ArrayLike, n_b: int) -> tuple[ArrayLike, ArrayLike]:
    """SE sqrt(sd_a^2/n_a + sd_b^2/n_b) and Welch-Satterthwaite df."""
    va = np.asarray(sd_a, dtype=float) ** 2 / n_a
    vb = np.asarray(sd_b, dtype=float) ** 2 / n_b
    se = np.sqrt(va + vb)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))
    if np.ndim(se):
        return se, df
    return float(se), float(df)


def paired_se_df(a: np.ndarray, b: np.ndarray, axis: int = 0) -> tuple[ArrayLike, ArrayLike]:
    """SE (SD of the differences / sqrt(n)) and df (n - 1) of a paired t-test."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.shape[axis]
    if n < 2:
        se = np.full(np.std(d, axis=axis).shape, np.nan)
    else:
        se = np.std(d, axis=axis, ddof=1) / np.sqrt(n)
    df = float(n - 1)
    if np.ndim(se):
        return se, np.full(np.shape(se), df)
    return float(se), df


def mde_percent(
    mde: ArrayLike, reference_mean: ArrayLike | None, sd: ArrayLike | None = None
) -> ArrayLike:
    """100 * mde / |reference mean|; NaN with no reference or a mean near zero.

    "Near zero" is the rule shared with neurofaune and source-analytics: the
    reference mean lies within one residual (pooled) SD of zero, |mean| < sd.
    For change scores, centred measures or log ratios a percent of such a mean
    is a large number that means nothing. Without an SD only an exact zero is
    refused.
    """
    if reference_mean is None:
        return np.nan
    ref = np.abs(np.asarray(reference_mean, dtype=float))
    floor = _NEAR_ZERO if sd is None else np.maximum(np.abs(np.asarray(sd, dtype=float)), _NEAR_ZERO)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(np.isfinite(ref) & (ref >= floor), 100.0 * np.asarray(mde) / ref, np.nan)
    return pct if pct.ndim else float(pct)


def mde_fields(
    se: float | None,
    df: float | None,
    reference_mean: float | None = None,
    alpha: float = MDE_ALPHA,
    power: float = MDE_POWER,
    sd: float | None = None,
) -> dict[str, object]:
    """
    The contract's columns for one tabular row.

    Returns se, df, mde, mde_percent, mde_power, mde_alpha, mde_note. When the
    SE cannot be computed (too few subjects, no variance), mde is NaN and
    mde_note says why.
    """
    note = ""
    if se is None or not np.isfinite(se):
        note = "not computed: standard error unavailable (fewer than two subjects in a group?)"
    elif se <= 0:
        note = "not computed: zero variance"
    elif df is not None and np.isfinite(df) and df <= 0:
        note = "not computed: no residual degrees of freedom"

    if note:
        mde = np.nan
        pct = np.nan
    else:
        mde = minimum_detectable_effect(se, df, alpha=alpha, power=power)
        pct = mde_percent(mde, reference_mean, sd)

    return {
        "se": np.nan if se is None else float(se),
        "df": np.nan if df is None else float(df),
        "mde": mde,
        "mde_percent": pct,
        "mde_power": power,
        "mde_alpha": alpha,
        "mde_note": note,
    }
