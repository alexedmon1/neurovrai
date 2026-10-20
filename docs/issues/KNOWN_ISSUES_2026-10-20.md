# Known issues found 2026-10-20

Found while adding the minimum detectable effect (MDE) to the tabular tests
(`feat/power-mde`). None is fixed yet; each is to be fixed later.

## Defects

1. **The TBSS GLM path cannot run.** `neurovrai/analysis/tbss/run_tbss_stats.py`
   calls `run_fsl_glm` (line 364) and `threshold_zstat` (line 381), which are
   defined nowhere in the repository, and `summarize_glm_results` (line 389),
   which exists in `analysis/stats/nilearn_glm.py` but is not imported. Any run
   with `method='glm'` raises `NameError`. `docs/GLM_USAGE.md` documents
   `run_fsl_glm` and imports it from `neurovrai.analysis.stats.glm_wrapper`, a
   module that does not exist.

2. **`statsmodels` is used but not declared.** `compute_group_difference`
   (`connectome/group_analysis.py`, line 663) imports
   `statsmodels.stats.multitest.multipletests`, and `pyproject.toml` does not
   list `statsmodels`, so connectome group statistics fail in a clean install.
   The two connectome tests skip in `make check` for the same reason.

3. **The connectome README's examples are wrong.** `neurovrai/connectome/README.md`
   reads `nbs_results['component_pvals']` (line 287); the key is
   `component_pvalues`. Its `compute_group_difference` example (line 310)
   unpacks two matrices and passes `method=`; the function returns a dict and
   takes no `method` argument.

4. **Two functions can write the same SE file.** `average_connectivity_matrices`
   (default prefix `group`) and `compute_group_difference` both save
   `{output_prefix}_se_matrix.npy`. Their default prefixes differ, but given the
   same prefix and output directory one overwrites the other. One of them
   should use a distinct name.

5. **The WMH script disagrees with itself about the groups.**
   `scripts/wmh_group_comparison.py` says in its docstring "mriglu: 1 =
   controlled gestational diabetes, 2 = uncontrolled", while the code prints
   and labels mriglu=1 as poor control and 2 as good control. One is wrong. It
   decides which group `mde_percent` (and the sign of `cohens_d`) is relative
   to, which is currently the mriglu=2 group.

## Follow-ups to the MDE work

- **Multiple-comparison correction.** The MDE uses a nominal alpha of 0.05 by
  decision; for the edgewise connectome tests (FDR across edges) the real bar is
  stricter, so the MDE is optimistic. To be added later.
- **Mann-Whitney.** `wmh_group_comparison.py` reports a Mann-Whitney p beside
  the t-test; the MDE describes the t-test only.
- **Voxelwise maps** (randomise/TFCE, nilearn second level) carry no MDE by
  design: "not computed: voxelwise map".
