"""Integration-tier template: end-to-end preprocessing vs derived-metric golden.

Runs only via `make integration` (marker: integration), locally or on HPC where
FSL/ANTs/MRtrix exist; NOT part of the fast PR gate. Because these tools aren't
bit-reproducible, compare robust derived metrics (Dice, correlation, summary)
within loose tolerance — never raw voxels.

Skipping template: wire to a real builder + tiny downsampled-real fixture, then
remove the skip. Skips cleanly when tools are absent, so never falsely fails.
"""

from __future__ import annotations

import shutil

import pytest

pytestmark = pytest.mark.integration

requires_fsl = pytest.mark.skipif(
    shutil.which("bet") is None,
    reason="FSL not installed; integration tier runs locally / on HPC",
)


@requires_fsl
def test_dwi_pipeline_derived_metrics(tmp_path):
    pytest.skip(
        "TEMPLATE: run the real DWI pipeline on a tiny downsampled fixture, then "
        "compare derived metrics to golden. Pattern:\n"
        "  from tests.regression._derived import dice, summary, assert_metrics_close\n"
        "  out_mask, fa_map = run_dwi_preprocess(fixtures.dwi_tiny(), tmp_path)\n"
        "  metrics = {'dice_brainmask': dice(out_mask, golden_mask),\n"
        "             **{f'fa_{k}': v for k, v in summary(fa_map).items()}}\n"
        "  assert_metrics_close(metrics, 'dwi_pipeline')"
    )
