#!/usr/bin/env python3
"""Re-segment preprocessed T1w images with the study config's anatomical.segmentation_method.

Runs only the segmentation step (run_t1w_segmentation) on derivatives/<subject>/anat/bias_corrected,
keeping skull stripping, N4 and registration. Previous outputs go to segmentation/superseded_<method>/.
Afterwards, everything made from the tissue maps must be regenerated (VBM, T1w/T2w WM mask, ASL
calibration, aCompCor, WMH).

    uv run python scripts/resegment_t1w.py --config /study/config.yaml --jobs 3 [SUBJECT ...]

FAST at 0.45 mm needs ~9 GB of memory per subject; choose --jobs to fit.
"""
import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _one(config_path: str, anat_dir: str) -> str:
    from neurovrai.config import load_config
    from neurovrai.preprocess.workflows.t1w_preprocess import run_t1w_segmentation
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s {Path(anat_dir).parent.name} %(message)s")
    maps = run_t1w_segmentation(load_config(Path(config_path)), Path(anat_dir))
    return f"{Path(anat_dir).parent.name}: " + ", ".join(f"{t}={p.resolve().name}" for t, p in maps.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=Path, help="study config.yaml")
    ap.add_argument('--derivatives', type=Path, help="default: <config's folder>/derivatives")
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('subjects', nargs='*', help="default: every subject with anat/bias_corrected")
    a = ap.parse_args()
    deriv = a.derivatives or a.config.resolve().parent / 'derivatives'
    subjects = a.subjects or sorted(p.parent.parent.name for p in deriv.glob('*/anat/bias_corrected'))
    anat_dirs = [deriv / s / 'anat' for s in subjects]
    missing = [d for d in anat_dirs if not (d / 'bias_corrected').is_dir()]
    if missing:
        print("no bias_corrected/ in: " + ", ".join(str(d) for d in missing), file=sys.stderr)
        return 2
    failed = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(_one, str(a.config), str(d)): d for d in anat_dirs}
        for fut in as_completed(futs):
            try:
                print("done " + fut.result(), flush=True)
            except Exception as e:  # report and carry on with the others
                failed += 1
                print(f"FAILED {futs[fut].parent.name}: {e}", flush=True)
    print(f"{len(anat_dirs) - failed} of {len(anat_dirs)} re-segmented")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
