"""Analysis folders to the results specification (Lumière) for neurovrai's randomise analyses.

neurovrai reads its randomise runs with neurofaune's read-out, as decided on 2026-10-06
(Lumière, docs/RESULTS_PRODUCERS.md §4.3): the design record
(``neurofaune.analysis.stats.design_record``), ``read_randomise`` (one tests row per
contrast and measure with its whole-mask effect, CI and MDE; one clusters row per cluster)
and ``write_readout_results`` (tables, column dictionaries, analysis.json, provenance.json,
figures). This module adds what is neurovrai's own:

- :func:`load_design_record` -- the design's record (``design.json``) when it has one;
  otherwise :func:`describe_neuroaider_design` writes one for a design neuroaider built
  (``design_summary.json`` + ``participants_matched.tsv``), checked by rebuilding
  ``design.mat`` from the participants table, so the rows are known, not assumed;
- :func:`fsl_atlas` -- region names for clusters from an FSL label atlas (JHU, ...);
- :func:`read_randomise_log` -- what randomise was actually run with (TFCE 2-D or 3-D,
  permutations, variance smoothing), read from its log rather than from code defaults;
- :func:`write_randomise_analysis` -- one analysis folder over several measures;
- :func:`tbss_results` -- the TBSS layout neurovrai writes (``analysis/tbss``).

Command line: ``python -m neurovrai.analysis.results tbss --study-root ... --design-dir ...``.
A study whose other analyses sit in its own layout calls :func:`write_randomise_analysis`
from its own script.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import logging
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: FSL's MNI152 (and FMRIB58_FA, which is in it) by its TemplateFlow name.
FSL_MNI = "MNI152NLin6Asym"


# --------------------------------------------------------------------------- provenance ---
def package_entry() -> dict[str, Any]:
    """neurovrai as a BIDS GeneratedBy entry: version, and the commit where it can be known.

    From a VCS install the commit is in the install's PEP 610 record; from an editable
    install (a checkout) it is the checkout's HEAD, with a note when tracked files differ.
    """
    entry: dict[str, Any] = {"Name": "neurovrai"}
    try:
        dist = md.distribution("neurovrai")
    except md.PackageNotFoundError:          # pragma: no cover - not installed
        entry["Version"] = "unknown"
        return entry
    entry["Version"] = dist.version
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    vcs = info.get("vcs_info") or {}
    if vcs.get("commit_id"):
        entry["CommitID"] = vcs["commit_id"]
        if vcs.get("requested_revision"):
            entry["RequestedRevision"] = vcs["requested_revision"]
        entry["CodeURL"] = info.get("url")
        return entry
    src = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return entry
    entry["CommitID"] = head
    if dirty:
        entry["Description"] = "editable checkout with uncommitted changes to tracked files"
    return entry


# ------------------------------------------------------------------------------- atlas ---
def fsl_atlas(atlas_id: str = "jhu-labels", resolution: float = 1.0, *, drop: Sequence[str] = ()):
    """An FSL atlas as neurofaune's ``Atlas``: parcellation, names, hemispheres.

    Names and label values come from FSL's atlas registry (fslpy), so a value means what
    FSL says it means. A probabilistic atlas is read through its summary image (FSL's
    max-probability, thresholded at 25%). Labels named in ``drop`` are set to 0.
    Hemisphere is read from a trailing " L" / " R" (JHU's convention) or a leading
    "Left " / "Right " (Harvard-Oxford subcortical).
    """
    from fsl.data import atlases
    from neurofaune.analysis.stats.readout import Atlas

    atlases.rescanAtlases()
    desc = atlases.getAtlasDescription(atlas_id)
    images = desc.images if desc.atlasType == "label" else desc.summaryImages
    path = None
    for img, pix in zip(images, desc.pixdims, strict=False):
        if abs(pix[0] - resolution) < 1e-3:
            path = img
    if path is None:
        raise ValueError(f"{atlas_id} has no {resolution} mm image")
    path = path if str(path).endswith((".nii", ".nii.gz")) else f"{path}.nii.gz"
    data = np.asarray(nib.load(path).dataobj, dtype=np.int32)
    names, hemi = {}, {}
    for label in desc.labels:
        if label.value == 0:
            continue
        if label.name in drop:
            data[data == label.value] = 0
            continue
        names[int(label.value)] = label.name
        m = re.search(r"\s([LR])$", label.name) or re.match(r"(Left|Right)\s", label.name)
        if m:
            hemi[int(label.value)] = {"L": "left", "R": "right", "Left": "left",
                                      "Right": "right"}[m.group(1)]
    return Atlas(data, names, hemi)


def combine_atlases(first, *others, offset: int = 100):
    """One parcellation from several on the same grid: the first's labels win where both
    label a voxel; each later atlas's labels are shifted by ``offset`` (then 2 x offset, ...)."""
    from neurofaune.analysis.stats.readout import Atlas

    data = first.parcellation.copy()
    names, hemi = dict(first.names), dict(first.hemispheres)
    for k, other in enumerate(others, start=1):
        if other.parcellation.shape != data.shape:
            raise ValueError("atlases on different grids")
        shift = k * offset
        if data.max() >= shift:
            raise ValueError(f"labels reach {data.max()}; use a larger offset than {offset}")
        fill = (data == 0) & (other.parcellation > 0)
        data[fill] = other.parcellation[fill] + shift
        names.update({v + shift: n for v, n in other.names.items()})
        hemi.update({v + shift: h for v, h in other.hemispheres.items()})
    return Atlas(data, names, hemi)


def harvard_oxford(resolution: float = 2.0):
    """Harvard-Oxford cortical, then subcortical where cortical has no label; subcortical's
    whole-hemisphere "Cerebral White Matter" / "Cerebral Cortex" labels left out."""
    drop = [f"{side} Cerebral {what}" for side in ("Left", "Right") for what in ("White Matter", "Cortex")]
    return combine_atlases(fsl_atlas("harvardoxford-cortical", resolution),
                           fsl_atlas("harvardoxford-subcortical", resolution, drop=drop))


# ------------------------------------------------------------------------------ design ---
def _ids(design_dir: Path) -> list[str]:
    tsv = design_dir / "participants_matched.tsv"
    ids = pd.read_csv(tsv, sep="\t", dtype=str)["participant_id"].tolist()
    order = design_dir / "subject_order.txt"
    if order.exists():
        listed = [ln.strip() for ln in order.read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]
        if listed and listed != ids:
            raise ValueError(f"{order} and {tsv} list different subjects or orders")
    return ids


def describe_neuroaider_design(
    design_dir: Path,
    levels: Mapping[str, Mapping[str, str]],
    *,
    covariate_meanings: Mapping[str, str] | None = None,
    data: Mapping[str, str] | None = None,
    summary: str | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """A design record (``neurofaune.design/1``) for a design neuroaider wrote.

    ``levels`` says what each factor level is, e.g. ``{"mriglu": {"1": "Controlled",
    "2": "Uncontrolled"}}``: that is study knowledge, never guessed. The participants
    table is rebuilt into a design matrix and compared with ``design.mat`` (to its six
    printed decimals); any difference raises, so the record's rows are the matrix's rows.
    Contrasts are read from ``design.con``: +1/-1 on two levels of one factor is a
    two-group test, +/-1 on one covariate a regression; anything else raises.
    """
    from neurofaune.analysis.stats.design_record import describe_design, read_vest

    design_dir = Path(design_dir)
    summ = json.loads((design_dir / "design_summary.json").read_text())
    X, _ = read_vest(design_dir / "design.mat")
    C, con_names = read_vest(design_dir / "design.con")
    C = np.atleast_2d(C)
    ids = _ids(design_dir)
    table = pd.read_csv(design_dir / "participants_matched.tsv", sep="\t", dtype=str)
    columns = summ["columns"]
    if X.shape != (len(ids), len(columns)):
        raise ValueError(f"design.mat is {X.shape}; {len(ids)} subjects x {len(columns)} columns described")
    covs = {c["name"]: c for c in summ.get("covariates", [])}
    facs = {f["name"]: f for f in summ.get("factors", [])}
    cov_meaning = dict(covariate_meanings or {})

    rebuilt, described, col_of = [], [], {}
    for j, name in enumerate(columns):
        if name in covs:
            v = table[name].astype(float).to_numpy()
            how = []
            if covs[name].get("mean_center"):
                v = v - v.mean()
                how.append("mean-centred")
            if covs[name].get("standardize"):
                v = v / v.std(ddof=1)
                how.append("standardised")
            rebuilt.append(v)
            described.append({"name": name, "kind": "covariate",
                              "meaning": f"{cov_meaning.get(name, name)}"
                                         + (f", {' and '.join(how)}" if how else "")})
            col_of[j] = ("covariate", name, None)
            continue
        factor = next((f for f in facs if name.startswith(f + "_")), None)
        if factor is None or facs[factor].get("coding") != "dummy":
            raise ValueError(f"column {name!r} is neither a covariate nor a dummy-coded factor level")
        level = name[len(factor) + 1:]
        label = (levels.get(factor) or {}).get(level)
        if not label:
            raise ValueError(f"what is {factor} = {level}? give it in levels")
        rebuilt.append((table[factor].astype(str).str.strip() == level).astype(float).to_numpy())
        described.append({"name": name, "kind": "group",
                          "meaning": f"1 for {label} ({factor} = {level}), else 0"})
        col_of[j] = ("group", factor, label)
    rebuilt = np.column_stack(rebuilt)
    if not np.allclose(rebuilt, X, atol=5e-6):
        bad = np.argwhere(~np.isclose(rebuilt, X, atol=5e-6))[:3].tolist()
        raise ValueError(f"design.mat does not match participants_matched.tsv (row, column): {bad}")

    names = summ.get("contrasts") or con_names or []
    if len(names) != len(C):
        raise ValueError(f"{len(C)} contrasts in design.con, {len(names)} names")
    covariate_names = [cov_meaning.get(n, n) for n in covs]
    contrasts = []
    for name, vec in zip(names, C, strict=True):
        nz = [j for j in range(len(vec)) if abs(vec[j]) > 1e-9]
        kinds = {col_of[j][0] for j in nz}
        if kinds == {"group"} and len(nz) == 2 and sorted(vec[nz].tolist()) == [-1.0, 1.0] \
                and col_of[nz[0]][1] == col_of[nz[1]][1]:
            a = col_of[nz[0] if vec[nz[0]] > 0 else nz[1]][2]
            b = col_of[nz[1] if vec[nz[0]] > 0 else nz[0]][2]
            adj = f", adjusting for {' and '.join(covariate_names)}" if covariate_names else ""
            contrasts.append({"name": name, "vector": vec.tolist(), "test_kind": "two_group",
                              "group_a": a, "group_b": b, "tests": f"{a} > {b}{adj}"})
        elif kinds == {"covariate"} and len(nz) == 1 and abs(vec[nz[0]]) == 1.0:
            cov = col_of[nz[0]][1]
            sign = "positive" if vec[nz[0]] > 0 else "negative"
            others = [cov_meaning.get(n, n) for n in covs if n != cov]
            groups = sorted({col_of[j][1] for j in col_of if col_of[j][0] == "group"})
            adj = others + [f"{g} group" for g in groups]
            contrasts.append({"name": name, "vector": vec.tolist(), "test_kind": "regression",
                              "tests": f"a {sign} association with {cov_meaning.get(cov, cov)}"
                                       + (f", adjusting for {' and '.join(adj)}" if adj else "")})
        else:
            raise ValueError(f"contrast {name} {vec.tolist()} is not a two-group or covariate contrast")

    groups: dict[str, int] = {}
    for j, (kind, _, label) in col_of.items():
        if kind == "group":
            groups[label] = int(X[:, j].sum())
    return describe_design(described, contrasts, rows=ids, rows_file="participants_matched.tsv",
                           data=dict(data) if data else None, groups=groups, summary=summary,
                           notes=list(notes) or None, n=len(ids))


def load_design_record(
    design_dir: Path,
    levels: Mapping[str, Mapping[str, str]] | None = None,
    **describe,
) -> dict[str, Any]:
    """The design's record: its ``design.json``, checked against design.mat / design.con.

    A design without one gets one from :func:`describe_neuroaider_design` (which needs
    ``levels``), written beside it.
    """
    from neurofaune.analysis.stats.design_record import (
        check_design_record,
        read_design_record,
        write_design_record,
    )

    design_dir = Path(design_dir)
    record = read_design_record(design_dir)
    if record is not None:
        check_design_record(record, design_mat=design_dir / "design.mat",
                            design_con=design_dir / "design.con")
        return record
    if not levels:
        raise ValueError(f"{design_dir} has no design.json; give the factor levels to describe it")
    record = describe_neuroaider_design(design_dir, levels, **describe)
    write_design_record(design_dir, record)
    return record


# --------------------------------------------------------------------------- randomise ---
@dataclass
class RandomiseRun:
    """One randomise run: its output folder, the 4-D input it was given, and its prefix."""

    run_dir: Path
    data: Path
    prefix: str = "randomise"


def read_randomise_log(run_dir: Path) -> dict[str, Any]:
    """What randomise was run with, from the ``randomise options:`` line of its log."""
    log = Path(run_dir) / "randomise.log"
    if not log.exists():
        raise FileNotFoundError(f"{log}: no log to read the randomise options from")
    first = log.read_text(errors="replace").splitlines()[0]
    if not first.startswith("randomise options:"):
        raise ValueError(f"{log} does not start with randomise's options line")
    args = shlex.split(first.split(":", 1)[1])
    opts: dict[str, Any] = {"command": "randomise " + " ".join(args)}
    if "--T2" in args:
        opts["tfce"] = "2-D TFCE"
    elif "-T" in args:
        opts["tfce"] = "3-D TFCE"
    else:
        opts["tfce"] = None
    if "-n" in args:
        opts["n_permutations"] = int(args[args.index("-n") + 1])
    if "-v" in args:
        opts["variance_smoothing_mm"] = float(args[args.index("-v") + 1])
    opts["uncorrected_p"] = "--uncorrp" in args
    if not log.read_text(errors="replace").rstrip().endswith("Finished, exiting."):
        raise ValueError(f"{log}: randomise did not finish")
    return opts


def write_randomise_analysis(
    output_dir: Path,
    runs: Mapping[str, RandomiseRun],
    *,
    design_dir: Path,
    design_record: Mapping[str, Any],
    mask: Path,
    mask_name: str,
    analysis_id: str,
    title: str,
    description: str,
    analysis_type: str,
    modality: str,
    space: str,
    background: Path | None = None,
    atlas=None,
    alpha: float = 0.05,
    measure_column: str = "metric",
    caveats: Sequence[str] = (),
    plane: str = "axial",
    role: str = "exploratory",
    run: Mapping[str, Any] | str | None = None,
    derived_from: Sequence[Mapping[str, Any]] | None = None,
    references: Sequence[Mapping[str, str]] = (),
    strict: bool = True,
):
    """Read every measure's randomise run and write one analysis folder over them.

    The inference (TFCE 2-D or 3-D, permutations, variance smoothing) is read from each
    run's log and must agree across measures; runs without uncorrected p maps are named
    in a caveat. Run folders must be inside ``output_dir`` (the maps are listed by
    relative path).
    """
    from neurofaune.analysis.stats.readout import read_randomise
    from neurofaune.analysis.stats.readout_results import now, write_readout_results

    started = now()
    output_dir = Path(output_dir)
    logs = {m: read_randomise_log(r.run_dir) for m, r in runs.items()}
    for key in ("tfce", "n_permutations", "variance_smoothing_mm"):
        seen = {logs[m].get(key) for m in logs}
        if len(seen) != 1:
            raise ValueError(f"measures were run with different {key}: "
                             f"{ {m: logs[m].get(key) for m in logs} }")
    first = next(iter(logs.values()))
    if not first["tfce"]:
        raise ValueError("these runs are not TFCE; describe their inference by hand")
    inference = first["tfce"]
    caveats = list(caveats)
    if first.get("variance_smoothing_mm"):
        inference += f" with {first['variance_smoothing_mm']:g} mm variance smoothing (pseudo-t)"
        caveats.append(f"randomise smoothed the variance (-v {first['variance_smoothing_mm']:g}, sigma "
                       "in mm), so the voxel statistics are pseudo-t; the whole-mask effect sizes and "
                       "CIs in tests.csv are from an unsmoothed fit of each subject's mask mean.")
    no_uncorr = [m for m in logs if not logs[m]["uncorrected_p"]]
    if no_uncorr:
        caveats.append(f"{', '.join(no_uncorr)} were run without --uncorrp: no uncorrected p maps, so "
                       "their uncorrected extent is not reported.")

    contrast_names = [c["name"] for c in design_record["contrasts"]]
    column_names = [c["name"] for c in design_record["columns"]]
    tests, clusters = [], []
    for m, r in runs.items():
        t, c = read_randomise(r.run_dir, r.data, Path(design_dir) / "design.mat", mask,
                              prefix=r.prefix, atlas=atlas, column_names=column_names,
                              contrast_names=contrast_names, design_con=Path(design_dir) / "design.con",
                              alpha=alpha, labels={measure_column: m})
        tests.append(t)
        clusters.append(c)
    tests = pd.concat(tests, ignore_index=True)
    clusters = pd.concat([c for c in clusters if len(c)], ignore_index=True) if any(len(c) for c in clusters) \
        else pd.DataFrame()

    axes = "".join(nib.aff2axcodes(nib.load(str(mask)).affine))
    prefixes = {r.prefix for r in runs.values()}
    if len(prefixes) != 1:
        raise ValueError(f"runs use different prefixes {prefixes}")
    return write_readout_results(
        output_dir, tests, clusters, analysis_id=analysis_id, title=title, description=description,
        analysis_type=analysis_type, modality=modality, measure_column=measure_column,
        measures=list(runs), run_dirs={m: r.run_dir for m, r in runs.items()}, prefix=prefixes.pop(),
        n_permutations=first["n_permutations"], alpha=alpha, mask_name=mask_name, space=space,
        inference=inference, role=role, design_record=design_record, started=started,
        inputs=[{"path": str(r.data), "measure": m} for m, r in runs.items()],
        settings={"randomise": {m: logs[m]["command"] for m in logs},
                  "design": str(design_dir)},
        caveats=caveats, mask=mask, background=background, axes=axes, plane=plane,
        references=references, run=run, generated_by=[package_entry()], derived_from=derived_from,
        strict=strict)


# ------------------------------------------------------------------------------- TBSS ---
TBSS_METRICS = ("FA", "MD", "AD", "RD", "MK", "AK", "RK", "KFA", "FICVF", "ODI", "FISO")


def tbss_results(
    study_root: Path,
    *,
    design_dir: Path,
    name: str,
    title: str,
    description: str,
    run_name: str = "randomise_output",
    metrics: Sequence[str] | None = None,
    levels: Mapping[str, Mapping[str, str]] | None = None,
    covariate_meanings: Mapping[str, str] | None = None,
    run: Mapping[str, Any] | str | None = None,
    caveats: Sequence[str] = (),
    strict: bool = True,
):
    """TBSS in neurovrai's layout: one analysis folder, ``<study>/analysis/tbss``.

    Reads ``analysis/tbss/<metric>/<run_name>`` for every metric that has one, with
    ``analysis/tbss/stats/all_<metric>_skeletonised.nii.gz`` and the skeleton mask, and the
    design in ``design_dir`` (its design.json, or one written for it from ``levels``).
    Background: FMRIB58_FA 1 mm, the TBSS registration target, in FSL's MNI152; regions
    from the JHU white-matter labels.
    """
    from lumiere.spec import analysis_id

    study_root = Path(study_root)
    tbss = study_root / "analysis" / "tbss"
    metrics = [m for m in (metrics or TBSS_METRICS) if (tbss / m / run_name).is_dir()]
    if not metrics:
        raise FileNotFoundError(f"no <metric>/{run_name} in {tbss}")
    record = load_design_record(design_dir, levels, covariate_meanings=covariate_meanings,
                                summary=description)
    fsldir = Path(os.environ.get("FSLDIR", "/usr/local/fsl"))
    runs = {m: RandomiseRun(tbss / m / run_name, tbss / "stats" / f"all_{m}_skeletonised.nii.gz")
            for m in metrics}
    return write_randomise_analysis(
        tbss, runs, design_dir=design_dir, design_record=record,
        mask=tbss / "stats" / "mean_FA_skeleton_mask.nii.gz", mask_name="TBSS skeleton",
        analysis_id=analysis_id("dwi", "tbss", name), title=title, description=description,
        analysis_type="tbss", modality="dwi", space=FSL_MNI,
        background=fsldir / "data" / "standard" / "FMRIB58_FA_1mm.nii.gz",
        atlas=fsl_atlas("jhu-labels", 1.0), run=run, caveats=caveats, strict=strict)


# -------------------------------------------------------------------------------- CLI ---
def _levels(items: Sequence[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for item in items:
        m = re.fullmatch(r"([^:=]+):([^=]+)=(.+)", item)
        if not m:
            raise SystemExit(f"--level {item!r}: write factor:level=Label, e.g. mriglu:1=Controlled")
        out.setdefault(m.group(1), {})[m.group(2)] = m.group(3)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m neurovrai.analysis.results",
                                description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="family", required=True)
    t = sub.add_parser("tbss", help="analysis/tbss as one analysis folder")
    t.add_argument("--study-root", type=Path, required=True)
    t.add_argument("--design-dir", type=Path, required=True)
    t.add_argument("--run-name", default="randomise_output",
                   help="each metric's randomise folder under analysis/tbss/<metric>/")
    t.add_argument("--level", action="append", default=[],
                   help="for a design without design.json: factor:level=Label (repeat)")
    t.add_argument("--covariate", action="append", default=[],
                   help="for a design without design.json: name=meaning (repeat)")
    t.add_argument("--name", required=True, help="the analysis name in its id, dwi/tbss/<name>")
    t.add_argument("--title", required=True)
    t.add_argument("--description", required=True)
    t.add_argument("--metrics", nargs="*")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = tbss_results(a.study_root, design_dir=a.design_dir, name=a.name, title=a.title,
                          description=a.description, run_name=a.run_name, metrics=a.metrics,
                          levels=_levels(a.level) or None,
                          covariate_meanings=dict(c.split("=", 1) for c in a.covariate) or None)
    print("conforms" if report is not None and report.ok else "does not conform (see the log)")
    return 0 if report is not None and report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
