"""neurovrai's analysis folders: design records, randomise logs, and a whole folder that conforms.

No FSL: a randomise run is written by hand (maps and log), as neurofaune's tests do.
"""
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurovrai.analysis import results as R

SHAPE = (8, 7, 5)
AFFINE = np.diag([-2.0, 2.0, 2.0, 1.0])
LEVELS = {"mriglu": {"1": "Controlled", "2": "Uncontrolled"}}


def _save(path, arr):
    nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), AFFINE), str(path))


def _vest(path, M, names=()):
    M = np.atleast_2d(M)
    head = [f"/NumWaves {M.shape[1]}", f"/NumPoints {M.shape[0]}" if path.suffix == ".mat"
            else f"/NumContrasts {M.shape[0]}"]
    head += [f"/ContrastName{i + 1} {n}" for i, n in enumerate(names)]
    rows = "\n".join(" ".join(f"{v:.6f}" for v in r) for r in M)
    path.write_text("\n".join(head) + "\n/Matrix\n" + rows + "\n")


def _neuroaider_design(d, n=12, age_shift=0.0):
    """A design laid out as neuroaider writes it: sex, age (centred), mriglu_1, mriglu_2."""
    rng = np.random.default_rng(1)
    d.mkdir(parents=True, exist_ok=True)
    ids = [f"sub-{i:02d}" for i in range(n)]
    group = np.array([1] * (n // 2) + [2] * (n - n // 2))
    sex = rng.integers(1, 3, n)
    age = rng.integers(25, 45, n).astype(float)
    pd.DataFrame({"participant_id": ids, "mriglu": group, "sex": sex, "age": age}).to_csv(
        d / "participants_matched.tsv", sep="\t", index=False)
    X = np.column_stack([sex - sex.mean(), age + age_shift - (age + age_shift).mean(),
                         group == 1, group == 2]).astype(float)
    names = ["mriglu_positive", "mriglu_negative", "age_positive"]
    _vest(d / "design.mat", X)
    _vest(d / "design.con", np.array([[0, 0, 1, -1], [0, 0, -1, 1], [0, 1, 0, 0]], float), names)
    (d / "design_summary.json").write_text(json.dumps({
        "columns": ["sex", "age", "mriglu_1", "mriglu_2"], "contrasts": names,
        "covariates": [{"name": "sex", "mean_center": True}, {"name": "age", "mean_center": True}],
        "factors": [{"name": "mriglu", "coding": "dummy", "levels": [1, 2]}]}))
    return ids, group, X


def test_a_neuroaider_design_is_described_from_its_participants(tmp_path):
    ids, group, _ = _neuroaider_design(tmp_path / "d")
    rec = R.describe_neuroaider_design(tmp_path / "d", LEVELS, covariate_meanings={"age": "age at scan"})
    assert rec["rows"]["ids"] == ids
    assert rec["groups"] == {"Controlled": 6, "Uncontrolled": 6}
    c1, c2, c3 = rec["contrasts"]
    assert (c1["test_kind"], c1["group_a"], c1["group_b"]) == ("two_group", "Controlled", "Uncontrolled")
    assert c2["group_a"] == "Uncontrolled"
    assert c3["test_kind"] == "regression" and "age at scan" in c3["tests"]


def test_a_design_whose_age_is_not_the_participants_age_is_refused(tmp_path):
    """IRC805, 2026-11-16: the designs' age was another variable than the table beside them."""
    _neuroaider_design(tmp_path / "d", age_shift=0.0)
    m = (tmp_path / "d" / "design.mat").read_text().splitlines()
    m[3] = m[3].replace(m[3].split()[1], "9.000000", 1)          # one subject's age changed
    (tmp_path / "d" / "design.mat").write_text("\n".join(m) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        R.describe_neuroaider_design(tmp_path / "d", LEVELS)


def test_an_unnamed_level_is_refused(tmp_path):
    _neuroaider_design(tmp_path / "d")
    with pytest.raises(ValueError, match="what is mriglu = 2"):
        R.describe_neuroaider_design(tmp_path / "d", {"mriglu": {"1": "Controlled"}})


def _log(run_dir, flags, finished=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "randomise.log").write_text(
        f"randomise options: -i all.nii.gz -o {run_dir}/randomise -d design.mat -t design.con {flags} \n"
        "Loading Data: \n" + ("Finished, exiting.\n" if finished else "Starting permutation 12\n"))


def test_the_log_says_how_randomise_was_run(tmp_path):
    _log(tmp_path / "a", "-n 5000 -T --T2 -x")
    assert R.read_randomise_log(tmp_path / "a") | {"command": ""} == {
        "command": "", "tfce": "2-D TFCE", "n_permutations": 5000, "uncorrected_p": False}
    _log(tmp_path / "b", "-n 500 -T -v 5.0 --uncorrp")
    o = R.read_randomise_log(tmp_path / "b")
    assert (o["tfce"], o["variance_smoothing_mm"], o["uncorrected_p"]) == ("3-D TFCE", 5.0, True)
    _log(tmp_path / "c", "-n 500 -T", finished=False)
    with pytest.raises(ValueError, match="did not finish"):
        R.read_randomise_log(tmp_path / "c")


def _run(run_dir, data_file, X, C, mask, rng, uncorrp=True, flags="-n 5000 -T --T2"):
    """A randomise run on disk for two-group data with an effect in a block of the mask."""
    n = X.shape[0]
    data = rng.normal(0, 1, SHAPE + (n,))
    block = np.zeros(SHAPE, bool)
    block[2:5, 2:5, 2] = True
    data[block] += 2.0 * X[:, 2]
    data[~mask] = 0
    _save(data_file, data)
    _log(run_dir, flags + (" --uncorrp" if uncorrp else ""))
    for c in range(1, len(C) + 1):
        _save(run_dir / f"randomise_tstat{c}.nii.gz", np.where(mask, rng.normal(0, 1, SHAPE), 0))
        _save(run_dir / f"randomise_tfce_corrp_tstat{c}.nii.gz",
              np.where(block & (c == 1), 0.99, 0.3) * mask)
        if uncorrp:
            _save(run_dir / f"randomise_tfce_p_tstat{c}.nii.gz", np.where(block, 0.999, 0.4) * mask)


def test_a_two_measure_analysis_folder_conforms(tmp_path):
    from lumiere import check
    from neurofaune.analysis.stats.readout import Atlas

    rng = np.random.default_rng(0)
    design = tmp_path / "design"
    ids, group, X = _neuroaider_design(design)
    record = R.load_design_record(design, LEVELS, covariate_meanings={"age": "age at scan"})
    assert (design / "design.json").exists()
    assert R.load_design_record(design) == json.loads((design / "design.json").read_text())
    C = np.array([c["vector"] for c in record["contrasts"]])

    out = tmp_path / "tbss"
    stats = out / "stats"
    stats.mkdir(parents=True)
    mask = np.zeros(SHAPE, bool)
    mask[1:7, 1:6, 1:4] = True
    _save(stats / "mask.nii.gz", mask)
    runs = {}
    for m, uncorrp in (("FA", False), ("MD", True)):
        runs[m] = R.RandomiseRun(out / m / "randomise_ageyr", stats / f"all_{m}.nii.gz")
        _run(runs[m].run_dir, runs[m].data, X, C, mask, rng, uncorrp=uncorrp)
    parc = np.zeros(SHAPE, np.int32)
    parc[mask] = 1
    parc[4:][mask[4:]] = 2
    atlas = Atlas(parc, {1: "Tract L", 2: "Tract R"}, {1: "left", 2: "right"})

    rep = R.write_randomise_analysis(
        out, runs, design_dir=design, design_record=record, mask=stats / "mask.nii.gz",
        mask_name="TBSS skeleton", analysis_id="dwi/tbss/mriglu", title="t", description="d",
        analysis_type="tbss", modality="dwi", space=R.FSL_MNI, atlas=atlas, strict=True)
    assert rep.ok, rep.errors
    assert check(out)[0].ok

    a = json.loads((out / "analysis.json").read_text())
    assert a["inference"]["method"].startswith("FSL randomise, 2-D TFCE, 5000 permutations")
    assert any("FA were run without --uncorrp" in c for c in a["caveats"])
    assert a["design"]["groups"] == {"Controlled": 6, "Uncontrolled": 6}
    assert {m["axes"] for m in a["maps"]} == {"LAS"}
    prov = json.loads((out / "provenance.json").read_text())
    assert [g["Name"] for g in prov["generated_by"]][:2] == ["neurovrai", "neurofaune"]
    tests = pd.read_csv(out / "tests.csv")
    assert set(tests.metric) == {"FA", "MD"} and len(tests) == 6


def test_measures_run_differently_are_refused(tmp_path):
    rng = np.random.default_rng(0)
    design = tmp_path / "design"
    _, _, X = _neuroaider_design(design)
    record = R.load_design_record(design, LEVELS)
    C = np.array([c["vector"] for c in record["contrasts"]])
    mask = np.ones(SHAPE, bool)
    runs = {}
    for m, flags in (("FA", "-n 5000 -T --T2"), ("MD", "-n 5000 -T")):
        runs[m] = R.RandomiseRun(tmp_path / m, tmp_path / f"all_{m}.nii.gz")
        _run(runs[m].run_dir, runs[m].data, X, C, mask, rng, flags=flags)
    _save(tmp_path / "mask.nii.gz", mask)
    with pytest.raises(ValueError, match="different tfce"):
        R.write_randomise_analysis(tmp_path, runs, design_dir=design, design_record=record,
                                   mask=tmp_path / "mask.nii.gz", mask_name="m", analysis_id="dwi/tbss/x",
                                   title="t", description="d", analysis_type="tbss", modality="dwi",
                                   space="S")


def test_combined_atlases_keep_the_first_and_shift_the_rest():
    from neurofaune.analysis.stats.readout import Atlas

    a = Atlas(np.array([[1, 0], [0, 0]], np.int32), {1: "Cortex A"})
    b = Atlas(np.array([[3, 3], [0, 4]], np.int32), {3: "Left Thalamus", 4: "Right Putamen"},
              {3: "left", 4: "right"})
    c = R.combine_atlases(a, b)
    assert c.parcellation.tolist() == [[1, 103], [0, 104]]
    assert c.names == {1: "Cortex A", 103: "Left Thalamus", 104: "Right Putamen"}
    assert c.hemispheres == {103: "left", 104: "right"}
