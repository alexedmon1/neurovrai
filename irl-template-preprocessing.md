---
author: Alex Edmondson
affiliation: CCHMC
email: alex.edmondson@cchmc.org
study: {{STUDY_NAME}}
study_code: {{STUDY_CODE}}               # short code, e.g. IRC805, STUDY01
modalities: [{{MODALITIES}}]             # subset of: T1w, T2w, DWI, fMRI, ASL
phase: preprocessing
project_dir: ~/research/{{STUDY_NAME}}/preprocessing
study_dir: /mnt/arborea/{{STUDY_NAME}}
pipeline_dir: /home/edm9fd/sandbox/neurovrai
pipeline_venv: /home/edm9fd/sandbox/neurovrai/.venv/bin/python
mni_template: /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz
---

<!-- AI Instructions:
Preprocessing-phase IRL project for a neurovrai human MRI study.
- $PROJECT (this repo) is small, git-tracked; holds plan, QC notes, activity log
- $STUDY on /mnt/arborea holds all big data and pipeline artifacts
- All preprocessing outputs (derivatives, transforms, QC HTML) go to $STUDY; never into $PROJECT
- neurovrai is invoked with `uv run python ...` from $PIPELINE; do NOT activate a venv and call python directly
- Long jobs (batch preproc, BEDPOSTX, eddy) MUST use nohup, logs to $STUDY/logs
- Before launching CPU/GPU-heavy jobs: `ps aux | grep -E '(eddy|bedpostx|probtrackx|python.*run_simple_pipeline)'`
- GPU check before eddy/BEDPOSTX: `nvidia-smi`
-->

# {{STUDY_NAME}} — Preprocessing Plan

## 📁 Paths — Single source of truth

### `$PROJECT` — this IRL project (small, git-tracked on home drive)

- **`$PROJECT`** — `~/research/{{STUDY_NAME}}/preprocessing` — this repo root
- **`$PLAN`** — `$PROJECT/plans` — main-plan.md, activity log, CSV log
- **`$QC_NOTES`** — `$PROJECT/qc-notes` — markdown notes on QC passes and exclusion decisions

### `$STUDY` — study data (big, on arborea, shared across all projects)

- **`$STUDY`** — `/mnt/arborea/{{STUDY_NAME}}` — study data root
- **`$DICOM`** — `$STUDY/raw/dicom` — raw DICOM files (per-subject subfolders)
- **`$BIDS`** — `$STUDY/raw/bids` — BIDS-converted NIfTI data
- **`$DERIV`** — `$STUDY/derivatives` — preprocessed per-subject outputs (`anat/`, `dwi/`, `func/`, `asl/`)
- **`$TRANSFORMS`** — `$STUDY/transforms` — per-subject spatial transforms (ANTs, FLIRT/FNIRT)
- **`$WORK`** — `$STUDY/work` — Nipype working dirs (safe to wipe after QC)
- **`$QC`** — `$STUDY/qc` — HTML QC reports per modality
- **`$EXCL`** — `$STUDY/exclusions` — canonical per-modality exclusion CSVs
- **`$LOGS`** — `$STUDY/logs` — nohup log destinations
- **`$CONFIG`** — `$STUDY/config.yaml` — pipeline configuration (authoritative)
- **`$MANIFEST`** — `$STUDY/study_manifest.json` — data inventory from `init_study.py`

### Pipeline (read-only)

- **`$PIPELINE`** — `/home/edm9fd/sandbox/neurovrai` — neurovrai repo
- **`$MNI`** — `/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz`

Rule: every section below refers to these by shorthand. If you need a new absolute path, add it here first.

---

## 🔧 First Time Setup — Run once when establishing the study

1. **Verify FSL/ANTs/CUDA environment**:
   ```bash
   cd $PIPELINE
   uv run python verify_environment.py
   echo $FSLDIR; nvidia-smi | head -5
   ```
2. **Initialize study** (creates `$STUDY/{raw,derivatives,work,qc,transforms,logs}` + `$CONFIG` + `$MANIFEST`):
   ```bash
   cd $PIPELINE
   uv run python scripts/init_study.py $STUDY \
       --name "{{STUDY_NAME}}" \
       --code {{STUDY_CODE}} \
       --dicom-root $DICOM
   # or --bids-root $BIDS if already BIDS-converted
   # add --freesurfer-dir ... if using FreeSurfer for connectivity
   ```
3. **Review `$CONFIG`** and edit for this study (TR, readout time, tedana mode, acompcor, use_cuda)
4. **Initialize exclusion CSVs** with headers `subject,session,reason,date_added`:
   ```bash
   for mod in anat dwi func asl; do
     echo "subject,session,reason,date_added" > $EXCL/${mod}_exclusions.csv
   done
   ```
5. **Snapshot `$CONFIG` and `$MANIFEST`** into `$PROJECT` (copy, don't symlink) so the version used by preprocessing is committed in this repo's git history
6. **Commit baseline** in `$PROJECT`: plan, config/manifest snapshot, empty QC notes

### Common skill library
<!-- Uncomment to use -->
<!-- Install Quarto: https://github.com/posit-dev/skills/tree/main/quarto/authoring -->

---

## ✅ Before Each Loop

- **Clean git tree** in `$PROJECT`: `git status`
- **Running-jobs check**: `ps aux | grep -E '(eddy|bedpostx|probtrackx|python.*run_simple_pipeline|python.*run_vbm|python.*run_tbss)'`
- **GPU free** (if launching eddy/BEDPOSTX): `nvidia-smi`
- **Disk check**: `df -h /mnt/arborea`
- **Pipeline version**: `cd $PIPELINE && git log -1` — record for reproducibility
- Any step that writes to `$DERIV` or `$TRANSFORMS` must be idempotent (re-run = no-op or byte-identical output)
- Only `## One-Time Instructions` is plan-editable without explicit permission

---

## 🔁 Instruction Loop — Define the preprocessing work for each iteration

<!-- 👤 AUTHOR AREA: Edit each loop. -->

### Loop task (current)

- **Phase:** <!-- DICOM→BIDS conversion | anat preproc | DWI preproc | fMRI preproc | ASL preproc | QC pass | exclusion triage -->
- **Subjects:** <!-- which batch (sub-001..sub-050) -->
- **Modality:** <!-- T1w / T2w / DWI / fMRI / ASL / all -->
- **Expected output:** <!-- $DERIV/{subject}/{mod}/, $TRANSFORMS/{subject}/ -->

### Command templates

**Per-subject, all modalities (DICOM input):**
```bash
cd $PIPELINE
nohup uv run python run_simple_pipeline.py \
    --subject {{SUBJECT}} \
    --dicom-dir $DICOM/{{SUBJECT}} \
    --config $CONFIG \
    --parallel-modalities \
    > $LOGS/{{SUBJECT}}_preproc_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Per-subject, all modalities (BIDS input):**
```bash
cd $PIPELINE
nohup uv run python run_simple_pipeline.py \
    --subject {{SUBJECT}} \
    --nifti-dir $BIDS/{{SUBJECT}} \
    --config $CONFIG \
    --parallel-modalities \
    > $LOGS/{{SUBJECT}}_preproc_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Per-subject, subset of modalities:**
```bash
cd $PIPELINE
nohup uv run python run_simple_pipeline.py \
    --subject {{SUBJECT}} --dicom-dir $DICOM/{{SUBJECT}} --config $CONFIG \
    --skip-func --skip-asl \
    > $LOGS/{{SUBJECT}}_preproc_anat_dwi_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Batch across subjects** (sequentially, one at a time to avoid GPU contention for eddy):
```bash
for s in $(cat subjects.txt); do
  cd $PIPELINE
  uv run python run_simple_pipeline.py --subject $s --dicom-dir $DICOM/$s --config $CONFIG \
      >> $LOGS/batch_preproc_$(date +%Y%m%d).log 2>&1
done
```

**BEDPOSTX** (required for structural connectivity; GPU strongly recommended):
```bash
cd $PIPELINE
nohup bedpostx_gpu $DERIV/{{SUBJECT}}/dwi -n 3 -w 1 -b 1000 \
    > $LOGS/{{SUBJECT}}_bedpostx_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Registration/QC pass:**
- Review HTML reports in `$QC` and `$DERIV/{subject}/{mod}/qc/`
- For each failure, add a row to the appropriate `$EXCL/*_exclusions.csv` with a concrete `reason`
- Record rationale in `$QC_NOTES/{{SUBJECT}}_{{MOD}}_qc.md`

### One-Time Instructions — Tasks that should only execute once

<!-- 👤 AUTHOR AREA: Add tasks. Move to Completed once done. -->

- [ ] `init_study.py` for `$STUDY`
- [ ] Validate DICOM/BIDS discovery (`--discover-only`)
- [ ] Customize `$CONFIG` for this study (TR, readout, tedana, acompcor)
- [ ] Initialize empty exclusion CSVs
- [ ] Snapshot `$CONFIG` + `$MANIFEST` into `$PROJECT`
- [ ] First full batch of anatomical preprocessing
- [ ] First full batch of DWI preprocessing (+ BEDPOSTX if needed)
- [ ] First full batch of fMRI preprocessing
- [ ] First full batch of ASL preprocessing (if in scope)
- [ ] First registration QC pass, per modality

#### Completed (don't re-run)
<!-- Move checked items here with date -->

### Formatting Guidelines

- **QC notes** → `$QC_NOTES/{subject}_{modality}_qc.md` with: date, QC items reviewed, failures + reason, exclusions added
- **Exclusion rows** — every row in `$EXCL/*.csv` needs a concrete `reason` string; expand rationale in `$QC_NOTES` if needed
- **Paths** — always shorthand from `## Paths`; never `../../`

---

## 📝 After Each Loop

- **Update activity log** (`$PLAN/main-plan-activity.md`, append 1–2 lines):
  - Phase, subjects processed, modalities, outputs produced
  - Timestamp (UTC), `$PROJECT` git hash, `$PIPELINE` git hash
  - Exclusions added this loop (count + pointer to `$QC_NOTES/`)

- **Update plan log** (`$PLAN/main-plan-log.csv`):
  `timestamp,phase,subject,modality,output_path,status,project_hash,pipeline_hash`

- **Commit `$PROJECT`** — plan edits, QC notes, log updates, config/manifest snapshots only
  - Never commit anything from `$STUDY`; `$STUDY` is not under this repo's git
  - Commit message: `preproc: {subject_range} {modality} — {outcome}`

- **Feedback to AUTHOR**:
  1. Phase progress, subjects remaining
  2. QC findings needing attention (motion, coverage, registration failures)
  3. Pipeline issues worth filing upstream in neurovrai

---

## 📚 Skill Library — Community skills (optional)
<!-- Uncomment to use -->

---

## 📌 Study-specific conventions

### Modality pipelines (neurovrai default)
- **Anatomical (T1w/T2w):** N4 bias → BET → Atropos seg → FLIRT/FNIRT to MNI
- **DWI:** TOPUP → GPU eddy → DTI fit → DKI/NODDI (multi-shell) → FMRIB58 normalization
- **fMRI:** MCFLIRT → TEDANA (multi-echo) or ICA-AROMA (single-echo) → ACompCor → bandpass → smooth → MNI normalize
- **ASL:** MCFLIRT → label-control → CBF quantification → M0 calibration → PVC

### Transform chain
```
Subject functional (BOLD / DWI / ASL) → Subject T1w → MNI152
                (FLIRT rigid/BBR)          (FNIRT)
```

### Exclusion system
- `$EXCL/*.csv` is the only source of truth for downstream analyses
- Never apply ad-hoc filters outside this system
- Every exclusion carries a `reason` string
- Detailed rationale lives in `$QC_NOTES`

### Handoff to analysis projects
When preprocessing is complete (all subjects, all in-scope modalities, exclusions populated, BEDPOSTX done if needed), tag a `preprocessing-v1` commit in `$PROJECT` and note it in the activity log. Analysis projects (see `neurovrai-analysis` template) will reference this snapshot.
