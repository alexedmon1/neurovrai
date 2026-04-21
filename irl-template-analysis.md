---
author: Alex Edmondson
affiliation: CCHMC
email: alex.edmondson@cchmc.org
study: {{STUDY_NAME}}
project_name: {{PROJECT_NAME}}        # e.g. "vbm-age", "tbss-group", "fc-default-mode"
project_slug: {{PROJECT_SLUG}}        # filesystem-safe: matches project_dir leaf
phase: analysis
modalities: [{{MODALITIES}}]          # subset of: T1w, DWI, fMRI, ASL
project_dir: ~/research/{{STUDY_NAME}}/{{PROJECT_SLUG}}
study_dir: /mnt/arborea/{{STUDY_NAME}}
pipeline_dir: /home/edm9fd/sandbox/neurovrai
pipeline_venv: /home/edm9fd/sandbox/neurovrai/.venv/bin/python
---

<!-- AI Instructions:
Analysis-phase IRL project for a neurovrai human MRI study.
- Preprocessing is DONE — $DERIV, $TRANSFORMS, $EXCL are populated and maintained by the preprocessing project
- $PROJECT (this repo) is small, git-tracked; holds plan + result writeups + per-project scripts
- $STUDY on /mnt/arborea holds all big data, SHARED across projects
- This project's analysis outputs go to $ANALYSIS_OUT and $CONN_OUT (namespaced by project slug)
- Result summaries (markdown, small figures, small tables) live in $RESULTS inside $PROJECT
- neurovrai is invoked with `uv run python ...` from $PIPELINE
- Long jobs (randomise, probtrackx2, permutation NBS) MUST use nohup, logs to $STUDY/logs
- Before launching CPU/GPU-heavy jobs: `ps aux | grep -E '(randomise|probtrackx|python.*run_vbm|python.*run_tbss)'`
-->

# {{PROJECT_NAME}} ({{STUDY_NAME}}) — Analysis Plan

## 📁 Paths — Single source of truth

### `$PROJECT` — this IRL project (small, git-tracked on home drive)

- **`$PROJECT`** — `~/research/{{STUDY_NAME}}/{{PROJECT_SLUG}}` — this repo root
- **`$PLAN`** — `$PROJECT/plans` — main-plan.md, activity log, CSV log
- **`$RESULTS`** — `$PROJECT/results` — markdown summaries, figures, small tables
- **`$SCRIPTS`** — `$PROJECT/scripts` — wrappers specific to this analysis
- **`$DESIGNS`** — `$PROJECT/designs` — FSL design matrices, participant CSVs

### `$STUDY` — study data (big, on arborea, shared across projects)

- **`$STUDY`** — `/mnt/arborea/{{STUDY_NAME}}`
- **`$BIDS`** — `$STUDY/raw/bids` — read-only
- **`$DERIV`** — `$STUDY/derivatives` — preprocessed subjects (maintained by preprocessing project)
- **`$TRANSFORMS`** — `$STUDY/transforms`
- **`$EXCL`** — `$STUDY/exclusions` — canonical CSVs (shared, not edited here)
- **`$ANALYSIS_OUT`** — `$STUDY/analysis/{{PROJECT_SLUG}}` — VBM/TBSS/resting outputs, namespaced to THIS project
- **`$CONN_OUT`** — `$STUDY/connectivity/{{PROJECT_SLUG}}` — functional + structural connectivity outputs
- **`$LOGS`** — `$STUDY/logs`
- **`$CONFIG`** — `$STUDY/config.yaml`

### Pipeline (read-only)

- **`$PIPELINE`** — `/home/edm9fd/sandbox/neurovrai`

Rule: every section below refers to these by shorthand. Add new paths here before using them.

---

## 🔧 First Time Setup — Run once when starting this analysis

<!-- 👤 AUTHOR AREA: Fill in scope before first loop -->

### Research question
<!-- 2–4 sentences: what does this project investigate? What's the hypothesis? -->

### In-scope
- **Modalities:** <!-- T1w (VBM), DWI (TBSS, structural FC), fMRI (resting, network, task), ASL (CBF) -->
- **Primary analyses:** <!-- VBM, TBSS, ReHo/fALFF, MELODIC, functional connectivity, structural connectivity, NBS, graph metrics -->
- **Atlases:** <!-- HarvardOxford cort, HarvardOxford sub, Juelich, Schaefer_200, Desikan-Killiany -->
- **Contrasts / targets:** <!-- group comparison, regression target (age, cognitive score), interaction -->

### Out-of-scope
<!-- What this project will NOT touch; prevents scope creep -->

### Verify preprocessing is complete
```bash
ls $DERIV | wc -l                      # expect: all subjects preprocessed
ls $DERIV/sub-*/anat/brain.nii.gz | wc -l  # T1w coverage
ls $DERIV/sub-*/dwi/dti/FA.nii.gz | wc -l  # DWI coverage
ls $DERIV/sub-*/func/preprocessed_bold.nii.gz | wc -l  # fMRI coverage
ls $EXCL/*.csv                         # expect: populated CSVs per modality
```
If any are missing, redirect to the preprocessing project before proceeding.

### Create project-namespaced output dirs
```bash
mkdir -p $ANALYSIS_OUT $CONN_OUT $DESIGNS
```

### Common skill library
<!-- Uncomment to use -->
<!-- Install Scientific Writing: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/scientific-writing -->
<!-- Install PubMed Search: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/pubmed-database -->
<!-- Install PPTX Posters: https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/pptx-posters -->

---

## ✅ Before Each Loop

- **Clean git tree** in `$PROJECT`: `git status`
- **Running-jobs check**: `ps aux | grep -E '(randomise|probtrackx|python.*run_vbm|python.*run_tbss|python.*batch_functional)'`
- **Disk check**: `df -h /mnt/arborea`
- **Exclusions current**: `ls -la $EXCL/*.csv` — changes since last loop? (Maintained by preprocessing project.)
- **Pipeline version**: `cd $PIPELINE && git log -1` — record hash
- Scripts must be **idempotent** — re-running produces identical output or no-op
- Only `## One-Time Instructions` is plan-editable without explicit permission

---

## 🔁 Instruction Loop — Define the work for each iteration

<!-- 👤 AUTHOR AREA: Edit each loop. -->

### Loop task (current)

- **Analysis:** <!-- VBM GM, TBSS FA, ReHo, fALFF, MELODIC, functional FC (HarvardOxford), structural FC (Schaefer_200), NBS, graph metrics -->
- **Subjects:** <!-- all / subset -->
- **Tissue / metric:** <!-- GM, FA, MD, ReHo, fALFF, coherence, partial correlation -->
- **Contrast / target:** <!-- group comparison, regression, interaction -->
- **Exclusion CSV:** <!-- $EXCL/dwi_exclusions.csv, $EXCL/func_exclusions.csv, etc. -->
- **Output dir:** <!-- $ANALYSIS_OUT/{vbm_GM, tbss_FA, reho} or $CONN_OUT/{func, struct} -->

### Command templates

**VBM (voxel-based morphometry):**
```bash
cd $PIPELINE
nohup uv run python scripts/analysis/run_vbm_group_analysis.py \
    --study-root $STUDY \
    --output-dir $ANALYSIS_OUT/vbm_{{TISSUE}} \
    --method randomise \
    --tissue {{TISSUE}} \
    --design $DESIGNS/{{DESIGN}}.mat \
    --contrast $DESIGNS/{{DESIGN}}.con \
    --n-permutations 5000 \
    > $LOGS/vbm_{{TISSUE}}_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**TBSS (tract-based spatial statistics):**
```bash
cd $PIPELINE
nohup uv run python -m neurovrai.analysis.tbss.prepare_tbss \
    --derivatives-dir $DERIV --output-dir $ANALYSIS_OUT/tbss \
    --metric {{METRIC}} \
    > $LOGS/tbss_prep_{{METRIC}}_$(date +%Y%m%d_%H%M).log 2>&1 &

# after prepare completes:
nohup uv run python -m neurovrai.analysis.tbss.run_tbss_stats \
    --tbss-dir $ANALYSIS_OUT/tbss \
    --design-dir $DESIGNS/tbss \
    --n-permutations 5000 \
    > $LOGS/tbss_stats_{{METRIC}}_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Resting-state (ReHo / fALFF):**
```bash
cd $PIPELINE
uv run python scripts/run_reho_falff_analysis.py \
    --study-root $STUDY \
    --output-dir $ANALYSIS_OUT/resting \
    --exclusion-csv $EXCL/func_exclusions.csv \
    > $LOGS/resting_$(date +%Y%m%d_%H%M).log 2>&1
```

**MELODIC (group ICA):**
```bash
cd $PIPELINE
nohup uv run python scripts/run_melodic_analysis.py \
    --derivatives-dir $DERIV \
    --output-dir $ANALYSIS_OUT/melodic \
    --exclusion-csv $EXCL/func_exclusions.csv \
    --n-components {{N_COMPONENTS}} \
    > $LOGS/melodic_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Functional connectivity (batch):**
```bash
cd $PIPELINE
nohup uv run python -m neurovrai.connectome.batch_functional_connectivity \
    --study-root $STUDY \
    --atlases {{ATLAS_LIST}} \
    --output-dir $CONN_OUT/func \
    --exclusion-csv $EXCL/func_exclusions.csv \
    > $LOGS/fc_batch_$(date +%Y%m%d_%H%M).log 2>&1 &
```

**Structural connectivity (per subject, GPU):**
```bash
cd $PIPELINE
nohup uv run python -m neurovrai.connectome.run_structural_connectivity \
    --subject {{SUBJECT}} \
    --derivatives-dir $DERIV \
    --atlas {{ATLAS}} \
    --config $CONFIG \
    --output-dir $CONN_OUT/struct \
    --batch-mode --use-gpu \
    > $LOGS/sc_{{SUBJECT}}_{{ATLAS}}_$(date +%Y%m%d_%H%M).log 2>&1 &
```

### One-Time Instructions — Tasks that should only execute once

<!-- 👤 AUTHOR AREA: Add tasks. Move to Completed once done. -->

- [ ] Build participants CSV + covariate table (age, sex, group, cognitive scores) → `$DESIGNS/participants.csv`
- [ ] Generate FSL design matrices for primary contrast → `$DESIGNS/*.mat|*.con`
- [ ] Validate atlas registration for all non-excluded subjects
- [ ] Draft preregistration / analysis plan summary in `$RESULTS/00_analysis_plan.md`

#### Completed (don't re-run)
<!-- Move checked items here with date -->

### Formatting Guidelines

- **Result summaries** → `$RESULTS/{analysis_name}.md` with: design, N, exclusions applied, significant clusters/edges (p_FWE<0.05), effect direction, figure refs
- **Figures** → `$RESULTS/figures/` as PNG/SVG; never inline binary blobs
- **Tables** → `$RESULTS/tables/` as small CSVs; render markdown summary table in the `.md`
- **Large outputs** (NIFTI maps, full HTML reports, connectivity matrices) live in `$ANALYSIS_OUT` / `$CONN_OUT`, not in `$PROJECT`
- **Paths in reports** — always shorthand from `## Paths` or absolute; never `../../`
- **Number formatting** — p-values 3 sig figs, effect sizes 2 decimals, N as integer

---

## 📝 After Each Loop

- **Update activity log** (`$PLAN/main-plan-activity.md`, append 1–2 lines):
  - Analysis, design, output path
  - Timestamp (UTC), `$PROJECT` hash, `$PIPELINE` hash
  - Sessions excluded beyond canonical CSVs (with reason)

- **Update plan log** (`$PLAN/main-plan-log.csv`):
  `timestamp,analysis,modality,metric,target,n_subjects,output_dir,status,project_hash,pipeline_hash`

- **Commit `$PROJECT`** — plan edits, new result writeups, figures, tables, scripts, designs
  - Never commit anything from `$STUDY`; only `$PROJECT` is under this repo's git
  - Message format: `{analysis}: {one-line result}` (e.g. `VBM GM: sig cluster in right hippocampus`)

- **Feedback to AUTHOR**:
  1. What was done, results summary, next steps
  2. Idempotency or stale `## One-Time Instructions` issues
  3. Critical reasoning errors or QC concerns
  4. Pipeline quirks worth filing upstream in neurovrai

---

## 📚 Skill Library — Community skills (optional)
<!-- Uncomment to use -->
<!-- Install Scientific Writing -->
<!-- Install BioRx Search -->
<!-- Install Flowcharts -->

---

## 📌 Study-specific conventions

### Exclusion system
- `$EXCL/*.csv` is the only source of truth (maintained by the preprocessing project)
- Scripts consume via `--exclusion-csv`
- Never apply ad-hoc filters
- Flag any exclusion that would change N by >5% for author review before proceeding

### Design naming
<!-- 👤 AUTHOR AREA: Define the naming scheme for this project's designs -->
- Voxelwise: `{contrast}_{covariates}` (e.g. `group_age_sex`)
- Connectivity: `{atlas}_{metric}_{contrast}`

### Output namespacing
All outputs that this project writes to `$STUDY` go under `{{PROJECT_SLUG}}`:
- `$STUDY/analysis/{{PROJECT_SLUG}}/...`
- `$STUDY/connectivity/{{PROJECT_SLUG}}/...`

This prevents collisions with other analysis projects on the same study.
