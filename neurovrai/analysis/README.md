# neurovrai.analysis

**Group-level statistical analyses for neuroimaging data.**

## Overview

This module provides group-level analyses following a two-stage pipeline pattern:

| Analysis | Stage 1 (Preparation) | Stage 2 (Statistics) | Status |
|----------|----------------------|---------------------|--------|
| **VBM** | Tissue normalization, smoothing | FSL randomise / nilearn GLM | Production |
| **TBSS** | FSL TBSS pipeline (1-4) | FSL randomise | Production |
| **ReHo/fALFF** | Metric computation, z-scoring | FSL randomise / nilearn GLM | Production |
| **MELODIC** | Group ICA | Dual regression | Production |
| **ASL** | CBF normalization | FSL randomise | Planned |

## Module Structure

```
analysis/
├── anat/
│   └── vbm_workflow.py        # Voxel-Based Morphometry
├── tbss/
│   ├── prepare_tbss.py        # TBSS data preparation
│   └── run_tbss_stats.py      # TBSS statistical analysis
├── func/
│   ├── reho.py                # Regional Homogeneity
│   ├── falff.py               # fALFF computation
│   ├── resting_workflow.py    # Combined ReHo + fALFF
│   ├── melodic.py             # Group ICA
│   └── dual_regression.py     # Dual regression
├── stats/
│   ├── design_matrix.py       # Design matrix generation
│   ├── randomise_wrapper.py   # FSL randomise wrapper
│   ├── nilearn_glm.py         # Nilearn GLM alternative
│   ├── cluster_report.py      # Cluster extraction
│   └── enhanced_cluster_report.py  # Atlas-based reporting
└── utils/
    ├── design_validation.py   # Design-to-data validation
    └── modality_subjects.py   # Subject discovery
```

---

## VBM (Voxel-Based Morphometry)

Analyze structural brain differences at the voxel level.

### Python API

```python
from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data, run_vbm_analysis
from pathlib import Path

# Stage 1: Prepare data
prepare_vbm_data(
    subjects=['sub-001', 'sub-002', 'sub-003'],
    derivatives_dir=Path('/derivatives'),
    output_dir=Path('/analysis/vbm'),
    tissue_type='GM',        # GM, WM, or CSF
    smoothing_fwhm=4.0       # mm
)

# Stage 2: Run statistics
# Option A: FSL randomise (nonparametric, TFCE correction)
run_vbm_analysis(
    vbm_dir=Path('/analysis/vbm/GM'),
    participants_file=Path('participants.csv'),
    formula='age + sex + group',
    contrasts={'age_positive': [0, 1, 0, 0]},
    method='randomise',
    n_permutations=5000
)

# Option B: nilearn GLM (parametric, FDR/Bonferroni correction)
run_vbm_analysis(
    vbm_dir=Path('/analysis/vbm/GM'),
    participants_file=Path('participants.csv'),
    formula='age + sex + group',
    contrasts={'age_positive': [0, 1, 0, 0]},
    method='glm',
    z_threshold=2.3
)
```

### CLI Usage

```bash
uv run python scripts/analysis/run_vbm_group_analysis.py \
    --study-root /mnt/data/my_study \
    --method randomise \
    --tissue GM \
    --n-permutations 5000
```

### Outputs

```
analysis/vbm/GM/
├── smoothed/
│   └── sub-*_GM_smooth.nii.gz      # Smoothed tissue maps
├── randomise_output/                # FSL randomise
│   ├── randomise_tfce_corrp_tstat1.nii.gz
│   └── randomise_tstat1.nii.gz
├── glm_output/                      # nilearn GLM
│   ├── contrast_name_z_map.nii.gz
│   └── fdr_corrected/
└── cluster_reports/
    └── contrast_report.html
```

---

## TBSS (Tract-Based Spatial Statistics)

White matter analysis using DTI metrics projected onto a skeleton.

### Python API

```python
from neurovrai.analysis.tbss.prepare_tbss import prepare_tbss_data
from neurovrai.analysis.tbss.run_tbss_stats import run_tbss_statistics
from pathlib import Path

# Stage 1: Prepare TBSS data
prepare_tbss_data(
    derivatives_dir=Path('/derivatives'),
    output_dir=Path('/analysis/tbss'),
    subjects=['sub-001', 'sub-002'],  # or None for auto-discovery
    metric='FA'  # FA, MD, AD, RD
)

# Stage 2: Run statistics
run_tbss_statistics(
    tbss_dir=Path('/analysis/tbss'),
    design_dir=Path('/data/designs/tbss'),  # Pre-generated designs
    n_permutations=5000
)
```

### CLI Usage

```bash
# Prepare data
uv run python -m neurovrai.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --metric FA \
    --output-dir /analysis/tbss

# Run statistics
uv run python -m neurovrai.analysis.tbss.run_tbss_stats \
    --tbss-dir /analysis/tbss \
    --design-dir /data/designs/tbss \
    --n-permutations 5000
```

### FSL TBSS Pipeline

The workflow runs the standard FSL TBSS steps:
1. `tbss_1_preproc` - Preprocessing and erosion
2. `tbss_2_reg -T` - Registration to FMRIB58_FA
3. `tbss_3_postreg -S` - Create mean FA and skeleton
4. `tbss_4_prestats 0.2` - Project onto skeleton

### Outputs

```
analysis/tbss/
├── origdata/                        # Original FA images
├── FA/
│   ├── all_FA_skeletonised.nii.gz   # 4D skeleton volume
│   └── mean_FA_skeleton.nii.gz      # Skeleton mask
├── randomise_output/
│   └── randomise_tfce_corrp_tstat1.nii.gz
├── subject_manifest.json            # Included/excluded subjects
└── subject_list.txt                 # Subject order
```

---

## Resting-State fMRI (ReHo/fALFF)

Compute regional homogeneity and fractional amplitude of low-frequency fluctuations.

### Python API

```python
from neurovrai.analysis.func.resting_workflow import run_resting_state_analysis
from neurovrai.analysis.func.reho import compute_reho
from neurovrai.analysis.func.falff import compute_falff
from pathlib import Path

# Combined workflow
results = run_resting_state_analysis(
    func_file=Path('preprocessed_bold.nii.gz'),
    mask_file=Path('brain_mask.nii.gz'),
    output_dir=Path('/derivatives/sub-001/func'),
    compute_reho=True,
    compute_falff=True,
    neighborhood=27,        # ReHo: 7, 19, or 27 voxels
    freq_range=(0.01, 0.08) # fALFF frequency band
)

# Individual metrics
reho_map = compute_reho(
    func_file=Path('preprocessed_bold.nii.gz'),
    mask_file=Path('brain_mask.nii.gz'),
    neighborhood=27
)

falff_map = compute_falff(
    func_file=Path('preprocessed_bold.nii.gz'),
    mask_file=Path('brain_mask.nii.gz'),
    tr=2.0,
    freq_range=(0.01, 0.08)
)
```

### Performance

| Metric | Voxels | Time |
|--------|--------|------|
| ReHo | 136,000 | ~7 min |
| fALFF | 136,000 | ~22 sec |

### Outputs

```
derivatives/{subject}/func/
├── reho.nii.gz           # Regional homogeneity map
├── reho_z.nii.gz         # Z-score normalized
├── falff.nii.gz          # fALFF map
└── falff_z.nii.gz        # Z-score normalized
```

---

## MELODIC (Group ICA)

Group-level independent component analysis for resting-state networks.

### Python API

```python
from neurovrai.analysis.func.melodic import run_melodic_group_analysis
from neurovrai.analysis.func.dual_regression import run_dual_regression
from pathlib import Path

# Group ICA
run_melodic_group_analysis(
    derivatives_dir=Path('/derivatives'),
    output_dir=Path('/analysis/melodic'),
    subjects=['sub-001', 'sub-002', 'sub-003'],
    expected_tr=2.0,
    n_components=20,  # Or None for automatic
    approach='concat'  # 'concat' or 'tensor'
)

# Dual regression for subject-specific maps
run_dual_regression(
    melodic_dir=Path('/analysis/melodic'),
    func_files=[Path('sub-001/func/preprocessed.nii.gz'), ...],
    output_dir=Path('/analysis/melodic/dual_regression'),
    des_norm=True
)
```

### CLI Usage

```bash
uv run python scripts/run_melodic_irc805.py \
    --derivatives-dir /derivatives \
    --output-dir /analysis/melodic \
    --n-components 20
```

### Outputs

```
analysis/melodic/
├── melodic_IC.nii.gz     # Independent components
├── melodic_mix           # Time courses
├── melodic_FTmix         # Frequency spectra
├── report.html           # Component visualization
└── dual_regression/
    ├── dr_stage1_*.txt   # Subject time courses
    └── dr_stage2_*.nii.gz # Subject spatial maps
```

---

## Statistical Framework

### Design Matrix Generation

```python
from neurovrai.analysis.stats.design_matrix import create_design_matrix

design, contrasts = create_design_matrix(
    participants_file=Path('participants.csv'),
    formula='age + sex + group',
    output_dir=Path('/analysis/designs'),
    demean_continuous=True
)
```

### FSL Randomise Wrapper

```python
from neurovrai.analysis.stats.randomise_wrapper import run_randomise

run_randomise(
    input_4d=Path('all_FA_skeletonised.nii.gz'),
    design_mat=Path('design.mat'),
    design_con=Path('design.con'),
    output_prefix='randomise',
    mask=Path('mean_FA_skeleton.nii.gz'),
    n_permutations=5000,
    tfce=True
)
```

### Nilearn GLM Alternative

```python
from neurovrai.analysis.stats.nilearn_glm import run_second_level_glm

results = run_second_level_glm(
    input_4d=Path('GM_smooth_4d.nii.gz'),
    design_matrix=design_df,
    contrast_def={'age': [0, 1, 0, 0]},
    output_dir=Path('/analysis/glm'),
    z_threshold=2.3
)
```

### Enhanced Cluster Reporting

```python
from neurovrai.analysis.stats.enhanced_cluster_report import EnhancedClusterReport

report = EnhancedClusterReport(
    stat_map=Path('randomise_tstat1.nii.gz'),
    corrp_map=Path('randomise_tfce_corrp_tstat1.nii.gz'),
    atlas='jhu',  # or 'harvard-oxford'
    threshold=0.95,
    output_dir=Path('/analysis/cluster_reports')
)
report.generate_report()
```

---

## Atlas Support

| Atlas | Tissue | Analysis | Regions |
|-------|--------|----------|---------|
| Harvard-Oxford | Grey matter | VBM, ReHo, fALFF | 48 cortical + 21 subcortical |
| JHU ICBM-DTI-81 | White matter | TBSS | 48 white matter tracts |

---

## Configuration

Add analysis parameters to your `config.yaml`:

```yaml
analysis:
  n_permutations: 5000
  threshold: 0.05

  vbm:
    tissue_type: GM
    smooth_fwhm: 4.0
    modulated: true

  tbss:
    skeleton_threshold: 0.2

  resting_state:
    reho_neighborhood: 27
    falff_freq_range: [0.01, 0.08]

  melodic:
    n_components: 20
    approach: concat
```

---

## Workflow Pattern

All analysis modules follow a two-stage pattern:

### Stage 1: Data Preparation
- Discover subjects with completed preprocessing
- Validate data completeness
- Prepare metrics (normalize, smooth, concatenate)
- Generate subject manifest
- Output: 4D volume + subject list

### Stage 2: Statistical Analysis
- Load pre-generated design matrices
- Validate design-to-data alignment
- Run FSL randomise or nilearn GLM
- Extract significant clusters
- Generate HTML reports with atlas localization
- Output: Statistical maps + cluster reports

This separation allows:
- Testing preparation independently of statistics
- Running multiple statistical models on same data
- Clear provenance tracking
- Easier debugging

---

## References

- **FSL randomise**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise
- **FSL TBSS**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS
- **FSL MELODIC**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC
- **TFCE**: Smith & Nichols, NeuroImage 2009
- **ReHo**: Zang et al., NeuroImage 2004
- **fALFF**: Zou et al., J Neurosci Methods 2008
