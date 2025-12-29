# neurovrai.analysis

**Group-level statistical analyses for neuroimaging data.**

## Overview

This module provides group-level analyses following a two-stage pipeline pattern:

| Analysis | Stage 1 (Preparation) | Stage 2 (Statistics) | Status |
|----------|----------------------|---------------------|--------|
| **VBM** | Tissue normalization, smoothing | FSL randomise / nilearn GLM | Production |
| **WMH** | T2w normalization, WM masking | Intensity thresholding, tract analysis | Production |
| **T1-T2-ratio** | Ratio computation, WM masking | FSL randomise / nilearn GLM | Production |
| **TBSS** | FSL TBSS pipeline (1-4) | FSL randomise | Production |
| **ReHo/fALFF** | Metric computation, z-scoring | FSL randomise / nilearn GLM | Production |
| **MELODIC** | Group ICA | Dual regression | Production |
| **ASL** | CBF normalization | FSL randomise | Planned |

## Module Structure

```
analysis/
├── anat/
│   ├── vbm_workflow.py        # Voxel-Based Morphometry
│   ├── wmh_detection.py       # WMH detection algorithm
│   ├── wmh_workflow.py        # WMH analysis orchestration
│   ├── wmh_reporting.py       # WMH HTML report generation
│   ├── t1t2ratio_workflow.py  # T1w/T2w ratio analysis
│   └── t1t2ratio_reporting.py # T1-T2-ratio HTML report generation
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

## WMH (White Matter Hyperintensity Detection)

Detect and quantify white matter hyperintensities using T2w imaging with tract-level analysis.

### Processing Pipeline

1. **T2w Input**: Use preprocessed T2w from `t2w_preprocess.py` if available, otherwise register raw T2w → T1w
2. **MNI Normalization**: Apply existing T1w→MNI ANTs transform to T2w
3. **Clean WM Mask**: Create WM mask excluding CSF (with dilatable buffer) and GM
4. **WMH Detection**: Intensity thresholding within WM (mean + SD × threshold)
5. **Lesion Labeling**: Connected component analysis with minimum cluster filtering
6. **Tract Analysis**: Map lesions to JHU ICBM-DTI-81 white matter atlas

**Note**: When preprocessed T2w is found at `{derivatives}/{subject}/t2w/registered/t2w_to_t1w.nii.gz`, the WMH workflow uses it directly and skips the internal registration step. This ensures consistency with other analyses (e.g., T1-T2-ratio) and improves efficiency.

### Python API

```python
from neurovrai.analysis.anat.wmh_workflow import (
    run_wmh_analysis_single,
    run_wmh_analysis_batch,
    generate_group_summary
)
from neurovrai.analysis.anat.wmh_reporting import generate_wmh_html_report
from pathlib import Path

# Single subject
results = run_wmh_analysis_single(
    subject='sub-001',
    study_root=Path('/mnt/data/study'),
    output_dir=Path('/mnt/data/study/hyperintensities'),
    sd_threshold=3.0,        # Detection sensitivity (lower = more sensitive)
    min_cluster_size=5,      # Minimum voxels per lesion
    csf_dilate=1,            # CSF buffer iterations (~2mm each)
    gm_dilate=0              # GM buffer iterations (optional)
)

print(f"Detected {results['wmh_summary']['n_lesions']} lesions")
print(f"Total volume: {results['wmh_summary']['total_volume_mm3']:.2f} mm³")

# Batch processing (parallel)
batch_results = run_wmh_analysis_batch(
    study_root=Path('/mnt/data/study'),
    output_dir=Path('/mnt/data/study/hyperintensities'),
    n_jobs=4,
    sd_threshold=3.0
)

# Generate group summary and HTML report
group_df = generate_group_summary(Path('/mnt/data/study/hyperintensities'))
report_path = generate_wmh_html_report(Path('/mnt/data/study/hyperintensities'))
```

### CLI Usage

```bash
# Single subject
uv run python -m neurovrai.analysis.anat.wmh_workflow single \
    --subject sub-001 \
    --study-root /mnt/data/study \
    --output-dir /mnt/data/study/hyperintensities \
    --sd-threshold 3.0 \
    --min-cluster-size 5 \
    --csf-dilate 1

# Batch processing
uv run python -m neurovrai.analysis.anat.wmh_workflow batch \
    --study-root /mnt/data/study \
    --output-dir /mnt/data/study/hyperintensities \
    --n-jobs 4

# Generate HTML report
uv run python -m neurovrai.analysis.anat.wmh_workflow report \
    --hyperintensities-dir /mnt/data/study/hyperintensities
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sd_threshold` | 3.0 | Detection threshold (mean + SD × threshold) |
| `min_cluster_size` | 5 | Minimum voxels per lesion |
| `wm_threshold` | 0.7 | WM probability for inclusion |
| `csf_exclude_threshold` | 0.1 | CSF probability for exclusion |
| `csf_dilate` | 1 | CSF dilation iterations (~2mm buffer each) |
| `gm_exclude_threshold` | 0.5 | GM probability for exclusion |
| `gm_dilate` | 0 | GM dilation iterations (optional) |

### Input Requirements

```
{study_root}/
├── bids/{subject}/anat/
│   └── *T2W*.nii.gz                   # T2w image(s)
├── derivatives/{subject}/anat/
│   ├── brain/*_brain.nii.gz           # T1w brain-extracted
│   └── segmentation/
│       ├── csf.nii.gz                 # CSF probability map
│       ├── gm.nii.gz                  # GM probability map
│       └── wm.nii.gz                  # WM probability map
└── transforms/{subject}/
    └── t1w-mni-composite.h5           # T1w→MNI ANTs transform
```

### Outputs

**Per Subject** (`{output_dir}/{subject}/`):
```
├── t2w_mni.nii.gz           # T2w normalized to MNI
├── wm_mask_mni.nii.gz       # Clean WM mask in MNI
├── wmh_mask.nii.gz          # Binary WMH detection mask
├── wmh_labeled.nii.gz       # Labeled lesion map (IDs: 1, 2, 3, ...)
├── wmh_metrics.json         # Complete results with metadata
├── wmh_tract_counts.csv     # Per-tract lesion counts and volumes
├── wmh_lesions.csv          # Per-lesion detailed metrics
└── wmh_visualization.png    # Multi-slice overlay
```

**Group Level** (`{output_dir}/group/`):
```
├── wmh_summary.csv          # Per-subject summary statistics
├── wmh_by_tract.csv         # Aggregated tract-wise statistics
└── wmh_report.html          # Interactive HTML report
```

---

## T1-T2-ratio (T1w/T2w Ratio Analysis)

Compute T1w/T2w ratio as a proxy for myelin content in white matter. Based on [Du et al. 2019 (PMID: 30408230)](https://pubmed.ncbi.nlm.nih.gov/30408230/).

### Processing Pipeline

1. **Load Preprocessed Data**: T1w bias-corrected and T2w registered to T1w from derivatives
2. **Compute Ratio**: T1w / T2w in native space (with T2w threshold for zero handling)
3. **MNI Normalization**: Apply existing T1w→MNI transform to ratio map
4. **WM Masking**: Create WM mask from tissue probability maps (threshold 0.5)
5. **Smoothing**: Apply 4mm FWHM Gaussian kernel
6. **Group Statistics**: FSL randomise or nilearn GLM within WM mask

### Python API

```python
from neurovrai.analysis.anat.t1t2ratio_workflow import (
    prepare_t1t2ratio_single,
    prepare_t1t2ratio_batch,
    run_t1t2ratio_analysis
)
from neurovrai.analysis.anat.t1t2ratio_reporting import generate_t1t2ratio_html_report
from pathlib import Path

# Single subject preparation
results = prepare_t1t2ratio_single(
    subject='sub-001',
    study_root=Path('/mnt/data/study'),
    output_dir=Path('/mnt/data/study/analysis/t1t2ratio'),
    wm_threshold=0.5,        # WM probability threshold
    smooth_fwhm=4.0,         # Smoothing kernel (mm)
    overwrite=False
)

print(f"Mean ratio: {results['statistics']['ratio_mean']:.3f}")
print(f"WM voxels: {results['statistics']['n_wm_voxels']}")

# Batch processing (parallel)
batch_results = prepare_t1t2ratio_batch(
    study_root=Path('/mnt/data/study'),
    output_dir=Path('/mnt/data/study/analysis/t1t2ratio'),
    subjects=None,           # Auto-discover subjects
    n_jobs=4,
    wm_threshold=0.5,
    smooth_fwhm=4.0
)

# Group statistics (requires design matrices)
stats_results = run_t1t2ratio_analysis(
    t1t2ratio_dir=Path('/mnt/data/study/analysis/t1t2ratio'),
    design_dir=Path('/mnt/data/study/designs/t1t2ratio'),
    method='randomise',      # 'randomise' or 'glm'
    n_permutations=5000,
    tfce=True
)

# Generate HTML report
report_path = generate_t1t2ratio_html_report(
    Path('/mnt/data/study/analysis/t1t2ratio')
)
```

### CLI Usage

```bash
# Single subject
uv run python -m neurovrai.analysis.anat.t1t2ratio_workflow prepare-single \
    --subject sub-001 \
    --study-root /mnt/data/study \
    --output-dir /mnt/data/study/analysis/t1t2ratio \
    --wm-threshold 0.5 \
    --smooth-fwhm 4.0

# Batch processing
uv run python -m neurovrai.analysis.anat.t1t2ratio_workflow prepare-batch \
    --study-root /mnt/data/study \
    --output-dir /mnt/data/study/analysis/t1t2ratio \
    --n-jobs 4

# Group statistics
uv run python -m neurovrai.analysis.anat.t1t2ratio_workflow analyze \
    --t1t2ratio-dir /mnt/data/study/analysis/t1t2ratio \
    --design-dir /mnt/data/study/designs/t1t2ratio \
    --method randomise \
    --n-permutations 5000

# Generate HTML report
uv run python -m neurovrai.analysis.anat.t1t2ratio_workflow report \
    --t1t2ratio-dir /mnt/data/study/analysis/t1t2ratio
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wm_threshold` | 0.5 | WM probability threshold for mask creation |
| `smooth_fwhm` | 4.0 | Gaussian smoothing FWHM (mm) |
| `min_t2w_threshold` | 10.0 | Minimum T2w intensity (avoid division artifacts) |
| `n_jobs` | 4 | Parallel workers for batch processing |
| `n_permutations` | 5000 | Permutations for FSL randomise |
| `tfce` | True | Use TFCE correction in randomise |

### Input Requirements

```
{study_root}/
├── derivatives/{subject}/
│   ├── anat/
│   │   ├── bias_corrected/*_n4.nii.gz    # N4 bias-corrected T1w
│   │   └── segmentation/wm.nii.gz        # WM probability map
│   └── t2w/
│       └── registered/t2w_to_t1w.nii.gz  # T2w registered to T1w
└── transforms/{subject}/
    └── t1w-mni-composite.h5              # T1w→MNI ANTs transform
```

**Prerequisites**:
- T1w preprocessing (`t1w_preprocess.py`) with N4 bias correction
- T2w preprocessing (`t2w_preprocess.py`) with T1w registration
- Tissue segmentation with WM probability map

### Outputs

**Per Subject** (`{output_dir}/{subject}/`):
```
├── t1t2_ratio.nii.gz              # Ratio in native T1w space
├── t1t2_ratio_mni.nii.gz          # Ratio normalized to MNI
├── wm_mask_mni.nii.gz             # WM mask in MNI space
├── t1t2_ratio_mni_wm.nii.gz       # Ratio masked to WM
├── t1t2_ratio_mni_wm_smooth.nii.gz # Smoothed (analysis-ready)
└── t1t2ratio_metrics.json         # Statistics and metadata
```

**Group Level** (`{output_dir}/group/`):
```
├── merged_t1t2ratio.nii.gz        # 4D merged volume
├── group_wm_mask.nii.gz           # Intersection WM mask
├── t1t2ratio_summary.csv          # Per-subject statistics
└── t1t2ratio_report.html          # Interactive HTML report
```

**Statistics** (`{output_dir}/stats/`):
```
└── randomise_*/
    ├── randomise_tstat1.nii.gz
    ├── randomise_tfce_corrp_tstat1.nii.gz
    └── cluster_reports/
        └── t1t2ratio_cluster_report.html
```

### Reference

Du G, Lewis MM, Sica C, Kong L, Huang X. (2019). Magnetic resonance T1w/T2w ratio: A parsimonious marker for Parkinson disease. Ann Neurol. 85(1):96-104. [PMID: 30408230](https://pubmed.ncbi.nlm.nih.gov/30408230/)

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
