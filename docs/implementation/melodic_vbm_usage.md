# MELODIC and VBM Analysis Usage Guide

This guide provides comprehensive examples for using the MELODIC (group ICA) and VBM (Voxel-Based Morphometry) analysis modules.

## Table of Contents
- [MELODIC Group ICA Analysis](#melodic-group-ica-analysis)
- [VBM Analysis](#vbm-analysis)
- [Configuration](#configuration)
- [Complete Workflow Examples](#complete-workflow-examples)

---

## MELODIC Group ICA Analysis

MELODIC performs group-level Independent Component Analysis to identify spatial patterns of functional connectivity across subjects.

### Prerequisites

**Required:**
- Preprocessed functional data in MNI space for all subjects
- Data should be:
  - Detrended
  - Bandpass filtered (0.01-0.08 Hz)
  - Nuisance-regressed (motion, aCompCor)
  - Spatially smoothed (typically 6mm FWHM)

**Location:**
- Preprocessed data: `{derivatives_dir}/{subject}/func/preprocessed_bold_mni.nii.gz`

### Basic Usage

#### Command Line

**Option 1: Using subject list file**

```bash
# Create subject list file
cat > subjects.txt <<EOF
/study/derivatives/IRC805-0580101/func/preprocessed_bold_mni.nii.gz
/study/derivatives/IRC805-0590101/func/preprocessed_bold_mni.nii.gz
/study/derivatives/IRC805-0600101/func/preprocessed_bold_mni.nii.gz
EOF

# Run MELODIC
python neurovrai/analysis/func/melodic.py \
    --subjects-file subjects.txt \
    --output-dir /study/analysis/melodic \
    --n-components 20 \
    --approach concat
```

**Option 2: Auto-discover from derivatives**

```bash
python neurovrai/analysis/func/melodic.py \
    --derivatives-dir /study/derivatives \
    --subjects IRC805-0580101 IRC805-0590101 IRC805-0600101 \
    --output-dir /study/analysis/melodic \
    --n-components auto \
    --approach concat
```

#### Python API

```python
from pathlib import Path
from neurovrai.analysis.func.melodic import (
    run_melodic_group_ica,
    prepare_subjects_for_melodic
)

# Option 1: Manually specify subject files
subject_files = [
    Path('/study/derivatives/IRC805-0580101/func/preprocessed_bold_mni.nii.gz'),
    Path('/study/derivatives/IRC805-0590101/func/preprocessed_bold_mni.nii.gz'),
    Path('/study/derivatives/IRC805-0600101/func/preprocessed_bold_mni.nii.gz'),
]

results = run_melodic_group_ica(
    subject_files=subject_files,
    output_dir=Path('/study/analysis/melodic'),
    tr=1.029,
    n_components=20,
    approach='concat'
)

# Option 2: Auto-discover from derivatives
subjects = ['IRC805-0580101', 'IRC805-0590101', 'IRC805-0600101']

subject_files = prepare_subjects_for_melodic(
    derivatives_dir=Path('/study/derivatives'),
    subjects=subjects,
    output_file=Path('/study/analysis/melodic/subject_list.txt')
)

results = run_melodic_group_ica(
    subject_files=subject_files,
    output_dir=Path('/study/analysis/melodic'),
    tr=1.029,
    n_components='auto',  # Automatic estimation
    approach='concat'
)
```

### Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `n_components` | int or 'auto' | 'auto' | Number of ICA components. Use 'auto' for automatic estimation, or specify (typically 15-30) |
| `approach` | 'concat' or 'tica' | 'concat' | ICA approach. 'concat' is faster, 'tica' is more sophisticated |
| `sep_vn` | bool | True | Separate variance normalization per subject (recommended) |
| `mm_thresh` | float (0-1) | 0.5 | Mixture model threshold |
| `tr` | float | auto | Repetition time in seconds (read from header if not provided) |
| `mask_file` | Path | MNI152 2mm | Brain mask |
| `bg_image` | Path | MNI152 2mm | Background image for reports |

### Outputs

```
{output_dir}/
├── melodic_IC.nii.gz          # Independent component spatial maps (4D)
├── melodic_mix                # Time courses matrix
├── melodic_FTmix              # Frequency domain time courses
├── melodic_list.txt           # List of input files
├── melodic_summary.json       # Analysis metadata
├── stats/                     # Statistical images and thresholds
├── report/                    # HTML report with visualizations
└── log.txt                    # MELODIC log file
```

### Typical Workflow

```python
from pathlib import Path
from neurovrai.analysis.func.melodic import run_melodic_group_ica

# 1. Collect all preprocessed subjects
subject_files = list(Path('/study/derivatives').glob('*/func/preprocessed_bold_mni.nii.gz'))

# 2. Run MELODIC with 20 components
results = run_melodic_group_ica(
    subject_files=subject_files,
    output_dir=Path('/study/analysis/melodic'),
    n_components=20,
    approach='concat'
)

# 3. View results
print(f"Analyzed {results['n_subjects']} subjects")
print(f"Component maps: {results['outputs']['component_maps']}")
print(f"Report: {results['outputs']['report_dir']}")
```

### Next Steps After MELODIC

After MELODIC identifies group-level components, you can:

1. **Dual Regression**: Get subject-specific versions of components
2. **Component Classification**: Identify signal vs noise components
3. **Statistical Analysis**: Compare components across groups

---

## VBM Analysis

VBM performs group-level statistical analysis of brain structure using tissue probability maps from anatomical preprocessing.

### Prerequisites

**Required:**
- Anatomical preprocessing completed for all subjects
- Required outputs from anatomical preprocessing:
  - Tissue segmentation: `{derivatives_dir}/{subject}/anat/segmentation/pve_*.nii.gz`
  - Transform to MNI: `{derivatives_dir}/{subject}/anat/transforms/highres2standard_warp.nii.gz`
- Participant demographics file (TSV format with 'participant_id' column)

### Workflow Overview

VBM is a **two-step process**:

1. **Prepare**: Normalize and smooth tissue maps for all subjects
2. **Analyze**: Run group-level statistics

### Step 1: Prepare VBM Data

#### Command Line

```bash
python neurovrai/analysis/anat/vbm_workflow.py prepare \
    --derivatives-dir /study/derivatives \
    --subjects IRC805-0580101 IRC805-0590101 IRC805-0600101 \
    --output-dir /study/analysis/vbm \
    --tissue GM \
    --smooth-fwhm 4.0
```

#### Python API

```python
from pathlib import Path
from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data

subjects = ['IRC805-0580101', 'IRC805-0590101', 'IRC805-0600101']

results = prepare_vbm_data(
    subjects=subjects,
    derivatives_dir=Path('/study/derivatives'),
    output_dir=Path('/study/analysis/vbm'),
    tissue_type='GM',      # Grey matter
    smooth_fwhm=4.0,       # 4mm smoothing
    modulate=False         # No modulation (standard VBM)
)

print(f"Prepared {results['n_subjects']} subjects")
print(f"Merged data: {results['merged_file']}")
```

#### Preparation Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `tissue_type` | 'GM', 'WM', 'CSF' | 'GM' | Tissue type to analyze |
| `smooth_fwhm` | float | 4.0 | Smoothing kernel FWHM in mm (0 = no smoothing) |
| `modulate` | bool | False | Modulate by Jacobian to preserve volume |
| `reference` | Path | MNI152 2mm | Reference template |

#### Preparation Outputs

```
{output_dir}/
├── subjects/
│   ├── IRC805-0580101_GM_mni.nii.gz         # Normalized tissue maps
│   ├── IRC805-0580101_GM_mni_smooth.nii.gz  # Smoothed normalized maps
│   └── ...
├── merged_GM.nii.gz          # 4D merged volume (all subjects)
├── group_mask.nii.gz         # Group mask
├── subject_list.txt          # List of subjects
└── vbm_info.json            # Metadata
```

### Step 2: Run VBM Statistics

#### Create Participants File

Create a TSV file with participant demographics:

```tsv
participant_id	age	sex	group
IRC805-0580101	45.2	1	patient
IRC805-0590101	38.7	0	control
IRC805-0600101	52.1	1	patient
```

Where:
- `participant_id`: Must match subject IDs from preparation step
- Other columns: Variables for statistical analysis (continuous or categorical)
- Sex coding: 0=female, 1=male (or your preferred coding)

#### Command Line

```bash
python neurovrai/analysis/anat/vbm_workflow.py analyze \
    --vbm-dir /study/analysis/vbm \
    --participants /study/participants.tsv \
    --design "age + sex" \
    --contrasts age_positive,age_negative \
    --n-permutations 5000
```

#### Python API

```python
from pathlib import Path
from neurovrai.analysis.anat.vbm_workflow import run_vbm_analysis

# Define contrasts
# Format: contrast_name: [intercept, age, sex, ...]
contrasts = {
    'age_positive': [0, 1, 0],    # Positive age effect
    'age_negative': [0, -1, 0],   # Negative age effect
    'sex_difference': [0, 0, 1]   # Sex differences (controlling for age)
}

results = run_vbm_analysis(
    vbm_dir=Path('/study/analysis/vbm'),
    participants_file=Path('/study/participants.tsv'),
    formula='age + sex',
    contrasts=contrasts,
    n_permutations=5000,
    tfce=True,
    cluster_threshold=0.95  # p < 0.05
)

print(f"Analysis complete for {results['n_subjects']} subjects")
print(f"Statistical maps: {results['randomise_dir']}")
```

#### Analysis Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `formula` | str | required | Design formula (e.g., 'age + sex') |
| `contrasts` | dict | required | Contrast name to vector mapping |
| `n_permutations` | int | 5000 | Number of permutations for randomise |
| `tfce` | bool | True | Use Threshold-Free Cluster Enhancement |
| `cluster_threshold` | float | 0.95 | Cluster significance (0.95 = p<0.05) |

#### Analysis Outputs

```
{vbm_dir}/stats/
├── design.mat                      # Design matrix
├── design.con                      # Contrasts
├── randomise_output/               # Statistical maps
│   ├── vbm_tstat1.nii.gz          # T-statistic map (contrast 1)
│   ├── vbm_tfce_corrp_tstat1.nii.gz  # Corrected p-values
│   └── ...
├── cluster_reports/                # Cluster tables and visualizations
│   ├── age_positive_report.html   # HTML report
│   └── ...
└── vbm_results.json               # Analysis summary
```

---

## Configuration

Add to your `config.yaml`:

```yaml
# Resting-state analysis
resting_analysis:
  # MELODIC group ICA
  melodic:
    enabled: false
    n_components: auto  # or specific number like 20
    approach: concat    # concat (temporal concatenation) or tica (tensor ICA)
    sep_vn: true       # Separate variance normalization
    mm_thresh: 0.5     # Mixture model threshold
  # ReHo parameters
  reho:
    neighborhood: 27   # 7, 19, or 27
  # fALFF parameters
  falff:
    low_freq: 0.01    # Hz
    high_freq: 0.08   # Hz

# VBM analysis
vbm:
  enabled: false
  tissue_type: GM      # GM, WM, or CSF
  smooth_fwhm: 4.0    # Smoothing kernel FWHM in mm (0 = no smoothing)
  modulate: false     # Modulate by Jacobian to preserve volume
  n_permutations: 5000
  tfce: true          # Use Threshold-Free Cluster Enhancement
  cluster_threshold: 0.95  # p < 0.05
```

---

## Complete Workflow Examples

### Example 1: MELODIC Analysis for IRC805 Study

```python
#!/usr/bin/env python3
"""
Run MELODIC group ICA on IRC805 resting-state data
"""
from pathlib import Path
from neurovrai.analysis.func.melodic import prepare_subjects_for_melodic, run_melodic_group_ica

# Study paths
derivatives_dir = Path('/mnt/bytopia/IRC805/derivatives')
analysis_dir = Path('/mnt/bytopia/IRC805/analysis')
melodic_dir = analysis_dir / 'melodic'

# Get list of subjects with preprocessed data
# (Assuming preprocessing is complete)
subjects = [
    'IRC805-0580101', 'IRC805-0590101', 'IRC805-0600101',
    'IRC805-0610101', 'IRC805-0620101', 'IRC805-0630101',
    # Add all subjects...
]

# Step 1: Collect preprocessed data
print("Collecting preprocessed data...")
subject_files = prepare_subjects_for_melodic(
    derivatives_dir=derivatives_dir,
    subjects=subjects,
    output_file=melodic_dir / 'subject_list.txt'
)

print(f"Found {len(subject_files)} subjects with preprocessed data")

# Step 2: Run MELODIC with 20 components
print("\nRunning MELODIC group ICA...")
results = run_melodic_group_ica(
    subject_files=subject_files,
    output_dir=melodic_dir,
    n_components=20,
    approach='concat',
    tr=1.029
)

print("\nMELODIC Analysis Complete!")
print(f"Component maps: {results['outputs']['component_maps']}")
print(f"Report: {results['outputs']['report_dir']}")
print(f"Summary: {melodic_dir / 'melodic_summary.json'}")
```

### Example 2: VBM Analysis for IRC805 Study

```python
#!/usr/bin/env python3
"""
Run VBM analysis on IRC805 anatomical data
"""
from pathlib import Path
import pandas as pd
from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data, run_vbm_analysis

# Study paths
derivatives_dir = Path('/mnt/bytopia/IRC805/derivatives')
analysis_dir = Path('/mnt/bytopia/IRC805/analysis')
vbm_dir = analysis_dir / 'vbm'

# Get subjects with anatomical preprocessing
subjects = [
    'IRC805-0580101', 'IRC805-0590101', 'IRC805-0600101',
    'IRC805-0610101', 'IRC805-0620101', 'IRC805-0630101',
    # Add all subjects...
]

# Step 1: Prepare VBM data
print("Preparing VBM data...")
prep_results = prepare_vbm_data(
    subjects=subjects,
    derivatives_dir=derivatives_dir,
    output_dir=vbm_dir,
    tissue_type='GM',
    smooth_fwhm=4.0,
    modulate=False
)

print(f"\nPrepared {prep_results['n_subjects']} subjects")
print(f"Merged data: {prep_results['merged_file']}")

# Step 2: Create participants file
participants_data = {
    'participant_id': subjects,
    'age': [45.2, 38.7, 52.1, 41.5, 49.8, 36.2],  # Example ages
    'sex': [1, 0, 1, 0, 1, 1],  # 0=female, 1=male
    'group': [1, 0, 1, 0, 1, 0]  # 0=control, 1=patient
}
participants_df = pd.DataFrame(participants_data)
participants_file = analysis_dir / 'participants.tsv'
participants_df.to_csv(participants_file, sep='\t', index=False)

print(f"\nCreated participants file: {participants_file}")

# Step 3: Run group statistics
print("\nRunning VBM group statistics...")

# Define contrasts based on design matrix columns
# Formula: 'age + sex + group' creates design matrix:
# [intercept, age, sex, group]
contrasts = {
    'age_positive': [0, 1, 0, 0],      # Positive age effect
    'age_negative': [0, -1, 0, 0],     # Negative age effect
    'group_difference': [0, 0, 0, 1],  # Patient > Control
}

stats_results = run_vbm_analysis(
    vbm_dir=vbm_dir,
    participants_file=participants_file,
    formula='age + sex + group',
    contrasts=contrasts,
    n_permutations=5000,
    tfce=True
)

print("\nVBM Analysis Complete!")
print(f"Statistical maps: {stats_results['randomise_dir']}")
print(f"Cluster reports: {stats_results['cluster_reports']}")
print(f"Results summary: {vbm_dir / 'stats' / 'vbm_results.json'}")
```

### Example 3: Batch Processing Multiple Analyses

```python
#!/usr/bin/env python3
"""
Run both MELODIC and VBM analyses in batch
"""
from pathlib import Path
from neurovrai.analysis.func.melodic import prepare_subjects_for_melodic, run_melodic_group_ica
from neurovrai.analysis.anat.vbm_workflow import prepare_vbm_data, run_vbm_analysis

# Configuration
study_root = Path('/mnt/bytopia/IRC805')
derivatives_dir = study_root / 'derivatives'
analysis_dir = study_root / 'analysis'

subjects = list(derivatives_dir.glob('IRC805-*'))
subject_ids = [s.name for s in subjects]

print(f"Found {len(subject_ids)} subjects")

# 1. MELODIC Analysis
print("\n" + "=" * 80)
print("MELODIC ANALYSIS")
print("=" * 80)

melodic_dir = analysis_dir / 'melodic'
subject_files = prepare_subjects_for_melodic(
    derivatives_dir=derivatives_dir,
    subjects=subject_ids
)

if len(subject_files) >= 10:  # Need minimum subjects for MELODIC
    melodic_results = run_melodic_group_ica(
        subject_files=subject_files,
        output_dir=melodic_dir,
        n_components=20,
        approach='concat'
    )
    print(f"✓ MELODIC complete: {melodic_results['n_subjects']} subjects")
else:
    print(f"✗ Need at least 10 subjects, found {len(subject_files)}")

# 2. VBM Analysis
print("\n" + "=" * 80)
print("VBM ANALYSIS")
print("=" * 80)

vbm_dir = analysis_dir / 'vbm'

# Grey matter
print("\nAnalyzing grey matter...")
gm_results = prepare_vbm_data(
    subjects=subject_ids,
    derivatives_dir=derivatives_dir,
    output_dir=vbm_dir / 'GM',
    tissue_type='GM',
    smooth_fwhm=4.0
)
print(f"✓ GM preparation complete: {gm_results['n_subjects']} subjects")

# White matter
print("\nAnalyzing white matter...")
wm_results = prepare_vbm_data(
    subjects=subject_ids,
    derivatives_dir=derivatives_dir,
    output_dir=vbm_dir / 'WM',
    tissue_type='WM',
    smooth_fwhm=4.0
)
print(f"✓ WM preparation complete: {wm_results['n_subjects']} subjects")

print("\n" + "=" * 80)
print("BATCH ANALYSIS COMPLETE")
print("=" * 80)
```

---

## Troubleshooting

### MELODIC Issues

**Problem: "Dimension mismatch" errors**
- Solution: Ensure all functional images are in the same space (MNI152) with same resolution
- Check: `fslinfo image.nii.gz` for each subject

**Problem: MELODIC takes very long**
- Solution: Use `approach='concat'` instead of `'tica'`
- Reduce number of subjects for initial testing

### VBM Issues

**Problem: "Transform not found" during preparation**
- Solution: Ensure anatomical preprocessing completed successfully
- Check for: `{derivatives_dir}/{subject}/anat/transforms/highres2standard_warp.nii.gz`

**Problem: "No significant clusters" in results**
- Try: Lower the cluster threshold or increase number of permutations
- Check: Sample size adequate (VBM typically needs 20+ subjects per group)

---

## References

- **MELODIC**: Beckmann, C.F., & Smith, S.M. (2004). Probabilistic independent component analysis for functional magnetic resonance imaging. IEEE TMI, 23(2), 137-152.

- **VBM**: Ashburner, J., & Friston, K.J. (2000). Voxel-based morphometry—the methods. Neuroimage, 11(6), 805-821.

- **FSL MELODIC**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC
- **FSL Randomise**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise
