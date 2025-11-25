# neurovrai.analysis

**Status**: In Development (Phase 3)

## Overview

Group-level statistical analyses for neuroimaging data using a modular two-stage approach.

## Analysis Architecture

All analysis modules follow a **two-stage pipeline** design:

### Stage 1: Preparation
Data preparation and metric calculation to prepare for statistical analysis.

### Stage 2: Statistical Analysis
FSL randomise with TFCE + enhanced cluster reporting with atlas localization.

### Subject Selection
Each module supports flexible subject selection:
- **Option A**: Provide explicit subject filelist (text file, one subject per line)
- **Option B**: Auto-discover all preprocessed subjects in `{study_root}/derivatives/`

This allows for:
- Testing with subset of subjects
- Excluding subjects with QC failures
- Running analyses on specific cohorts
- Maximum flexibility for complex study designs

---

## Analysis Modules

### 1. VBM (Voxel-Based Morphometry)

**Purpose**: Statistical comparison of grey/white matter volume across groups

**Status**: ✅ Production-Ready

#### Stage 1: VBM Preparation
```bash
prepare_vbm \
  --derivatives-dir /study/derivatives \
  --output-dir /study/analysis/vbm \
  [--subject-list subjects.txt | --auto-discover] \
  --tissue-type [GM|WM|CSF] \
  --smooth-fwhm 6
```

**Operations**:
- Discover subjects with completed anatomical preprocessing
- Extract tissue probability maps (GM/WM/CSF)
- Smooth with configurable FWHM
- Register to MNI152 template space
- Concatenate into 4D volume
- Generate subject QC report

**Outputs**:
- `{tissue}_4d.nii.gz` - 4D concatenated volume
- `subjects.txt` - Final subject list (for randomise)
- `qc_report.html` - Tissue segmentation QC

#### Stage 2: VBM Statistics
```bash
run_vbm_stats \
  --input /study/analysis/vbm/GM_4d.nii.gz \
  --design design.mat \
  --contrasts design.con \
  --output-dir /study/analysis/vbm/GM/stats \
  --n-perm 5000 \
  --threshold 0.05 \
  --atlas harvard-oxford
```

**Operations**:
- Run FSL randomise with TFCE
- Extract significant clusters
- Generate HTML reports with Harvard-Oxford grey matter atlas
- Create tri-planar visualizations
- Compile summary statistics

**Outputs**:
- `randomise_tstat*.nii.gz` - T-statistic maps
- `randomise_tfce_corrp_tstat*.nii.gz` - TFCE-corrected p-value maps
- `cluster_reports/{contrast}_report.html` - Enhanced cluster reports
- `cluster_reports/images/` - Tri-planar visualizations

**Atlas**: Harvard-Oxford Cortical & Subcortical (grey matter)

---

### 2. TBSS (Tract-Based Spatial Statistics)

**Purpose**: Voxel-wise statistics on white matter skeleton

**Status**: ✅ Production-Ready

#### Stage 1: TBSS Preparation
```bash
prepare_tbss \
  --derivatives-dir /study/derivatives \
  --output-dir /study/analysis/tbss \
  [--subject-list subjects.txt | --auto-discover] \
  --metric [FA|MD|AD|RD|MK|FICVF|ODI]
```

**Operations**:
- Discover subjects with DTI/DKI/NODDI metrics
- Run FSL TBSS pipeline (steps 1-4)
  1. Erode FA images
  2. Register to FMRIB58_FA template
  3. Create mean FA and skeleton
  4. Project metric onto skeleton
- Concatenate skeletonized metrics into 4D volume
- Generate skeleton QC report

**Outputs**:
- `FA/stats/all_FA_skeletonised.nii.gz` - 4D skeleton volume
- `stats/mean_FA_skeleton.nii.gz` - Group skeleton mask
- `subjects.txt` - Final subject list
- `qc_report.html` - Registration and projection QC

#### Stage 2: TBSS Statistics
```bash
run_tbss_stats \
  --input /study/analysis/tbss/FA/stats/all_FA_skeletonised.nii.gz \
  --skeleton /study/analysis/tbss/stats/mean_FA_skeleton.nii.gz \
  --design design.mat \
  --contrasts design.con \
  --output-dir /study/analysis/tbss/FA/stats \
  --n-perm 5000 \
  --threshold 0.05 \
  --atlas jhu
```

**Operations**:
- Run FSL randomise with TFCE on skeletonized data
- Threshold with skeleton mask
- Extract significant clusters
- Generate HTML reports with JHU white matter atlas
- Create skeleton overlay visualizations

**Outputs**:
- `randomise_tstat*.nii.gz` - T-statistic maps (skeleton space)
- `randomise_tfce_corrp_tstat*.nii.gz` - TFCE-corrected p-values
- `cluster_reports/{contrast}_report.html` - Enhanced cluster reports
- `cluster_reports/images/` - Skeleton overlay visualizations

**Atlas**: JHU ICBM-DTI-81 White Matter (white matter tracts)

---

### 3. ReHo/fALFF (Regional Homogeneity / fALFF)

**Purpose**: Resting-state functional connectivity metrics

**Status**: ✅ Production-Ready

#### Stage 1: ReHo/fALFF Preparation
```bash
prepare_reho_falff \
  --derivatives-dir /study/derivatives \
  --output-dir /study/analysis/resting_state \
  [--subject-list subjects.txt | --auto-discover] \
  --metric [reho|falff|both] \
  --reho-neighborhood 27 \
  --falff-freq-range 0.01 0.08
```

**Operations**:
- Discover subjects with preprocessed functional data
- Compute ReHo maps (Kendall's W in local neighborhood)
- Compute fALFF maps (fractional ALFF in 0.01-0.08 Hz)
- Z-score normalize maps
- Concatenate into 4D volumes
- Generate tSNR and metric QC report

**Outputs**:
- `reho_4d.nii.gz` - 4D ReHo z-score maps
- `falff_4d.nii.gz` - 4D fALFF z-score maps
- `subjects.txt` - Final subject list
- `qc_report.html` - tSNR and metric distribution QC

#### Stage 2: ReHo/fALFF Statistics
```bash
run_reho_falff_stats \
  --input /study/analysis/resting_state/reho_4d.nii.gz \
  --design design.mat \
  --contrasts design.con \
  --output-dir /study/analysis/resting_state/reho/stats \
  --n-perm 5000 \
  --threshold 0.05 \
  --atlas harvard-oxford
```

**Operations**:
- Run FSL randomise with TFCE
- Extract significant clusters
- Generate HTML reports with Harvard-Oxford grey matter atlas
- Create tri-planar visualizations
- Compile summary statistics

**Outputs**:
- `randomise_tstat*.nii.gz` - T-statistic maps
- `randomise_tfce_corrp_tstat*.nii.gz` - TFCE-corrected p-values
- `cluster_reports/{contrast}_report.html` - Enhanced cluster reports
- `cluster_reports/images/` - Tri-planar visualizations

**Atlas**: Harvard-Oxford Cortical & Subcortical (grey matter)

---

### 4. MELODIC (Group ICA)

**Purpose**: Identify consistent resting-state networks across subjects

**Status**: 🚧 In Development

#### Stage 1: MELODIC Group ICA
```bash
prepare_melodic \
  --derivatives-dir /study/derivatives \
  --output-dir /study/analysis/melodic \
  [--subject-list subjects.txt | --auto-discover] \
  --n-components 20 \
  --tr 2.0
```

**Operations**:
- Discover subjects with preprocessed functional data
- Temporal concatenation of 4D functional data
- Run FSL MELODIC group ICA
- Identify resting-state network components
- Generate component spatial maps and time courses

**Outputs**:
- `melodic_IC.nii.gz` - Independent component spatial maps
- `melodic_mix` - Component time courses
- `melodic_FTmix` - Frequency domain mixing matrix
- `subjects.txt` - Final subject list
- `report.html` - Component visualization

#### Stage 2: Dual Regression + Statistics
```bash
run_melodic_dr_stats \
  --melodic-dir /study/analysis/melodic \
  --design design.mat \
  --contrasts design.con \
  --output-dir /study/analysis/melodic/dr/stats \
  --n-perm 5000 \
  --threshold 0.05
```

**Operations**:
- Run dual regression for subject-specific networks
- Spatial regression: get subject time courses
- Temporal regression: get subject spatial maps
- Run FSL randomise on spatial maps
- Generate network-specific cluster reports

**Outputs**:
- `dr_stage1_ic{N}.nii.gz` - Subject time courses
- `dr_stage2_ic{N}.nii.gz` - Subject spatial maps (4D)
- `randomise_ic{N}_tstat*.nii.gz` - Network-specific statistics
- `cluster_reports/ic{N}_{contrast}_report.html` - Network cluster reports

**Atlas**: Harvard-Oxford (for network anatomical localization)

---

### 5. ASL Group Analysis

**Purpose**: Group-level cerebral blood flow statistics

**Status**: 📋 Planned

#### Stage 1: CBF Preparation
```bash
prepare_asl_group \
  --derivatives-dir /study/derivatives \
  --output-dir /study/analysis/asl \
  [--subject-list subjects.txt | --auto-discover] \
  --cbf-type [mean|gm|wm]
```

**Operations**:
- Discover subjects with ASL preprocessing completed
- Extract mean CBF, GM CBF, or WM CBF maps
- Register to MNI152 template space
- Concatenate into 4D volume
- Generate CBF distribution QC report

**Outputs**:
- `cbf_mean_4d.nii.gz` - 4D concatenated CBF maps
- `subjects.txt` - Final subject list
- `qc_report.html` - CBF distribution and registration QC

#### Stage 2: ASL Statistics
```bash
run_asl_stats \
  --input /study/analysis/asl/cbf_mean_4d.nii.gz \
  --design design.mat \
  --contrasts design.con \
  --output-dir /study/analysis/asl/stats \
  --n-perm 5000 \
  --threshold 0.05 \
  --atlas harvard-oxford
```

**Operations**:
- Run FSL randomise with TFCE
- Extract significant clusters
- Generate HTML reports with Harvard-Oxford atlas
- Create perfusion overlay visualizations

**Outputs**:
- `randomise_tstat*.nii.gz` - T-statistic maps
- `randomise_tfce_corrp_tstat*.nii.gz` - TFCE-corrected p-values
- `cluster_reports/{contrast}_report.html` - Enhanced cluster reports

**Atlas**: Harvard-Oxford Cortical & Subcortical (grey matter)

---

## Design Matrix and Contrasts

All Stage 2 analyses require:
- `design.mat` - FSL vest format design matrix
- `design.con` - FSL vest format contrast file

### FSL Vest Format

**Design Matrix** (`design.mat`):
```
/NumWaves N_PREDICTORS
/NumPoints N_SUBJECTS
/Matrix
<design matrix values>
```

**Contrast File** (`design.con`):
```
/NumWaves N_PREDICTORS
/NumContrasts N_CONTRASTS
/Matrix
<contrast matrix values>
```

### Example: Age and Group Effects

Design matrix columns:
1. Intercept (mean)
2. Age (demeaned continuous)
3. Group (binary: patient=1, control=0)
4. Sex (binary: M=1, F=0)

Typical contrasts:
1. Age positive: `[0, 1, 0, 0]` - regions increase with age
2. Age negative: `[0, -1, 0, 0]` - regions decrease with age
3. Group difference: `[0, 0, 1, 0]` - patient > control
4. Sex difference: `[0, 0, 0, 1]` - M > F

### Utilities

Create design matrices programmatically:
```python
from neurovrai.analysis.stats.design import create_design_matrix

design, contrasts = create_design_matrix(
    demographics=df,
    formula='age + group + sex',
    output_dir=Path('analysis/stats')
)
```

---

## Atlas-Based Cluster Reporting

All analyses use enhanced cluster reporting with anatomical atlas localization.

### Available Atlases

| Atlas | Tissue Type | Analysis Modules | Regions |
|-------|-------------|------------------|---------|
| **Harvard-Oxford Cortical & Subcortical** | Grey matter | VBM, ReHo, fALFF, ASL | 48 cortical + 21 subcortical |
| **JHU ICBM-DTI-81** | White matter | TBSS | 48 white matter tracts |

### Cluster Report Features

**For each significant cluster**:
- **Anatomical localization**: Region names with percentage coverage
- **Peak statistics**: T-stat, p-value (corrected), peak coordinates (MNI)
- **Cluster size**: Number of voxels
- **Tri-planar visualization**: Axial/Coronal/Sagittal slices at peak
- **Interactive HTML**: Sortable tables, embedded images

**Report Format**:
- Modern responsive HTML with gradient styling
- Cluster cards with statistics and location breakdown
- Embedded mosaic images (no external dependencies)
- Summary statistics at top

---

## Quality Control

Each Stage 1 preparation module generates QC reports:

### VBM QC
- Tissue segmentation quality
- Registration accuracy to MNI152
- Volume distributions across subjects

### TBSS QC
- FA registration to FMRIB58_FA
- Skeleton projection accuracy
- Metric distributions on skeleton

### ReHo/fALFF QC
- Temporal SNR (tSNR) maps
- Metric distributions
- Z-score normalization check

### MELODIC QC
- Component spatial maps
- Component time courses
- Temporal frequency spectra
- Component classification (signal vs artifact)

### ASL QC
- CBF value distributions (GM/WM/whole brain)
- Registration to MNI152
- Outlier detection

---

## Command-Line Interface

### Unified Analysis Scripts

Each module will have two CLI scripts in `scripts/`:

#### Preparation
```bash
# Stage 1: Preparation (auto-discover all subjects)
python scripts/prepare_{module}.py \
  --derivatives-dir /study/derivatives \
  --output-dir /study/analysis/{module} \
  --config config.yaml

# Stage 1: Preparation (explicit subject list)
python scripts/prepare_{module}.py \
  --derivatives-dir /study/derivatives \
  --output-dir /study/analysis/{module} \
  --subject-list subjects.txt \
  --config config.yaml
```

#### Statistical Analysis
```bash
# Stage 2: Statistics
python scripts/run_{module}_stats.py \
  --input /study/analysis/{module}/{metric}_4d.nii.gz \
  --design /study/analysis/{module}/design.mat \
  --contrasts /study/analysis/{module}/design.con \
  --output-dir /study/analysis/{module}/stats \
  --config config.yaml
```

### Configuration File Support

All scripts support YAML configuration:
```yaml
# analysis_config.yaml
derivatives_dir: /mnt/bytopia/IRC805/derivatives
analysis_dir: /mnt/bytopia/IRC805/analysis

# Subject selection
subject_list: null  # null = auto-discover
# subject_list: /path/to/subjects.txt  # or explicit list

# Statistical parameters
n_permutations: 5000
threshold: 0.05

# Module-specific
vbm:
  tissue_type: GM
  smooth_fwhm: 6

tbss:
  metric: FA

reho_falff:
  reho_neighborhood: 27
  falff_freq_range: [0.01, 0.08]

melodic:
  n_components: 20
```

---

## Implementation Status

| Module | Stage 1 | Stage 2 | Atlas | Status |
|--------|---------|---------|-------|--------|
| **VBM** | ✅ Complete | ✅ Complete | Harvard-Oxford | Production |
| **TBSS** | ✅ Complete | ✅ Complete | JHU | Production |
| **ReHo/fALFF** | ✅ Complete | ✅ Complete | Harvard-Oxford | Production |
| **MELODIC** | 🚧 In Progress | 📋 Planned | Harvard-Oxford | Development |
| **ASL Group** | 📋 Planned | 📋 Planned | Harvard-Oxford | Planned |

**Legend**:
- ✅ Complete and tested
- 🚧 In development
- 📋 Planned but not started

---

## Testing Strategy

### Unit Tests
- Subject discovery functions
- Design matrix generation
- Atlas loading and localization
- 4D volume concatenation

### Integration Tests
- Full Stage 1 pipeline on synthetic data
- Full Stage 2 pipeline with known statistical effects
- End-to-end test with sample dataset

### Validation Tests
- Compare TBSS results with FSL's built-in pipeline
- Verify atlas localization with known coordinates
- Cross-check statistics with manual GLM

---

## Development Timeline

**Current Status**: Stage 1 and 2 implemented for VBM, TBSS, ReHo/fALFF

**Next Steps**:
1. **Refactor existing modules** to follow two-stage pattern
2. **Implement MELODIC** Stage 1 and 2
3. **Add ASL group analysis** Stage 1 and 2
4. **Create unified CLI** with config file support
5. **Add comprehensive tests** for all modules
6. **Generate documentation** with example workflows

**Estimated Completion**: 2-3 weeks for full implementation and testing

---

## References

### FSL Tools
- randomise: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise
- TBSS: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS
- MELODIC: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MELODIC

### Atlases
- Harvard-Oxford: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
- JHU ICBM-DTI-81: Mori et al., MRI Atlas of Human White Matter (2005)

### Methods Papers
- TBSS: Smith et al., NeuroImage 2006
- TFCE: Smith & Nichols, NeuroImage 2009
- ReHo: Zang et al., NeuroImage 2004
- fALFF: Zou et al., Journal of Neuroscience Methods 2008
- MELODIC/ICA: Beckmann & Smith, NeuroImage 2004
