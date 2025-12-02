# GLM Analysis Guide

## Overview

The GLM (General Linear Model) module provides parametric statistical analysis as an alternative or complement to FSL's randomise nonparametric permutation testing. Both methods use the same design matrices and contrasts, making it easy to compare results.

## When to Use GLM vs Randomise

### Use GLM when:
- **Speed is important** - GLM is much faster than randomise (seconds vs minutes/hours)
- **Exploratory analysis** - Quick hypothesis testing before full permutation testing
- **Data meets assumptions** - Normally distributed residuals, homogeneous variance
- **Large sample sizes** - Parametric assumptions typically hold well with n > 30

### Use Randomise when:
- **Gold standard needed** - Nonparametric, no assumptions about data distribution
- **Small sample sizes** - More robust with limited data (n < 30)
- **Non-normal data** - Violations of normality or variance assumptions
- **Publication requirements** - Many journals prefer nonparametric methods for neuroimaging

### Use Both when:
- **Comparison and validation** - Check agreement between parametric and nonparametric results
- **Robustness testing** - Identify findings that replicate across methods
- **Method development** - Understanding how analysis choices affect results

## Key Differences

| Feature | GLM (Parametric) | Randomise (Nonparametric) |
|---------|-----------------|--------------------------|
| **Speed** | Fast (~seconds) | Slow (~minutes to hours) |
| **Assumptions** | Normality, homogeneity of variance | Minimal (exchangeability) |
| **Multiple comparisons** | Cluster thresholding, FDR | TFCE, cluster-based permutation |
| **P-values** | Parametric (from t/F distributions) | Empirical (from permutations) |
| **Typical use** | Exploratory, large samples | Final analysis, small samples |

## Architecture

The GLM module (`neurovrai/analysis/stats/glm_wrapper.py`) provides:

1. **`run_fsl_glm()`** - Execute FSL's fsl_glm for parametric fitting
2. **`threshold_zstat()`** - Cluster-based thresholding of z-statistic maps
3. **`summarize_glm_results()`** - Extract significant findings across contrasts

## Usage Examples

### 1. TBSS Analysis with GLM

```bash
# Run parametric GLM analysis only
python -m neurovrai.analysis.tbss.run_tbss_stats \
    --data-dir /study/analysis/tbss_FA/ \
    --participants participants.csv \
    --formula "age + sex + exposure" \
    --contrasts-file contrasts.yaml \
    --output-dir /study/analysis/tbss_FA/model_glm/ \
    --method glm \
    --z-threshold 2.3

# Run both randomise and GLM for comparison
python -m neurovrai.analysis.tbss.run_tbss_stats \
    --data-dir /study/analysis/tbss_FA/ \
    --participants participants.csv \
    --formula "age + sex + exposure" \
    --contrasts-file contrasts.yaml \
    --output-dir /study/analysis/tbss_FA/model_both/ \
    --method both \
    --z-threshold 2.3
```

### 2. Functional Group Analysis with GLM

```python
from pathlib import Path
from neurovrai.analysis.stats.glm_wrapper import run_fsl_glm, threshold_zstat

# Paths
image_4d = Path('/study/analysis/func/reho/all_reho_4D.nii.gz')
design_mat = Path('/study/analysis/func/reho/design.mat')
contrast_con = Path('/study/analysis/func/reho/design.con')
mask = Path('/study/analysis/func/reho/group_mask.nii.gz')

# Run GLM
glm_result = run_fsl_glm(
    input_file=image_4d,
    design_mat=design_mat,
    contrast_con=contrast_con,
    output_dir=Path('/study/analysis/func/reho/glm_output/'),
    mask=mask
)

# Threshold results
threshold_result = threshold_zstat(
    zstat_file=Path(glm_result['output_files']['zstat']),
    output_dir=Path('/study/analysis/func/reho/glm_output/thresholded/'),
    z_threshold=2.3,  # Approximately p < 0.01
    cluster_threshold=10,  # Minimum 10 voxels
    mask=mask
)
```

### 3. Standalone GLM Analysis

```python
from pathlib import Path
from neurovrai.analysis.stats import (
    generate_design_files,
    run_fsl_glm,
    summarize_glm_results
)

# Step 1: Generate design matrix and contrasts
design_result = generate_design_files(
    participants_file=Path('participants.csv'),
    formula='age + sex + exposure',
    contrasts=[
        {'name': 'age_positive', 'vector': [0, 1, 0, 0]},
        {'name': 'exposure_positive', 'vector': [0, 0, 0, 1]}
    ],
    output_dir=Path('/study/analysis/model1/'),
    subject_list_file=Path('subject_list.txt')
)

# Step 2: Run GLM
glm_result = run_fsl_glm(
    input_file=Path('/study/data/all_FA_skeletonised.nii.gz'),
    design_mat=Path(design_result['design_mat_file']),
    contrast_con=Path(design_result['contrast_con_file']),
    output_dir=Path('/study/analysis/model1/glm_output/'),
    mask=Path('/study/data/mean_FA_skeleton_mask.nii.gz')
)

# Step 3: Summarize results
summary = summarize_glm_results(
    output_dir=Path('/study/analysis/model1/glm_output/'),
    contrast_names=['age_positive', 'exposure_positive'],
    z_threshold=2.3
)

print(f"Analysis complete. Found {len(summary['contrasts'])} contrasts.")
for contrast in summary['contrasts']:
    if contrast['significant']:
        print(f"  ✓ {contrast['name']}: {contrast['n_positive_voxels']} pos, "
              f"{contrast['n_negative_voxels']} neg voxels")
```

## Output Files

### GLM Outputs

GLM analysis creates the following files in `{output_dir}/glm_output/`:

```
glm_output/
├── glm.log                     # Execution log
├── glm_pe.nii.gz               # Parameter estimates
├── glm_tstat.nii.gz            # T-statistics
├── glm_zstat.nii.gz            # Z-statistics (primary output)
├── glm_cope.nii.gz             # Contrast of parameter estimates
├── glm_varcope.nii.gz          # Variance estimates
└── thresholded/
    ├── zstat_contrast1_thresh.nii.gz
    ├── zstat_contrast2_thresh.nii.gz
    └── ...
```

### Comparison with Randomise Outputs

When using `--method both`, outputs are organized as:

```
{output_dir}/
├── design.mat                  # Design matrix (shared)
├── design.con                  # Contrasts (shared)
├── design_summary.json         # Design metadata
├── randomise_output/           # Nonparametric results
│   ├── randomise_tstat1.nii.gz
│   ├── randomise_tfce_corrp_tstat1.nii.gz
│   └── ...
├── glm_output/                 # Parametric results
│   ├── glm_zstat.nii.gz
│   ├── thresholded/
│   └── ...
└── cluster_reports_randomise/  # Cluster reports (randomise only)
```

## Interpretation

### Z-Score Thresholds

Common z-score thresholds and their approximate p-values:

| Z-threshold | One-tailed p | Two-tailed p | Usage |
|------------|--------------|--------------|-------|
| 1.64 | 0.05 | 0.10 | Liberal screening |
| 1.96 | 0.025 | 0.05 | Standard threshold |
| 2.33 | 0.01 | 0.02 | Conservative |
| 2.58 | 0.005 | 0.01 | Very conservative |
| 3.09 | 0.001 | 0.002 | Highly stringent |

**Recommended**: Use z = 2.3 (approximately p < 0.01) for exploratory analysis, then validate significant findings with randomise.

### Multiple Comparison Correction

GLM uses **cluster-based thresholding**:
1. Threshold z-statistic map at chosen z-value
2. Identify connected clusters of suprathreshold voxels
3. Filter clusters smaller than minimum size

Randomise uses **TFCE** (Threshold-Free Cluster Enhancement):
- No arbitrary threshold selection
- Integrates cluster extent and height
- More sensitive to spatially extended signals

### Comparing Methods

When running both methods, look for:

1. **Agreement** - Findings significant in both methods (high confidence)
2. **GLM-only** - May be false positives or parametric assumption violations
3. **Randomise-only** - May reflect distributional features not captured by GLM
4. **Spatial concordance** - Similar cluster locations but different extents

## Best Practices

### 1. Always Start with GLM for Exploratory Analysis

```bash
# Quick GLM check (seconds)
python -m neurovrai.analysis.tbss.run_tbss_stats \
    --method glm \
    --z-threshold 2.3

# If significant, validate with randomise (minutes)
python -m neurovrai.analysis.tbss.run_tbss_stats \
    --method randomise \
    --n-permutations 5000
```

### 2. Use Consistent Thresholds

For fair comparison between methods:
- GLM z-threshold 2.3 ≈ randomise p < 0.01
- GLM z-threshold 1.96 ≈ randomise p < 0.05

### 3. Report Both Methods for Important Findings

```
"Age showed a positive association with FA in bilateral corticospinal tracts
(parametric GLM: z > 2.3, 547 voxels; nonparametric randomise with TFCE:
p < 0.01, 521 voxels; Figure 2)."
```

### 4. Check Assumptions for GLM

Before trusting GLM results, verify:
- **Normality**: Check residual distributions
- **Homoscedasticity**: Variance should be constant across fitted values
- **Outliers**: Extreme values can disproportionately affect parametric tests

If assumptions violated → use randomise exclusively

## Command-Line Reference

### TBSS Statistical Analysis

```bash
python -m neurovrai.analysis.tbss.run_tbss_stats \
    --data-dir <prepared_data>     # TBSS skeleton data
    --participants <csv_file>      # Participant demographics
    --formula "predictor1 + ..."   # Model formula
    --contrasts-file <yaml_file>   # Contrast specifications
    --output-dir <output_path>     # Results directory
    --method {randomise|glm|both}  # Statistical method
    --z-threshold <float>          # GLM threshold (default: 2.3)
    --n-permutations <int>         # Randomise permutations (default: 5000)
    --cluster-threshold <float>    # Randomise threshold (default: 0.95)
    --min-cluster-size <int>       # Min voxels per cluster (default: 10)
```

### Functional Group Analysis

Modify `run_func_group_analysis.py`:

```python
# Line 395: Choose method
method = 'glm'  # or 'randomise' or 'both'
```

Then run:

```bash
python run_func_group_analysis.py
```

## Python API Reference

### `run_fsl_glm()`

Execute FSL's fsl_glm for parametric analysis.

```python
def run_fsl_glm(
    input_file: Path,           # 4D input volume
    design_mat: Path,           # FSL design matrix (.mat)
    contrast_con: Path,         # FSL contrasts (.con)
    output_dir: Path,           # Output directory
    mask: Optional[Path] = None,# Binary mask
    demean: bool = False,       # Demean data temporally
    variance_normalization: bool = False  # Variance normalization
) -> Dict
```

**Returns**: Dictionary with execution results and output file paths

### `threshold_zstat()`

Apply cluster-based thresholding to z-statistic maps.

```python
def threshold_zstat(
    zstat_file: Path,           # Z-statistic map
    output_dir: Path,           # Output directory
    z_threshold: float = 2.3,   # Z-score threshold
    cluster_threshold: int = 10,# Min cluster size
    mask: Optional[Path] = None # Analysis mask
) -> Dict
```

**Returns**: Dictionary with cluster statistics for each contrast

### `summarize_glm_results()`

Summarize significant findings across all contrasts.

```python
def summarize_glm_results(
    output_dir: Path,                    # GLM output directory
    contrast_names: Optional[List[str]] = None,  # Contrast names
    z_threshold: float = 2.3             # Significance threshold
) -> Dict
```

**Returns**: Dictionary with summary statistics and significant contrasts

## Troubleshooting

### GLM outputs are empty

**Problem**: `glm_zstat.nii.gz` has all zeros

**Solution**:
1. Check design matrix matches number of volumes in 4D image
2. Verify contrast vectors have correct length
3. Check mask includes non-zero voxels

### Different results between GLM and Randomise

**Expected**: Some differences are normal due to:
- Different assumptions (parametric vs nonparametric)
- Different multiple comparison methods (cluster vs TFCE)
- Different thresholds (z vs permutation-based p)

**Concerning** if:
- No overlap in significant regions
- Opposite effect directions
- Suggests assumption violations → use randomise

### FSL GLM not found

**Problem**: `fsl_glm not found` error

**Solution**:
```bash
# Check FSL installation
echo $FSLDIR

# Load FSL environment
source $FSLDIR/etc/fslconf/fsl.sh

# Verify fsl_glm available
which fsl_glm
```

## References

1. **FSL GLM Documentation**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/GLM
2. **Randomise Documentation**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise
3. **TFCE Paper**: Smith, S.M. & Nichols, T.E. (2009). Threshold-free cluster enhancement. NeuroImage, 44(1), 83-98.
4. **Parametric vs Nonparametric**: Eklund, A., et al. (2016). Cluster failure: Why fMRI inferences for spatial extent have inflated false-positive rates. PNAS, 113(28), 7900-7905.

## Support

For issues or questions:
1. Check FSL installation: `which fsl_glm`
2. Verify input files exist and are valid NIfTI format
3. Review log files in `{output_dir}/glm_output/glm.log`
4. Compare with randomise results to identify inconsistencies
