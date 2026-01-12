# Cohen's d Effect Size Maps from FSL Randomise

## Overview

We've implemented functionality to generate standardized effect size maps (Cohen's d) from FSL randomise t-statistic outputs. This provides interpretable measures of effect magnitude that are independent of sample size, addressing the limitation that t-statistics conflate effect size with sample size.

## Why Effect Sizes Matter

- **T-statistics** increase with both effect size AND sample size
- **Cohen's d** provides a standardized measure independent of N
- Allows comparison across studies with different sample sizes
- Provides clinical interpretation: d=0.2 (small), d=0.5 (medium), d=0.8 (large)

## Implementation

### Module: `neurovrai/analysis/stats/effect_size.py`

Key functions:
- `t_to_cohens_d()`: Converts t-statistics to Cohen's d
- `create_effect_size_maps()`: Generates NIfTI maps from randomise outputs
- `batch_effect_size_calculation()`: Processes all contrasts automatically

### Conversion Formulas

For **two-sample t-tests**:
```
d = t * sqrt(1/n1 + 1/n2)
```

For **one-sample or paired t-tests**:
```
d = t / sqrt(n)
```

## Usage

### Basic Command Line Usage

```bash
# Calculate effect sizes from existing randomise results
python scripts/analysis/calculate_effect_sizes.py \
    --randomise-dir results/tbss_randomise \
    --design-file design.csv \
    --output-dir results/effect_sizes

# Specify sample sizes manually
python scripts/analysis/calculate_effect_sizes.py \
    --randomise-dir results/randomise_output \
    --n1 60 --n2 60 \
    --design-type two_sample \
    --output-dir results/effect_sizes
```

### Python API Usage

```python
from neurovrai.analysis.stats.effect_size import create_effect_size_maps

# Create effect size maps from t-statistic file
results = create_effect_size_maps(
    tstat_file=Path("randomise_tstat1.nii.gz"),
    output_dir=Path("effect_sizes/"),
    design_info={
        'n1': 60,
        'n2': 60,
        'design_type': 'two_sample',
        'contrast_name': 'GDM_vs_Control'
    },
    corrp_file=Path("randomise_tfce_corrp_tstat1.nii.gz")  # Optional
)
```

## Outputs

For each contrast, the following files are generated:

1. **Uncorrected Cohen's d map** (`contrast_cohens_d_uncorrected.nii.gz`)
   - Full effect size map without thresholding
   - Useful for understanding overall effect patterns

2. **Thresholded Cohen's d map** (`contrast_cohens_d_p0.05.nii.gz`)
   - Only voxels surviving multiple comparison correction
   - Created if corrected p-value file is provided

3. **Hedges' g map** (`contrast_hedges_g_uncorrected.nii.gz`)
   - Bias-corrected effect size (important for small samples)

4. **Confidence interval maps**
   - Lower and upper 95% CI bounds
   - Useful for understanding effect size precision

5. **Statistics JSON** (`contrast_effect_size_stats.json`)
   - Summary statistics (mean, median, range)
   - Effect size distribution (% small/medium/large)

## Example Output

```
GDM_vs_Control:
  Mean Cohen's d: 0.421
  Effect size distribution:
    Small (0.2-0.5): 62.3%
    Medium (0.5-0.8): 28.1%
    Large (>0.8): 9.6%
  Significant voxels (p<0.05): 12,453
  Mean d (significant only): 0.687
```

## Advantages Over T-Statistics Alone

1. **Interpretability**: Effect sizes have established benchmarks
2. **Comparability**: Can compare across studies with different N
3. **Clinical Relevance**: Magnitude of effect, not just significance
4. **Power Analysis**: Can be used for future study planning

## Integration with Grant Applications

For grant applications, effect size maps provide:

- Evidence of **meaningful** (not just significant) differences
- Support for power calculations in future studies
- Comparison with published effect sizes in the field
- Transparent reporting of both corrected and uncorrected effects

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Rosenthal, R. (1991). Meta-analytic procedures for social research
- Lakens, D. (2013). Calculating and reporting effect sizes. Front Psychol, 4, 863