# Dual Regression Analysis Guide

## Overview

Dual regression is a technique for obtaining subject-specific spatial maps and time courses from group Independent Component Analysis (ICA). After performing group ICA with MELODIC, dual regression allows you to:

1. Extract subject-specific time courses for each group ICA component
2. Generate subject-specific spatial maps for each component
3. Perform group-level statistical comparisons on subject-specific maps

## When to Use Dual Regression

### Use Dual Regression when:
- **Subject-specific analysis** - You want individual subject maps from group ICA components
- **Statistical inference** - You need to compare groups or test for covariate effects
- **Network analysis** - You're studying functional connectivity networks at the individual level
- **Longitudinal studies** - Tracking changes in network expression over time

### Typical Workflow:
1. **Group ICA** (MELODIC) - Identify common functional networks across all subjects
2. **Dual Regression** - Obtain subject-specific versions of these networks
3. **Statistical Analysis** - Compare networks between groups or test associations with covariates

## How Dual Regression Works

Dual regression is a two-stage process:

### Stage 1: Spatial Regression
Regress the group ICA spatial maps against each subject's 4D fMRI data to obtain subject-specific time courses.

```
For each subject:
  For each ICA component:
    Time_Course[subject, component] = regress(Group_IC_Map[component], Subject_4D_Data)
```

**Output**: Subject-specific time courses for each component

### Stage 2: Temporal Regression
Regress the subject-specific time courses against the subject's 4D data to obtain subject-specific spatial maps.

```
For each subject:
  Spatial_Map[subject] = regress(Time_Courses[subject], Subject_4D_Data)
```

**Output**: Subject-specific spatial maps (one 4D volume per subject with all components)

### Stage 3: Group Statistics (Optional)
If a design matrix and contrasts are provided, perform group-level statistical analysis using FSL randomise with permutation testing.

```
For each ICA component:
  Statistical_Maps[component] = randomise(Subject_Spatial_Maps[component], Design, Contrasts)
```

**Output**: Group statistical maps (one set per component)

## Usage Examples

### 1. Basic Dual Regression (No Statistics)

```python
from pathlib import Path
from neurovrai.analysis.func.dual_regression import run_dual_regression

# Run dual regression
results = run_dual_regression(
    group_ic_maps=Path('/study/melodic/melodic_IC.nii.gz'),
    subject_files=[
        Path('/study/derivatives/sub-001/func/preprocessed_mni.nii.gz'),
        Path('/study/derivatives/sub-002/func/preprocessed_mni.nii.gz'),
        # ... more subjects
    ],
    output_dir=Path('/study/analysis/dual_regression')
)

# Access outputs
print(f"Processed {results['n_subjects']} subjects")
print(f"Components: {results['n_components']}")
print(f"Subject spatial maps: {len(results['stage2_files'])}")
```

### 2. Dual Regression with Group Statistics

```python
from pathlib import Path
from neurovrai.analysis.func.dual_regression import run_dual_regression

# Run dual regression with statistical inference
results = run_dual_regression(
    group_ic_maps=Path('/study/melodic/melodic_IC.nii.gz'),
    subject_files=[Path(f) for f in subject_list],
    output_dir=Path('/study/analysis/dual_regression'),
    design_mat=Path('/study/design.mat'),  # FSL design matrix
    contrast_con=Path('/study/design.con'),  # FSL contrasts
    n_permutations=5000  # For randomise
)

# Statistical maps will be in stage 3 outputs
for ic_name, stat_files in results['stage3_files'].items():
    print(f"{ic_name}: {len(stat_files)} statistical maps")
```

### 3. Complete MELODIC + Dual Regression Workflow

```python
from pathlib import Path
from neurovrai.analysis.func.melodic import run_melodic_group_ica
from neurovrai.analysis.func.dual_regression import run_dual_regression

# Step 1: Group ICA with MELODIC
melodic_results = run_melodic_group_ica(
    subject_files=[
        Path('/study/derivatives/sub-001/func/preprocessed_mni.nii.gz'),
        Path('/study/derivatives/sub-002/func/preprocessed_mni.nii.gz'),
        # ... more subjects
    ],
    output_dir=Path('/study/analysis/melodic'),
    n_components=20,  # or 'auto'
    approach='concat'
)

print(f"MELODIC identified {melodic_results['n_components']} components")

# Step 2: Dual regression for subject-specific maps
dr_results = run_dual_regression(
    group_ic_maps=Path(melodic_results['melodic_ic']),
    subject_files=[Path(f) for f in melodic_results['subject_files']],
    output_dir=Path('/study/analysis/dual_regression'),
    design_mat=Path('/study/design.mat'),
    contrast_con=Path('/study/design.con'),
    n_permutations=5000
)

print(f"Dual regression complete: {dr_results['n_subjects']} subjects")
```

### 4. Command-Line Usage

```bash
# Create subject list file
ls /study/derivatives/*/func/preprocessed_mni.nii.gz > subjects.txt

# Run dual regression (no statistics)
python -m neurovrai.analysis.func.dual_regression \
    --group-ics /study/melodic/melodic_IC.nii.gz \
    --subjects-file subjects.txt \
    --output-dir /study/dual_regression

# Run with group statistics
python -m neurovrai.analysis.func.dual_regression \
    --group-ics /study/melodic/melodic_IC.nii.gz \
    --subjects-file subjects.txt \
    --output-dir /study/dual_regression \
    --design /study/design.mat \
    --contrasts /study/design.con \
    --n-permutations 5000
```

## Output Files

Dual regression creates the following outputs in `{output_dir}`:

```
dual_regression/
├── dr_stage1_subject00000/          # Stage 1: Subject 1 time courses
│   └── [component time course files]
├── dr_stage1_subject00001/          # Stage 1: Subject 2 time courses
│   └── [component time course files]
├── dr_stage2_subject00000.nii.gz    # Stage 2: Subject 1 spatial maps (4D)
├── dr_stage2_subject00001.nii.gz    # Stage 2: Subject 2 spatial maps (4D)
├── dr_stage3_ic0000_*.nii.gz        # Stage 3: Statistics for component 1
├── dr_stage3_ic0001_*.nii.gz        # Stage 3: Statistics for component 2
├── subject_list.txt                 # List of input files
├── command.log                      # Dual regression command output
├── dual_regression_TIMESTAMP.log    # Execution log
└── dual_regression_results.json     # Results summary
```

### Stage 2 Outputs (Most Important)

**Stage 2 files** (`dr_stage2_subjectXXXXX.nii.gz`) contain subject-specific spatial maps:
- One 4D file per subject
- Each volume corresponds to one ICA component
- Units: Z-scores (standardized)
- These are used for group-level statistics in Stage 3

### Stage 3 Outputs (Statistical Inference)

If a design matrix is provided, **Stage 3 files** contain group statistical maps:
- One set of files per ICA component
- Format: `dr_stage3_ic####_tstat#.nii.gz` (t-statistics)
- Format: `dr_stage3_ic####_tfce_corrp_tstat#.nii.gz` (TFCE-corrected p-values)
- Interpret like any randomise output

## Design Matrix and Contrasts

Dual regression uses standard FSL design matrices and contrasts. See the main analysis documentation for creating these.

### Example: Group Comparison

**Design Matrix** (`design.mat`):
- Column 1: Intercept (all 1's)
- Column 2: Group (0 for controls, 1 for patients)

**Contrasts** (`design.con`):
- Contrast 1: Patients > Controls `[0, 1]`
- Contrast 2: Controls > Patients `[0, -1]`

This will test for group differences in each ICA component network.

### Example: Correlation with Covariate

**Design Matrix**:
- Column 1: Intercept (all 1's)
- Column 2: Age (demeaned)
- Column 3: Sex (0/1)

**Contrasts**:
- Contrast 1: Age positive `[0, 1, 0]`
- Contrast 2: Age negative `[0, -1, 0]`

This tests for age-related changes in network expression, controlling for sex.

## Best Practices

### 1. Preprocessing Requirements

Dual regression requires that:
- **All subjects are in the same space** (e.g., MNI152)
- **Preprocessing is consistent** across subjects
- **Same TR** for all subjects
- **Group ICA was performed on the same data** (or spatially normalized equivalents)

### 2. Choosing Number of Components

For group ICA before dual regression:
- **Automatic estimation**: Let MELODIC estimate (good default)
- **Low dimensional** (10-30 components): Captures major networks (DMN, sensorimotor, visual)
- **High dimensional** (50-100 components): Captures fine-grained sub-networks
- **Very high dimensional** (>100): May capture noise, harder to interpret

**Recommendation**: Start with automatic estimation or 20-30 components for typical resting-state studies.

### 3. Interpreting Results

**Stage 2 (Subject Maps)**:
- Z-scores represent how strongly each voxel expresses the component
- Higher values = stronger expression of the network pattern
- Can be used for individual-level analysis or visualization

**Stage 3 (Group Statistics)**:
- Use TFCE-corrected p-value maps (`tfce_corrp_tstat#.nii.gz`)
- Threshold at p < 0.05 (1 - p > 0.95 in corrp map)
- Report component name/number and contrast
- Visualize significant clusters on standard brain

### 4. Multiple Comparison Correction

Stage 3 uses FSL randomise with:
- **TFCE** (Threshold-Free Cluster Enhancement) for spatial correction
- **Permutation testing** (default 5000 permutations) for family-wise error control
- Correction is **per component** (not across all components)

**Important**: Testing multiple components increases multiple comparison burden. Consider:
- Bonferroni correction across components (divide alpha by number of components)
- Focus on a priori components of interest (e.g., DMN, executive control network)
- Report number of components tested

### 5. Quality Control

Before dual regression:
- **Inspect MELODIC components** visually - remove clear artifacts (motion, CSF, vessels)
- **Check component time courses** - should not correlate with motion parameters
- **Verify spatial patterns** - should match known functional networks

After dual regression:
- **Check stage 2 maps** - should show spatial patterns similar to group IC maps
- **Verify statistical maps** - check that effects are anatomically plausible
- **Compare across contrasts** - opposite contrasts should show opposite effects

## Troubleshooting

### Dimension Mismatch Errors

**Problem**: Subject data doesn't match group ICA spatial dimensions

**Solution**:
- Ensure all subjects and group ICs are in same space (e.g., MNI152 2mm)
- Re-run preprocessing to normalize all data to common space
- Check that group ICA was run on normalized data

### No Stage 3 Outputs

**Problem**: Dual regression completed but no statistical maps generated

**Possible causes**:
1. No design matrix provided (Stage 3 is optional)
2. Design matrix path was incorrect
3. Randomise failed (check `command.log`)

**Solution**: Check design matrix exists and is valid FSL format

### Memory Issues

**Problem**: Dual regression fails with memory errors

**Solution**:
- Reduce number of subjects per run
- Reduce spatial resolution (downsample to 3mm or 4mm)
- Use cluster computing for large datasets
- Reduce number of permutations (though 5000 is recommended)

### FSL Not Found

**Problem**: `dual_regression: command not found`

**Solution**:
```bash
# Check FSL installation
echo $FSLDIR

# Load FSL environment
source $FSLDIR/etc/fslconf/fsl.sh

# Verify dual_regression available
which dual_regression
```

## Performance Considerations

### Computation Time

Approximate times for 30 subjects, 20 components, 200 time points:

- **Stage 1+2 (spatial/temporal regression)**: 5-10 minutes
- **Stage 3 (5000 permutations)**: 1-3 hours per component
  - Total Stage 3 time: 20-60 hours for 20 components
  - Can run components in parallel

### Parallelization

FSL's dual_regression:
- Processes subjects in parallel automatically
- Stage 3 (randomise) runs independently per component
- Can manually parallelize by running multiple components simultaneously

### Optimization Tips

1. **Use fewer permutations** for exploratory analysis (e.g., 500-1000)
2. **Downsample data** to 3mm or 4mm resolution
3. **Select components of interest** instead of testing all components
4. **Use cluster computing** for large datasets (>50 subjects)

## References

1. **Dual Regression Method**: Filippini et al. (2009). "Distinct patterns of brain activity in young carriers of the APOE-ε4 allele." PNAS, 106(17), 7209-7214.
2. **FSL Dual Regression**: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/DualRegression
3. **MELODIC/ICA**: Beckmann & Smith (2004). "Probabilistic independent component analysis for functional magnetic resonance imaging." IEEE TMI, 23(2), 137-152.
4. **TFCE**: Smith & Nichols (2009). "Threshold-free cluster enhancement: addressing problems of smoothing, threshold dependence and localisation in cluster inference." NeuroImage, 44(1), 83-98.

## Support

For issues or questions:
1. Check FSL installation: `which dual_regression`
2. Verify all inputs are in the same space
3. Review log files: `command.log` and `dual_regression_TIMESTAMP.log`
4. Ensure preprocessing included spatial normalization to standard space
