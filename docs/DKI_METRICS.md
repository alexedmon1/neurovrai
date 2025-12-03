# DKI Metric Naming Convention

## Overview

DKI (Diffusion Kurtosis Imaging) metrics use **short form names** in the CLI for user convenience, but the underlying file detection supports **multiple naming conventions** including DIPY's descriptive file names.

## Supported DKI Metrics

| Short Name | Full Name | DIPY File Name | Description |
|------------|-----------|----------------|-------------|
| **MK** | Mean Kurtosis | `mean_kurtosis.nii.gz` | Average kurtosis across all directions |
| **AK** | Axial Kurtosis | `axial_kurtosis.nii.gz` | Kurtosis along principal diffusion direction |
| **RK** | Radial Kurtosis | `radial_kurtosis.nii.gz` | Kurtosis perpendicular to principal direction |
| **KFA** | Kurtosis FA | `kurtosis_fa.nii.gz` | Fractional anisotropy of kurtosis tensor |

## File Detection Strategy

The code searches for DKI metrics in the following order:

1. **DIPY naming** (primary): `mean_kurtosis.nii.gz`, `axial_kurtosis.nii.gz`, etc.
2. **Short form**: `MK.nii.gz`, `AK.nii.gz`, etc.
3. **Legacy naming**: `dki_mk.nii.gz`, `dki_ak.nii.gz`, etc.

This ensures compatibility with:
- DIPY preprocessing output (descriptive names)
- Manual file naming (short form)
- Legacy pipelines (prefixed names)

## Expected Directory Structure

```
derivatives/
└── {subject}/
    └── dwi/
        └── dki/
            ├── mean_kurtosis.nii.gz       # MK - Required for discovery
            ├── axial_kurtosis.nii.gz      # AK
            ├── radial_kurtosis.nii.gz     # RK
            ├── kurtosis_fa.nii.gz         # KFA
            ├── kurtosis_tensor.nii.gz     # Full kurtosis tensor
            ├── mean_diffusivity.nii.gz    # MD (also from DKI)
            ├── axial_diffusivity.nii.gz   # AD
            └── radial_diffusivity.nii.gz  # RD
```

## CLI Usage

### TBSS Preparation

Use **short form names** (MK, AK, RK, KFA) in all commands:

```bash
# Mean Kurtosis
python -m neurovrai.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --metric MK \
    --fa-skeleton-dir /study/analysis/tbss/ \
    --output-dir /study/analysis/tbss/

# Axial Kurtosis
python -m neurovrai.analysis.tbss.prepare_tbss \
    --config config.yaml \
    --metric AK \
    --fa-skeleton-dir /study/analysis/tbss/ \
    --output-dir /study/analysis/tbss/
```

The code will automatically find the corresponding `mean_kurtosis.nii.gz` and `axial_kurtosis.nii.gz` files.

### Design Matrix Creation

Design matrix creation automatically discovers subjects with DKI data by looking for `mean_kurtosis.nii.gz`:

```python
from neurovrai.analysis.stats.design_matrix_matching import create_matched_design_for_analysis

result = create_matched_design_for_analysis(
    participants_file=Path('participants.tsv'),
    derivatives_dir=Path('/study/derivatives'),
    formula='age + sex + group',
    analysis_type='dki',  # Automatically finds subjects with MK files
    output_dir=Path('/study/designs/dki')
)
```

## Implementation Details

### File Mapping (prepare_tbss.py)

```python
DKI_FILE_MAPPING = {
    'MK': 'mean_kurtosis',
    'AK': 'axial_kurtosis',
    'RK': 'radial_kurtosis',
    'KFA': 'kurtosis_fa'
}
```

### File Discovery (collect_subject_data)

```python
file_name = DKI_FILE_MAPPING.get(metric, metric.lower())
possible_files = [
    dwi_dir / "dki" / f"{file_name}.nii.gz",  # DIPY naming
    dwi_dir / "dki" / f"{metric}.nii.gz",      # Short form
    dwi_dir / "dki" / f"dki_{metric.lower()}.nii.gz",  # Legacy
]
```

## Benefits

1. **User-Friendly**: Short CLI names (MK, AK) are easier to type
2. **Flexible**: Works with multiple naming conventions
3. **Compatible**: Supports DIPY output without renaming
4. **Clear**: File mapping makes intent explicit in code
5. **Maintainable**: Centralized mapping dictionary

## Related Documentation

- DTI metrics: FA, MD, AD, RD (in `derivatives/{subject}/dwi/dti/`)
- NODDI metrics: FICVF, ODI, FISO (in `derivatives/{subject}/dwi/noddi/`)
- See `neurovrai/analysis/tbss/prepare_tbss.py` for implementation
