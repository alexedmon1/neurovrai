# Project Dependencies Reference

Complete list of all dependencies in `pyproject.toml` with their purpose.

## Core Neuroimaging Packages

| Package | Version | Purpose |
|---------|---------|---------|
| **nipype** | ≥1.10.0 | Workflow engine for FSL/FreeSurfer/ANTs interfaces |
| **nibabel** | ≥5.3.2 | NIfTI file I/O and manipulation |
| **dipy** | ≥1.11.0 | Diffusion imaging: DTI, DKI, NODDI models |
| **dmri-amico** | ≥2.0.3 | Fast microstructure modeling (NODDI/SANDI/ActiveAx, 100x faster) |
| **tedana** | ≥25.0.1 | Multi-echo fMRI denoising (ICA-based) |
| **nilearn** | ≥0.10.0,<0.11 | Required by TEDANA (fMRI utilities) |
| **pydicom** | ≥3.0.1 | DICOM file reading and metadata extraction |

## Scientific Computing

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | ≥1.24,<3.0 | Array operations, numerical computing |
| **scipy** | ≥1.14 | Scientific algorithms (optimization, statistics) |
| **pandas** | ≥2.0 | Data tables for QC metrics and results |
| **scikit-learn** | ≥1.0 | Machine learning for ACompCor (PCA-based confounds) |

## Visualization & QC

| Package | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | ≥3.5.0 | QC plots and visualizations |
| **seaborn** | ≥0.12.0 | Enhanced QC visualizations (distributions, heatmaps) |

## Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| **click** | ≥8.1.0 | Command-line interface framework |
| **pyyaml** | ≥6.0 | YAML configuration file parsing |
| **packaging** | ≥20.0 | Version comparison for environment verification |

## Optional/Development

| Package | Version | Purpose |
|---------|---------|---------|
| **heudiconv** | ≥1.3.4 | DICOM to BIDS conversion (optional utility) |
| **spyder-kernels** | ≥3.1.0 | For Spyder IDE integration (optional) |

## External Dependencies (Not in pyproject.toml)

These must be installed separately on the system:

### Required

- **FSL** ≥6.0 - Neuroimaging tools (FLIRT, FNIRT, BET, TOPUP, eddy, BEDPOSTX)
  - Set `FSLDIR` environment variable
  - Command: `module load fsl` or `source $FSLDIR/etc/fslconf/fsl.sh`

- **ANTs** ≥2.0 - N4BiasFieldCorrection, Atropos segmentation
  - Usually bundled with Nipype or available via `apt-get install ants`

- **dcm2niix** - DICOM to NIfTI converter
  - Required for DICOM conversion step
  - Install: `apt-get install dcm2niix` or download from GitHub

### Optional

- **FreeSurfer** ≥7.0 - Cortical reconstruction (NOT production-ready in pipeline)
  - Set `FREESURFER_HOME` and `SUBJECTS_DIR`
  - Currently only detection/extraction hooks implemented

- **CUDA** ≥9.0 - For GPU acceleration
  - Required for `eddy_cuda` and `probtrackx2_gpu`
  - Load with: `module load cuda`

## Version Constraints Explained

### NumPy <3.0
TEDANA and other packages not yet compatible with NumPy 2.0's breaking changes.

### nilearn <0.11
TEDANA 25.x requires nilearn 0.10.x due to API changes in 0.11+.

### Python ≥3.13
Pipeline uses modern Python features (pattern matching, type hints).

## Dependency Management

### Installing All Dependencies

```bash
# Install/update all packages
uv sync

# Verify installation
uv run python verify_environment.py
```

### Adding New Packages

```bash
# Option 1: Edit pyproject.toml manually, then:
uv sync

# Option 2: Use uv to add (updates pyproject.toml automatically)
uv add package-name
```

### Updating Packages

```bash
# Update all to latest compatible versions
uv sync --upgrade

# Update specific package
uv pip install --upgrade package-name
```

## Checking Installed Versions

```bash
# List all packages
uv pip list

# Check specific package
uv pip show nipype

# Verify critical packages
uv pip list | grep -E "nipype|dipy|amico|tedana|nibabel"
```

## Common Issues

### "ModuleNotFoundError: No module named 'X'"

**Cause:** Package not installed or wrong environment

**Solution:**
```bash
rm -rf .venv
uv sync
```

### "ImportError: cannot import name 'X' from 'Y'"

**Cause:** Version incompatibility

**Solution:**
```bash
# Check versions match constraints
uv pip list | grep package-name

# If needed, reinstall with correct version
uv pip install 'package-name>=X.Y'
```

### "FSLDIR not set"

**Cause:** FSL not loaded

**Solution:**
```bash
module load fsl
# OR
source /usr/local/fsl/etc/fslconf/fsl.sh
```

## Development Dependencies

For development work, you may want to add:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1.0",
]
```

Install with: `uv sync --extra dev`

## License Compatibility

All packages use permissive licenses compatible with academic/commercial use:
- BSD: nibabel, dipy, nilearn, scikit-learn
- Apache 2.0: nipype, tedana
- MIT: click, pyyaml, matplotlib, seaborn, pandas
- LGPL: scipy, numpy

## Total Package Count

- **Python packages in pyproject.toml:** 17
- **Including sub-dependencies:** ~103
- **External system tools required:** 3 (FSL, ANTs, dcm2niix)
- **Optional system tools:** 2 (FreeSurfer, CUDA)

## Maintenance

Dependencies are reviewed and updated:
- Monthly: Security updates
- Quarterly: Version compatibility checks
- Annually: Major version upgrades (with testing)

Last updated: 2025-11-16
