# Configuration Guide

## Overview

The MRI preprocessing pipeline uses YAML configuration files to define all processing parameters, file paths, and sequence mappings. This approach eliminates hardcoded values and makes the pipeline adaptable to different studies and scanners.

## Configuration File Structure

Configurations use a two-tier system:
1. **`configs/default.yaml`**: Default parameters for all studies
2. **`configs/your_study.yaml`**: Study-specific overrides and paths

Study-specific configs inherit from defaults and override only what's needed.

---

## Auto-Generation from DICOM

### Quick Start: Generate Config from DICOMs

The easiest way to create a study configuration is to auto-generate it from DICOM headers:

```bash
mri-preprocess config init \
  --dicom-dir /path/to/subject/dicoms \
  --output configs/my_study.yaml \
  --study-name "My Study" \
  --study-code "STUDY01"
```

This will:
1. Scan DICOM headers to detect available sequences
2. Extract scanner parameters (TR, TE, voxel size, etc.)
3. Auto-populate sequence mappings
4. Generate a complete config with recommended defaults
5. Flag any sequences that don't match known patterns for manual review

### Manual Review Required

After auto-generation, **review and edit** the config:
- Verify sequence mappings are correct
- Set study-specific paths
- Adjust preprocessing parameters as needed
- Add any custom sequences not auto-detected

---

## Mandatory Parameters

These parameters **must** be specified in your study config:

### Study Information

```yaml
study:
  name: "My Research Study"           # REQUIRED: Full study name
  code: "STUDY01"                      # REQUIRED: Short study code
  base_dir: "/path/to/study"           # REQUIRED: Root directory for outputs
```

### Paths

```yaml
paths:
  rawdata: "/path/to/bids/rawdata"                    # REQUIRED: NIfTI data location
  derivatives: "/path/to/bids/derivatives"            # REQUIRED: Preprocessed outputs

  # Optional but recommended
  sourcedata: "/path/to/dicom"                        # Original DICOMs
  logs: "${study.base_dir}/logs"                      # Execution logs
  transforms: "${study.base_dir}/transforms"          # Transformation registry
```

### Sequence Mappings

At minimum, specify **one modality** you want to process:

```yaml
sequence_mappings:
  t1w:
    - "MPRAGE"                         # REQUIRED if running anatomical preprocessing
    - "T1_WEIGHTED"
```

**Full example**:
```yaml
sequence_mappings:
  t1w:
    - "MPRAGE"
    - "3D_T1_TFE_SAG_CS3"
  t2w:
    - "T2_SPACE"
    - "T2W_CS5"
  dwi:
    - "DTI"
    - "ep2d_diff"
  rest:
    - "BOLD_REST"
    - "RESTING_ME3_MB3"
  fmap:
    - "gre_field_mapping"
    - "SE_EPI"
```

---

## Optional Parameters

These have sensible defaults in `configs/default.yaml` but can be overridden:

### FSL Configuration

```yaml
fsl:
  fsldir: /usr/local/fsl              # Default: from $FSLDIR environment variable
  output_type: NIFTI_GZ               # Default: NIFTI_GZ (options: NIFTI, NIFTI_GZ)
```

### MNI Templates

```yaml
templates:
  mni152_t1_2mm: /path/to/MNI152_T1_2mm_brain.nii.gz
  mni152_mask_2mm: /path/to/MNI152_T1_2mm_brain_mask.nii.gz
  # Defaults to FSL templates if not specified
```

### Anatomical Preprocessing

```yaml
anatomical:
  reorient: true                      # Default: true - reorient to standard

  bias_correction:
    method: fast                      # Default: fast (options: fast, ants)
    img_type: 1                       # Default: 1 (T1-weighted)
    output_biascorrected: true

  skull_strip:
    method: bet                       # Default: bet (options: bet, ants)
    frac: 0.5                         # Default: 0.5 (BET fractional intensity)
    reduce_bias: true                 # Default: true

  registration:
    to_mni: true                      # Default: true
    dof: 12                           # Default: 12 (degrees of freedom)
    cost_func: bbr                    # Default: bbr (boundary-based registration)
    nonlinear: true                   # Default: true (use FNIRT)
    use_fnirt: true

  segmentation:
    run: false                        # Default: false (only if needed for ACompCor)
    number_classes: 3                 # Default: 3 (GM, WM, CSF)
```

### Diffusion Preprocessing

```yaml
diffusion:
  shells:
    merge_multishell: true            # Default: true - combine shells
    extract_b0: true                  # Default: true

  eddy:
    use_cuda: true                    # Default: true (requires GPU)
    num_threads: 1                    # Default: 1
    acqp_file: /path/to/acqp.txt      # REQUIRED for eddy
    index_file: /path/to/index.txt    # REQUIRED for eddy

  skull_strip:
    frac: 0.5                         # Default: 0.5

  dtifit:
    save_tensor: true                 # Default: true
    sse: true                         # Default: true

  bedpostx:
    run: false                        # Default: false (very slow - 8-20hrs)
    use_gpu: true                     # Default: true
    n_fibres: 3                       # Default: 3
    burn_in: 200                      # Default: 200
    n_jumps: 5000                     # Default: 5000
    sample_every: 25                  # Default: 25

  probtrackx2:
    run: false                        # Default: false (requires BEDPOSTX)
    n_samples: 5000                   # Default: 5000
    curvature_threshold: 0.2          # Default: 0.2
    loopcheck: true                   # Default: true
```

### Functional Preprocessing

```yaml
functional:
  multi_echo:
    enabled: true                     # Default: true (auto-detect from file count)
    tedana:
      run: true                       # Default: true for multi-echo
      tedpca: auto                    # Default: auto (options: kundu, aic, kic, mdl)
      tedort: false                   # Default: false
      gscontrol: null                 # Default: null (options: t1c, mir, gsr)
      fittype: curvefit               # Default: curvefit

  motion_correction:
    method: mcflirt                   # Default: mcflirt
    cost: leastsquares                # Default: leastsquares
    dof: 6                            # Default: 6
    stages: 4                         # Default: 4
    interpolation: sinc               # Default: sinc
    save_plots: true                  # Default: true
    reference_echo: middle            # Default: middle (for multi-echo)

  skull_strip:
    frac: 0.3                         # Default: 0.3 (more liberal for functional)
    functional: true                  # Default: true

  smoothing:
    fwhm: 6                           # Default: 6mm

  normalize:
    method: grand_mean                # Default: grand_mean
    target: 1000                      # Default: 1000

  nuisance_regression:
    ica_aroma:
      run: true                       # Default: true
      denoise_type: both              # Default: both (options: aggr, nonaggr, both)
    acompcor:
      run: true                       # Default: true
      num_components: 6               # Default: 6
      mask_type: combined             # Default: combined (options: csf, wm, combined)

  temporal_filter:
    method: afni_bandpass             # Default: afni_bandpass
    highpass: 0.001                   # Default: 0.001 Hz
    lowpass: 0.08                     # Default: 0.08 Hz

  coregistration:
    dof: 6                            # Default: 6
    cost_func: mutualinfo             # Default: mutualinfo
```

### Execution Settings

```yaml
execution:
  plugin: MultiProc                   # Default: MultiProc (options: Linear, MultiProc)
  n_procs: 2                          # Default: 2 (set based on CPU cores)
  remove_unnecessary_outputs: true    # Default: true
  keep_inputs: true                   # Default: true
  stop_on_first_crash: false          # Default: false (set to true for debugging)
  crashdump_dir: null                 # Default: null (use for debugging)
```

### Logging

```yaml
logging:
  level: INFO                         # Default: INFO (options: DEBUG, INFO, WARNING, ERROR)
  log_directory: logs                 # Default: logs
  workflow_log_directory: workflow_logs
```

### Anonymization

```yaml
anonymization:
  enabled: true                       # Default: false (set to true for shared data)
  strip_headers: true                 # Default: false
  dcm2niix_flags: "-ba y"             # Default: "" (BIDS anonymization)
  remove_fields:                      # Fields to strip from JSON sidecars
    - PatientName
    - PatientID
    - PatientBirthDate
    - AcquisitionDate
    - SeriesDate
```

---

## Environment Variable Substitution

Use `${VAR}` syntax to reference environment variables or other config values:

```yaml
study:
  base_dir: "/data/studies/mystudy"

paths:
  rawdata: "${study.base_dir}/rawdata"
  derivatives: "${study.base_dir}/derivatives"
  logs: "${study.base_dir}/logs"

fsl:
  fsldir: "${FSLDIR}"                 # From environment variable
```

---

## Subject-Specific Overrides

You can override parameters for specific subjects by passing them via CLI:

```bash
mri-preprocess anat \
  --config configs/my_study.yaml \
  --subject sub-001 \
  --override anatomical.skull_strip.frac=0.3
```

---

## Validation

Validate your configuration before running:

```bash
mri-preprocess config validate --config configs/my_study.yaml
```

This checks:
- All mandatory parameters are present
- File paths exist
- Sequence mappings are non-empty
- Parameter values are valid
- FSL environment is set up correctly

---

## Example: Minimal Configuration

Bare minimum config for anatomical preprocessing only:

```yaml
study:
  name: "Quick Test"
  code: "TEST"
  base_dir: "/data/test"

paths:
  rawdata: "/data/test/rawdata"
  derivatives: "/data/test/derivatives"

sequence_mappings:
  t1w:
    - "MPRAGE"
```

---

## Example: Full Multi-Modal Study

Complete configuration for all modalities:

```yaml
study:
  name: "Multimodal Brain Study"
  code: "MBS2024"
  base_dir: "/data/mbs2024"

paths:
  sourcedata: "/data/mbs2024/sourcedata"
  rawdata: "/data/mbs2024/rawdata"
  derivatives: "/data/mbs2024/derivatives"
  transforms: "/data/mbs2024/transforms"
  logs: "/data/mbs2024/logs"

sequence_mappings:
  t1w:
    - "MPRAGE"
    - "T1_3D_TFE"
  t2w:
    - "T2_SPACE"
  dwi:
    - "DTI_64dir"
    - "ep2d_diff_b1000"
    - "ep2d_diff_b2000"
  rest:
    - "BOLD_REST_ME"
  fmap:
    - "gre_field_mapping"

diffusion:
  eddy:
    acqp_file: "/data/mbs2024/configs/acqp.txt"
    index_file: "/data/mbs2024/configs/index.txt"
  bedpostx:
    run: true
    use_gpu: true

functional:
  multi_echo:
    enabled: true
    tedana:
      run: true

execution:
  n_procs: 8

anonymization:
  enabled: true
  strip_headers: true
```

---

## Tips and Best Practices

### 1. Start with Auto-Generation
Always start by auto-generating from DICOMs, then customize.

### 2. Use Separate Configs Per Study
Don't modify `default.yaml` - create study-specific configs.

### 3. Version Control Your Configs
Keep configs in git with your analysis code.

### 4. Test with One Subject First
Validate your config on a single subject before batch processing.

### 5. Document Custom Sequences
Add comments to explain non-standard sequence names:

```yaml
sequence_mappings:
  rest:
    - "BOLD_REST_ME"          # Multi-echo, 3 echoes, TR=1104ms
    - "rs_fMRI_custom"        # Custom sequence from pilot data
```

### 6. Use Descriptive Study Codes
Choose codes that make sense in file paths:
- ✅ `ADHD_2024`, `AGING_MRI`, `TRAUMA_STUDY`
- ❌ `STUDY1`, `TEST`, `ABC`

### 7. Set Realistic n_procs
Don't exceed available CPU cores:

```yaml
execution:
  n_procs: 4  # For 8-core machine, leave room for system
```

---

## Troubleshooting

### Config not loading?
- Check YAML syntax (indentation matters!)
- Validate with: `mri-preprocess config validate`
- Look for tabs (use spaces only)

### Sequences not detected?
- Run auto-generation to see what scanner uses
- Check DICOM headers manually: `dcmdump dicom_file.dcm | grep SeriesDescription`
- Add exact sequence name to mappings

### Paths not resolving?
- Use absolute paths, not relative
- Check environment variable substitution
- Verify directories exist before running

---

## Reference

### Sequence Mapping Strategy

**Pattern Matching**: Sequence mappings use substring matching. The pipeline looks for any of the listed strings in the DICOM SeriesDescription:

```yaml
sequence_mappings:
  t1w:
    - "MPRAGE"        # Matches "MPRAGE", "T1_MPRAGE", "MPRAGE_SAG"
    - "T1"            # Matches "T1_3D", "3D_T1_TFE", etc.
```

**Best Practice**: Be specific to avoid false matches:
- ✅ `"T1_WEIGHTED"` (specific)
- ❌ `"T1"` (too broad, might match "MT1" or "T1rho")

### Required Files for Diffusion

Eddy correction requires acquisition parameters:

**acqp.txt**: Acquisition parameters (one line per unique phase encoding)
```
0 -1 0 0.05    # AP direction, readout time 0.05s
0 1 0 0.05     # PA direction, readout time 0.05s
```

**index.txt**: Maps each volume to an acqp line
```
1 1 1 1 1 ... (for all AP volumes)
2 2 2 2 2 ... (for all PA volumes)
```

Generate these with: `mri-preprocess dwi create-acqp --help`

---

For more documentation, see:
- `docs/workflows.md` - Individual workflow parameters
- `docs/cli.md` - Command-line interface
- `TESTING.md` - Development testing guide
