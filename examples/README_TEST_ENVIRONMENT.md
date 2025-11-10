# MRI Preprocessing Development & Testing Environment

This directory contains the development testing environment for the `human-mri-preprocess` refactoring project.

## Purpose

Test each development step with real data to ensure:
1. Code works with actual MRI data (not just theory)
2. Outputs are valid and interpretable
3. Workflows complete without errors
4. Performance is acceptable

## Directory Structure

```
/mnt/bytopia/development/mri-preprocess/
├── test-data/              # Input data for testing
│   └── sub-0580101/        # Test subject
│       ├── sourcedata/     # Original DICOM data (symlinked)
│       ├── rawdata/        # Converted NIfTI data (BIDS-like)
│       └── derivatives/    # Preprocessed outputs
├── outputs/                # Pipeline outputs organized by phase
│   ├── phase01-structure/
│   ├── phase02-config/
│   ├── phase04-dicom/
│   ├── phase05-anatomical/
│   ├── phase06-diffusion/
│   ├── phase07-functional/
│   └── phase08-myelin/
├── logs/                   # Execution logs and workflow graphs
└── configs/                # Test configurations
```

## Test Subject: sub-0580101

**Source**: `/mnt/bytopia/IRC805/raw/dicom/IRC805-0580101`

**Available Modalities**:
- T1w: 3D_T1_TFE_SAG_CS3
- T2w: T2W_CS5_OF1_TR2500, T2W_Sagittal_Reformat
- Resting-state fMRI: Multi-echo (RESTING_ME3_MB3_SENSE3, fMRI_CORRECTION_MB3_ME3)
- DWI: Multi-shell (DTI_2shell_b1000_b2000_MB4, DTI_1shell_b3000_MB4)
- Fieldmaps: SE_EPI_Posterior
- ASL: pCASL sequences

## ⚠️ DE-IDENTIFICATION REQUIREMENTS

**CRITICAL**: All outputs from this test subject will be used as example data. All files and file contents MUST be de-identified:

### NIfTI Headers
- Strip patient name, ID, birthdate, scan date
- Use `dcm2niix -ba y` flag for BIDS anonymization
- Or use `pydeface` / `mri_deface` for structural images
- Replace dates with relative "days from session 1" if needed

### JSON Sidecars
- Remove: PatientName, PatientID, PatientBirthDate, AcquisitionDate, SeriesDate
- Keep: Sequence parameters (TR, TE, etc.) for reproducibility
- Preserve scanning parameters but strip identifiers

### File Naming
- **Folder names CAN use**: `sub-0580101` (generic code, non-identifying)
- **File names MUST be generic**:
  - ✅ `sub-0580101_T1w.nii.gz`
  - ❌ `IRC805-0580101_...` or patient name

### Implementation
- Add `--anonymize` flag to DICOM converter (Step 4.1)
- Add header stripping to all workflow outputs
- Validate de-identification before using as example data

## Testing Workflow

### Phase-by-Phase Testing

After completing each phase, test with sub-0580101:

**Phase 4 (DICOM Conversion)**:
```bash
cd ~/sandbox/human-mri-preprocess
mri-preprocess convert \
  --dicom-dir /mnt/bytopia/development/mri-preprocess/test-data/sub-0580101/sourcedata/dicom \
  --output-dir /mnt/bytopia/development/mri-preprocess/test-data/sub-0580101/rawdata \
  --anonymize
```

**Phase 5 (Anatomical)**:
```bash
mri-preprocess anat \
  --bids-dir /mnt/bytopia/development/mri-preprocess/test-data/sub-0580101/rawdata \
  --subject 0580101 \
  --output-dir /mnt/bytopia/development/mri-preprocess/outputs/phase05-anatomical
```

**Phase 6 (Diffusion)**:
```bash
mri-preprocess dwi \
  --bids-dir /mnt/bytopia/development/mri-preprocess/test-data/sub-0580101/rawdata \
  --subject 0580101 \
  --output-dir /mnt/bytopia/development/mri-preprocess/outputs/phase06-diffusion
```

**Phase 7 (Functional - Multi-echo with TEDANA)**:
```bash
mri-preprocess func \
  --bids-dir /mnt/bytopia/development/mri-preprocess/test-data/sub-0580101/rawdata \
  --subject 0580101 \
  --output-dir /mnt/bytopia/development/mri-preprocess/outputs/phase07-functional
```

**Full Pipeline Test**:
```bash
mri-preprocess run \
  --config /mnt/bytopia/development/mri-preprocess/configs/test_subject.yaml \
  --subject 0580101 \
  --steps all
```

## Validation Checklist

After each phase, verify:

- [ ] **Outputs exist**: Check expected files are created
- [ ] **No errors**: Review logs for crashes or warnings
- [ ] **Headers valid**: Use `fslinfo` / `nib-ls` to check NIfTI integrity
- [ ] **De-identified**: Inspect headers with `nib-ls -H` - no patient info
- [ ] **Workflow graph**: Visualize `.dot` files to understand pipeline
- [ ] **Visual QC**: Open in FSLeyes to spot obvious preprocessing errors
- [ ] **Transform reuse**: Verify transforms saved/loaded correctly

## Success Criteria

- All preprocessing steps complete without errors
- Output images pass visual QC
- All files are de-identified (safe for sharing)
- Transformation registry works (no duplicate computations)
- Multi-echo TEDANA runs successfully
- Example data ready for documentation

## Notes

- This is a **development environment** - outputs may change as code evolves
- Keep logs of each test run for debugging
- Document any issues or edge cases discovered
- Use this subject to test edge cases (e.g., missing modalities)
