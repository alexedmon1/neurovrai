# Testing Strategy

## Development Testing Environment

**Location**: `/mnt/bytopia/development/mri-preprocess/`

This external directory contains:
- Test data (sub-0580101 from IRC805 study)
- Test outputs organized by phase
- Execution logs
- Test configurations

**See**: `/mnt/bytopia/development/mri-preprocess/README.md` for detailed testing procedures

## Test Subject

**Subject**: sub-0580101
- Multi-echo resting-state fMRI (3 echoes)
- Multi-shell diffusion (b1000, b2000, b3000)
- T1w and T2w structural
- Fieldmaps and ASL

## Testing Workflow

After implementing each major workflow phase:

1. Run workflow on sub-0580101
2. Validate outputs (structure, content, quality)
3. Visual QC in FSLeyes
4. Check de-identification (no patient info in headers)
5. Document issues before proceeding to next phase

## Configuration

Test configuration: `/mnt/bytopia/development/mri-preprocess/configs/test_subject_0580101.yaml`

## De-identification

**Critical**: All outputs must be anonymized
- DICOM→NIfTI: Use `dcm2niix -ba y`
- Strip patient info from headers
- Generic BIDS file naming
- Safe to share as example data

See PROJECT_PLAN.md "Testing Environment" section for complete strategy.
