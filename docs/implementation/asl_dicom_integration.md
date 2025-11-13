# ASL DICOM Parameter Extraction Integration

## Overview

Automated DICOM parameter extraction has been integrated into the ASL preprocessing workflow, allowing acquisition parameters to be automatically extracted from DICOM files when available.

## Implementation

### Location
`mri_preprocess/workflows/asl_preprocess.py`

### Features

1. **Automatic Detection**: When `dicom_dir` parameter is provided, the workflow automatically searches for DICOM files
2. **Parameter Extraction**: Extracts validated Philips pCASL parameters from DICOM private tags:
   - Labeling duration (τ): tag (2005,140a)
   - Post-labeling delay (PLD): tag (2005,1442)
   - Label-control order: tag (2005,1429)
   - Background suppression: tag (2005,1412)
3. **Parameter Override**: Extracted DICOM parameters override config defaults
4. **Source Logging**: Clear indication of parameter source (DICOM vs config)
5. **Fallback**: Gracefully falls back to config values if DICOM extraction fails

### Usage

```python
from mri_preprocess.workflows.asl_preprocess import run_asl_preprocessing

results = run_asl_preprocessing(
    config=config,
    subject='IRC805-0580101',
    asl_file=Path('asl_source.nii.gz'),
    output_dir=Path('/mnt/bytopia/IRC805'),
    dicom_dir=Path('/path/to/dicom/directory'),  # NEW parameter
    # labeling_duration and post_labeling_delay will be overridden by DICOM
)
```

### Workflow Behavior

**With DICOM directory provided:**
```
Attempting to extract ASL parameters from DICOM...
  DICOM directory: /path/to/dicom
  Using DICOM file: MR1104000001.dcm
  Found labeling duration in DICOM: 1.932 s
  Found PLD in DICOM: 2.031 s

ASL Acquisition Parameters:
  Labeling duration (τ): 1.932 s [DICOM]
  Post-labeling delay (PLD): 2.031 s [DICOM]
  Label-control order: control_first [DICOM]
```

**Without DICOM directory:**
```
ASL Acquisition Parameters (from config):
  Labeling duration (τ): 1.800 s
  Post-labeling delay (PLD): 1.800 s
  Label-control order: control_first
```

## Validation

### Test Results (IRC805-0580101)

**Test Script**: `archive/tests/test_asl_dicom_integration.py`

**Results**:
- ✓ DICOM parameter extraction integrated
- ✓ Parameters extracted and used in quantification
- ✓ Parameter sources logged (DICOM vs config)
- ✓ Accurate extraction: τ=1.932s, PLD=2.031s

**Comparison with Manual Extraction**:
- Manual DICOM inspection: τ=1.932s, PLD=2.031s
- Automated extraction: τ=1.932s, PLD=2.031s
- **Result**: Perfect match ✓

## Benefits

1. **Eliminates Manual Parameter Entry**: No need to manually look up acquisition parameters
2. **Reduces Human Error**: Automated extraction is more reliable than manual transcription
3. **Scanner-Specific Accuracy**: Uses actual scanner parameters instead of protocol defaults
4. **Audit Trail**: Parameters are logged with their source for reproducibility
5. **Backward Compatible**: Works with existing code that doesn't provide DICOM directory

## Technical Details

### DICOM Tag Mapping (Philips Ingenia Elition X 3T)

| Parameter | DICOM Tag | Description |
|-----------|-----------|-------------|
| Labeling Duration (τ) | (2005,140a) | Time arterial blood is labeled |
| Post-Labeling Delay (PLD) | (2005,1442) | Time between labeling and imaging |
| Background Suppression | (2005,1412) | Number of BS pulses |
| Volume Type | (2005,1429) | CONTROL or LABEL identifier |

### Parameter Precedence

1. **DICOM** (if dicom_dir provided and extraction succeeds)
2. **Config** (if DICOM not available or extraction fails)
3. **Function defaults** (fallback if neither available)

### Error Handling

The integration includes robust error handling:
- Missing DICOM directory → log warning, use config
- No DICOM files found → log warning, use config
- DICOM read error → log warning, use config
- Invalid parameter values → log warning, use config

This ensures the workflow always completes successfully, even if DICOM extraction fails.

## References

- Philips pCASL DICOM specification (private tags group 2005)
- Alsop et al. (2015). Recommended implementation of arterial spin-labeled perfusion MRI. *MRM*, 73(1)
- `mri_preprocess/utils/dicom_asl_params.py`: DICOM extraction utility
- `docs/implementation/asl_parameter_investigation.md`: Parameter validation study

## Maintenance Notes

### Adding New Scanners

To support additional scanner vendors:
1. Identify vendor-specific DICOM tags for ASL parameters
2. Update `dicom_asl_params.py` with new tag mappings
3. Add vendor detection logic
4. Test on example DICOM files
5. Update documentation

### Parameter Validation

Extracted parameters include sanity checks:
- τ: 1.0 < value < 3.0 seconds
- PLD: 1.0 < value < 4.0 seconds
- Invalid values are rejected and config defaults are used

## Status

**Implementation Status**: ✅ COMPLETE
**Testing Status**: ✅ VALIDATED on Philips Ingenia Elition X 3T
**Documentation Status**: ✅ COMPLETE
**Integration Status**: ✅ MERGED into main workflow
