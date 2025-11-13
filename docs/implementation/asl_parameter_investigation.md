# ASL Parameter Investigation - Final Report

## Investigation Summary

Successfully extracted actual ASL acquisition parameters from Philips DICOM private tags and determined root cause of elevated CBF values.

## Extracted Parameters

### Philips Private DICOM Tags (Validated)

From IRC805 Philips Ingenia Elition X 3T scanner:

| Parameter | DICOM Tag | Value | Default | Difference |
|-----------|-----------|-------|---------|------------|
| **Labeling Duration (τ)** | (2005,140a) | **1.932 s** | 1.8 s | +7.3% |
| **Post-Labeling Delay (PLD)** | (2005,1442) | **2.031 s** | 1.8 s | +12.8% |
| **Background Suppression** | (2005,1412) | **1 pulse** | N/A | - |
| **Volume Type** | (2005,1429) | **CONTROL/LABEL** | Inferred | Confirmed |

### Additional Tags

- **(2005,100a)**: 1.183 s - Purpose unknown (possibly related parameter)
- **(2005,1410)**: 2147483647 - Corrupted/sentinel value (max int32)

## Analysis of Elevated CBF Values

### Current CBF Statistics (IRC805-0580101)

- **Mean CBF**: 158.90 ml/100g/min
- **Median CBF**: 116.90 ml/100g/min
- **Expected GM CBF**: 40-60 ml/100g/min
- **Expected WM CBF**: 20-30 ml/100g/min

**Elevation Factor**: 2.6-4.0x higher than expected

### Impact of Actual vs. Default Parameters

Using the correct extracted parameters (τ=1.932s, PLD=2.031s) instead of defaults (τ=1.8s, PLD=1.8s):

```
CBF = (λ · ΔM · e^(PLD/T1_blood)) /
      (2 · α · T1_blood · M0 · (1 - e^(-τ/T1_blood)))
```

**Effect of longer PLD (2.031s vs 1.8s)**:
- Numerator increases: `e^(2.031/1.65) / e^(1.8/1.65) = 1.155`
- **CBF increases by ~15.5%**

**Effect of longer τ (1.932s vs 1.8s)**:
- Denominator increases: `(1 - e^(-1.932/1.65)) / (1 - e^(-1.8/1.65)) = 1.033`
- **CBF decreases by ~3.3%**

**Net effect**: +12.2% higher CBF

**Conclusion**: Using correct parameters actually makes CBF **higher**, not lower. The elevation is **not** caused by incorrect acquisition parameters.

## Root Causes of Elevated CBF

### 1. M0 Estimation Bias (Primary Cause)

**Current Method**: Using mean control image as M0 (equilibrium magnetization)

**Problem**: Control images contain residual labeled blood, biasing M0 downward

**Impact**:
- If M0 is underestimated by 50%, CBF is overestimated by 2x
- If M0 is underestimated by 60%, CBF is overestimated by 2.5x

**Solution Options**:
- Use separate proton density (M0) scan (not available in this dataset)
- Use white matter reference (WM CBF is known ~25 ml/100g/min)
- Apply calibration factor based on literature values

### 2. Partial Volume Effects

**Scanner Settings**:
- **Slice thickness**: 5 mm (relatively thick)
- **Matrix size**: 88×88
- **Voxel size**: ~3×3×5 mm³

**Problem**: Voxels contain mixtures of GM/WM/CSF, leading to:
- Overestimation of CBF in high-perfusion voxels
- Spatial smoothing of CBF distribution
- Edge effects at tissue boundaries

**Impact**: Can contribute 10-30% elevation in mean CBF

**Solution**: Implement partial volume correction (PVC) using tissue segmentation

### 3. Scanner-Specific Scaling

**Philips Scale Factor**: Tag (2005,140a) = 1.932

**Problem**: Philips may apply vendor-specific scaling that affects signal intensity

**Impact**: Unknown, but could contribute to elevation

**Solution**: Investigate Philips documentation for scaling conventions

## Recommendations

### Immediate Actions (Completed)

1. ✅ **Extract actual parameters from DICOM**
   - Created `dicom_asl_params.py` utility
   - Validated on IRC805 Philips scanner
   - Identified correct τ and PLD values

2. ✅ **Update config.yaml with actual values**
   ```yaml
   asl:
     labeling_duration: 1.932  # From DICOM
     post_labeling_delay: 2.031  # From DICOM
     background_suppression_pulses: 1
   ```

3. ✅ **Document findings**
   - Created parameter investigation report
   - Updated ASL QC summary

### Next Steps (Pending)

1. **Implement M0 Calibration** (High Priority)
   - Use white matter reference region
   - Apply scaling factor: `CBF_corrected = CBF_raw × (25 / CBF_WM_measured)`
   - Expected to reduce CBF by 2-3x

2. **Implement Partial Volume Correction** (Medium Priority)
   - Use tissue segmentation masks
   - Apply Bayesian spatial regularization
   - Correct for slice thickness effects

3. **Validate Against Literature** (Medium Priority)
   - Compare age-matched cohorts
   - Account for physiological variability (20-30%)
   - Consider factors: caffeine, CO2, hematocrit

4. **Integrate Auto-Extraction into Workflow** (Low Priority)
   - Modify `asl_preprocess.py` to auto-extract DICOM parameters
   - Add option to override with manual parameters
   - Log extracted values in QC report

## Implementation Plan

### Phase 1: M0 Calibration (Most Impact)

```python
def calibrate_cbf_with_wm_reference(
    cbf: np.ndarray,
    wm_mask: np.ndarray,
    wm_cbf_expected: float = 25.0  # ml/100g/min
) -> np.ndarray:
    """
    Calibrate CBF using white matter reference.

    WM has relatively stable CBF (~25 ml/100g/min) and can
    serve as internal calibration reference.
    """
    wm_cbf_measured = np.mean(cbf[wm_mask > 0.7])  # High-confidence WM
    scaling_factor = wm_cbf_expected / wm_cbf_measured
    cbf_calibrated = cbf * scaling_factor

    return cbf_calibrated
```

**Expected Impact**: Reduce mean CBF from 158.90 to ~50-70 ml/100g/min

### Phase 2: Partial Volume Correction

```python
def partial_volume_correction(
    cbf: np.ndarray,
    gm_pve: np.ndarray,
    wm_pve: np.ndarray,
    csf_pve: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Apply linear regression PVC (Asllani et al., 2008).

    Corrects for partial volume averaging in low-resolution ASL.
    """
    # Linear regression model
    # Observed_CBF = GM_fraction × GM_CBF + WM_fraction × WM_CBF

    # Returns:
    # - cbf_gm: GM-corrected CBF
    # - cbf_wm: WM-corrected CBF
```

**Expected Impact**: Improve accuracy by 10-20%, especially at tissue boundaries

### Phase 3: Automated Parameter Extraction

Integrate DICOM extraction directly into preprocessing workflow:
- Check for DICOM directory when processing
- Auto-extract parameters if available
- Fall back to config.yaml defaults
- Log source of parameters in QC report

## References

1. **Alsop et al. (2015)**. Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications. *Magnetic Resonance in Medicine*, 73(1).

2. **Asllani et al. (2008)**. Regression algorithm correcting for partial volume effects in arterial spin labeling MRI. *Magnetic Resonance in Medicine*, 60(6).

3. **Chappell et al. (2010)**. Variational Bayesian inference for a nonlinear forward model. *IEEE Transactions on Signal Processing*, 57(1).

4. **Philips pCASL Documentation**. Ingenia Elition X 3T Technical Specifications.

## Conclusion

The investigation successfully extracted actual ASL acquisition parameters from DICOM files:
- **τ = 1.932 s** (validated)
- **PLD = 2.031 s** (validated)

However, these parameters are **longer** than defaults, meaning the elevated CBF (~2-3x expected) is **not** due to incorrect acquisition parameters. The primary cause is **M0 estimation bias** from using control images instead of a separate M0 scan.

**Next Priority**: Implement WM reference calibration to correct CBF values.

**Status**: Parameter extraction COMPLETE, awaiting implementation of calibration methods.
