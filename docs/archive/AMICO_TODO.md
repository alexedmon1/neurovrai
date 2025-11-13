# AMICO Integration - TODO Items

**Date:** 2025-11-12

## Current Status

### ✅ Working
- **NODDI (AMICO)**: Successfully implemented and validated
  - Runtime: 30-33 seconds for 205,876 voxels
  - All 4 outputs generated correctly (ficvf, odi, fiso, dir)
  - 100x faster than DIPY implementation

### ⏳ Blocked

#### 1. SANDI (Soma And Neurite Density Imaging)
**Status:** Blocked - requires STEJSKALTANNER scheme format

**Issue:**
- AMICO's SANDI model requires `VERSION: STEJSKALTANNER` scheme format
- Needs full gradient timing parameters: TE, δ (small delta), Δ (big delta)
- Current BIDS JSON has: TE (127 ms), TR (3.18529 s)
- Missing: δ (gradient pulse duration), Δ (pulse separation)

**BIDS JSON Location:**
```
/mnt/bytopia/development/mri-preprocess/rawdata/sub-0580101/dwi/sub-0580101_dwi.json
```

**Next Steps:**
1. Check DICOM headers for gradient timing parameters
2. Contact scanner team for acquisition protocol details
3. Estimate δ and Δ from literature values for Philips Ingenia Elition X
4. Consider using scanner-specific defaults (typically δ=12-30ms, Δ=35-50ms for clinical scanners)

**Reference:**
- SANDI paper: Palombo et al. (2020) NeuroImage 215:116835
- AMICO scheme format: https://github.com/daducci/AMICO/wiki/Scheme-format

#### 2. ActiveAx (Axon Diameter Distribution)
**Status:** Investigation needed

**Issue:**
- Not found in installed AMICO package (dmri-amico 2.1.0)
- User indicated it should be available on AMICO GitHub
- Need to check if:
  - It's in a newer version
  - It's under a different name (e.g., CylinderGPD)
  - It requires separate installation
  - It's been deprecated or replaced

**Next Steps:**
1. Browse AMICO GitHub repository: https://github.com/daducci/AMICO
2. Check for ActiveAx in source code, examples, or documentation
3. Look for related models: CylinderGPD, AxCaliber
4. Check if development branch has newer models
5. Review AMICO publications for model availability

**Alternative Models Available:**
- CylinderGPD: Gamma-distributed cylinder model
- CylinderZeppelinBall: Standard cylinder model
- These may provide similar axon diameter information

## Implementation Notes

### NODDI Output File Naming
AMICO uses lowercase 'fit_' prefix, not uppercase 'FIT_':
```python
metric_map = {
    'fit_NDI.nii.gz': 'ficvf.nii.gz',   # Neurite Density Index
    'fit_ODI.nii.gz': 'odi.nii.gz',     # Orientation Dispersion Index
    'fit_FWF.nii.gz': 'fiso.nii.gz',    # Free Water Fraction
    'fit_dir.nii.gz': 'dir.nii.gz'      # Fiber direction
}
```

### SANDI Parameter Format
AMICO expects arrays, not min/max/step:
```python
# Soma radii: array in meters
Rs_um = np.arange(1.0, 12.0 + 0.5, 0.5)  # μm
Rs_m = Rs_um * 1e-6  # convert to meters

ae.model.set(
    d_is=0.003,    # Intra-soma diffusivity (mm²/s)
    Rs=Rs_m,       # Soma radii array (meters)
    d_in=np.linspace(0.00025, 0.003, 5),  # Intra-neurite diffusivities
    d_isos=np.linspace(0.00025, 0.003, 5)  # Extra-cellular diffusivities
)
```

### pkg_resources Deprecation
AMICO uses deprecated `pkg_resources` API. This is an upstream issue in AMICO's code.
Warning is visible and documented but not suppressed, as it should be fixed in AMICO itself.

## Test Results

### Test Script: `test_amico_only.py`
**Configuration:**
- Subject: IRC805-0580101
- DWI: 112×112×72×220 volumes
- B-values: 0, 1000, 2000, 3000 s/mm²
- Mask: 205,876 voxels

**Results:**
```
✓ NODDI: 33 seconds - SUCCESS
✗ SANDI: STEJSKALTANNER scheme error - BLOCKED
✗ ActiveAx: Not available in AMICO 2.1.0 - INVESTIGATING
```

## Documentation Status

### Complete
- ✅ `AMICO_MODELS_DOCUMENTATION.md` (1000+ lines)
- ✅ `AMICO_INTEGRATION_SUMMARY.md`
- ✅ `mri_preprocess/workflows/amico_models.py` (fully documented)
- ✅ NODDI implementation validated

### Pending
- ⏳ SANDI troubleshooting guide
- ⏳ ActiveAx availability confirmation
- ⏳ Gradient timing parameter extraction methods

## Priority Actions

1. **High Priority:**
   - Investigate ActiveAx on AMICO GitHub
   - Determine gradient timing parameters for SANDI

2. **Medium Priority:**
   - Test alternative models (CylinderGPD) as ActiveAx replacement
   - Create gradient timing extraction utilities

3. **Low Priority:**
   - Compare AMICO NODDI vs DIPY NODDI outputs
   - Generate QC reports for NODDI outputs
   - Add NODDI to main preprocessing workflow

## References

### AMICO
- GitHub: https://github.com/daducci/AMICO
- Paper: Daducci et al. (2015) NeuroImage 105:32-44
- PyPI: https://pypi.org/project/dmri-amico/

### Models
- NODDI: Zhang et al. (2012) NeuroImage 61(4):1000-1016
- SANDI: Palombo et al. (2020) NeuroImage 215:116835
- ActiveAx: Alexander et al. (2010) NeuroImage 52(4):1374-1389

---

**Last Updated:** 2025-11-12
**Status:** NODDI working, SANDI/ActiveAx require investigation
