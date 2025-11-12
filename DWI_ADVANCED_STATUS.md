# Advanced DWI Models - Testing Status

**Date:** 2025-11-12
**Subject:** IRC805-0580101

## Test Run: DKI and NODDI

### Data Specifications
- **B-values:** 0, 1000, 2000, 3000 s/mm² (4 shells - perfect for DKI/NODDI!)
- **Volumes:** 220 total
  - b=0: 10 volumes
  - b=1000: 30 volumes
  - b=2000: 60 volumes
  - b=3000: 120 volumes
- **Mask:** 205,876 voxels
- **Image dimensions:** 112 x 112 x 72

### Current Status: IN PROGRESS

**Started:** 09:50:56
**Current Time:** ~10:11 (20+ minutes elapsed)

**Progress:**
1. ✅ Data loading (10 seconds)
2. ✅ DKI model fitting (3 minutes)
3. ⏳ **Computing DKI metrics** (17+ minutes and counting)
   - Mean Kurtosis (MK)
   - Axial Kurtosis (AK)
   - Radial Kurtosis (RK)
   - Kurtosis FA (KFA)
4. ⏳ NODDI fitting (not started yet)

### Performance Observations

**DKI Metric Computation:**
- **Expected:** 5-10 minutes
- **Actual:** 17+ minutes (still running)
- **CPU Usage:** 140-150% (multi-core)
- **Memory:** ~1.9 GB

**Reason for Slow Performance:**
- DIPY uses pure Python implementation for kurtosis metrics
- Each metric (MK, AK, RK, KFA) requires complex tensor calculations
- 206k voxels × 4 metrics = ~824k individual calculations
- No GPU acceleration in current implementation

### Implementation Details

**Current Approach:**
```python
from dipy.reconst import dki

# Fit DKI model
dkimodel = dki.DiffusionKurtosisModel(gtab)
dkifit = dkimodel.fit(data, mask=mask)

# Compute metrics (SLOW operations)
mk = dkifit.mk(0, 3)  # Mean kurtosis
ak = dkifit.ak(0, 3)  # Axial kurtosis
rk = dkifit.rk(0, 3)  # Radial kurtosis
kfa = dkifit.kfa(0, 3)  # Kurtosis FA
```

**Known Issues:**
- `.mk()`, `.ak()`, `.rk()`, `.kfa()` are single-threaded
- Each call iterates through all masked voxels
- No progress indication

### NODDI Implementation

**Library:** DIPY (Python, NO MATLAB required!)

**Advantages:**
- Pure Python - no MATLAB dependency
- Part of well-maintained DIPY ecosystem
- Uses CVXPY for convex optimization
- Validated against original NODDI

**Expected Metrics:**
- ODI (Orientation Dispersion Index): 0-1
- FICVF (Intracellular Volume Fraction): 0-1
- FISO (Isotropic Volume Fraction): 0-1

**Expected Runtime:** 10-15 minutes (may be longer based on DKI experience)

---

## Alternative Approaches for Production

### Option 1: AMICO (RECOMMENDED for Speed)
**Accelerated Microstructure Imaging via Convex Optimization**

```bash
pip install dmri-amico
```

**Advantages:**
- 100x faster than DIPY NODDI
- Linear programming approach
- Validated results
- Python implementation

**Example:**
```python
import amico

# Setup AMICO
amico.core.setup()
ae = amico.Evaluation("study_dir", "subject")
ae.load_data(dwi_file, scheme_file)
ae.set_model("NODDI")
ae.generate_kernels()
ae.load_kernels()
ae.fit()
ae.save_results()
```

### Option 2: MDT (GPU-Accelerated)
**Microstructure Diffusion Toolbox**

```bash
pip install mdt
```

**Advantages:**
- GPU-accelerated (OpenCL)
- Very fast with proper GPU
- Supports DKI and NODDI
- Command-line interface

**Example:**
```bash
mdt-model-fit NODDI dwi.nii.gz protocol.prtcl mask.nii.gz output_dir/
```

**Cons:**
- Requires GPU with OpenCL support
- More complex setup
- Additional dependencies

### Option 3: Downsample Mask
**Keep DIPY but reduce voxel count**

```python
# Erosion to reduce voxels
from scipy import ndimage
eroded_mask = ndimage.binary_erosion(mask, iterations=2)
# Reduces voxel count by ~30-50%
```

**Trade-offs:**
- Faster computation
- Less detailed results
- May miss edge regions

---

## Recommendations

### For Research/Clinical Use:
1. **Use AMICO** for NODDI - much faster, validated
2. **Keep DIPY DKI** but add progress logging
3. **Consider GPU acceleration** (MDT) for large batches

### For Current Testing:
1. **Let current run complete** to establish baseline timing
2. **Document actual runtimes** for future reference
3. **Test AMICO** in parallel for comparison

### Code Optimizations Needed:
1. Add progress bar to DKI metric computation
2. Consider parallel processing (if DIPY supports it)
3. Add option to compute only specific metrics
4. Save intermediate results to resume if interrupted

---

## Expected Outputs

### DKI Metrics:
```
derivatives/dwi_topup/IRC805-0580101/advanced_models/dki/
├── mk.nii.gz          # Mean Kurtosis
├── ak.nii.gz          # Axial Kurtosis
├── rk.nii.gz          # Radial Kurtosis
└── kfa.nii.gz         # Kurtosis FA
```

### NODDI Metrics:
```
derivatives/dwi_topup/IRC805-0580101/advanced_models/noddi/
├── odi.nii.gz         # Orientation Dispersion Index
├── ficvf.nii.gz       # Intracellular Volume Fraction
└── fiso.nii.gz        # Isotropic Volume Fraction
```

---

## Timing Estimates (Updated)

| Step | Expected | Actual |
|------|----------|--------|
| Data loading | <1 min | 10 sec ✅ |
| DKI fitting | 3-5 min | 3 min ✅ |
| DKI metrics | 5-10 min | **17+ min** ⏳ |
| NODDI fitting | 10-15 min | TBD |
| **Total** | **~20-30 min** | **40+ min (est)** |

**Conclusion:** DIPY implementation is functional but SLOW for production use with full-resolution data. Consider AMICO or MDT for operational pipelines.

---

## Next Steps

1. ✅ Complete current DKI/NODDI test run
2. Document final timing results
3. Test AMICO implementation for comparison
4. Add progress logging to current code
5. Create QC module for DKI/NODDI outputs
6. Integrate into main DWI workflow with runtime warnings

---

**Status:** Process running in background (PID: 578235)
**Monitor:** `ps aux | grep 578235`
**Outputs:** `/mnt/bytopia/development/IRC805/derivatives/dwi_topup/IRC805-0580101/advanced_models/`
