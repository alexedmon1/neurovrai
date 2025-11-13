# ActiveAx Implementation with AMICO

**Date:** 2025-11-12
**Status:** ✅ IMPLEMENTED (Testing pending)

## Discovery

ActiveAx is implemented in AMICO as **`CylinderZeppelinBall`** model, not as "ActiveAx" directly.

## Model: CylinderZeppelinBall

### Description
The Cylinder-Zeppelin-Ball model is a three-compartment diffusion model designed for white matter microstructure characterization:

1. **Cylinder compartment**: Restricted diffusion within axons (specific radii)
2. **Zeppelin compartment**: Hindered extra-axonal diffusion (tensors)
3. **Ball compartment**: Isotropic diffusion (CSF/free water)

This is equivalent to ActiveAx from Alexander et al. (2010).

### Parameters

```python
ae.set_model("CylinderZeppelinBall")
ae.model.set(
    d_par=1.7e-3,     # Parallel diffusivity (mm²/s)
    Rs=Rs_m,          # Axon radii array (meters)
    d_perps=np.array([1.19e-3, 0.85e-3, 0.51e-3, 0.17e-3]),  # Perpendicular diffusivities
    d_isos=np.array([3.0e-3])  # Isotropic diffusivity (CSF)
)
```

#### Parameter Details

**`d_par` (Parallel Diffusivity)**
- Value: 1.7e-3 mm²/s
- Represents diffusion along axons
- Same for both intra-axonal and extra-axonal compartments
- Literature range: 1.5-2.0e-3 mm²/s

**`Rs` (Axon Radii)**
- Array of radii in **meters** (not μm!)
- Default implementation: 0.1-10 μm diameter → 0.05-5 μm radius
- Sampled with 0.5 μm diameter steps (0.25 μm radius steps)
- Typical human brain: 0.5-5 μm diameter
  - Motor pathways: larger (1-5 μm)
  - Sensory pathways: smaller (0.5-2 μm)

**`d_perps` (Perpendicular Diffusivities)**
- Array: [1.19e-3, 0.85e-3, 0.51e-3, 0.17e-3] mm²/s
- Represents extra-axonal hindered diffusion perpendicular to fibers
- Multiple values capture varying degrees of restriction
- Lower values = more restricted (higher axon density)

**`d_isos` (Isotropic Diffusivities)**
- Array: [3.0e-3] mm²/s
- Represents CSF and free water
- Standard CSF diffusivity at body temperature

### Expected Outputs

Based on AMICO's naming conventions for other models, CylinderZeppelinBall should produce:

```
activeax/
├── ficvf.nii.gz      # Intra-axonal volume fraction (axon density) [0-1]
├── diam.nii.gz       # Mean axon diameter [μm]
├── dir.nii.gz        # Principal fiber direction (3D vector)
└── fvf_tot.nii.gz    # Total fiber volume fraction [0-1]
```

**Note:** Actual output file names will be discovered during testing. AMICO may use:
- `FIT_v_*.nii.gz` for volume fractions
- `FIT_d.nii.gz` for diameter
- `FIT_dir.nii.gz` for direction

### Important Notes

**Gradient Requirements:**
- Like ActiveAx, CylinderZeppelinBall requires **STEJSKALTANNER** scheme
- Needs full gradient timing specification: TE, δ (small delta), Δ (big delta)
- **This is the same issue as SANDI**
- Standard FSL bval/bvec format may not be sufficient
- May need to use `amico.util.fsl2scheme()` with additional parameters

**Gradient Strength Limitations:**
- Reliable axon diameter estimation requires strong gradients (>300 mT/m)
- Standard clinical scanners (~80 mT/m) provide limited diameter sensitivity
- However, FICVF (axon density) should still be reliable
- Consider diameter estimates with caution on clinical data

## Implementation

### Location
`mri_preprocess/workflows/amico_models.py:425-594`

### Function
`fit_activeax_amico()` - Now uses CylinderZeppelinBall model

### Key Changes
1. Changed model name: `"ActiveAx"` → `"CylinderZeppelinBall"`
2. Updated parameters to match AMICO API (arrays, not min/max/step)
3. Convert diameter (μm) to radius (meters) for Rs parameter
4. Updated output directory path

### Code Example

```python
from pathlib import Path
from mri_preprocess.workflows.advanced_diffusion import run_advanced_diffusion_models

results = run_advanced_diffusion_models(
    dwi_file=Path('dwi_eddy_corrected.nii.gz'),
    bval_file=Path('dwi.bval'),
    bvec_file=Path('dwi_rotated.bvec'),
    mask_file=Path('dwi_mask.nii.gz'),
    output_dir=Path('advanced_models'),
    fit_dki=False,
    fit_noddi=True,
    fit_sandi=True,
    fit_activeax=True,    # Now available!
    use_amico=True
)

# Access ActiveAx outputs
if 'activeax' in results:
    print(f"Axon density: {results['activeax']['ficvf']}")
    print(f"Mean diameter: {results['activeax']['diam']}")
```

## Testing

### Test Script
`test_amico_only.py` - Updated to enable ActiveAx

### Expected Behavior
1. **If scheme conversion succeeds:**
   - CylinderZeppelinBall should fit successfully
   - Runtime: ~3-6 minutes for 205k voxels
   - Outputs generated in `advanced_models_amico/activeax/`

2. **If scheme error occurs:**
   - Same STEJSKALTANNER error as SANDI
   - Would need to add gradient timing to scheme file

### Test Command
```bash
source .venv/bin/activate
python test_amico_only.py 2>&1 | tee activeax_test.log
```

## Potential Issues

### Issue 1: STEJSKALTANNER Scheme Requirement
**Symptom:** `[ ERROR ] This model requires a "VERSION: STEJSKALTANNER" scheme`

**Cause:**
- CylinderZeppelinBall (like SANDI) requires full gradient specification
- FSL bval/bvec doesn't include δ and Δ timing

**Solution Options:**
1. Extract from DICOM headers or BIDS JSON (if available)
2. Estimate from scanner protocol (TE, gradient strength)
3. Contact scanner operator for protocol parameters
4. Use typical values for scanner model

**IRC805 Data:**
- TE = 127 ms (from BIDS JSON)
- Missing: δ (small delta), Δ (big delta)
- Scanner: Philips Ingenia Elition X

### Issue 2: Output File Naming
**Symptom:** Files generated but not found at expected locations

**Cause:**
- AMICO's actual output names may differ from expected
- Already happened with NODDI (fit_NDI vs FIT_ICVF)

**Solution:**
- After first test run, check actual output directory
- Update `metric_map` in `amico_models.py:574-575`

## Comparison: AMICO vs Literature

| Aspect | AMICO CylinderZeppelinBall | Original ActiveAx |
|--------|---------------------------|-------------------|
| **Implementation** | Convex optimization | Non-linear fitting |
| **Speed** | ~3-6 min | 30-60+ min |
| **Model** | Same 3-compartment | Same 3-compartment |
| **Outputs** | FICVF, diameter, direction | Same |
| **Accuracy** | Equivalent | Baseline |
| **Diameter range** | Configurable (0.1-10 μm) | Typically 0.1-20 μm |

## References

### Primary Papers
1. **Panagiotaki et al. (2012)**
   "Compartment models of the diffusion MR signal in brain white matter: A taxonomy and comparison"
   *NeuroImage* 59:2241-2254
   - Original description of Cylinder-Zeppelin-Ball model

2. **Alexander et al. (2010)**
   "Orientationally invariant indices of axon diameter and density from diffusion MRI"
   *NeuroImage* 52(4):1374-1389
   - Original ActiveAx paper

3. **Daducci et al. (2015)**
   "Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data"
   *NeuroImage* 105:32-44
   - AMICO framework

### AMICO Documentation
- GitHub: https://github.com/daducci/AMICO
- Help: `help(amico.models.CylinderZeppelinBall)`

## Next Steps

1. **Test on IRC805 data**
   - Run `test_amico_only.py`
   - Check if STEJSKALTANNER error occurs
   - Verify output files are generated

2. **If STEJSKALTANNER error:**
   - Same resolution path as SANDI
   - Extract/estimate gradient timing parameters
   - Create proper scheme file

3. **If successful:**
   - Validate output file naming
   - Check physiological plausibility of results
   - Compare with literature values
   - Document QC procedures

4. **Quality Control:**
   - Check FICVF range [0-1]
   - Verify diameter values are physiological (0.5-5 μm typical)
   - Inspect fiber direction maps
   - Compare with DTI metrics

---

**Status:** Implementation complete, awaiting testing
**Blocked by:** Same STEJSKALTANNER scheme issue as SANDI
**Workaround:** May work if `fsl2scheme()` is sufficient
