# Future Enhancements

This document tracks planned improvements to make additional parameters configurable via `config.yaml`.

## Medium Priority: Advanced Model Parameters

### 1. Tractography Parameters

**Status**: Partially implemented (hardcoded defaults)
**Files**: `mri_preprocess/workflows/tractography.py`
**Priority**: Medium (useful for advanced users)

**Current State**: All parameters are hardcoded as function defaults (lines 56-59):
```python
def run_atlas_based_tractography(
    ...
    n_samples: int = 5000,
    n_steps: int = 2000,
    step_length: float = 0.5,
    curvature_threshold: float = 0.2,
    ...
):
```

**Proposed Config Section**:
```yaml
diffusion:
  tractography:
    n_samples: 5000          # Number of streamlines to generate
    n_steps: 2000            # Maximum steps per streamline
    step_length: 0.5         # Step size in mm
    curvature_threshold: 0.2 # Maximum curvature (radians)
```

**Implementation Notes**:
- These parameters affect tractography quality vs computation time
- Typical values:
  - n_samples: 1000-10000 (higher = more accurate connectivity, slower)
  - n_steps: 1000-5000 (higher = longer streamlines allowed)
  - step_length: 0.2-1.0 mm (smaller = smoother but slower)
  - curvature_threshold: 0.1-0.3 (lower = stricter anatomical constraints)

---

### 2. AMICO Model Parameters

**Status**: Not implemented (hardcoded in model functions)
**Files**: `mri_preprocess/workflows/amico_models.py`
**Priority**: Low-Medium (expert users only)

#### 2a. NODDI Parameters

**Current State**: Hardcoded (lines 105-106):
```python
parallel_diffusivity: float = 1.7e-3  # mm²/s
isotropic_diffusivity: float = 3.0e-3  # mm²/s
```

**Proposed Config Section**:
```yaml
diffusion:
  advanced_models:
    noddi:
      parallel_diffusivity: 1.7e-3   # Intra-axonal diffusivity (mm²/s)
      isotropic_diffusivity: 3.0e-3  # CSF diffusivity (mm²/s)
```

**Notes**:
- Biophysical model parameters based on literature
- Rarely changed unless studying specific pathology
- Values from Jelescu et al., 2016

---

#### 2b. SANDI Parameters

**Current State**: Multiple hardcoded values (lines 258, 424, 430, 432-433):
```python
soma_radius_range: Tuple[float, float] = (1.0, 12.0)  # μm
Rs_um = np.arange(1.0, 12.5, 0.5)  # 0.5 μm steps
d_is = 0.003  # Intra-soma diffusivity
d_in = np.linspace(0.00025, 0.003, 5)  # Intra-neurite diffusivities
d_isos = np.linspace(0.00025, 0.003, 5)  # Extra-cellular diffusivities
```

**Proposed Config Section**:
```yaml
diffusion:
  advanced_models:
    sandi:
      soma_radius_min: 1.0          # Minimum soma radius (μm)
      soma_radius_max: 12.0         # Maximum soma radius (μm)
      soma_radius_step: 0.5         # Radius discretization (μm)
      intra_soma_diffusivity: 0.003 # mm²/s
      intra_neurite_diffusivity:    # Range for optimization
        min: 0.00025
        max: 0.003
        n_values: 5
      extracellular_diffusivity:    # Range for optimization
        min: 0.00025
        max: 0.003
        n_values: 5
```

**Notes**:
- SANDI estimates soma and neurite compartments separately
- Soma radius range affects model sensitivity to cell body size
- Diffusivity ranges define search space for optimization
- Values from Palombo et al., 2020

---

#### 2c. ActiveAx Parameters

**Current State**: Multiple hardcoded values (lines 496, 668, 678, 680-681):
```python
axon_diameter_range: Tuple[float, float] = (0.1, 10.0)  # μm
diam_um = np.arange(0.1, 10.5, 0.5)  # 0.5 μm steps
d_par = 1.7e-3  # Parallel diffusivity
d_perps = np.array([1.19e-3, 0.85e-3, 0.51e-3, 0.17e-3])  # Perpendicular
d_isos = np.array([3.0e-3])  # Isotropic
```

**Proposed Config Section**:
```yaml
diffusion:
  advanced_models:
    activeax:
      axon_diameter_min: 0.1           # Minimum diameter (μm)
      axon_diameter_max: 10.0          # Maximum diameter (μm)
      axon_diameter_step: 0.5          # Diameter discretization (μm)
      parallel_diffusivity: 1.7e-3     # Intra-axonal (mm²/s)
      perpendicular_diffusivities:     # Hindered extra-axonal
        - 1.19e-3
        - 0.85e-3
        - 0.51e-3
        - 0.17e-3
      isotropic_diffusivity: 3.0e-3    # CSF (mm²/s)
```

**Notes**:
- ActiveAx estimates axon diameter distribution
- Diameter range should match expected axon sizes in ROI
- Perpendicular diffusivities model hindered extra-axonal diffusion
- Values from Alexander et al., 2010

---

## Implementation Strategy

### Phase 1: Tractography (Recommended Next)
1. Add `tractography` section to `config.yaml` template
2. Update `create_config.py` to include tractography defaults
3. Modify `tractography.py` to extract parameters from config
4. Test with different parameter values

**Estimated Time**: 30-60 minutes
**Impact**: High (tractography parameters commonly tuned by users)

---

### Phase 2: AMICO Models (Optional)
1. Add `noddi`, `sandi`, `activeax` subsections under `advanced_models`
2. Update `create_config.py` with detailed comments
3. Modify each AMICO function to extract parameters from config
4. Add parameter validation (e.g., diffusivity > 0)

**Estimated Time**: 2-3 hours
**Impact**: Low (expert users only, rarely need to change)

**Considerations**:
- AMICO parameters are biophysical model constants
- Changing them requires deep understanding of microstructure models
- Most users should use defaults from literature
- Only implement if users request customization

---

## Current Status

✅ **Option 1 Complete** (High Priority):
- BET fractional intensity for all modalities
- N4 bias correction parameters
- Atropos segmentation parameters

🔄 **Option 2 Pending** (Medium Priority):
- Tractography parameters (recommended next)
- AMICO model parameters (optional)

---

## References

### Tractography
- Basser PJ, et al. (2000). "In vivo fiber tractography using DT-MRI data." Magnetic Resonance in Medicine.
- Tournier JD, et al. (2012). "MRtrix: Diffusion tractography in crossing fiber regions." International Journal of Imaging Systems and Technology.

### NODDI
- Zhang H, et al. (2012). "NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain." NeuroImage.
- Jelescu IO, et al. (2016). "Design and validation of a clinical protocol for quantitative NODDI." NeuroImage.

### SANDI
- Palombo M, et al. (2020). "SANDI: A compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI." NeuroImage.

### ActiveAx
- Alexander DC, et al. (2010). "Orientationally invariant indices of axon diameter and density from diffusion MRI." NeuroImage.
- Assaf Y, et al. (2008). "AxCaliber: A method for measuring axon diameter distribution from diffusion MRI." Magnetic Resonance in Medicine.
