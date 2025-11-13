# AMICO Microstructure Models - Complete Output Documentation

**Date:** 2025-11-12
**Version:** 1.0

## Overview

This document describes all output metrics from the AMICO (Accelerated Microstructure Imaging via Convex Optimization) implementation of advanced diffusion MRI models. AMICO provides 100-1000x faster computation compared to traditional optimization approaches while maintaining equivalent accuracy.

## Table of Contents

1. [NODDI (Neurite Orientation Dispersion and Density)](#noddi)
2. [SANDI (Soma And Neurite Density Imaging)](#sandi)
3. [ActiveAx (Axon Diameter Distribution)](#activeax)
4. [DKI (Diffusion Kurtosis Imaging)](#dki)
5. [Interpretation Guidelines](#interpretation-guidelines)
6. [Data Requirements](#data-requirements)
7. [Quality Control](#quality-control)

---

## NODDI

**Neurite Orientation Dispersion and Density Imaging**

**Purpose:** Model neurite microstructure (axons + dendrites) using a three-compartment tissue model.

### Model Compartments

1. **Intracellular (restricted):** Inside neurites, modeled as zero-radius sticks
2. **Extracellular (hindered):** Around neurites, anisotropic Gaussian diffusion
3. **CSF (free water):** Isotropic fast diffusion

### Output Files

All outputs are located in: `derivatives/dwi_topup/{subject}/advanced_models/noddi/`

#### `ficvf.nii.gz` - Intracellular Volume Fraction (Neurite Density)

**Full Name:** Fiber Intracellular Volume Fraction
**Also called:** Neurite Density Index (NDI)

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- **High values (0.6-0.9):** Dense neurite packing (white matter tracts, cortex)
- **Medium values (0.3-0.6):** Moderate neurite density (gray matter)
- **Low values (0.0-0.3):** Sparse neurites (CSF, damaged tissue)

**Biological Meaning:**
- Proportion of tissue volume occupied by neurites (axons + dendrites)
- Reflects axonal/dendritic density
- Sensitive to neurodegeneration, development, plasticity

**Typical Values:**
- Corpus callosum: 0.7-0.9
- Cortical gray matter: 0.4-0.6
- Deep gray matter: 0.3-0.5
- CSF/ventricles: <0.1

**Clinical Relevance:**
- ↓ in neurodegeneration (Alzheimer's, MS, ALS)
- ↑ during brain development
- ↓ after stroke or traumatic injury
- Marker of white matter integrity

---

#### `odi.nii.gz` - Orientation Dispersion Index

**Full Name:** Orientation Dispersion Index

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- **0.0:** Perfectly aligned fibers (coherent bundle)
- **0.5:** Moderate dispersion (crossing fibers, fanning)
- **1.0:** Maximally dispersed (isotropic orientation)

**Biological Meaning:**
- Angular variability of neurite orientations
- Measures fiber organization and coherence
- Independent of neurite density (ODI vs FICVF)

**Typical Values:**
- Major WM tracts (corpus callosum): 0.1-0.3 (low dispersion)
- Cortical gray matter: 0.6-0.9 (high dispersion)
- Subcortical nuclei: 0.7-0.9
- Crossing fiber regions: 0.4-0.7

**Clinical Relevance:**
- ↑ in cortical development (dendritic branching)
- ↑ in white matter regions with crossing fibers
- Changes in plasticity and reorganization
- Complements FA (more specific to orientation complexity)

**Relationship to FA:**
- FA is affected by both dispersion AND density
- ODI isolates dispersion from density
- Low ODI + High FICVF = Coherent dense tract (corpus callosum)
- High ODI + High FICVF = Dense but dispersed (cortex)

---

#### `fiso.nii.gz` - Isotropic Volume Fraction (Free Water)

**Full Name:** CSF/Free Water Volume Fraction
**Also called:** FISO, ISOVF

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- **High values (0.8-1.0):** CSF-dominated (ventricles, sulci)
- **Medium values (0.3-0.8):** Partial volume with CSF
- **Low values (0.0-0.3):** Tissue-dominated (white/gray matter)

**Biological Meaning:**
- Proportion of tissue with fast, unrestricted isotropic diffusion
- Primarily CSF in normal tissue
- Can reflect edema, inflammation, or atrophy

**Typical Values:**
- Lateral ventricles: 0.9-1.0
- Deep white matter: 0.0-0.1
- Cortical gray matter: 0.1-0.2
- Periventricular regions: 0.2-0.4

**Clinical Relevance:**
- ↑ in edema (vasogenic, cytotoxic)
- ↑ in inflammation (MS lesions)
- ↑ with atrophy (compensatory CSF)
- ↓ in tumors (restricted diffusion)

**Quality Control:**
- Should be near 1.0 in ventricles
- Should be near 0.0 in major white matter tracts
- If FISO is high everywhere, check data quality or model assumptions

---

#### `dir.nii.gz` - Principal Fiber Direction

**Dimensions:** X × Y × Z × 3 (3D vector field)

**Interpretation:**
- 3D unit vector indicating principal neurite orientation
- Similar to primary eigenvector from DTI
- Components: [x, y, z] direction cosines

**Visualization:**
- RGB color-coded maps (standard convention):
  - **Red:** Left-Right (x-direction)
  - **Green:** Anterior-Posterior (y-direction)
  - **Blue:** Superior-Inferior (z-direction)

**Use Cases:**
- Tractography seed directions
- White matter orientation analysis
- Comparison with DTI primary eigenvector
- Registration quality control

---

## SANDI

**Soma And Neurite Density Imaging**

**Purpose:** Separate neuronal cell body (soma) contributions from neurites, particularly useful for gray matter analysis.

### Model Compartments

1. **Soma (restricted sphere):** Cell bodies with variable radius
2. **Neurite (stick):** Axons and dendrites (thin cylinders)
3. **Extra-cellular (hindered):** Extracellular space
4. **CSF (free water):** Fast isotropic diffusion

### Output Files

All outputs are located in: `derivatives/dwi_topup/{subject}/advanced_models/sandi/`

#### `fsoma.nii.gz` - Soma Volume Fraction

**Full Name:** Soma Volume Fraction

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- **High values (0.3-0.6):** Soma-rich regions (cortical layers, nuclei)
- **Medium values (0.1-0.3):** Mixed soma/neurite
- **Low values (0.0-0.1):** Soma-sparse (white matter)

**Biological Meaning:**
- Proportion of tissue occupied by neuronal cell bodies
- Reflects neuron density and soma size
- Primarily non-zero in gray matter

**Typical Values:**
- Cortical gray matter: 0.2-0.5
- Subcortical nuclei (caudate, putamen): 0.3-0.6
- Hippocampus: 0.2-0.4
- White matter: 0.0-0.1 (very low)

**Clinical Relevance:**
- ↓ in neuronal loss (Alzheimer's, neurodegeneration)
- ↑ in development (increased neuron packing)
- Changes in cortical pathology
- Marker of gray matter integrity

**Gray Matter Specificity:**
- SANDI is particularly valuable for gray matter
- Separates soma from dendrites (both contribute to NODDI FICVF)
- More specific neuronal marker than NODDI alone

---

#### `fneurite.nii.gz` - Neurite Volume Fraction

**Full Name:** Neurite Volume Fraction (SANDI-specific)

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- Similar to NODDI FICVF but excluding soma contribution
- **High values (0.5-0.8):** Dense neurites (white matter)
- **Medium values (0.2-0.5):** Moderate neurites (gray matter)
- **Low values (0.0-0.2):** Sparse neurites

**Biological Meaning:**
- Proportion of tissue occupied by neurites (excluding somas)
- Axon and dendrite density
- Complements FSOMA for complete tissue characterization

**Typical Values:**
- White matter tracts: 0.6-0.8
- Cortical gray matter: 0.3-0.5
- Subcortical gray matter: 0.2-0.4

**Relationship to NODDI:**
- **NODDI FICVF** = SANDI FSOMA + SANDI FNEURITE
- SANDI separates what NODDI combines
- More detailed compartmentalization

---

#### `fec.nii.gz` - Extra-cellular Volume Fraction

**Full Name:** Extra-cellular Volume Fraction

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- **High values (0.4-0.7):** Large extracellular space
- **Medium values (0.2-0.4):** Moderate extracellular space
- **Low values (0.0-0.2):** Dense cellular packing

**Biological Meaning:**
- Proportion of tissue that is extracellular space
- Reflects tissue porosity and cellular density
- Includes interstitial fluid and ECM

**Typical Values:**
- White matter: 0.1-0.3
- Gray matter: 0.2-0.4
- Pathological tissue: 0.3-0.7 (edema, inflammation)

**Clinical Relevance:**
- ↑ in edema (interstitial fluid accumulation)
- ↑ in inflammation
- ↑ with cellular loss (atrophy)
- Changes in blood-brain barrier dysfunction

**Volume Conservation:**
- FSOMA + FNEURITE + FEC + FCSF ≈ 1.0
- Useful quality check

---

#### `fcsf.nii.gz` - CSF Volume Fraction

**Full Name:** CSF/Free Water Volume Fraction

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- Same as NODDI FISO
- **High values (0.8-1.0):** CSF-dominated
- **Low values (0.0-0.2):** Tissue-dominated

**See NODDI FISO documentation above for detailed interpretation.**

---

#### `rsoma.nii.gz` - Soma Radius

**Full Name:** Mean Soma Radius

**Range:** 1.0 - 12.0 μm (micrometers)
**Default search range:** 1-12 μm

**Interpretation:**
- **Large somas (8-12 μm):** Motor neurons, pyramidal neurons
- **Medium somas (5-8 μm):** Typical cortical neurons
- **Small somas (2-5 μm):** Granule cells, small interneurons

**Biological Meaning:**
- Average radius of neuronal cell bodies in each voxel
- Reflects neuron type and size distribution
- Region-specific neuron populations

**Typical Values:**
- Motor cortex: 8-12 μm (large pyramidal neurons)
- Sensory cortex: 6-9 μm
- Cerebellum granule layer: 3-5 μm (small granule cells)
- Hippocampus: 7-10 μm

**Clinical Relevance:**
- Changes in neurodevelopment (soma growth)
- ↓ in atrophy or neurodegeneration
- Disease-specific alterations
- Neuron type vulnerability

**Important Notes:**
- Only meaningful where FSOMA > 0.1 (gray matter)
- Requires high b-values (≥3000 s/mm²) for sensitivity
- Voxel averages over multiple neuron types
- May be biased by SNR and model assumptions

---

#### `dir.nii.gz` - Principal Neurite Direction

**Same as NODDI DIR - see NODDI documentation above.**

---

## ActiveAx

**Axon Diameter Distribution Modeling**

**Purpose:** Estimate axon diameter distribution in white matter by separating intra-axonal (restricted) and extra-axonal (hindered) diffusion.

### Model Compartments

1. **Intra-axonal (restricted cylinder):** Inside axons with varying diameter
2. **Extra-axonal (hindered):** Extracellular space around axons

### Output Files

All outputs are located in: `derivatives/dwi_topup/{subject}/advanced_models/activeax/`

#### `ficvf.nii.gz` - Intra-axonal Volume Fraction

**Full Name:** Intra-axonal Volume Fraction
**Also called:** Axon Density

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- **High values (0.6-0.9):** Dense axon packing
- **Medium values (0.3-0.6):** Moderate axon density
- **Low values (0.0-0.3):** Sparse axons

**Biological Meaning:**
- Proportion of white matter volume occupied by axons
- Reflects axonal density (not neurite density like NODDI)
- White matter specific (not applicable to gray matter)

**Typical Values:**
- Corpus callosum (genu): 0.7-0.9
- Corpus callosum (splenium): 0.6-0.8
- Internal capsule: 0.6-0.8
- Corona radiata: 0.5-0.7

**Clinical Relevance:**
- ↓ in demyelinating diseases (MS)
- ↓ in axonal degeneration
- ↓ in traumatic brain injury
- Marker of white matter integrity

**Difference from NODDI FICVF:**
- ActiveAx: Axons only (white matter specific)
- NODDI: All neurites (axons + dendrites, WM + GM)
- ActiveAx may give slightly different values in WM

---

#### `diam.nii.gz` - Mean Axon Diameter

**Full Name:** Mean Axon Diameter

**Range:** 0.1 - 10.0 μm (micrometers)
**Default search range:** 0.1-10 μm
**Physiological range in human brain:** 0.2-5 μm

**Interpretation:**
- **Large axons (2-5 μm):** Motor pathways, fast conduction
- **Medium axons (1-2 μm):** Association fibers
- **Small axons (0.5-1 μm):** Sensory pathways, slow conduction

**Biological Meaning:**
- Average diameter of axons in each voxel
- Related to conduction velocity (larger = faster)
- Reflects fiber type composition

**Typical Values:**
- Corpus callosum (motor fibers): 1.5-3.0 μm
- Corticospinal tract: 2.0-4.0 μm (large motor axons)
- Optic radiation: 0.8-1.5 μm
- Association fibers: 1.0-2.0 μm

**Clinical Relevance:**
- Changes in development (axon growth)
- Selective vulnerability (large axons in ALS)
- Tract-specific pathology
- Functional implications (conduction speed)

**CRITICAL TECHNICAL NOTE:**

⚠️ **Axon diameter estimation is extremely challenging and requires:**

1. **Very strong diffusion gradients:** >300 mT/m (clinical scanners typically 40-80 mT/m)
2. **Specialized hardware:** Human Connectom scanner, high-performance gradients
3. **Optimized acquisition:** Long diffusion times, high b-values, many directions

**On standard clinical scanners:**
- Diameter estimates may be **unreliable**
- Systematic underestimation common
- High sensitivity to noise and modeling assumptions
- FICVF estimates remain robust

**Recommendations:**
- Use diameter maps **cautiously** on standard scanners
- Focus on FICVF for clinical applications
- Interpret diameter as **relative** rather than absolute values
- Validate against histology when possible

**Quality Checks:**
- Values consistently <0.5 μm or >5 μm are suspect
- Check against known anatomy (motor tracts should be larger)
- Compare with literature values for your scanner

---

#### `dir.nii.gz` - Principal Fiber Direction

**Same as NODDI DIR - see NODDI documentation above.**

---

#### `fvf_tot.nii.gz` - Total Fiber Volume Fraction

**Full Name:** Total Fiber Volume Fraction

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- Sum of FICVF across all axon diameter bins
- Should be very similar to FICVF in practice
- Represents total axonal occupancy

**Biological Meaning:**
- Total white matter volume occupied by fibers
- Aggregate measure across diameter distribution

**Use Case:**
- Quality control (should match FICVF closely)
- Model fitting validation

---

## DKI

**Diffusion Kurtosis Imaging**

**Implementation:** DIPY (not AMICO - AMICO does not support DKI)

**Purpose:** Measure non-Gaussian diffusion to characterize tissue microstructure complexity.

### Output Files

All outputs are located in: `derivatives/dwi_topup/{subject}/advanced_models/dki/`

#### `mk.nii.gz` - Mean Kurtosis

**Full Name:** Mean Kurtosis

**Range:** 0.0 - 3.0 (unitless, typical range)
**Higher values possible but rare in healthy tissue**

**Interpretation:**
- **High values (1.5-3.0):** Complex microstructure, high restriction
- **Medium values (0.8-1.5):** Moderate complexity
- **Low values (0.0-0.8):** Simple microstructure, less restriction

**Biological Meaning:**
- Deviation from Gaussian diffusion (DTI assumption)
- Reflects microstructural complexity
- Sensitive to tissue heterogeneity, barriers, compartments

**Typical Values:**
- Deep gray matter (caudate, putamen): 1.0-1.5
- White matter tracts: 0.8-1.2
- Cortical gray matter: 0.8-1.2
- CSF: ~0 (Gaussian diffusion)

**Clinical Relevance:**
- ↑ in brain development (increasing complexity)
- ↓ in aging (microstructure loss)
- ↑ in tumors (cellular density)
- ↓ in neurodegenerative disease (structural breakdown)
- Sensitive to stroke (acute ↑, chronic ↓)

**Advantages over DTI:**
- Captures non-Gaussian effects (barriers, compartments)
- More sensitive to gray matter microstructure
- Detects changes FA might miss

---

#### `ak.nii.gz` - Axial Kurtosis

**Full Name:** Axial Kurtosis
**Also called:** Parallel Kurtosis

**Range:** 0.0 - 3.0 (unitless, typical range)

**Interpretation:**
- Kurtosis **along** the principal diffusion direction
- **High values:** Strong restriction parallel to fibers
- **Low values:** Less restriction along fibers

**Biological Meaning:**
- Non-Gaussian diffusion along primary fiber orientation
- Reflects microstructure parallel to tract direction
- Sensitive to intra-axonal restrictions, beading, varicosities

**Typical Values:**
- Major WM tracts (corpus callosum): 0.6-1.2
- Crossing fiber regions: 0.8-1.5
- Gray matter: 0.7-1.2

**Clinical Relevance:**
- ↓ in axonal degeneration (less intra-axonal barriers)
- ↑ in axonal beading/swelling (injury)
- Changes in demyelination
- Complements axial diffusivity (AD) from DTI

**Relationship to AD:**
- AD measures magnitude of diffusion along axis
- AK measures how non-Gaussian that diffusion is
- Both provide info about axonal integrity

---

#### `rk.nii.gz` - Radial Kurtosis

**Full Name:** Radial Kurtosis
**Also called:** Perpendicular Kurtosis

**Range:** 0.0 - 3.0 (unitless, typical range)

**Interpretation:**
- Kurtosis **perpendicular** to principal diffusion direction
- **High values:** Strong restriction across fibers
- **Low values:** Less restriction perpendicular to fibers

**Biological Meaning:**
- Non-Gaussian diffusion across fiber orientation
- Reflects barriers perpendicular to tracts (myelin, membranes)
- Sensitive to myelination and extra-axonal space

**Typical Values:**
- Heavily myelinated WM (corpus callosum): 1.0-2.0
- Lightly myelinated WM: 0.8-1.5
- Gray matter: 0.8-1.3

**Clinical Relevance:**
- ↓ in demyelination (MS, leukodystrophies)
- ↑ in development (myelination)
- ↓ in aging (myelin breakdown)
- More specific to myelin than RD

**Relationship to RD:**
- RD measures magnitude of diffusion across fibers
- RK measures how non-Gaussian that diffusion is
- RK more sensitive to myelin microstructure than RD

**Myelin Sensitivity:**
- RK is particularly sensitive to myelin integrity
- Complements FA for white matter assessment
- May detect early demyelination before FA changes

---

#### `kfa.nii.gz` - Kurtosis Fractional Anisotropy

**Full Name:** Kurtosis Fractional Anisotropy

**Range:** 0.0 - 1.0 (unitless)

**Interpretation:**
- **1.0:** Maximally anisotropic kurtosis (directional restriction)
- **0.0:** Isotropic kurtosis (uniform restriction)

**Biological Meaning:**
- Anisotropy of the kurtosis tensor (not diffusion tensor)
- Directional dependence of non-Gaussian diffusion
- Reflects organizational complexity

**Typical Values:**
- Major WM tracts: 0.3-0.6
- Crossing fiber regions: 0.2-0.4
- Gray matter: 0.1-0.3
- CSF: ~0

**Clinical Relevance:**
- Reflects fiber coherence and organization
- Sensitive to crossing fibers (↓ where fibers cross)
- Changes in white matter pathology
- Complements standard FA

**Relationship to FA:**
- FA: Anisotropy of diffusion magnitude
- KFA: Anisotropy of diffusion complexity
- Both measure directionality, different aspects
- KFA can be high even when FA is moderate

**Use Cases:**
- White matter organization assessment
- Crossing fiber detection (low KFA)
- Microstructural anisotropy beyond DTI

---

## Interpretation Guidelines

### Comparing Models

#### NODDI vs SANDI

**When to use NODDI:**
- Whole brain analysis (white + gray matter)
- Focus on neurite density and organization
- Simpler model, faster computation
- Well-validated across many studies

**When to use SANDI:**
- Gray matter specific questions
- Distinguish soma from neurites
- Neuron density and soma size
- Cortical/subcortical analysis

**Relationship:**
- NODDI FICVF ≈ SANDI FSOMA + SANDI FNEURITE
- NODDI FISO ≈ SANDI FCSF
- SANDI provides finer compartmentalization

#### NODDI vs ActiveAx

**When to use NODDI:**
- General neurite microstructure
- Gray + white matter
- Faster, more robust
- Standard clinical scanners

**When to use ActiveAx:**
- White matter specific questions
- Axon diameter estimation (with caveats)
- High-end scanner with strong gradients
- Research applications

**Relationship:**
- Both measure intra-axonal volume fraction
- ActiveAx adds diameter information
- ActiveAx more specialized, less robust

#### DKI Complements All Models

**DKI + NODDI/SANDI/ActiveAx:**
- DKI measures complexity (how non-Gaussian)
- Other models explain WHY it's non-Gaussian
- Use together for complete picture
- DKI is fast pre-screening, models for detailed analysis

### Regional Interpretation

#### White Matter

**Primary metrics:**
- FICVF (NODDI, ActiveAx): Axon density
- ODI (NODDI): Fiber dispersion
- RK (DKI): Myelin integrity
- AK (DKI): Intra-axonal complexity
- Axon diameter (ActiveAx): If strong gradients available

**Example - Corpus Callosum:**
- High FICVF (0.7-0.9): Dense axons
- Low ODI (0.1-0.3): Coherent bundle
- High RK (1.0-2.0): Well-myelinated
- Diameter variation: Motor (large) vs sensory (small) portions

#### Gray Matter

**Primary metrics:**
- FSOMA (SANDI): Neuron density
- FNEURITE (SANDI): Dendrite density
- ODI (NODDI): Dendritic dispersion
- MK (DKI): Overall complexity
- Soma radius (SANDI): Neuron type/size

**Example - Motor Cortex:**
- High FSOMA (0.3-0.5): Dense neurons
- High FNEURITE (0.3-0.5): Dense dendrites
- High ODI (0.7-0.9): Dispersed dendrites
- Large soma radius (8-12 μm): Pyramidal neurons
- High MK (1.0-1.5): Complex microstructure

#### Pathology

**Example - MS Lesion:**
- ↓ FICVF: Axon loss
- ↓ RK: Demyelination
- ↑ FISO/FCSF: Edema/inflammation
- Normal or ↑ MK (acute), ↓ MK (chronic)
- ↑ FEC (SANDI): Increased extracellular space

**Example - Alzheimer's Disease:**
- ↓ FSOMA: Neuronal loss
- ↓ FNEURITE: Dendritic loss
- ↓ MK: Reduced complexity
- ↑ FISO: Atrophy with compensatory CSF

---

## Data Requirements

### B-value Requirements

| Model | Minimum b-values | Recommended | Maximum useful |
|-------|-----------------|-------------|----------------|
| **DKI** | ≥2 non-zero shells | b=0, 1000, 2000 | 3000 |
| **NODDI** | ≥2 non-zero shells | b=0, 1000, 2000 | 3000 |
| **SANDI** | ≥3 non-zero shells | b=0, 1000, 2000, 3000+ | 5000+ |
| **ActiveAx** | ≥2 non-zero shells | b=0, 1000, 2000, 3000+ | 4000-6000 |

### Angular Sampling

| Model | Minimum directions/shell | Recommended |
|-------|-------------------------|-------------|
| **DKI** | 30 | 60 |
| **NODDI** | 30 | 60 |
| **SANDI** | 30 | 90 |
| **ActiveAx** | 60 | 90-120 |

### SNR Requirements

- **DKI:** Moderate SNR (>10)
- **NODDI:** Moderate SNR (>10-15)
- **SANDI:** High SNR (>15-20) for soma sensitivity
- **ActiveAx:** Very high SNR (>20-30) for diameter accuracy

### Scanner Requirements

#### Standard Clinical Scanner (OK):
- DKI ✓
- NODDI ✓
- SANDI ✓ (with high b-values)
- ActiveAx ⚠️ (FICVF only, diameter estimates unreliable)

#### High-Performance Research Scanner (Optimal):
- DKI ✓✓
- NODDI ✓✓
- SANDI ✓✓
- ActiveAx ✓✓ (reliable diameter estimation)

**Gradient strength:**
- Clinical: 40-80 mT/m (NODDI, DKI, SANDI okay; ActiveAx limited)
- Connectom: 300 mT/m (all models optimal, reliable diameter)

---

## Quality Control

### Automated Checks

#### Range Checks

```python
# Expected ranges for quality control
QC_RANGES = {
    # NODDI
    'noddi/ficvf': (0.0, 1.0),      # Should be bounded
    'noddi/odi': (0.0, 1.0),        # Should be bounded
    'noddi/fiso': (0.0, 1.0),       # Should be bounded

    # SANDI
    'sandi/fsoma': (0.0, 1.0),
    'sandi/fneurite': (0.0, 1.0),
    'sandi/fec': (0.0, 1.0),
    'sandi/fcsf': (0.0, 1.0),
    'sandi/rsoma': (1.0, 12.0),     # μm

    # ActiveAx
    'activeax/ficvf': (0.0, 1.0),
    'activeax/diam': (0.1, 10.0),   # μm

    # DKI
    'dki/mk': (0.0, 3.0),            # Typical range
    'dki/ak': (0.0, 3.0),
    'dki/rk': (0.0, 3.0),
    'dki/kfa': (0.0, 1.0)
}
```

#### Volume Conservation (SANDI)

```python
# SANDI compartments should sum to ~1.0
total = fsoma + fneurite + fec + fcsf
assert 0.95 < total.mean() < 1.05, "Volume fractions don't sum to 1"
```

#### Physiological Validation

```python
# CSF should be near 1.0 in ventricles
ventricle_fiso = fiso[ventricle_mask]
assert ventricle_fiso.mean() > 0.8, "CSF not detected in ventricles"

# White matter should have high FICVF
wm_ficvf = ficvf[white_matter_mask]
assert wm_ficvf.mean() > 0.5, "Unexpectedly low WM neurite density"
```

### Visual QC

#### What to Check

1. **Anatomical correspondence:**
   - High FICVF in major WM tracts
   - High FISO in ventricles
   - High FSOMA in gray matter nuclei

2. **No artifacts:**
   - No edge effects (wraparound)
   - No motion-corrupted slices
   - No unrealistic values

3. **Model-specific:**
   - DKI: MK > 0 everywhere in brain
   - NODDI: FISO near 1.0 in CSF
   - SANDI: FSOMA near 0 in white matter
   - ActiveAx: Reasonable diameter values (0.5-5 μm in WM)

#### Recommended Visualizations

```bash
# Overlay on T1w for anatomical context
fsleyes T1w.nii.gz \
    noddi/ficvf.nii.gz -cm hot -dr 0 1 \
    noddi/odi.nii.gz -cm cool -dr 0 1

# Check CSF in ventricles
fsleyes T1w.nii.gz \
    noddi/fiso.nii.gz -cm red-yellow -dr 0 1

# Fiber directions (RGB)
fsleyes T1w.nii.gz \
    noddi/dir.nii.gz -ot rgbvector
```

---

## References

### NODDI
Zhang, H., et al. (2012). "NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain." *NeuroImage* 61(4):1000-1016.

### SANDI
Palombo, M., et al. (2020). "SANDI: A compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI." *NeuroImage* 215:116835.

### ActiveAx
Alexander, D.C., et al. (2010). "Orientationally invariant indices of axon diameter and density from diffusion MRI." *NeuroImage* 52(4):1374-1389.

### DKI
Jensen, J.H., et al. (2005). "Diffusional kurtosis imaging: The quantification of non-gaussian water diffusion by means of magnetic resonance imaging." *Magnetic Resonance in Medicine* 53(6):1432-1440.

### AMICO
Daducci, A., et al. (2015). "Accelerated Microstructure Imaging via Convex Optimization (AMICO) from diffusion MRI data." *NeuroImage* 105:32-44.

---

## File Organization Summary

```
derivatives/dwi_topup/{subject}/advanced_models/
├── dki/
│   ├── mk.nii.gz           # Mean Kurtosis
│   ├── ak.nii.gz           # Axial Kurtosis
│   ├── rk.nii.gz           # Radial Kurtosis
│   └── kfa.nii.gz          # Kurtosis FA
├── noddi/
│   ├── ficvf.nii.gz        # Neurite density
│   ├── odi.nii.gz          # Orientation dispersion
│   ├── fiso.nii.gz         # Free water fraction
│   └── dir.nii.gz          # Fiber direction (3D vector)
├── sandi/
│   ├── fsoma.nii.gz        # Soma volume fraction
│   ├── fneurite.nii.gz     # Neurite volume fraction
│   ├── fec.nii.gz          # Extra-cellular fraction
│   ├── fcsf.nii.gz         # CSF fraction
│   ├── rsoma.nii.gz        # Soma radius (μm)
│   └── dir.nii.gz          # Neurite direction (3D vector)
└── activeax/
    ├── ficvf.nii.gz        # Intra-axonal volume fraction
    ├── diam.nii.gz         # Mean axon diameter (μm)
    ├── dir.nii.gz          # Fiber direction (3D vector)
    └── fvf_tot.nii.gz      # Total fiber volume fraction
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-12
**Status:** Complete

For questions or issues, refer to the individual model documentation in the module docstrings or consult the references above.
