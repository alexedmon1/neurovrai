# ASL (Arterial Spin Labeling) Preprocessing Plan

## Overview

ASL is a non-invasive perfusion imaging technique that measures cerebral blood flow (CBF) by magnetically labeling arterial blood water as an endogenous tracer.

**Data Type**: pCASL (pseudo-continuous ASL)
**Scanner**: Philips Ingenia Elition X 3T
**Sequence**: 2D FEEPI, TR=4.32s, TE=15.82ms
**Resolution**: 2.5×2.5×6mm (22 slices)
**Volumes**: 66 (33 label-control pairs)

---

## ASL Preprocessing Pipeline

### Standard Preprocessing Steps

1. **Motion Correction**
   - MCFLIRT for inter-volume motion correction
   - Register all volumes to middle volume
   - Generate motion parameters for QC

2. **Label-Control Subtraction**
   - Separate label and control volumes
   - Pairwise subtraction: Control - Label = ΔM (perfusion-weighted signal)
   - Average all subtraction pairs → mean ΔM image

3. **Brain Extraction**
   - BET on mean ΔM or mean control image
   - Use aggressive settings (frac=0.3) for low-intensity perfusion images

4. **Registration to Anatomical Space**
   - Register mean control image to T1w (BBR or correlation ratio)
   - Apply transform to perfusion maps

5. **CBF Quantification**
   - Convert ΔM to absolute CBF (ml/100g/min)
   - Requires acquisition parameters:
     - Labeling duration (τ)
     - Post-labeling delay (PLD/TI)
     - Blood T1 relaxation time
     - Labeling efficiency (α)
     - Partition coefficient (λ)

6. **Spatial Normalization** (optional)
   - Reuse anatomical→MNI152 transforms
   - Concatenate: ASL→anat→MNI152

7. **Partial Volume Correction** (optional)
   - Use tissue segmentation (GM/WM/CSF)
   - Correct for partial volume effects in low-resolution ASL

8. **Quality Control**
   - Motion parameters
   - tSNR in GM regions
   - CBF distribution (mean, std in GM/WM)
   - Visual inspection of CBF maps

---

## Implementation Strategy

### Tools to Use

**Primary**: FSL tools (MCFLIRT, BET, FLIRT, fslmaths)
**CBF Quantification**: Custom Python implementation using standard ASL equations
**Alternative**: ASL toolbox (if available) or FSL BASIL

### Workflow Architecture

**Location**: `mri_preprocess/workflows/asl_preprocess.py`

**Main function**: `run_asl_preprocessing()`

**Utilities**: `mri_preprocess/utils/asl_cbf.py` for CBF quantification

---

## CBF Quantification Equation

The standard ASL single-compartment model:

```
CBF = (λ · ΔM · e^(PLD/T1_blood)) / (2 · α · T1_blood · M0 · (1 - e^(-τ/T1_blood)))
```

Where:
- **ΔM**: Perfusion-weighted signal (Control - Label)
- **M0**: Equilibrium magnetization (from control or proton density image)
- **λ**: Blood-brain partition coefficient (typically 0.9 ml/g)
- **α**: Labeling efficiency (typically 0.85 for pCASL)
- **τ**: Labeling duration (typically 1.8s for pCASL)
- **PLD**: Post-labeling delay (time from end of labeling to imaging)
- **T1_blood**: Blood T1 relaxation time at 3T (typically 1650ms)

---

## Required Acquisition Parameters

From DICOM/JSON metadata:
- **TR**: RepetitionTime (4.32s from JSON) ✓
- **TE**: EchoTime (15.82ms from JSON) ✓
- **τ (labeling duration)**: Need to extract from DICOM private tags
- **PLD (post-labeling delay)**: Need to extract from DICOM private tags
- **Label/Control order**: Need to determine (usually alternating: C-L-C-L...)

**Action**: Check Philips DICOM private tags or scanner protocol for pCASL parameters

---

## Workflow Structure

```
derivatives/{subject}/asl/
├── motion_corrected.nii.gz              # Motion-corrected 4D time series
├── motion_params.txt                    # MCFLIRT parameters
├── control_mean.nii.gz                  # Mean control image
├── label_mean.nii.gz                    # Mean label image
├── perfusion_mean.nii.gz                # Mean perfusion-weighted image (ΔM)
├── brain_mask.nii.gz                    # Brain mask
├── cbf.nii.gz                           # Cerebral blood flow map
├── cbf_gm.nii.gz                        # CBF masked to gray matter
├── transforms/
│   ├── asl_to_anat.mat                  # ASL→anatomical transform
│   └── asl_to_MNI152_warp.nii.gz        # Concatenated warp (reuses anat transforms)
├── normalized/
│   └── cbf_normalized.nii.gz            # CBF in MNI152 space
└── qc/
    ├── motion_qc.tsv
    ├── cbf_qc.tsv
    └── asl_qc_report.html
```

---

## Implementation Phases

### Phase 1: Basic Preprocessing (Priority 1)
- [x] Research ASL pipeline requirements
- [ ] Implement motion correction
- [ ] Implement label-control separation and subtraction
- [ ] Implement brain extraction
- [ ] Generate mean perfusion images

### Phase 2: CBF Quantification (Priority 1)
- [ ] Extract acquisition parameters from DICOM
- [ ] Implement CBF quantification equation
- [ ] Validate CBF values (expected GM: 40-60 ml/100g/min, WM: 20-30 ml/100g/min)

### Phase 3: Registration & Normalization (Priority 2)
- [ ] Implement ASL→anatomical registration
- [ ] Implement spatial normalization (reuse anatomical transforms)
- [ ] Generate normalized CBF maps

### Phase 4: Quality Control (Priority 2)
- [ ] Implement motion QC
- [ ] Implement CBF QC metrics
- [ ] Generate QC report

### Phase 5: Advanced Processing (Priority 3)
- [ ] Implement partial volume correction
- [ ] Integrate FreeSurfer segmentation for better GM/WM masks
- [ ] Multi-PLD support (if available)

---

## Testing Plan

**Test Subject**: IRC805-0580101

1. Run basic preprocessing on SOURCE file
2. Verify motion correction quality
3. Check label-control subtraction
4. Validate CBF quantification (expected values in physiological range)
5. Visual QC of CBF maps

---

## References

- **ASLtbx**: SPM-based ASL analysis toolbox
- **FSL BASIL**: Bayesian inference for ASL
- **ExploreASL**: Comprehensive ASL analysis pipeline
- **ASL White Paper** (Alsop et al., MRM 2015): Recommended implementation for pCASL

---

## Next Steps

1. Create `mri_preprocess/workflows/asl_preprocess.py`
2. Create `mri_preprocess/utils/asl_cbf.py` for CBF quantification
3. Extract Philips-specific pCASL parameters from DICOM
4. Implement and test basic preprocessing on IRC805-0580101
