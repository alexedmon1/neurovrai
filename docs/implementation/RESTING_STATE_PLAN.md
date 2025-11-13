# Resting State fMRI Preprocessing Plan

## Dataset Overview

### IRC805 Multi-Echo Resting State fMRI
- **Scanner**: Philips Ingenia Elition X 3T
- **Sequence**: Multi-echo gradient echo EPI (FEEPI)
- **Multi-band acceleration**: MB3 (factor 3)
- **SENSE acceleration**: 3
- **Echoes**: 3 (TE1=10.0ms, TE2≈30ms, TE3≈50ms - need to verify)
- **TR**: 1.029 seconds
- **Flip angle**: 65°
- **Slice thickness**: 3mm
- **Volumes**: ~450 timepoints (need to verify with fslinfo)
- **Total scan time**: ~7.7 minutes

### Data Location
- **Subject**: IRC805-0580101
- **Path**: `/mnt/bytopia/IRC805/subjects/IRC805-0580101/nifti/rest/`
- **Files**:
  - `501_IRC805-0580101_WIP_RESTING_ME3_MB3_SENSE3_e1.nii.gz` (296 MB)
  - `501_IRC805-0580101_WIP_RESTING_ME3_MB3_SENSE3_e2.nii.gz` (266 MB)
  - `501_IRC805-0580101_WIP_RESTING_ME3_MB3_SENSE3_e3.nii.gz` (251 MB)
- **Field map correction scans** (also multi-echo):
  - `301/401_IRC805-0580101_WIP_fMRI_CORRECTION_MB3_ME3_SENSE3_3mm_TR1104_TE15_DTE20_e[1-3].nii.gz`

## Preprocessing Pipeline Overview

### Modern Approach (RECOMMENDED)

**Pipeline**: TEDANA → MCFLIRT → ICA-AROMA → ACompCor → Spatial Smoothing → Registration to MNI

#### Step 1: Multi-Echo Denoising with TEDANA ✨
**Purpose**: Remove thermal noise and motion artifacts using optimal combination of echoes

```python
from tedana import workflows

workflows.tedana_workflow(
    data=[e1_file, e2_file, e3_file],
    tes=[10.0, 30.0, 50.0],  # Echo times in ms (verify from JSON)
    out_dir=tedana_dir,
    mask=func_mask,  # Optional: provide mask from first echo BET
    tedpca='kundu',  # Or 'aic', 'kic', 'mdl'
    tree='kundu',    # Or 'minimal'
    verbose=True
)
```

**Outputs**:
- `desc-optcom_bold.nii.gz`: Optimally combined data (use for further preprocessing)
- `desc-denoised_bold.nii.gz`: Denoised data (after component removal)
- `desc-MEICA_components.nii.gz`: Independent component timeseries
- `desc-MEICA_mixing.tsv`: Mixing matrix
- `desc-tedana_metrics.tsv`: Component metrics (kappa, rho, variance explained)

**Benefits**:
- Removes thermal noise (exponentially decays with TE)
- Identifies BOLD signal (scales with TE)
- Automatic identification of motion/non-BOLD components
- Better tSNR than single-echo

#### Step 2: Motion Correction (MCFLIRT)
**Input**: TEDANA optimally combined data
**Tool**: FSL MCFLIRT

```python
mcflirt = Node(fsl.MCFLIRT(
    cost='leastsquares',
    save_plots=True,
    save_mats=True,
    save_rms=True,
    output_type='NIFTI_GZ'
), name='motion_correction')
```

**Outputs**:
- Motion-corrected functional data
- Motion parameters (6 DOF: 3 translations, 3 rotations)
- RMS plots for QC

#### Step 3: ICA-AROMA (Motion Artifact Removal)
**Purpose**: Additional motion artifact removal using ICA

```python
aroma = Node(fsl.ICA_AROMA(
    denoise_type='both',  # Produces both non-aggressive and aggressive
    TR=1.029
), name='ica_aroma')
```

**Note**: May be partially redundant with TEDANA, but provides additional cleaning

#### Step 4: Nuisance Regression (ACompCor)
**Purpose**: Remove physiological noise from CSF and white matter

**Requirements**:
- CSF mask from anatomical FAST segmentation
- WM mask from anatomical FAST segmentation

```python
# Extract tissue masks
csf_mask = derivatives/anat_preproc/IRC805-0580101/fast_seg_0.nii.gz
wm_mask = derivatives/anat_preproc/IRC805-0580101/fast_seg_2.nii.gz

# Combine and erode for motion mask
motion_mask = ImageMaths(op_string='-add').run(
    in_file=wm_mask,
    in_file2=csf_mask
)
eroded_mask = ErodeImage().run(in_file=motion_mask.outputs.out_file)

# Run ACompCor
acompcor = Node(confounds.ACompCor(
    num_components=6,
    repetition_time=1.029
), name='acompcor')

# Regress out nuisance components
glm = Node(fsl.GLM(
    output_type='NIFTI_GZ',
    out_res_name='residuals.nii.gz'
), name='nuisance_regression')
```

#### Step 5: Temporal Filtering
**Purpose**: Bandpass filter to retain resting state frequencies

```python
bandpass = Node(afni.Bandpass(
    highpass=0.001,  # 0.001 Hz
    lowpass=0.08,    # 0.08 Hz (standard resting state band)
    tr=1.029,
    outputtype='NIFTI_GZ'
), name='bandpass')
```

#### Step 6: Spatial Smoothing
**Purpose**: Improve SNR and anatomical correspondence

```python
smooth = Node(fsl.Smooth(
    fwhm=6,  # 6mm FWHM (standard for 3mm voxels)
    output_type='NIFTI_GZ'
), name='spatial_smooth')
```

#### Step 7: Registration to MNI Space
**Strategy**: Use pre-computed T1w→MNI transforms from anatomical workflow

```python
# Functional → T1w (BBR registration)
func_to_t1 = Node(fsl.FLIRT(
    dof=6,
    cost_func='bbr',  # Boundary-based registration
    output_type='NIFTI_GZ'
), name='func_to_t1')

# Concatenate transforms: func→T1→MNI
concat_xfm = Node(fsl.ConvertXFM(
    concat_xfm=True
), name='concat_transforms')

# Apply combined transform
apply_warp = Node(fsl.ApplyWarp(
    ref_file=MNI152_2mm,
    output_type='NIFTI_GZ'
), name='warp_to_mni')
```

### Alternative: Legacy Workflow (Without TEDANA)

For comparison with existing processed data or if TEDANA fails:

**Pipeline**: MCFLIRT on middle echo → ICA-AROMA → Coregister to T1w → Warp to MNI → ACompCor → Bandpass → Smooth

This is the approach in `archive/rest/rest-preproc-dev.py` but without multi-echo advantages.

## Implementation Steps

### Phase 1: Verify Data Quality ✅
```bash
# Check echo times in all JSON files
for echo in e1 e2 e3; do
  echo "Echo: $echo"
  grep "EchoTime" 501_*_${echo}.json
done

# Check dimensions and timepoints
for echo in e1 e2 e3; do
  echo "Echo: $echo"
  fslinfo 501_*_${echo}.nii.gz | grep -E "^dim[0-4]|^pixdim[1-4]"
done

# Quick visual QC
fsleyes 501_*_e1.nii.gz &
```

### Phase 2: Update func_preprocess.py 🔧

**File**: `mri_preprocess/workflows/func_preprocess.py`

Add implementations for:
1. Multi-echo detection and TEDANA integration
2. Motion correction (MCFLIRT)
3. ICA-AROMA
4. ACompCor with tissue masks from anatomical workflow
5. Bandpass filtering
6. Spatial smoothing
7. Registration to MNI using TransformRegistry

### Phase 3: Create Test Script 🧪

```python
#!/usr/bin/env python3
"""
test_rest_preprocessing.py - Test resting state fMRI preprocessing
"""
from pathlib import Path
from mri_preprocess.workflows.func_preprocess import run_func_preprocessing

# Configuration
config = {
    'tr': 1.029,
    'te': [10.0, 30.0, 50.0],  # Verify from JSON
    'highpass': 0.001,
    'lowpass': 0.08,
    'fwhm': 6,
    'tedana': {
        'enabled': True,
        'tedpca': 'kundu',
        'tree': 'kundu'
    },
    'aroma': {
        'enabled': True,
        'denoise_type': 'both'
    },
    'acompcor': {
        'enabled': True,
        'num_components': 6
    }
}

# Paths
study_root = Path('/mnt/bytopia/IRC805')
subject = 'IRC805-0580101'

# Multi-echo functional files
func_files = [
    study_root / f'subjects/{subject}/nifti/rest/501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e1.nii.gz',
    study_root / f'subjects/{subject}/nifti/rest/501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e2.nii.gz',
    study_root / f'subjects/{subject}/nifti/rest/501_{subject}_WIP_RESTING_ME3_MB3_SENSE3_e3.nii.gz'
]

# Anatomical tissue masks (from anat_preprocess workflow)
anat_dir = study_root / f'derivatives/anat_preproc/{subject}'
csf_mask = anat_dir / 'fast_seg_0.nii.gz'  # CSF
wm_mask = anat_dir / 'fast_seg_2.nii.gz'   # White matter

# Run preprocessing
results = run_func_preprocessing(
    config=config,
    subject=subject,
    func_file=func_files,  # List for multi-echo
    output_dir=study_root,  # Study root, not derivatives
    csf_mask=csf_mask,
    wm_mask=wm_mask
)

print(f"Preprocessed data: {results['preprocessed']}")
print(f"Derivatives directory: {results['derivatives_dir']}")
```

### Phase 4: Quality Control 📊

After preprocessing, perform QC:

```bash
# 1. Check TEDANA reports
firefox derivatives/func_preproc/IRC805-0580101/tedana/report.html

# 2. Check motion parameters
fsleyes derivatives/func_preproc/IRC805-0580101/mcflirt_motion.par

# 3. Visualize preprocessed data
fsleyes \
  $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
  derivatives/func_preproc/IRC805-0580101/func_mni.nii.gz &

# 4. Compute tSNR
fslmaths func_mni.nii.gz -Tmean mean_func
fslmaths func_mni.nii.gz -Tstd std_func
fslmaths mean_func -div std_func tsnr_func

# 5. Check registration quality
overlay 1 0 \
  $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz -a \
  mean_func.nii.gz 500 10000 \
  registration_check.nii.gz
```

### Phase 5: Connectivity Analysis (Optional) 🔗

After preprocessing, can perform:

1. **Seed-based correlation**
2. **Dual regression with ICA templates** (FSL MELODIC)
3. **Graph theory analysis**
4. **Dynamic functional connectivity**

Utilities exist in `archive/rest/`:
- `rest_dualregress.py`: Dual regression with spatial maps
- `rest_corrmat_generation.py`: Connectivity matrix generation

## Expected Runtime

### Per Subject
- **TEDANA**: ~5-10 minutes
- **Motion correction**: ~5 minutes
- **ICA-AROMA**: ~15-20 minutes
- **ACompCor + nuisance regression**: ~5 minutes
- **Spatial smoothing**: ~2 minutes
- **Registration to MNI**: ~5 minutes
- **Total**: ~45-60 minutes per subject

### For IRC805 cohort (~100 subjects)
- Serial: ~75-100 hours
- Parallel (10 subjects): ~7.5-10 hours

## Key Differences from DWI Processing

| Aspect | DWI | rs-fMRI |
|--------|-----|---------|
| **Main artifact** | Eddy currents, motion | Physiological noise, motion |
| **Correction** | TOPUP + eddy | TEDANA + ICA-AROMA |
| **Output space** | Native or T1w | Usually MNI for group analysis |
| **Temporal aspect** | Independent volumes | Time series correlations |
| **Primary QC** | Visual inspection, FA maps | tSNR, motion params, connectivity |

## Next Steps

1. ✅ Verify echo times from JSON metadata
2. 🔧 Implement TEDANA integration in `func_preprocess.py`
3. 🔧 Add MCFLIRT, ICA-AROMA, ACompCor nodes
4. 🔧 Implement registration to MNI with TransformRegistry
5. 🧪 Test on IRC805-0580101
6. 📊 Create QC report generation
7. 🚀 Deploy to full IRC805 cohort

## References

- TEDANA documentation: https://tedana.readthedocs.io/
- ICA-AROMA paper: Pruim et al. (2015) NeuroImage
- ACompCor paper: Behzadi et al. (2007) NeuroImage
- Multi-echo fMRI review: Kundu et al. (2017) NeuroImage
