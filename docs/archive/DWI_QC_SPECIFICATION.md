# DWI Quality Control Specification

**Version**: 1.0
**Date**: November 11, 2025
**Status**: Implementation Specification

## Overview

This document specifies the outputs, storage locations, and implementation details for DWI preprocessing quality control and validation.

## Directory Structure

All QC outputs will be stored in a centralized QC directory at the study root level, organized by modality:

```
{study_root}/
├── derivatives/
│   ├── dwi_topup/{subject}/       # DWI preprocessing outputs
│   ├── anat_preproc/{subject}/    # Anatomical preprocessing outputs
│   └── func_preproc/{subject}/    # Functional preprocessing outputs
│
└── qc/                            # NEW: Centralized Quality Control
    ├── dwi/                       # DWI-specific QC
    │   └── {subject}/
    │       ├── reports/           # HTML/PDF QC reports
    │       ├── metrics/           # Quantitative metrics (JSON/CSV)
    │       ├── images/            # Visual QC images (PNG/JPG)
    │       ├── motion/            # Motion parameters and plots
    │       ├── topup/             # TOPUP-specific QC
    │       └── comparisons/       # Before/after comparisons
    │
    ├── rest/                      # Resting-state fMRI QC (future)
    │   └── {subject}/
    │       ├── reports/
    │       ├── metrics/
    │       ├── images/
    │       ├── motion/
    │       ├── carpet_plots/
    │       └── ica_aroma/
    │
    └── anat/                      # Anatomical QC (future)
        └── {subject}/
            ├── reports/
            ├── metrics/
            ├── images/
            ├── segmentation/
            └── registration/
```

**Rationale for Centralized QC Structure**:
- All QC outputs in one place for easy access
- Modality-specific organization (dwi/, rest/, anat/)
- Cross-modality comparisons easier (e.g., DWI-T1w registration QC)
- Group-level QC summaries can aggregate across modalities
- Cleaner separation between processing outputs and QC

## QC Outputs Specification

### 1. TOPUP Quality Control

**Purpose**: Validate distortion correction effectiveness

#### Outputs

##### 1.1 Field Map Visualization
**Files**:
- `qc/dwi/{subject}/topup/fieldmap_sagittal.png` - Sagittal view of field map
- `qc/dwi/{subject}/topup/fieldmap_coronal.png` - Coronal view of field map
- `qc/dwi/{subject}/topup/fieldmap_axial.png` - Axial view of field map

**Content**:
- Heatmap overlay showing field distortions
- Color scale indicating distortion magnitude
- Anatomical slices at key locations

**Generation Method**:
```python
# Using nibabel + matplotlib
import nibabel as nib
import matplotlib.pyplot as plt

fieldmap = nib.load('topup_results_fieldcoef.nii.gz')
# Create multi-slice visualization with colormap
```

---

##### 1.2 Before/After Comparison
**Files**:
- `qc/dwi/{subject}/topup/b0_comparison_uncorrected.png` - B0 before TOPUP
- `qc/dwi/{subject}/topup/b0_comparison_corrected.png` - B0 after TOPUP
- `qc/dwi/{subject}/topup/b0_comparison_overlay.png` - Side-by-side comparison
- `qc/dwi/{subject}/topup/difference_map.png` - Difference image

**Content**:
- Edge detection overlay showing geometric distortion
- Checkerboard comparison
- Difference map highlighting corrections

**Generation Method**:
```python
# Using FSL slicesdir or custom Python visualization
from nilearn import plotting
plotting.plot_anat(corrected_b0, title="Corrected B0")
```

---

##### 1.3 TOPUP Convergence Metrics
**Files**:
- `qc/dwi/{subject}/topup/convergence_plot.png` - SSD vs iteration plot
- `qc/dwi/{subject}/metrics/topup_convergence.json` - Convergence data

**Content** (JSON):
```json
{
  "subject": "IRC805-0580101",
  "date": "2025-11-11",
  "topup_version": "FSL 6.0.7",
  "convergence": {
    "iterations": 12,
    "converged": true,
    "final_ssd": 2853.27,
    "initial_ssd": 5667.87,
    "improvement_percent": 49.7,
    "ssd_by_iteration": [5667.87, 2483940, 212153, ...],
    "convergence_rate": "normal"
  },
  "field_statistics": {
    "mean_displacement_mm": 2.3,
    "max_displacement_mm": 8.5,
    "std_displacement_mm": 1.1
  }
}
```

---

### 2. Motion Parameters QC

**Purpose**: Identify excessive motion and outlier volumes

#### Outputs

##### 2.1 Motion Plots
**Files**:
- `qc/dwi/{subject}/motion/translation_plot.png` - Translation over time (X, Y, Z)
- `qc/dwi/{subject}/motion/rotation_plot.png` - Rotation over time (pitch, roll, yaw)
- `qc/dwi/{subject}/motion/displacement_plot.png` - Frame-wise displacement
- `qc/dwi/{subject}/motion/outliers_plot.png` - Volumes flagged as outliers

**Content**:
- Time series plots of 6 motion parameters
- Threshold lines for acceptable motion (e.g., 2mm, 2°)
- Red markers for outlier volumes
- Summary statistics

**Generation Method**:
```python
# Parse eddy's .eddy_movement_rms file
import pandas as pd
import matplotlib.pyplot as plt

motion_data = pd.read_csv('eddy_corrected.eddy_movement_rms', sep='\s+')
plt.plot(motion_data['abs_displacement'])
```

---

##### 2.2 Motion Metrics
**Files**:
- `qc/dwi/{subject}/metrics/motion_parameters.json` - Comprehensive motion stats
- `qc/dwi/{subject}/metrics/motion_summary.csv` - Per-volume motion table

**Content** (JSON):
```json
{
  "subject": "IRC805-0580101",
  "total_volumes": 220,
  "motion_summary": {
    "mean_fd_mm": 0.25,
    "max_fd_mm": 1.82,
    "std_fd_mm": 0.31,
    "mean_rotation_deg": 0.12,
    "max_rotation_deg": 0.87,
    "outlier_volumes": [45, 112, 156],
    "outlier_count": 3,
    "outlier_percent": 1.4,
    "volumes_above_2mm": 0,
    "volumes_above_1mm": 12
  },
  "translation_mm": {
    "x": {"mean": 0.08, "max": 0.65, "std": 0.12},
    "y": {"mean": 0.11, "max": 0.89, "std": 0.15},
    "z": {"mean": 0.06, "max": 0.45, "std": 0.09}
  },
  "rotation_deg": {
    "pitch": {"mean": 0.05, "max": 0.41, "std": 0.07},
    "roll": {"mean": 0.07, "max": 0.52, "std": 0.09},
    "yaw": {"mean": 0.04, "max": 0.38, "std": 0.06}
  }
}
```

**Content** (CSV):
```csv
volume,translation_x,translation_y,translation_z,rotation_x,rotation_y,rotation_z,fd_mm,is_outlier
0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,False
1,0.12,0.08,0.05,0.02,0.03,0.01,0.15,False
2,0.15,0.11,0.07,0.04,0.05,0.02,0.22,False
...
45,1.25,0.87,0.42,0.35,0.28,0.18,1.82,True
...
```

---

### 3. SNR and Signal Quality

**Purpose**: Assess image quality improvements from preprocessing

#### Outputs

##### 3.1 SNR Measurements
**Files**:
- `qc/dwi/{subject}/metrics/snr_analysis.json` - SNR before/after correction
- `qc/dwi/{subject}/images/snr_comparison.png` - Visual SNR comparison

**Content** (JSON):
```json
{
  "subject": "IRC805-0580101",
  "b0_snr": {
    "uncorrected": 18.5,
    "topup_corrected": 22.3,
    "improvement_percent": 20.5
  },
  "dwi_snr_by_shell": {
    "b1000": {"mean": 12.3, "std": 2.1},
    "b2000": {"mean": 9.8, "std": 1.7},
    "b3000": {"mean": 7.5, "std": 1.4}
  },
  "snr_by_region": {
    "white_matter": 15.2,
    "gray_matter": 11.8,
    "csf": 8.3
  }
}
```

**Calculation Method**:
```python
# SNR = mean(signal) / std(noise)
# Noise estimated from background region or residuals
signal_roi = brain_mask * dwi_data
noise_roi = background_mask * dwi_data
snr = np.mean(signal_roi) / np.std(noise_roi)
```

---

##### 3.2 Residual Analysis
**Files**:
- `qc/dwi/{subject}/images/eddy_residuals.png` - Residual map
- `qc/dwi/{subject}/metrics/residual_stats.json` - Residual statistics

**Content**:
- Mean squared error per volume
- Spatial distribution of residuals
- Shell-specific residual patterns

---

### 4. DTI Metric Quality

**Purpose**: Validate DTI fitting and detect artifacts

#### Outputs

##### 4.1 FA Map Validation
**Files**:
- `qc/dwi/{subject}/images/fa_map_overview.png` - FA map with ROI overlays
- `qc/dwi/{subject}/images/fa_histogram.png` - FA distribution histogram
- `qc/dwi/{subject}/metrics/fa_statistics.json` - FA ROI statistics

**Content** (JSON):
```json
{
  "subject": "IRC805-0580101",
  "fa_statistics": {
    "global": {
      "mean": 0.38,
      "std": 0.15,
      "min": 0.0,
      "max": 0.95,
      "median": 0.42
    },
    "white_matter": {
      "mean": 0.52,
      "std": 0.08
    },
    "gray_matter": {
      "mean": 0.18,
      "std": 0.05
    },
    "corpus_callosum": {
      "mean": 0.68,
      "std": 0.06
    }
  },
  "expected_ranges": {
    "white_matter": [0.4, 0.7],
    "gray_matter": [0.1, 0.3],
    "corpus_callosum": [0.6, 0.8]
  },
  "quality_flags": {
    "values_in_expected_range": true,
    "excessive_low_fa": false,
    "nan_values": 0,
    "inf_values": 0
  }
}
```

---

##### 4.2 MD Map Validation
**Files**:
- `qc/dwi/{subject}/images/md_map_overview.png` - MD map overview
- `qc/dwi/{subject}/metrics/md_statistics.json` - MD ROI statistics

**Content**:
- Similar structure to FA statistics
- Expected MD ranges: WM (0.7-0.9 × 10⁻³ mm²/s), GM (0.8-1.0)

---

##### 4.3 Tensor Fit Quality
**Files**:
- `qc/dwi/{subject}/images/tensor_fitting_residuals.png` - Fitting residuals
- `qc/dwi/{subject}/metrics/tensor_fit_quality.json` - Goodness of fit metrics

**Content**:
```json
{
  "subject": "IRC805-0580101",
  "tensor_fitting": {
    "mean_r2": 0.94,
    "std_r2": 0.08,
    "poor_fit_voxels_percent": 2.3,
    "negative_eigenvalues": 0
  }
}
```

---

### 5. Visual Inspection Images

**Purpose**: Enable rapid visual QC by expert reviewers

#### Outputs

##### 5.1 Multi-Slice Mosaics
**Files**:
- `qc/dwi/{subject}/images/b0_mosaic.png` - B0 image mosaic (post-TOPUP)
- `qc/dwi/{subject}/images/fa_mosaic.png` - FA map mosaic
- `qc/dwi/{subject}/images/md_mosaic.png` - MD map mosaic
- `qc/dwi/{subject}/images/colored_fa_mosaic.png` - Color FA with direction encoding

**Content**:
- 4×6 grid of axial slices covering whole brain
- Consistent slice spacing
- Color FA: Red=L/R, Green=A/P, Blue=S/I

---

##### 5.2 Edge Overlays
**Files**:
- `qc/dwi/{subject}/images/fa_edges_on_b0.png` - FA edges on B0 (check registration)
- `qc/dwi/{subject}/images/mask_overlay.png` - Brain mask overlay

---

### 6. Comprehensive QC Report

**Purpose**: Single document for overall quality assessment

#### Outputs

##### 6.1 HTML Report
**File**: `qc/dwi/{subject}/reports/dwi_qc_report.html`

**Sections**:
1. **Subject Information**
   - Subject ID, scan date, acquisition parameters
   - Processing date and pipeline version

2. **TOPUP Summary**
   - Convergence plot
   - Field map visualization
   - Before/after comparison
   - Key metrics (iterations, SSD improvement)

3. **Motion Summary**
   - Motion plots (translation, rotation, FD)
   - Outlier detection
   - Summary statistics table

4. **Image Quality**
   - SNR measurements
   - Signal quality by shell
   - Residual analysis

5. **DTI Metrics**
   - FA/MD statistics
   - Histogram distributions
   - ROI-based validation
   - Quality flags

6. **Visual QC Gallery**
   - Multi-slice mosaics
   - Edge overlays
   - Colored FA

7. **Overall Quality Rating**
   - Pass/Fail/Review status
   - Automatic flagging of issues
   - Reviewer notes section

**Generation Method**:
```python
# Using Jinja2 templates or reportlab
from jinja2 import Template
template = Template(open('qc_template.html').read())
html = template.render(qc_data=metrics)
```

---

##### 6.2 PDF Report (Optional)
**File**: `qc/dwi/{subject}/reports/dwi_qc_report.pdf`

Same content as HTML but in PDF format for archiving.

---

### 7. Multi-Subject Summary

**Purpose**: Batch QC for multiple subjects

#### Outputs

##### 7.1 Cohort Summary
**File**: `{study_root}/derivatives/dwi_topup/group_qc_summary.csv`

**Content**:
```csv
subject,scan_date,topup_converged,topup_iterations,mean_fd_mm,outlier_volumes,fa_mean_wm,md_mean_wm,qc_status
IRC805-0580101,2025-11-11,True,12,0.25,3,0.52,0.82,PASS
IRC805-0580102,2025-11-12,True,10,0.31,5,0.51,0.84,PASS
IRC805-0580103,2025-11-13,True,15,0.89,18,0.48,0.88,REVIEW
...
```

---

##### 7.2 Group QC Dashboard
**File**: `{study_root}/derivatives/dwi_topup/group_qc_dashboard.html`

**Content**:
- Table of all subjects with key metrics
- Sortable/filterable columns
- Links to individual subject reports
- Aggregate statistics and distributions
- Outlier detection across cohort

---

## Implementation Priority

### Phase 1 (Week 1)
1. ✅ TOPUP convergence metrics (1.3)
2. ✅ Motion parameters extraction and plotting (2.1, 2.2)
3. ✅ Basic DTI statistics (4.1, 4.2)

### Phase 2 (Week 2)
4. ✅ Visual inspection images (5.1, 5.2)
5. ✅ TOPUP field map visualization (1.1, 1.2)
6. ✅ HTML QC report (6.1)

### Phase 3 (Week 3)
7. ⏳ SNR analysis (3.1, 3.2)
8. ⏳ Multi-subject summary (7.1, 7.2)
9. ⏳ Automated quality flags and thresholds

---

## Storage and Organization

### File Naming Convention
```
{subject}_dwi_qc_{metric}_{view}.{ext}

Examples:
IRC805-0580101_dwi_qc_motion_translation.png
IRC805-0580101_dwi_qc_fa_statistics.json
IRC805-0580101_dwi_qc_topup_convergence.json
```

### Total Storage Estimate per Subject
- Images (PNG): ~10-20 MB
- Metrics (JSON/CSV): ~1-2 MB
- Reports (HTML/PDF): ~5-10 MB
- **Total**: ~20-30 MB per subject

For 100 subjects: ~2-3 GB

---

## Quality Thresholds

### Automatic PASS/FAIL Criteria

**FAIL if**:
- TOPUP did not converge (iterations > 20 or increasing SSD)
- Mean FD > 2.0 mm
- Outlier volumes > 20%
- FA mean in WM < 0.3 or > 0.8
- NaN or Inf values in DTI maps

**REVIEW if**:
- Mean FD > 1.0 mm
- Outlier volumes > 10%
- FA mean in WM outside [0.4, 0.7]
- SNR improvement < 5%

**PASS**:
- All metrics within expected ranges

---

## Future Enhancements

1. **Interactive QC Interface**
   - Web-based dashboard with real-time updates
   - Manual QC ratings and notes
   - Issue tracking system

2. **Machine Learning QC**
   - Automated artifact detection
   - Outlier detection across cohort
   - Predicted quality ratings

3. **BIDS-Compatible QC**
   - Store QC metrics in BIDS-compatible format
   - Integration with MRIQC

4. **Real-Time Monitoring**
   - QC metrics calculated during processing
   - Early termination for poor-quality data

---

## Appendix A: Anatomical QC Specification (Future)

### Directory Structure
```
{study_root}/qc/anat/{subject}/
├── reports/              # HTML/PDF QC reports
├── metrics/              # Quantitative metrics (JSON/CSV)
├── images/               # Visual QC images
├── segmentation/         # Tissue segmentation QC
└── registration/         # Registration QC
```

### Key QC Outputs

#### 1. Brain Extraction QC
**Files**:
- `qc/anat/{subject}/images/brain_extraction_overlay.png` - BET mask overlay
- `qc/anat/{subject}/metrics/brain_extraction_stats.json` - Volume statistics

**Metrics**:
- Brain volume (cc)
- Mask coverage percentage
- Edge accuracy (visual inspection)

---

#### 2. Bias Field Correction QC
**Files**:
- `qc/anat/{subject}/images/bias_field_before_after.png` - Before/after comparison
- `qc/anat/{subject}/metrics/bias_field_stats.json` - Intensity statistics

**Metrics**:
- Intensity uniformity improvement
- Mean intensity by tissue class before/after

---

#### 3. Tissue Segmentation QC
**Files**:
- `qc/anat/{subject}/segmentation/tissue_prob_maps.png` - GM/WM/CSF probability maps
- `qc/anat/{subject}/segmentation/tissue_overlay.png` - Segmentation overlay on T1w
- `qc/anat/{subject}/metrics/segmentation_stats.json` - Volume statistics

**Metrics** (JSON):
```json
{
  "subject": "IRC805-0580101",
  "segmentation": {
    "gm_volume_cc": 650.5,
    "wm_volume_cc": 520.3,
    "csf_volume_cc": 210.8,
    "total_brain_volume_cc": 1381.6,
    "gm_percent": 47.1,
    "wm_percent": 37.7,
    "csf_percent": 15.3
  },
  "quality_flags": {
    "volumes_in_expected_range": true,
    "segmentation_overlap": 0.92
  }
}
```

---

#### 4. Registration to MNI QC
**Files**:
- `qc/anat/{subject}/registration/t1w_to_mni_edges.png` - Edge overlay on MNI template
- `qc/anat/{subject}/registration/t1w_to_mni_checkerboard.png` - Checkerboard comparison
- `qc/anat/{subject}/registration/registration_quality.json` - Registration metrics

**Metrics**:
- Normalized mutual information (NMI)
- Dice coefficient for tissue overlap
- Visual inspection rating

**Quality Checks**:
- No misalignment in major structures (corpus callosum, ventricles)
- Smooth warping without edge artifacts
- Symmetric registration

---

#### 5. Anatomical QC Report
**File**: `qc/anat/{subject}/reports/anat_qc_report.html`

**Sections**:
1. Subject info and scan parameters
2. Brain extraction summary
3. Bias correction results
4. Tissue segmentation volumes and overlays
5. Registration quality (affine + nonlinear)
6. Overall quality rating (Pass/Fail/Review)

---

## Appendix B: Resting-State fMRI QC Specification (Future)

### Directory Structure
```
{study_root}/qc/rest/{subject}/
├── reports/              # HTML/PDF QC reports
├── metrics/              # Quantitative metrics (JSON/CSV)
├── images/               # Visual QC images
├── motion/               # Motion parameters
├── carpet_plots/         # Carpet/grayplot visualizations
└── ica_aroma/            # ICA-AROMA component classification
```

### Key QC Outputs

#### 1. Motion Parameters QC
**Files**:
- `qc/rest/{subject}/motion/translation_plot.png` - Translation over time
- `qc/rest/{subject}/motion/rotation_plot.png` - Rotation over time
- `qc/rest/{subject}/motion/framewise_displacement.png` - FD time series
- `qc/rest/{subject}/metrics/motion_summary.json` - Motion statistics

**Metrics**:
- Mean FD, max FD, std FD
- Percentage of high-motion volumes (>0.5mm, >0.2mm thresholds)
- DVARS (temporal derivative of signals)

---

#### 2. Temporal SNR (tSNR)
**Files**:
- `qc/rest/{subject}/images/tsnr_map.png` - tSNR map overlay
- `qc/rest/{subject}/metrics/tsnr_stats.json` - tSNR by ROI

**Metrics** (JSON):
```json
{
  "subject": "IRC805-0580101",
  "tsnr": {
    "global_mean": 45.2,
    "gray_matter_mean": 52.3,
    "white_matter_mean": 38.7,
    "csf_mean": 15.2
  },
  "quality_flags": {
    "acceptable_tsnr": true,
    "threshold_gm": 40.0
  }
}
```

---

#### 3. Carpet Plot (Grayplot)
**Files**:
- `qc/rest/{subject}/carpet_plots/carpet_plot.png` - Full time series visualization
- `qc/rest/{subject}/carpet_plots/carpet_plot_denoised.png` - After nuisance regression

**Content**:
- Voxel time series organized by tissue type (cortex, subcortex, cerebellum, WM, CSF)
- Color intensity = BOLD signal
- Overlaid FD trace
- Overlaid respiration/cardiac traces (if available)

**Purpose**:
- Identify motion artifacts (horizontal stripes)
- Check effectiveness of denoising
- Detect physiological noise patterns

---

#### 4. ICA-AROMA Classification
**Files**:
- `qc/rest/{subject}/ica_aroma/motion_components.png` - Components classified as motion
- `qc/rest/{subject}/ica_aroma/signal_components.png` - Components classified as signal
- `qc/rest/{subject}/metrics/ica_aroma_summary.json` - Component classification

**Metrics**:
- Total components: 40
- Motion components: 12
- Signal components: 28
- Variance explained by motion components: 15.3%

---

#### 5. Nuisance Regression QC
**Files**:
- `qc/rest/{subject}/images/acompcor_components.png` - ACompCor components from WM/CSF
- `qc/rest/{subject}/images/confound_correlation_matrix.png` - Correlation between confounds
- `qc/rest/{subject}/metrics/nuisance_regression_stats.json` - Variance explained

**Metrics**:
- Variance explained by each confound regressor
- Number of ACompCor components retained
- Correlation between motion and signals

---

#### 6. Functional Connectivity QC
**Files**:
- `qc/rest/{subject}/images/connectivity_matrix.png` - Sample connectivity matrix
- `qc/rest/{subject}/images/dmn_connectivity.png` - Default mode network
- `qc/rest/{subject}/metrics/connectivity_stats.json` - Network metrics

**Metrics**:
- Mean within-network connectivity (DMN, FPN, SN, etc.)
- Distance-dependent connectivity decay
- QC-FC correlation (motion vs connectivity)

---

#### 7. Multi-Echo QC (if applicable)
**Files**:
- `qc/rest/{subject}/images/tedana_component_table.png` - TEDANA component classification
- `qc/rest/{subject}/metrics/tedana_summary.json` - Echo-dependent metrics

**Metrics**:
- Number of accepted/rejected components
- Variance explained by accepted components
- Kappa/Rho values for component classification

---

#### 8. Resting-State QC Report
**File**: `qc/rest/{subject}/reports/rest_qc_report.html`

**Sections**:
1. Subject info and scan parameters (TR, TEs, volumes, duration)
2. Motion summary (FD plots, outlier volumes, scrubbing)
3. tSNR maps and statistics
4. Carpet plots (before/after denoising)
5. ICA-AROMA classification (if used)
6. ACompCor components (WM/CSF masks)
7. Nuisance regression effectiveness
8. Sample connectivity matrices
9. Overall quality rating (Pass/Fail/Review)

**Quality Flags**:
- High motion: Mean FD > 0.5 mm or >20% volumes FD > 0.5mm
- Low tSNR: GM tSNR < 40
- Poor denoising: <50% variance explained by noise components
- Connectivity outlier: QC-FC correlation > 0.3

---

## Appendix C: Cross-Modality QC

### Purpose
Validate registration and integration across modalities

### Directory Structure
```
{study_root}/qc/cross_modality/{subject}/
├── dwi_to_t1w/           # DWI→T1w registration QC
├── rest_to_t1w/          # fMRI→T1w registration QC  
└── reports/              # Combined QC reports
```

### Key Outputs

#### DWI→T1w Registration QC
**Files**:
- `qc/cross_modality/{subject}/dwi_to_t1w/fa_on_t1w_edges.png`
- `qc/cross_modality/{subject}/dwi_to_t1w/registration_quality.json`

**Metrics**:
- Boundary-based registration cost
- Dice overlap of brain masks
- Visual inspection rating

---

#### fMRI→T1w Registration QC
**Files**:
- `qc/cross_modality/{subject}/rest_to_t1w/mean_bold_on_t1w_edges.png`
- `qc/cross_modality/{subject}/rest_to_t1w/registration_quality.json`

**Metrics**:
- BBR cost function value
- Tissue class alignment
- Visual inspection rating

---

