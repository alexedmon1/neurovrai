# Overnight Pipeline Run - IRC805-0580101

## Status: RUNNING
**Started**: 2025-11-13 23:30 UTC
**Log File**: `logs/overnight_run_0580101.log`
**Subject**: IRC805-0580101

## Current Progress
- ✅ DICOM conversion started
- ✅ Configuration validation passed
- ✅ Anatomical workflow started
- 🔄 DICOM conversion in progress

## What's Running
The complete end-to-end preprocessing pipeline with all fixes applied:

1. **DICOM → NIfTI Conversion**
   - anat: 5 files
   - dwi: 4 files
   - func: 3 files (multi-echo)
   - asl: 3 files

2. **Anatomical Preprocessing** (in progress)
   - N4 bias correction
   - Skull stripping (BET)
   - Tissue segmentation (ANTs Atropos)
   - Registration to MNI152
   - QC report generation

3. **DWI Preprocessing** (queued)
   - TOPUP distortion correction
   - GPU-accelerated eddy correction
   - DTI fitting
   - **Advanced models** (NEW - auto-enabled for multi-shell)
     - DKI (Diffusion Kurtosis Imaging)
     - NODDI (Neurite Orientation Dispersion and Density)
   - Spatial normalization to FMRIB58_FA

4. **Functional Preprocessing** (queued)
   - Motion correction
   - TEDANA multi-echo denoising
   - BBR registration to anatomical
   - ACompCor nuisance regression
   - Spatial normalization to MNI152

5. **ASL Preprocessing** (queued)
   - Motion correction
   - Label-control subtraction
   - CBF quantification with M0 calibration
   - Registration to anatomical
   - Optional normalization to MNI152

## Fixes Applied (Session 2025-11-13)

### Critical Bug Fixes
1. ✅ QC directory hierarchy (anat, ASL)
2. ✅ Transform centralization via TransformRegistry
3. ✅ DWI work directory structure
4. ✅ Functional workflow parameter error

### New Features
5. ✅ Advanced diffusion models integration (DKI, NODDI)

## Expected Runtime
- **Anatomical**: ~15-20 minutes
- **DWI** (with advanced models): ~45-60 minutes
- **Functional** (TEDANA): ~2-3 hours
- **ASL**: ~15-20 minutes

**Total Estimated Time**: 3-4 hours

## Monitoring Commands

### Check if pipeline is running
```bash
ps aux | grep run_continuous_pipeline | grep -v grep
```

### View latest log output
```bash
tail -50 logs/overnight_run_0580101.log
```

### Monitor progress
```bash
tail -f logs/overnight_run_0580101.log | grep -E "(✓|Step|complete|ERROR)"
```

### Check for errors
```bash
grep -E "ERROR|Failed|Traceback" logs/overnight_run_0580101.log | grep -v "FileNotFoundError.*tmpfile"
```

## Expected Output Structure

```
/mnt/bytopia/IRC805/
├── derivatives/IRC805-0580101/
│   ├── anat/
│   │   ├── brain.nii.gz
│   │   ├── segmentation/
│   │   └── transforms/
│   ├── dwi/
│   │   ├── dti/                # Standard DTI metrics
│   │   ├── dki/                # DKI metrics (NEW)
│   │   │   ├── mean_kurtosis.nii.gz
│   │   │   ├── axial_kurtosis.nii.gz
│   │   │   └── radial_kurtosis.nii.gz
│   │   └── noddi/              # NODDI metrics (NEW)
│   │       ├── ficvf.nii.gz    # Neurite density
│   │       ├── odi.nii.gz      # Orientation dispersion
│   │       └── fiso.nii.gz     # Isotropic fraction
│   ├── func/
│   │   └── preprocessed_bold.nii.gz
│   └── asl/
│       └── cbf_map.nii.gz
├── qc/IRC805-0580101/
│   ├── anat/
│   ├── dwi/
│   ├── func/
│   └── asl/
├── transforms/IRC805-0580101/
│   ├── ASL_to_T1w.mat
│   ├── func_to_T1w.mat
│   └── T1w_to_MNI152.nii.gz
└── work/IRC805-0580101/
    ├── anat_preprocess/
    ├── dwi_preprocess/
    ├── func_preprocess/
    └── asl_preprocess/
```

## Troubleshooting

If pipeline fails:
1. Check error log: `grep ERROR logs/overnight_run_0580101.log`
2. Review full traceback: `tail -100 logs/overnight_run_0580101.log`
3. Clean restart:
   ```bash
   rm -rf /mnt/bytopia/IRC805/work/IRC805-0580101
   source .venv/bin/activate
   python run_continuous_pipeline.py \
     --subject IRC805-0580101 \
     --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-0580101 \
     --study-root /mnt/bytopia/IRC805 \
     --config config.yaml \
     > logs/overnight_run_0580101.log 2>&1 &
   ```

## Next Session Checklist

- [ ] Check if pipeline completed successfully
- [ ] Review QC reports in `qc/IRC805-0580101/`
- [ ] Verify DKI/NODDI outputs exist and are valid
- [ ] Check all transforms in `transforms/IRC805-0580101/`
- [ ] Validate directory hierarchy matches specification
- [ ] Review any error messages in log
- [ ] Test pipeline on second subject if successful

## Git Status
- ✅ All changes committed (commit 3d66271)
- ✅ Pushed to origin/master
- ✅ Session report created: `SESSION_REPORT_2025-11-13.md`
