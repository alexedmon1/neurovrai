#!/usr/bin/env python3
"""
ASL (Arterial Spin Labeling) Preprocessing Workflow

Workflow features:
1. Motion correction (MCFLIRT)
2. Label-control separation and subtraction
3. Brain extraction
4. CBF quantification using standard kinetic model
5. M0 calibration with white matter reference (optional)
6. Partial volume correction (optional)
7. Tissue-specific CBF statistics
8. Registration to anatomical space
9. Spatial normalization to MNI152 (reuses anatomical transforms)
10. Quality control metrics

Key integrations:
- Reuses anatomical→MNI152 transforms for efficient normalization
- Uses tissue segmentation from anatomical workflow for tissue-specific CBF
- M0 calibration corrects for estimation bias (reduces elevated CBF)
- PVC corrects for partial volume averaging in low-resolution ASL
- Compatible with pCASL and PASL acquisition types
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any
import nibabel as nib
import numpy as np

from neurovrai.preprocess.utils.asl_cbf import (
    separate_label_control,
    compute_perfusion_weighted_signal,
    quantify_cbf,
    compute_tissue_specific_cbf,
    calibrate_cbf_with_wm_reference,
    apply_partial_volume_correction,
    save_asl_outputs
)
from neurovrai.preprocess.utils.dicom_asl_params import extract_asl_parameters
from neurovrai.preprocess.qc.asl_qc import (
    compute_asl_motion_qc,
    compute_cbf_qc,
    compute_perfusion_tsnr,
    generate_asl_qc_report
)
from neurovrai.utils.transforms import create_transform_registry

logger = logging.getLogger(__name__)


def run_asl_preprocessing(
    config: Dict[str, Any],
    subject: str,
    asl_file: Path,
    output_dir: Path,
    t1w_brain: Optional[Path] = None,
    gm_mask: Optional[Path] = None,
    wm_mask: Optional[Path] = None,
    csf_mask: Optional[Path] = None,
    dicom_dir: Optional[Path] = None,
    label_control_order: str = 'control_first',
    labeling_duration: float = 1.8,
    post_labeling_delay: float = 1.8,
    normalize_to_mni: bool = False
) -> Dict[str, Path]:
    """
    Run complete ASL preprocessing workflow.

    This function implements standard ASL preprocessing pipeline:
    1. Motion correction with MCFLIRT
    2. Label-control separation and subtraction
    3. Brain extraction
    4. CBF quantification
    5. Registration to anatomical space (optional)
    6. Spatial normalization to MNI152 (optional, reuses anatomical transforms)

    Parameters
    ----------
    config : dict
        Configuration dictionary with ASL-specific parameters
    subject : str
        Subject identifier
    asl_file : Path
        4D ASL time series (label-control pairs)
    output_dir : Path
        Study root directory (e.g., /mnt/bytopia/IRC805)
        Derivatives will be saved to: {output_dir}/derivatives/{subject}/asl/
    t1w_brain : Path, optional
        Brain-extracted T1w for registration
    gm_mask : Path, optional
        Gray matter mask in T1w space
    wm_mask : Path, optional
        White matter mask in T1w space
    csf_mask : Path, optional
        CSF mask in T1w space
    dicom_dir : Path, optional
        Directory containing ASL DICOM files for automatic parameter extraction
        If provided, will extract acquisition parameters from DICOM and override config
    label_control_order : str
        Label-control ordering ('control_first' or 'label_first')
    labeling_duration : float
        Labeling duration (τ) in seconds (default: 1.8s for pCASL)
        Overridden by DICOM if dicom_dir is provided
    post_labeling_delay : float
        Post-labeling delay (PLD) in seconds (default: 1.8s)
        Overridden by DICOM if dicom_dir is provided
    normalize_to_mni : bool
        Perform spatial normalization to MNI152 (default: False)

    Returns
    -------
    dict
        Dictionary with output file paths

    Examples
    --------
    >>> from pathlib import Path
    >>> results = run_asl_preprocessing(
    ...     config=config,
    ...     subject='IRC805-0580101',
    ...     asl_file=Path('asl_source.nii.gz'),
    ...     output_dir=Path('/mnt/bytopia/IRC805'),
    ...     labeling_duration=1.8,
    ...     post_labeling_delay=1.8
    ... )
    >>> print(results['cbf'])
    """
    logger.info("="*70)
    logger.info("ASL PREPROCESSING")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"ASL file: {asl_file}")
    logger.info("")

    # Setup directory structure
    # output_dir is the derivatives base (e.g., /mnt/bytopia/IRC805/derivatives)
    # Use standardized hierarchy: {outdir}/{subject}/{modality}/
    outdir = Path(output_dir)
    derivatives_dir = outdir / subject / 'asl'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    # Derive study root from output_dir (derivatives directory)
    # output_dir is derivatives, so study_root is one level up
    study_root = outdir.parent

    # Work directory
    work_dir = study_root / 'work' / subject / 'asl_preprocess'
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Study root: {study_root}")
    logger.info(f"Derivatives output: {derivatives_dir}")
    logger.info(f"Working directory: {work_dir}")
    logger.info("")

    # Extract BET parameters from config
    asl_config = config.get('asl', {})
    bet_frac = asl_config.get('bet', {}).get('frac', 0.3)  # Default: 0.3 for ASL

    # Automatic DICOM parameter extraction (if available)
    dicom_params = None
    if dicom_dir and dicom_dir.exists():
        logger.info("Attempting to extract ASL parameters from DICOM...")
        logger.info(f"  DICOM directory: {dicom_dir}")

        try:
            # Find ASL DICOM files
            import glob
            dicom_files = list(dicom_dir.glob('*.dcm')) + list(dicom_dir.glob('*.DCM'))

            if dicom_files:
                # Extract from first DICOM file
                dicom_file = dicom_files[0]
                logger.info(f"  Using DICOM file: {dicom_file.name}")

                dicom_params = extract_asl_parameters(dicom_file)

                # Override parameters with DICOM values if available
                param_sources = {}

                if 'labeling_duration' in dicom_params:
                    labeling_duration = dicom_params['labeling_duration']
                    param_sources['labeling_duration'] = 'DICOM'
                else:
                    param_sources['labeling_duration'] = 'config'

                if 'post_labeling_delay' in dicom_params:
                    post_labeling_delay = dicom_params['post_labeling_delay']
                    param_sources['post_labeling_delay'] = 'DICOM'
                else:
                    param_sources['post_labeling_delay'] = 'config'

                if 'label_control_order' in dicom_params:
                    label_control_order = dicom_params['label_control_order']
                    param_sources['label_control_order'] = 'DICOM'
                else:
                    param_sources['label_control_order'] = 'config'

                logger.info("")
                logger.info("ASL Acquisition Parameters:")
                logger.info(f"  Labeling duration (τ): {labeling_duration:.3f} s [{param_sources['labeling_duration']}]")
                logger.info(f"  Post-labeling delay (PLD): {post_labeling_delay:.3f} s [{param_sources['post_labeling_delay']}]")
                logger.info(f"  Label-control order: {label_control_order} [{param_sources['label_control_order']}]")
                logger.info("")
            else:
                logger.warning("  No DICOM files found in directory")
                logger.info("  Using parameters from config")
                logger.info("")
        except Exception as e:
            logger.warning(f"  Failed to extract DICOM parameters: {e}")
            logger.info("  Using parameters from config")
            logger.info("")
    else:
        logger.info("ASL Acquisition Parameters (from config):")
        logger.info(f"  Labeling duration (τ): {labeling_duration:.3f} s")
        logger.info(f"  Post-labeling delay (PLD): {post_labeling_delay:.3f} s")
        logger.info(f"  Label-control order: {label_control_order}")
        logger.info("")

    results = {}

    # Step 1: Motion correction
    logger.info("Step 1: Motion correction (MCFLIRT)")
    logger.info("  Registering all volumes to middle volume...")

    mcflirt_output = work_dir / 'asl_mcf.nii.gz'
    # MCFLIRT appends .par to the output name, so if we use 'asl_mcf' it creates 'asl_mcf.nii.par'
    mcflirt_params = work_dir / 'asl_mcf.nii.par'

    mcflirt_cmd = [
        'mcflirt',
        '-in', str(asl_file),
        '-out', str(mcflirt_output.with_suffix('')),  # mcflirt adds .nii.gz
        '-plots',
        '-report'
    ]

    subprocess.run(mcflirt_cmd, check=True, capture_output=True)
    logger.info(f"  Motion corrected: {mcflirt_output}")
    logger.info(f"  Motion parameters: {mcflirt_params}")
    results['motion_corrected'] = mcflirt_output
    results['motion_params'] = mcflirt_params
    logger.info("")

    # Load motion-corrected data
    logger.info("Step 2: Label-control separation and subtraction")
    asl_img = nib.load(mcflirt_output)
    asl_data = asl_img.get_fdata()
    affine = asl_img.affine
    header = asl_img.header

    logger.info(f"  ASL data shape: {asl_data.shape}")
    logger.info(f"  Label-control order: {label_control_order}")

    # Separate label and control
    control_volumes, label_volumes = separate_label_control(
        asl_data,
        order=label_control_order
    )

    # Compute perfusion-weighted signal
    delta_m_4d, delta_m_mean = compute_perfusion_weighted_signal(
        control_volumes,
        label_volumes,
        method='simple'
    )

    # Save 4D perfusion-weighted signal for tSNR calculation
    perfusion_4d_file = work_dir / 'perfusion_4d.nii.gz'
    nib.save(nib.Nifti1Image(delta_m_4d, affine, header), perfusion_4d_file)
    results['perfusion_4d'] = perfusion_4d_file

    # Compute mean control and label
    control_mean = np.mean(control_volumes, axis=3)
    label_mean = np.mean(label_volumes, axis=3)

    logger.info("")

    # Step 3: Brain extraction
    logger.info("Step 3: Brain extraction")
    logger.info("  Extracting brain from mean control image...")

    # Save mean control for BET
    control_mean_file = work_dir / 'control_mean.nii.gz'
    nib.save(nib.Nifti1Image(control_mean, affine, header), control_mean_file)

    # Run BET on mean control image
    brain_mask_file = work_dir / 'brain_mask.nii.gz'

    bet_cmd = [
        'bet',
        str(control_mean_file),
        str(work_dir / 'brain'),
        '-f', str(bet_frac),  # Configurable - ASL has low contrast
        '-m',  # Generate mask
        '-R'   # Robust
    ]

    subprocess.run(bet_cmd, check=True, capture_output=True)
    logger.info(f"  Brain mask: {brain_mask_file}")
    results['brain_mask'] = brain_mask_file

    # Load mask
    brain_mask = nib.load(brain_mask_file).get_fdata()
    n_brain_voxels = np.sum(brain_mask > 0)
    logger.info(f"  Brain voxels: {n_brain_voxels}")
    logger.info("")

    # Step 4: CBF quantification
    logger.info("Step 4: CBF quantification")
    logger.info("  Using standard single-compartment kinetic model")

    # Use mean control as M0 (equilibrium magnetization)
    m0 = control_mean

    # Quantify CBF
    cbf = quantify_cbf(
        delta_m=delta_m_mean,
        m0=m0,
        labeling_duration=labeling_duration,
        post_labeling_delay=post_labeling_delay,
        labeling_efficiency=config.get('labeling_efficiency', 0.85),
        t1_blood=config.get('t1_blood', 1.65),
        blood_brain_partition=config.get('blood_brain_partition', 0.9),
        mask=brain_mask
    )

    logger.info("")

    # Step 5: M0 Calibration with White Matter Reference (if WM mask provided)
    cbf_calibrated = None
    calibration_info = None
    if wm_mask and config.get('asl', {}).get('apply_m0_calibration', True):
        logger.info("Step 5: M0 Calibration with White Matter Reference")
        logger.info("  Correcting for M0 estimation bias...")
        logger.info("  Resampling tissue masks to ASL space...")

        # Resample WM mask from anatomical to ASL space
        wm_mask_asl = work_dir / 'wm_mask_asl.nii.gz'
        flirt_resample_cmd = [
            'flirt',
            '-in', str(wm_mask),
            '-ref', str(control_mean_file),  # ASL reference
            '-out', str(wm_mask_asl),
            '-applyxfm',  # Apply identity transform (just resample)
            '-interp', 'nearestneighbour'  # Preserve mask values
        ]
        subprocess.run(flirt_resample_cmd, check=True, capture_output=True)
        logger.info(f"  WM mask resampled: {wm_mask_asl}")

        # Load resampled WM mask
        wm_data = nib.load(wm_mask_asl).get_fdata()

        try:
            cbf_calibrated, calibration_info = calibrate_cbf_with_wm_reference(
                cbf=cbf,
                wm_mask=wm_data,
                wm_cbf_expected=config.get('asl', {}).get('wm_cbf_reference', 25.0),
                wm_threshold=0.7
            )

            logger.info(f"  Calibration successful:")
            logger.info(f"    Measured WM CBF: {calibration_info['wm_cbf_measured']:.2f} ml/100g/min")
            logger.info(f"    Expected WM CBF: {calibration_info['wm_cbf_expected']:.2f} ml/100g/min")
            logger.info(f"    Scaling factor: {calibration_info['scaling_factor']:.3f}")
            logger.info(f"    Calibrated WM CBF: {calibration_info['wm_cbf_calibrated']:.2f} ml/100g/min")
            logger.info("")

        except Exception as e:
            logger.warning(f"  M0 calibration failed: {e}")
            logger.warning("  Proceeding with uncalibrated CBF")
            logger.info("")

    # Step 6: Partial Volume Correction (if enabled and tissue masks provided)
    cbf_gm_pvc = None
    cbf_wm_pvc = None
    if config.get('asl', {}).get('apply_pvc', False) and gm_mask and wm_mask and csf_mask:
        logger.info("Step 6: Partial Volume Correction")
        logger.info("  Applying linear regression PVC method...")
        logger.info("  Resampling tissue masks to ASL space...")

        # Resample tissue masks from anatomical to ASL space
        gm_mask_asl = work_dir / 'gm_mask_asl.nii.gz'
        csf_mask_asl = work_dir / 'csf_mask_asl.nii.gz'

        # Resample GM mask
        flirt_gm_cmd = [
            'flirt', '-in', str(gm_mask), '-ref', str(control_mean_file),
            '-out', str(gm_mask_asl), '-applyxfm', '-interp', 'nearestneighbour'
        ]
        subprocess.run(flirt_gm_cmd, check=True, capture_output=True)

        # Resample CSF mask
        flirt_csf_cmd = [
            'flirt', '-in', str(csf_mask), '-ref', str(control_mean_file),
            '-out', str(csf_mask_asl), '-applyxfm', '-interp', 'nearestneighbour'
        ]
        subprocess.run(flirt_csf_cmd, check=True, capture_output=True)

        # Load resampled tissue masks (wm_data may already be loaded from calibration)
        gm_data = nib.load(gm_mask_asl).get_fdata()
        if 'wm_data' not in locals():  # Load if not already loaded in calibration
            wm_mask_asl = work_dir / 'wm_mask_asl.nii.gz'
            flirt_wm_cmd = [
                'flirt', '-in', str(wm_mask), '-ref', str(control_mean_file),
                '-out', str(wm_mask_asl), '-applyxfm', '-interp', 'nearestneighbour'
            ]
            subprocess.run(flirt_wm_cmd, check=True, capture_output=True)
            wm_data = nib.load(wm_mask_asl).get_fdata()
        csf_data = nib.load(csf_mask_asl).get_fdata()

        # Use calibrated CBF if available, otherwise use uncalibrated
        cbf_to_correct = cbf_calibrated if cbf_calibrated is not None else cbf

        try:
            cbf_gm_pvc, cbf_wm_pvc = apply_partial_volume_correction(
                cbf=cbf_to_correct,
                gm_pve=gm_data,
                wm_pve=wm_data,
                csf_pve=csf_data,
                brain_mask=brain_mask
            )

            # Compute statistics for PVC-corrected maps
            gm_high_conf = gm_data > 0.7
            wm_high_conf = wm_data > 0.7

            gm_cbf_pvc_mean = np.mean(cbf_gm_pvc[gm_high_conf])
            wm_cbf_pvc_mean = np.mean(cbf_wm_pvc[wm_high_conf])

            logger.info(f"  PVC-corrected GM CBF: {gm_cbf_pvc_mean:.2f} ml/100g/min")
            logger.info(f"  PVC-corrected WM CBF: {wm_cbf_pvc_mean:.2f} ml/100g/min")
            logger.info("")

        except Exception as e:
            logger.warning(f"  Partial volume correction failed: {e}")
            logger.warning("  Proceeding without PVC")
            logger.info("")

    # Save outputs
    logger.info("Saving ASL outputs...")
    saved_files = save_asl_outputs(
        output_dir=derivatives_dir,
        subject=subject,
        control_mean=control_mean,
        label_mean=label_mean,
        delta_m_mean=delta_m_mean,
        cbf=cbf,
        affine=affine,
        header=header
    )
    results.update(saved_files)

    # Save calibrated CBF if available
    if cbf_calibrated is not None:
        cbf_calibrated_file = derivatives_dir / f'{subject}_cbf_calibrated.nii.gz'
        nib.save(nib.Nifti1Image(cbf_calibrated, affine, header), cbf_calibrated_file)
        results['cbf_calibrated'] = cbf_calibrated_file
        results['calibration_info'] = calibration_info
        logger.info(f"  CBF (calibrated): {cbf_calibrated_file}")

    # Save PVC-corrected CBF maps if available
    if cbf_gm_pvc is not None:
        cbf_gm_pvc_file = derivatives_dir / f'{subject}_cbf_gm_pvc.nii.gz'
        nib.save(nib.Nifti1Image(cbf_gm_pvc, affine, header), cbf_gm_pvc_file)
        results['cbf_gm_pvc'] = cbf_gm_pvc_file
        logger.info(f"  CBF GM (PVC): {cbf_gm_pvc_file}")

    if cbf_wm_pvc is not None:
        cbf_wm_pvc_file = derivatives_dir / f'{subject}_cbf_wm_pvc.nii.gz'
        nib.save(nib.Nifti1Image(cbf_wm_pvc, affine, header), cbf_wm_pvc_file)
        results['cbf_wm_pvc'] = cbf_wm_pvc_file
        logger.info(f"  CBF WM (PVC): {cbf_wm_pvc_file}")

    # Copy brain mask to derivatives
    brain_mask_out = derivatives_dir / f'{subject}_brain_mask.nii.gz'
    nib.save(nib.Nifti1Image(brain_mask, affine, header), brain_mask_out)
    results['brain_mask'] = brain_mask_out
    logger.info("")

    # Step 7: Tissue-specific CBF statistics (if masks provided)
    if gm_mask and wm_mask and csf_mask:
        logger.info("Step 7: Computing tissue-specific CBF statistics")

        # Check if resampled masks exist from previous steps, otherwise resample now
        gm_mask_asl = work_dir / 'gm_mask_asl.nii.gz'
        wm_mask_asl = work_dir / 'wm_mask_asl.nii.gz'
        csf_mask_asl = work_dir / 'csf_mask_asl.nii.gz'

        if not gm_mask_asl.exists():
            logger.info("  Resampling tissue masks to ASL space...")
            for mask_file, mask_asl in [(gm_mask, gm_mask_asl), (wm_mask, wm_mask_asl), (csf_mask, csf_mask_asl)]:
                flirt_cmd = [
                    'flirt', '-in', str(mask_file), '-ref', str(control_mean_file),
                    '-out', str(mask_asl), '-applyxfm', '-interp', 'nearestneighbour'
                ]
                subprocess.run(flirt_cmd, check=True, capture_output=True)

        # Load resampled tissue masks
        gm_data = nib.load(gm_mask_asl).get_fdata()
        wm_data = nib.load(wm_mask_asl).get_fdata()
        csf_data = nib.load(csf_mask_asl).get_fdata()

        # Compute tissue-specific statistics
        tissue_cbf = compute_tissue_specific_cbf(
            cbf=cbf,
            gm_mask=gm_data,
            wm_mask=wm_data,
            csf_mask=csf_data
        )

        results['tissue_cbf'] = tissue_cbf
        logger.info("")

    # Step 8: Registration to anatomical space (if T1w provided)
    if t1w_brain:
        logger.info("Step 8: Registration to anatomical space")
        logger.info("  Registering mean control to T1w...")

        # Create temp transform file
        import tempfile
        temp_mat = Path(tempfile.mktemp(suffix='_asl_to_anat.mat'))

        flirt_cmd = [
            'flirt',
            '-in', str(control_mean_file),
            '-ref', str(t1w_brain),
            '-omat', str(temp_mat),
            '-dof', '6',
            '-cost', 'corratio'  # Good for different contrasts
        ]

        subprocess.run(flirt_cmd, check=True, capture_output=True)

        # Save to TransformRegistry
        registry = create_transform_registry(config, subject)
        asl_to_anat_mat = registry.save_linear_transform(
            transform_file=temp_mat,
            source_space='ASL',
            target_space='T1w',
            source_image=control_mean_file,
            reference=t1w_brain
        )
        temp_mat.unlink()  # Clean up temp file

        logger.info(f"  ASL→anatomical transform: {asl_to_anat_mat}")
        results['asl_to_anat'] = asl_to_anat_mat
        logger.info("")

        # Apply transform to CBF map
        logger.info("  Transforming CBF to anatomical space...")
        cbf_anat = derivatives_dir / f'{subject}_cbf_anat.nii.gz'

        flirt_apply_cmd = [
            'flirt',
            '-in', str(results['cbf']),
            '-ref', str(t1w_brain),
            '-applyxfm',
            '-init', str(asl_to_anat_mat),
            '-out', str(cbf_anat)
        ]

        subprocess.run(flirt_apply_cmd, check=True, capture_output=True)
        logger.info(f"  CBF in anatomical space: {cbf_anat}")
        results['cbf_anat'] = cbf_anat
        logger.info("")

    # Step 9: Spatial normalization to MNI152 (optional)
    if normalize_to_mni and t1w_brain and asl_to_anat_mat:
        logger.info("Step 9: Spatial normalization to MNI152")

        # Get anatomical transforms from TransformRegistry
        registry = create_transform_registry(config, subject)
        anat_transforms = registry.get_nonlinear_transform('T1w', 'MNI152')

        if anat_transforms:
            t1w_to_mni_warp, t1w_to_mni_affine = anat_transforms
            logger.info("  Reusing anatomical transforms for normalization")
            logger.info(f"  ASL→anat: {asl_to_anat_mat}")
            logger.info(f"  Anat→MNI affine: {t1w_to_mni_affine}")
            logger.info(f"  Anat→MNI warp: {t1w_to_mni_warp}")

            # Concatenate transforms
            import os
            fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')
            mni_template = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm.nii.gz'

            # Create temp file for concatenated warp
            import tempfile
            temp_warp = Path(tempfile.mktemp(suffix='_asl_to_mni_warp.nii.gz'))

            convertwarp_cmd = [
                'convertwarp',
                '--ref=' + str(mni_template),
                '--premat=' + str(asl_to_anat_mat),
                '--warp1=' + str(t1w_to_mni_warp),
                '--out=' + str(temp_warp)
            ]

            subprocess.run(convertwarp_cmd, check=True, capture_output=True)

            # Save concatenated warp to TransformRegistry
            # Note: This is a composite warp, so we save it as a special case
            # For now, just copy to registry directory manually since it's composite
            registry_dir = registry.subject_dir
            asl_to_mni_warp = registry_dir / f'ASL_to_MNI152_warp.nii.gz'
            import shutil
            shutil.copy2(temp_warp, asl_to_mni_warp)
            temp_warp.unlink()

            logger.info(f"  Concatenated warp: {asl_to_mni_warp}")

            # Apply to CBF
            normalized_dir = derivatives_dir / 'normalized'
            normalized_dir.mkdir(parents=True, exist_ok=True)

            cbf_mni = normalized_dir / f'{subject}_cbf_normalized.nii.gz'

            applywarp_cmd = [
                'applywarp',
                '--in=' + str(results['cbf']),
                '--ref=' + str(mni_template),
                '--warp=' + str(asl_to_mni_warp),
                '--out=' + str(cbf_mni),
                '--interp=trilinear'
            ]

            subprocess.run(applywarp_cmd, check=True, capture_output=True)
            logger.info(f"  CBF in MNI152 space: {cbf_mni}")
            results['cbf_normalized'] = cbf_mni
            logger.info("")
        else:
            logger.warning("  Anatomical transforms not found - skipping normalization")
            logger.warning("  Run anatomical preprocessing first to generate transforms")
            logger.info("")

    # Step 10: Quality Control (optional)
    if config.get('run_qc', True):
        logger.info("Step 10: Quality Control")

        # QC goes to study-level QC directory: {study_root}/qc/{subject}/asl/
        qc_dir = outdir.parent / 'qc' / subject / 'asl'
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Motion QC
        logger.info("  Computing motion QC metrics...")
        motion_qc = compute_asl_motion_qc(
            motion_file=results['motion_params'],
            output_dir=qc_dir,
            fd_threshold=0.5
        )

        # CBF QC
        logger.info("  Computing CBF QC metrics...")

        # Use resampled masks for QC if they exist, otherwise use original (will cause error)
        gm_mask_for_qc = work_dir / 'gm_mask_asl.nii.gz' if (work_dir / 'gm_mask_asl.nii.gz').exists() else gm_mask
        wm_mask_for_qc = work_dir / 'wm_mask_asl.nii.gz' if (work_dir / 'wm_mask_asl.nii.gz').exists() else wm_mask
        csf_mask_for_qc = work_dir / 'csf_mask_asl.nii.gz' if (work_dir / 'csf_mask_asl.nii.gz').exists() else csf_mask

        cbf_qc = compute_cbf_qc(
            cbf_file=results['cbf'],
            mask_file=results['brain_mask'],
            output_dir=qc_dir,
            gm_mask=gm_mask_for_qc if gm_mask else None,
            wm_mask=wm_mask_for_qc if wm_mask else None,
            csf_mask=csf_mask_for_qc if csf_mask else None
        )

        # Perfusion tSNR (if we have the 4D perfusion file)
        tsnr_qc = None
        perfusion_4d_file = work_dir / 'perfusion_4d.nii.gz'
        if perfusion_4d_file.exists():
            logger.info("  Computing perfusion tSNR...")
            tsnr_qc = compute_perfusion_tsnr(
                perfusion_file=perfusion_4d_file,
                mask_file=results['brain_mask'],
                output_dir=qc_dir
            )

        # Generate HTML QC report
        logger.info("  Generating QC report...")
        qc_report = qc_dir / f'{subject}_asl_qc_report.html'
        generate_asl_qc_report(
            subject=subject,
            motion_metrics=motion_qc,
            cbf_metrics=cbf_qc,
            tsnr_metrics=tsnr_qc,
            output_file=qc_report
        )

        results['qc_report'] = qc_report
        results['qc_dir'] = qc_dir
        logger.info(f"  QC report: {qc_report}")
        logger.info("")

    logger.info("="*70)
    logger.info("ASL PREPROCESSING COMPLETE")
    logger.info("="*70)
    logger.info("")
    logger.info("Key outputs:")
    logger.info(f"  CBF map: {results['cbf']}")
    if 'cbf_anat' in results:
        logger.info(f"  CBF (anatomical space): {results['cbf_anat']}")
    if 'cbf_normalized' in results:
        logger.info(f"  CBF (MNI152 space): {results['cbf_normalized']}")
    logger.info("")

    return results
