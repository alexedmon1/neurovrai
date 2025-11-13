#!/usr/bin/env python3
"""
ASL (Arterial Spin Labeling) Preprocessing Workflow

Workflow features:
1. Motion correction (MCFLIRT)
2. Label-control separation and subtraction
3. Perfusion-weighted signal (ΔM) computation
4. Brain extraction
5. CBF quantification using standard kinetic model
6. Registration to anatomical space
7. Spatial normalization to MNI152 (reuses anatomical transforms)
8. Quality control metrics

Key integrations:
- Reuses anatomical→MNI152 transforms for efficient normalization
- Uses tissue segmentation from anatomical workflow for tissue-specific CBF
- Compatible with pCASL and PASL acquisition types
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any
import nibabel as nib
import numpy as np

from mri_preprocess.utils.asl_cbf import (
    separate_label_control,
    compute_perfusion_weighted_signal,
    quantify_cbf,
    compute_tissue_specific_cbf,
    save_asl_outputs
)
from mri_preprocess.qc.asl_qc import (
    compute_asl_motion_qc,
    compute_cbf_qc,
    compute_perfusion_tsnr,
    generate_asl_qc_report
)

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
    label_control_order : str
        Label-control ordering ('control_first' or 'label_first')
    labeling_duration : float
        Labeling duration (τ) in seconds (default: 1.8s for pCASL)
    post_labeling_delay : float
        Post-labeling delay (PLD) in seconds (default: 1.8s)
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
    outdir = Path(output_dir)
    derivatives_dir = outdir / 'derivatives' / subject / 'asl'
    derivatives_dir.mkdir(parents=True, exist_ok=True)

    # Work directory
    work_dir = outdir / 'work' / subject / 'asl_preprocess'
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Derivatives output: {derivatives_dir}")
    logger.info(f"Working directory: {work_dir}")
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
        '-f', '0.3',  # Aggressive for low-intensity ASL
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

    # Copy brain mask to derivatives
    brain_mask_out = derivatives_dir / f'{subject}_brain_mask.nii.gz'
    nib.save(nib.Nifti1Image(brain_mask, affine, header), brain_mask_out)
    results['brain_mask'] = brain_mask_out
    logger.info("")

    # Step 5: Tissue-specific CBF (if masks provided)
    if gm_mask and wm_mask and csf_mask:
        logger.info("Step 5: Computing tissue-specific CBF")

        # Load tissue masks
        gm_data = nib.load(gm_mask).get_fdata()
        wm_data = nib.load(wm_mask).get_fdata()
        csf_data = nib.load(csf_mask).get_fdata()

        # Compute tissue-specific statistics
        tissue_cbf = compute_tissue_specific_cbf(
            cbf=cbf,
            gm_mask=gm_data,
            wm_mask=wm_data,
            csf_mask=csf_data
        )

        results['tissue_cbf'] = tissue_cbf
        logger.info("")

    # Step 6: Registration to anatomical space (if T1w provided)
    if t1w_brain:
        logger.info("Step 6: Registration to anatomical space")
        logger.info("  Registering mean control to T1w...")

        transforms_dir = derivatives_dir / 'transforms'
        transforms_dir.mkdir(parents=True, exist_ok=True)

        asl_to_anat_mat = transforms_dir / f'{subject}_asl_to_anat.mat'

        flirt_cmd = [
            'flirt',
            '-in', str(control_mean_file),
            '-ref', str(t1w_brain),
            '-omat', str(asl_to_anat_mat),
            '-dof', '6',
            '-cost', 'corratio'  # Good for different contrasts
        ]

        subprocess.run(flirt_cmd, check=True, capture_output=True)
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

    # Step 7: Spatial normalization to MNI152 (optional)
    if normalize_to_mni and t1w_brain and asl_to_anat_mat:
        logger.info("Step 7: Spatial normalization to MNI152")

        # Get anatomical transforms
        anat_derivatives_dir = outdir / 'derivatives' / subject / 'anat'
        anat_transforms_dir = anat_derivatives_dir / 'transforms'

        t1w_to_mni_affine = anat_transforms_dir / f'{subject}_T1w_to_MNI152_affine.mat'
        t1w_to_mni_warp = anat_transforms_dir / f'{subject}_T1w_to_MNI152_warp.nii.gz'

        if t1w_to_mni_affine.exists() and t1w_to_mni_warp.exists():
            logger.info("  Reusing anatomical transforms for normalization")
            logger.info(f"  ASL→anat: {asl_to_anat_mat}")
            logger.info(f"  Anat→MNI affine: {t1w_to_mni_affine}")
            logger.info(f"  Anat→MNI warp: {t1w_to_mni_warp}")

            # Concatenate transforms
            import os
            fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')
            mni_template = Path(fsldir) / 'data' / 'standard' / 'MNI152_T1_2mm.nii.gz'

            asl_to_mni_warp = transforms_dir / f'{subject}_asl_to_MNI152_warp.nii.gz'

            convertwarp_cmd = [
                'convertwarp',
                '--ref=' + str(mni_template),
                '--premat=' + str(asl_to_anat_mat),
                '--warp1=' + str(t1w_to_mni_warp),
                '--out=' + str(asl_to_mni_warp)
            ]

            subprocess.run(convertwarp_cmd, check=True, capture_output=True)
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

    # Step 8: Quality Control (optional)
    if config.get('run_qc', True):
        logger.info("Step 8: Quality Control")

        qc_dir = derivatives_dir / 'qc'
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
        cbf_qc = compute_cbf_qc(
            cbf_file=results['cbf'],
            mask_file=results['brain_mask'],
            output_dir=qc_dir,
            gm_mask=gm_mask if gm_mask else None,
            wm_mask=wm_mask if wm_mask else None,
            csf_mask=csf_mask if csf_mask else None
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
