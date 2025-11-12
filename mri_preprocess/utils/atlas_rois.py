#!/usr/bin/env python3
"""
Atlas-based ROI extraction utilities for tractography.

This module provides functions to:
- Warp standard atlases to subject DWI space
- Extract specific ROIs as binary masks
- Prepare seed/target masks for probtrackx2
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import subprocess
import nibabel as nib
import numpy as np


# Atlas definitions
HARVARD_OXFORD_CORTICAL_LABELS = {
    'frontal_pole_l': 1,
    'frontal_pole_r': 2,
    'insular_cortex_l': 3,
    'insular_cortex_r': 4,
    'superior_frontal_gyrus_l': 5,
    'superior_frontal_gyrus_r': 6,
    'middle_frontal_gyrus_l': 7,
    'middle_frontal_gyrus_r': 8,
    'inferior_frontal_gyrus_pars_triangularis_l': 9,
    'inferior_frontal_gyrus_pars_triangularis_r': 10,
    'inferior_frontal_gyrus_pars_opercularis_l': 11,
    'inferior_frontal_gyrus_pars_opercularis_r': 12,
    'precentral_gyrus_l': 13,
    'precentral_gyrus_r': 14,
    'temporal_pole_l': 15,
    'temporal_pole_r': 16,
    'superior_temporal_gyrus_anterior_l': 17,
    'superior_temporal_gyrus_anterior_r': 18,
    'superior_temporal_gyrus_posterior_l': 19,
    'superior_temporal_gyrus_posterior_r': 20,
    'middle_temporal_gyrus_anterior_l': 21,
    'middle_temporal_gyrus_anterior_r': 22,
    'middle_temporal_gyrus_posterior_l': 23,
    'middle_temporal_gyrus_posterior_r': 24,
    'middle_temporal_gyrus_temporooccipital_l': 25,
    'middle_temporal_gyrus_temporooccipital_r': 26,
    'inferior_temporal_gyrus_anterior_l': 27,
    'inferior_temporal_gyrus_anterior_r': 28,
    'inferior_temporal_gyrus_posterior_l': 29,
    'inferior_temporal_gyrus_posterior_r': 30,
    'inferior_temporal_gyrus_temporooccipital_l': 31,
    'inferior_temporal_gyrus_temporooccipital_r': 32,
    'postcentral_gyrus_l': 33,
    'postcentral_gyrus_r': 34,
    'superior_parietal_lobule_l': 35,
    'superior_parietal_lobule_r': 36,
    'supramarginal_gyrus_anterior_l': 37,
    'supramarginal_gyrus_anterior_r': 38,
    'supramarginal_gyrus_posterior_l': 39,
    'supramarginal_gyrus_posterior_r': 40,
    'angular_gyrus_l': 41,
    'angular_gyrus_r': 42,
    'lateral_occipital_cortex_superior_l': 43,
    'lateral_occipital_cortex_superior_r': 44,
    'lateral_occipital_cortex_inferior_l': 45,
    'lateral_occipital_cortex_inferior_r': 46,
    'intracalcarine_cortex_l': 47,
    'intracalcarine_cortex_r': 48,
    'frontal_medial_cortex_l': 49,
    'frontal_medial_cortex_r': 50,
}

HARVARD_OXFORD_SUBCORTICAL_LABELS = {
    'cerebral_white_matter_l': 1,
    'cerebral_white_matter_r': 2,
    'cerebral_cortex_l': 3,
    'cerebral_cortex_r': 4,
    'lateral_ventricle_l': 5,
    'lateral_ventricle_r': 6,
    'thalamus_l': 7,
    'thalamus_r': 8,
    'caudate_l': 9,
    'caudate_r': 10,
    'putamen_l': 11,
    'putamen_r': 12,
    'pallidum_l': 13,
    'pallidum_r': 14,
    'brain_stem': 15,
    'hippocampus_l': 16,
    'hippocampus_r': 17,
    'amygdala_l': 18,
    'amygdala_r': 19,
    'accumbens_l': 20,
    'accumbens_r': 21,
}

JHU_WM_LABELS = {
    'middle_cerebellar_peduncle': 1,
    'pontine_crossing_tract': 2,
    'genu_of_corpus_callosum': 3,
    'body_of_corpus_callosum': 4,
    'splenium_of_corpus_callosum': 5,
    'fornix': 6,
    'corticospinal_tract_r': 7,
    'corticospinal_tract_l': 8,
    'medial_lemniscus_r': 9,
    'medial_lemniscus_l': 10,
    'inferior_cerebellar_peduncle_r': 11,
    'inferior_cerebellar_peduncle_l': 12,
    'superior_cerebellar_peduncle_r': 13,
    'superior_cerebellar_peduncle_l': 14,
    'cerebral_peduncle_r': 15,
    'cerebral_peduncle_l': 16,
    'anterior_limb_of_internal_capsule_r': 17,
    'anterior_limb_of_internal_capsule_l': 18,
    'posterior_limb_of_internal_capsule_r': 19,
    'posterior_limb_of_internal_capsule_l': 20,
    'retrolenticular_part_of_internal_capsule_r': 21,
    'retrolenticular_part_of_internal_capsule_l': 22,
    'anterior_corona_radiata_r': 23,
    'anterior_corona_radiata_l': 24,
    'superior_corona_radiata_r': 25,
    'superior_corona_radiata_l': 26,
    'posterior_corona_radiata_r': 27,
    'posterior_corona_radiata_l': 28,
    'posterior_thalamic_radiation_r': 29,
    'posterior_thalamic_radiation_l': 30,
    'sagittal_stratum_r': 31,
    'sagittal_stratum_l': 32,
    'external_capsule_r': 33,
    'external_capsule_l': 34,
    'cingulum_cingulate_gyrus_r': 35,
    'cingulum_cingulate_gyrus_l': 36,
    'cingulum_hippocampus_r': 37,
    'cingulum_hippocampus_l': 38,
    'fornix_stria_terminalis_r': 39,
    'fornix_stria_terminalis_l': 40,
    'superior_longitudinal_fasciculus_r': 41,
    'superior_longitudinal_fasciculus_l': 42,
    'superior_fronto_occipital_fasciculus_r': 43,
    'superior_fronto_occipital_fasciculus_l': 44,
    'uncinate_fasciculus_r': 45,
    'uncinate_fasciculus_l': 46,
    'tapetum_r': 47,
    'tapetum_l': 48,
}


def get_fsl_atlas_path(atlas_name: str) -> Path:
    """
    Get path to FSL atlas.

    Parameters
    ----------
    atlas_name : str
        Atlas name. Options:
        - 'HarvardOxford-cortical'
        - 'HarvardOxford-subcortical'
        - 'JHU-ICBM-labels-1mm'
        - 'JHU-ICBM-labels-2mm'
        - 'JHU-ICBM-tracts-maxprob-1mm'
        - 'JHU-ICBM-tracts-maxprob-2mm'

    Returns
    -------
    Path
        Path to atlas file
    """
    import os
    fsldir = os.environ.get('FSLDIR', '/usr/local/fsl')
    atlas_dir = Path(fsldir) / 'data' / 'atlases'

    atlas_files = {
        'HarvardOxford-cortical': 'HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz',
        'HarvardOxford-subcortical': 'HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz',
        'JHU-ICBM-labels-1mm': 'JHU/JHU-ICBM-labels-1mm.nii.gz',
        'JHU-ICBM-labels-2mm': 'JHU/JHU-ICBM-labels-2mm.nii.gz',
        'JHU-ICBM-tracts-1mm': 'JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz',
        'JHU-ICBM-tracts-2mm': 'JHU/JHU-ICBM-tracts-maxprob-thr25-2mm.nii.gz',
    }

    if atlas_name not in atlas_files:
        raise ValueError(
            f"Unknown atlas: {atlas_name}. "
            f"Available: {list(atlas_files.keys())}"
        )

    atlas_path = atlas_dir / atlas_files[atlas_name]

    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    return atlas_path


def warp_atlas_to_dwi(
    atlas_file: Path,
    dwi_reference: Path,
    mni_to_dwi_warp: Optional[Path] = None,
    output_file: Optional[Path] = None,
    interpolation: str = 'nn'
) -> Path:
    """
    Warp atlas from MNI space to DWI space.

    Parameters
    ----------
    atlas_file : Path
        Atlas in MNI space
    dwi_reference : Path
        Reference DWI volume (typically b0 or FA map)
    mni_to_dwi_warp : Path, optional
        Warp field from MNI to DWI space. If None, uses linear registration.
    output_file : Path, optional
        Output file path. If None, creates one automatically.
    interpolation : str
        Interpolation method: 'nn' (nearest neighbor) for labels, 'trilinear' for continuous

    Returns
    -------
    Path
        Path to warped atlas in DWI space
    """
    if output_file is None:
        output_file = Path(str(atlas_file).replace('.nii.gz', '_dwi.nii.gz'))

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if mni_to_dwi_warp and Path(mni_to_dwi_warp).exists():
        # Use nonlinear warp
        cmd = [
            'applywarp',
            '--in=' + str(atlas_file),
            '--ref=' + str(dwi_reference),
            '--warp=' + str(mni_to_dwi_warp),
            '--out=' + str(output_file),
            '--interp=' + interpolation
        ]
    else:
        # Use linear registration only
        print("WARNING: No warp field provided, using FLIRT for linear registration")
        cmd = [
            'flirt',
            '-in', str(atlas_file),
            '-ref', str(dwi_reference),
            '-out', str(output_file),
            '-interp', 'nearestneighbour' if interpolation == 'nn' else 'trilinear',
            '-applyxfm'
        ]

    subprocess.run(cmd, check=True)
    print(f"Warped atlas to DWI space: {output_file}")

    return output_file


def extract_roi_mask(
    atlas_file: Path,
    roi_label: Union[int, List[int]],
    output_file: Path,
    binarize: bool = True
) -> Path:
    """
    Extract specific ROI(s) from atlas as binary mask.

    Parameters
    ----------
    atlas_file : Path
        Atlas file with integer labels
    roi_label : int or list of int
        Label value(s) to extract
    output_file : Path
        Output mask file
    binarize : bool
        If True, output is binary (0 or 1). If False, preserves label values.

    Returns
    -------
    Path
        Path to ROI mask
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load atlas
    atlas_img = nib.load(str(atlas_file))
    atlas_data = atlas_img.get_fdata()

    # Extract ROI(s)
    if isinstance(roi_label, int):
        roi_label = [roi_label]

    mask = np.zeros_like(atlas_data)
    for label in roi_label:
        mask[atlas_data == label] = 1 if binarize else label

    # Save mask
    mask_img = nib.Nifti1Image(mask.astype(np.float32), atlas_img.affine, atlas_img.header)
    nib.save(mask_img, str(output_file))

    print(f"Extracted ROI mask: {output_file}")
    print(f"  Labels: {roi_label}")
    print(f"  Voxels: {int(mask.sum())}")

    return output_file


def prepare_probtrackx_rois(
    dwi_reference: Path,
    seed_regions: List[str],
    target_regions: Optional[List[str]] = None,
    atlas: str = 'HarvardOxford-subcortical',
    output_dir: Path = Path('./probtrackx_rois'),
    mni_to_dwi_warp: Optional[Path] = None
) -> Dict[str, Dict[str, Path]]:
    """
    Prepare seed and target ROIs for probtrackx2 from standard atlases.

    Parameters
    ----------
    dwi_reference : Path
        Reference DWI volume (b0 or FA)
    seed_regions : list of str
        List of region names to use as seeds
    target_regions : list of str, optional
        List of region names to use as targets. If None, uses all other regions.
    atlas : str
        Atlas to use ('HarvardOxford-subcortical', 'HarvardOxford-cortical', 'JHU-ICBM-tracts-2mm')
    output_dir : Path
        Output directory for ROI masks
    mni_to_dwi_warp : Path, optional
        Warp from MNI to DWI space

    Returns
    -------
    dict
        Dictionary with 'seeds' and 'targets' subdicts containing region_name: mask_path

    Examples
    --------
    >>> rois = prepare_probtrackx_rois(
    ...     dwi_reference=Path('FA.nii.gz'),
    ...     seed_regions=['hippocampus_l', 'hippocampus_r'],
    ...     target_regions=['thalamus_l', 'thalamus_r'],
    ...     atlas='HarvardOxford-subcortical'
    ... )
    >>> print(rois['seeds']['hippocampus_l'])
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get atlas and label dictionary
    atlas_file = get_fsl_atlas_path(atlas)

    if 'HarvardOxford-cortical' in atlas:
        label_dict = HARVARD_OXFORD_CORTICAL_LABELS
    elif 'HarvardOxford-subcortical' in atlas:
        label_dict = HARVARD_OXFORD_SUBCORTICAL_LABELS
    elif 'JHU' in atlas:
        label_dict = JHU_WM_LABELS
    else:
        raise ValueError(f"No label dictionary for atlas: {atlas}")

    # Warp atlas to DWI space
    print(f"\nWarping {atlas} to DWI space...")
    atlas_dwi = warp_atlas_to_dwi(
        atlas_file=atlas_file,
        dwi_reference=dwi_reference,
        mni_to_dwi_warp=mni_to_dwi_warp,
        output_file=output_dir / f'{atlas}_dwi.nii.gz',
        interpolation='nn'
    )

    # Extract seed ROIs
    print(f"\nExtracting {len(seed_regions)} seed ROIs...")
    seeds = {}
    for region in seed_regions:
        if region not in label_dict:
            print(f"WARNING: Region '{region}' not found in {atlas}")
            continue

        label = label_dict[region]
        mask_file = output_dir / 'seeds' / f'{region}.nii.gz'
        extract_roi_mask(atlas_dwi, label, mask_file)
        seeds[region] = mask_file

    # Extract target ROIs
    targets = {}
    if target_regions:
        print(f"\nExtracting {len(target_regions)} target ROIs...")
        for region in target_regions:
            if region not in label_dict:
                print(f"WARNING: Region '{region}' not found in {atlas}")
                continue

            label = label_dict[region]
            mask_file = output_dir / 'targets' / f'{region}.nii.gz'
            extract_roi_mask(atlas_dwi, label, mask_file)
            targets[region] = mask_file

    print(f"\n{'='*60}")
    print("ROI preparation complete!")
    print(f"  Seeds: {len(seeds)}")
    print(f"  Targets: {len(targets)}")
    print(f"  Output directory: {output_dir}")

    return {'seeds': seeds, 'targets': targets, 'atlas_dwi': atlas_dwi}


# CLI usage
if __name__ == '__main__':
    import sys

    # Example: Prepare hippocampus-thalamus connectivity ROIs
    if len(sys.argv) < 2:
        print("Usage: python atlas_rois.py <dwi_reference>")
        print("\nExample:")
        print("  python atlas_rois.py FA.nii.gz")
        sys.exit(1)

    dwi_ref = Path(sys.argv[1])

    rois = prepare_probtrackx_rois(
        dwi_reference=dwi_ref,
        seed_regions=['hippocampus_l', 'hippocampus_r'],
        target_regions=['thalamus_l', 'thalamus_r'],
        atlas='HarvardOxford-subcortical',
        output_dir=Path('./probtrackx_rois')
    )

    print("\nGenerated ROI files:")
    for roi_type, roi_dict in rois.items():
        if isinstance(roi_dict, dict):
            for name, path in roi_dict.items():
                print(f"  {roi_type}/{name}: {path}")
