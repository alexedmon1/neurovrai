#!/usr/bin/env python3
"""
Atlas to Functional Space Transformation

Transform atlases from various source spaces to functional native space for
correlation-based functional connectivity analysis.

Supported Source Spaces:
    - MNI152: Standard atlases (Schaefer, AAL, Harvard-Oxford)
    - FreeSurfer: Subject-specific parcellations (aparc+aseg, Desikan-Killiany)

Transform Chains:
    MNI152 → T1w → Functional (via ANTs inverse + ANTs/FLIRT inverse)
    FreeSurfer → T1w → Functional (via registration + ANTs/FLIRT inverse)
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import json

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Atlas Configuration (extends DWI atlas configs for functional)
# =============================================================================

FUNC_ATLAS_CONFIGS = {
    # FSL atlases (already in MNI space)
    'harvardoxford_cort': {
        'file': '/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'space': 'MNI152',
        'description': 'Harvard-Oxford Cortical (48 regions)',
        'n_rois': 48
    },
    'harvardoxford_sub': {
        'file': '/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'space': 'MNI152',
        'description': 'Harvard-Oxford Subcortical (21 regions)',
        'n_rois': 21
    },
    'juelich': {
        'file': '/usr/local/fsl/data/atlases/Juelich/Juelich-maxprob-thr25-2mm.nii.gz',
        'labels': None,
        'space': 'MNI152',
        'description': 'Juelich Histological Atlas',
        'n_rois': None
    },

    # Schaefer functional parcellations (MNI space)
    'schaefer_100': {
        'file': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz',
        'labels': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_100Parcels_7Networks_order.txt',
        'space': 'MNI152',
        'description': 'Schaefer 100 parcels (7 Networks)',
        'n_rois': 100
    },
    'schaefer_200': {
        'file': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz',
        'labels': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_200Parcels_7Networks_order.txt',
        'space': 'MNI152',
        'description': 'Schaefer 200 parcels (7 Networks)',
        'n_rois': 200
    },
    'schaefer_400': {
        'file': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz',
        'labels': '/usr/local/fsl/data/atlases/Schaefer/Schaefer2018_400Parcels_7Networks_order.txt',
        'space': 'MNI152',
        'description': 'Schaefer 400 parcels (7 Networks)',
        'n_rois': 400
    },

    # FreeSurfer-based atlases (subject-specific)
    'desikan_killiany': {
        'source': 'freesurfer',
        'atlas_type': 'aparc+aseg',
        'space': 'freesurfer',
        'description': 'Desikan-Killiany cortical (68) + subcortical',
        'n_rois': 85
    },
    'destrieux': {
        'source': 'freesurfer',
        'atlas_type': 'aparc.a2009s+aseg',
        'space': 'freesurfer',
        'description': 'Destrieux cortical (148) + subcortical',
        'n_rois': 165
    },
}


class FuncAtlasTransformer:
    """
    Transform atlases from various spaces to functional native space.

    This class handles the complexity of transforming atlases from:
    - MNI152 standard space
    - FreeSurfer subject-specific space

    All transformed to the subject's functional native space for connectivity.

    Parameters
    ----------
    subject : str
        Subject identifier
    derivatives_dir : Path
        Path to derivatives directory
    fs_subjects_dir : Path, optional
        Path to FreeSurfer SUBJECTS_DIR
    config : dict, optional
        Configuration dictionary

    Examples
    --------
    >>> transformer = FuncAtlasTransformer(
    ...     subject='IRC805-0580101',
    ...     derivatives_dir=Path('/mnt/bytopia/IRC805/derivatives'),
    ...     fs_subjects_dir=Path('/mnt/bytopia/IRC805/freesurfer')
    ... )
    >>> atlas_func = transformer.transform_atlas_to_func(
    ...     atlas_name='desikan_killiany',
    ...     output_file=Path('atlases/desikan_killiany_func.nii.gz')
    ... )
    """

    def __init__(
        self,
        subject: str,
        derivatives_dir: Path,
        fs_subjects_dir: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        self.subject = subject
        self.derivatives_dir = Path(derivatives_dir)
        self.fs_subjects_dir = Path(fs_subjects_dir) if fs_subjects_dir else None
        self.config = config or {}

        # Subject directories
        self.subject_anat = self.derivatives_dir / subject / 'anat'
        self.subject_func = self.derivatives_dir / subject / 'func'
        self.subject_fs = self.fs_subjects_dir / subject if self.fs_subjects_dir else None

        # Discover available transforms
        self._discover_transforms()

    def _discover_transforms(self):
        """Discover available transformation files for this subject."""
        self.transforms = {}

        # T1w brain
        brain_candidates = [
            self.subject_anat / 'brain' / 'brain.nii.gz',
            self.subject_anat / 'brain.nii.gz',
        ]
        for candidate in brain_candidates:
            if candidate.exists():
                self.transforms['t1w_brain'] = candidate
                break

        # Also check for glob pattern
        if 't1w_brain' not in self.transforms:
            brain_dir = self.subject_anat / 'brain'
            if brain_dir.exists():
                brains = list(brain_dir.glob('*brain.nii.gz'))
                if brains:
                    self.transforms['t1w_brain'] = brains[0]

        # ANTs T1w → MNI composite (for MNI → T1w inverse)
        ants_composite = self.subject_anat / 'transforms' / 'ants_Composite.h5'
        if ants_composite.exists():
            self.transforms['t1w_to_mni_ants'] = ants_composite

        # Functional reference
        func_ref_candidates = [
            self.subject_func / 'brain' / 'func_brain.nii.gz',
            self.subject_func / 'func_brain.nii.gz',
            self.subject_func / 'mean_bold.nii.gz',
            self.subject_func / 'registration' / 'func_mean.nii.gz',
        ]
        for candidate in func_ref_candidates:
            if candidate.exists():
                self.transforms['func_ref'] = candidate
                break

        # Functional mask - also try glob pattern for variable naming
        mask_candidates = [
            self.subject_func / 'brain' / 'func_mask.nii.gz',
            self.subject_func / 'func_mask.nii.gz',
        ]
        for candidate in mask_candidates:
            if candidate.exists():
                self.transforms['func_mask'] = candidate
                break

        # Glob fallback for func_mask (variable naming convention)
        if 'func_mask' not in self.transforms:
            brain_dir = self.subject_func / 'brain'
            if brain_dir.exists():
                masks = list(brain_dir.glob('*brain_mask.nii.gz'))
                if masks:
                    self.transforms['func_mask'] = masks[0]

        # Functional → T1w transform (need inverse for T1w → Func)
        reg_dir = self.subject_func / 'registration'
        func_to_t1w_candidates = [
            reg_dir / 'func_to_t1w0GenericAffine.mat',
            reg_dir / 'func_to_t1w.mat',
            reg_dir / 'func2anat.mat',
        ]
        for candidate in func_to_t1w_candidates:
            if candidate.exists():
                self.transforms['func_to_t1w_mat'] = candidate
                break

        # Func → MNI composite (for MNI → Func inverse)
        func_to_mni = reg_dir / 'func_to_mni_Composite.h5'
        if func_to_mni.exists():
            self.transforms['func_to_mni_ants'] = func_to_mni

        # FreeSurfer subject directory
        if self.subject_fs and self.subject_fs.exists():
            self.transforms['fs_subject_dir'] = self.subject_fs

    def get_available_transforms(self) -> Dict[str, bool]:
        """
        Check which transform chains are available.

        Returns
        -------
        dict
            Keys: 'mni_to_func', 'fs_to_func'
            Values: bool indicating availability
        """
        has_func_ref = 'func_ref' in self.transforms or 'func_mask' in self.transforms

        available = {
            'mni_to_func': (
                has_func_ref and
                ('func_to_mni_ants' in self.transforms or
                 ('t1w_to_mni_ants' in self.transforms and 'func_to_t1w_mat' in self.transforms))
            ),
            'fs_to_func': (
                'fs_subject_dir' in self.transforms and
                't1w_brain' in self.transforms and
                has_func_ref and
                'func_to_t1w_mat' in self.transforms
            ),
            'resample_only': has_func_ref  # Can always resample if we have a reference
        }
        return available

    def _get_func_reference(self) -> Path:
        """Get functional reference image."""
        if 'func_ref' in self.transforms:
            return self.transforms['func_ref']
        elif 'func_mask' in self.transforms:
            return self.transforms['func_mask']
        else:
            raise FileNotFoundError(
                f"No functional reference found for {self.subject}. "
                "Need func_brain.nii.gz or func_mask.nii.gz"
            )

    def transform_mni_atlas_to_func(
        self,
        atlas_mni: Path,
        output_file: Path,
        intermediate_dir: Optional[Path] = None
    ) -> Path:
        """
        Transform MNI-space atlas to functional native space.

        Uses either:
        - Direct: MNI → Func (via ANTs inverse of func_to_mni)
        - Chain: MNI → T1w (ANTs inverse) → Func (FLIRT inverse)

        Parameters
        ----------
        atlas_mni : Path
            Atlas in MNI152 space
        output_file : Path
            Output atlas in functional space
        intermediate_dir : Path, optional
            Directory for intermediate files

        Returns
        -------
        Path
            Transformed atlas
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        intermediate_dir = intermediate_dir or output_file.parent
        func_ref = self._get_func_reference()

        logger.info(f"Transforming MNI atlas to functional space: {atlas_mni.name}")

        # Preferred: Direct MNI → Func if we have func_to_mni composite
        if 'func_to_mni_ants' in self.transforms:
            logger.info("  Using direct MNI → Func (ANTs inverse)")
            func_to_mni = self.transforms['func_to_mni_ants']

            cmd = [
                'antsApplyTransforms',
                '-d', '3',
                '-i', str(atlas_mni),
                '-r', str(func_ref),
                '-o', str(output_file),
                '-n', 'GenericLabel',
                '-t', f'[{str(func_to_mni)},1]'  # ,1 = inverse
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ANTs MNI→Func failed: {result.stderr}")

        # Fallback: MNI → T1w → Func chain
        elif 't1w_to_mni_ants' in self.transforms and 'func_to_t1w_mat' in self.transforms:
            logger.info("  Using chain: MNI → T1w → Func")

            t1w_brain = self.transforms['t1w_brain']
            mni_to_t1w_warp = self.transforms['t1w_to_mni_ants']
            func_to_t1w_mat = self.transforms['func_to_t1w_mat']

            # Step 1: MNI → T1w using ANTs inverse
            atlas_in_t1w = intermediate_dir / f'{atlas_mni.stem}_in_t1w.nii.gz'

            logger.info("    Step 1: MNI → T1w (ANTs inverse)")
            cmd_ants = [
                'antsApplyTransforms',
                '-d', '3',
                '-i', str(atlas_mni),
                '-r', str(t1w_brain),
                '-o', str(atlas_in_t1w),
                '-n', 'GenericLabel',
                '-t', f'[{str(mni_to_t1w_warp)},1]'
            ]

            result = subprocess.run(cmd_ants, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ANTs MNI→T1w failed: {result.stderr}")

            # Step 2: T1w → Func using FLIRT inverse
            # First compute inverse of func_to_t1w
            t1w_to_func_mat = intermediate_dir / 't1w_to_func.mat'

            logger.info("    Step 2: Computing T1w → Func inverse transform")
            cmd_invert = [
                'convert_xfm',
                '-omat', str(t1w_to_func_mat),
                '-inverse', str(func_to_t1w_mat)
            ]
            result = subprocess.run(cmd_invert, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"convert_xfm inverse failed: {result.stderr}")

            # Apply T1w → Func transform
            logger.info("    Step 3: T1w → Func (FLIRT inverse)")
            cmd_flirt = [
                'flirt',
                '-in', str(atlas_in_t1w),
                '-ref', str(func_ref),
                '-applyxfm',
                '-init', str(t1w_to_func_mat),
                '-out', str(output_file),
                '-interp', 'nearestneighbour'
            ]

            result = subprocess.run(cmd_flirt, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FLIRT T1w→Func failed: {result.stderr}")

            # Clean up intermediate
            if atlas_in_t1w.exists():
                atlas_in_t1w.unlink()

        else:
            raise RuntimeError(
                f"No MNI→Func transform available for {self.subject}. "
                "Need func_to_mni_Composite.h5 or (t1w_to_mni_ants + func_to_t1w.mat)"
            )

        logger.info(f"  Output: {output_file}")
        return output_file

    def transform_fs_atlas_to_func(
        self,
        atlas_type: str,
        output_file: Path,
        intermediate_dir: Optional[Path] = None
    ) -> Path:
        """
        Transform FreeSurfer atlas to functional native space.

        Chain: FreeSurfer → T1w → Func

        Parameters
        ----------
        atlas_type : str
            FreeSurfer atlas type: 'aparc+aseg', 'aparc.a2009s+aseg', 'aseg'
        output_file : Path
            Output atlas in functional space
        intermediate_dir : Path, optional
            Directory for intermediate files

        Returns
        -------
        Path
            Transformed atlas
        """
        available = self.get_available_transforms()
        if not available['fs_to_func']:
            raise RuntimeError(
                f"FreeSurfer→Func transform not available for {self.subject}. "
                "Missing FreeSurfer outputs, T1w brain, or func_to_t1w transform."
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        intermediate_dir = intermediate_dir or output_file.parent

        fs_subject_dir = self.transforms['fs_subject_dir']
        t1w_brain = self.transforms['t1w_brain']
        func_ref = self._get_func_reference()
        func_to_t1w_mat = self.transforms['func_to_t1w_mat']

        logger.info(f"Transforming FreeSurfer {atlas_type} to functional space")

        # Step 1: Use mri_vol2vol to resample atlas directly to T1w space
        # This is fast because FreeSurfer was run on the same T1w - uses --regheader
        atlas_in_t1w = intermediate_dir / f'{atlas_type}_in_t1w.nii.gz'
        atlas_mgz = fs_subject_dir / 'mri' / f'{atlas_type}.mgz'

        if not atlas_mgz.exists():
            raise FileNotFoundError(f"FreeSurfer atlas not found: {atlas_mgz}")

        logger.info("  Step 1: FreeSurfer → T1w (mri_vol2vol --regheader)")
        cmd_vol2vol = [
            'mri_vol2vol',
            '--mov', str(atlas_mgz),
            '--targ', str(t1w_brain),
            '--regheader',
            '--o', str(atlas_in_t1w),
            '--interp', 'nearest',
            '--no-save-reg'
        ]
        result = subprocess.run(cmd_vol2vol, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mri_vol2vol failed: {result.stderr}")

        # Step 2: T1w → Func using ANTs inverse transform
        # Check if we have ANTs affine (func_to_t1w0GenericAffine.mat)
        if str(func_to_t1w_mat).endswith('.mat') and 'GenericAffine' in str(func_to_t1w_mat):
            # ANTs affine - use antsApplyTransforms with inverse
            logger.info("  Step 2: T1w → Func (ANTs inverse)")
            cmd_ants = [
                'antsApplyTransforms',
                '-d', '3',
                '-i', str(atlas_in_t1w),
                '-r', str(func_ref),
                '-o', str(output_file),
                '-n', 'NearestNeighbor',
                '-t', f'[{str(func_to_t1w_mat)},1]'  # ,1 = inverse
            ]
            result = subprocess.run(cmd_ants, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"antsApplyTransforms failed: {result.stderr}")
        else:
            # FSL FLIRT matrix - use convert_xfm and flirt
            t1w_to_func_mat = intermediate_dir / 't1w_to_func.mat'

            logger.info("  Step 2a: Computing T1w → Func inverse transform")
            cmd_invert = [
                'convert_xfm',
                '-omat', str(t1w_to_func_mat),
                '-inverse', str(func_to_t1w_mat)
            ]
            result = subprocess.run(cmd_invert, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"convert_xfm inverse failed: {result.stderr}")

            logger.info("  Step 2b: T1w → Func (FLIRT inverse)")
            cmd_flirt = [
                'flirt',
                '-in', str(atlas_in_t1w),
                '-ref', str(func_ref),
                '-applyxfm',
                '-init', str(t1w_to_func_mat),
                '-out', str(output_file),
                '-interp', 'nearestneighbour'
            ]
            result = subprocess.run(cmd_flirt, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FLIRT T1w→Func failed: {result.stderr}")

        # Clean up intermediate files
        if atlas_in_t1w.exists() and atlas_in_t1w != output_file:
            atlas_in_t1w.unlink()

        logger.info(f"  Output: {output_file}")
        return output_file

    def _extract_freesurfer_atlas(
        self,
        fs_subject_dir: Path,
        atlas_type: str,
        output_file: Path
    ):
        """Extract atlas from FreeSurfer directory."""
        # Map atlas type to file
        atlas_files = {
            'aparc+aseg': 'mri/aparc+aseg.mgz',
            'aparc.a2009s+aseg': 'mri/aparc.a2009s+aseg.mgz',
            'aseg': 'mri/aseg.mgz',
        }

        if atlas_type not in atlas_files:
            raise ValueError(f"Unknown FreeSurfer atlas type: {atlas_type}")

        atlas_mgz = fs_subject_dir / atlas_files[atlas_type]
        if not atlas_mgz.exists():
            raise FileNotFoundError(f"FreeSurfer atlas not found: {atlas_mgz}")

        # Convert MGZ to NIfTI
        cmd = ['mri_convert', str(atlas_mgz), str(output_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mri_convert failed: {result.stderr}")

    def _compute_fs_to_t1w_transform(
        self,
        fs_orig: Path,
        t1w_brain: Path,
        output_mat: Path
    ):
        """Compute FreeSurfer → T1w transform."""
        if output_mat.exists():
            logger.debug(f"Using existing FS→T1w transform: {output_mat}")
            return

        # Convert FS orig to NIfTI for FLIRT
        fs_orig_nii = output_mat.parent / 'fs_orig.nii.gz'
        cmd_convert = ['mri_convert', str(fs_orig), str(fs_orig_nii)]
        result = subprocess.run(cmd_convert, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mri_convert failed: {result.stderr}")

        # Register FS → T1w
        logger.info("  Computing FreeSurfer → T1w transform")
        cmd = [
            'flirt',
            '-in', str(fs_orig_nii),
            '-ref', str(t1w_brain),
            '-omat', str(output_mat),
            '-dof', '6',  # Rigid body
            '-cost', 'corratio'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FLIRT FS→T1w registration failed: {result.stderr}")

        # Clean up
        if fs_orig_nii.exists():
            fs_orig_nii.unlink()

    def resample_atlas_to_func(
        self,
        atlas_file: Path,
        output_file: Path
    ) -> Path:
        """
        Simple resampling of atlas to functional space (no transform chain).

        Uses nilearn's resample_to_img for nearest-neighbor interpolation.
        This is the fallback when proper transforms aren't available.

        Parameters
        ----------
        atlas_file : Path
            Atlas file (any space)
        output_file : Path
            Output resampled atlas

        Returns
        -------
        Path
            Resampled atlas
        """
        from nilearn.image import resample_to_img

        output_file.parent.mkdir(parents=True, exist_ok=True)
        func_ref = self._get_func_reference()

        logger.info(f"Resampling atlas to functional space: {atlas_file.name}")

        atlas_img = nib.load(atlas_file)
        func_img = nib.load(func_ref)

        resampled = resample_to_img(
            atlas_img,
            func_img,
            interpolation='nearest'
        )

        nib.save(resampled, output_file)
        logger.info(f"  Output: {output_file}")

        return output_file

    def transform_atlas_to_func(
        self,
        atlas_name: str,
        output_file: Path,
        intermediate_dir: Optional[Path] = None
    ) -> Tuple[Path, List[str]]:
        """
        Transform atlas to functional space (main entry point).

        Automatically determines source space and applies appropriate transform.

        Parameters
        ----------
        atlas_name : str
            Atlas name from FUNC_ATLAS_CONFIGS
        output_file : Path
            Output atlas in functional space
        intermediate_dir : Path, optional
            Directory for intermediate files

        Returns
        -------
        tuple of (Path, list)
            (atlas_in_func, roi_names)
        """
        if atlas_name not in FUNC_ATLAS_CONFIGS:
            raise ValueError(
                f"Unknown atlas: {atlas_name}. "
                f"Available: {list(FUNC_ATLAS_CONFIGS.keys())}"
            )

        config = FUNC_ATLAS_CONFIGS[atlas_name]
        space = config['space']

        logger.info(f"Transforming atlas '{atlas_name}' to functional space")
        logger.info(f"  Source space: {space}")

        if space == 'MNI152':
            atlas_file = Path(config['file'])
            if not atlas_file.exists():
                raise FileNotFoundError(f"Atlas file not found: {atlas_file}")

            # Try proper transform first, fall back to resampling
            available = self.get_available_transforms()
            if available['mni_to_func']:
                result = self.transform_mni_atlas_to_func(
                    atlas_mni=atlas_file,
                    output_file=output_file,
                    intermediate_dir=intermediate_dir
                )
            else:
                logger.warning("No proper MNI→Func transform, using resampling")
                result = self.resample_atlas_to_func(atlas_file, output_file)

        elif space == 'freesurfer':
            atlas_type = config['atlas_type']
            result = self.transform_fs_atlas_to_func(
                atlas_type=atlas_type,
                output_file=output_file,
                intermediate_dir=intermediate_dir
            )

        else:
            raise ValueError(f"Unsupported atlas space: {space}")

        # Get ROI names
        roi_names = self._get_roi_names(atlas_name, result)

        return result, roi_names

    def _get_roi_names(self, atlas_name: str, atlas_file: Path) -> List[str]:
        """Extract ROI names from atlas."""
        config = FUNC_ATLAS_CONFIGS[atlas_name]

        if config['space'] == 'freesurfer':
            from neurovrai.preprocess.utils.freesurfer_utils import (
                get_desikan_killiany_labels
            )

            # Get labels from atlas image
            img = nib.load(atlas_file)
            data = img.get_fdata().astype(int)
            unique_labels = sorted(set(data.flatten()) - {0})

            label_map = get_desikan_killiany_labels()

            roi_names = [
                label_map.get(label, f'Unknown_{label}')
                for label in unique_labels
            ]

        else:
            # For MNI atlases, use numbered labels
            img = nib.load(atlas_file)
            data = img.get_fdata().astype(int)
            unique_labels = sorted(set(data.flatten()) - {0})
            roi_names = [f'ROI_{label}' for label in unique_labels]

        return roi_names


# =============================================================================
# Convenience Functions
# =============================================================================

def prepare_atlas_for_fc(
    subject: str,
    atlas_name: str,
    derivatives_dir: Path,
    output_dir: Path,
    fs_subjects_dir: Optional[Path] = None,
    config: Optional[Dict] = None
) -> Tuple[Path, List[str], Dict[str, Any]]:
    """
    Prepare atlas in functional space for connectivity analysis (main entry point).

    Parameters
    ----------
    subject : str
        Subject identifier
    atlas_name : str
        Atlas name from FUNC_ATLAS_CONFIGS
    derivatives_dir : Path
        Path to derivatives directory
    output_dir : Path
        Output directory for transformed atlas
    fs_subjects_dir : Path, optional
        FreeSurfer SUBJECTS_DIR
    config : dict, optional
        Configuration dictionary

    Returns
    -------
    tuple of (Path, list, dict)
        (atlas_in_func, roi_names, metadata)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformer = FuncAtlasTransformer(
        subject=subject,
        derivatives_dir=derivatives_dir,
        fs_subjects_dir=fs_subjects_dir,
        config=config
    )

    # Output file
    output_file = output_dir / f'{atlas_name}_in_func.nii.gz'
    intermediate_dir = output_dir / 'intermediate'

    # Transform atlas
    atlas_func, roi_names = transformer.transform_atlas_to_func(
        atlas_name=atlas_name,
        output_file=output_file,
        intermediate_dir=intermediate_dir
    )

    # Save ROI names
    roi_names_file = output_dir / f'{atlas_name}_roi_names.txt'
    with open(roi_names_file, 'w') as f:
        f.write('\n'.join(roi_names))

    # Metadata
    atlas_config = FUNC_ATLAS_CONFIGS[atlas_name]
    metadata = {
        'subject': subject,
        'atlas_name': atlas_name,
        'atlas_description': atlas_config['description'],
        'source_space': atlas_config['space'],
        'n_rois_expected': atlas_config.get('n_rois', len(roi_names)),
        'n_rois_actual': len(roi_names),
        'atlas_in_func': str(atlas_func),
        'roi_names_file': str(roi_names_file),
        'available_transforms': transformer.get_available_transforms()
    }

    # Save metadata
    metadata_file = output_dir / f'{atlas_name}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Atlas preparation complete for {subject}")
    logger.info(f"  Atlas: {atlas_func}")
    logger.info(f"  ROIs: {len(roi_names)}")

    return atlas_func, roi_names, metadata


def list_available_fc_atlases() -> Dict[str, Dict]:
    """List all available atlas configurations for functional connectivity."""
    return FUNC_ATLAS_CONFIGS.copy()
