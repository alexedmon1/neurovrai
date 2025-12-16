#!/usr/bin/env python3
"""
Atlas to DWI Space Transformation

Transform atlases from various source spaces to DWI native space for
tractography-based structural connectivity analysis.

Supported Source Spaces:
    - MNI152: Standard atlases (Schaefer, AAL, Harvard-Oxford)
    - FreeSurfer: Subject-specific parcellations (aparc+aseg, Desikan-Killiany)
    - FMRIB58_FA: White matter atlases (JHU ICBM-DTI-81)

Transform Chains:
    MNI152 → T1w → DWI (via ANTs inverse + FLIRT)
    FreeSurfer → T1w → DWI (via FLIRT chain)
    FMRIB58 → DWI (via inverse warp from DWI normalization)
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
# Atlas Configuration
# =============================================================================

ATLAS_CONFIGS = {
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
        'n_rois': 85  # 68 cortical + 17 subcortical
    },
    'destrieux': {
        'source': 'freesurfer',
        'atlas_type': 'aparc.a2009s+aseg',
        'space': 'freesurfer',
        'description': 'Destrieux cortical (148) + subcortical',
        'n_rois': 165
    },

    # White matter atlases (FMRIB58 space)
    'jhu_icbm_labels': {
        'file': '/usr/local/fsl/data/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz',
        'labels': '/usr/local/fsl/data/atlases/JHU-labels.xml',
        'space': 'FMRIB58',
        'description': 'JHU ICBM-DTI-81 white matter labels (48 regions)',
        'n_rois': 48
    },
    'jhu_icbm_tracts': {
        'file': '/usr/local/fsl/data/atlases/JHU/JHU-ICBM-tracts-maxprob-thr25-2mm.nii.gz',
        'labels': '/usr/local/fsl/data/atlases/JHU-tracts.xml',
        'space': 'FMRIB58',
        'description': 'JHU ICBM white matter tracts (20 regions)',
        'n_rois': 20
    },
}


class DWIAtlasTransformer:
    """
    Transform atlases from various spaces to DWI native space.

    This class handles the complexity of transforming atlases from:
    - MNI152 standard space
    - FreeSurfer subject-specific space
    - FMRIB58_FA template space

    All transformed to the subject's DWI native space for tractography.

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
    >>> transformer = DWIAtlasTransformer(
    ...     subject='IRC805-0580101',
    ...     derivatives_dir=Path('/mnt/bytopia/IRC805/derivatives'),
    ...     fs_subjects_dir=Path('/mnt/bytopia/IRC805/freesurfer')
    ... )
    >>> atlas_dwi = transformer.transform_atlas_to_dwi(
    ...     atlas_name='schaefer_200',
    ...     output_file=Path('atlases/schaefer_200_dwi.nii.gz')
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
        self.subject_dwi = self.derivatives_dir / subject / 'dwi'
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

        # DWI reference (b0 brain)
        dwi_candidates = [
            self.subject_dwi / 'brain' / 'b0_brain.nii.gz',
            self.subject_dwi / 'b0_brain.nii.gz',
            self.subject_dwi / 'nodif_brain.nii.gz',
        ]
        for candidate in dwi_candidates:
            if candidate.exists():
                self.transforms['dwi_brain'] = candidate
                break

        # Check for DWI mask as alternative
        if 'dwi_brain' not in self.transforms:
            mask_candidates = [
                self.subject_dwi / 'brain' / 'b0_brain_mask.nii.gz',
                self.subject_dwi / 'mask.nii.gz',
            ]
            for mask in mask_candidates:
                if mask.exists():
                    self.transforms['dwi_mask'] = mask
                    break

        # FMRIB58 inverse warp (from DWI normalization)
        fmrib_inverse = self.subject_dwi / 'normalized' / 'fmrib58_to_fa_warp.nii.gz'
        if fmrib_inverse.exists():
            self.transforms['fmrib58_to_dwi_warp'] = fmrib_inverse

        # FA image (for FMRIB58 transforms)
        fa_candidates = [
            self.subject_dwi / 'dti' / 'dtifit__FA.nii.gz',
            self.subject_dwi / 'FA.nii.gz',
        ]
        for candidate in fa_candidates:
            if candidate.exists():
                self.transforms['fa'] = candidate
                break

        # FreeSurfer transforms (will be computed on demand)
        if self.subject_fs and self.subject_fs.exists():
            self.transforms['fs_subject_dir'] = self.subject_fs

    def get_available_transforms(self) -> Dict[str, bool]:
        """
        Check which transform chains are available.

        Returns
        -------
        dict
            Keys: 'mni_to_dwi', 'fs_to_dwi', 'fmrib58_to_dwi'
            Values: bool indicating availability
        """
        available = {
            'mni_to_dwi': (
                't1w_brain' in self.transforms and
                't1w_to_mni_ants' in self.transforms and
                ('dwi_brain' in self.transforms or 'fa' in self.transforms)
            ),
            'fs_to_dwi': (
                'fs_subject_dir' in self.transforms and
                't1w_brain' in self.transforms and
                ('dwi_brain' in self.transforms or 'fa' in self.transforms)
            ),
            'fmrib58_to_dwi': (
                'fmrib58_to_dwi_warp' in self.transforms and
                'fa' in self.transforms
            )
        }
        return available

    def _get_dwi_reference(self) -> Path:
        """Get DWI reference image (b0 brain or FA)."""
        if 'dwi_brain' in self.transforms:
            return self.transforms['dwi_brain']
        elif 'fa' in self.transforms:
            return self.transforms['fa']
        else:
            raise FileNotFoundError(
                f"No DWI reference found for {self.subject}. "
                "Need b0_brain.nii.gz or FA.nii.gz"
            )

    def _compute_t1w_to_dwi_transform(self, output_dir: Path) -> Path:
        """Compute T1w → DWI transform if not already available."""
        from neurovrai.preprocess.utils.freesurfer_transforms import (
            compute_t1w_to_dwi_transform
        )

        t1w_to_dwi_mat = output_dir / 't1w_to_dwi.mat'

        if t1w_to_dwi_mat.exists():
            logger.debug(f"Using existing T1w→DWI transform: {t1w_to_dwi_mat}")
            return t1w_to_dwi_mat

        t1w_brain = self.transforms['t1w_brain']
        dwi_ref = self._get_dwi_reference()

        logger.info("Computing T1w → DWI transform")
        t1w_to_dwi, _ = compute_t1w_to_dwi_transform(
            t1w_brain=t1w_brain,
            dwi_b0_brain=dwi_ref,
            output_dir=output_dir
        )

        self.transforms['t1w_to_dwi'] = t1w_to_dwi
        return t1w_to_dwi

    def transform_mni_atlas_to_dwi(
        self,
        atlas_mni: Path,
        output_file: Path,
        intermediate_dir: Optional[Path] = None
    ) -> Path:
        """
        Transform MNI-space atlas to DWI native space.

        Chain: MNI → T1w (ANTs inverse) → DWI (FLIRT)

        Parameters
        ----------
        atlas_mni : Path
            Atlas in MNI152 space
        output_file : Path
            Output atlas in DWI space
        intermediate_dir : Path, optional
            Directory for intermediate files

        Returns
        -------
        Path
            Transformed atlas
        """
        available = self.get_available_transforms()
        if not available['mni_to_dwi']:
            raise RuntimeError(
                f"MNI→DWI transform not available for {self.subject}. "
                "Missing T1w brain, ANTs composite, or DWI reference."
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        intermediate_dir = intermediate_dir or output_file.parent

        logger.info(f"Transforming MNI atlas to DWI space: {atlas_mni.name}")

        # Get transforms
        t1w_brain = self.transforms['t1w_brain']
        mni_to_t1w_warp = self.transforms['t1w_to_mni_ants']
        dwi_ref = self._get_dwi_reference()

        # Ensure T1w→DWI transform exists
        t1w_to_dwi_mat = self._compute_t1w_to_dwi_transform(intermediate_dir)

        # Step 1: MNI → T1w using ANTs inverse
        atlas_in_t1w = intermediate_dir / f'{atlas_mni.stem}_in_t1w.nii.gz'

        logger.info("  Step 1: MNI → T1w (ANTs inverse)")
        cmd_ants = [
            'antsApplyTransforms',
            '-d', '3',
            '-i', str(atlas_mni),
            '-r', str(t1w_brain),
            '-o', str(atlas_in_t1w),
            '-n', 'GenericLabel',
            '-t', f'[{str(mni_to_t1w_warp)},1]'  # ,1 = inverse
        ]

        result = subprocess.run(cmd_ants, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ANTs MNI→T1w failed: {result.stderr}")

        # Step 2: T1w → DWI using FLIRT
        logger.info("  Step 2: T1w → DWI (FLIRT)")
        cmd_flirt = [
            'flirt',
            '-in', str(atlas_in_t1w),
            '-ref', str(dwi_ref),
            '-applyxfm',
            '-init', str(t1w_to_dwi_mat),
            '-out', str(output_file),
            '-interp', 'nearestneighbour'
        ]

        result = subprocess.run(cmd_flirt, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FLIRT T1w→DWI failed: {result.stderr}")

        # Clean up intermediate
        if atlas_in_t1w.exists():
            atlas_in_t1w.unlink()

        logger.info(f"  Output: {output_file}")
        return output_file

    def transform_fs_atlas_to_dwi(
        self,
        atlas_type: str,
        output_file: Path,
        intermediate_dir: Optional[Path] = None
    ) -> Path:
        """
        Transform FreeSurfer atlas to DWI native space.

        Chain: FreeSurfer → T1w → DWI

        Parameters
        ----------
        atlas_type : str
            FreeSurfer atlas type: 'aparc+aseg', 'aparc.a2009s+aseg', 'aseg'
        output_file : Path
            Output atlas in DWI space
        intermediate_dir : Path, optional
            Directory for intermediate files

        Returns
        -------
        Path
            Transformed atlas
        """
        from neurovrai.preprocess.utils.freesurfer_utils import (
            extract_freesurfer_atlas
        )
        from neurovrai.preprocess.utils.freesurfer_transforms import (
            compute_all_transforms,
            transform_atlas_to_dwi
        )

        available = self.get_available_transforms()
        if not available['fs_to_dwi']:
            raise RuntimeError(
                f"FreeSurfer→DWI transform not available for {self.subject}. "
                "Missing FreeSurfer outputs, T1w brain, or DWI reference."
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        intermediate_dir = intermediate_dir or output_file.parent

        fs_subject_dir = self.transforms['fs_subject_dir']
        t1w_brain = self.transforms['t1w_brain']
        dwi_ref = self._get_dwi_reference()

        logger.info(f"Transforming FreeSurfer {atlas_type} to DWI space (using mri_vol2vol)")

        # Get FreeSurfer atlas file
        aparc_file = fs_subject_dir / 'mri' / f'{atlas_type}.mgz'
        if not aparc_file.exists():
            raise FileNotFoundError(f"FreeSurfer atlas not found: {aparc_file}")

        # Step 1: FS → T1w using mri_vol2vol (fast!)
        atlas_in_t1w = intermediate_dir / f'{atlas_type}_in_t1w.nii.gz'
        logger.info(f"  Step 1: FS → T1w (mri_vol2vol)")

        cmd_fs_to_t1w = [
            'mri_vol2vol',
            '--mov', str(aparc_file),
            '--targ', str(t1w_brain),
            '--o', str(atlas_in_t1w),
            '--regheader',  # Fast header-based registration
            '--nearest'  # Nearest neighbor for label preservation
        ]

        result = subprocess.run(cmd_fs_to_t1w, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mri_vol2vol FS→T1w failed: {result.stderr}")

        logger.info(f"    Atlas in T1w space: {atlas_in_t1w.name}")

        # Step 2: T1w → DWI using FLIRT
        logger.info(f"  Step 2: T1w → DWI (FLIRT)")

        # Compute T1w→DWI transform if needed
        from neurovrai.preprocess.utils.freesurfer_transforms import compute_t1w_to_dwi_transform

        transforms_dir = intermediate_dir / 'transforms'
        transforms_dir.mkdir(parents=True, exist_ok=True)

        t1w_to_dwi_mat = transforms_dir / 't1w_to_dwi.mat'
        if not t1w_to_dwi_mat.exists():
            t1w_to_dwi_mat, _ = compute_t1w_to_dwi_transform(
                t1w_brain=t1w_brain,
                dwi_b0_brain=dwi_ref,
                output_dir=transforms_dir
            )

        # Apply T1w→DWI transform
        cmd_t1w_to_dwi = [
            'flirt',
            '-in', str(atlas_in_t1w),
            '-ref', str(dwi_ref),
            '-applyxfm',
            '-init', str(t1w_to_dwi_mat),
            '-out', str(output_file),
            '-interp', 'nearestneighbour'
        ]

        result = subprocess.run(cmd_t1w_to_dwi, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FLIRT T1w→DWI failed: {result.stderr}")

        logger.info(f"  Output: {output_file}")
        return output_file

    def transform_fmrib58_atlas_to_dwi(
        self,
        atlas_fmrib58: Path,
        output_file: Path
    ) -> Path:
        """
        Transform FMRIB58-space atlas to DWI native space.

        Uses inverse warp from DWI normalization.

        Parameters
        ----------
        atlas_fmrib58 : Path
            Atlas in FMRIB58_FA space
        output_file : Path
            Output atlas in DWI space

        Returns
        -------
        Path
            Transformed atlas
        """
        available = self.get_available_transforms()
        if not available['fmrib58_to_dwi']:
            raise RuntimeError(
                f"FMRIB58→DWI transform not available for {self.subject}. "
                "Missing inverse warp from DWI normalization."
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        inverse_warp = self.transforms['fmrib58_to_dwi_warp']
        fa = self.transforms['fa']

        logger.info(f"Transforming FMRIB58 atlas to DWI space: {atlas_fmrib58.name}")

        # Use FSL applywarp with inverse warp
        cmd = [
            'applywarp',
            '--in=' + str(atlas_fmrib58),
            '--ref=' + str(fa),
            '--warp=' + str(inverse_warp),
            '--out=' + str(output_file),
            '--interp=nn'  # Nearest neighbor for labels
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"applywarp FMRIB58→DWI failed: {result.stderr}")

        logger.info(f"  Output: {output_file}")
        return output_file

    def transform_atlas_to_dwi(
        self,
        atlas_name: str,
        output_file: Path,
        intermediate_dir: Optional[Path] = None
    ) -> Tuple[Path, List[str]]:
        """
        Transform atlas to DWI space (main entry point).

        Automatically determines source space and applies appropriate transform.

        Parameters
        ----------
        atlas_name : str
            Atlas name from ATLAS_CONFIGS
        output_file : Path
            Output atlas in DWI space
        intermediate_dir : Path, optional
            Directory for intermediate files

        Returns
        -------
        tuple of (Path, list)
            (atlas_in_dwi, roi_names)
        """
        if atlas_name not in ATLAS_CONFIGS:
            raise ValueError(
                f"Unknown atlas: {atlas_name}. "
                f"Available: {list(ATLAS_CONFIGS.keys())}"
            )

        config = ATLAS_CONFIGS[atlas_name]
        space = config['space']

        logger.info(f"Transforming atlas '{atlas_name}' to DWI space")
        logger.info(f"  Source space: {space}")

        if space == 'MNI152':
            atlas_file = Path(config['file'])
            if not atlas_file.exists():
                raise FileNotFoundError(f"Atlas file not found: {atlas_file}")

            result = self.transform_mni_atlas_to_dwi(
                atlas_mni=atlas_file,
                output_file=output_file,
                intermediate_dir=intermediate_dir
            )

        elif space == 'freesurfer':
            atlas_type = config['atlas_type']
            result = self.transform_fs_atlas_to_dwi(
                atlas_type=atlas_type,
                output_file=output_file,
                intermediate_dir=intermediate_dir
            )

        elif space == 'FMRIB58':
            atlas_file = Path(config['file'])
            if not atlas_file.exists():
                raise FileNotFoundError(f"Atlas file not found: {atlas_file}")

            result = self.transform_fmrib58_atlas_to_dwi(
                atlas_fmrib58=atlas_file,
                output_file=output_file
            )

        else:
            raise ValueError(f"Unsupported atlas space: {space}")

        # Get ROI names
        roi_names = self._get_roi_names(atlas_name, result)

        return result, roi_names

    def _get_roi_names(self, atlas_name: str, atlas_file: Path) -> List[str]:
        """Extract ROI names from atlas."""
        config = ATLAS_CONFIGS[atlas_name]

        if config['space'] == 'freesurfer':
            from neurovrai.preprocess.utils.freesurfer_utils import (
                get_desikan_killiany_labels,
                get_subcortical_labels
            )

            # Get labels from atlas image
            img = nib.load(atlas_file)
            data = img.get_fdata().astype(int)
            unique_labels = sorted(set(data.flatten()) - {0})

            if config['atlas_type'] == 'aparc+aseg':
                label_map = get_desikan_killiany_labels()
            else:
                label_map = get_desikan_killiany_labels()

            roi_names = [
                label_map.get(label, f'Unknown_{label}')
                for label in unique_labels
            ]

        else:
            # For MNI/FMRIB58 atlases, use numbered labels
            img = nib.load(atlas_file)
            data = img.get_fdata().astype(int)
            unique_labels = sorted(set(data.flatten()) - {0})
            roi_names = [f'ROI_{label}' for label in unique_labels]

        return roi_names


# =============================================================================
# Convenience Functions
# =============================================================================

def prepare_atlas_for_tractography(
    subject: str,
    atlas_name: str,
    derivatives_dir: Path,
    output_dir: Path,
    fs_subjects_dir: Optional[Path] = None,
    config: Optional[Dict] = None
) -> Tuple[Path, List[str], Dict[str, Any]]:
    """
    Prepare atlas in DWI space for tractography (main entry point).

    Parameters
    ----------
    subject : str
        Subject identifier
    atlas_name : str
        Atlas name from ATLAS_CONFIGS
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
        (atlas_in_dwi, roi_names, metadata)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transformer = DWIAtlasTransformer(
        subject=subject,
        derivatives_dir=derivatives_dir,
        fs_subjects_dir=fs_subjects_dir,
        config=config
    )

    # Output file
    output_file = output_dir / f'{atlas_name}_in_dwi.nii.gz'
    intermediate_dir = output_dir / 'intermediate'

    # Transform atlas
    atlas_dwi, roi_names = transformer.transform_atlas_to_dwi(
        atlas_name=atlas_name,
        output_file=output_file,
        intermediate_dir=intermediate_dir
    )

    # Save ROI names
    roi_names_file = output_dir / f'{atlas_name}_roi_names.txt'
    with open(roi_names_file, 'w') as f:
        f.write('\n'.join(roi_names))

    # Metadata
    atlas_config = ATLAS_CONFIGS[atlas_name]
    metadata = {
        'subject': subject,
        'atlas_name': atlas_name,
        'atlas_description': atlas_config['description'],
        'source_space': atlas_config['space'],
        'n_rois_expected': atlas_config.get('n_rois', len(roi_names)),
        'n_rois_actual': len(roi_names),
        'atlas_in_dwi': str(atlas_dwi),
        'roi_names_file': str(roi_names_file),
        'available_transforms': transformer.get_available_transforms()
    }

    # Save metadata
    metadata_file = output_dir / f'{atlas_name}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Atlas preparation complete for {subject}")
    logger.info(f"  Atlas: {atlas_dwi}")
    logger.info(f"  ROIs: {len(roi_names)}")

    return atlas_dwi, roi_names, metadata


def list_available_atlases() -> Dict[str, Dict]:
    """List all available atlas configurations."""
    return ATLAS_CONFIGS.copy()
