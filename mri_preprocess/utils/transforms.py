#!/usr/bin/env python3
"""
Transformation registry for managing coordinate space transformations.

Centralizes storage and retrieval of spatial transformations (e.g., T1w→MNI)
to avoid duplicate computation across preprocessing workflows.

Key principle: Compute once, reuse everywhere.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import json
from datetime import datetime
import shutil


class TransformRegistry:
    """
    Registry for storing and retrieving spatial transformations.

    Manages transformations between different coordinate spaces,
    ensuring each transformation is computed only once and reused
    across all workflows that need it.

    Parameters
    ----------
    transforms_dir : Path
        Base directory for storing transformations
    subject : str
        Subject identifier
    session : str, optional
        Session identifier (for multi-session studies)

    Examples
    --------
    >>> registry = TransformRegistry(Path("/data/transforms"), "sub-001")
    >>> registry.save_linear_transform(
    ...     transform_file=Path("t1w_to_mni_affine.mat"),
    ...     source_space="T1w",
    ...     target_space="MNI152",
    ...     reference=Path("MNI152_T1_2mm_brain.nii.gz")
    ... )
    >>> affine = registry.get_linear_transform("T1w", "MNI152")
    """

    def __init__(
        self,
        transforms_dir: Path,
        subject: str,
        session: Optional[str] = None
    ):
        self.transforms_dir = Path(transforms_dir)
        self.subject = subject
        self.session = session

        # Create subject-specific transform directory
        self.subject_dir = self._get_subject_dir()
        self.subject_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize registry metadata
        self.metadata_file = self.subject_dir / 'transforms.json'
        self.metadata = self._load_metadata()

    def _get_subject_dir(self) -> Path:
        """Get subject-specific transforms directory."""
        if not self.subject.startswith('sub-'):
            subject = f'sub-{self.subject}'
        else:
            subject = self.subject

        subject_dir = self.transforms_dir / subject

        if self.session:
            if not self.session.startswith('ses-'):
                session = f'ses-{self.session}'
            else:
                session = self.session
            subject_dir = subject_dir / session

        return subject_dir

    def _load_metadata(self) -> Dict:
        """Load transformation metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'subject': self.subject,
                'session': self.session,
                'created': datetime.now().isoformat(),
                'transforms': {}
            }

    def _save_metadata(self):
        """Save transformation metadata to JSON file."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_transform_key(self, source_space: str, target_space: str) -> str:
        """Generate unique key for transformation."""
        return f"{source_space}_to_{target_space}"

    def save_linear_transform(
        self,
        transform_file: Path,
        source_space: str,
        target_space: str,
        reference: Optional[Path] = None,
        source_image: Optional[Path] = None
    ) -> Path:
        """
        Save a linear (affine) transformation matrix.

        Parameters
        ----------
        transform_file : Path
            Path to FLIRT .mat file
        source_space : str
            Source coordinate space (e.g., 'T1w', 'DWI')
        target_space : str
            Target coordinate space (e.g., 'MNI152', 'T1w')
        reference : Path, optional
            Reference image used for transformation
        source_image : Path, optional
            Source image that was transformed

        Returns
        -------
        Path
            Path to saved transformation file

        Examples
        --------
        >>> registry.save_linear_transform(
        ...     transform_file=Path("t1w_to_mni_affine.mat"),
        ...     source_space="T1w",
        ...     target_space="MNI152"
        ... )
        """
        transform_file = Path(transform_file)

        if not transform_file.exists():
            raise FileNotFoundError(f"Transform file not found: {transform_file}")

        # Build destination path
        key = self._get_transform_key(source_space, target_space)
        dest_file = self.subject_dir / f"{key}_affine.mat"

        # Copy transform file
        shutil.copy2(transform_file, dest_file)

        # Update metadata
        self.metadata['transforms'][key] = {
            'type': 'linear',
            'source_space': source_space,
            'target_space': target_space,
            'affine_file': str(dest_file),
            'reference': str(reference) if reference else None,
            'source_image': str(source_image) if source_image else None,
            'created': datetime.now().isoformat()
        }

        self._save_metadata()

        return dest_file

    def save_nonlinear_transform(
        self,
        warp_file: Path,
        affine_file: Path,
        source_space: str,
        target_space: str,
        reference: Optional[Path] = None,
        source_image: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """
        Save a nonlinear transformation (warp field + affine).

        Parameters
        ----------
        warp_file : Path
            Path to FNIRT warp field (.nii.gz)
        affine_file : Path
            Path to affine matrix (.mat) used with warp
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space
        reference : Path, optional
            Reference image
        source_image : Path, optional
            Source image

        Returns
        -------
        tuple
            (warp_path, affine_path) - paths to saved files

        Examples
        --------
        >>> registry.save_nonlinear_transform(
        ...     warp_file=Path("t1w_to_mni_warp.nii.gz"),
        ...     affine_file=Path("t1w_to_mni_affine.mat"),
        ...     source_space="T1w",
        ...     target_space="MNI152"
        ... )
        """
        warp_file = Path(warp_file)
        affine_file = Path(affine_file)

        if not warp_file.exists():
            raise FileNotFoundError(f"Warp file not found: {warp_file}")
        if not affine_file.exists():
            raise FileNotFoundError(f"Affine file not found: {affine_file}")

        # Build destination paths
        key = self._get_transform_key(source_space, target_space)
        dest_warp = self.subject_dir / f"{key}_warp.nii.gz"
        dest_affine = self.subject_dir / f"{key}_affine.mat"

        # Copy files
        shutil.copy2(warp_file, dest_warp)
        shutil.copy2(affine_file, dest_affine)

        # Update metadata
        self.metadata['transforms'][key] = {
            'type': 'nonlinear',
            'source_space': source_space,
            'target_space': target_space,
            'warp_file': str(dest_warp),
            'affine_file': str(dest_affine),
            'reference': str(reference) if reference else None,
            'source_image': str(source_image) if source_image else None,
            'created': datetime.now().isoformat()
        }

        self._save_metadata()

        return dest_warp, dest_affine

    def get_linear_transform(
        self,
        source_space: str,
        target_space: str
    ) -> Optional[Path]:
        """
        Get a linear transformation matrix.

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space

        Returns
        -------
        Path or None
            Path to transformation file, or None if not found

        Examples
        --------
        >>> affine = registry.get_linear_transform("T1w", "MNI152")
        >>> if affine:
        ...     # Use transformation
        ...     pass
        """
        key = self._get_transform_key(source_space, target_space)

        if key not in self.metadata['transforms']:
            return None

        transform_info = self.metadata['transforms'][key]
        affine_file = Path(transform_info['affine_file'])

        if not affine_file.exists():
            print(f"Warning: Transform file missing: {affine_file}")
            return None

        return affine_file

    def get_nonlinear_transform(
        self,
        source_space: str,
        target_space: str
    ) -> Optional[Tuple[Path, Path]]:
        """
        Get a nonlinear transformation (warp + affine).

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space

        Returns
        -------
        tuple or None
            (warp_file, affine_file) or None if not found

        Examples
        --------
        >>> result = registry.get_nonlinear_transform("T1w", "MNI152")
        >>> if result:
        ...     warp, affine = result
        ...     # Use transformations
        """
        key = self._get_transform_key(source_space, target_space)

        if key not in self.metadata['transforms']:
            return None

        transform_info = self.metadata['transforms'][key]

        if transform_info['type'] != 'nonlinear':
            print(f"Warning: Transform {key} is not nonlinear")
            return None

        warp_file = Path(transform_info['warp_file'])
        affine_file = Path(transform_info['affine_file'])

        if not warp_file.exists() or not affine_file.exists():
            print(f"Warning: Transform files missing: {warp_file}, {affine_file}")
            return None

        return warp_file, affine_file

    def has_transform(
        self,
        source_space: str,
        target_space: str,
        transform_type: Optional[str] = None
    ) -> bool:
        """
        Check if a transformation exists.

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space
        transform_type : str, optional
            Type of transform ('linear' or 'nonlinear')
            If None, checks for any type

        Returns
        -------
        bool
            True if transformation exists

        Examples
        --------
        >>> if registry.has_transform("T1w", "MNI152", "nonlinear"):
        ...     print("T1w→MNI nonlinear transform available")
        """
        key = self._get_transform_key(source_space, target_space)

        if key not in self.metadata['transforms']:
            return False

        transform_info = self.metadata['transforms'][key]

        # Check type if specified
        if transform_type and transform_info['type'] != transform_type:
            return False

        # Verify files exist
        if transform_info['type'] == 'linear':
            affine_file = Path(transform_info['affine_file'])
            return affine_file.exists()
        else:  # nonlinear
            warp_file = Path(transform_info['warp_file'])
            affine_file = Path(transform_info['affine_file'])
            return warp_file.exists() and affine_file.exists()

    def list_transforms(self) -> List[Dict]:
        """
        List all available transformations.

        Returns
        -------
        list
            List of transformation info dictionaries

        Examples
        --------
        >>> transforms = registry.list_transforms()
        >>> for t in transforms:
        ...     print(f"{t['source_space']} → {t['target_space']} ({t['type']})")
        """
        return [
            {
                'key': key,
                **info
            }
            for key, info in self.metadata['transforms'].items()
        ]

    def get_inverse_transform(
        self,
        source_space: str,
        target_space: str
    ) -> Optional[Path]:
        """
        Get inverse transformation (target→source).

        Note: Inverse transformations must be computed and saved separately.
        This method checks if the inverse exists in the registry.

        Parameters
        ----------
        source_space : str
            Original source space (becomes target for inverse)
        target_space : str
            Original target space (becomes source for inverse)

        Returns
        -------
        Path or None
            Path to inverse transformation, or None if not found

        Examples
        --------
        >>> # After computing MNI→T1w inverse
        >>> inv_warp = registry.get_inverse_transform("T1w", "MNI152")
        """
        # Swap source and target
        return self.get_linear_transform(target_space, source_space)

    def save_inverse_warp(
        self,
        inverse_warp_file: Path,
        source_space: str,
        target_space: str,
        reference: Optional[Path] = None
    ) -> Path:
        """
        Save an inverse warp field (for FNIRT nonlinear transforms).

        Parameters
        ----------
        inverse_warp_file : Path
            Path to inverse warp field
        source_space : str
            Original source space
        target_space : str
            Original target space
        reference : Path, optional
            Reference image

        Returns
        -------
        Path
            Path to saved inverse warp

        Examples
        --------
        >>> # Save MNI→T1w inverse
        >>> registry.save_inverse_warp(
        ...     inverse_warp_file=Path("mni_to_t1w_invwarp.nii.gz"),
        ...     source_space="T1w",
        ...     target_space="MNI152"
        ... )
        """
        inverse_warp_file = Path(inverse_warp_file)

        if not inverse_warp_file.exists():
            raise FileNotFoundError(f"Inverse warp file not found: {inverse_warp_file}")

        # Build destination path (swap source/target)
        key = self._get_transform_key(target_space, source_space)
        dest_file = self.subject_dir / f"{key}_invwarp.nii.gz"

        # Copy file
        shutil.copy2(inverse_warp_file, dest_file)

        # Update metadata
        self.metadata['transforms'][key] = {
            'type': 'inverse_warp',
            'source_space': target_space,  # Swapped
            'target_space': source_space,  # Swapped
            'inverse_warp_file': str(dest_file),
            'reference': str(reference) if reference else None,
            'created': datetime.now().isoformat()
        }

        self._save_metadata()

        return dest_file

    def get_metadata(self) -> Dict:
        """
        Get full registry metadata.

        Returns
        -------
        dict
            Complete metadata dictionary
        """
        return self.metadata.copy()

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate that all registered transformations exist on disk.

        Returns
        -------
        tuple
            (is_valid, missing_files) - validation result and list of missing files

        Examples
        --------
        >>> is_valid, missing = registry.validate()
        >>> if not is_valid:
        ...     print(f"Missing files: {missing}")
        """
        missing_files = []

        for key, info in self.metadata['transforms'].items():
            if info['type'] == 'linear':
                affine_file = Path(info['affine_file'])
                if not affine_file.exists():
                    missing_files.append(str(affine_file))

            elif info['type'] == 'nonlinear':
                warp_file = Path(info['warp_file'])
                affine_file = Path(info['affine_file'])
                if not warp_file.exists():
                    missing_files.append(str(warp_file))
                if not affine_file.exists():
                    missing_files.append(str(affine_file))

            elif info['type'] == 'inverse_warp':
                inv_warp_file = Path(info['inverse_warp_file'])
                if not inv_warp_file.exists():
                    missing_files.append(str(inv_warp_file))

        return len(missing_files) == 0, missing_files


def create_transform_registry(
    config: Dict,
    subject: str,
    session: Optional[str] = None
) -> TransformRegistry:
    """
    Create a TransformRegistry from configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    subject : str
        Subject identifier
    session : str, optional
        Session identifier

    Returns
    -------
    TransformRegistry
        Initialized registry

    Examples
    --------
    >>> config = load_config("study.yaml")
    >>> registry = create_transform_registry(config, "sub-001")
    """
    transforms_dir = Path(config['paths']['transforms'])
    return TransformRegistry(transforms_dir, subject, session)
