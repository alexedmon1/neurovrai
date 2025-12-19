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
        # Use subject ID directly without adding 'sub-' prefix
        subject_dir = self.transforms_dir / self.subject

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

    def save_ants_composite_transform(
        self,
        composite_file: Path,
        source_space: str,
        target_space: str,
        reference: Optional[Path] = None,
        source_image: Optional[Path] = None
    ) -> Path:
        """
        Save an ANTs composite transformation (includes all stages).

        Parameters
        ----------
        composite_file : Path
            Path to ANTs composite transform (.h5 or .mat)
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
        Path
            Path to saved composite transform

        Examples
        --------
        >>> registry.save_ants_composite_transform(
        ...     composite_file=Path("ants_Composite.h5"),
        ...     source_space="T1w",
        ...     target_space="MNI152"
        ... )
        """
        composite_file = Path(composite_file)

        if not composite_file.exists():
            raise FileNotFoundError(f"Composite transform file not found: {composite_file}")

        # Build destination path
        key = self._get_transform_key(source_space, target_space)
        # Preserve original extension (.h5 or .mat)
        ext = composite_file.suffix
        dest_file = self.subject_dir / f"{key}_composite{ext}"

        # Copy file
        shutil.copy2(composite_file, dest_file)

        # Update metadata
        self.metadata['transforms'][key] = {
            'type': 'ants_composite',
            'method': 'ants',
            'source_space': source_space,
            'target_space': target_space,
            'composite_file': str(dest_file),
            'reference': str(reference) if reference else None,
            'source_image': str(source_image) if source_image else None,
            'created': datetime.now().isoformat()
        }

        self._save_metadata()

        return dest_file

    def get_transform_method(
        self,
        source_space: str,
        target_space: str
    ) -> Optional[str]:
        """
        Get the registration method used for a transformation.

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space

        Returns
        -------
        str or None
            'ants', 'fsl', or None if not found

        Examples
        --------
        >>> method = registry.get_transform_method("T1w", "MNI152")
        >>> if method == 'ants':
        ...     # Use ANTs tools
        >>> elif method == 'fsl':
        ...     # Use FSL tools
        """
        key = self._get_transform_key(source_space, target_space)

        if key not in self.metadata['transforms']:
            return None

        transform_info = self.metadata['transforms'][key]

        # Check explicit method field
        if 'method' in transform_info:
            return transform_info['method']

        # Infer from type for backward compatibility
        if transform_info['type'] == 'ants_composite':
            return 'ants'
        else:
            return 'fsl'  # Default for linear/nonlinear types

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
        Get a nonlinear transformation.

        For FSL: Returns (warp_file, affine_file)
        For ANTs: Returns (composite_file, composite_file) for compatibility

        Parameters
        ----------
        source_space : str
            Source coordinate space
        target_space : str
            Target coordinate space

        Returns
        -------
        tuple or None
            (warp/composite, affine/composite) or None if not found

        Examples
        --------
        >>> result = registry.get_nonlinear_transform("T1w", "MNI152")
        >>> if result:
        ...     warp, affine = result
        ...     # Check registration method to determine how to use these
        ...     method = registry.get_transform_method("T1w", "MNI152")
        """
        key = self._get_transform_key(source_space, target_space)

        if key not in self.metadata['transforms']:
            return None

        transform_info = self.metadata['transforms'][key]

        # Handle ANTs composite transforms
        if transform_info['type'] == 'ants_composite':
            composite_file = Path(transform_info['composite_file'])
            if not composite_file.exists():
                print(f"Warning: Composite transform missing: {composite_file}")
                return None
            # Return composite file as both warp and affine for compatibility
            return composite_file, composite_file

        # Handle FSL nonlinear transforms
        if transform_info['type'] != 'nonlinear':
            print(f"Warning: Transform {key} is not nonlinear or ants_composite")
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
        elif transform_info['type'] == 'ants_composite':
            composite_file = Path(transform_info['composite_file'])
            return composite_file.exists()
        else:  # nonlinear (FSL)
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

            elif info['type'] == 'ants_composite':
                composite_file = Path(info['composite_file'])
                if not composite_file.exists():
                    missing_files.append(str(composite_file))

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


# =============================================================================
# Standardized Transform Naming Convention
# =============================================================================
#
# All transforms follow: {source}-{target}-{type}.{ext}
#
# Standard transforms:
#   func-t1w-affine.mat       - Functional to T1w (affine, FSL or ANTs)
#   t1w-mni-affine.mat        - T1w to MNI (affine only)
#   t1w-mni-warp.nii.gz       - T1w to MNI (nonlinear warp field)
#   t1w-mni-composite.h5      - T1w to MNI (ANTs composite)
#   func-mni-composite.h5     - Functional to MNI (ANTs composite, combines func->t1w + t1w->mni)
#   dwi-t1w-affine.mat        - DWI to T1w
#   dwi-fmrib58-affine.mat    - DWI FA to FMRIB58 (affine)
#   dwi-fmrib58-warp.nii.gz   - DWI FA to FMRIB58 (nonlinear)
#   asl-t1w-affine.mat        - ASL to T1w
#   fs-t1w-affine.lta         - FreeSurfer to T1w
#
# Location: {study_root}/transforms/{subject}/
# =============================================================================


def get_transform_path(
    study_root: Path,
    subject: str,
    source: str,
    target: str,
    transform_type: str,
    extension: Optional[str] = None
) -> Path:
    """
    Get the standardized path for a transform file.

    Parameters
    ----------
    study_root : Path
        Study root directory
    subject : str
        Subject identifier
    source : str
        Source space (e.g., 'func', 't1w', 'dwi', 'asl', 'fs')
    target : str
        Target space (e.g., 't1w', 'mni', 'fmrib58')
    transform_type : str
        Transform type (e.g., 'affine', 'warp', 'composite')
    extension : str, optional
        File extension (auto-detected if None)

    Returns
    -------
    Path
        Full path to transform file

    Examples
    --------
    >>> path = get_transform_path(study_root, 'IRC805-0580101', 'func', 'mni', 'composite')
    >>> # Returns: {study_root}/transforms/IRC805-0580101/func-mni-composite.h5
    """
    # Auto-detect extension based on transform type
    if extension is None:
        if transform_type == 'composite':
            extension = '.h5'
        elif transform_type == 'warp':
            extension = '.nii.gz'
        elif transform_type == 'affine':
            extension = '.mat'
        else:
            extension = '.mat'

    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = '.' + extension

    filename = f"{source}-{target}-{transform_type}{extension}"
    return Path(study_root) / 'transforms' / subject / filename


def find_transform(
    study_root: Path,
    subject: str,
    source: str,
    target: str,
    prefer_composite: bool = True
) -> Optional[Path]:
    """
    Find a transform file, checking multiple types.

    Parameters
    ----------
    study_root : Path
        Study root directory
    subject : str
        Subject identifier
    source : str
        Source space
    target : str
        Target space
    prefer_composite : bool
        If True, prefer composite over separate affine+warp

    Returns
    -------
    Path or None
        Path to transform file if found

    Examples
    --------
    >>> transform = find_transform(study_root, 'IRC805-0580101', 'func', 'mni')
    >>> if transform:
    ...     print(f"Found: {transform}")
    """
    transforms_dir = Path(study_root) / 'transforms' / subject

    if not transforms_dir.exists():
        return None

    # Check order based on preference
    if prefer_composite:
        check_order = ['composite', 'warp', 'affine']
    else:
        check_order = ['affine', 'warp', 'composite']

    for xfm_type in check_order:
        if xfm_type == 'composite':
            path = transforms_dir / f"{source}-{target}-composite.h5"
        elif xfm_type == 'warp':
            path = transforms_dir / f"{source}-{target}-warp.nii.gz"
        else:
            path = transforms_dir / f"{source}-{target}-affine.mat"

        if path.exists():
            return path

    return None


def get_func_to_mni_transform(study_root: Path, subject: str) -> Optional[Path]:
    """Get functional to MNI composite transform."""
    return find_transform(study_root, subject, 'func', 'mni')


def get_t1w_to_mni_transform(study_root: Path, subject: str) -> Optional[Path]:
    """Get T1w to MNI transform (composite or warp)."""
    return find_transform(study_root, subject, 't1w', 'mni')


def get_func_to_t1w_transform(study_root: Path, subject: str) -> Optional[Path]:
    """Get functional to T1w affine transform."""
    return find_transform(study_root, subject, 'func', 't1w', prefer_composite=False)


def get_dwi_to_t1w_transform(study_root: Path, subject: str) -> Optional[Path]:
    """Get DWI to T1w affine transform."""
    return find_transform(study_root, subject, 'dwi', 't1w', prefer_composite=False)


def get_asl_to_t1w_transform(study_root: Path, subject: str) -> Optional[Path]:
    """Get ASL to T1w affine transform."""
    return find_transform(study_root, subject, 'asl', 't1w', prefer_composite=False)


def list_available_transforms(study_root: Path, subject: str) -> List[Dict]:
    """
    List all available transforms for a subject.

    Returns
    -------
    list
        List of dicts with 'source', 'target', 'type', 'path' keys
    """
    transforms_dir = Path(study_root) / 'transforms' / subject

    if not transforms_dir.exists():
        return []

    transforms = []
    for f in transforms_dir.glob('*'):
        if f.name == 'transforms.json':
            continue

        # Parse filename: source-target-type.ext
        name = f.stem
        if name.endswith('.nii'):  # Handle .nii.gz
            name = name[:-4]

        parts = name.split('-')
        if len(parts) >= 3:
            transforms.append({
                'source': parts[0],
                'target': parts[1],
                'type': '-'.join(parts[2:]),
                'path': f
            })

    return transforms


def save_transform(
    transform_file: Path,
    study_root: Path,
    subject: str,
    source: str,
    target: str,
    transform_type: str
) -> Path:
    """
    Save a transform to the standardized location.

    Parameters
    ----------
    transform_file : Path
        Source transform file to copy
    study_root : Path
        Study root directory
    subject : str
        Subject identifier
    source : str
        Source space
    target : str
        Target space
    transform_type : str
        Transform type

    Returns
    -------
    Path
        Destination path
    """
    transform_file = Path(transform_file)
    if not transform_file.exists():
        raise FileNotFoundError(f"Transform file not found: {transform_file}")

    # Get destination path
    dest = get_transform_path(
        study_root, subject, source, target, transform_type,
        extension=transform_file.suffix if not transform_file.name.endswith('.nii.gz') else '.nii.gz'
    )

    # Create directory if needed
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Copy file
    shutil.copy2(transform_file, dest)

    return dest
