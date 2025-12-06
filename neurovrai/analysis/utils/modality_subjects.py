#!/usr/bin/env python3
"""
Modality-Aware Subject Detection

Automatically detects which subjects have data for specific MRI modalities
and filters participants data accordingly.

This ensures design matrices only include subjects with available MRI data
and maintains correct subject ordering.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def find_subjects_for_modality(
    study_root: Path,
    modality: str,
    metric: str = None
) -> List[str]:
    """
    Find all subjects that have MRI data for a specific modality

    Args:
        study_root: Study root directory (e.g., /mnt/bytopia/IRC805)
        modality: Modality type ('vbm', 'asl', 'reho', 'falff', 'tbss')
        metric: For TBSS, specify metric (e.g., 'FA', 'MD', 'MK', etc.)

    Returns:
        Sorted list of subject IDs with data for this modality

    Examples:
        >>> find_subjects_for_modality(Path('/mnt/bytopia/IRC805'), 'vbm')
        ['IRC805-0580101', 'IRC805-1580101', ...]

        >>> find_subjects_for_modality(Path('/mnt/bytopia/IRC805'), 'tbss', 'FA')
        ['IRC805-0580101', 'IRC805-1640101', ...]
    """
    study_root = Path(study_root)
    subjects = []

    if modality == 'vbm':
        # Look for VBM smoothed GM files
        vbm_subjects_dir = study_root / 'analysis' / 'anat' / 'vbm' / 'subjects'
        if vbm_subjects_dir.exists():
            for subj_file in vbm_subjects_dir.glob('IRC805-*_GM_mni_smooth.nii.gz'):
                # Extract subject ID: IRC805-XXXXXXX from IRC805-XXXXXXX_GM_mni_smooth.nii.gz
                subject_id = subj_file.name.split('_GM_mni_smooth')[0]
                subjects.append(subject_id)
        else:
            logger.warning(f"VBM subjects directory not found: {vbm_subjects_dir}")

    elif modality == 'asl':
        # Look for ASL CBF MNI files
        derivatives_dir = study_root / 'derivatives'
        for subj_dir in sorted(derivatives_dir.glob('IRC805-*/asl')):
            subject_id = subj_dir.parent.name
            cbf_file = subj_dir / f'{subject_id}_cbf_mni.nii.gz'
            if cbf_file.exists():
                subjects.append(subject_id)

    elif modality == 'reho':
        # Look for ReHo MNI files
        derivatives_dir = study_root / 'derivatives'
        for subj_dir in sorted(derivatives_dir.glob('IRC805-*/func')):
            subject_id = subj_dir.parent.name
            reho_file = subj_dir / 'reho' / 'reho_mni_zscore_masked.nii.gz'
            if reho_file.exists():
                subjects.append(subject_id)

    elif modality == 'falff':
        # Look for fALFF MNI files
        derivatives_dir = study_root / 'derivatives'
        for subj_dir in sorted(derivatives_dir.glob('IRC805-*/func')):
            subject_id = subj_dir.parent.name
            falff_file = subj_dir / 'falff' / 'falff_mni_zscore_masked.nii.gz'
            if falff_file.exists():
                subjects.append(subject_id)

    elif modality == 'tbss':
        # Look for TBSS metric files
        if metric is None:
            raise ValueError("TBSS modality requires 'metric' parameter (e.g., 'FA', 'MD')")

        tbss_metric_dir = study_root / 'analysis' / 'tbss' / metric
        if tbss_metric_dir.exists():
            for metric_file in sorted(tbss_metric_dir.glob(f'IRC805-*_{metric}.nii.gz')):
                # Extract subject ID: IRC805-XXXXXXX from IRC805-XXXXXXX_FA.nii.gz
                subject_id = metric_file.name.split(f'_{metric}.nii.gz')[0]
                subjects.append(subject_id)
        else:
            logger.warning(f"TBSS {metric} directory not found: {tbss_metric_dir}")

    else:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Supported: vbm, asl, reho, falff, tbss"
        )

    # Return sorted list (ensures consistent ordering)
    return sorted(subjects)


def load_participants_for_modality(
    participants_file: Path,
    study_root: Path,
    modality: str,
    metric: str = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load participants data and filter to subjects with MRI data for modality

    Args:
        participants_file: Path to master participants file (e.g., gludata.csv)
        study_root: Study root directory
        modality: Modality type ('vbm', 'asl', 'reho', 'falff', 'tbss')
        metric: For TBSS, specify metric (e.g., 'FA', 'MD')

    Returns:
        Tuple of (filtered DataFrame, list of subject IDs in order)

    Examples:
        >>> df, subjects = load_participants_for_modality(
        ...     Path('gludata.csv'),
        ...     Path('/mnt/bytopia/IRC805'),
        ...     'vbm'
        ... )
        >>> len(df)
        23
        >>> subjects[:3]
        ['IRC805-0580101', 'IRC805-1580101', 'IRC805-1640101']
    """
    # Find subjects with MRI data for this modality
    subjects_with_data = find_subjects_for_modality(study_root, modality, metric)

    if len(subjects_with_data) == 0:
        raise ValueError(
            f"No subjects found with {modality} data. "
            f"Check that data exists at {study_root}"
        )

    logger.info(f"Found {len(subjects_with_data)} subjects with {modality} data")

    # Load participants file
    if participants_file.suffix == '.csv':
        df = pd.read_csv(participants_file)
    else:
        df = pd.read_csv(participants_file, sep='\t')

    # Create participant_id column if needed
    if 'participant_id' not in df.columns:
        if 'Subject' in df.columns:
            # gludata.csv format: Subject column with numeric IDs
            # IMPORTANT: Pad with zeros to 7 digits (e.g., 580101 -> 0580101)
            df['participant_id'] = 'IRC805-' + df['Subject'].astype(str).str.zfill(7)
        else:
            raise ValueError(
                "Participants file must have 'participant_id' or 'Subject' column"
            )

    # Standardize column names (handle case variations)
    column_mapping = {}
    for col in df.columns:
        if col.lower() == 'age' and col != 'age':
            column_mapping[col] = 'age'
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info(f"Standardized column names: {column_mapping}")

    # Filter to subjects with MRI data
    df_filtered = df[df['participant_id'].isin(subjects_with_data)].copy()

    # Sort by participant_id to ensure consistent ordering
    df_filtered = df_filtered.sort_values('participant_id').reset_index(drop=True)

    # Get ordered subject list
    ordered_subjects = df_filtered['participant_id'].tolist()

    logger.info(f"Filtered participants to {len(df_filtered)} subjects with {modality} data")

    # Check for missing subjects
    missing_subjects = set(subjects_with_data) - set(ordered_subjects)
    if missing_subjects:
        logger.warning(
            f"{len(missing_subjects)} subjects with MRI data not found in participants file: "
            f"{sorted(missing_subjects)[:5]}..."
        )

    return df_filtered, ordered_subjects


def save_subject_list(
    subject_ids: List[str],
    output_file: Path,
    modality: str = None
):
    """
    Save ordered subject list to file

    Creates a text file documenting the exact order of subjects in the design matrix.
    This is critical for FSL randomise, which requires subject order to match
    the rows in the design matrix and the volumes in the 4D merged image.

    Args:
        subject_ids: List of subject IDs in order
        output_file: Output file path (e.g., 'subject_order.txt')
        modality: Optional modality name for header

    Example file format:
        # Subject order for VBM analysis
        # N=23 subjects
        # Generated: 2025-12-05
        IRC805-0580101
        IRC805-1580101
        IRC805-1640101
        ...
    """
    from datetime import datetime

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# Subject order")
        if modality:
            f.write(f" for {modality} analysis")
        f.write("\n")
        f.write(f"# N={len(subject_ids)} subjects\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        f.write("# CRITICAL: This order MUST match:\n")
        f.write("#   1. Rows in design.mat\n")
        f.write("#   2. Volumes in 4D merged image\n")
        f.write("#   3. Subject order in analysis scripts\n")
        f.write("#\n")

        # Write subjects
        for subject_id in subject_ids:
            f.write(f"{subject_id}\n")

    logger.info(f"Saved subject list ({len(subject_ids)} subjects) to {output_file}")


def get_modality_info(modality: str) -> dict:
    """
    Get information about a specific modality

    Args:
        modality: Modality name

    Returns:
        Dictionary with modality metadata
    """
    modality_info = {
        'vbm': {
            'name': 'Voxel-Based Morphometry',
            'short_name': 'VBM',
            'data_type': 'structural',
            'typical_n': '20-30',
            'file_pattern': '*_GM_mni_smooth.nii.gz'
        },
        'asl': {
            'name': 'Arterial Spin Labeling',
            'short_name': 'ASL',
            'data_type': 'perfusion',
            'typical_n': '15-25',
            'file_pattern': '*_cbf_mni.nii.gz'
        },
        'reho': {
            'name': 'Regional Homogeneity',
            'short_name': 'ReHo',
            'data_type': 'functional',
            'typical_n': '15-25',
            'file_pattern': 'reho_mni_zscore_masked.nii.gz'
        },
        'falff': {
            'name': 'Fractional ALFF',
            'short_name': 'fALFF',
            'data_type': 'functional',
            'typical_n': '15-25',
            'file_pattern': 'falff_mni_zscore_masked.nii.gz'
        },
        'tbss': {
            'name': 'Tract-Based Spatial Statistics',
            'short_name': 'TBSS',
            'data_type': 'diffusion',
            'typical_n': '15-25',
            'file_pattern': 'IRC805-*_{metric}.nii.gz'
        }
    }

    return modality_info.get(modality, {
        'name': modality,
        'short_name': modality.upper(),
        'data_type': 'unknown',
        'typical_n': 'N/A',
        'file_pattern': 'unknown'
    })
