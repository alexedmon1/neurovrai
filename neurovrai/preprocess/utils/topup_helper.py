#!/usr/bin/env python3
"""
Helper utilities for generating TOPUP acquisition parameter files.

This module provides functions to create acqparams.txt and index.txt files
required by FSL's TOPUP and eddy correction tools.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import nibabel as nib
import numpy as np


def get_nvols(nifti_file: Path) -> int:
    """
    Get number of volumes in a 4D NIfTI file.

    Parameters
    ----------
    nifti_file : Path
        Path to NIfTI file

    Returns
    -------
    int
        Number of volumes (timepoints)
    """
    img = nib.load(str(nifti_file))
    shape = img.shape
    if len(shape) == 4:
        return shape[3]
    return 1


def create_acqparams_file(
    pe_directions: List[str],
    readout_times: List[float],
    output_file: Path
) -> Path:
    """
    Create acqparams.txt file for TOPUP/eddy.

    The acqparams file specifies the phase encoding direction and readout time
    for each unique acquisition in your data.

    Parameters
    ----------
    pe_directions : list of str
        Phase encoding directions. Valid values:
        - 'AP' or 'j-' : Anterior-Posterior (0 -1 0)
        - 'PA' or 'j' : Posterior-Anterior (0 1 0)
        - 'LR' or 'i-' : Left-Right (-1 0 0)
        - 'RL' or 'i' : Right-Left (1 0 0)
    readout_times : list of float
        Total readout time (in seconds) for each acquisition.
        Typically 0.04-0.08 for modern sequences.
    output_file : Path
        Output path for acqparams.txt

    Returns
    -------
    Path
        Path to created acqparams.txt file

    Examples
    --------
    >>> # Two acquisitions: main DWI (AP) and reverse PE (PA)
    >>> create_acqparams_file(
    ...     pe_directions=['AP', 'PA'],
    ...     readout_times=[0.05, 0.05],
    ...     output_file=Path('acqparams.txt')
    ... )

    Notes
    -----
    The acqparams.txt format is:
        PE_x PE_y PE_z readout_time

    For TOPUP, you need at least 2 lines (forward and reverse PE).
    For eddy, all volumes should map to one of these lines via index.txt.
    """
    # Map direction strings to vectors
    direction_map = {
        'AP': (0, -1, 0),
        'j-': (0, -1, 0),
        'PA': (0, 1, 0),
        'j': (0, 1, 0),
        'LR': (-1, 0, 0),
        'i-': (-1, 0, 0),
        'RL': (1, 0, 0),
        'i': (1, 0, 0),
    }

    output_file = Path(output_file)

    with open(output_file, 'w') as f:
        for pe_dir, readout in zip(pe_directions, readout_times):
            if pe_dir not in direction_map:
                raise ValueError(
                    f"Invalid PE direction: {pe_dir}. "
                    f"Valid options: {list(direction_map.keys())}"
                )

            pe_vec = direction_map[pe_dir]
            f.write(f"{pe_vec[0]} {pe_vec[1]} {pe_vec[2]} {readout:.6f}\n")

    print(f"Created acqparams file: {output_file}")
    print(f"  {len(pe_directions)} unique acquisitions")

    return output_file


def create_index_file(
    dwi_files: List[Path],
    acqparams_indices: List[int],
    output_file: Path
) -> Path:
    """
    Create index.txt file for eddy correction.

    The index file maps each volume in the merged DWI data to a line number
    in the acqparams.txt file.

    Parameters
    ----------
    dwi_files : list of Path
        List of DWI NIfTI files (in merge order)
    acqparams_indices : list of int
        For each DWI file, the 1-indexed line number in acqparams.txt.
        Example: If both shells use AP direction (line 1), use [1, 1]
    output_file : Path
        Output path for index.txt

    Returns
    -------
    Path
        Path to created index.txt file

    Examples
    --------
    >>> # Two shells, both acquired with AP direction (line 1 in acqparams)
    >>> create_index_file(
    ...     dwi_files=[Path('b1000.nii.gz'), Path('b2000.nii.gz')],
    ...     acqparams_indices=[1, 1],
    ...     output_file=Path('index.txt')
    ... )

    Notes
    -----
    - Indices are 1-based (not 0-based)
    - The index file should have as many entries as total volumes in merged DWI
    - All volumes from one acquisition get the same index
    """
    output_file = Path(output_file)

    if len(dwi_files) != len(acqparams_indices):
        raise ValueError(
            f"Number of DWI files ({len(dwi_files)}) must match "
            f"number of acqparams indices ({len(acqparams_indices)})"
        )

    indices = []
    for dwi_file, acqparam_idx in zip(dwi_files, acqparams_indices):
        nvols = get_nvols(dwi_file)
        indices.extend([acqparam_idx] * nvols)
        print(f"  {dwi_file.name}: {nvols} volumes -> acqparams line {acqparam_idx}")

    # Write as space-separated single line
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, indices)))

    print(f"Created index file: {output_file}")
    print(f"  Total volumes: {len(indices)}")

    return output_file


def create_topup_files_for_multishell(
    dwi_files: List[Path],
    pe_direction: str,
    readout_time: float,
    output_dir: Path,
    acqparams_name: str = 'acqparams.txt',
    index_name: str = 'index.txt'
) -> Tuple[Path, Path]:
    """
    Convenience function to create both acqparams.txt and index.txt for standard
    multi-shell DWI with reverse PE images.

    This assumes:
    - All DWI shells acquired with same PE direction
    - You have reverse PE images for TOPUP
    - Standard setup: acqparams has 2 lines (forward PE, reverse PE)

    Parameters
    ----------
    dwi_files : list of Path
        DWI files to be merged
    pe_direction : str
        Phase encoding direction of main DWI acquisition ('AP', 'PA', 'LR', 'RL')
    readout_time : float
        Total readout time in seconds (typically 0.04-0.08)
    output_dir : Path
        Output directory for files
    acqparams_name : str
        Filename for acqparams file
    index_name : str
        Filename for index file

    Returns
    -------
    tuple
        (acqparams_file, index_file) paths

    Examples
    --------
    >>> # Two-shell DWI with AP phase encoding
    >>> acqparams, index = create_topup_files_for_multishell(
    ...     dwi_files=[Path('b1000.nii.gz'), Path('b2000.nii.gz')],
    ...     pe_direction='AP',
    ...     readout_time=0.05,
    ...     output_dir=Path('/data/dwi')
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine reverse PE direction
    reverse_map = {
        'AP': 'PA',
        'PA': 'AP',
        'LR': 'RL',
        'RL': 'LR',
        'j-': 'j',
        'j': 'j-',
        'i-': 'i',
        'i': 'i-',
    }

    if pe_direction not in reverse_map:
        raise ValueError(f"Unknown PE direction: {pe_direction}")

    reverse_pe = reverse_map[pe_direction]

    print(f"\n{'='*60}")
    print("Creating TOPUP acquisition parameter files")
    print(f"{'='*60}")
    print(f"PE direction: {pe_direction} (reverse: {reverse_pe})")
    print(f"Readout time: {readout_time}s")
    print(f"Output directory: {output_dir}")
    print()

    # Create acqparams file
    # Line 1: Main DWI PE direction
    # Line 2: Reverse PE direction (for TOPUP)
    acqparams_file = create_acqparams_file(
        pe_directions=[pe_direction, reverse_pe],
        readout_times=[readout_time, readout_time],
        output_file=output_dir / acqparams_name
    )

    print()

    # Create index file
    # All DWI volumes map to line 1 (main PE direction)
    index_file = create_index_file(
        dwi_files=dwi_files,
        acqparams_indices=[1] * len(dwi_files),  # All use line 1
        output_file=output_dir / index_name
    )

    print(f"\n{'='*60}")
    print("Files created successfully!")
    print(f"{'='*60}")
    print(f"acqparams: {acqparams_file}")
    print(f"index: {index_file}")
    print()
    print("Next steps:")
    print("1. Review the generated files")
    print("2. Update your config to point to these files:")
    print(f"   diffusion:")
    print(f"     topup:")
    print(f"       encoding_file: {acqparams_file}")
    print(f"     eddy:")
    print(f"       acqp_file: {acqparams_file}")
    print(f"       index_file: {index_file}")

    return acqparams_file, index_file


# Example usage
if __name__ == '__main__':
    import sys

    # Example: Create files for IRC805 study
    study_dir = Path('/mnt/bytopia/IRC805')
    dwi_dir = study_dir / 'subjects' / 'IRC805_001' / 'nifti' / 'dwi'

    if dwi_dir.exists():
        # Find DWI files
        from glob import glob
        dwi_files = [
            Path(f) for f in sorted(glob(str(dwi_dir / '*DTI*shell*.nii.gz')))
        ]

        if dwi_files:
            print(f"Found {len(dwi_files)} DWI files:")
            for f in dwi_files:
                print(f"  - {f.name}")

            # Create parameter files
            # Adjust PE direction and readout time based on your protocol
            acqparams, index = create_topup_files_for_multishell(
                dwi_files=dwi_files,
                pe_direction='AP',  # Change to match your acquisition
                readout_time=0.05,   # Adjust based on your protocol
                output_dir=study_dir / 'dwi_params'
            )
        else:
            print("No DWI files found")
    else:
        print(f"Directory not found: {dwi_dir}")
        print("\nUsage:")
        print("  Edit this script to set your study directory and DWI files")
        print("  Or import functions in your own script:")
        print("    from neurovrai.preprocess.utils.topup_helper import create_topup_files_for_multishell")
