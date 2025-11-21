#!/usr/bin/env python3
"""
Test TBSS Workflow with Synthetic Data

Creates synthetic test data to validate the complete TBSS analysis pipeline:
1. Generates fake FA maps for synthetic subjects
2. Creates participants CSV with demographic data
3. Creates contrasts YAML file
4. Runs prepare_tbss.py (data preparation)
5. Runs run_tbss_stats.py (statistical analysis)

This test validates the integration of all components without requiring real data.
"""

import json
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml


def create_synthetic_fa_data(
    output_dir: Path,
    n_subjects: int = 20,
    img_shape: tuple = (20, 20, 20),
    add_signal: bool = True
) -> list:
    """
    Create synthetic FA maps for test subjects

    Args:
        output_dir: Output directory for synthetic data
        n_subjects: Number of subjects to create
        img_shape: Shape of 3D FA volumes
        add_signal: Whether to add realistic signal patterns

    Returns:
        List of subject IDs created
    """
    print(f"Creating synthetic FA data for {n_subjects} subjects...")

    subjects = []

    for i in range(1, n_subjects + 1):
        subject_id = f"test-sub-{i:03d}"
        subjects.append(subject_id)

        # Create subject directory
        subject_dir = output_dir / subject_id / "dwi" / "dti"
        subject_dir.mkdir(parents=True, exist_ok=True)

        # Generate FA data
        if add_signal:
            # Create realistic FA pattern with some structure
            fa_data = np.random.uniform(0.2, 0.8, img_shape)

            # Add white matter-like structure in center
            center = np.array(img_shape) // 2
            for x in range(img_shape[0]):
                for y in range(img_shape[1]):
                    for z in range(img_shape[2]):
                        dist = np.sqrt((x - center[0])**2 +
                                     (y - center[1])**2 +
                                     (z - center[2])**2)
                        if dist < 5:
                            fa_data[x, y, z] += 0.2  # Higher FA in "white matter"

            # Clip to valid FA range
            fa_data = np.clip(fa_data, 0, 1)
        else:
            # Simple random FA values
            fa_data = np.random.uniform(0.3, 0.7, img_shape)

        # Create NIfTI image
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0  # 2mm isotropic
        img = nib.Nifti1Image(fa_data, affine)

        # Save
        fa_file = subject_dir / "FA.nii.gz"
        nib.save(img, fa_file)

    print(f"✓ Created {n_subjects} synthetic FA maps in {output_dir}")
    return subjects


def create_participants_csv(
    subjects: list,
    output_file: Path,
    add_missing: int = 2
) -> pd.DataFrame:
    """
    Create participants CSV with demographic data

    Args:
        subjects: List of subject IDs
        output_file: Path to save CSV
        add_missing: Number of subjects to exclude (simulate missing data)

    Returns:
        DataFrame with participant data
    """
    print(f"\nCreating participants CSV with {len(subjects)} subjects...")

    # Generate demographic data
    data = {
        'subject_id': subjects,
        'age': np.random.randint(25, 75, len(subjects)),
        'sex': np.random.choice(['M', 'F'], len(subjects)),
        'group': np.random.choice(['control', 'patient'], len(subjects)),
        'exposure': np.random.uniform(0, 10, len(subjects))
    }

    df = pd.DataFrame(data)

    # Remove some subjects to simulate missing data
    if add_missing > 0:
        df = df.iloc[:-add_missing]
        print(f"  Removed {add_missing} subjects to simulate missing data")

    df.to_csv(output_file, index=False)
    print(f"✓ Created participants CSV: {output_file}")
    print(f"  Total subjects: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    return df


def create_contrasts_yaml(output_file: Path):
    """
    Create contrasts YAML file for statistical tests

    Args:
        output_file: Path to save YAML file
    """
    print(f"\nCreating contrasts YAML...")

    contrasts_data = {
        'contrasts': [
            {
                'name': 'age_positive',
                'vector': [0, 1, 0, 0],  # intercept, age, sex_M, exposure
                'description': 'Positive association with age'
            },
            {
                'name': 'sex_MvsF',
                'vector': [0, 0, 1, 0],
                'description': 'Male vs Female'
            },
            {
                'name': 'exposure_negative',
                'vector': [0, 0, 0, -1],
                'description': 'Negative association with exposure'
            }
        ]
    }

    with open(output_file, 'w') as f:
        yaml.dump(contrasts_data, f, default_flow_style=False)

    print(f"✓ Created contrasts YAML: {output_file}")
    print(f"  Number of contrasts: {len(contrasts_data['contrasts'])}")


def create_test_data(output_dir: Path) -> dict:
    """
    Create all test data needed for TBSS workflow

    Args:
        output_dir: Base directory for test data

    Returns:
        Dictionary with paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CREATING SYNTHETIC TEST DATA FOR TBSS")
    print("=" * 80)

    # Create directory structure
    derivatives_dir = output_dir / "derivatives"
    analysis_dir = output_dir / "analysis"

    # Create synthetic FA data
    subjects = create_synthetic_fa_data(
        output_dir=derivatives_dir,
        n_subjects=20,
        img_shape=(20, 20, 20),
        add_signal=True
    )

    # Create participants CSV (with 2 missing subjects)
    participants_file = output_dir / "participants.csv"
    participants_df = create_participants_csv(
        subjects=subjects,
        output_file=participants_file,
        add_missing=2
    )

    # Create contrasts YAML
    contrasts_file = output_dir / "contrasts.yaml"
    create_contrasts_yaml(contrasts_file)

    print("\n" + "=" * 80)
    print("TEST DATA CREATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  Derivatives: {derivatives_dir}")
    print(f"  Participants CSV: {participants_file}")
    print(f"  Contrasts YAML: {contrasts_file}")
    print("=" * 80)

    return {
        'output_dir': output_dir,
        'derivatives_dir': derivatives_dir,
        'analysis_dir': analysis_dir,
        'participants_file': participants_file,
        'contrasts_file': contrasts_file,
        'subjects': subjects,
        'n_subjects_with_data': len(subjects),
        'n_subjects_in_csv': len(participants_df)
    }


if __name__ == '__main__':
    # Create test data in temporary directory
    test_dir = Path('/tmp/tbss_test')

    # Clean up if exists
    if test_dir.exists():
        print(f"Cleaning up existing test directory: {test_dir}")
        shutil.rmtree(test_dir)

    # Create test data
    test_data = create_test_data(test_dir)

    # Save test data info
    info_file = test_dir / "test_data_info.json"
    with open(info_file, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        json_data = {k: str(v) if isinstance(v, Path) else v
                     for k, v in test_data.items()}
        json.dump(json_data, f, indent=2)

    print(f"\nTest data info saved to: {info_file}")
    print("\nNext steps:")
    print("1. Review the generated data")
    print("2. Run prepare_tbss.py on the synthetic data")
    print("3. Run run_tbss_stats.py to complete the workflow")
    print("\nExample commands:")
    print(f"  # Prepare TBSS data")
    print(f"  python -m neurovrai.analysis.tbss.prepare_tbss \\")
    print(f"      --config config.yaml \\")
    print(f"      --metric FA \\")
    print(f"      --output-dir {test_data['analysis_dir']}/tbss_FA/")
    print()
    print(f"  # Run statistical analysis")
    print(f"  python -m neurovrai.analysis.tbss.run_tbss_stats \\")
    print(f"      --data-dir {test_data['analysis_dir']}/tbss_FA/ \\")
    print(f"      --participants {test_data['participants_file']} \\")
    print(f"      --formula 'age + sex + exposure' \\")
    print(f"      --contrasts-file {test_data['contrasts_file']} \\")
    print(f"      --output-dir {test_data['analysis_dir']}/tbss_FA/model1/ \\")
    print(f"      --n-permutations 100")  # Fewer for testing
