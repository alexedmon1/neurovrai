#!/usr/bin/env python3
"""
Connectome Module Test Script

Comprehensive testing of the neurovrai connectome module including:
- Synthetic data generation (functional timeseries, atlases)
- ROI extraction (discrete and probabilistic atlases)
- Functional connectivity computation
- Fisher z-transformation
- Matrix thresholding
- Partial correlation
- Output validation

Usage:
    python archive/tests/test_connectome.py

Or with uv:
    uv run python archive/tests/test_connectome.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurovrai.connectome import (
    extract_roi_timeseries,
    extract_roi_values,
    load_atlas,
    compute_functional_connectivity,
    compute_correlation_matrix,
    fisher_z_transform,
    threshold_matrix,
)


def create_synthetic_functional_data(
    output_dir: Path,
    n_timepoints: int = 200,
    image_shape: tuple = (20, 20, 20),
    tr: float = 2.0
) -> Path:
    """
    Create synthetic 4D functional MRI data with realistic properties

    Args:
        output_dir: Output directory
        n_timepoints: Number of timepoints
        image_shape: 3D spatial dimensions
        tr: Repetition time in seconds

    Returns:
        Path to created NIfTI file
    """
    print(f"\nCreating synthetic functional data...")
    print(f"  Shape: {image_shape} x {n_timepoints} timepoints")
    print(f"  TR: {tr} s")

    # Create 4D data with realistic properties
    data = np.zeros(image_shape + (n_timepoints,))

    # Add "brain" regions with different signal characteristics
    center_x, center_y, center_z = np.array(image_shape) // 2

    # Region 1: High correlation area (left hemisphere)
    region1_signal = np.random.randn(n_timepoints)
    region1_signal = np.convolve(region1_signal, np.ones(5)/5, mode='same')  # Smooth

    for x in range(center_x - 5, center_x):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 3, center_z + 3):
                noise = np.random.randn(n_timepoints) * 0.3
                data[x, y, z, :] = region1_signal + noise + 100  # Add baseline

    # Region 2: Correlated with Region 1 (right hemisphere)
    region2_signal = 0.7 * region1_signal + 0.3 * np.random.randn(n_timepoints)
    region2_signal = np.convolve(region2_signal, np.ones(5)/5, mode='same')

    for x in range(center_x + 1, center_x + 6):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 3, center_z + 3):
                noise = np.random.randn(n_timepoints) * 0.3
                data[x, y, z, :] = region2_signal + noise + 100

    # Region 3: Uncorrelated (posterior)
    region3_signal = np.random.randn(n_timepoints)
    region3_signal = np.convolve(region3_signal, np.ones(5)/5, mode='same')

    for x in range(center_x - 2, center_x + 3):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 8, center_z - 3):
                noise = np.random.randn(n_timepoints) * 0.3
                data[x, y, z, :] = region3_signal + noise + 100

    # Create NIfTI image
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 3.0  # 3mm isotropic
    img = nib.Nifti1Image(data, affine)

    # Set TR in header
    img.header.set_zooms((3.0, 3.0, 3.0, tr))

    # Save
    output_file = output_dir / "synthetic_func.nii.gz"
    nib.save(img, output_file)

    print(f"   Created: {output_file}")

    return output_file


def create_discrete_atlas(
    output_dir: Path,
    image_shape: tuple = (20, 20, 20),
    n_rois: int = 3
) -> tuple:
    """
    Create synthetic discrete (integer-labeled) atlas

    Args:
        output_dir: Output directory
        image_shape: 3D spatial dimensions
        n_rois: Number of ROIs

    Returns:
        Tuple of (atlas_file, labels_file)
    """
    print(f"\nCreating discrete atlas with {n_rois} ROIs...")

    atlas_data = np.zeros(image_shape, dtype=np.int32)

    center_x, center_y, center_z = np.array(image_shape) // 2

    # ROI 1: Left hemisphere
    for x in range(center_x - 5, center_x):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 3, center_z + 3):
                atlas_data[x, y, z] = 1

    # ROI 2: Right hemisphere
    for x in range(center_x + 1, center_x + 6):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 3, center_z + 3):
                atlas_data[x, y, z] = 2

    # ROI 3: Posterior
    for x in range(center_x - 2, center_x + 3):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 8, center_z - 3):
                atlas_data[x, y, z] = 3

    # Create NIfTI
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 3.0
    img = nib.Nifti1Image(atlas_data, affine)

    atlas_file = output_dir / "discrete_atlas.nii.gz"
    nib.save(img, atlas_file)

    # Create labels file
    labels_file = output_dir / "discrete_atlas_labels.txt"
    with open(labels_file, 'w') as f:
        f.write("1 Left_Region\n")
        f.write("2 Right_Region\n")
        f.write("3 Posterior_Region\n")

    print(f"   Atlas: {atlas_file}")
    print(f"   Labels: {labels_file}")

    return atlas_file, labels_file


def create_probabilistic_atlas(
    output_dir: Path,
    image_shape: tuple = (20, 20, 20),
    n_rois: int = 3
) -> Path:
    """
    Create synthetic probabilistic (4D) atlas

    Args:
        output_dir: Output directory
        image_shape: 3D spatial dimensions
        n_rois: Number of ROIs

    Returns:
        Path to atlas file
    """
    print(f"\nCreating probabilistic atlas with {n_rois} ROIs...")

    atlas_data = np.zeros(image_shape + (n_rois,))

    center_x, center_y, center_z = np.array(image_shape) // 2

    # ROI 0: Left hemisphere with smooth probabilities
    for x in range(center_x - 5, center_x):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 3, center_z + 3):
                dist = np.sqrt((x - (center_x - 2.5))**2 +
                             (y - center_y)**2 +
                             (z - center_z)**2)
                prob = np.exp(-dist / 2.0)
                atlas_data[x, y, z, 0] = prob

    # ROI 1: Right hemisphere
    for x in range(center_x + 1, center_x + 6):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 3, center_z + 3):
                dist = np.sqrt((x - (center_x + 3.5))**2 +
                             (y - center_y)**2 +
                             (z - center_z)**2)
                prob = np.exp(-dist / 2.0)
                atlas_data[x, y, z, 1] = prob

    # ROI 2: Posterior
    for x in range(center_x - 2, center_x + 3):
        for y in range(center_y - 3, center_y + 3):
            for z in range(center_z - 8, center_z - 3):
                dist = np.sqrt((x - center_x)**2 +
                             (y - center_y)**2 +
                             (z - (center_z - 5.5))**2)
                prob = np.exp(-dist / 2.0)
                atlas_data[x, y, z, 2] = prob

    # Create NIfTI
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 3.0
    img = nib.Nifti1Image(atlas_data, affine)

    atlas_file = output_dir / "probabilistic_atlas.nii.gz"
    nib.save(img, atlas_file)

    print(f"   Atlas: {atlas_file}")

    return atlas_file


def create_3d_volume(
    output_dir: Path,
    image_shape: tuple = (20, 20, 20)
) -> Path:
    """
    Create synthetic 3D volume (e.g., FA map)

    Args:
        output_dir: Output directory
        image_shape: 3D spatial dimensions

    Returns:
        Path to volume file
    """
    print(f"\nCreating synthetic 3D volume (FA-like)...")

    # Create volume with gradient pattern
    data = np.zeros(image_shape)
    center_x, center_y, center_z = np.array(image_shape) // 2

    for x in range(image_shape[0]):
        for y in range(image_shape[1]):
            for z in range(image_shape[2]):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
                # Higher values in center (white matter-like)
                data[x, y, z] = 0.8 * np.exp(-dist / 5.0) + 0.1

    # Add some noise
    data += np.random.randn(*image_shape) * 0.05
    data = np.clip(data, 0, 1)

    # Create NIfTI
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 3.0
    img = nib.Nifti1Image(data, affine)

    volume_file = output_dir / "synthetic_FA.nii.gz"
    nib.save(img, volume_file)

    print(f"   Volume: {volume_file}")

    return volume_file


def test_roi_extraction_discrete(func_file: Path, atlas_file: Path, labels_file: Path, output_dir: Path):
    """Test ROI extraction with discrete atlas"""
    print("\n" + "=" * 80)
    print("TEST 1: ROI Extraction with Discrete Atlas")
    print("=" * 80)

    try:
        # Load atlas
        atlas = load_atlas(atlas_file, labels_file=labels_file, is_probabilistic=False)
        print(f" Loaded discrete atlas: {atlas.n_rois} ROIs")

        # Extract timeseries
        timeseries, roi_names = extract_roi_timeseries(
            data_file=func_file,
            atlas=atlas,
            min_voxels=5,
            statistic='mean'
        )

        print(f" Extracted timeseries shape: {timeseries.shape}")
        print(f" ROI names: {roi_names}")

        # Validate
        assert timeseries.ndim == 2, "Timeseries should be 2D"
        assert len(roi_names) == timeseries.shape[1], "Number of names should match columns"
        assert len(roi_names) <= atlas.n_rois, "Should not have more ROIs than atlas"

        print(" TEST 1 PASSED")
        return True, timeseries, roi_names

    except Exception as e:
        print(f" TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_roi_extraction_probabilistic(func_file: Path, atlas_file: Path, output_dir: Path):
    """Test ROI extraction with probabilistic atlas"""
    print("\n" + "=" * 80)
    print("TEST 2: ROI Extraction with Probabilistic Atlas")
    print("=" * 80)

    try:
        # Load atlas
        atlas = load_atlas(atlas_file, is_probabilistic=True)
        print(f" Loaded probabilistic atlas: {atlas.n_rois} ROIs")

        # Extract timeseries
        timeseries, roi_names = extract_roi_timeseries(
            data_file=func_file,
            atlas=atlas,
            min_voxels=5,
            statistic='mean'
        )

        print(f" Extracted timeseries shape: {timeseries.shape}")
        print(f" ROI names: {roi_names}")

        # Validate
        assert timeseries.ndim == 2, "Timeseries should be 2D"
        assert len(roi_names) == timeseries.shape[1], "Number of names should match columns"

        print(" TEST 2 PASSED")
        return True, timeseries, roi_names

    except Exception as e:
        print(f" TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_roi_values_extraction(volume_file: Path, atlas_file: Path, labels_file: Path, output_dir: Path):
    """Test ROI value extraction from 3D volume"""
    print("\n" + "=" * 80)
    print("TEST 3: ROI Value Extraction from 3D Volume")
    print("=" * 80)

    try:
        # Extract values
        roi_values, voxel_counts = extract_roi_values(
            data_file=volume_file,
            atlas=atlas_file,
            labels_file=labels_file,
            min_voxels=5,
            statistic='mean'
        )

        print(f" Extracted {len(roi_values)} ROI values")
        for roi_name, value in roi_values.items():
            n_voxels = voxel_counts[roi_name]
            print(f"  {roi_name}: {value:.4f} ({n_voxels} voxels)")

        # Validate
        assert len(roi_values) > 0, "Should extract at least one ROI"
        assert len(roi_values) == len(voxel_counts), "Values and counts should match"
        assert all(0 <= v <= 1 for v in roi_values.values()), "FA values should be in [0,1]"

        print(" TEST 3 PASSED")
        return True

    except Exception as e:
        print(f" TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_functional_connectivity(timeseries: np.ndarray, roi_names: list, output_dir: Path):
    """Test functional connectivity computation"""
    print("\n" + "=" * 80)
    print("TEST 4: Functional Connectivity Computation")
    print("=" * 80)

    try:
        # Compute FC with various options
        fc_dir = output_dir / "fc_results"
        fc_results = compute_functional_connectivity(
            timeseries=timeseries,
            roi_names=roi_names,
            method='pearson',
            fisher_z=True,
            threshold=None,
            output_dir=fc_dir,
            output_prefix='test_fc'
        )

        fc_matrix = fc_results['connectivity_matrix']

        print(f" Computed connectivity matrix shape: {fc_matrix.shape}")
        print(f" Method: {fc_results['method']}")
        print(f" Fisher z: {fc_results['fisher_z']}")
        print(f" Mean connectivity: {fc_results['summary']['mean_connectivity']:.4f}")
        print(f" Std connectivity: {fc_results['summary']['std_connectivity']:.4f}")

        # Validate
        n_rois = len(roi_names)
        assert fc_matrix.shape == (n_rois, n_rois), f"Matrix should be {n_rois}x{n_rois}"
        assert np.allclose(fc_matrix, fc_matrix.T), "Matrix should be symmetric"
        assert np.allclose(np.diag(fc_matrix), 0), "Diagonal should be 0 (Fisher z of 1.0)"

        # Check output files
        assert (fc_dir / "test_fc_matrix.npy").exists(), "Should create numpy file"
        assert (fc_dir / "test_fc_matrix.csv").exists(), "Should create CSV file"
        assert (fc_dir / "test_fc_summary.json").exists(), "Should create summary file"
        assert (fc_dir / "test_fc_roi_names.txt").exists(), "Should create ROI names file"

        print(" TEST 4 PASSED")
        return True, fc_matrix

    except Exception as e:
        print(f" TEST 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_partial_correlation(timeseries: np.ndarray, roi_names: list, output_dir: Path):
    """Test partial correlation computation"""
    print("\n" + "=" * 80)
    print("TEST 5: Partial Correlation")
    print("=" * 80)

    try:
        # Compute partial correlation
        fc_dir = output_dir / "partial_fc_results"
        fc_results = compute_functional_connectivity(
            timeseries=timeseries,
            roi_names=roi_names,
            partial=True,
            fisher_z=True,
            output_dir=fc_dir,
            output_prefix='partial_fc'
        )

        partial_matrix = fc_results['connectivity_matrix']

        print(f" Computed partial correlation matrix shape: {partial_matrix.shape}")
        print(f" Method: {fc_results['method']}")
        print(f" Mean connectivity: {fc_results['summary']['mean_connectivity']:.4f}")

        # Validate
        n_rois = len(roi_names)
        assert partial_matrix.shape == (n_rois, n_rois), "Matrix should be square"
        assert np.allclose(partial_matrix, partial_matrix.T), "Matrix should be symmetric"

        print(" TEST 5 PASSED")
        return True

    except Exception as e:
        print(f" TEST 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_thresholding(fc_matrix: np.ndarray, output_dir: Path):
    """Test matrix thresholding"""
    print("\n" + "=" * 80)
    print("TEST 6: Matrix Thresholding")
    print("=" * 80)

    try:
        # Test thresholding
        threshold = 0.5
        thresholded = threshold_matrix(fc_matrix, threshold=threshold, absolute=True, binarize=False)

        # Count edges
        n_edges_original = np.sum(fc_matrix != 0) / 2
        n_edges_thresholded = np.sum(thresholded != 0) / 2

        print(f" Threshold: {threshold}")
        print(f" Original edges: {n_edges_original}")
        print(f" Thresholded edges: {n_edges_thresholded}")

        # Validate
        assert n_edges_thresholded <= n_edges_original, "Thresholding should reduce edges"
        assert np.allclose(thresholded, thresholded.T), "Should remain symmetric"
        assert np.all(np.abs(thresholded[thresholded != 0]) >= threshold), "Non-zero values should exceed threshold"

        print(" TEST 6 PASSED")
        return True

    except Exception as e:
        print(f" TEST 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner"""
    print("=" * 80)
    print("CONNECTOME MODULE TEST SUITE")
    print("=" * 80)

    # Create temporary directory for test data
    test_dir = Path(tempfile.mkdtemp(prefix="connectome_test_"))
    print(f"\nTest directory: {test_dir}")

    try:
        # Generate synthetic data
        print("\n" + "=" * 80)
        print("GENERATING SYNTHETIC TEST DATA")
        print("=" * 80)

        func_file = create_synthetic_functional_data(test_dir)
        discrete_atlas, labels_file = create_discrete_atlas(test_dir)
        prob_atlas = create_probabilistic_atlas(test_dir)
        volume_file = create_3d_volume(test_dir)

        # Run tests
        results = []

        # Test 1: Discrete atlas ROI extraction
        success, timeseries, roi_names = test_roi_extraction_discrete(
            func_file, discrete_atlas, labels_file, test_dir
        )
        results.append(("ROI Extraction (Discrete)", success))

        if not success:
            print("\n Critical test failed, aborting remaining tests")
            sys.exit(1)

        # Test 2: Probabilistic atlas ROI extraction
        success, _, _ = test_roi_extraction_probabilistic(func_file, prob_atlas, test_dir)
        results.append(("ROI Extraction (Probabilistic)", success))

        # Test 3: 3D volume value extraction
        success = test_roi_values_extraction(volume_file, discrete_atlas, labels_file, test_dir)
        results.append(("ROI Value Extraction", success))

        # Test 4: Functional connectivity
        success, fc_matrix = test_functional_connectivity(timeseries, roi_names, test_dir)
        results.append(("Functional Connectivity", success))

        if not success:
            print("\n Critical test failed, aborting remaining tests")
            sys.exit(1)

        # Test 5: Partial correlation
        success = test_partial_correlation(timeseries, roi_names, test_dir)
        results.append(("Partial Correlation", success))

        # Test 6: Thresholding
        success = test_thresholding(fc_matrix, test_dir)
        results.append(("Matrix Thresholding", success))

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        all_passed = True
        for test_name, passed in results:
            status = " PASSED" if passed else " FAILED"
            print(f"{test_name:40s} {status}")
            if not passed:
                all_passed = False

        print("=" * 80)

        if all_passed:
            print("\n<*** ALL TESTS PASSED! ***")
            print(f"\nTest outputs saved to: {test_dir}")
            print("Review the outputs to validate correctness.")
            return 0
        else:
            print("\nL SOME TESTS FAILED")
            print(f"\nTest outputs saved to: {test_dir}")
            print("Review the outputs and error messages above.")
            return 1

    except Exception as e:
        print(f"\n TEST SUITE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Optionally clean up (comment out to keep test data)
        # print(f"\nCleaning up: {test_dir}")
        # shutil.rmtree(test_dir)
        pass


if __name__ == '__main__':
    sys.exit(main())
