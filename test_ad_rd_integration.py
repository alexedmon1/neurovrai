#!/usr/bin/env python3
"""
Test AD/RD calculation integration in DWI preprocessing workflow.

This script verifies that the calculate_ad_rd function works correctly
with actual DTI data.
"""

from pathlib import Path
from neurovrai.preprocess.utils.dti_metrics import calculate_ad_rd, validate_dti_metrics

def test_ad_rd_calculation():
    """Test AD/RD calculation on existing subject data."""

    # Test on one subject that already has DTI processed
    test_subject = 'IRC805-1580101'
    dti_dir = Path(f'/mnt/bytopia/IRC805/derivatives/{test_subject}/dwi/dti')

    print("="*80)
    print("Testing AD/RD Calculation Integration")
    print("="*80)
    print(f"Test subject: {test_subject}")
    print(f"DTI directory: {dti_dir}")
    print()

    if not dti_dir.exists():
        print(f"❌ ERROR: DTI directory not found: {dti_dir}")
        return False

    # Check for eigenvalue files
    print("Step 1: Checking for eigenvalue files...")
    l1_file = dti_dir / "dtifit__L1.nii.gz"
    l2_file = dti_dir / "dtifit__L2.nii.gz"
    l3_file = dti_dir / "dtifit__L3.nii.gz"

    for f in [l1_file, l2_file, l3_file]:
        if f.exists():
            print(f"  ✓ Found: {f.name}")
        else:
            print(f"  ✗ Missing: {f.name}")
            return False

    print()

    # Calculate AD and RD
    print("Step 2: Calculating AD and RD...")
    ad_file, rd_file = calculate_ad_rd(dti_dir, prefix='dtifit__')

    if not ad_file or not rd_file:
        print("  ✗ Failed to calculate AD/RD")
        return False

    print(f"  ✓ Created: {ad_file.name}")
    print(f"  ✓ Created: {rd_file.name}")
    print()

    # Validate all DTI metrics
    print("Step 3: Validating all DTI metrics...")
    results = validate_dti_metrics(dti_dir, required_metrics=['FA', 'MD', 'AD', 'RD'])

    all_valid = True
    for metric, info in results.items():
        if info['exists']:
            print(f"  ✓ {metric}: exists, shape={info.get('shape')}, mean={info.get('mean', 0):.6f}")
        else:
            print(f"  ✗ {metric}: missing")
            all_valid = False

    print()

    if all_valid:
        print("="*80)
        print("✓ AD/RD Integration Test PASSED")
        print("="*80)
        return True
    else:
        print("="*80)
        print("✗ AD/RD Integration Test FAILED")
        print("="*80)
        return False

if __name__ == '__main__':
    import sys
    success = test_ad_rd_calculation()
    sys.exit(0 if success else 1)
