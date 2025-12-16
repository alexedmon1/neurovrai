#!/usr/bin/env python3
"""Test only the post-workflow processing (ACompCor → Bandpass → Smooth)"""

from pathlib import Path
import subprocess

# Paths
subject = 'IRC805-2350101'
derivatives_dir = Path('/mnt/bytopia/IRC805/derivatives') / subject / 'func'
work_dir = Path('/mnt/bytopia/IRC805/work') / subject

# Find denoised output
denoised_file = derivatives_dir / 'denoised' / 'denoised_func_data_nonaggr.nii.gz'
print(f"Denoised file: {denoised_file}")
print(f"Exists: {denoised_file.exists()}")
print(f"Size: {denoised_file.stat().st_size / 1024 / 1024:.1f} MB" if denoised_file.exists() else "N/A")

# Test 1: Can we read it with nibabel?
print("\nTest 1: Loading with nibabel...")
try:
    import nibabel as nib
    img = nib.load(str(denoised_file))
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {img.get_data_dtype()}")
    print("  ✓ Successfully loaded")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: Can we run a simple FSL command on it?
print("\nTest 2: Running fslinfo...")
try:
    result = subprocess.run(
        ['fslinfo', str(denoised_file)],
        capture_output=True,
        text=True,
        timeout=10
    )
    print(f"  Exit code: {result.returncode}")
    if result.returncode == 0:
        print("  ✓ fslinfo succeeded")
        print(result.stdout[:200])
    else:
        print(f"  ✗ fslinfo failed: {result.stderr}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 3: Can we run 3dBandpass?
print("\nTest 3: Testing 3dBandpass...")
output_file = work_dir / 'test_bandpass.nii.gz'
output_file.parent.mkdir(parents=True, exist_ok=True)

try:
    cmd = [
        '3dBandpass',
        '-prefix', str(output_file),
        '0.001', '0.08',
        str(denoised_file)
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60
    )
    print(f"  Exit code: {result.returncode}")
    if result.returncode == 0:
        print("  ✓ 3dBandpass succeeded")
        if output_file.exists():
            print(f"  Output size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"  ✗ 3dBandpass failed")
        print(f"  STDERR: {result.stderr[:500]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "="*70)
print("Diagnostic test complete")
