#!/usr/bin/env python3
"""
Environment Verification Script

Checks that all required dependencies are properly installed.
Run this before using the pipeline to catch issues early.

Usage:
    uv run python verify_environment.py
"""

import sys


def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name} - {e}")
        return False


def check_version(module_name, min_version=None):
    """Check module version."""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        version_str = f" (v{version})" if version != 'unknown' else ""

        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"⚠ {module_name}{version_str} - Need >={min_version}")
                return False

        print(f"✓ {module_name}{version_str}")
        return True
    except ImportError:
        print(f"✗ {module_name} - Not installed")
        return False


def main():
    """Run all verification checks."""
    print("="*70)
    print("MRI Preprocessing Pipeline - Environment Verification")
    print("="*70)
    print()

    all_ok = True

    # Core dependencies
    print("Core Dependencies:")
    print("-" * 70)
    all_ok &= check_version('nipype', '1.10.0')
    all_ok &= check_version('nibabel', '5.3.0')
    all_ok &= check_version('numpy', '1.24')
    all_ok &= check_version('scipy', '1.14')
    all_ok &= check_version('pandas', '2.0')
    print()

    # Neuroimaging packages
    print("Neuroimaging Packages:")
    print("-" * 70)
    all_ok &= check_version('dipy', '1.11.0')
    all_ok &= check_import('amico', 'dmri-amico')
    all_ok &= check_version('tedana', '25.0.0')
    all_ok &= check_version('nilearn', '0.10.0')
    print()

    # Data handling
    print("Data Handling:")
    print("-" * 70)
    all_ok &= check_version('pydicom', '3.0.0')
    all_ok &= check_import('yaml', 'PyYAML')
    print()

    # Analysis and visualization
    print("Analysis & Visualization:")
    print("-" * 70)
    all_ok &= check_import('sklearn', 'scikit-learn')
    all_ok &= check_import('matplotlib')
    all_ok &= check_import('seaborn')
    print()

    # Optional but recommended
    print("Optional Dependencies:")
    print("-" * 70)
    check_import('heudiconv')
    print()

    # Check FSL environment
    print("External Tools:")
    print("-" * 70)
    import os
    fsl_dir = os.environ.get('FSLDIR')
    if fsl_dir:
        print(f"✓ FSL: {fsl_dir}")
    else:
        print("⚠ FSLDIR not set - FSL may not be available")
        all_ok = False
    print()

    # Summary
    print("="*70)
    if all_ok:
        print("✓ All required dependencies are installed and ready!")
        print("✓ You can now run the pipeline")
        return 0
    else:
        print("✗ Some dependencies are missing or outdated")
        print()
        print("To fix:")
        print("  1. Run: uv sync")
        print("  2. Check FSL is loaded: echo $FSLDIR")
        print("  3. Re-run this script")
        return 1


if __name__ == '__main__':
    sys.exit(main())
