#!/usr/bin/env python3
"""
Complete Anatomical QC Test Suite.

Tests all anatomical QC modules on real or synthetic data:
- Skull Strip QC
- Segmentation QC
- Registration QC
"""

import sys
from pathlib import Path
import logging
import argparse
import json

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from mri_preprocess.qc.anat.skull_strip_qc import SkullStripQualityControl
from mri_preprocess.qc.anat.segmentation_qc import SegmentationQualityControl
from mri_preprocess.qc.anat.registration_qc import RegistrationQualityControl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_anatomical_data(study_root: Path, subject: str):
    """
    Find anatomical preprocessing outputs for a subject.

    Parameters
    ----------
    study_root : Path
        Study root directory
    subject : str
        Subject identifier

    Returns
    -------
    dict
        Paths to anatomical files
    """
    logger.info(f"Searching for anatomical data for {subject}...")

    # Check standard derivatives locations
    search_dirs = [
        study_root / 'derivatives' / 'anat_preproc' / subject,
        study_root / 'derivatives' / 'anatomical' / subject,
        study_root / subject / 'anat',
        study_root / 'subjects' / subject / 'anat'
    ]

    files = {
        't1w': None,
        'brain': None,
        'mask': None,
        'csf': None,
        'gm': None,
        'wm': None,
        'registered': None,
        'registered_mask': None
    }

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        logger.info(f"Checking: {search_dir}")

        # Look for T1w
        if files['t1w'] is None:
            for pattern in ['*T1w.nii.gz', '*t1w.nii.gz', '*T1*.nii.gz']:
                candidates = list(search_dir.glob(pattern))
                candidates = [c for c in candidates if 'brain' not in c.name.lower() and 'mask' not in c.name.lower()]
                if candidates:
                    files['t1w'] = candidates[0]
                    logger.info(f"  Found T1w: {files['t1w'].name}")
                    break

        # Look for brain-extracted
        if files['brain'] is None:
            for pattern in ['*brain.nii.gz', '*bet.nii.gz']:
                candidates = list(search_dir.glob(pattern))
                candidates = [c for c in candidates if 'mask' not in c.name.lower() and 'pve' not in c.name.lower()]
                if candidates:
                    files['brain'] = candidates[0]
                    logger.info(f"  Found brain: {files['brain'].name}")
                    break

        # Look for brain mask
        if files['mask'] is None:
            for pattern in ['*brain_mask.nii.gz', '*mask.nii.gz']:
                candidates = list(search_dir.glob(pattern))
                candidates = [c for c in candidates if 'MNI' not in c.name and 'mni' not in c.name]
                if candidates:
                    files['mask'] = candidates[0]
                    logger.info(f"  Found mask: {files['mask'].name}")
                    break

        # Look for tissue segmentations (FAST outputs)
        if files['csf'] is None:
            for pattern in ['*pve_0.nii.gz', '*csf*.nii.gz']:
                candidates = list(search_dir.glob(pattern))
                if candidates:
                    files['csf'] = candidates[0]
                    logger.info(f"  Found CSF: {files['csf'].name}")
                    break

        if files['gm'] is None:
            for pattern in ['*pve_1.nii.gz', '*gm*.nii.gz', '*GM*.nii.gz']:
                candidates = list(search_dir.glob(pattern))
                if candidates:
                    files['gm'] = candidates[0]
                    logger.info(f"  Found GM: {files['gm'].name}")
                    break

        if files['wm'] is None:
            for pattern in ['*pve_2.nii.gz', '*wm*.nii.gz', '*WM*.nii.gz']:
                candidates = list(search_dir.glob(pattern))
                if candidates:
                    files['wm'] = candidates[0]
                    logger.info(f"  Found WM: {files['wm'].name}")
                    break

        # Look for registered outputs
        if files['registered'] is None:
            for pattern in ['*MNI152*.nii.gz', '*mni152*.nii.gz', '*2mm.nii.gz', '*std*.nii.gz']:
                candidates = list(search_dir.glob(pattern))
                candidates = [c for c in candidates if 'mask' not in c.name.lower()]
                if candidates:
                    files['registered'] = candidates[0]
                    logger.info(f"  Found registered: {files['registered'].name}")
                    break

        if files['registered_mask'] is None:
            for pattern in ['*MNI152*mask*.nii.gz', '*mni152*mask*.nii.gz', '*std*mask*.nii.gz']:
                candidates = list(search_dir.glob(pattern))
                if candidates:
                    files['registered_mask'] = candidates[0]
                    logger.info(f"  Found registered mask: {files['registered_mask'].name}")
                    break

    return files


def run_complete_qc(subject: str, anat_dir: Path, qc_dir: Path, files: dict):
    """
    Run all anatomical QC modules.

    Parameters
    ----------
    subject : str
        Subject identifier
    anat_dir : Path
        Directory containing anatomical outputs
    qc_dir : Path
        QC output directory
    files : dict
        Paths to anatomical files

    Returns
    -------
    dict
        Combined QC results
    """
    all_results = {
        'subject': subject,
        'skull_strip': {},
        'segmentation': {},
        'registration': {}
    }

    logger.info("")
    logger.info("="*70)
    logger.info("RUNNING COMPLETE ANATOMICAL QC SUITE")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"Anat directory: {anat_dir}")
    logger.info(f"QC output: {qc_dir}")
    logger.info("")

    # 1. Skull Strip QC
    if files['mask'] is not None:
        logger.info("="*70)
        logger.info("1. SKULL STRIP QC")
        logger.info("="*70)

        skull_qc = SkullStripQualityControl(
            subject=subject,
            anat_dir=anat_dir,
            qc_dir=qc_dir / 'skull_strip'
        )

        try:
            skull_results = skull_qc.run_qc(
                t1w_file=files['t1w'],
                brain_file=files['brain'],
                mask_file=files['mask']
            )
            all_results['skull_strip'] = skull_results

            logger.info("")
            logger.info("Skull Strip QC Summary:")
            if 'mask_stats' in skull_results and skull_results['mask_stats']:
                stats = skull_results['mask_stats']
                logger.info(f"  Brain volume: {stats.get('brain_volume_cm3', 0):.2f} cm³")
                logger.info(f"  N voxels: {stats.get('n_voxels', 0)}")

            if 'quality' in skull_results and skull_results['quality']:
                quality = skull_results['quality']
                logger.info(f"  Quality pass: {quality.get('quality_pass', False)}")
                if quality.get('quality_flags'):
                    logger.info(f"  Flags: {quality['quality_flags']}")

            logger.info("  ✓ Skull Strip QC completed")
        except Exception as e:
            logger.error(f"Skull Strip QC failed: {e}")
            all_results['skull_strip']['error'] = str(e)
    else:
        logger.warning("Skipping Skull Strip QC - no brain mask found")

    logger.info("")

    # 2. Segmentation QC
    if files['csf'] is not None or files['gm'] is not None or files['wm'] is not None:
        logger.info("="*70)
        logger.info("2. SEGMENTATION QC")
        logger.info("="*70)

        seg_qc = SegmentationQualityControl(
            subject=subject,
            anat_dir=anat_dir,
            qc_dir=qc_dir / 'segmentation'
        )

        try:
            seg_results = seg_qc.run_qc(
                csf_file=files['csf'],
                gm_file=files['gm'],
                wm_file=files['wm'],
                threshold=0.5
            )
            all_results['segmentation'] = seg_results

            logger.info("")
            logger.info("Segmentation QC Summary:")
            if 'volumes' in seg_results and seg_results['volumes']:
                volumes = seg_results['volumes']
                if 'csf' in volumes:
                    logger.info(f"  CSF: {volumes['csf']['volume_cm3']:.2f} cm³ ({volumes['csf']['fraction']*100:.1f}%)")
                if 'gm' in volumes:
                    logger.info(f"  GM:  {volumes['gm']['volume_cm3']:.2f} cm³ ({volumes['gm']['fraction']*100:.1f}%)")
                if 'wm' in volumes:
                    logger.info(f"  WM:  {volumes['wm']['volume_cm3']:.2f} cm³ ({volumes['wm']['fraction']*100:.1f}%)")

            if 'validation' in seg_results and seg_results['validation']:
                validation = seg_results['validation']
                logger.info(f"  GM/WM ratio: {validation.get('gm_wm_ratio', 0):.2f}")
                logger.info(f"  Quality pass: {validation.get('quality_pass', False)}")
                if validation.get('quality_flags'):
                    logger.info(f"  Flags: {validation['quality_flags']}")

            logger.info("  ✓ Segmentation QC completed")
        except Exception as e:
            logger.error(f"Segmentation QC failed: {e}")
            all_results['segmentation']['error'] = str(e)
    else:
        logger.warning("Skipping Segmentation QC - no tissue maps found")

    logger.info("")

    # 3. Registration QC
    if files['registered'] is not None:
        logger.info("="*70)
        logger.info("3. REGISTRATION QC")
        logger.info("="*70)

        reg_qc = RegistrationQualityControl(
            subject=subject,
            anat_dir=anat_dir,
            qc_dir=qc_dir / 'registration'
        )

        try:
            reg_results = reg_qc.run_qc(
                registered_file=files['registered'],
                template_file=None,  # Use FSL MNI152 template
                registered_mask=files['registered_mask'],
                template_mask=None  # Use FSL MNI152 mask
            )
            all_results['registration'] = reg_results

            logger.info("")
            logger.info("Registration QC Summary:")
            if 'metrics' in reg_results and reg_results['metrics']:
                metrics = reg_results['metrics']
                if 'correlation' in metrics:
                    logger.info(f"  Correlation: {metrics['correlation']:.4f}")
                if 'dice_coefficient' in metrics:
                    logger.info(f"  Dice coefficient: {metrics['dice_coefficient']:.4f}")
                if 'mad' in metrics:
                    logger.info(f"  MAD: {metrics['mad']:.4f}")
                logger.info(f"  Quality pass: {metrics.get('quality_pass', False)}")
                if metrics.get('quality_flags'):
                    logger.info(f"  Flags: {metrics['quality_flags']}")

            logger.info("  ✓ Registration QC completed")
        except Exception as e:
            logger.error(f"Registration QC failed: {e}")
            all_results['registration']['error'] = str(e)
    else:
        logger.warning("Skipping Registration QC - no registered image found")

    return all_results


def print_summary(results: dict, qc_dir: Path):
    """Print final summary of QC results."""
    logger.info("")
    logger.info("="*70)
    logger.info("ANATOMICAL QC COMPLETE")
    logger.info("="*70)
    logger.info(f"Subject: {results['subject']}")
    logger.info("")

    # Overall status
    modules_run = 0
    modules_passed = 0

    if 'skull_strip' in results and results['skull_strip'] and 'error' not in results['skull_strip']:
        modules_run += 1
        if results['skull_strip'].get('quality', {}).get('quality_pass', False):
            modules_passed += 1
        status = "✓ PASS" if results['skull_strip'].get('quality', {}).get('quality_pass', False) else "✗ FAIL"
        logger.info(f"Skull Strip QC:    {status}")

    if 'segmentation' in results and results['segmentation'] and 'error' not in results['segmentation']:
        modules_run += 1
        if results['segmentation'].get('validation', {}).get('quality_pass', False):
            modules_passed += 1
        status = "✓ PASS" if results['segmentation'].get('validation', {}).get('quality_pass', False) else "✗ FAIL"
        logger.info(f"Segmentation QC:   {status}")

    if 'registration' in results and results['registration'] and 'error' not in results['registration']:
        modules_run += 1
        if results['registration'].get('metrics', {}).get('quality_pass', False):
            modules_passed += 1
        status = "✓ PASS" if results['registration'].get('metrics', {}).get('quality_pass', False) else "✗ FAIL"
        logger.info(f"Registration QC:   {status}")

    logger.info("")
    logger.info(f"Modules run: {modules_run}")
    logger.info(f"Modules passed: {modules_passed}")
    logger.info("")
    logger.info(f"QC outputs: {qc_dir}")
    logger.info("")

    # List output files
    logger.info("Generated outputs:")
    for module in ['skull_strip', 'segmentation', 'registration']:
        if module in results and 'outputs' in results[module]:
            for key, path in results[module]['outputs'].items():
                if path and Path(path).exists():
                    logger.info(f"  {Path(path).name}")


def main():
    """Run complete anatomical QC test."""
    parser = argparse.ArgumentParser(description='Run complete anatomical QC suite')
    parser.add_argument('--subject', type=str, required=True, help='Subject identifier')
    parser.add_argument('--study-root', type=Path, required=True, help='Study root directory')
    parser.add_argument('--anat-dir', type=Path, help='Anatomical preprocessing directory (optional, will auto-detect)')
    parser.add_argument('--qc-dir', type=Path, help='QC output directory (default: {study-root}/qc/anat/{subject})')

    args = parser.parse_args()

    subject = args.subject
    study_root = Path(args.study_root)

    if not study_root.exists():
        logger.error(f"Study root does not exist: {study_root}")
        return 1

    # Set QC directory
    if args.qc_dir:
        qc_dir = Path(args.qc_dir)
    else:
        qc_dir = study_root / 'qc' / 'anat' / subject

    # Find or use anatomical directory
    if args.anat_dir:
        anat_dir = Path(args.anat_dir)
        if not anat_dir.exists():
            logger.error(f"Anatomical directory does not exist: {anat_dir}")
            return 1
    else:
        anat_dir = None

    # Find anatomical files
    files = find_anatomical_data(study_root, subject)

    # Determine anatomical directory from found files
    if anat_dir is None:
        for file_path in files.values():
            if file_path is not None:
                anat_dir = file_path.parent
                break

    if anat_dir is None:
        logger.error(f"No anatomical data found for subject {subject}")
        logger.info("Searched locations:")
        logger.info(f"  - {study_root}/derivatives/anat_preproc/{subject}")
        logger.info(f"  - {study_root}/derivatives/anatomical/{subject}")
        logger.info(f"  - {study_root}/{subject}/anat")
        logger.info(f"  - {study_root}/subjects/{subject}/anat")
        return 1

    # Check if any files were found
    if all(v is None for v in files.values()):
        logger.error(f"No anatomical preprocessing outputs found in {anat_dir}")
        logger.info("Expected files:")
        logger.info("  - T1w image (original or bias-corrected)")
        logger.info("  - Brain-extracted image (*brain.nii.gz)")
        logger.info("  - Brain mask (*brain_mask.nii.gz)")
        logger.info("  - Tissue segmentations (*pve_0/1/2.nii.gz)")
        logger.info("  - Registered to MNI152 (*MNI152*.nii.gz)")
        return 1

    # Run QC
    try:
        results = run_complete_qc(subject, anat_dir, qc_dir, files)

        # Save combined results
        combined_json = qc_dir / 'combined_qc_results.json'
        combined_json.parent.mkdir(parents=True, exist_ok=True)

        with open(combined_json, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            results_serializable = json.loads(json.dumps(results, default=str))
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Saved combined results: {combined_json}")

        # Print summary
        print_summary(results, qc_dir)

        return 0

    except Exception as e:
        logger.error(f"QC failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
