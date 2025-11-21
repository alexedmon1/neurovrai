#!/usr/bin/env python3
"""
TBSS Data Preparation Workflow

Prepares DTI data for group-level statistical analysis using FSL's TBSS pipeline.

This workflow:
1. Collects DTI metric maps (FA, MD, AD, RD) from preprocessed subjects
2. Validates data availability and logs missing subjects
3. Runs FSL TBSS pipeline (tbss_1_preproc through tbss_4_prestats)
4. Generates subject manifest and analysis-ready 4D volumes
5. Outputs skeleton-projected data ready for randomise

Separation of Concerns:
- This module ONLY prepares data (run once)
- Statistical analysis (design matrix, randomise) is handled separately
- Allows iterating on statistical models without re-running TBSS

Usage:
    python -m neurovrai.analysis.tbss.prepare_tbss \\
        --config config.yaml \\
        --metric FA \\
        --output-dir /study/analysis/tbss_FA/
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from neurovrai.config import load_config


class SubjectData:
    """Container for subject data and validation status"""

    def __init__(self, subject_id: str, metric_file: Optional[Path] = None,
                 included: bool = False, exclusion_reason: Optional[str] = None):
        self.subject_id = subject_id
        self.metric_file = metric_file
        self.included = included
        self.exclusion_reason = exclusion_reason


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"prepare_tbss_{timestamp}.log"

    logger = logging.getLogger("prepare_tbss")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def discover_subjects(derivatives_dir: Path, logger: logging.Logger) -> List[str]:
    """
    Discover all subjects in derivatives directory

    Args:
        derivatives_dir: Path to derivatives directory
        logger: Logger instance

    Returns:
        List of subject IDs
    """
    subjects = []

    if not derivatives_dir.exists():
        logger.error(f"Derivatives directory not found: {derivatives_dir}")
        return subjects

    # Look for subject directories
    for subject_dir in sorted(derivatives_dir.iterdir()):
        if subject_dir.is_dir() and not subject_dir.name.startswith('.'):
            subjects.append(subject_dir.name)

    logger.info(f"Discovered {len(subjects)} subjects in {derivatives_dir}")
    return subjects


def collect_subject_data(
    subject_id: str,
    derivatives_dir: Path,
    metric: str,
    logger: logging.Logger
) -> SubjectData:
    """
    Collect and validate metric file for a single subject

    Args:
        subject_id: Subject identifier
        derivatives_dir: Path to derivatives directory
        metric: DTI metric (FA, MD, AD, RD)
        logger: Logger instance

    Returns:
        SubjectData object with validation status
    """
    # Look for metric file in subject's DWI directory
    dwi_dir = derivatives_dir / subject_id / "dwi"

    # Try different possible filenames
    possible_files = [
        dwi_dir / "dti" / f"{metric}.nii.gz",
        dwi_dir / "dti" / f"dti_{metric}.nii.gz",
        dwi_dir / "dti" / f"dtifit__{metric}.nii.gz",  # neurovrai preprocessing output
        dwi_dir / f"{metric}.nii.gz",
        dwi_dir / f"dti_{metric}.nii.gz",
        dwi_dir / f"dtifit__{metric}.nii.gz",
    ]

    for metric_file in possible_files:
        if metric_file.exists():
            logger.info(f"✓ {subject_id}: Found {metric} at {metric_file.relative_to(derivatives_dir)}")
            return SubjectData(
                subject_id=subject_id,
                metric_file=metric_file,
                included=True
            )

    # File not found
    logger.warning(f"✗ {subject_id}: {metric} file not found in {dwi_dir.relative_to(derivatives_dir)}")
    return SubjectData(
        subject_id=subject_id,
        included=False,
        exclusion_reason=f"{metric} file not found"
    )


def copy_to_tbss_structure(
    subjects_data: List[SubjectData],
    output_dir: Path,
    metric: str,
    logger: logging.Logger
) -> Path:
    """
    Copy metric files to TBSS work directory

    tbss_1_preproc expects files in the work directory, then it:
    - Moves originals to origdata/
    - Creates processed files in FA/

    Files should be named: <subject_id>.nii.gz (NO metric suffix - tbss adds it)

    Args:
        subjects_data: List of validated subject data
        output_dir: Output directory for TBSS analysis (work directory)
        metric: DTI metric name
        logger: Logger instance

    Returns:
        Path to work directory containing input files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying {metric} files to TBSS work directory...")

    included_subjects = [s for s in subjects_data if s.included]

    for subject_data in included_subjects:
        # tbss_1_preproc adds _FA suffix automatically, so don't include metric in filename
        dest_file = output_dir / f"{subject_data.subject_id}.nii.gz"
        shutil.copy2(subject_data.metric_file, dest_file)
        logger.info(f"  Copied: {subject_data.subject_id}")

    logger.info(f"Copied {len(included_subjects)} files to {output_dir}")
    return output_dir


def run_tbss_pipeline(
    tbss_input_dir: Path,
    output_dir: Path,
    metric: str,
    logger: logging.Logger
) -> bool:
    """
    Execute FSL TBSS pipeline (steps 1-4) using direct FSL commands

    Args:
        tbss_input_dir: Directory containing input files
        output_dir: Output directory
        metric: DTI metric
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("=" * 80)
        logger.info("Running FSL TBSS Pipeline")
        logger.info("=" * 80)

        # Change to output directory (TBSS scripts expect to run in parent of origdata)
        original_dir = os.getcwd()
        os.chdir(output_dir)

        # Set environment variable to run locally without cluster submission
        env = os.environ.copy()
        env['FSLSUBALREADYRUN'] = 'true'

        # Step 1: Preprocessing (erodes and zero-ends images)
        # tbss_1_preproc processes files in work dir, moves them to origdata/, creates FA/
        logger.info("\n[TBSS Step 1] Running tbss_1_preproc...")
        cmd = 'tbss_1_preproc *.nii.gz'
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode != 0:
            logger.error(f"tbss_1_preproc failed: {result.stderr}")
            os.chdir(original_dir)
            return False
        logger.info(f"  ✓ Completed: Processed images in {output_dir}")

        # Verify FA directory was created by tbss_1_preproc
        fa_dir = output_dir / "FA"
        if not fa_dir.exists():
            logger.error(f"FA directory not created by tbss_1_preproc: {fa_dir}")
            os.chdir(original_dir)
            return False

        # Step 2: Registration to target (run from work directory, script CDs into FA/)
        logger.info("\n[TBSS Step 2] Running tbss_2_reg...")
        cmd = ['tbss_2_reg', '-T']  # -T = use FMRIB58_FA target
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode != 0:
            logger.error(f"tbss_2_reg failed: {result.stderr}")
            os.chdir(original_dir)
            return False
        logger.info(f"  ✓ Completed: Registered all subjects to FMRIB58_FA template")

        # Step 3: Post-registration (run from work directory, script CDs into FA/)
        logger.info("\n[TBSS Step 3] Running tbss_3_postreg...")
        cmd = ['tbss_3_postreg', '-S']  # -S = use FMRIB58_FA mean and skeleton
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode != 0:
            logger.error(f"tbss_3_postreg failed: {result.stderr}")
            os.chdir(original_dir)
            return False
        logger.info(f"  ✓ Completed: Created mean FA and skeleton (threshold=0.2)")

        # Step 4: Project to skeleton (run from work directory, script CDs into FA/)
        logger.info(f"\n[TBSS Step 4] Running tbss_4_prestats for {metric}...")
        cmd = ['tbss_4_prestats', '0.2']  # Threshold for skeleton projection
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode != 0:
            logger.error(f"tbss_4_prestats failed: {result.stderr}")
            os.chdir(original_dir)
            return False
        logger.info(f"  ✓ Completed: Projected {metric} onto skeleton")

        # Return to original directory
        os.chdir(original_dir)

        logger.info("\n" + "=" * 80)
        logger.info("FSL TBSS Pipeline Completed Successfully")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"TBSS pipeline failed: {str(e)}", exc_info=True)
        # Ensure we return to original directory
        try:
            os.chdir(original_dir)
        except:
            pass
        return False


def generate_manifest(
    subjects_data: List[SubjectData],
    output_dir: Path,
    metric: str,
    logger: logging.Logger
) -> Dict:
    """
    Generate manifest file documenting included/excluded subjects

    Args:
        subjects_data: List of subject data with validation status
        output_dir: Output directory
        metric: DTI metric
        logger: Logger instance

    Returns:
        Manifest dictionary
    """
    included = [s for s in subjects_data if s.included]
    excluded = [s for s in subjects_data if not s.included]

    manifest = {
        "analysis_type": "TBSS",
        "metric": metric,
        "date_prepared": datetime.now().isoformat(),
        "total_subjects_discovered": len(subjects_data),
        "subjects_included": len(included),
        "subjects_excluded": len(excluded),
        "included_subject_ids": [s.subject_id for s in included],
        "excluded_subjects": [
            {
                "subject_id": s.subject_id,
                "reason": s.exclusion_reason
            }
            for s in excluded
        ]
    }

    # Write manifest
    manifest_file = output_dir / "subject_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nSubject Manifest:")
    logger.info(f"  Total discovered: {manifest['total_subjects_discovered']}")
    logger.info(f"  Included: {manifest['subjects_included']}")
    logger.info(f"  Excluded: {manifest['subjects_excluded']}")

    if excluded:
        logger.info(f"\n  Excluded subjects:")
        for exc in manifest['excluded_subjects']:
            logger.info(f"    - {exc['subject_id']}: {exc['reason']}")

    logger.info(f"\n  Manifest saved to: {manifest_file}")

    # Also write simple subject list for easy reference
    subject_list_file = output_dir / "subject_list.txt"
    with open(subject_list_file, 'w') as f:
        for s in included:
            f.write(f"{s.subject_id}\n")
    logger.info(f"  Subject list saved to: {subject_list_file}")

    return manifest


def prepare_tbss_analysis(
    config: Dict,
    metric: str,
    output_dir: Path,
    subjects: Optional[List[str]] = None
) -> Dict:
    """
    Main workflow: Prepare TBSS analysis from preprocessed DTI data

    Args:
        config: Configuration dictionary
        metric: DTI metric to analyze (FA, MD, AD, RD)
        output_dir: Output directory for prepared analysis
        subjects: Optional list of specific subjects to include (if None, discovers all)

    Returns:
        Dictionary containing preparation results and manifest
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("TBSS Data Preparation Workflow")
    logger.info("=" * 80)
    logger.info(f"Metric: {metric}")
    logger.info(f"Output directory: {output_dir}")

    # Get derivatives directory
    derivatives_dir = Path(config['derivatives_dir'])
    logger.info(f"Derivatives directory: {derivatives_dir}")

    # Discover or use provided subjects
    if subjects is None:
        subjects = discover_subjects(derivatives_dir, logger)
    else:
        logger.info(f"Using provided subject list: {len(subjects)} subjects")

    if not subjects:
        logger.error("No subjects found or provided!")
        return {"success": False, "error": "No subjects found"}

    # Collect and validate data for each subject
    logger.info(f"\nValidating {metric} data for {len(subjects)} subjects...")
    logger.info("-" * 80)

    subjects_data = []
    for subject_id in subjects:
        subject_data = collect_subject_data(
            subject_id=subject_id,
            derivatives_dir=derivatives_dir,
            metric=metric,
            logger=logger
        )
        subjects_data.append(subject_data)

    # Check if we have any valid subjects
    included_count = sum(1 for s in subjects_data if s.included)
    if included_count == 0:
        logger.error(f"No subjects have valid {metric} data!")
        return {"success": False, "error": "No valid subject data found"}

    logger.info("-" * 80)
    logger.info(f"Validation complete: {included_count}/{len(subjects)} subjects have valid data")

    # Copy files to TBSS structure
    logger.info("\n" + "=" * 80)
    tbss_input_dir = copy_to_tbss_structure(
        subjects_data=subjects_data,
        output_dir=output_dir,
        metric=metric,
        logger=logger
    )

    # Run TBSS pipeline
    success = run_tbss_pipeline(
        tbss_input_dir=tbss_input_dir,
        output_dir=output_dir,
        metric=metric,
        logger=logger
    )

    if not success:
        return {"success": False, "error": "TBSS pipeline failed"}

    # Generate manifest
    logger.info("\n" + "=" * 80)
    manifest = generate_manifest(
        subjects_data=subjects_data,
        output_dir=output_dir,
        metric=metric,
        logger=logger
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TBSS PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Subjects included: {manifest['subjects_included']}")
    logger.info(f"Subjects excluded: {manifest['subjects_excluded']}")
    logger.info("\nNext steps:")
    logger.info("  1. Review subject_manifest.json for excluded subjects")
    logger.info("  2. Filter your participants.csv to match subject_list.txt")
    logger.info("  3. Run statistical analysis with run_tbss_stats.py")
    logger.info("=" * 80)

    return {
        "success": True,
        "manifest": manifest,
        "output_dir": str(output_dir)
    }


def main():
    """Command-line interface for TBSS preparation"""
    parser = argparse.ArgumentParser(
        description="Prepare DTI data for TBSS group analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare FA analysis for all subjects
  python -m neurovrai.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --metric FA \\
      --output-dir /study/analysis/tbss_FA/

  # Prepare MD analysis for specific subjects
  python -m neurovrai.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --metric MD \\
      --subjects sub-001 sub-002 sub-003 \\
      --output-dir /study/analysis/tbss_MD/
        """
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to config.yaml file'
    )

    parser.add_argument(
        '--metric',
        type=str,
        required=True,
        choices=['FA', 'MD', 'AD', 'RD'],
        help='DTI metric to analyze'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for prepared TBSS analysis'
    )

    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Specific subjects to include (default: discover all)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run preparation
    result = prepare_tbss_analysis(
        config=config,
        metric=args.metric,
        output_dir=args.output_dir,
        subjects=args.subjects
    )

    # Exit with appropriate code
    exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
