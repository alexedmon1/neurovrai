#!/usr/bin/env python3
"""
TBSS Data Preparation Workflow

Prepares DTI, DKI, and NODDI data for group-level statistical analysis using FSL's TBSS pipeline.

This workflow:
1. For FA: Runs full TBSS pipeline (tbss_1_preproc through tbss_4_prestats) to create skeleton
2. For non-FA metrics: Uses tbss_non_FA to project onto existing FA skeleton
3. Supports DTI (FA, MD, AD, RD), DKI (MK, AK, RK, KFA), and NODDI (FICVF, ODI, FISO) metrics
4. Validates data availability and logs missing subjects
5. Generates subject manifest and analysis-ready 4D volumes
6. Outputs skeleton-projected data ready for statistical analysis

Workflow Order:
1. MUST run with --metric FA first to create the FA skeleton
2. Then run with other metrics (MD, AD, RD, MK, etc.) to project onto FA skeleton

Separation of Concerns:
- This module ONLY prepares data (run once per metric)
- Statistical analysis (design matrix, randomise) is handled separately
- Allows iterating on statistical models without re-running TBSS

Usage:
    # Step 1: Create FA skeleton (REQUIRED FIRST)
    python -m neurovrai.analysis.tbss.prepare_tbss \\
        --config config.yaml \\
        --metric FA \\
        --output-dir /study/analysis/tbss/

    # Step 2: Project other metrics onto FA skeleton
    python -m neurovrai.analysis.tbss.prepare_tbss \\
        --config config.yaml \\
        --metric MD \\
        --fa-skeleton-dir /study/analysis/tbss/ \\
        --output-dir /study/analysis/tbss/
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


# Supported metrics organized by modality
METRIC_GROUPS = {
    'DTI': ['FA', 'MD', 'AD', 'RD'],
    'DKI': ['MK', 'AK', 'RK', 'KFA'],
    'NODDI': ['FICVF', 'ODI', 'FISO']
}

ALL_METRICS = METRIC_GROUPS['DTI'] + METRIC_GROUPS['DKI'] + METRIC_GROUPS['NODDI']

# Mapping from short metric names to actual file names (for DIPY output)
DKI_FILE_MAPPING = {
    'MK': 'mean_kurtosis',
    'AK': 'axial_kurtosis',
    'RK': 'radial_kurtosis',
    'KFA': 'kurtosis_fa'
}


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
        metric: DTI, DKI, or NODDI metric
        logger: Logger instance

    Returns:
        SubjectData object with validation status
    """
    # Look for metric file in subject's DWI directory
    dwi_dir = derivatives_dir / subject_id / "dwi"

    # Determine which subdirectory to search based on metric type
    if metric in METRIC_GROUPS['DTI']:
        # DTI metrics: FA, MD, AD, RD
        possible_files = [
            dwi_dir / "dti" / f"{metric}.nii.gz",
            dwi_dir / "dti" / f"dti_{metric}.nii.gz",
            dwi_dir / "dti" / f"dtifit__{metric}.nii.gz",  # neurovrai preprocessing output
            dwi_dir / f"{metric}.nii.gz",
            dwi_dir / f"dti_{metric}.nii.gz",
            dwi_dir / f"dtifit__{metric}.nii.gz",
        ]
    elif metric in METRIC_GROUPS['DKI']:
        # DKI metrics: MK, AK, RK, KFA
        # Map short names to DIPY output file names
        file_name = DKI_FILE_MAPPING.get(metric, metric.lower())
        possible_files = [
            dwi_dir / "dki" / f"{file_name}.nii.gz",  # DIPY naming: mean_kurtosis.nii.gz
            dwi_dir / "dki" / f"{metric}.nii.gz",      # Short form: MK.nii.gz
            dwi_dir / "dki" / f"dki_{metric.lower()}.nii.gz",  # Legacy: dki_mk.nii.gz
            dwi_dir / f"{file_name}.nii.gz",
        ]
    elif metric in METRIC_GROUPS['NODDI']:
        # NODDI metrics: FICVF, ODI, FISO
        metric_lower = metric.lower()
        possible_files = [
            dwi_dir / "noddi" / f"{metric_lower}.nii.gz",
            dwi_dir / "noddi" / f"noddi_{metric_lower}.nii.gz",
            dwi_dir / "noddi_amico" / f"{metric_lower}.nii.gz",  # AMICO output
            dwi_dir / f"{metric_lower}.nii.gz",
        ]
    else:
        logger.error(f"Unknown metric: {metric}")
        return SubjectData(
            subject_id=subject_id,
            included=False,
            exclusion_reason=f"Unknown metric type: {metric}"
        )

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


def validate_fa_skeleton(fa_skeleton_dir: Path, logger: logging.Logger) -> Dict:
    """
    Validate that FA skeleton has been created and is ready for non-FA projection

    Args:
        fa_skeleton_dir: Directory containing FA TBSS preparation
        logger: Logger instance

    Returns:
        Dictionary with paths to required FA skeleton files

    Raises:
        FileNotFoundError: If FA skeleton is not found or incomplete
    """
    fa_skeleton_dir = Path(fa_skeleton_dir)

    logger.info("=" * 80)
    logger.info("Validating FA Skeleton")
    logger.info("=" * 80)
    logger.info(f"FA skeleton directory: {fa_skeleton_dir}")

    # Check for required FA skeleton files
    # Note: TBSS creates skeleton files in stats/ directory after tbss_4_prestats
    required_files = {
        'fa_dir': fa_skeleton_dir / 'FA',
        'stats_dir': fa_skeleton_dir / 'stats',
        'mean_fa': fa_skeleton_dir / 'stats' / 'mean_FA.nii.gz',
        'mean_fa_skeleton': fa_skeleton_dir / 'stats' / 'mean_FA_skeleton.nii.gz',
        'all_fa_skeletonised': fa_skeleton_dir / 'stats' / 'all_FA_skeletonised.nii.gz',
        'subject_list': fa_skeleton_dir / 'subject_list.txt'
    }

    # Validate each required file
    missing_files = []
    for name, path in required_files.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")

    if missing_files:
        error_msg = (
            f"FA skeleton not found or incomplete in {fa_skeleton_dir}\n"
            f"Missing files:\n  " + "\n  ".join(missing_files) + "\n\n"
            f"You MUST run FA preparation first:\n"
            f"  python -m neurovrai.analysis.tbss.prepare_tbss \\\n"
            f"      --config config.yaml \\\n"
            f"      --metric FA \\\n"
            f"      --output-dir {fa_skeleton_dir}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("✓ FA skeleton validation successful")
    logger.info(f"  FA directory: {required_files['fa_dir']}")
    logger.info(f"  Stats directory: {required_files['stats_dir']}")
    logger.info(f"  Mean FA skeleton: {required_files['mean_fa_skeleton']}")
    logger.info(f"  Skeletonised FA: {required_files['all_fa_skeletonised']}")

    return required_files


def run_tbss_non_fa(
    metric: str,
    fa_skeleton_dir: Path,
    output_dir: Path,
    subjects_data: List[SubjectData],
    logger: logging.Logger
) -> bool:
    """
    Run tbss_non_FA to project non-FA metric onto FA skeleton

    Args:
        metric: Non-FA metric name (e.g., 'MD', 'MK', 'FICVF')
                For DKI: Uses short form (MK, AK, RK, KFA) but handles DIPY file names
        fa_skeleton_dir: Directory containing FA skeleton
        output_dir: Output directory (typically same as fa_skeleton_dir)
        subjects_data: List of subject data
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("=" * 80)
        logger.info(f"Running tbss_non_FA for {metric}")
        logger.info("=" * 80)

        # Validate FA skeleton exists
        fa_files = validate_fa_skeleton(fa_skeleton_dir, logger)

        # Read subject list from FA preparation to ensure same order
        with open(fa_files['subject_list'], 'r') as f:
            fa_subject_order = [line.strip() for line in f]

        logger.info(f"FA skeleton has {len(fa_subject_order)} subjects")

        # Create directory for the new metric
        # tbss_non_FA expects files in: {METRIC}/{subject_id}.nii.gz (no metric suffix)
        metric_dir = output_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Copy metric files to metric directory with proper naming
        logger.info(f"\nCopying {metric} files to {metric}/ directory...")
        included_subjects = [s for s in subjects_data if s.included]

        # Create mapping of subject_id to metric file
        subject_to_file = {s.subject_id: s.metric_file for s in included_subjects}

        # Copy files in FA subject order
        copied_count = 0
        for subject_id in fa_subject_order:
            if subject_id not in subject_to_file:
                logger.warning(f"  ! {subject_id}: Not found in {metric} data (was in FA)")
                continue

            # tbss_non_FA expects: {METRIC}/{subject_id}.nii.gz (WITHOUT metric suffix)
            dest_file = metric_dir / f"{subject_id}.nii.gz"
            shutil.copy2(subject_to_file[subject_id], dest_file)
            copied_count += 1
            logger.info(f"  ✓ {subject_id}")

        logger.info(f"\nCopied {copied_count}/{len(fa_subject_order)} subjects")

        if copied_count < len(fa_subject_order):
            logger.warning(
                f"Warning: Only {copied_count}/{len(fa_subject_order)} subjects have {metric} data. "
                f"Statistical analysis will use only these subjects."
            )

        # Run tbss_non_FA
        logger.info(f"\nRunning tbss_non_FA...")
        original_dir = os.getcwd()
        os.chdir(output_dir)

        env = os.environ.copy()
        env['FSLSUBALREADYRUN'] = 'true'

        cmd = ['tbss_non_FA', metric]
        logger.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )

        os.chdir(original_dir)

        if result.returncode != 0:
            logger.error(f"tbss_non_FA failed: {result.stderr}")
            return False

        # Verify output was created
        stats_dir = output_dir / "stats"
        expected_output = stats_dir / f"all_{metric}_skeletonised.nii.gz"

        if not expected_output.exists():
            logger.error(f"Expected output not found: {expected_output}")
            return False

        logger.info(f"✓ Successfully created: {expected_output}")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"tbss_non_FA failed: {str(e)}", exc_info=True)
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
    subjects: Optional[List[str]] = None,
    fa_skeleton_dir: Optional[Path] = None
) -> Dict:
    """
    Main workflow: Prepare TBSS analysis from preprocessed DTI/DKI/NODDI data

    Workflow:
    - For FA: Runs full TBSS pipeline to create skeleton
    - For non-FA: Uses tbss_non_FA to project onto existing FA skeleton

    Args:
        config: Configuration dictionary
        metric: Metric to analyze (FA, MD, AD, RD, MK, AK, RK, KFA, FICVF, ODI, FISO)
        output_dir: Output directory for prepared analysis
        subjects: Optional list of specific subjects to include (if None, discovers all)
        fa_skeleton_dir: Required for non-FA metrics. Directory containing FA skeleton.
                        If None, uses output_dir (assumes FA is in same location)

    Returns:
        Dictionary containing preparation results and manifest
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    # Determine if this is FA or non-FA metric
    is_fa = (metric == 'FA')

    logger.info("=" * 80)
    logger.info("TBSS Data Preparation Workflow")
    logger.info("=" * 80)
    logger.info(f"Metric: {metric}")
    logger.info(f"Workflow: {'Full TBSS pipeline (FA skeleton creation)' if is_fa else 'tbss_non_FA projection'}")
    logger.info(f"Output directory: {output_dir}")

    # For non-FA metrics, determine FA skeleton directory
    if not is_fa:
        if fa_skeleton_dir is None:
            fa_skeleton_dir = output_dir
            logger.info(f"FA skeleton directory: {fa_skeleton_dir} (using output_dir)")
        else:
            fa_skeleton_dir = Path(fa_skeleton_dir)
            logger.info(f"FA skeleton directory: {fa_skeleton_dir}")

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

    # Route to appropriate workflow: FA (full pipeline) or non-FA (tbss_non_FA)
    if is_fa:
        # FA: Run full TBSS pipeline to create skeleton
        logger.info("\n" + "=" * 80)
        logger.info("FA METRIC: Running full TBSS pipeline")
        logger.info("=" * 80)

        tbss_input_dir = copy_to_tbss_structure(
            subjects_data=subjects_data,
            output_dir=output_dir,
            metric=metric,
            logger=logger
        )

        success = run_tbss_pipeline(
            tbss_input_dir=tbss_input_dir,
            output_dir=output_dir,
            metric=metric,
            logger=logger
        )

        if not success:
            return {"success": False, "error": "TBSS pipeline failed"}

    else:
        # Non-FA: Use tbss_non_FA to project onto FA skeleton
        logger.info("\n" + "=" * 80)
        logger.info(f"NON-FA METRIC ({metric}): Using tbss_non_FA projection")
        logger.info("=" * 80)

        success = run_tbss_non_fa(
            metric=metric,
            fa_skeleton_dir=fa_skeleton_dir,
            output_dir=output_dir,
            subjects_data=subjects_data,
            logger=logger
        )

        if not success:
            return {"success": False, "error": f"tbss_non_FA failed for {metric}"}

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

    if is_fa:
        logger.info("\nNext steps:")
        logger.info("  1. Review subject_manifest.json for excluded subjects")
        logger.info("  2. Prepare other metrics (MD, AD, RD, MK, etc.) using --fa-skeleton-dir")
        logger.info("     Example:")
        logger.info(f"       python -m neurovrai.analysis.tbss.prepare_tbss \\")
        logger.info(f"           --config config.yaml \\")
        logger.info(f"           --metric MD \\")
        logger.info(f"           --fa-skeleton-dir {output_dir} \\")
        logger.info(f"           --output-dir {output_dir}")
        logger.info("  3. Run statistical analysis with run_tbss_stats.py")
    else:
        logger.info("\nNext steps:")
        logger.info(f"  1. Review subject_manifest.json for excluded subjects")
        logger.info(f"  2. Prepare additional metrics if needed")
        logger.info(f"  3. Run statistical analysis with run_tbss_stats.py on {metric}")

    logger.info("=" * 80)

    return {
        "success": True,
        "manifest": manifest,
        "output_dir": str(output_dir)
    }


def main():
    """Command-line interface for TBSS preparation"""
    parser = argparse.ArgumentParser(
        description="Prepare DTI/DKI/NODDI data for TBSS group analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. MUST run with --metric FA first to create the FA skeleton
  2. Then run with other metrics to project onto the FA skeleton

Examples:
  # Step 1: Prepare FA skeleton (REQUIRED FIRST)
  python -m neurovrai.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --metric FA \\
      --output-dir /study/analysis/tbss/

  # Step 2: Project DTI metrics onto FA skeleton
  python -m neurovrai.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --metric MD \\
      --fa-skeleton-dir /study/analysis/tbss/ \\
      --output-dir /study/analysis/tbss/

  # Step 3: Project DKI metrics (if available)
  python -m neurovrai.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --metric MK \\
      --fa-skeleton-dir /study/analysis/tbss/ \\
      --output-dir /study/analysis/tbss/

  # Step 4: Project NODDI metrics (if available)
  python -m neurovrai.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --metric FICVF \\
      --fa-skeleton-dir /study/analysis/tbss/ \\
      --output-dir /study/analysis/tbss/

  # Use subjects from design matrix (RECOMMENDED)
  python -m neurovrai.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --metric FA \\
      --subject-list /study/designs/dki/subject_list.txt \\
      --output-dir /study/analysis/tbss/

  # Or specify subjects manually
  python -m neurovrai.analysis.tbss.prepare_tbss \\
      --config config.yaml \\
      --metric FA \\
      --subjects IRC805-0580101 IRC805-1230101 \\
      --output-dir /study/analysis/tbss/

Supported Metrics:
  DTI:    FA, MD, AD, RD
  DKI:    MK (mean kurtosis), AK (axial), RK (radial), KFA
  NODDI:  FICVF (neurite density), ODI (dispersion), FISO (isotropic)
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
        choices=ALL_METRICS,
        help='Metric to analyze (DTI: FA/MD/AD/RD, DKI: MK/AK/RK/KFA, NODDI: FICVF/ODI/FISO)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for prepared TBSS analysis'
    )

    parser.add_argument(
        '--fa-skeleton-dir',
        type=Path,
        help='Directory containing FA skeleton (required for non-FA metrics). '
             'If not specified, uses --output-dir'
    )

    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='Specific subjects to include (default: discover all)'
    )

    parser.add_argument(
        '--subject-list',
        type=Path,
        help='Path to text file with subject IDs (one per line). '
             'If provided, only these subjects will be used in the exact order specified. '
             'This should match your design matrix subject order.'
    )

    args = parser.parse_args()

    # Validate mutually exclusive subject specification
    if args.subjects and args.subject_list:
        parser.error("Cannot specify both --subjects and --subject-list")

    # Read subject list from file if provided
    subjects = None
    if args.subject_list:
        if not args.subject_list.exists():
            parser.error(f"Subject list file not found: {args.subject_list}")

        with open(args.subject_list, 'r') as f:
            subjects = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(subjects)} subjects from {args.subject_list}")
        print(f"Subjects will be processed in the order specified in the file.")
    elif args.subjects:
        subjects = args.subjects

    # Validate that FA skeleton dir is provided or same as output_dir for non-FA metrics
    if args.metric != 'FA' and args.fa_skeleton_dir is None:
        print(f"Warning: --fa-skeleton-dir not specified for non-FA metric '{args.metric}'")
        print(f"Using --output-dir as FA skeleton location: {args.output_dir}")
        print("This assumes FA skeleton was already created in this directory.\n")

    # Load config
    config = load_config(args.config)

    # Run preparation
    result = prepare_tbss_analysis(
        config=config,
        metric=args.metric,
        output_dir=args.output_dir,
        subjects=subjects,
        fa_skeleton_dir=args.fa_skeleton_dir
    )

    # Exit with appropriate code
    exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
