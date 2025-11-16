#!/usr/bin/env python3
"""
Continuous MRI Preprocessing Pipeline

This script implements a streaming/continuous pipeline that:
1. Monitors DICOM conversion progress
2. Starts anatomical workflow as soon as anatomical files are available
3. Starts other modality workflows as their files become available
4. Runs workflows concurrently as resources allow

This is ideal for large datasets where conversion takes significant time.

Usage:
    python run_continuous_pipeline.py \
        --subject IRC805-1580101 \
        --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-1580101 \
        --config config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
import concurrent.futures
import subprocess
import threading

# Import utilities
from mri_preprocess.config import load_config
from mri_preprocess.utils.config_validator import validate_all_workflows

# Import workflows
from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing
from mri_preprocess.workflows.dwi_preprocess import run_dwi_multishell_topup_preprocessing
from mri_preprocess.workflows.func_preprocess import run_func_preprocessing
from mri_preprocess.workflows.asl_preprocess import run_asl_preprocessing


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ContinuousPipelineOrchestrator:
    """
    Orchestrates a continuous streaming MRI preprocessing pipeline.

    Monitors DICOM conversion and starts workflows as data becomes available.
    """

    def __init__(
        self,
        subject: str,
        config: Dict,
        study_root: Path,
        dicom_dir: Path,
        modalities: Optional[List[str]] = None
    ):
        """
        Initialize continuous pipeline orchestrator.

        Args:
            subject: Subject identifier
            config: Configuration dictionary
            study_root: Study root directory
            dicom_dir: DICOM directory
            modalities: List of modalities to process (default: all)
        """
        self.subject = subject
        self.config = config
        self.study_root = Path(study_root)
        self.dicom_dir = Path(dicom_dir)

        # Default to all modalities if not specified
        self.modalities = modalities or ['anat', 'dwi', 'func', 'asl']

        # Setup directories
        self.bids_dir = self.study_root / 'bids' / subject
        self.derivatives_dir = self.study_root / 'derivatives'
        self.work_dir = self.study_root / 'work' / subject  # Subject-specific work directory
        self.qc_dir = self.study_root / 'qc'

        # Create directories
        for dir_path in [self.derivatives_dir, self.work_dir, self.qc_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.converted_modalities: Set[str] = set()
        self.completed_workflows: Set[str] = set()
        self.conversion_process = None
        self.conversion_complete = False

        # Thread safety
        self.lock = threading.Lock()

        # Results
        self.results = {
            'anatomical': None,
            'dwi': None,
            'functional': None,
            'asl': None
        }

        logger.info("="*70)
        logger.info(f"INITIALIZING CONTINUOUS PIPELINE FOR {subject}")
        logger.info("="*70)
        logger.info(f"  Study root: {study_root}")
        logger.info(f"  DICOM directory: {dicom_dir}")
        logger.info(f"  Modalities: {', '.join(self.modalities)}")
        logger.info("")

    def run(self) -> Dict:
        """
        Run the continuous pipeline.

        Returns:
            Dictionary with results from each workflow
        """
        try:
            # Step 1: Validate configuration
            logger.info("STEP 1: Configuration Validation")
            logger.info("-" * 70)
            validation_results = self._validate_config()
            logger.info("")

            if not all(validation_results.values()):
                logger.error("Configuration validation failed")
                return self.results

            # Step 2: Start DICOM conversion in background
            logger.info("STEP 2: Starting DICOM Conversion (Background)")
            logger.info("-" * 70)
            self._start_dicom_conversion()
            logger.info("")

            # Step 3: Monitor and run workflows as data becomes available
            logger.info("STEP 3: Continuous Workflow Execution")
            logger.info("-" * 70)
            self._run_continuous_workflows()
            logger.info("")

            # Step 4: Wait for all workflows to complete
            logger.info("STEP 4: Waiting for All Workflows to Complete")
            logger.info("-" * 70)
            self._wait_for_completion()
            logger.info("")

            # Step 5: Summary
            self._print_summary()

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self.results

    def _validate_config(self) -> Dict[str, bool]:
        """Validate configuration for all workflows."""
        return validate_all_workflows(self.config, has_dicom=True)

    def _start_dicom_conversion(self):
        """Start DICOM conversion in background subprocess."""
        from mri_preprocess.utils.dicom_converter import convert_subject_dicoms

        # Create conversion function for subprocess
        def run_conversion():
            try:
                logger.info("  Starting DICOM to NIfTI conversion...")
                output_dir = self.study_root / 'bids'

                results = convert_subject_dicoms(
                    subject=self.subject,
                    dicom_dir=self.dicom_dir,
                    output_dir=output_dir
                )

                with self.lock:
                    self.conversion_complete = True

                logger.info("  ✓ DICOM conversion complete")
                return results

            except Exception as e:
                logger.error(f"  ✗ DICOM conversion failed: {e}")
                with self.lock:
                    self.conversion_complete = True
                return None

        # Start conversion in thread
        self.conversion_thread = threading.Thread(target=run_conversion, daemon=False)
        self.conversion_thread.start()
        logger.info("  DICOM conversion running in background...")

    def _check_modality_ready(self, modality: str) -> bool:
        """
        Check if a modality's NIfTI files are available.

        Args:
            modality: Modality to check ('anat', 'dwi', 'func', 'asl')

        Returns:
            True if files exist, False otherwise
        """
        modality_dir = self.bids_dir / modality
        if not modality_dir.exists():
            return False

        # Check for modality-specific files
        patterns = {
            'anat': '*T1*.nii.gz',
            'dwi': '*DTI*.nii.gz',
            'func': '*RESTING*.nii.gz',
            'asl': '*pCASL*.nii.gz'
        }

        pattern = patterns.get(modality, '*.nii.gz')
        files = list(modality_dir.glob(pattern))

        return len(files) > 0

    def _run_continuous_workflows(self):
        """
        Monitor conversion and start workflows as data becomes available.

        Workflow execution order:
        1. Anatomical (runs first, required by all others)
        2. DWI, Functional, ASL (run in parallel after anatomical completes)
        """
        # Track running workflow futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        running_futures = {}

        anatomical_started = False
        anatomical_complete = False

        # Monitoring loop
        logger.info("  Monitoring for converted files...")
        logger.info("")

        check_interval = 10  # Check every 10 seconds
        max_checks = 120     # Maximum 20 minutes (120 * 10s)
        checks = 0

        while checks < max_checks:
            checks += 1
            time.sleep(check_interval)

            # Check anatomical first (highest priority)
            if 'anat' in self.modalities and not anatomical_started:
                if self._check_modality_ready('anat'):
                    logger.info("  ✓ Anatomical files ready - starting anatomical workflow")
                    future = executor.submit(self._run_anatomical)
                    running_futures['anatomical'] = future
                    anatomical_started = True

            # Check if anatomical is complete
            if anatomical_started and not anatomical_complete:
                if running_futures['anatomical'].done():
                    try:
                        result = running_futures['anatomical'].result()
                        with self.lock:
                            self.results['anatomical'] = result
                            self.completed_workflows.add('anat')
                        anatomical_complete = True
                        logger.info("  ✓ Anatomical workflow complete - other modalities can now start")
                        logger.info("")
                    except Exception as e:
                        logger.error(f"  ✗ Anatomical workflow failed: {e}")
                        anatomical_complete = True  # Don't block other modalities

            # Once anatomical is complete, start other modalities as they become ready
            if anatomical_complete:
                # Check DWI
                if 'dwi' in self.modalities and 'dwi' not in running_futures:
                    if self._check_modality_ready('dwi'):
                        logger.info("  ✓ DWI files ready - starting DWI workflow")
                        future = executor.submit(self._run_dwi)
                        running_futures['dwi'] = future

                # Check Functional
                if 'func' in self.modalities and 'functional' not in running_futures:
                    if self._check_modality_ready('func'):
                        logger.info("  ✓ Functional files ready - starting functional workflow")
                        future = executor.submit(self._run_functional)
                        running_futures['functional'] = future

                # Check ASL
                if 'asl' in self.modalities and 'asl' not in running_futures:
                    if self._check_modality_ready('asl'):
                        logger.info("  ✓ ASL files ready - starting ASL workflow")
                        future = executor.submit(self._run_asl)
                        running_futures['asl'] = future

            # Check if all workflows are started
            expected_workflows = set()
            if 'anat' in self.modalities:
                expected_workflows.add('anatomical')
            if 'dwi' in self.modalities:
                expected_workflows.add('dwi')
            if 'func' in self.modalities:
                expected_workflows.add('functional')
            if 'asl' in self.modalities:
                expected_workflows.add('asl')

            all_started = expected_workflows.issubset(set(running_futures.keys()))

            if all_started:
                logger.info("  All workflows started - monitoring completion...")
                break

            # Check if we've been monitoring too long
            # Continue monitoring while workflows are running, even if conversion is complete
            # Only exit if max_checks reached AND conversion is complete
            with self.lock:
                conversion_done = self.conversion_complete

            if checks >= max_checks and conversion_done:
                logger.warning(f"  ⚠ Max monitoring time reached ({max_checks * check_interval}s)")
                # List any modalities that didn't start
                missing = []
                for mod in self.modalities:
                    workflow_name = 'anatomical' if mod == 'anat' else 'functional' if mod == 'func' else mod
                    if workflow_name not in running_futures:
                        missing.append(mod)
                if missing:
                    logger.warning(f"  ⚠ Workflows not started: {', '.join(missing)}")
                break

        # Store futures for completion check
        self.running_futures = running_futures
        self.executor = executor

    def _wait_for_completion(self):
        """Wait for all running workflows to complete."""
        if not hasattr(self, 'running_futures'):
            return

        # Wait for conversion thread
        if hasattr(self, 'conversion_thread'):
            logger.info("  Waiting for DICOM conversion to complete...")
            self.conversion_thread.join()

        # Wait for all workflow futures
        logger.info("  Waiting for all workflows to complete...")
        for workflow_name, future in self.running_futures.items():
            try:
                result = future.result(timeout=7200)  # 2 hour timeout per workflow
                with self.lock:
                    if workflow_name not in self.results or self.results[workflow_name] is None:
                        self.results[workflow_name] = result
                logger.info(f"    ✓ {workflow_name} complete")
            except Exception as e:
                logger.error(f"    ✗ {workflow_name} failed: {e}")

        # Shutdown executor
        self.executor.shutdown(wait=True)

    def _run_anatomical(self) -> Optional[Dict]:
        """Run anatomical preprocessing workflow."""
        try:
            # Find T1w file
            anat_dir = self.bids_dir / 'anat'
            t1w_files = list(anat_dir.glob('*T1*.nii.gz'))

            if not t1w_files:
                logger.error("    T1w file not found")
                return None

            t1w_file = t1w_files[0]
            logger.info(f"    Input: {t1w_file.name}")

            results = run_anat_preprocessing(
                config=self.config,
                subject=self.subject,
                t1w_file=t1w_file,
                output_dir=self.derivatives_dir,
                work_dir=self.work_dir
            )

            return results

        except Exception as e:
            logger.error(f"    Anatomical preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _run_dwi(self) -> Optional[Dict]:
        """Run DWI preprocessing workflow."""
        try:
            # Find DWI files
            dwi_dir = self.bids_dir / 'dwi'
            dwi_files = list(dwi_dir.glob('*DTI*.nii.gz'))

            if not dwi_files:
                logger.warning("    No DWI files found")
                return None

            # Find bval/bvec files
            bval_files = [f.with_suffix('').with_suffix('.bval') for f in dwi_files]
            bvec_files = [f.with_suffix('').with_suffix('.bvec') for f in dwi_files]

            # Find reverse phase files for TOPUP
            rev_phase_files = list(dwi_dir.glob('*SE_EPI*.nii.gz'))

            logger.info(f"    Found {len(dwi_files)} DWI files")
            logger.info(f"    Found {len(rev_phase_files)} reverse phase files")

            results = run_dwi_multishell_topup_preprocessing(
                config=self.config,
                subject=self.subject,
                dwi_files=dwi_files,
                bval_files=bval_files,
                bvec_files=bvec_files,
                rev_phase_files=rev_phase_files if rev_phase_files else None,
                output_dir=self.derivatives_dir,
                work_dir=self.work_dir
            )

            return results

        except Exception as e:
            logger.error(f"    DWI preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _run_functional(self) -> Optional[Dict]:
        """Run functional preprocessing workflow."""
        try:
            # Find functional files
            func_dir = self.bids_dir / 'func'
            func_files = list(func_dir.glob('*RESTING*.nii.gz'))

            if not func_files:
                logger.warning("    No functional files found")
                return None

            # Get anatomical reference from anatomical workflow results
            if 'anatomical' not in self.results or not self.results['anatomical']:
                logger.error("    Anatomical preprocessing not completed")
                return None

            anat_results = self.results['anatomical']
            t1w_brain = anat_results.get('brain')

            if not t1w_brain or not Path(t1w_brain).exists():
                logger.error("    Anatomical brain file not found")
                return None

            # Multi-echo detection
            is_multi_echo = len(func_files) > 1 or 'ME' in func_files[0].name

            logger.info(f"    Found {len(func_files)} functional files")
            logger.info(f"    Multi-echo: {is_multi_echo}")

            results = run_func_preprocessing(
                config=self.config,
                subject=self.subject,
                func_file=func_files,  # Fixed: parameter name is 'func_file' (singular)
                output_dir=self.derivatives_dir,
                anat_derivatives=self.derivatives_dir / self.subject / 'anat',
                work_dir=self.work_dir
            )

            return results

        except Exception as e:
            logger.error(f"    Functional preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _run_asl(self) -> Optional[Dict]:
        """Run ASL preprocessing workflow."""
        try:
            # Find ASL files
            asl_dir = self.bids_dir / 'asl'
            asl_files = list(asl_dir.glob('*pCASL*.nii.gz'))

            if not asl_files:
                logger.warning("    No ASL files found")
                return None

            # Use the SOURCE file if available
            source_files = [f for f in asl_files if 'SOURCE' in f.name]
            asl_file = source_files[0] if source_files else asl_files[0]

            # Get anatomical reference and tissue masks from anatomical workflow results
            if 'anatomical' not in self.results or not self.results['anatomical']:
                logger.error("    Anatomical preprocessing not completed")
                return None

            anat_results = self.results['anatomical']
            t1w_brain = anat_results.get('brain')
            gm_mask = anat_results.get('gm_prob')
            wm_mask = anat_results.get('wm_prob')
            csf_mask = anat_results.get('csf_prob')

            if not t1w_brain or not Path(t1w_brain).exists():
                logger.error("    Anatomical brain file not found")
                return None

            # Find DICOM directory for parameter extraction
            dicom_asl_dir = None
            if self.dicom_dir:
                # Find ASL DICOM subdirectory
                date_dirs = list(self.dicom_dir.glob('*'))
                for date_dir in date_dirs:
                    asl_subdirs = list(date_dir.glob('*pCASL*'))
                    if asl_subdirs:
                        dicom_asl_dir = asl_subdirs[0]
                        break

            logger.info(f"    Input: {asl_file.name}")
            logger.info(f"    DICOM parameters: {'Yes' if dicom_asl_dir else 'No (using config)'}")

            results = run_asl_preprocessing(
                config=self.config,
                subject=self.subject,
                asl_file=asl_file,
                output_dir=self.derivatives_dir,
                t1w_brain=t1w_brain,
                gm_mask=gm_mask if gm_mask and Path(gm_mask).exists() else None,
                wm_mask=wm_mask if wm_mask and Path(wm_mask).exists() else None,
                csf_mask=csf_mask if csf_mask and Path(csf_mask).exists() else None,
                dicom_dir=dicom_asl_dir
            )

            return results

        except Exception as e:
            logger.error(f"    ASL preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _print_summary(self):
        """Print pipeline execution summary."""
        logger.info("="*70)
        logger.info("CONTINUOUS PIPELINE SUMMARY")
        logger.info("="*70)
        logger.info("")

        # Workflows
        for workflow_name in ['anatomical', 'dwi', 'functional', 'asl']:
            mod_name = {
                'anatomical': 'anat',
                'functional': 'func',
                'dwi': 'dwi',
                'asl': 'asl'
            }[workflow_name]

            if mod_name not in self.modalities:
                continue

            result = self.results.get(workflow_name)
            if result:
                logger.info(f"  {workflow_name.capitalize()}: ✓ Complete")
            else:
                logger.info(f"  {workflow_name.capitalize()}: ✗ Failed or not run")

        logger.info("")
        logger.info(f"Results saved to: {self.derivatives_dir}")
        logger.info(f"Working files in: {self.work_dir}")
        logger.info("")
        logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run continuous MRI preprocessing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--subject',
        required=True,
        help='Subject identifier (e.g., IRC805-1580101)'
    )
    parser.add_argument(
        '--dicom-dir',
        type=Path,
        required=True,
        help='Path to subject DICOM directory'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Path to configuration file (default: config.yaml)'
    )

    # Optional arguments
    parser.add_argument(
        '--study-root',
        type=Path,
        help='Study root directory (default: inferred from DICOM directory)'
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        choices=['anat', 'dwi', 'func', 'asl'],
        help='Modalities to process (default: all)'
    )

    args = parser.parse_args()

    # Load configuration
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Determine study root
    if args.study_root:
        study_root = args.study_root
    else:
        # Infer study root from DICOM directory structure
        # Expected: {study_root}/raw/dicom/{subject} → {study_root}
        # E.g., /mnt/bytopia/IRC805/raw/dicom/IRC805-1580101 → /mnt/bytopia/IRC805
        dicom_parent = args.dicom_dir.parent  # .../raw/dicom
        if dicom_parent.name == 'dicom':
            study_root = dicom_parent.parent.parent  # 3 levels up: .../raw/dicom → .../raw → .../study
        else:
            # Fallback: assume 2 levels up
            study_root = args.dicom_dir.parent.parent

        logger.info(f"Auto-detected study root: {study_root}")

    # Override study root in config if provided
    if args.study_root:
        config['project_dir'] = str(study_root)

    # Create and run orchestrator
    orchestrator = ContinuousPipelineOrchestrator(
        subject=args.subject,
        config=config,
        study_root=study_root,
        dicom_dir=args.dicom_dir,
        modalities=args.modalities
    )

    results = orchestrator.run()

    # Exit with appropriate code
    all_success = all(
        results.get(workflow) is not None
        for workflow in ['anatomical', 'dwi', 'functional', 'asl']
        if workflow.replace('anatomical', 'anat').replace('functional', 'func') in (args.modalities or ['anat', 'dwi', 'func', 'asl'])
    )

    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    main()
