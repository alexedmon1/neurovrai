#!/usr/bin/env python3
"""
Complete MRI Preprocessing Pipeline Orchestrator

This script runs the complete preprocessing pipeline from DICOM or NIfTI:

1. DICOM Conversion (if starting from DICOM)
   - Automatic modality detection
   - Parameter extraction from DICOM headers
   - Organized NIfTI output structure

2. Configuration Validation
   - Validates config for each modality workflow
   - Warns about missing optional parameters

3. Preprocessing Workflows (with dependency management)
   - Anatomical preprocessing (runs first - required by all)
   - DWI, Functional, ASL preprocessing (run in parallel after anatomical)

4. Quality Control
   - Automated QC for all modalities

Usage:
    # From DICOM directory
    python run_full_pipeline.py \\
        --subject IRC805-1580101 \\
        --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-1580101 \\
        --config config.yaml

    # From NIfTI directory (DICOM conversion skipped)
    python run_full_pipeline.py \\
        --subject IRC805-1580101 \\
        --nifti-dir /mnt/bytopia/IRC805/nifti/IRC805-1580101 \\
        --config config.yaml

    # Specify which modalities to run
    python run_full_pipeline.py \\
        --subject IRC805-1580101 \\
        --dicom-dir /mnt/bytopia/IRC805/raw/dicom/IRC805-1580101 \\
        --modalities anat dwi func \\
        --config config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures
import time

# Import utilities
from mri_preprocess.config import load_config
from mri_preprocess.utils.dicom_converter import convert_subject_dicoms
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
        logging.FileHandler('logs/pipeline_orchestrator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the complete MRI preprocessing pipeline.

    Handles:
    - Input detection (DICOM vs NIfTI)
    - DICOM conversion if needed
    - Config validation
    - Workflow execution with dependencies
    - Parallel execution where possible
    """

    def __init__(
        self,
        subject: str,
        config: Dict,
        study_root: Path,
        dicom_dir: Optional[Path] = None,
        nifti_dir: Optional[Path] = None,
        modalities: Optional[List[str]] = None
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            subject: Subject identifier
            config: Configuration dictionary
            study_root: Study root directory
            dicom_dir: DICOM directory (if starting from DICOM)
            nifti_dir: NIfTI directory (if starting from NIfTI)
            modalities: List of modalities to process (default: all)
        """
        self.subject = subject
        self.config = config
        self.study_root = Path(study_root)
        self.dicom_dir = Path(dicom_dir) if dicom_dir else None
        self.nifti_dir = Path(nifti_dir) if nifti_dir else None

        # Default to all modalities if not specified
        self.modalities = modalities or ['anat', 'dwi', 'func', 'asl']

        # Setup directories
        self.derivatives_dir = self.study_root / 'derivatives'
        self.work_dir = self.study_root / 'work'
        self.qc_dir = self.study_root / 'qc'

        # Create directories
        for dir_path in [self.derivatives_dir, self.work_dir, self.qc_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.results = {
            'dicom_conversion': None,
            'anatomical': None,
            'dwi': None,
            'functional': None,
            'asl': None
        }

        logger.info("="*70)
        logger.info(f"INITIALIZING PIPELINE FOR {subject}")
        logger.info("="*70)
        logger.info(f"  Study root: {study_root}")
        logger.info(f"  Modalities: {', '.join(self.modalities)}")
        logger.info(f"  Input type: {'DICOM' if dicom_dir else 'NIfTI'}")
        logger.info("")

    def run(self) -> Dict:
        """
        Run the complete pipeline.

        Returns:
            Dictionary with results from each workflow
        """
        try:
            # Step 1: DICOM Conversion (if needed)
            if self.dicom_dir:
                logger.info("STEP 1: DICOM Conversion")
                logger.info("-" * 70)
                self.results['dicom_conversion'] = self._run_dicom_conversion()
                logger.info("")
            else:
                logger.info("STEP 1: DICOM Conversion - SKIPPED (using existing NIfTI)")
                logger.info("")

            # Step 2: Configuration Validation
            logger.info("STEP 2: Configuration Validation")
            logger.info("-" * 70)
            validation_results = self._validate_config()
            logger.info("")

            # Check if any required workflows are invalid
            invalid_workflows = [
                mod for mod in self.modalities
                if mod == 'anat' and not validation_results.get('anatomical', False)
                or mod == 'dwi' and not validation_results.get('dwi', False)
                or mod == 'func' and not validation_results.get('functional', False)
                or mod == 'asl' and not validation_results.get('asl', False)
            ]

            if invalid_workflows:
                logger.error(f"Configuration invalid for: {', '.join(invalid_workflows)}")
                logger.error("Please fix configuration and retry")
                return self.results

            # Step 3: Anatomical Preprocessing (MUST run first)
            if 'anat' in self.modalities:
                logger.info("STEP 3: Anatomical Preprocessing")
                logger.info("-" * 70)
                self.results['anatomical'] = self._run_anatomical()
                logger.info("")

                if not self.results['anatomical']:
                    logger.error("Anatomical preprocessing failed - cannot continue")
                    return self.results
            else:
                logger.warning("STEP 3: Anatomical Preprocessing - SKIPPED")
                logger.warning("  Warning: Other modalities may fail without anatomical preprocessing")
                logger.info("")

            # Step 4: Other Modalities (can run in parallel)
            other_modalities = [m for m in self.modalities if m != 'anat']
            if other_modalities:
                logger.info("STEP 4: Modality-Specific Preprocessing (Parallel)")
                logger.info("-" * 70)
                self._run_parallel_workflows(other_modalities)
                logger.info("")

            # Step 5: Summary
            self._print_summary()

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self.results

    def _run_dicom_conversion(self) -> Optional[Dict]:
        """Run DICOM to NIfTI conversion."""
        try:
            # Determine output directory
            # BIDS structure: /study_root/bids/{subject}/{modality}/
            if self.nifti_dir:
                # User provided nifti_dir, use parent to get bids dir
                # E.g., /path/to/bids/{subject} → /path/to/bids
                output_dir = self.nifti_dir.parent
            else:
                # Default to bids subdirectory
                output_dir = self.study_root / 'bids'

            results = convert_subject_dicoms(
                subject=self.subject,
                dicom_dir=self.dicom_dir,
                output_dir=output_dir
            )

            # Update nifti_dir to point to converted files
            # Converter creates: {output_dir}/{subject}/{modality}/
            self.nifti_dir = output_dir / self.subject

            return results

        except Exception as e:
            logger.error(f"DICOM conversion failed: {e}")
            return None

    def _validate_config(self) -> Dict[str, bool]:
        """Validate configuration for all workflows."""
        has_dicom = self.dicom_dir is not None
        return validate_all_workflows(self.config, has_dicom=has_dicom)

    def _run_anatomical(self) -> Optional[Dict]:
        """Run anatomical preprocessing workflow."""
        try:
            # Find T1w file
            t1w_file = self._find_file('anat', '*T1*.nii.gz')
            if not t1w_file:
                logger.error("T1w file not found")
                return None

            logger.info(f"  Input: {t1w_file.name}")

            results = run_anat_preprocessing(
                config=self.config,
                subject=self.subject,
                t1w_file=t1w_file,
                output_dir=self.derivatives_dir,
                work_dir=self.work_dir
            )

            logger.info("  ✓ Anatomical preprocessing complete")
            return results

        except Exception as e:
            logger.error(f"Anatomical preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _run_dwi(self) -> Optional[Dict]:
        """Run DWI preprocessing workflow."""
        try:
            logger.info("  DWI Preprocessing")

            # Find DWI files
            dwi_files = list((self.nifti_dir / 'dwi').glob('*DTI*.nii.gz'))
            if not dwi_files:
                logger.warning("    No DWI files found")
                return None

            # Find bval/bvec files
            bval_files = [f.with_suffix('').with_suffix('.bval') for f in dwi_files]
            bvec_files = [f.with_suffix('').with_suffix('.bvec') for f in dwi_files]

            # Find reverse phase files for TOPUP
            rev_phase_files = list((self.nifti_dir / 'dwi').glob('*SE_EPI*.nii.gz'))

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

            logger.info("    ✓ DWI preprocessing complete")
            return results

        except Exception as e:
            logger.error(f"    ✗ DWI preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _run_functional(self) -> Optional[Dict]:
        """Run functional preprocessing workflow."""
        try:
            logger.info("  Functional Preprocessing")

            # Find functional files
            func_files = list((self.nifti_dir / 'func').glob('*RESTING*.nii.gz'))
            if not func_files:
                logger.warning("    No functional files found")
                return None

            # Get anatomical reference
            anat_dir = self.derivatives_dir / self.subject / 'anat'
            t1w_brain = anat_dir / 'brain.nii.gz'

            if not t1w_brain.exists():
                logger.error("    Anatomical preprocessing outputs not found")
                return None

            # Multi-echo detection
            is_multi_echo = len(func_files) > 1 or 'ME' in func_files[0].name

            logger.info(f"    Found {len(func_files)} functional files")
            logger.info(f"    Multi-echo: {is_multi_echo}")

            results = run_func_preprocessing(
                config=self.config,
                subject=self.subject,
                func_files=func_files,
                output_dir=self.derivatives_dir,
                t1w_brain=t1w_brain,
                work_dir=self.work_dir
            )

            logger.info("    ✓ Functional preprocessing complete")
            return results

        except Exception as e:
            logger.error(f"    ✗ Functional preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _run_asl(self) -> Optional[Dict]:
        """Run ASL preprocessing workflow."""
        try:
            logger.info("  ASL Preprocessing")

            # Find ASL files
            asl_files = list((self.nifti_dir / 'asl').glob('*pCASL*.nii.gz'))
            if not asl_files:
                logger.warning("    No ASL files found")
                return None

            # Use the SOURCE file if available
            source_files = [f for f in asl_files if 'SOURCE' in f.name]
            asl_file = source_files[0] if source_files else asl_files[0]

            # Get anatomical reference and tissue masks
            anat_dir = self.derivatives_dir / self.subject / 'anat'
            t1w_brain = anat_dir / 'brain.nii.gz'
            seg_dir = anat_dir / 'segmentation'

            gm_mask = seg_dir / 'POSTERIOR_02.nii.gz'
            wm_mask = seg_dir / 'POSTERIOR_03.nii.gz'
            csf_mask = seg_dir / 'POSTERIOR_01.nii.gz'

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
                gm_mask=gm_mask if gm_mask.exists() else None,
                wm_mask=wm_mask if wm_mask.exists() else None,
                csf_mask=csf_mask if csf_mask.exists() else None,
                dicom_dir=dicom_asl_dir
            )

            logger.info("    ✓ ASL preprocessing complete")
            return results

        except Exception as e:
            logger.error(f"    ✗ ASL preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _run_parallel_workflows(self, modalities: List[str]):
        """Run multiple workflows in parallel."""
        workflow_funcs = {
            'dwi': self._run_dwi,
            'func': self._run_functional,
            'asl': self._run_asl
        }

        # Determine max workers based on config
        max_workers = min(len(modalities), self.config.get('execution', {}).get('n_procs', 4))

        logger.info(f"  Running {len(modalities)} workflows with {max_workers} parallel workers")
        logger.info("")

        # Execute workflows in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_modality = {
                executor.submit(workflow_funcs[mod]): mod
                for mod in modalities if mod in workflow_funcs
            }

            for future in concurrent.futures.as_completed(future_to_modality):
                modality = future_to_modality[future]
                try:
                    result = future.result()
                    self.results[modality] = result
                except Exception as e:
                    logger.error(f"  ✗ {modality} workflow failed: {e}")

    def _find_file(self, modality: str, pattern: str) -> Optional[Path]:
        """Find a file in the NIfTI directory."""
        modality_dir = self.nifti_dir / modality
        if not modality_dir.exists():
            return None

        files = list(modality_dir.glob(pattern))
        return files[0] if files else None

    def _print_summary(self):
        """Print pipeline execution summary."""
        logger.info("="*70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*70)
        logger.info("")

        # DICOM conversion
        if self.results['dicom_conversion']:
            logger.info("DICOM Conversion: ✓")
        elif self.dicom_dir:
            logger.info("DICOM Conversion: ✗")
        else:
            logger.info("DICOM Conversion: - (skipped)")

        # Workflows
        for modality in ['anatomical', 'dwi', 'functional', 'asl']:
            if modality == 'anatomical':
                mod_name = 'anat'
            elif modality == 'functional':
                mod_name = 'func'
            else:
                mod_name = modality

            if mod_name not in self.modalities:
                continue

            result = self.results.get(modality)
            if result:
                logger.info(f"{modality.capitalize()}: ✓")
            elif result is None and mod_name in self.modalities:
                logger.info(f"{modality.capitalize()}: ✗")
            else:
                logger.info(f"{modality.capitalize()}: - (not requested)")

        logger.info("")
        logger.info(f"Results saved to: {self.derivatives_dir}")
        logger.info(f"Working files in: {self.work_dir}")
        logger.info("")
        logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run complete MRI preprocessing pipeline',
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
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Path to configuration file (default: config.yaml)'
    )

    # Input type (DICOM or NIfTI)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--dicom-dir',
        type=Path,
        help='Path to subject DICOM directory (for starting from DICOM)'
    )
    input_group.add_argument(
        '--nifti-dir',
        type=Path,
        help='Path to subject NIfTI directory (for starting from NIfTI)'
    )

    # Optional arguments
    parser.add_argument(
        '--study-root',
        type=Path,
        help='Study root directory (default: inferred from input directory)'
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
    # Priority: 1) --study-root argument, 2) config project_dir, 3) infer from paths
    if args.study_root:
        study_root = args.study_root
    elif 'project_dir' in config:
        # Use project_dir from config (most reliable)
        study_root = Path(config['project_dir'])
        logger.info(f"Using study root from config: {study_root}")
    elif args.dicom_dir:
        # Fallback: Assume study root is 2 levels up from subject DICOM dir
        study_root = args.dicom_dir.parent.parent
        logger.warning(f"Inferring study root from DICOM path: {study_root}")
    else:
        # Fallback: Assume study root is 1 level up from subject NIfTI dir
        study_root = args.nifti_dir.parent
        logger.warning(f"Inferring study root from NIfTI path: {study_root}")

    # Override study root in config if provided via argument
    if args.study_root:
        config['project_dir'] = str(study_root)

    # Create and run orchestrator
    orchestrator = PipelineOrchestrator(
        subject=args.subject,
        config=config,
        study_root=study_root,
        dicom_dir=args.dicom_dir,
        nifti_dir=args.nifti_dir,
        modalities=args.modalities
    )

    results = orchestrator.run()

    # Exit with appropriate code
    all_success = all(
        results.get(mod) is not None
        for mod in ['anatomical', 'dwi', 'functional', 'asl']
        if mod in (args.modalities or ['anat', 'dwi', 'func', 'asl'])
    )

    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    main()
