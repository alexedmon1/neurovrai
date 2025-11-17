#!/usr/bin/env python3
"""
Configuration Validators for MRI Preprocessing Workflows

This module provides validation functions for each modality workflow,
ensuring all required parameters are present before execution.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Base configuration validator."""

    @staticmethod
    def check_required_keys(
        config: Dict,
        required_keys: List[str],
        section: str = ""
    ) -> Tuple[bool, List[str]]:
        """
        Check if required keys exist in config.

        Args:
            config: Configuration dictionary
            required_keys: List of required key paths (e.g., 'anatomical.bet.frac')
            section: Section name for logging

        Returns:
            Tuple of (is_valid, missing_keys)
        """
        missing_keys = []

        for key_path in required_keys:
            keys = key_path.split('.')
            current = config

            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    missing_keys.append(key_path)
                    break
                current = current[key]

        is_valid = len(missing_keys) == 0
        return is_valid, missing_keys

    @staticmethod
    def check_file_exists(file_path: Optional[Path], param_name: str) -> bool:
        """
        Check if a file exists.

        Args:
            file_path: Path to check
            param_name: Parameter name for logging

        Returns:
            True if exists or None, False otherwise
        """
        if file_path is None:
            return True  # Optional parameter

        if not Path(file_path).exists():
            logger.warning(f"{param_name} not found: {file_path}")
            return False

        return True


class AnatomicalConfigValidator(ConfigValidator):
    """Validator for anatomical preprocessing workflow."""

    REQUIRED_KEYS = [
        'anatomical.bet.frac',
        'templates.mni152_t1_2mm'
    ]

    OPTIONAL_KEYS = [
        'anatomical.registration_method',
        'anatomical.run_qc'
    ]

    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate anatomical preprocessing configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, missing_required, missing_optional)
        """
        logger.info("Validating anatomical preprocessing configuration...")

        # Check required keys
        is_valid, missing_required = cls.check_required_keys(
            config, cls.REQUIRED_KEYS, 'anatomical'
        )

        # Check optional keys
        _, missing_optional = cls.check_required_keys(
            config, cls.OPTIONAL_KEYS, 'anatomical'
        )

        # Check template file exists
        mni_template = Path(config.get('templates', {}).get('mni152_t1_2mm', ''))
        if not cls.check_file_exists(mni_template, 'MNI152 template'):
            missing_required.append('templates.mni152_t1_2mm (file not found)')
            is_valid = False

        # Log results
        if is_valid:
            logger.info("  ✓ Anatomical config valid")
        else:
            logger.error("  ✗ Anatomical config invalid")
            for key in missing_required:
                logger.error(f"    Missing required: {key}")

        if missing_optional:
            logger.warning("  ⚠ Missing optional parameters (will use defaults):")
            for key in missing_optional:
                logger.warning(f"    {key}")

        return is_valid, missing_required, missing_optional


class DWIConfigValidator(ConfigValidator):
    """Validator for DWI preprocessing workflow."""

    REQUIRED_KEYS = [
        'diffusion.denoise_method',
        'diffusion.eddy_config.flm',
        'diffusion.eddy_config.slm'
    ]

    OPTIONAL_KEYS = [
        'diffusion.topup.readout_time',
        'diffusion.eddy_config.use_cuda',
        'diffusion.run_qc'
    ]

    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate DWI preprocessing configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, missing_required, missing_optional)
        """
        logger.info("Validating DWI preprocessing configuration...")

        # Check required keys
        is_valid, missing_required = cls.check_required_keys(
            config, cls.REQUIRED_KEYS, 'diffusion'
        )

        # Check optional keys
        _, missing_optional = cls.check_required_keys(
            config, cls.OPTIONAL_KEYS, 'diffusion'
        )

        # Log results
        if is_valid:
            logger.info("  ✓ DWI config valid")
        else:
            logger.error("  ✗ DWI config invalid")
            for key in missing_required:
                logger.error(f"    Missing required: {key}")

        if missing_optional:
            logger.warning("  ⚠ Missing optional parameters (will use defaults):")
            for key in missing_optional:
                logger.warning(f"    {key}")

        # Check for TOPUP parameters if reverse phase data expected
        if 'topup' not in config.get('diffusion', {}):
            logger.warning("  ⚠ No TOPUP config - reverse phase data will not be used")

        return is_valid, missing_required, missing_optional


class FunctionalConfigValidator(ConfigValidator):
    """Validator for functional MRI preprocessing workflow."""

    REQUIRED_KEYS = [
        'functional.tr',
        'functional.highpass',
        'functional.lowpass'
    ]

    OPTIONAL_KEYS = [
        'functional.te',
        'functional.fwhm',
        'functional.tedana.enabled',
        'functional.aroma.enabled',
        'functional.acompcor.enabled',
        'functional.run_qc'
    ]

    @classmethod
    def validate(cls, config: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate functional preprocessing configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, missing_required, missing_optional)
        """
        logger.info("Validating functional preprocessing configuration...")

        # Check required keys
        is_valid, missing_required = cls.check_required_keys(
            config, cls.REQUIRED_KEYS, 'functional'
        )

        # Check optional keys
        _, missing_optional = cls.check_required_keys(
            config, cls.OPTIONAL_KEYS, 'functional'
        )

        # Check multi-echo specific requirements
        func_config = config.get('functional', {})
        if func_config.get('tedana', {}).get('enabled', False):
            if 'te' not in func_config or not isinstance(func_config['te'], list):
                logger.error("  ✗ TEDANA enabled but no echo times (te) provided")
                missing_required.append('functional.te (list of echo times)')
                is_valid = False
            elif len(func_config['te']) < 2:
                logger.error("  ✗ TEDANA requires at least 2 echo times")
                missing_required.append('functional.te (need ≥2 echoes)')
                is_valid = False

        # Log results
        if is_valid:
            logger.info("  ✓ Functional config valid")
        else:
            logger.error("  ✗ Functional config invalid")
            for key in missing_required:
                logger.error(f"    Missing required: {key}")

        if missing_optional:
            logger.warning("  ⚠ Missing optional parameters (will use defaults):")
            for key in missing_optional:
                logger.warning(f"    {key}")

        return is_valid, missing_required, missing_optional


class ASLConfigValidator(ConfigValidator):
    """Validator for ASL preprocessing workflow."""

    REQUIRED_KEYS = [
        'asl.labeling_type',
        'asl.labeling_efficiency',
        'asl.t1_blood',
        'asl.blood_brain_partition'
    ]

    OPTIONAL_KEYS = [
        'asl.labeling_duration',
        'asl.post_labeling_delay',
        'asl.label_control_order',
        'asl.apply_m0_calibration',
        'asl.apply_pvc',
        'asl.normalize_to_mni',
        'asl.run_qc'
    ]

    @classmethod
    def validate(cls, config: Dict, has_dicom: bool = False) -> Tuple[bool, List[str], List[str]]:
        """
        Validate ASL preprocessing configuration.

        Args:
            config: Configuration dictionary
            has_dicom: Whether DICOM directory is available for parameter extraction

        Returns:
            Tuple of (is_valid, missing_required, missing_optional)
        """
        logger.info("Validating ASL preprocessing configuration...")

        # Check required keys
        is_valid, missing_required = cls.check_required_keys(
            config, cls.REQUIRED_KEYS, 'asl'
        )

        # Check optional keys
        _, missing_optional = cls.check_required_keys(
            config, cls.OPTIONAL_KEYS, 'asl'
        )

        # Check for acquisition parameters (required if no DICOM)
        if not has_dicom:
            acq_params = ['asl.labeling_duration', 'asl.post_labeling_delay']
            for param in acq_params:
                if param not in [k for k in cls.OPTIONAL_KEYS]:
                    continue
                keys = param.split('.')
                current = config
                for key in keys:
                    if key not in current:
                        logger.error(f"  ✗ {param} required when no DICOM available")
                        missing_required.append(param)
                        is_valid = False
                        break
                    current = current[key]

        # Log results
        if is_valid:
            logger.info("  ✓ ASL config valid")
            if has_dicom:
                logger.info("    (acquisition parameters will be extracted from DICOM)")
        else:
            logger.error("  ✗ ASL config invalid")
            for key in missing_required:
                logger.error(f"    Missing required: {key}")

        if missing_optional:
            logger.warning("  ⚠ Missing optional parameters (will use defaults):")
            for key in missing_optional:
                logger.warning(f"    {key}")

        return is_valid, missing_required, missing_optional


def validate_all_workflows(config: Dict, has_dicom: bool = False) -> Dict[str, bool]:
    """
    Validate configuration for all workflows.

    Args:
        config: Configuration dictionary
        has_dicom: Whether DICOM is available for parameter extraction

    Returns:
        Dictionary mapping workflow name to validation status
    """
    logger.info("="*70)
    logger.info("VALIDATING CONFIGURATION FOR ALL WORKFLOWS")
    logger.info("="*70)
    logger.info("")

    results = {}

    # Validate each workflow
    validators = {
        'anatomical': AnatomicalConfigValidator,
        'dwi': DWIConfigValidator,
        'functional': FunctionalConfigValidator,
        'asl': ASLConfigValidator
    }

    for workflow, validator_class in validators.items():
        if workflow == 'asl':
            is_valid, _, _ = validator_class.validate(config, has_dicom=has_dicom)
        else:
            is_valid, _, _ = validator_class.validate(config)

        results[workflow] = is_valid
        logger.info("")

    # Summary
    logger.info("="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)

    all_valid = all(results.values())
    for workflow, is_valid in results.items():
        status = "✓" if is_valid else "✗"
        logger.info(f"  {status} {workflow}: {'VALID' if is_valid else 'INVALID'}")

    logger.info("")

    if all_valid:
        logger.info("All workflows validated successfully")
    else:
        logger.warning("Some workflows have invalid configurations")

    return results


if __name__ == '__main__':
    # Example usage
    from neurovrai.config import load_config

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Load and validate config
    config = load_config(Path('config.yaml'))
    results = validate_all_workflows(config)

    # Exit with error if any workflow invalid
    import sys
    sys.exit(0 if all(results.values()) else 1)
