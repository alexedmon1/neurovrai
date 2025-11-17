#!/usr/bin/env python3
"""
Configuration loader for MRI preprocessing pipeline.

Handles:
- Loading YAML configuration files
- Merging study configs with defaults
- Environment variable substitution
- Configuration validation
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters."""
    pass


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    file_path : Path
        Path to YAML file

    Returns
    -------
    dict
        Loaded configuration

    Raises
    ------
    ConfigurationError
        If file doesn't exist or YAML is invalid
    """
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {file_path}: {e}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Parameters
    ----------
    base : dict
        Base configuration (defaults)
    override : dict
        Override configuration (study-specific)

    Returns
    -------
    dict
        Merged configuration (override takes precedence)
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def substitute_variables(config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Substitute environment variables and config references in strings.

    Supports:
    - ${ENV_VAR} - environment variables
    - ${config.key.subkey} - references to other config values

    Parameters
    ----------
    config : dict
        Configuration dictionary
    context : dict, optional
        Context for variable substitution (defaults to config itself)

    Returns
    -------
    dict
        Configuration with substituted values
    """
    if context is None:
        context = config

    def substitute_string(value: str) -> str:
        """Substitute variables in a single string."""
        # Pattern: ${VAR} or ${config.key}
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_path = match.group(1)

            # Try environment variable first
            if var_path in os.environ:
                return os.environ[var_path]

            # Try config reference (e.g., ${study.base_dir})
            try:
                parts = var_path.split('.')
                val = context
                for part in parts:
                    val = val[part]
                return str(val)
            except (KeyError, TypeError):
                # Variable not found - leave as is
                return match.group(0)

        return re.sub(pattern, replacer, value)

    def process_value(value: Any) -> Any:
        """Recursively process values."""
        if isinstance(value, str):
            return substitute_string(value)
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(item) for item in value]
        else:
            return value

    return process_value(config)


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize configuration to standard format.

    Handles two formats:
    1. Study config (config_generator.py format):
       - study: {name, code, base_dir}
       - paths: {rawdata, derivatives, ...}
       - sequence_mappings: {...}

    2. Preprocessing params (current format):
       - project_dir: /path/to/study
       - rawdata_dir, derivatives_dir, work_dir
       - anatomical, diffusion, functional params

    Converts study config to preprocessing params format for consistency.

    Parameters
    ----------
    config : dict
        Configuration to normalize

    Returns
    -------
    dict
        Normalized configuration
    """
    # If already in preprocessing params format, return as-is
    if 'project_dir' in config:
        return config

    # Convert study config format to preprocessing params format
    if 'study' in config and 'base_dir' in config['study']:
        normalized = config.copy()

        # Map study.base_dir to project_dir
        normalized['project_dir'] = config['study']['base_dir']

        # Map paths if they exist
        if 'paths' in config:
            paths = config['paths']
            if 'rawdata' in paths:
                normalized['rawdata_dir'] = paths['rawdata']
            if 'derivatives' in paths:
                normalized['derivatives_dir'] = paths['derivatives']
            if 'work' in paths:
                normalized['work_dir'] = paths['work']

        # Substitute variables after mapping
        normalized = substitute_variables(normalized)

        return normalized

    # If neither format is detected, return as-is
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration has all required parameters.

    Parameters
    ----------
    config : dict
        Configuration to validate

    Raises
    ------
    ConfigurationError
        If required parameters are missing or invalid
    """
    # Check if this is a study config (config_generator.py format)
    if 'study' in config:
        # Required study fields
        required_study_fields = ['name', 'code', 'base_dir']
        for field in required_study_fields:
            if field not in config.get('study', {}):
                raise ConfigurationError(f"Missing required study field: study.{field}")

        # Required paths
        required_paths = ['rawdata', 'derivatives']
        for path in required_paths:
            if path not in config.get('paths', {}):
                raise ConfigurationError(f"Missing required path: paths.{path}")

        # At least one modality must be specified
        if 'sequence_mappings' not in config:
            raise ConfigurationError("Missing sequence_mappings section")

        if not config['sequence_mappings']:
            raise ConfigurationError("sequence_mappings is empty - specify at least one modality")

    # Check if this is preprocessing params config (current format)
    elif 'project_dir' in config:
        # Verify project_dir exists
        project_dir = Path(config['project_dir'])
        if not project_dir.exists():
            print(f"Warning: project_dir does not exist: {project_dir}")

        # At least one modality config should be present
        modalities = ['anatomical', 'diffusion', 'functional']
        has_modality = any(mod in config for mod in modalities)
        if not has_modality:
            print("Warning: No modality configurations found (anatomical, diffusion, functional)")

    else:
        # Unknown format - this is okay for default.yaml
        return

    # Check for common mistakes
    if 'fsl' in config:
        fsldir = config['fsl'].get('fsldir')
        if fsldir and not fsldir.startswith('${') and not Path(fsldir).exists():
            print(f"Warning: FSLDIR not found: {fsldir}")

    print("✓ Configuration validation passed")


def load_config(config_path: Path, validate: bool = True) -> Dict[str, Any]:
    """
    Load and process configuration file.

    This is the main entry point for loading configs. It:
    1. Loads the study config
    2. Loads and merges default config
    3. Substitutes variables
    4. Normalizes to standard format
    5. Validates the result

    Parameters
    ----------
    config_path : Path
        Path to study-specific configuration file
    validate : bool
        Whether to validate the configuration

    Returns
    -------
    dict
        Processed configuration

    Raises
    ------
    ConfigurationError
        If configuration is invalid
    """
    config_path = Path(config_path)

    # Load study config
    study_config = load_yaml(config_path)

    # Find and load default config
    # Look for default.yaml in same directory as study config
    default_path = config_path.parent / 'default.yaml'
    if not default_path.exists():
        # Try relative to package
        package_dir = Path(__file__).parent.parent
        default_path = package_dir / 'configs' / 'default.yaml'

    if default_path.exists():
        default_config = load_yaml(default_path)
        # Merge: defaults + study overrides
        config = merge_configs(default_config, study_config)
    else:
        # No default config found - use study config only
        config = study_config

    # Substitute variables
    config = substitute_variables(config)

    # Normalize to standard format (handles both config formats)
    config = normalize_config(config)

    # Validate
    if validate:
        validate_config(config)

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from config using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    key_path : str
        Dot-separated path (e.g., 'anatomical.skull_strip.frac')
    default : any
        Default value if key not found

    Returns
    -------
    any
        Value at key_path, or default if not found

    Examples
    --------
    >>> get_config_value(config, 'anatomical.skull_strip.frac')
    0.5
    >>> get_config_value(config, 'missing.key', default=0.3)
    0.3
    """
    try:
        parts = key_path.split('.')
        value = config
        for part in parts:
            value = value[part]
        return value
    except (KeyError, TypeError):
        return default
