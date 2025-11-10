#!/usr/bin/env python3
"""
Command-line interface for MRI preprocessing pipeline.

Usage:
    mri-preprocess config init --dicom-dir /data/DICOM --study-name "My Study" --study-code STUDY01
    mri-preprocess convert --config study.yaml --subject sub-001
    mri-preprocess run anatomical --config study.yaml --subject sub-001
    mri-preprocess run diffusion --config study.yaml --subject sub-001
    mri-preprocess run functional --config study.yaml --subject sub-001
    mri-preprocess run all --config study.yaml --subject sub-001
"""

import click
import logging
from pathlib import Path
from typing import Optional

from mri_preprocess.config import load_config
from mri_preprocess.config_generator import auto_generate_config
from mri_preprocess.dicom.bids_converter import convert_and_organize
from mri_preprocess.workflows.anat_preprocess import run_anat_preprocessing
from mri_preprocess.workflows.dwi_preprocess import run_dwi_preprocessing
from mri_preprocess.utils.file_finder import find_subject_files
from mri_preprocess.utils.bids import get_subject_dir, get_workflow_dir


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='0.1.0')
def main():
    """MRI Preprocessing Pipeline - FSL-based preprocessing for structural and functional MRI."""
    pass


@main.group()
def config():
    """Configuration management commands."""
    pass


@config.command('init')
@click.option('--dicom-dir', type=click.Path(exists=True), required=True,
              help='Directory containing DICOM files')
@click.option('--study-name', required=True, help='Full study name')
@click.option('--study-code', required=True, help='Short study code')
@click.option('--base-dir', type=click.Path(), help='Base directory for study data')
@click.option('--output', '-o', type=click.Path(), help='Output YAML file path')
def config_init(dicom_dir, study_name, study_code, base_dir, output):
    """
    Auto-generate configuration from DICOM directory.
    
    Scans DICOM headers to detect sequences and create a study-specific config file.
    
    Example:
        mri-preprocess config init --dicom-dir /data/DICOM/sub-001 \
                                    --study-name "My Study" \
                                    --study-code STUDY01 \
                                    --output configs/study01.yaml
    """
    click.echo(f"Generating configuration from DICOM directory: {dicom_dir}")
    
    if output is None:
        output = f"configs/{study_code.lower()}.yaml"
    
    try:
        auto_generate_config(
            dicom_dir=Path(dicom_dir),
            study_name=study_name,
            study_code=study_code,
            base_dir=Path(base_dir) if base_dir else None,
            output_path=Path(output)
        )
        click.echo(f"\n✓ Configuration generated: {output}")
        click.echo("\nNext steps:")
        click.echo(f"  1. Review and edit: {output}")
        click.echo(f"  2. Run: mri-preprocess config validate --config {output}")
        
    except Exception as e:
        click.echo(f"Error generating config: {e}", err=True)
        raise click.Abort()


@config.command('validate')
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Configuration file to validate')
def config_validate(config):
    """
    Validate configuration file.
    
    Example:
        mri-preprocess config validate --config configs/study01.yaml
    """
    click.echo(f"Validating configuration: {config}")
    
    try:
        cfg = load_config(Path(config), validate=True)
        click.echo("✓ Configuration is valid")
        click.echo(f"\nStudy: {cfg['study']['name']}")
        click.echo(f"Code: {cfg['study']['code']}")
        click.echo(f"\nModalities configured:")
        for modality in cfg.get('sequence_mappings', {}).keys():
            click.echo(f"  - {modality}")
            
    except Exception as e:
        click.echo(f"✗ Configuration error: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Configuration file')
@click.option('--subject', '-s', required=True, help='Subject ID')
@click.option('--session', help='Session ID (optional)')
@click.option('--dicom-dir', type=click.Path(exists=True),
              help='DICOM directory (overrides config)')
def convert(config, subject, session, dicom_dir):
    """
    Convert DICOM to BIDS-organized NIfTI.
    
    Example:
        mri-preprocess convert --config configs/study01.yaml --subject sub-001
    """
    click.echo(f"Converting DICOM to NIfTI for {subject}")
    
    try:
        cfg = load_config(Path(config))
        
        # Get paths from config
        sourcedata_dir = Path(cfg['paths']['sourcedata'])
        rawdata_dir = Path(cfg['paths']['rawdata'])
        
        if dicom_dir is None:
            dicom_dir = get_subject_dir(sourcedata_dir, subject, session)
        else:
            dicom_dir = Path(dicom_dir)
        
        # Convert and organize
        result = convert_and_organize(
            dicom_dir=dicom_dir,
            rawdata_dir=rawdata_dir,
            subject=subject,
            config=cfg,
            session=session
        )
        
        if result['success']:
            click.echo(f"✓ Conversion successful")
            click.echo(f"\nOrganized files:")
            for modality, files in result['organized_files'].items():
                click.echo(f"  {modality}: {len(files)} files")
        else:
            click.echo(f"✗ Conversion failed", err=True)
            raise click.Abort()
            
    except Exception as e:
        click.echo(f"Error during conversion: {e}", err=True)
        raise click.Abort()


@main.group()
def run():
    """Run preprocessing workflows."""
    pass


@run.command('anatomical')
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Configuration file')
@click.option('--subject', '-s', required=True, help='Subject ID')
@click.option('--session', help='Session ID (optional)')
@click.option('--t1w-file', type=click.Path(exists=True),
              help='T1w file (if not auto-detected)')
def run_anatomical(config, subject, session, t1w_file):
    """
    Run anatomical (T1w) preprocessing.
    
    This workflow:
      - Reorients to standard space
      - Skull strips with BET
      - Bias field correction with FAST
      - Registers to MNI152 (linear + nonlinear)
      - Saves transforms to TransformRegistry
    
    Example:
        mri-preprocess run anatomical --config configs/study01.yaml --subject sub-001
    """
    click.echo(f"Running anatomical preprocessing for {subject}")
    
    try:
        cfg = load_config(Path(config))
        
        # Find T1w file if not specified
        if t1w_file is None:
            rawdata_dir = Path(cfg['paths']['rawdata'])
            subject_dir = get_subject_dir(rawdata_dir, subject, session)
            
            files = find_subject_files(subject_dir, 't1w', cfg, session)
            if not files['images']:
                click.echo(f"✗ No T1w files found for {subject}", err=True)
                raise click.Abort()
            
            t1w_file = files['images'][0]
            click.echo(f"Using T1w file: {t1w_file}")
        else:
            t1w_file = Path(t1w_file)
        
        # Setup paths
        derivatives_dir = Path(cfg['paths']['derivatives'])
        work_dir = get_workflow_dir(
            Path(cfg['paths'].get('work', '/tmp/work')),
            subject,
            'anat-preprocess',
            session
        )
        
        # Run workflow
        results = run_anat_preprocessing(
            config=cfg,
            subject=subject,
            t1w_file=t1w_file,
            output_dir=derivatives_dir,
            work_dir=work_dir,
            session=session
        )
        
        click.echo(f"\n✓ Anatomical preprocessing complete")
        click.echo(f"\nOutputs:")
        for key, path in results.items():
            if path:
                click.echo(f"  {key}: {path}")
        
    except Exception as e:
        click.echo(f"Error during anatomical preprocessing: {e}", err=True)
        raise click.Abort()


@run.command('diffusion')
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Configuration file')
@click.option('--subject', '-s', required=True, help='Subject ID')
@click.option('--session', help='Session ID (optional)')
@click.option('--bedpostx/--no-bedpostx', default=False,
              help='Run BEDPOSTX for tractography (slow)')
def run_diffusion(config, subject, session, bedpostx):
    """
    Run diffusion (DWI) preprocessing.
    
    This workflow:
      - Eddy current and motion correction
      - DTI tensor fitting
      - Optionally runs BEDPOSTX
      - Warps to MNI using transforms from TransformRegistry
    
    Example:
        mri-preprocess run diffusion --config configs/study01.yaml --subject sub-001
    """
    click.echo(f"Running diffusion preprocessing for {subject}")
    
    try:
        cfg = load_config(Path(config))
        
        # Find DWI files
        rawdata_dir = Path(cfg['paths']['rawdata'])
        subject_dir = get_subject_dir(rawdata_dir, subject, session)
        
        files = find_subject_files(subject_dir, 'dwi', cfg, session)
        if not files['images']:
            click.echo(f"✗ No DWI files found for {subject}", err=True)
            raise click.Abort()
        
        dwi_file = files['images'][0]
        bval_file = files['bval'][0] if files['bval'] else None
        bvec_file = files['bvec'][0] if files['bvec'] else None
        
        if not bval_file or not bvec_file:
            click.echo(f"✗ Missing bval/bvec files", err=True)
            raise click.Abort()
        
        click.echo(f"Using DWI file: {dwi_file}")
        
        # Setup paths
        derivatives_dir = Path(cfg['paths']['derivatives'])
        work_dir = get_workflow_dir(
            Path(cfg['paths'].get('work', '/tmp/work')),
            subject,
            'dwi-preprocess',
            session
        )
        
        # Run workflow
        results = run_dwi_preprocessing(
            config=cfg,
            subject=subject,
            dwi_file=dwi_file,
            bval_file=bval_file,
            bvec_file=bvec_file,
            output_dir=derivatives_dir,
            work_dir=work_dir,
            session=session,
            run_bedpostx=bedpostx
        )
        
        click.echo(f"\n✓ Diffusion preprocessing complete")
        click.echo(f"\nOutputs:")
        for key, path in results.items():
            if path:
                click.echo(f"  {key}: {path}")
        
    except Exception as e:
        click.echo(f"Error during diffusion preprocessing: {e}", err=True)
        raise click.Abort()


@run.command('all')
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Configuration file')
@click.option('--subject', '-s', required=True, help='Subject ID')
@click.option('--session', help='Session ID (optional)')
@click.option('--bedpostx/--no-bedpostx', default=False,
              help='Run BEDPOSTX for tractography (slow)')
def run_all(config, subject, session, bedpostx):
    """
    Run complete preprocessing pipeline.
    
    Runs anatomical, diffusion, and functional preprocessing in sequence.
    
    Example:
        mri-preprocess run all --config configs/study01.yaml --subject sub-001
    """
    click.echo(f"Running complete preprocessing pipeline for {subject}")
    
    # Run anatomical first (to compute transforms)
    click.echo("\n" + "="*60)
    click.echo("STEP 1: Anatomical Preprocessing")
    click.echo("="*60)
    ctx = click.get_current_context()
    ctx.invoke(run_anatomical, config=config, subject=subject, session=session)
    
    # Run diffusion (uses anatomical transforms)
    click.echo("\n" + "="*60)
    click.echo("STEP 2: Diffusion Preprocessing")
    click.echo("="*60)
    ctx.invoke(run_diffusion, config=config, subject=subject, session=session, bedpostx=bedpostx)
    
    click.echo("\n" + "="*60)
    click.echo("✓ Complete preprocessing pipeline finished")
    click.echo("="*60)


if __name__ == '__main__':
    main()
