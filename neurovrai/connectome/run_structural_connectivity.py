#!/usr/bin/env python3
"""
Structural Connectivity Analysis Runner

Complete pipeline for computing structural connectivity matrices from diffusion MRI data.

This script orchestrates:
1. BEDPOSTX fiber orientation modeling (if not already complete)
2. Atlas transformation to DWI space (MNI, FreeSurfer, or FMRIB58 atlases)
3. Atlas preparation for probtrackx2
4. Probabilistic tractography (probtrackx2 network mode)
5. Connectivity matrix construction
6. Graph metrics computation (optional)

Usage:
    # Using standard atlas (automatically transformed to DWI space)
    python -m neurovrai.connectome.run_structural_connectivity \\
        --subject sub-001 \\
        --derivatives-dir /study/derivatives \\
        --atlas schaefer_200 \\
        --output-dir /study/connectome/structural

    # Using FreeSurfer atlas (requires --fs-subjects-dir)
    python -m neurovrai.connectome.run_structural_connectivity \\
        --subject sub-001 \\
        --derivatives-dir /study/derivatives \\
        --atlas desikan_killiany \\
        --fs-subjects-dir /study/freesurfer \\
        --output-dir /study/connectome/structural

    # With pre-computed atlas in DWI space
    python -m neurovrai.connectome.run_structural_connectivity \\
        --subject sub-001 \\
        --bedpostx-dir /study/derivatives/sub-001/dwi.bedpostX \\
        --atlas-file /study/atlases/schaefer_400_dwi.nii.gz \\
        --output-dir /study/connectome/structural/sub-001

Requirements:
    - Completed DWI preprocessing (eddy correction)
    - FSL installed (probtrackx2, bedpostx)
    - For MNI atlases: anatomical preprocessing with MNI registration
    - For FreeSurfer atlases: completed FreeSurfer recon-all
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add neurovrai to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurovrai.connectome.structural_connectivity import (
    compute_structural_connectivity,
    run_bedpostx,
    validate_bedpostx_outputs,
    StructuralConnectivityError
)
from neurovrai.connectome.graph_metrics import compute_graph_metrics
from neurovrai.connectome.atlas_dwi_transform import (
    ATLAS_CONFIGS,
    prepare_atlas_for_tractography,
)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"structural_connectivity_{timestamp}.log"

    logger = logging.getLogger("structural_connectivity")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def find_atlas_in_dwi_space(
    derivatives_dir: Path,
    subject: str,
    atlas_name: str = 'schaefer_400'
) -> Optional[Path]:
    """
    Find atlas file in subject's DWI space

    Args:
        derivatives_dir: Path to derivatives directory
        subject: Subject ID
        atlas_name: Atlas name (e.g., 'schaefer_400', 'aal')

    Returns:
        Path to atlas file or None if not found
    """
    # Common atlas locations in DWI space
    possible_paths = [
        derivatives_dir / subject / 'dwi' / f'{atlas_name}_dwi.nii.gz',
        derivatives_dir / subject / 'dwi' / 'parcellation' / f'{atlas_name}.nii.gz',
        derivatives_dir / subject / 'parcellation' / f'{atlas_name}_dwi.nii.gz',
        derivatives_dir / 'parcellations' / subject / f'{atlas_name}_dwi.nii.gz',
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def transform_atlas_to_dwi(
    subject: str,
    atlas_name: str,
    derivatives_dir: Path,
    output_dir: Path,
    fs_subjects_dir: Optional[Path] = None,
    logger: logging.Logger = None
) -> tuple:
    """
    Transform atlas from MNI/FreeSurfer space to DWI space

    Args:
        subject: Subject ID
        atlas_name: Atlas name from ATLAS_CONFIGS
        derivatives_dir: Path to derivatives directory
        output_dir: Output directory for transformed atlas
        fs_subjects_dir: FreeSurfer subjects directory (for FS atlases)
        logger: Logger instance

    Returns:
        Tuple of (atlas_dwi_path, roi_names, transform_info)

    Raises:
        StructuralConnectivityError: If transformation fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if atlas_name not in ATLAS_CONFIGS:
        available = list(ATLAS_CONFIGS.keys())
        raise StructuralConnectivityError(
            f"Unknown atlas: {atlas_name}. Available: {available}"
        )

    atlas_config = ATLAS_CONFIGS[atlas_name]
    logger.info(f"Transforming atlas '{atlas_name}' to DWI space...")
    logger.info(f"  Atlas type: {atlas_config.get('source', atlas_config.get('space', 'unknown'))}")

    # Check if FreeSurfer atlas but no FS dir provided
    if atlas_config.get('source') == 'freesurfer' and fs_subjects_dir is None:
        raise StructuralConnectivityError(
            f"Atlas '{atlas_name}' requires FreeSurfer. "
            f"Provide --fs-subjects-dir argument."
        )

    try:
        atlas_dwi_file, roi_names, transform_info = prepare_atlas_for_tractography(
            subject=subject,
            atlas_name=atlas_name,
            derivatives_dir=derivatives_dir,
            output_dir=output_dir,
            fs_subjects_dir=fs_subjects_dir,
        )

        logger.info(f"  Atlas transformed: {atlas_dwi_file.name}")
        logger.info(f"  ROIs: {len(roi_names)}")

        return atlas_dwi_file, roi_names, transform_info

    except Exception as e:
        raise StructuralConnectivityError(f"Atlas transformation failed: {e}")


def prepare_dwi_for_bedpostx(
    derivatives_dir: Path,
    subject: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Prepare DWI data in BEDPOSTX format

    BEDPOSTX expects a directory with:
    - data.nii.gz: Eddy-corrected DWI
    - nodif_brain_mask.nii.gz: Brain mask
    - bvals: b-values
    - bvecs: b-vectors (eddy-rotated)

    Args:
        derivatives_dir: Path to derivatives directory
        subject: Subject ID
        output_dir: Optional output directory (default: creates in dwi/)

    Returns:
        Path to BEDPOSTX input directory

    Raises:
        FileNotFoundError: If required files not found
    """
    dwi_dir = derivatives_dir / subject / 'dwi'

    if not dwi_dir.exists():
        raise FileNotFoundError(f"DWI directory not found: {dwi_dir}")

    # Create BEDPOSTX input directory
    if output_dir is None:
        bedpostx_input_dir = dwi_dir / 'bedpostx_input'
    else:
        bedpostx_input_dir = output_dir

    bedpostx_input_dir.mkdir(parents=True, exist_ok=True)

    # Find source files (multiple possible names from preprocessing)
    dwi_source_files = [
        'eddy_corrected.nii.gz',
        'dwi_eddy_corrected.nii.gz',
        'dwi_preprocessed.nii.gz'
    ]

    mask_source_files = [
        'nodif_brain_mask.nii.gz',
        'b0_brain_mask.nii.gz',
        'dwi_mask.nii.gz'
    ]

    # Find and link/copy files
    import shutil

    # DWI data
    dwi_found = False
    for fname in dwi_source_files:
        source = dwi_dir / fname
        if source.exists():
            dest = bedpostx_input_dir / 'data.nii.gz'
            if not dest.exists():
                shutil.copy2(source, dest)
            dwi_found = True
            logging.info(f"  Linked DWI: {fname}")
            break

    if not dwi_found:
        raise FileNotFoundError(
            f"Eddy-corrected DWI not found in {dwi_dir}. "
            f"Expected one of: {dwi_source_files}"
        )

    # Brain mask
    mask_found = False
    for fname in mask_source_files:
        source = dwi_dir / fname
        if source.exists():
            dest = bedpostx_input_dir / 'nodif_brain_mask.nii.gz'
            if not dest.exists():
                shutil.copy2(source, dest)
            mask_found = True
            logging.info(f"  Linked mask: {fname}")
            break

    if not mask_found:
        raise FileNotFoundError(
            f"Brain mask not found in {dwi_dir}. "
            f"Expected one of: {mask_source_files}"
        )

    # bvals and bvecs
    for fname in ['bvals', 'bvecs']:
        source = dwi_dir / fname
        dest = bedpostx_input_dir / fname

        # Try with eddy rotation suffix
        if not source.exists():
            source = dwi_dir / f'{fname}_eddy'

        if not source.exists():
            raise FileNotFoundError(f"Required file not found: {fname} in {dwi_dir}")

        if not dest.exists():
            shutil.copy2(source, dest)
        logging.info(f"  Linked: {fname}")

    logging.info(f"✓ BEDPOSTX input directory prepared: {bedpostx_input_dir}")

    return bedpostx_input_dir


def run_structural_connectivity_analysis(
    subject: str,
    derivatives_dir: Optional[Path] = None,
    bedpostx_dir: Optional[Path] = None,
    atlas_file: Optional[Path] = None,
    atlas_name: str = 'schaefer_200',
    output_dir: Optional[Path] = None,
    fs_subjects_dir: Optional[Path] = None,
    n_samples: int = 5000,
    run_bedpostx_if_needed: bool = True,
    use_gpu: bool = False,
    compute_graph: bool = True,
    threshold: Optional[float] = None,
    avoid_ventricles: bool = True,
    config: Optional[Dict] = None
) -> Dict:
    """
    Complete structural connectivity analysis workflow

    Args:
        subject: Subject ID
        derivatives_dir: Path to derivatives directory
        bedpostx_dir: Path to existing BEDPOSTX output (skip if provided)
        atlas_file: Path to atlas already in DWI space (skips transformation)
        atlas_name: Atlas name from ATLAS_CONFIGS (default: 'schaefer_200')
        output_dir: Output directory for results
        fs_subjects_dir: FreeSurfer subjects directory (required for FS atlases)
        n_samples: Number of tractography samples (default: 5000)
        run_bedpostx_if_needed: Run BEDPOSTX if not found (default: True)
        use_gpu: Use GPU for BEDPOSTX (default: False)
        compute_graph: Compute graph metrics (default: True)
        threshold: Optional threshold for weak connections
        avoid_ventricles: Exclude ventricles from tractography (default: True)
        config: Optional config dictionary for all tractography settings

    Returns:
        Dictionary with analysis results

    Raises:
        StructuralConnectivityError: If analysis fails
    """
    # Setup logging
    if output_dir is None:
        if derivatives_dir is not None:
            output_dir = derivatives_dir.parent / 'connectome' / 'structural' / subject
        else:
            output_dir = Path('connectome') / 'structural' / subject

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("STRUCTURAL CONNECTIVITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Subject: {subject}")
    if derivatives_dir:
        logger.info(f"Derivatives: {derivatives_dir}")
    if bedpostx_dir:
        logger.info(f"BEDPOSTX directory: {bedpostx_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Tractography samples: {n_samples}")

    # Step 1: Find or run BEDPOSTX
    if bedpostx_dir is None:
        if derivatives_dir is None:
            raise StructuralConnectivityError(
                "Either bedpostx_dir or derivatives_dir must be provided"
            )

        # Check for existing BEDPOSTX output
        dwi_dir = derivatives_dir / subject / 'dwi'
        possible_bedpostx_dirs = [
            dwi_dir / 'bedpostx',  # Standard location after migration
            derivatives_dir / subject / 'dwi.bedpostX',
            derivatives_dir / subject / f'{subject}_dwi.bedpostX',
            dwi_dir / 'bedpostx_input.bedpostX',
        ]

        bedpostx_dir_found = None
        for bpx_dir in possible_bedpostx_dirs:
            if bpx_dir.exists():
                try:
                    validate_bedpostx_outputs(bpx_dir)
                    bedpostx_dir_found = bpx_dir
                    logger.info(f"Found existing BEDPOSTX output: {bpx_dir}")
                    break
                except StructuralConnectivityError:
                    logger.warning(f"Incomplete BEDPOSTX found: {bpx_dir}")
                    continue

        if bedpostx_dir_found is not None:
            bedpostx_dir = bedpostx_dir_found
        elif run_bedpostx_if_needed:
            logger.info("\n[Step 1] No BEDPOSTX output found. Running BEDPOSTX...")

            # Prepare input directory
            bedpostx_input_dir = prepare_dwi_for_bedpostx(
                derivatives_dir=derivatives_dir,
                subject=subject
            )

            # Run BEDPOSTX
            bedpostx_dir = run_bedpostx(
                dwi_dir=bedpostx_input_dir,
                n_fibers=2,
                use_gpu=use_gpu
            )

            logger.info(f"✓ BEDPOSTX complete: {bedpostx_dir}")
        else:
            raise StructuralConnectivityError(
                f"No BEDPOSTX output found and run_bedpostx_if_needed=False. "
                f"Run BEDPOSTX manually or set run_bedpostx_if_needed=True."
            )
    else:
        bedpostx_dir = Path(bedpostx_dir)
        logger.info(f"Using existing BEDPOSTX: {bedpostx_dir}")

    # Step 2: Find or transform atlas to DWI space
    roi_names = None
    if atlas_file is None:
        if derivatives_dir is None:
            raise StructuralConnectivityError(
                "atlas_file must be provided if derivatives_dir is None"
            )

        # First check if atlas already exists in DWI space
        atlas_file = find_atlas_in_dwi_space(
            derivatives_dir=derivatives_dir,
            subject=subject,
            atlas_name=atlas_name
        )

        if atlas_file is not None:
            logger.info(f"Found existing atlas in DWI space: {atlas_file}")
        else:
            # Transform atlas from MNI/FreeSurfer space to DWI space
            logger.info(f"\n[Step 2] Transforming atlas '{atlas_name}' to DWI space...")

            atlas_file, roi_names, transform_info = transform_atlas_to_dwi(
                subject=subject,
                atlas_name=atlas_name,
                derivatives_dir=derivatives_dir,
                output_dir=output_dir / 'atlas',
                fs_subjects_dir=fs_subjects_dir,
                logger=logger
            )

            logger.info(f"✓ Atlas transformed: {atlas_file}")
    else:
        atlas_file = Path(atlas_file)
        if not atlas_file.exists():
            raise StructuralConnectivityError(f"Atlas file not found: {atlas_file}")
        logger.info(f"Using provided atlas file: {atlas_file}")

    # Step 3: Compute structural connectivity
    logger.info("\n[Step 3] Computing structural connectivity...")

    sc_results = compute_structural_connectivity(
        bedpostx_dir=bedpostx_dir,
        atlas_file=atlas_file,
        output_dir=output_dir,
        n_samples=n_samples,
        threshold=threshold,
        avoid_ventricles=avoid_ventricles,
        subject=subject,
        derivatives_dir=derivatives_dir,
        fs_subjects_dir=fs_subjects_dir,
        config=config
    )

    # Step 4: Compute graph metrics (optional)
    if compute_graph:
        logger.info("\n[Step 4] Computing graph metrics...")

        try:
            graph_metrics = compute_graph_metrics(
                connectivity_matrix=sc_results['connectivity_matrix'],
                roi_names=sc_results['roi_names']
            )

            # Save graph metrics
            graph_metrics_file = output_dir / 'graph_metrics.json'
            with open(graph_metrics_file, 'w') as f:
                json.dump(graph_metrics, f, indent=2)

            logger.info(f"✓ Graph metrics saved: {graph_metrics_file}")

            sc_results['graph_metrics'] = graph_metrics

        except Exception as e:
            logger.error(f"Graph metrics computation failed: {e}")
            logger.warning("Continuing without graph metrics")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Subject: {subject}")
    logger.info(f"ROIs: {sc_results['metadata']['n_rois']}")
    logger.info(f"Connections: {sc_results['metadata']['n_connections']}")
    logger.info(f"Density: {sc_results['metadata']['connection_density']:.3f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    return sc_results


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Structural connectivity analysis using probtrackx2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analysis with MNI atlas (auto-transformed to DWI space)
  python -m neurovrai.connectome.run_structural_connectivity \\
      --subject IRC805-0580101 \\
      --derivatives-dir /study/derivatives \\
      --atlas schaefer_200 \\
      --output-dir /study/connectome/structural

  # Using FreeSurfer atlas (requires FS subjects dir)
  python -m neurovrai.connectome.run_structural_connectivity \\
      --subject IRC805-0580101 \\
      --derivatives-dir /study/derivatives \\
      --atlas desikan_killiany \\
      --fs-subjects-dir /study/freesurfer \\
      --output-dir /study/connectome/structural

  # With pre-computed atlas in DWI space
  python -m neurovrai.connectome.run_structural_connectivity \\
      --subject IRC805-0580101 \\
      --bedpostx-dir /study/derivatives/IRC805-0580101/dwi.bedpostX \\
      --atlas-file /study/atlases/schaefer_200_dwi.nii.gz \\
      --output-dir /study/connectome/structural

  # High-resolution tractography with GPU
  python -m neurovrai.connectome.run_structural_connectivity \\
      --subject IRC805-0580101 \\
      --derivatives-dir /study/derivatives \\
      --atlas schaefer_400 \\
      --use-gpu \\
      --n-samples 10000 \\
      --output-dir /study/connectome/structural

Available atlases: """ + ', '.join(ATLAS_CONFIGS.keys()) + """
        """
    )

    parser.add_argument(
        '--subject',
        type=str,
        required=True,
        help='Subject ID'
    )

    parser.add_argument(
        '--derivatives-dir',
        type=Path,
        help='Path to derivatives directory'
    )

    parser.add_argument(
        '--bedpostx-dir',
        type=Path,
        help='Path to existing BEDPOSTX output (skip BEDPOSTX if provided)'
    )

    parser.add_argument(
        '--atlas-file',
        type=Path,
        help='Path to atlas already in DWI space (skips transformation)'
    )

    parser.add_argument(
        '--atlas',
        type=str,
        choices=list(ATLAS_CONFIGS.keys()),
        default='schaefer_200',
        help='Atlas to use (default: schaefer_200)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results'
    )

    parser.add_argument(
        '--fs-subjects-dir',
        type=Path,
        help='FreeSurfer subjects directory (required for FreeSurfer atlases)'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=5000,
        help='Number of tractography samples per voxel (default: 5000)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        help='Threshold for weak connections (0-1, default: None)'
    )

    parser.add_argument(
        '--no-bedpostx',
        action='store_true',
        help='Do not run BEDPOSTX even if not found (will error)'
    )

    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration for BEDPOSTX'
    )

    parser.add_argument(
        '--no-graph-metrics',
        action='store_true',
        help='Skip graph metrics computation'
    )

    parser.add_argument(
        '--no-avoid-ventricles',
        action='store_true',
        help='Disable ventricle avoidance (NOT recommended - ventricles contain CSF, not white matter)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml file for tractography settings'
    )

    args = parser.parse_args()

    # Validation
    if args.bedpostx_dir is None and args.derivatives_dir is None:
        parser.error("Either --bedpostx-dir or --derivatives-dir must be provided")

    # Load config if provided
    config = None
    if args.config:
        if not args.config.exists():
            parser.error(f"Config file not found: {args.config}")
        try:
            import yaml
            with open(args.config) as f:
                config = yaml.safe_load(f)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            parser.error(f"Failed to load config: {e}")

    try:
        results = run_structural_connectivity_analysis(
            subject=args.subject,
            derivatives_dir=args.derivatives_dir,
            bedpostx_dir=args.bedpostx_dir,
            atlas_file=args.atlas_file,
            atlas_name=args.atlas,
            output_dir=args.output_dir,
            fs_subjects_dir=args.fs_subjects_dir,
            n_samples=args.n_samples,
            run_bedpostx_if_needed=not args.no_bedpostx,
            use_gpu=args.use_gpu,
            compute_graph=not args.no_graph_metrics,
            threshold=args.threshold,
            avoid_ventricles=not args.no_avoid_ventricles,
            config=config
        )

        print("\n✓ Analysis complete!")
        print(f"  Output: {results['output_dir']}")
        print(f"  ROIs: {results['metadata']['n_rois']}")
        print(f"  Connections: {results['metadata']['n_connections']}")

        sys.exit(0)

    except StructuralConnectivityError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
