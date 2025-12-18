#!/usr/bin/env python3
"""
Batch ReHo and fALFF Analysis with MNI Normalization

Workflow:
1. Compute ReHo and fALFF in native space for each subject
2. Normalize z-scored maps to MNI152 using func->t1w->MNI transforms
3. Merge all subjects' normalized maps into 4D files
4. Run FSL randomise for group statistics

Uses ANTs transforms from functional preprocessing:
- func_to_t1w0GenericAffine.mat (func -> T1w)
- ants_Composite.h5 (T1w -> MNI)
"""

import logging
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to file and console"""
    logger = logging.getLogger('batch_reho_falff')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def compute_subject_reho_falff(
    subject: str,
    func_file: Path,
    mask_file: Path,
    output_dir: Path,
    tr: float = 1.029,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Compute ReHo and fALFF for a single subject

    Args:
        subject: Subject ID
        func_file: Preprocessed 4D functional file
        mask_file: Brain mask
        output_dir: Subject output directory
        tr: Repetition time
        logger: Logger instance

    Returns:
        Dictionary with paths to output files
    """
    from neurovrai.analysis.func.reho import compute_reho_map, compute_reho_zscore
    from neurovrai.analysis.func.falff import compute_falff_map, compute_falff_zscore

    if logger:
        logger.info(f"Processing {subject}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {'subject': subject, 'status': 'success'}

    # ReHo
    reho_dir = output_dir / 'reho'
    reho_dir.mkdir(exist_ok=True)
    reho_file = reho_dir / 'reho.nii.gz'
    reho_zscore_file = reho_dir / 'reho_zscore.nii.gz'

    if not reho_zscore_file.exists():
        compute_reho_map(func_file, mask_file, neighborhood=27, output_file=reho_file)
        compute_reho_zscore(reho_file, mask_file, output_file=reho_zscore_file)
    results['reho_zscore'] = str(reho_zscore_file)

    # fALFF
    falff_dir = output_dir / 'falff'
    falff_dir.mkdir(exist_ok=True)
    falff_zscore_file = falff_dir / 'falff_zscore.nii.gz'

    if not falff_zscore_file.exists():
        compute_falff_map(func_file, mask_file, tr=tr, output_dir=falff_dir)
        compute_falff_zscore(
            falff_dir / 'alff.nii.gz',
            falff_dir / 'falff.nii.gz',
            mask_file,
            output_dir=falff_dir
        )
    results['falff_zscore'] = str(falff_zscore_file)

    if logger:
        logger.info(f"  {subject}: ReHo and fALFF complete")

    return results


def normalize_to_mni_ants(
    input_file: Path,
    output_file: Path,
    func_to_t1w_mat: Path,
    t1w_to_mni_warp: Path,
    reference: Path,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Apply ANTs transforms to normalize functional map to MNI space

    Transform chain: native func -> T1w -> MNI

    Args:
        input_file: Input image in native func space
        output_file: Output image in MNI space
        func_to_t1w_mat: ANTs affine matrix (func -> T1w)
        t1w_to_mni_warp: ANTs composite warp (T1w -> MNI)
        reference: MNI template reference image
        logger: Logger instance

    Returns:
        True if successful
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use antsApplyTransforms with transform chain
    # Order: transforms are applied in reverse order (last listed = first applied)
    cmd = [
        'antsApplyTransforms',
        '-d', '3',
        '-i', str(input_file),
        '-r', str(reference),
        '-o', str(output_file),
        '-n', 'Linear',
        '-t', str(t1w_to_mni_warp),      # Second: T1w -> MNI
        '-t', str(func_to_t1w_mat),       # First: func -> T1w
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"ANTs transform failed: {e.stderr}")
        return False


def merge_subjects_to_4d(
    subject_files: List[Tuple[str, Path]],
    output_file: Path,
    subject_order_file: Path,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Merge subject maps into 4D file using fslmerge

    Args:
        subject_files: List of (subject_id, file_path) tuples
        output_file: Output 4D file
        subject_order_file: File to save subject order
        logger: Logger instance

    Returns:
        True if successful
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Sort by subject ID to ensure consistent order
    subject_files_sorted = sorted(subject_files, key=lambda x: x[0])

    # Save subject order
    with open(subject_order_file, 'w') as f:
        f.write(f"# Subject order for 4D merged file\n")
        f.write(f"# N={len(subject_files_sorted)} subjects\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n")
        for subj, _ in subject_files_sorted:
            f.write(f"{subj}\n")

    # Build fslmerge command
    files = [str(f) for _, f in subject_files_sorted]
    cmd = ['fslmerge', '-t', str(output_file)] + files

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if logger:
            logger.info(f"Merged {len(files)} subjects into {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"fslmerge failed: {e.stderr}")
        return False


def create_group_mask(
    subject_masks: List[Path],
    output_file: Path,
    threshold: float = 0.9,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Create group mask from individual masks

    Args:
        subject_masks: List of individual MNI-space masks
        output_file: Output group mask
        threshold: Proportion of subjects required (default: 90%)
        logger: Logger instance

    Returns:
        True if successful
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load first mask as reference
    ref_img = nib.load(subject_masks[0])
    ref_shape = ref_img.shape

    # Sum all masks
    mask_sum = np.zeros(ref_shape, dtype=np.float32)
    for mask_file in subject_masks:
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
        mask_sum += (mask_data > 0).astype(np.float32)

    # Threshold
    n_subjects = len(subject_masks)
    group_mask = (mask_sum >= (threshold * n_subjects)).astype(np.uint8)

    # Save
    group_mask_img = nib.Nifti1Image(group_mask, ref_img.affine, ref_img.header)
    nib.save(group_mask_img, output_file)

    n_voxels = np.sum(group_mask)
    if logger:
        logger.info(f"Group mask created: {n_voxels} voxels (threshold={threshold})")

    return True


def run_randomise(
    input_4d: Path,
    output_prefix: Path,
    design_mat: Path,
    design_con: Path,
    mask: Path,
    n_perm: int = 5000,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Run FSL randomise for group statistics

    Args:
        input_4d: 4D merged subject data
        output_prefix: Output prefix for randomise
        design_mat: Design matrix file
        design_con: Contrast file
        mask: Group mask
        n_perm: Number of permutations
        logger: Logger instance

    Returns:
        True if successful
    """
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'randomise',
        '-i', str(input_4d),
        '-o', str(output_prefix),
        '-d', str(design_mat),
        '-t', str(design_con),
        '-m', str(mask),
        '-n', str(n_perm),
        '-T',  # TFCE correction
        '-v', '5',  # Verbose
    ]

    if logger:
        logger.info(f"Running randomise with {n_perm} permutations...")
        logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if logger:
            logger.info("Randomise completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Randomise failed: {e.stderr}")
        return False


def run_batch_analysis(
    study_root: Path,
    subjects: List[str],
    design_dir: Path,
    output_dir: Path,
    analysis_type: str = 'reho',  # 'reho' or 'falff'
    tr: float = 1.029,
    n_perm: int = 5000,
    mni_template: Path = None,
    n_jobs: int = 4,
    skip_compute: bool = False,
    skip_normalize: bool = False,
) -> Dict:
    """
    Run complete batch analysis: compute -> normalize -> merge -> randomise

    Args:
        study_root: Study root directory
        subjects: List of subject IDs
        design_dir: Directory containing design.mat and design.con
        output_dir: Output directory for analysis
        analysis_type: 'reho' or 'falff'
        tr: Repetition time in seconds
        n_perm: Number of permutations for randomise
        mni_template: MNI template for normalization
        n_jobs: Number of parallel jobs
        skip_compute: Skip computing ReHo/fALFF (use existing)
        skip_normalize: Skip normalization (use existing)

    Returns:
        Dictionary with analysis results
    """
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"batch_{analysis_type}_{timestamp}.log"
    logger = setup_logging(log_file)

    logger.info("=" * 80)
    logger.info(f"BATCH {analysis_type.upper()} ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Study root: {study_root}")
    logger.info(f"Subjects: {len(subjects)}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Design: {design_dir}")

    # Set MNI template
    if mni_template is None:
        fsldir = Path('/usr/local/fsl')
        mni_template = fsldir / 'data/standard/MNI152_T1_2mm_brain.nii.gz'

    results = {
        'analysis_type': analysis_type,
        'timestamp': timestamp,
        'subjects': subjects,
        'compute': [],
        'normalize': [],
        'merge': None,
        'randomise': None
    }

    derivatives = study_root / 'derivatives'
    native_dir = output_dir / 'native'
    mni_dir = output_dir / 'mni'
    group_dir = output_dir / 'group'

    native_dir.mkdir(parents=True, exist_ok=True)
    mni_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Compute ReHo/fALFF in native space
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Computing ReHo/fALFF in native space")
    logger.info("=" * 80)

    for subject in subjects:
        subj_deriv = derivatives / subject
        func_file = subj_deriv / 'func' / f'{subject}_bold_bandpass_filtered.nii.gz'
        if not func_file.exists():
            logger.warning(f"  {subject}: func file not found, skipping")
            results['compute'].append({'subject': subject, 'status': 'skipped', 'error': 'func file not found'})
            continue

        # Find brain mask (variable naming convention)
        brain_dir = subj_deriv / 'func' / 'brain'
        mask_files = list(brain_dir.glob('*brain_mask.nii.gz'))
        if not mask_files:
            logger.warning(f"  {subject}: mask not found in {brain_dir}, skipping")
            results['compute'].append({'subject': subject, 'status': 'skipped', 'error': 'mask not found'})
            continue
        mask_file = mask_files[0]  # Use first found mask

        subj_output = native_dir / subject

        if skip_compute:
            # Check if output exists
            if analysis_type == 'reho':
                expected = subj_output / 'reho' / 'reho_zscore.nii.gz'
            else:
                expected = subj_output / 'falff' / 'falff_zscore.nii.gz'

            if expected.exists():
                logger.info(f"  {subject}: Using existing {analysis_type}")
                results['compute'].append({'subject': subject, 'status': 'existing'})
                continue

        try:
            compute_result = compute_subject_reho_falff(
                subject=subject,
                func_file=func_file,
                mask_file=mask_file,
                output_dir=subj_output,
                tr=tr,
                logger=logger
            )
            results['compute'].append(compute_result)
        except Exception as e:
            logger.error(f"  {subject}: Computation failed: {e}")
            results['compute'].append({'subject': subject, 'status': 'failed', 'error': str(e)})

    # =========================================================================
    # Step 2: Normalize to MNI space
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Normalizing to MNI space")
    logger.info("=" * 80)

    normalized_files = []
    normalized_masks = []

    for subject in subjects:
        subj_deriv = derivatives / subject

        # Get transforms
        func_to_t1w = subj_deriv / 'func' / 'registration' / 'func_to_t1w0GenericAffine.mat'
        t1w_to_mni = subj_deriv / 'anat' / 'transforms' / 'ants_Composite.h5'

        if not func_to_t1w.exists():
            logger.warning(f"  {subject}: func->t1w transform not found")
            results['normalize'].append({'subject': subject, 'status': 'skipped', 'error': 'func->t1w not found'})
            continue

        if not t1w_to_mni.exists():
            logger.warning(f"  {subject}: t1w->MNI transform not found")
            results['normalize'].append({'subject': subject, 'status': 'skipped', 'error': 't1w->MNI not found'})
            continue

        # Input and output files
        if analysis_type == 'reho':
            input_file = native_dir / subject / 'reho' / 'reho_zscore.nii.gz'
            output_file = mni_dir / f'{subject}_reho_zscore_mni.nii.gz'
        else:
            input_file = native_dir / subject / 'falff' / 'falff_zscore.nii.gz'
            output_file = mni_dir / f'{subject}_falff_zscore_mni.nii.gz'

        if not input_file.exists():
            logger.warning(f"  {subject}: Input file not found: {input_file}")
            results['normalize'].append({'subject': subject, 'status': 'skipped', 'error': 'input not found'})
            continue

        if skip_normalize and output_file.exists():
            logger.info(f"  {subject}: Using existing normalized file")
            normalized_files.append((subject, output_file))
            results['normalize'].append({'subject': subject, 'status': 'existing'})
            continue

        logger.info(f"  {subject}: Normalizing to MNI...")

        success = normalize_to_mni_ants(
            input_file=input_file,
            output_file=output_file,
            func_to_t1w_mat=func_to_t1w,
            t1w_to_mni_warp=t1w_to_mni,
            reference=mni_template,
            logger=logger
        )

        if success:
            normalized_files.append((subject, output_file))
            results['normalize'].append({'subject': subject, 'status': 'success', 'output': str(output_file)})

            # Also normalize the mask for group mask creation
            brain_dir = derivatives / subject / 'func' / 'brain'
            mask_files = list(brain_dir.glob('*brain_mask.nii.gz'))
            mask_file = mask_files[0] if mask_files else None

            if mask_file and mask_file.exists():
                mask_output = mni_dir / f'{subject}_mask_mni.nii.gz'
                normalize_to_mni_ants(
                    input_file=mask_file,
                    output_file=mask_output,
                    func_to_t1w_mat=func_to_t1w,
                    t1w_to_mni_warp=t1w_to_mni,
                    reference=mni_template,
                    logger=None  # Quiet
                )
                if mask_output.exists():
                    normalized_masks.append(mask_output)
        else:
            results['normalize'].append({'subject': subject, 'status': 'failed'})

    # =========================================================================
    # Step 3: Create group mask and merge to 4D
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Creating group mask and merging to 4D")
    logger.info("=" * 80)

    if len(normalized_files) < 2:
        logger.error("Not enough subjects normalized for group analysis")
        return results

    # Create group mask
    group_mask_file = group_dir / 'group_mask.nii.gz'
    if normalized_masks:
        create_group_mask(normalized_masks, group_mask_file, threshold=0.9, logger=logger)
    else:
        # Use MNI brain mask as fallback
        logger.warning("No individual masks, using MNI brain mask")
        mni_mask = Path('/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
        shutil.copy(mni_mask, group_mask_file)

    # Merge to 4D
    merged_4d = group_dir / f'all_{analysis_type}_4D.nii.gz'
    subject_order_file = group_dir / 'subject_order.txt'

    success = merge_subjects_to_4d(
        normalized_files,
        merged_4d,
        subject_order_file,
        logger=logger
    )

    if success:
        results['merge'] = {
            'status': 'success',
            'output_4d': str(merged_4d),
            'subject_order': str(subject_order_file),
            'n_subjects': len(normalized_files)
        }
    else:
        results['merge'] = {'status': 'failed'}
        return results

    # =========================================================================
    # Step 4: Copy design files and run randomise
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Running FSL randomise")
    logger.info("=" * 80)

    # Copy design files
    design_mat = design_dir / 'design.mat'
    design_con = design_dir / 'design.con'

    if not design_mat.exists() or not design_con.exists():
        logger.error(f"Design files not found in {design_dir}")
        return results

    shutil.copy(design_mat, group_dir / 'design.mat')
    shutil.copy(design_con, group_dir / 'design.con')

    # Verify subject order matches design matrix
    with open(design_mat) as f:
        for line in f:
            if line.startswith('/NumPoints'):
                n_design = int(line.split()[1])
                break

    if len(normalized_files) != n_design:
        logger.error(f"Subject count mismatch: {len(normalized_files)} files vs {n_design} in design matrix")
        logger.error("Please verify subject order and design matrix!")
        return results

    # Run randomise
    randomise_dir = group_dir / f'randomise_{n_perm}perm'
    randomise_dir.mkdir(exist_ok=True)

    randomise_prefix = randomise_dir / f'{analysis_type}_randomise'

    success = run_randomise(
        input_4d=merged_4d,
        output_prefix=randomise_prefix,
        design_mat=group_dir / 'design.mat',
        design_con=group_dir / 'design.con',
        mask=group_mask_file,
        n_perm=n_perm,
        logger=logger
    )

    if success:
        results['randomise'] = {
            'status': 'success',
            'output_dir': str(randomise_dir),
            'n_perm': n_perm
        }
    else:
        results['randomise'] = {'status': 'failed'}

    # Save results summary
    summary_file = output_dir / f'{analysis_type}_analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Summary: {summary_file}")
    logger.info(f"Results: {group_dir}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch ReHo/fALFF analysis with MNI normalization"
    )
    parser.add_argument(
        '--study-root',
        type=Path,
        required=True,
        help='Study root directory'
    )
    parser.add_argument(
        '--subjects-file',
        type=Path,
        help='File with list of subjects (one per line)'
    )
    parser.add_argument(
        '--design-dir',
        type=Path,
        required=True,
        help='Directory containing design.mat and design.con'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--analysis',
        type=str,
        choices=['reho', 'falff', 'both'],
        default='both',
        help='Analysis type (default: both)'
    )
    parser.add_argument(
        '--tr',
        type=float,
        default=1.029,
        help='Repetition time in seconds (default: 1.029)'
    )
    parser.add_argument(
        '--n-perm',
        type=int,
        default=5000,
        help='Number of permutations (default: 5000)'
    )
    parser.add_argument(
        '--skip-compute',
        action='store_true',
        help='Skip computing ReHo/fALFF (use existing)'
    )
    parser.add_argument(
        '--skip-normalize',
        action='store_true',
        help='Skip normalization (use existing)'
    )

    args = parser.parse_args()

    # Load subjects
    if args.subjects_file:
        with open(args.subjects_file) as f:
            subjects = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # Default: read from design directory
        subject_order = args.design_dir / 'subject_order.txt'
        if subject_order.exists():
            with open(subject_order) as f:
                subjects = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        else:
            raise ValueError("No subjects specified. Use --subjects-file or ensure design_dir has subject_order.txt")

    print(f"Subjects: {len(subjects)}")

    # Run analysis
    analyses = ['reho', 'falff'] if args.analysis == 'both' else [args.analysis]

    for analysis_type in analyses:
        print(f"\n{'='*80}")
        print(f"Running {analysis_type.upper()} analysis")
        print(f"{'='*80}")

        # Use appropriate design directory
        if args.analysis == 'both':
            design_dir = args.design_dir.parent / f'func_{analysis_type}'
        else:
            design_dir = args.design_dir

        # Output to the standard mriglu_analysis directory structure
        analysis_output = args.output_dir / analysis_type / 'mriglu_analysis'

        results = run_batch_analysis(
            study_root=args.study_root,
            subjects=subjects,
            design_dir=design_dir,
            output_dir=analysis_output,
            analysis_type=analysis_type,
            tr=args.tr,
            n_perm=args.n_perm,
            skip_compute=args.skip_compute,
            skip_normalize=args.skip_normalize,
        )

        print(f"\n{analysis_type.upper()} Results:")
        print(f"  Computed: {sum(1 for r in results['compute'] if r.get('status') == 'success')}/{len(subjects)}")
        print(f"  Normalized: {sum(1 for r in results['normalize'] if r.get('status') in ['success', 'existing'])}/{len(subjects)}")
        if results['merge']:
            print(f"  Merged: {results['merge'].get('status')}")
        if results['randomise']:
            print(f"  Randomise: {results['randomise'].get('status')}")
