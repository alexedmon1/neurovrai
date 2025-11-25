#!/usr/bin/env python3
"""
Run MELODIC group ICA on IRC805 resting-state data

Uses FSL MELODIC's built-in registration for subjects in native space.
For future subjects with normalize_to_mni=true, data will already be in MNI space.

Usage:
    python scripts/run_melodic_irc805.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from neurovrai.config import load_config
from neurovrai.analysis.func.melodic import run_melodic_group_ica

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Load config
    config = load_config(Path('config.yaml'))

    derivatives_dir = Path(config['derivatives_dir'])
    analysis_dir = Path(config['analysis_dir'])
    melodic_dir = analysis_dir / 'melodic'
    melodic_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("MELODIC GROUP ICA - IRC805 RESTING-STATE DATA")
    logger.info("=" * 80)
    logger.info(f"Derivatives: {derivatives_dir}")
    logger.info(f"Output: {melodic_dir}")
    logger.info("")

    # Collect preprocessed functional data (native space, ACompCor cleaned)
    logger.info("Collecting preprocessed functional data...")
    subject_files = []
    subjects_found = []

    for subject_dir in sorted(derivatives_dir.glob('IRC805-*')):
        subject = subject_dir.name
        func_dir = subject_dir / 'func'

        # Look for ACompCor cleaned data (best quality for ICA)
        func_file = func_dir / f'{subject}_bold_acompcor_cleaned.nii.gz'

        if not func_file.exists():
            # Fall back to regular preprocessed
            func_file = func_dir / f'{subject}_bold_preprocessed.nii.gz'

        if func_file.exists():
            subject_files.append(func_file)
            subjects_found.append(subject)
            logger.info(f"  ✓ {subject}: {func_file.name}")

    logger.info("")
    logger.info(f"Found {len(subject_files)} subjects with preprocessed data")

    if len(subject_files) < 10:
        logger.error("Need at least 10 subjects for MELODIC group ICA")
        logger.error(f"Only found {len(subject_files)} subjects")
        return 1

    # Save subject list
    subject_list_file = melodic_dir / 'subject_list.txt'
    with open(subject_list_file, 'w') as f:
        for subj, file in zip(subjects_found, subject_files):
            f.write(f'{subj}\t{file}\n')

    logger.info(f"Subject list saved to: {subject_list_file}")
    logger.info("")

    # Run MELODIC with built-in registration
    logger.info("=" * 80)
    logger.info("RUNNING MELODIC")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Settings:")
    logger.info(f"  Subjects: {len(subject_files)}")
    logger.info(f"  Components: 20 (fixed)")
    logger.info(f"  Approach: concat (temporal concatenation)")
    logger.info(f"  TR: 1.029 seconds")
    logger.info(f"  Registration: Built-in (subjects are in native space)")
    logger.info("")
    logger.info("This will take approximately 30-60 minutes...")
    logger.info("")

    try:
        results = run_melodic_group_ica(
            subject_files=subject_files,
            output_dir=melodic_dir,
            tr=1.029,  # IRC805 TR
            n_components=20,  # Standard for resting-state
            approach='concat',
            validate_inputs=True,
            sep_vn=True,
            mm_thresh=0.5,
            report=True
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("MELODIC COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Analyzed: {results['n_subjects']} subjects")
        logger.info(f"Components: {results['n_components']}")
        logger.info("")
        logger.info("Key outputs:")
        for name, path in results['outputs'].items():
            logger.info(f"  {name}: {path}")
        logger.info("")
        logger.info(f"Summary: {melodic_dir / 'melodic_summary.json'}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"MELODIC failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
