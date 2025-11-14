#!/usr/bin/env python3
"""Test functional preprocessing with multi-echo fixes"""

import logging
from pathlib import Path
from mri_preprocess.config import load_config
from mri_preprocess.workflows.func_preprocess import run_func_preprocessing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/func_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load config
config = load_config(Path('config.yaml'))

# Subject and paths
subject = 'IRC805-0580101'
study_root = Path('/mnt/bytopia/IRC805')
bids_dir = study_root / 'bids' / subject / 'func'
derivatives_dir = study_root / 'derivatives'
anat_derivatives = derivatives_dir / subject / 'anat'

# Find all RESTING echo files
func_files = sorted(list(bids_dir.glob('*RESTING*.nii.gz')))

logger.info("=" * 70)
logger.info("FUNCTIONAL PREPROCESSING TEST")
logger.info("=" * 70)
logger.info(f"Subject: {subject}")
logger.info(f"Found {len(func_files)} echo files:")
for f in func_files:
    size_mb = f.stat().st_size / (1024*1024)
    logger.info(f"  - {f.name} ({size_mb:.1f} MB)")
logger.info("")

# Run preprocessing
try:
    results = run_func_preprocessing(
        config=config,
        subject=subject,
        func_file=func_files,
        output_dir=derivatives_dir,
        anat_derivatives=anat_derivatives,
        work_dir=study_root / 'work' / subject
    )

    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)

    for key, value in results.items():
        if value and Path(str(value)).exists():
            logger.info(f"  ✓ {key}: {value}")
        else:
            logger.info(f"  ✗ {key}: {value}")

except Exception as e:
    logger.error(f"Preprocessing failed: {e}")
    import traceback
    logger.error(traceback.format_exc())
    raise
