#!/usr/bin/env python3
"""
Test DICOM to NIfTI Converter

This script tests the unified DICOM converter on a new subject.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mri_preprocess.utils.dicom_converter import convert_subject_dicoms

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_dicom_converter.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def test_dicom_conversion():
    """Test DICOM conversion for IRC805-1580101."""

    subject = 'IRC805-1580101'
    dicom_dir = Path(f'/mnt/bytopia/IRC805/raw/dicom/{subject}')
    output_dir = Path('/mnt/bytopia/IRC805/bids')  # Converter will create {subject}/{modality}/

    logger.info("="*70)
    logger.info("TESTING DICOM TO NIFTI CONVERSION")
    logger.info("="*70)
    logger.info(f"Subject: {subject}")
    logger.info(f"DICOM directory: {dicom_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    if not dicom_dir.exists():
        logger.error(f"DICOM directory not found: {dicom_dir}")
        return False

    try:
        results = convert_subject_dicoms(
            subject=subject,
            dicom_dir=dicom_dir,
            output_dir=output_dir
        )

        logger.info("")
        logger.info("="*70)
        logger.info("CONVERSION TEST RESULTS")
        logger.info("="*70)

        for modality, files in results.items():
            if files:
                logger.info(f"{modality.upper()}:")
                for file in files:
                    logger.info(f"  ✓ {file.name}")
                    # Check JSON sidecar exists
                    json_file = file.with_suffix('').with_suffix('.json')
                    if json_file.exists():
                        logger.info(f"    + {json_file.name}")
                    else:
                        logger.warning(f"    - No JSON sidecar")

        logger.info("")
        logger.info("="*70)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("="*70)

        return True

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == '__main__':
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    success = test_dicom_conversion()
    sys.exit(0 if success else 1)
