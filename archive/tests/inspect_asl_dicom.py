#!/usr/bin/env python3
"""
Detailed DICOM inspector for ASL parameters.

Examines all DICOM tags (standard and private) to identify ASL-specific fields.
"""

import pydicom
from pathlib import Path
import sys

def inspect_dicom(dicom_file: Path):
    """Inspect all DICOM tags in file."""
    print(f"Inspecting: {dicom_file}")
    print("=" * 80)

    dcm = pydicom.dcmread(dicom_file, force=True)

    print("\n### STANDARD DICOM TAGS ###\n")

    # Standard tags of interest
    standard_tags = [
        'SeriesDescription',
        'ProtocolName',
        'RepetitionTime',
        'EchoTime',
        'FlipAngle',
        'SliceThickness',
        'AcquisitionMatrix',
        'NumberOfTemporalPositions',
        'ImagesInAcquisition',
        'MRAcquisitionType',
        'ScanningSequence',
        'SequenceVariant',
        'PulseSequenceName'
    ]

    for tag in standard_tags:
        if hasattr(dcm, tag):
            print(f"{tag:30s}: {getattr(dcm, tag)}")

    print("\n### PHILIPS PRIVATE TAGS (Group 2005) ###\n")

    # Inspect all Philips private tags
    philips_tags = []
    for tag in dcm.keys():
        if tag.group == 0x2005:  # Philips private group
            try:
                element = dcm[tag]
                tag_str = f"({tag.group:04x},{tag.element:04x})"
                vr = element.VR
                value = element.value

                # Try to get a description
                keyword = element.keyword if hasattr(element, 'keyword') else ''

                philips_tags.append((tag_str, vr, keyword, value))
            except Exception as e:
                print(f"Error reading {tag}: {e}")

    # Sort by tag
    philips_tags.sort()

    for tag_str, vr, keyword, value in philips_tags:
        # Truncate long values
        val_str = str(value)
        if len(val_str) > 60:
            val_str = val_str[:57] + '...'

        print(f"{tag_str} {vr:4s} {keyword:30s}: {val_str}")

    print("\n### OTHER PRIVATE TAGS ###\n")

    # Check other private groups
    for tag in dcm.keys():
        if tag.group % 2 == 1 and tag.group != 0x2005:  # Odd groups are private
            try:
                element = dcm[tag]
                tag_str = f"({tag.group:04x},{tag.element:04x})"
                vr = element.VR
                value = element.value

                val_str = str(value)
                if len(val_str) > 60:
                    val_str = val_str[:57] + '...'

                print(f"{tag_str} {vr:4s}: {val_str}")
            except Exception as e:
                pass

    print("\n### ASL-RELATED SEQUENCES ###\n")

    # Check for ASL-specific sequences
    asl_sequences = [
        (0x0018, 0x9005),  # Pulse Sequence Name
        (0x0018, 0x9006),  # MR Acquisition Type
        (0x0018, 0x9008),  # Echo Pulse Sequence
        (0x0018, 0x9012),  # MR Timing and Related Parameters
        (0x0018, 0x9015),  # MR Diffusion Sequence
        (0x0018, 0x9028),  # MR Velocity Encoding Sequence
        (0x0018, 0x9087),  # MR Arterial Spin Labeling Sequence
    ]

    for tag in asl_sequences:
        if tag in dcm:
            print(f"{tag}: {dcm[tag].value}")

    print("\n=" * 80)

    # Try to extract ASL-specific numeric values from Philips tags
    print("\n### POTENTIAL ASL PARAMETERS ###\n")

    for tag_str, vr, keyword, value in philips_tags:
        # Look for numeric values in reasonable ranges for ASL
        if vr in ['DS', 'FL', 'FD', 'IS', 'SL', 'SS', 'UL', 'US']:
            try:
                num_val = float(value)

                # Check if it's in range for ASL parameters
                if 0.5 < num_val < 10:  # Likely τ or PLD (0.5-10 seconds)
                    print(f"{tag_str} {keyword:30s}: {num_val:.3f} (could be τ/PLD)")
                elif 10 < num_val < 5000:  # Likely milliseconds
                    print(f"{tag_str} {keyword:30s}: {num_val:.1f} ms ({num_val/1000:.3f} s)")
                elif 0.5 < num_val < 1.0:  # Likely efficiency
                    print(f"{tag_str} {keyword:30s}: {num_val:.3f} (could be α)")
            except (ValueError, TypeError):
                pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_asl_dicom.py <dicom_file>")
        sys.exit(1)

    dicom_file = Path(sys.argv[1])
    inspect_dicom(dicom_file)
