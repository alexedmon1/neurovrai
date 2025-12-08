"""
Atlas label parsing and ROI naming utilities.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# Atlas label file locations
ATLAS_LABEL_FILES = {
    'harvardoxford_cort': '/usr/local/fsl/data/atlases/HarvardOxford-Cortical.xml',
    'harvardoxford_sub': '/usr/local/fsl/data/atlases/HarvardOxford-Subcortical.xml',
    'schaefer100': '/mnt/arborea/atlases/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_100Parcels_17Networks_order_info.txt',
    'schaefer200': '/mnt/arborea/atlases/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_200Parcels_17Networks_order_info.txt',
    'schaefer400': '/mnt/arborea/atlases/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order_info.txt',
}

# Brain region groupings for Harvard-Oxford Cortical
HARVARD_OXFORD_REGIONS = {
    'Frontal': [
        'Frontal Pole',
        'Superior Frontal Gyrus',
        'Middle Frontal Gyrus',
        'Inferior Frontal Gyrus, pars triangularis',
        'Inferior Frontal Gyrus, pars opercularis',
        'Precentral Gyrus',
        'Frontal Medial Cortex',
        'Frontal Orbital Cortex',
        'Frontal Operculum Cortex',
        'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
        'Subcallosal Cortex',
        'Paracingulate Gyrus',
    ],
    'Temporal': [
        'Temporal Pole',
        'Superior Temporal Gyrus, anterior division',
        'Superior Temporal Gyrus, posterior division',
        'Middle Temporal Gyrus, anterior division',
        'Middle Temporal Gyrus, posterior division',
        'Middle Temporal Gyrus, temporooccipital part',
        'Inferior Temporal Gyrus, anterior division',
        'Inferior Temporal Gyrus, posterior division',
        'Inferior Temporal Gyrus, temporooccipital part',
        'Temporal Fusiform Cortex, anterior division',
        'Temporal Fusiform Cortex, posterior division',
        'Temporal Occipital Fusiform Cortex',
        'Parahippocampal Gyrus, anterior division',
        'Parahippocampal Gyrus, posterior division',
        'Planum Polare',
        "Heschl's Gyrus (includes H1 and H2)",
        'Planum Temporale',
    ],
    'Parietal': [
        'Postcentral Gyrus',
        'Superior Parietal Lobule',
        'Supramarginal Gyrus, anterior division',
        'Supramarginal Gyrus, posterior division',
        'Angular Gyrus',
        'Precuneous Cortex',
        'Parietal Operculum Cortex',
    ],
    'Occipital': [
        'Lateral Occipital Cortex, superior division',
        'Lateral Occipital Cortex, inferior division',
        'Intracalcarine Cortex',
        'Cuneal Cortex',
        'Lingual Gyrus',
        'Occipital Fusiform Gyrus',
        'Supracalcarine Cortex',
        'Occipital Pole',
    ],
    'Cingulate': [
        'Cingulate Gyrus, anterior division',
        'Cingulate Gyrus, posterior division',
    ],
    'Insula': [
        'Insular Cortex',
        'Central Opercular Cortex',
    ],
}


def parse_fsl_atlas_xml(xml_file: Path) -> Dict[int, str]:
    """
    Parse FSL atlas XML file to extract ROI labels.

    Parameters
    ----------
    xml_file : Path
        Path to XML label file

    Returns
    -------
    labels : Dict[int, str]
        Dictionary mapping ROI index to label name
    """
    if not xml_file.exists():
        logger.warning(f"Atlas XML file not found: {xml_file}")
        return {}

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        labels = {}
        for label in root.findall('.//label'):
            index = int(label.get('index'))
            name = label.text.strip()
            labels[index] = name

        logger.info(f"Loaded {len(labels)} ROI labels from {xml_file.name}")
        return labels

    except Exception as e:
        logger.error(f"Error parsing XML file {xml_file}: {e}")
        return {}


def parse_schaefer_labels(txt_file: Path) -> Dict[int, str]:
    """
    Parse Schaefer atlas label file to extract ROI labels.

    Format: ROI_NAME\nINDEX R G B ALPHA\n...

    Parameters
    ----------
    txt_file : Path
        Path to Schaefer label file

    Returns
    -------
    labels : Dict[int, str]
        Dictionary mapping ROI index to label name
    """
    if not txt_file.exists():
        logger.warning(f"Schaefer label file not found: {txt_file}")
        return {}

    try:
        labels = {}
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        # Parse pairs of lines (name, then index + colors)
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break

            name_line = lines[i].strip()
            index_line = lines[i + 1].strip()

            if not name_line or not index_line:
                continue

            # Parse index from index_line (format: "INDEX R G B ALPHA")
            parts = index_line.split()
            if len(parts) >= 1:
                index = int(parts[0])
                # Clean up the name (remove "17Networks_" prefix if present)
                clean_name = name_line.replace('17Networks_', '')
                labels[index - 1] = clean_name  # Schaefer uses 1-based indexing

        logger.info(f"Loaded {len(labels)} ROI labels from {txt_file.name}")
        return labels

    except Exception as e:
        logger.error(f"Error parsing Schaefer label file {txt_file}: {e}")
        return {}


def get_atlas_labels(atlas_name: str) -> Optional[Dict[int, str]]:
    """
    Get ROI labels for a given atlas.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas (e.g., 'harvardoxford_cort', 'schaefer200')

    Returns
    -------
    labels : Dict[int, str] or None
        Dictionary mapping ROI index to label name, or None if not available
    """
    if atlas_name not in ATLAS_LABEL_FILES:
        logger.debug(f"No label file configured for atlas: {atlas_name}")
        return None

    label_file = Path(ATLAS_LABEL_FILES[atlas_name])

    # Determine parser based on file extension
    if label_file.suffix == '.xml':
        return parse_fsl_atlas_xml(label_file)
    elif label_file.suffix == '.txt':
        return parse_schaefer_labels(label_file)
    else:
        logger.warning(f"Unknown label file format: {label_file}")
        return None


def get_roi_brain_region(roi_name: str, atlas_name: str = 'harvardoxford_cort') -> str:
    """
    Get the brain region group for a given ROI.

    Parameters
    ----------
    roi_name : str
        Name of the ROI
    atlas_name : str
        Name of the atlas

    Returns
    -------
    region : str
        Brain region name (e.g., 'Frontal', 'Temporal', 'Visual', 'Default')
    """
    if atlas_name == 'harvardoxford_cort':
        for region, roi_list in HARVARD_OXFORD_REGIONS.items():
            if roi_name in roi_list:
                return region
        return 'Other'

    elif atlas_name in ['schaefer100', 'schaefer200', 'schaefer400']:
        # Schaefer ROIs have format: LH_NetworkName_RegionName_#
        # Extract network name from ROI
        if '_VisCent' in roi_name or '_VisPeri' in roi_name:
            return 'Visual'
        elif '_SomMotA' in roi_name or '_SomMotB' in roi_name:
            return 'Somatomotor'
        elif '_DorsAttnA' in roi_name or '_DorsAttnB' in roi_name:
            return 'DorsalAttention'
        elif '_SalVentAttnA' in roi_name or '_SalVentAttnB' in roi_name:
            return 'VentralAttention'
        elif '_LimbicA' in roi_name or '_LimbicB' in roi_name:
            return 'Limbic'
        elif '_ContA' in roi_name or '_ContB' in roi_name or '_ContC' in roi_name:
            return 'Control'
        elif '_DefaultA' in roi_name or '_DefaultB' in roi_name or '_DefaultC' in roi_name:
            return 'DefaultMode'
        elif '_TempPar' in roi_name:
            return 'Temporoparietal'
        else:
            return 'Other'

    return 'Unknown'


def get_roi_labels_for_matrix(n_rois: int, atlas_name: str) -> List[str]:
    """
    Get ROI labels for a connectivity matrix.

    Parameters
    ----------
    n_rois : int
        Number of ROIs in the matrix
    atlas_name : str
        Name of the atlas

    Returns
    -------
    labels : List[str]
        List of ROI labels (generic if labels not available)
    """
    atlas_labels = get_atlas_labels(atlas_name)

    if atlas_labels is None:
        # Return generic labels
        return [f'ROI_{i+1}' for i in range(n_rois)]

    # Map indices to labels
    labels = []
    for i in range(n_rois):
        if i in atlas_labels:
            labels.append(atlas_labels[i])
        else:
            labels.append(f'ROI_{i+1}')

    return labels


def get_roi_order_by_region(roi_labels: List[str], atlas_name: str = 'harvardoxford_cort') -> List[int]:
    """
    Get ROI ordering indices grouped by brain region.

    Parameters
    ----------
    roi_labels : List[str]
        List of ROI label names
    atlas_name : str
        Name of the atlas

    Returns
    -------
    indices : List[int]
        Ordered indices for reordering ROIs by brain region
    """
    # Group ROIs by region
    region_rois = {}

    for idx, roi_name in enumerate(roi_labels):
        region = get_roi_brain_region(roi_name, atlas_name)
        if region not in region_rois:
            region_rois[region] = []
        region_rois[region].append(idx)

    # Define region order based on atlas
    if atlas_name == 'harvardoxford_cort':
        region_order = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Cingulate', 'Insula', 'Other']
    elif atlas_name in ['schaefer100', 'schaefer200', 'schaefer400']:
        # Order by Schaefer networks (functional systems)
        region_order = ['Visual', 'Somatomotor', 'DorsalAttention', 'VentralAttention',
                       'Limbic', 'Control', 'DefaultMode', 'Temporoparietal', 'Other']
    else:
        # Unknown atlas - return original order
        return list(range(len(roi_labels)))

    # Build ordered list
    ordered_indices = []
    for region in region_order:
        if region in region_rois:
            ordered_indices.extend(region_rois[region])

    # Add any regions not in the predefined order
    for region, indices in region_rois.items():
        if region not in region_order:
            ordered_indices.extend(indices)

    return ordered_indices
