"""Find tissue probability maps by tissue name, never by Atropos output position.

Atropos started from K-means writes POSTERIOR_01..03 in whatever order the clustering settles on, so
POSTERIOR_03 is not reliably white matter. ``standardize_atropos_tissues`` (t1w_preprocess.py) names
them csf/gm/wm.nii.gz; FSL FAST writes *_pve_0/1/2 in a fixed order (CSF, GM, WM). Those are the only
two sources read here.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

TISSUES = ('csf', 'gm', 'wm')
FAST_INDEX = {'csf': 0, 'gm': 1, 'wm': 2}


def find_tissue_map(seg_dir: Path, tissue: str) -> Optional[Path]:
    """Return the probability map for ``tissue`` ('csf', 'gm' or 'wm') in ``seg_dir``, or None.

    Looks for ``<tissue>.nii.gz`` first, then FSL FAST's ``*pve_<n>.nii.gz``. Unnamed Atropos
    posteriors are never used: if only those exist, a warning says to name them first.
    """
    tissue = tissue.lower()
    if tissue not in FAST_INDEX:
        raise ValueError(f"Unknown tissue {tissue!r}; expected one of {TISSUES}")
    seg_dir = Path(seg_dir)
    if not seg_dir.is_dir():
        return None

    named = seg_dir / f'{tissue}.nii.gz'
    if named.exists():
        return named

    fast = sorted(seg_dir.glob(f'*pve_{FAST_INDEX[tissue]}.nii.gz'))
    if fast:
        return fast[0]

    if any(seg_dir.glob('POSTERIOR_*.nii.gz')):
        logger.warning(
            f"{seg_dir} has only unnamed Atropos posteriors; their order is arbitrary, so no "
            f"{tissue.upper()} map is taken from them. Run standardize_atropos_tissues first."
        )
    return None


def find_tissue_maps(seg_dir: Path) -> Dict[str, Optional[Path]]:
    """Return {'csf': path, 'gm': path, 'wm': path} for ``seg_dir`` (None where not found)."""
    return {tissue: find_tissue_map(seg_dir, tissue) for tissue in TISSUES}
