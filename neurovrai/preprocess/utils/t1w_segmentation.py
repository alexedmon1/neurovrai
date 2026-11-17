"""T1w tissue segmentation: method choice, FSL FAST, naming and provenance.

``anatomical.segmentation_method`` chooses the method:

- ``ants`` (default): Atropos. Started from K-means it writes POSTERIOR_01..03 in arbitrary order,
  so they are named afterwards by mean T1w intensity (``standardize_atropos_tissues``). K-means can
  also settle on a tiny bright class (orbit, vessels) and merge GM into WM: IRC805 lost GM in 5 of 23.
- ``fsl`` (or ``fast``): FSL FAST, hidden Markov random field with its own bias-field estimation.
  It writes *_pve_0/1/2 in a fixed order (CSF, GM, WM), linked here as csf/gm/wm.nii.gz.

Either way, downstream code reads tissue maps by name (neurovrai/utils/tissue_maps.py). When the
method changes, the previous outputs are moved into ``segmentation/superseded_<method>/`` (never
deleted), so a stale gm.nii.gz cannot be picked up. ``segmentation/segmentation.json`` records the
method, its settings, the input and the tool version.

Config::

    anatomical:
      segmentation_method: fsl      # ants (Atropos) or fsl (FAST)
      fast:
        img_type: 1                 # 1 = T1-weighted
        number_classes: 3
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TISSUES = ('csf', 'gm', 'wm')
FAST_INDEX = {'csf': 0, 'gm': 1, 'wm': 2}
_ALIASES = {'ants': 'ants', 'atropos': 'ants', 'fsl': 'fsl', 'fast': 'fsl'}


def segmentation_method(config: Dict[str, Any]) -> str:
    """'ants' (Atropos) or 'fsl' (FAST) from ``anatomical.segmentation_method`` (default 'ants')."""
    raw = ((config or {}).get('anatomical', {}) or {}).get('segmentation_method', 'ants')
    method = _ALIASES.get(str(raw).strip().lower())
    if method is None:
        raise ValueError(f"anatomical.segmentation_method {raw!r}: expected 'ants' or 'fsl'")
    return method


def fast_settings(config: Dict[str, Any]) -> Dict[str, int]:
    fast = ((config or {}).get('anatomical', {}) or {}).get('fast', {}) or {}
    return {'img_type': int(fast.get('img_type', 1)),
            'number_classes': int(fast.get('number_classes', 3))}


def create_fast_node(config: Dict[str, Any], name: str = 'segmentation_fast'):
    """Nipype FSL FAST node (``fast -t <img_type> -n <number_classes>``) for the T1w brain."""
    from nipype import Node
    from nipype.interfaces import fsl

    s = fast_settings(config)
    node = Node(fsl.FAST(), name=name)
    node.inputs.img_type = s['img_type']
    node.inputs.number_classes = s['number_classes']
    node.inputs.output_type = 'NIFTI_GZ'
    return node


def detect_segmentation_method(seg_dir: Path) -> Optional[str]:
    """Which method wrote the outputs at the top of ``seg_dir`` ('ants', 'fsl' or None)."""
    seg_dir = Path(seg_dir)
    rec = seg_dir / 'segmentation.json'
    if rec.exists():
        try:
            return json.loads(rec.read_text()).get('method')
        except ValueError:
            pass
    if any(seg_dir.glob('POSTERIOR_*.nii.gz')):
        return 'ants'
    if any(seg_dir.glob('*pve_*.nii.gz')):
        return 'fsl'
    return None


def supersede_segmentation(seg_dir: Path, reason: str) -> Optional[Path]:
    """Move the files at the top of ``seg_dir`` into ``superseded_<method>[_<time>]/``.

    Returns the folder they went to, or None if there was nothing to move.
    """
    seg_dir = Path(seg_dir)
    files = [p for p in seg_dir.glob('*') if p.is_file() or p.is_symlink()] if seg_dir.is_dir() else []
    if not files:
        return None
    method = detect_segmentation_method(seg_dir) or 'unknown'
    dest = seg_dir / f'superseded_{method}'
    if dest.exists():
        dest = seg_dir / f"superseded_{method}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    dest.mkdir()
    for p in files:
        shutil.move(str(p), str(dest / p.name))
    (dest / 'SUPERSEDED.txt').write_text(f"{datetime.now().isoformat(timespec='seconds')}: {reason}\n")
    logger.info(f"  Previous segmentation ({method}) moved to {dest}")
    return dest


def prepare_segmentation_dir(seg_dir: Path, method: str) -> Optional[Path]:
    """Before a workflow run: supersede outputs another method wrote; keep same-method outputs."""
    seg_dir = Path(seg_dir)
    previous = detect_segmentation_method(seg_dir) if seg_dir.is_dir() else None
    if previous is not None and previous != method:
        return supersede_segmentation(seg_dir, f"segmentation_method changed {previous} -> {method}")
    return None


def name_fast_tissues(seg_dir: Path) -> Dict[str, Path]:
    """Link csf/gm/wm.nii.gz to FAST's *_pve_0/1/2.nii.gz (fixed order). Returns the named maps."""
    seg_dir = Path(seg_dir)
    named = {}
    for tissue, idx in FAST_INDEX.items():
        pve = sorted(seg_dir.glob(f'*pve_{idx}.nii.gz'))
        if len(pve) != 1:
            raise FileNotFoundError(f"expected one *pve_{idx}.nii.gz in {seg_dir}, found {len(pve)}")
        target = seg_dir / f'{tissue}.nii.gz'
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(pve[0].name)
        named[tissue] = target
    return named


def unlink_named_tissues(seg_dir: Path) -> None:
    """Remove csf/gm/wm.nii.gz links so they are remade from this run's outputs."""
    for tissue in TISSUES:
        p = Path(seg_dir) / f'{tissue}.nii.gz'
        if p.is_symlink():
            p.unlink()


def write_segmentation_record(seg_dir: Path, method: str, config: Dict[str, Any],
                              input_image: Optional[Path] = None) -> Path:
    """Write ``segmentation.json``: method, settings, input, tool version, which file is which."""
    seg_dir = Path(seg_dir)
    rec: Dict[str, Any] = {
        'method': method,
        'tool': 'FSL FAST' if method == 'fsl' else 'ANTs Atropos',
        'version': _tool_version(method),
        'input': str(input_image) if input_image else None,
        'written': datetime.now().isoformat(timespec='seconds'),
    }
    if method == 'fsl':
        rec['settings'] = fast_settings(config)
    else:
        rec['settings'] = dict(((config or {}).get('anatomical', {}) or {}).get('atropos', {}) or {})
    rec['tissues'] = {t: (Path(seg_dir) / f'{t}.nii.gz').resolve().name
                      for t in TISSUES if (Path(seg_dir) / f'{t}.nii.gz').exists()}
    out = seg_dir / 'segmentation.json'
    out.write_text(json.dumps(rec, indent=2))
    return out


def _tool_version(method: str) -> Optional[str]:
    try:
        if method == 'fsl':
            import os
            fsldir = os.environ.get('FSLDIR')
            if fsldir and (Path(fsldir) / 'etc' / 'fslversion').exists():
                return 'FSL ' + (Path(fsldir) / 'etc' / 'fslversion').read_text().split(':')[0].strip()
            return None
        out = subprocess.run(['antsRegistration', '--version'], capture_output=True, text=True, timeout=30)
        return out.stdout.splitlines()[0].strip() if out.stdout else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
