"""Pre-eddy DWI denoising and Gibbs-ringing removal (dipy-based), as in neurofaune.

Applied to each raw 4-D DWI series *before* merging, TOPUP and eddy, in this order:

  1. MP-PCA denoising (Marchenko-Pastur PCA, ``dipy.denoise.localpca.mppca``): removes thermal
     noise while preserving signal; most useful for the high-b shells feeding DKI / NODDI.
  2. Gibbs-ringing removal (``dipy.denoise.gibbs.gibbs_removal``, the Kellner subvoxel-shift method):
     suppresses truncation ripples at sharp boundaries. In-plane, so slices must lie along axis 2.

Per series, not on the merged data: MP-PCA assumes one noise level across the volumes it
decomposes, and separately acquired series can differ in scaling and noise (Philips scale slopes
differ per series). Each series still needs at least as many voxels in a patch, (2r + 1)^3, as it
has volumes for the Marchenko-Pastur fit to be reliable; a warning is logged otherwise.

Config (same keys and defaults as neurofaune; ``diffusion.denoise_method`` is a label only):

    diffusion:
      denoise:
        enabled: true      # MP-PCA
        patch_radius: 2
      degibbs:
        enabled: true      # Gibbs-ringing removal
        num_processes: 1

dipy is already a dependency, so no external tools (MRtrix) are needed. BLAS is held to one thread
inside both calls: MP-PCA decomposes one small matrix per voxel, and multithreaded OpenBLAS on
those ran more than 5x slower while using every core (IRC805, 112x112x72 volumes: 6 slices took
59 s single-threaded and over 300 s threaded). Run subjects in parallel instead. A record of what was done
(``denoise.json``, with the noise level MP-PCA estimated per series) and the noise maps go to
``derivatives/<subject>/dwi/denoise/``.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def denoise_dwi_mppca(
    in_file: Path,
    out_file: Path,
    patch_radius: int = 2,
    mask_file: Optional[Path] = None,
    sigma_file: Optional[Path] = None,
) -> Path:
    """MP-PCA denoise a 4-D DWI series. Returns ``out_file``.

    If ``sigma_file`` is given, the noise standard deviation MP-PCA estimated is saved there (3-D).
    """
    from dipy.denoise.localpca import mppca
    from threadpoolctl import threadpool_limits

    img = nib.load(str(in_file))
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = None
    if mask_file is not None:
        mask = np.asarray(nib.load(str(mask_file)).dataobj) > 0
    with threadpool_limits(limits=1, user_api='blas'):
        if sigma_file is not None:
            den, sigma = mppca(data, mask=mask, patch_radius=patch_radius, return_sigma=True)
            nib.save(nib.Nifti1Image(np.asarray(sigma, dtype=np.float32), img.affine), str(sigma_file))
        else:
            den = mppca(data, mask=mask, patch_radius=patch_radius)
    den = np.asarray(den, dtype=np.float32)
    np.clip(den, 0, None, out=den)  # denoising can produce small negatives
    nib.save(_float_image(den, img), str(out_file))
    return Path(out_file)


def degibbs_dwi(
    in_file: Path,
    out_file: Path,
    num_processes: int = 1,
) -> Path:
    """Remove Gibbs ringing from a 3-D/4-D image (slices along axis 2). Returns ``out_file``."""
    from dipy.denoise.gibbs import gibbs_removal
    from threadpoolctl import threadpool_limits

    img = nib.load(str(in_file))
    data = np.asarray(img.dataobj, dtype=np.float32)
    with threadpool_limits(limits=1, user_api='blas'):
        dg = gibbs_removal(data, slice_axis=2, inplace=False, num_processes=num_processes)
    dg = np.asarray(dg, dtype=np.float32)
    np.clip(dg, 0, None, out=dg)
    nib.save(_float_image(dg, img), str(out_file))
    return Path(out_file)


def _float_image(data: np.ndarray, like: nib.Nifti1Image) -> nib.Nifti1Image:
    """float32 image with ``like``'s geometry; scaling is already applied to ``data``."""
    hdr = like.header.copy()
    hdr.set_data_dtype(np.float32)
    out = nib.Nifti1Image(data, like.affine, hdr)
    out.header.set_slope_inter(1.0, 0.0)
    return out


def denoise_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """The denoise / degibbs settings from ``config['diffusion']``, with neurofaune's defaults."""
    diff = (config or {}).get('diffusion', {}) or {}
    den = diff.get('denoise', {}) or {}
    dg = diff.get('degibbs', {}) or {}
    return {
        'denoise': bool(den.get('enabled', True)),
        'patch_radius': int(den.get('patch_radius', 2)),
        'degibbs': bool(dg.get('enabled', True)),
        'num_processes': int(dg.get('num_processes', 1)),
    }


def _stem(path: Path) -> str:
    name = Path(path).name
    for ext in ('.nii.gz', '.nii'):
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(path).stem


def denoise_dwi_series(
    dwi_files: List[Path],
    config: Dict[str, Any],
    work_dir: Path,
    record_dir: Optional[Path] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Denoise and de-Gibbs each DWI series on its own, before they are merged.

    Returns the files to carry on with (the inputs unchanged if both steps are disabled) and the
    record, which is also written to ``record_dir/denoise.json`` with each series' noise map.
    Outputs in ``work_dir`` are reused when the record there says they came from the same input
    with the same settings.
    """
    from dipy import __version__ as dipy_version

    s = denoise_settings(config)
    record: Dict[str, Any] = {
        'denoise': {'enabled': s['denoise'], 'method': 'MP-PCA', 'implementation':
                    'dipy.denoise.localpca.mppca', 'patch_radius': s['patch_radius']},
        'degibbs': {'enabled': s['degibbs'], 'method': 'Kellner subvoxel shift', 'implementation':
                    'dipy.denoise.gibbs.gibbs_removal', 'slice_axis': 2},
        'dipy': dipy_version,
        'order': 'per series, before merging, TOPUP and eddy: MP-PCA, then Gibbs removal',
        'series': [],
    }
    if not (s['denoise'] or s['degibbs']):
        logger.info("  DWI denoising and Gibbs removal disabled in config")
        record['series'] = [{'input': str(f), 'output': str(f)} for f in dwi_files]
        return [Path(f) for f in dwi_files], record

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    old = _read_record(work_dir / 'denoise.json')
    settings_key = {k: record[k] for k in ('denoise', 'degibbs', 'dipy')}
    outputs: List[Path] = []
    for i, f in enumerate(dwi_files, 1):
        f = Path(f)
        stem = f"{i:02d}_{_stem(f)}"
        img = nib.load(str(f))
        nvols = img.shape[3] if img.ndim == 4 else 1
        out = work_dir / f"{stem}_{'denoised' if s['denoise'] and not s['degibbs'] else 'preproc'}.nii.gz"
        sigma_file = work_dir / f"{stem}_noise_sigma.nii.gz" if s['denoise'] else None
        entry: Dict[str, Any] = {'input': str(f), 'input_bytes': f.stat().st_size,
                                 'n_volumes': int(nvols), 'output': str(out)}
        prev = _matching_series(old, settings_key, entry)
        if prev is not None and out.exists() and (sigma_file is None or sigma_file.exists()):
            logger.info(f"  Series {i}: reusing {out.name}")
            entry.update({k: prev[k] for k in ('noise_sigma_median',) if k in prev})
        else:
            current = f
            if s['denoise']:
                patch_voxels = (2 * s['patch_radius'] + 1) ** 3
                if patch_voxels < nvols:
                    logger.warning(f"  Series {i}: patch of {patch_voxels} voxels < {nvols} volumes; "
                                   f"MP-PCA works best with patch_radius >= "
                                   f"{_min_radius(nvols)} for this series")
                den = work_dir / f"{stem}_denoised.nii.gz"
                logger.info(f"  Series {i}: MP-PCA denoising {f.name} ({nvols} volumes, "
                            f"patch radius {s['patch_radius']})")
                denoise_dwi_mppca(current, den, patch_radius=s['patch_radius'], sigma_file=sigma_file)
                current = den
            if s['degibbs']:
                logger.info(f"  Series {i}: Gibbs-ringing removal")
                degibbs_dwi(current, out, num_processes=s['num_processes'])
                if current != f and current != out:
                    current.unlink()  # the intermediate; the de-Gibbsed file is what carries on
                current = out
            if sigma_file is not None:
                entry['noise_sigma_median'] = _sigma_median(sigma_file, f)
        outputs.append(out)
        record['series'].append(entry)

    (work_dir / 'denoise.json').write_text(json.dumps(record, indent=2))
    if record_dir is not None:
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / 'denoise.json').write_text(json.dumps(record, indent=2))
        for p in work_dir.glob('*_noise_sigma.nii.gz'):
            shutil.copy2(p, record_dir / p.name)
    return outputs, record


def _min_radius(nvols: int) -> int:
    r = 1
    while (2 * r + 1) ** 3 < nvols:
        r += 1
    return r


def _sigma_median(sigma_file: Path, dwi_file: Path) -> float:
    """Median noise sigma over voxels with signal (mean DWI above 10% of its 99th percentile)."""
    sigma = np.asarray(nib.load(str(sigma_file)).dataobj, dtype=np.float32)
    mean = np.asarray(nib.load(str(dwi_file)).dataobj, dtype=np.float32)
    if mean.ndim == 4:
        mean = mean.mean(axis=3)
    sel = mean > 0.1 * np.percentile(mean, 99)
    return float(np.median(sigma[sel])) if sel.any() else float('nan')


def _read_record(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _matching_series(old: Optional[Dict[str, Any]], settings_key: Dict[str, Any],
                     entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not old or any(old.get(k) != v for k, v in settings_key.items()):
        return None
    for prev in old.get('series', []):
        if all(prev.get(k) == entry[k] for k in ('input', 'input_bytes', 'n_volumes', 'output')):
            return prev
    return None
