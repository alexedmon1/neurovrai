"""Select the volumes the diffusion tensor is fitted on.

The tensor model assumes the log signal falls linearly with b. That holds up to about b = 1000 s/mm²;
above it the signal is visibly non-Gaussian (kurtosis), and a tensor fitted across higher shells
absorbs that curvature: MD comes out roughly halved and FA biased. So FA, MD, AD and RD are fitted on
b=0 plus the shells up to ``max_bval``; DKI and NODDI keep every shell.

Config: ``diffusion.dti.max_bval`` (default 1000) and ``diffusion.dti.bval_tolerance`` (default 50,
for nominal shells written as e.g. 995 or 1005).
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

DEFAULT_MAX_BVAL = 1000.0
DEFAULT_TOLERANCE = 50.0
MIN_DIRECTIONS = 6  # a tensor has six unknowns


def dti_shell_settings(config: Dict[str, Any]) -> Tuple[float, float]:
    """(max_bval, tolerance) from ``config['diffusion']['dti']``, with the defaults."""
    dti = (config or {}).get('diffusion', {}).get('dti', {}) or {}
    return (float(dti.get('max_bval', DEFAULT_MAX_BVAL)),
            float(dti.get('bval_tolerance', DEFAULT_TOLERANCE)))


def dti_volume_index(bvals: np.ndarray, max_bval: float = DEFAULT_MAX_BVAL,
                     tolerance: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """Indices of the volumes with b <= max_bval + tolerance (b=0 included).

    Raises ValueError if fewer than six diffusion-weighted volumes remain.
    """
    bvals = np.asarray(bvals, dtype=float).ravel()
    keep = np.flatnonzero(bvals <= max_bval + tolerance)
    n_dw = int(np.sum(bvals[keep] > tolerance))
    if n_dw < MIN_DIRECTIONS:
        shells = sorted({int(round(b, -2)) for b in bvals})
        raise ValueError(
            f"Only {n_dw} diffusion-weighted volumes with b <= {max_bval:g} (shells {shells}); "
            f"a tensor needs at least {MIN_DIRECTIONS}. Set diffusion.dti.max_bval if this "
            f"protocol's lowest shell is higher."
        )
    return keep


def select_dti_volumes(dwi_file, bval_file, bvec_file, out_dir,
                       max_bval: float = DEFAULT_MAX_BVAL,
                       tolerance: float = DEFAULT_TOLERANCE) -> Tuple[str, str, str]:
    """Write the b <= max_bval subset of a DWI series; return (dwi, bval, bvec) paths.

    The bval/bvec files are always written to ``out_dir`` (dti.bval, dti.bvec) so the b-values the
    tensor was fitted on sit with the fit. When every volume is kept, the DWI image itself is not
    copied and its own path is returned.
    """
    import nibabel as nib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bvals = np.loadtxt(bval_file, ndmin=1).ravel()
    bvecs = np.loadtxt(bvec_file, ndmin=2)
    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T
    if bvecs.shape[1] != bvals.size:
        raise ValueError(f"{bval_file} has {bvals.size} values but {bvec_file} has {bvecs.shape[1]}")

    keep = dti_volume_index(bvals, max_bval, tolerance)

    out_bval = out_dir / 'dti.bval'
    out_bvec = out_dir / 'dti.bvec'
    np.savetxt(out_bval, bvals[keep][None, :], fmt='%g')
    np.savetxt(out_bvec, bvecs[:, keep], fmt='%.6f')

    img = nib.load(str(dwi_file))
    if img.shape[-1] != bvals.size:
        raise ValueError(f"{dwi_file} has {img.shape[-1]} volumes but {bval_file} has {bvals.size}")
    if keep.size == bvals.size:
        return str(dwi_file), str(out_bval), str(out_bvec)

    out_dwi = out_dir / 'dti_dwi.nii.gz'
    data = np.asanyarray(img.dataobj)[..., keep]
    hdr = img.header.copy()
    hdr.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, img.affine, hdr), str(out_dwi))
    return str(out_dwi), str(out_bval), str(out_bvec)


def select_dti_volumes_node(dwi_file, bval_file, bvec_file, max_bval, tolerance):
    """Nipype Function-node wrapper: writes into the node's working directory."""
    import os
    from neurovrai.preprocess.utils.dti_shells import select_dti_volumes
    return select_dti_volumes(dwi_file, bval_file, bvec_file, os.getcwd(), max_bval, tolerance)
