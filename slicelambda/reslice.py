"""Re-slicing a binary mask at a larger slice thickness.

Along the slice axis the native voxels (spacing ``T0``) are grouped into thick
slices of thickness ``T`` starting at ``-phase``. For each thick slice the exact
amount of lesion inside it is computed (box slice profile), and the slice is
labelled lesion by one of two rules:

* ``"nn"``: if the native voxel that contains the centre of the thick slice is lesion;
* ``"pv"``: if lesion fills at least half of the thickness of the slice.

Every native voxel then takes the label of the thick slice that contains its
centre, so the degraded mask lives on the original grid and can be compared with
the original voxel by voxel. The position of the slice grid relative to the
anatomy is arbitrary, so results are averaged over equally spaced phases in
``[0, T)`` (:func:`phases`).
"""
import numpy as np

RULES = ("nn", "pv")


def _slab_weights(n, T, phase, spacing=1.0):
    """Overlap in mm between thick slices (rows) and native voxels (columns)."""
    edges_v = np.arange(n + 1) * spacing
    n_slab = int(np.ceil((n * spacing + phase) / T))
    edges_s = -phase + np.arange(n_slab + 1) * T
    lo = np.maximum(edges_s[:-1, None], edges_v[None, :-1])
    hi = np.minimum(edges_s[1:, None], edges_v[None, 1:])
    return np.clip(hi - lo, 0, None), edges_s


def degrade(mask, T, rule="nn", phase=0.0, axis=2, spacing=1.0):
    """Mask re-sliced at thickness ``T`` (mm) and brought back to its own grid.

    Parameters
    ----------
    mask : array_like of bool
        Binary lesion mask.
    T : float
        Thickness of the thick slices, in mm (``T >= spacing``).
    rule : {"nn", "pv"}
        Labelling rule of a thick slice.
    phase : float
        Offset of the slice grid, in mm, in ``[0, T)``.
    axis : int
        Slice axis of the array.
    spacing : float
        Native spacing ``T0`` along the slice axis, in mm.
    """
    mask = np.asarray(mask, bool)
    m = np.moveaxis(mask, axis, -1)
    n = m.shape[-1]
    W, edges_s = _slab_weights(n, T, phase, spacing)
    if rule == "pv":
        occ = m.astype(np.float32) @ W.T.astype(np.float32)   # mm of lesion in each thick slice
        lab = occ >= 0.5 * T
    elif rule == "nn":
        centres = 0.5 * (edges_s[:-1] + edges_s[1:])
        idx = np.clip(np.floor(centres / spacing).astype(int), 0, n - 1)
        lab = m[..., idx]
    else:
        raise ValueError("rule must be 'nn' or 'pv', got %r" % (rule,))
    vox_centres = (np.arange(n) + 0.5) * spacing
    back = np.clip(np.searchsorted(edges_s, vox_centres) - 1, 0, lab.shape[-1] - 1)
    return np.moveaxis(lab[..., back], -1, axis)


def phases(T, n_phases):
    """``n_phases`` equally spaced phases in ``[0, T)``."""
    return np.linspace(0.0, T, n_phases, endpoint=False)


def si_axis(affine):
    """Index of the array axis aligned with superior-inferior (head-foot)."""
    from nibabel.orientations import aff2axcodes
    for i, c in enumerate(aff2axcodes(affine)):
        if c in "SI":
            return i
    raise ValueError("no superior-inferior axis in affine")


def coarsest_axis(zooms):
    """Index of the axis with the largest voxel spacing (the acquired slice axis)."""
    return int(np.argmax(np.asarray(zooms, float)[:3]))
