"""Re-slicing experiment: per-lesion Dice of a cohort of masks at several thicknesses.

Each mask is re-sliced at every thickness of a list, for several phases; every
original lesion is paired with its degraded pieces and its Dice is recorded.
For each scan and thickness the output gives, per depth class, the mean
per-lesion Dice (lost lesions count as 0) and the number of lesions; per lesion,
the Dice averaged over phases.
"""
from functools import partial
from multiprocessing import Pool

import numpy as np

from . import reslice as R
from .lesions import CLASSES, depth_class, label, match

# default thickness lists (mm), the first value is T0
T_MNI = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0)
T_LESJAK = (0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)


def run_mask(mask, axis, spacing, thicknesses, rule="nn", n_phases=6, connectivity=6, tau=0.10):
    """Re-slice one mask at every thickness and phase.

    Returns ``(class_rows, lesion_rows)``: one row per thickness with
    ``dice_all_e<class>`` and ``n_e<class>`` for every depth class, and one row per
    lesion and thickness with the phase-averaged Dice.
    """
    mask = np.asarray(mask) > 0
    if not mask.any():
        return [], []
    ref_labels = label(mask, connectivity)
    class_rows, lesion_rows = [], []
    for T in thicknesses:
        per_phase = []
        for ph in R.phases(T, n_phases):
            deg = R.degrade(mask, T, rule=rule, phase=float(ph), axis=axis, spacing=spacing)
            per_phase.append(match(mask, deg, connectivity=connectivity, tau=tau, axis=axis,
                                   ref_labels=ref_labels))
        ids = [r["lesion"] for r in per_phase[0]]
        extent = np.array([r["extent"] for r in per_phase[0]], float)
        n_vox = np.array([r["n_vox"] for r in per_phase[0]])
        dice = np.mean([[r["dice"] for r in p] for p in per_phase], axis=0)
        detected = np.mean([[r["detected"] for r in p] for p in per_phase], axis=0)
        cls = depth_class(extent)
        row = dict(T=float(T), rule=rule, n_phases=n_phases)
        for c in CLASSES:
            m = cls == c
            row["dice_all_e%s" % c] = float(dice[m].mean()) if m.any() else float("nan")
            row["n_e%s" % c] = int(m.sum())
        class_rows.append(row)
        for lid, e, nv, d, det in zip(ids, extent, n_vox, dice, detected):
            lesion_rows.append(dict(T=float(T), rule=rule, lesion=lid, extent=int(e),
                                    d_mm=float(e * spacing), n_vox=int(nv), dice=float(d),
                                    detected=float(det)))
    return class_rows, lesion_rows


def _one(item, thicknesses, rule, n_phases, connectivity, tau):
    import nibabel as nib
    img = nib.load(item["mask"])
    mask = np.asarray(img.dataobj) > 0
    zooms = img.header.get_zooms()[:3]
    axis = resolve_axis(item.get("axis", "si"), img.affine, zooms)
    spacing = float(zooms[axis])
    cr, lr = run_mask(mask, axis, spacing, thicknesses, rule, n_phases, connectivity, tau)
    meta = dict(scan_id=item["scan_id"], patient=item["patient"], spacing=spacing)
    return [dict(meta, **r) for r in cr], [dict(meta, **r) for r in lr]


def resolve_axis(spec, affine, zooms):
    """Slice axis from a specification: "si" (head-foot, from the affine),
    "coarsest" (largest voxel spacing) or an integer index."""
    if spec in (None, "", "si"):
        return R.si_axis(affine)
    if spec == "coarsest":
        return R.coarsest_axis(np.sqrt((np.asarray(affine, float)[:3, :3] ** 2).sum(axis=0)))
    return int(spec)


def run_manifest(items, thicknesses, rule="nn", n_phases=6, connectivity=6, tau=0.10, jobs=1,
                 progress=True):
    """Run :func:`run_mask` on every entry of a manifest.

    ``items`` is a list of dicts with ``scan_id``, ``patient``, ``mask`` (path to a
    NIfTI file) and optionally ``axis``. Returns ``(class_rows, lesion_rows)``.
    """
    fn = partial(_one, thicknesses=tuple(thicknesses), rule=rule, n_phases=n_phases,
                 connectivity=connectivity, tau=tau)
    class_rows, lesion_rows = [], []

    def collect(it):
        for k, (cr, lr) in enumerate(it, 1):
            class_rows.extend(cr)
            lesion_rows.extend(lr)
            if progress:
                print("  [%d/%d] %s" % (k, len(items), items[k - 1]["scan_id"]), flush=True)

    if jobs > 1:
        with Pool(jobs) as pool:
            collect(pool.imap(fn, items))
    else:
        collect(map(fn, items))
    return class_rows, lesion_rows
