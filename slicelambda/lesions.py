"""Lesions as connected components, pairing of original and degraded lesions,
per-lesion Dice and depth classes.

Two components, one in the reference (original) mask and one in the test
(degraded) mask, are linked when they share at least a fraction ``tau`` of the
smaller of the two. Linked components form groups (connected components of the
bipartite graph). The per-lesion Dice of a reference lesion is the Dice between
it and the union of the test components of its group; a lesion with no linked
component is lost and counts as 0.
"""
import numpy as np
from scipy import ndimage

# depth classes in slices: 1, 2, 3, 4, 5, 6-8, 9+
CLASSES = ("1", "2", "3", "4", "5", "6-8", "9+")
FIT_CLASSES = ("2", "3", "4", "5", "6-8", "9+")
_EDGES = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (9, 10 ** 6))


def structure(connectivity):
    return ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])


def label(mask, connectivity=6):
    """Connected components of a binary mask: (labels, number of components)."""
    return ndimage.label(np.asarray(mask) > 0, structure=structure(connectivity))


def depth_class(extent_slices):
    """Depth class of a lesion from its extent along the slice axis, in slices."""
    e = np.asarray(extent_slices)
    out = np.full(e.shape, "", dtype=object)
    for name, (lo, hi) in zip(CLASSES, _EDGES):
        out[(e >= lo) & (e <= hi)] = name
    return out


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _overlaps(lab_a, lab_b, n_b):
    m = (lab_a > 0) & (lab_b > 0)
    if not m.any():
        return {}
    key = lab_a[m].astype(np.int64) * (n_b + 1) + lab_b[m].astype(np.int64)
    k, c = np.unique(key, return_counts=True)
    return {(int(x // (n_b + 1)), int(x % (n_b + 1))): int(v) for x, v in zip(k, c)}


def match(ref, test, connectivity=6, tau=0.10, axis=None, ref_labels=None):
    """Per-lesion Dice of every lesion of ``ref`` against ``test``.

    Returns a list of dicts, one per reference lesion, with keys ``lesion`` (label),
    ``n_vox``, ``extent`` (slices along ``axis``, if given), ``detected`` and ``dice``.
    ``ref_labels`` = (labels, n) may be passed to avoid relabelling the reference.
    """
    lab_r, n_r = ref_labels if ref_labels is not None else label(ref, connectivity)
    lab_t, n_t = label(test, connectivity)
    size_r = np.bincount(lab_r.ravel(), minlength=n_r + 1)
    size_t = np.bincount(lab_t.ravel(), minlength=n_t + 1)
    inter = _overlaps(lab_r, lab_t, n_t)
    extents = {}
    if axis is not None:
        for i, sl in enumerate(ndimage.find_objects(lab_r), 1):
            if sl is not None:
                extents[i] = sl[axis].stop - sl[axis].start
    uf = _UnionFind(n_r + n_t + 2)
    for (i, j), v in inter.items():
        if v >= tau * min(size_r[i], size_t[j]):
            uf.union(i, n_r + j)
    groups = {}
    for i in range(1, n_r + 1):
        groups.setdefault(uf.find(i), ([], []))[0].append(i)
    for j in range(1, n_t + 1):
        groups.setdefault(uf.find(n_r + j), ([], []))[1].append(j)
    out = []
    for rs, ts in groups.values():
        for i in rs:
            inter_v = sum(inter.get((i, j), 0) for j in ts)
            tot_t = sum(int(size_t[j]) for j in ts)
            den = size_r[i] + tot_t
            out.append(dict(lesion=i, n_vox=int(size_r[i]), extent=extents.get(i, np.nan),
                            detected=bool(ts), dice=(2.0 * inter_v / den) if den else 0.0))
    out.sort(key=lambda r: r["lesion"])
    return out


def runs_of(mask, axis, connectivity=6):
    """Runs of every lesion along the slice axis.

    Returns a list of dicts with ``lesion``, ``extent`` (slices), ``offset`` (index of
    the first slice of the lesion's bounding box), ``columns`` (list of columns, each a
    list of (start, length) relative to ``offset``) and ``lbar`` (voxels / runs).
    """
    lab, _n = label(mask, connectivity)
    out = []
    for lid, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        q = np.moveaxis(lab[sl] == lid, axis, 0).reshape(lab[sl].shape[axis], -1)
        cols = []
        for col in q.T:
            if not col.any():
                continue
            d = np.diff(np.r_[0, col.astype(np.int8), 0])
            st = np.nonzero(d == 1)[0]
            en = np.nonzero(d == -1)[0]
            cols.append(list(zip(st.tolist(), (en - st).tolist())))
        n_runs = sum(len(c) for c in cols)
        out.append(dict(lesion=lid, extent=int(q.shape[0]), offset=int(sl[axis].start),
                        columns=cols, lbar=float(q.sum()) / max(n_runs, 1)))
    return out
