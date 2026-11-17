"""Run-length model: lambda predicted from the runs of the lesions alone.

Under the nn rule each column of voxels along the slice axis is degraded on its
own. In units of native slices (t = T / T0), with phase p, native index i falls
in thick slice k = floor((i - p) / t), which takes the value of native slice
j = floor(p + (k + 0.5) t + 0.5); back on the native grid, i takes the value of j.
The phase origin differs from :func:`slicelambda.reslice.degrade` by a constant
shift, which does not matter once results are averaged over the phase.

A lesion is a set of runs (start s_r, length L_r). Its Dice at phase p is
2 sum_r I_r(p) / (sum_r L_r + sum_r B_r(p)), with I_r and B_r the overlap and the
degraded length of run r; by periodicity and translation invariance,
I_r(p) = I(L_r, t, (p - s_r) mod t), so I and B are tabulated once per (L, t) on a
grid of phases and reused for every run (``exact``). Two variants are kept
for checks: ``old`` (length-weighted average of
the phase-averaged Dice of the runs) and ``column`` (columns with several runs
degraded whole, which also accounts for neighbouring runs merging).
"""
import numpy as np
from scipy.optimize import least_squares

from .lesions import FIT_CLASSES, depth_class, runs_of


def column_IB(A_idx, t, phis):
    """Overlap |A and B| and degraded length |B| of one column, for every phase.

    ``A_idx`` are the (absolute, integer) indices of the lesion voxels in the column.
    """
    A_idx = np.asarray(A_idx, int)
    lo, hi = A_idx.min(), A_idx.max()
    pad = int(np.ceil(2 * t)) + 2
    i = np.arange(lo - pad, hi + pad + 1)
    A = np.zeros(i.size, bool)
    A[A_idx - (lo - pad)] = True
    p = np.asarray(phis, float)[:, None]
    k = np.floor((i[None, :] - p) / t)
    j = np.floor(p + (k + 0.5) * t + 0.5).astype(int) - (lo - pad)
    ok = (j >= 0) & (j < i.size)
    Bm = np.zeros(j.shape, bool)
    Bm[ok] = A[j[ok]]
    return (A[None, :] & Bm).sum(1).astype(float), Bm.sum(1).astype(float)


class Tables:
    """I(L, t, psi) and B(L, t, psi) on a grid of ``nphi`` phases in [0, t)."""

    def __init__(self, nphi=240):
        self.nphi = nphi
        self.cache = {}

    def grid(self, t):
        return (np.arange(self.nphi) + 0.5) / self.nphi * t

    def get(self, L, t):
        key = (int(L), round(float(t), 6))
        if key not in self.cache:
            self.cache[key] = column_IB(np.arange(int(L)), float(t), self.grid(t))
        return self.cache[key]

    def D(self, L, t):
        """Phase-averaged Dice of a single run of L slices."""
        I, Bc = self.get(L, t)
        return float(np.mean(2 * I / (L + Bc)))

    def shift_index(self, s, t):
        """Table index of psi = (p_m - s) mod t for every phase p_m of the grid."""
        psi = np.mod(self.grid(t) - s, t)
        return np.minimum((psi / t * self.nphi).astype(int), self.nphi - 1)


def predict(cols, offset, t, tab, with_column=True):
    """Predicted Dice of one lesion at thickness ratio ``t``.

    ``cols`` are the columns of the lesion (lists of (start, length), start relative
    to ``offset``, the absolute index of the first slice of the lesion). Returns
    ``(old, exact, column)``; ``exact`` is the run-length prediction.
    """
    if t <= 1.0 + 1e-9:
        return 1.0, 1.0, 1.0
    runs = [(s + offset, L) for c in cols for (s, L) in c]
    Ls = np.array([L for _s, L in runs], float)
    V = Ls.sum()
    old = float(sum(L * tab.D(L, t) for L in Ls) / V)
    groups = {}
    for s, L in runs:
        groups[(s, L)] = groups.get((s, L), 0) + 1
    It = np.zeros(tab.nphi)
    Bt = np.zeros(tab.nphi)
    for (s, L), n in groups.items():
        I, Bc = tab.get(L, t)
        ix = tab.shift_index(s, t)
        It += n * I[ix]
        Bt += n * Bc[ix]
    exact = float(np.mean(2 * It / (V + Bt)))
    if not with_column:
        return old, exact, float("nan")
    Ic, Bcol = It.copy(), Bt.copy()
    p = tab.grid(t)
    for c in cols:
        if len(c) < 2:
            continue
        for s, L in c:
            I, Bc = tab.get(L, t)
            ix = tab.shift_index(s + offset, t)
            Ic -= I[ix]
            Bcol -= Bc[ix]
        idx = np.concatenate([np.arange(s, s + L) for s, L in c]) + offset
        I2, B2 = column_IB(idx, t, p)
        Ic += I2
        Bcol += B2
    column = float(np.mean(2 * Ic / (V + Bcol)))
    return old, exact, column


def fit_pooled(Y, X):
    """Fit R = a exp(-lambda x) (c = 0) to all values together; returns (a, lambda)."""
    Y, X = np.asarray(Y, float), np.asarray(X, float)
    ok = np.isfinite(Y) & np.isfinite(X)
    y, x = Y[ok], X[ok]
    r = least_squares(lambda p: p[0] * np.exp(-p[1] * x) - y, [0.95, 0.7], bounds=([0, 0], [1.5, 20]))
    return float(r.x[0]), float(r.x[1])


def lesion_predictions(mask, axis, spacing, thicknesses, nphi=240, connectivity=6, with_column=False):
    """Run-length prediction for every lesion of a mask and every thickness.

    Returns a list of dicts with ``lesion``, ``extent`` (slices), ``d_mm``, ``lbar``,
    ``T`` and ``pred`` (``exact``); with ``with_column`` also ``old`` and ``column``.
    """
    tab = Tables(nphi)
    out = []
    for les in runs_of(mask, axis, connectivity):
        for T in thicknesses:
            o, e, c = predict(les["columns"], les["offset"], float(T) / spacing, tab, with_column)
            row = dict(lesion=les["lesion"], extent=les["extent"], d_mm=les["extent"] * spacing,
                       lbar=les["lbar"], T=float(T), pred=e)
            if with_column:
                row.update(old=o, column=c)
            out.append(row)
    return out


def default_thicknesses(spacing):
    """Default thicknesses: the 1 mm grid for spacing 1 mm, the 0.8 mm grid
    for 0.8 mm, otherwise the 1 mm grid scaled by the spacing."""
    from .experiment import T_LESJAK, T_MNI
    if abs(spacing - 0.8) < 1e-6:
        return T_LESJAK
    return tuple(round(spacing * T, 6) for T in T_MNI)


def estimate_lambda(mask, axis, spacing, thicknesses=None, nphi=240, connectivity=6):
    """Decay constant of a mask from its runs alone.

    Lesions of two slices or more; R_l(T) from the run-length prediction for every T > T0; one fit of
    R = a exp(-lambda (T - T0) / d_l) to all lesions together. Returns a dict with
    ``lam``, ``a``, ``n_lesions`` and the per-lesion ``rows``.
    """
    thicknesses = default_thicknesses(spacing) if thicknesses is None else thicknesses
    Ts = [float(T) for T in thicknesses if T > spacing + 1e-6]
    rows = lesion_predictions(mask, axis, spacing, Ts, nphi, connectivity)
    keep = [r for r in rows if depth_class([r["extent"]])[0] in FIT_CLASSES]
    if not keep:
        raise ValueError("no lesion spans two slices or more")
    Y = np.array([r["pred"] for r in keep])
    X = np.array([(r["T"] - spacing) / r["d_mm"] for r in keep])
    a, lam = fit_pooled(Y, X)
    return dict(lam=lam, a=a, n_lesions=len({r["lesion"] for r in keep}), rows=rows)


def measured_lambda(lesion_rows, T0):
    """Same fit as :func:`estimate_lambda` on measured per-lesion curves (rows of
    :func:`slicelambda.experiment.run_mask`, or a CSV with ``extent``, ``d_mm``, ``T``,
    ``dice``). Returns (a, lambda)."""
    keep = [r for r in lesion_rows if float(r["T"]) > T0 + 1e-6
            and depth_class([int(r["extent"])])[0] in FIT_CLASSES]
    Y = np.array([float(r["dice"]) for r in keep])
    X = np.array([(float(r["T"]) - T0) / float(r["d_mm"]) for r in keep])
    return fit_pooled(Y, X)
