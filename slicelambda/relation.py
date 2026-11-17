"""The relation R(T) = c + a * exp(-lambda * (T - T0) / d), fitted on class curves.

For each depth class and thickness, the per-scan class means are averaged over
scans (one scan, one value), pooling patients. The relation is fitted on all
classes of two slices or more at once, for T > T0. Confidence intervals come from
a bootstrap that resamples patients with replacement, each with all its scans.
"""
import zlib

import numpy as np
from scipy.optimize import least_squares

from .lesions import CLASSES, FIT_CLASSES

# parameters c, a, lambda: start and bounds of the fit
P0 = np.array([0.0, 0.95, 0.70])
LO = np.array([0.0, 0.0, 0.0])
HI = np.array([1.0, 1.5, 5.0])
SEED = 20261024


def class_depth(cls, spacing, tail):
    """Depth d (mm) of a class: slices x spacing, 7 slices for 6-8, ``tail`` for 9+."""
    if cls == "9+":
        return float(tail)
    if cls == "6-8":
        return 7.0 * float(spacing)
    return float(cls) * float(spacing)


def tail_depth(lesion_rows, T0):
    """Depth of the class 9+: median extent (mm) of the lesions of nine slices or more."""
    seen = {}
    for r in lesion_rows:
        if abs(r["T"] - T0) < 1e-6 and r["extent"] >= 9:
            seen[(r["scan_id"], r["lesion"])] = r["d_mm"]
    return float(np.median(list(seen.values()))) if seen else 9.0 * T0


class Cohort:
    """Class curves of one population, ready for pooled fits and patient bootstrap.

    ``rows`` are per-scan class rows (from :mod:`slicelambda.experiment` or a CSV
    with columns ``patient``, ``T``, ``rule``, ``dice_all_e<class>``, ``n_e<class>``).
    """

    def __init__(self, rows, T0, spacing, tail, rule="nn", name="cohort"):
        self.name, self.T0, self.spacing, self.tail = name, float(T0), float(spacing), float(tail)
        acc = {}
        for row in rows:
            if row.get("rule", rule) != rule:
                continue
            T = float(row["T"])
            for c in CLASSES:
                d, n = row.get("dice_all_e%s" % c), row.get("n_e%s" % c)
                if d in (None, "") or n in (None, ""):
                    continue
                d, n = float(d), float(n)
                if not np.isfinite(d) or n <= 0:
                    continue
                a = acc.setdefault((str(row["patient"]), c, T), [0.0, 0])
                a[0] += d
                a[1] += 1
        if not acc:
            raise ValueError("no usable rows for rule %r" % rule)
        self.patients = sorted({k[0] for k in acc})
        keys = sorted({(k[1], k[2]) for k in acc}, key=lambda q: (CLASSES.index(q[0]), q[1]))
        pi = {p: i for i, p in enumerate(self.patients)}
        ki = {k: j for j, k in enumerate(keys)}
        self.S = np.zeros((len(self.patients), len(keys)))
        self.C = np.zeros((len(self.patients), len(keys)))
        for (p, c, T), (s, n) in acc.items():
            self.S[pi[p], ki[(c, T)]] = s
            self.C[pi[p], ki[(c, T)]] = n
        cls = np.array([k[0] for k in keys])
        T = np.array([k[1] for k in keys], float)
        d = np.array([class_depth(c, self.spacing, self.tail) for c in cls])
        self.X = dict(T=T, d=d, cls=cls, fit=np.isin(cls, FIT_CLASSES) & (np.abs(T - self.T0) > 1e-6))

    def pooled(self, w=None):
        """Class curves: mean over scans, patients weighted by ``w`` (bootstrap counts)."""
        w = np.ones(len(self.patients)) if w is None else np.asarray(w, float)
        num, den = w @ self.S, w @ self.C
        with np.errstate(invalid="ignore", divide="ignore"):
            y = num / den
        y[den == 0] = np.nan
        return y

    def x(self):
        return (self.X["T"] - self.T0) / self.X["d"]

    def fit(self, y=None, p_start=None, window=None):
        """Fit c, a, lambda on classes of two slices or more and T > T0.

        ``window`` = (min_d, max_d) in mm restricts the fit to classes of that depth.
        Returns a dict with ``c``, ``a``, ``lam``, ``r2``, ``rmse`` and ``n``.
        """
        y = self.pooled() if y is None else y
        m = self.X["fit"] & np.isfinite(y)
        if window is not None:
            m &= (self.X["d"] >= window[0] - 1e-9) & (self.X["d"] <= window[1] + 1e-9)
        x, ys = self.x()[m], y[m]
        eps = 1e-6 * np.maximum(HI - LO, 1.0)
        start = np.clip(P0 if p_start is None else np.asarray(p_start, float), LO + eps, HI - eps)
        f = lambda p: p[0] + p[1] * np.exp(-p[2] * x)
        r = least_squares(lambda p: f(p) - ys, start, bounds=(LO, HI), max_nfev=20000, x_scale="jac")
        e = ys - f(r.x)
        ss = float(((ys - ys.mean()) ** 2).sum())
        return dict(c=float(r.x[0]), a=float(r.x[1]), lam=float(r.x[2]), p=r.x, n=int(m.sum()),
                    r2=1.0 - float((e ** 2).sum()) / ss if ss > 0 else float("nan"),
                    rmse=float(np.sqrt(np.mean(e ** 2))))

    def bootstrap(self, n_boot=1000, seed=SEED, window=None):
        """95 % interval of lambda from a bootstrap of patients.

        The random stream depends on the seed and on the cohort name.
        """
        full = self.fit(window=window)
        rng = np.random.default_rng([seed, zlib.crc32(self.name.encode())])
        n = len(self.patients)
        lams = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            try:
                lams.append(self.fit(self.pooled(w), p_start=full["p"], window=window)["lam"])
            except Exception:
                pass
        lams = np.asarray(lams)
        return dict(full, lo=float(np.percentile(lams, 2.5)), hi=float(np.percentile(lams, 97.5)),
                    n_boot=int(len(lams)))


def predict(c, a, lam, T, T0, d):
    """Per-lesion Dice predicted by the relation at thickness T for depth d (mm)."""
    T = np.asarray(T, float)
    return np.where(T > T0, c + a * np.exp(-lam * (T - T0) / d), 1.0)
