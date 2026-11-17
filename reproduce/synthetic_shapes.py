#!/usr/bin/env python
"""The run-length model on 35 synthetic shapes with runs of mixed lengths.

No data needed. Each shape is a single lesion on an isotropic grid, slice axis = 0,
re-sliced with the nn rule at t = 1.25, 1.5, 2, 3 and 5. Five quantities, all
averaged over the phase:

  mis     re-slicing + pairing of the degraded pieces, as in slicelambda measure (24 phases)
  iso     re-slicing, Dice of the isolated lesion, no pairing (24 phases)
  exact   run-length prediction (240 phases)
  column  the same prediction with every column holding several runs degraded whole
  old     previous form: length-weighted average of the Dice of the runs

Errors (prediction minus measurement): run-length model = exact - mis; previous form = old - mis;
independence = exact - column; pairing = iso - mis. Mean and, in brackets, maximum
absolute value. Runs in a few seconds.

    python reproduce/synthetic_shapes.py [--csv synthetic.csv]
"""
import argparse

import numpy as np

from slicelambda import reslice as R
from slicelambda.lesions import label, match, runs_of
from slicelambda.runs import Tables, predict

T_SYN = (1.25, 1.5, 2.0, 3.0, 5.0)
FAMILIES = ("aligned cylinders", "oblique cylinders", "ellipsoids", "meandering tubes")


def shapes():
    """(family, name, boolean volume z-y-x, slice axis = 0)."""
    out = []
    N = 48
    z, y, x = np.mgrid[0:N, 0:24, 0:24].astype(float)
    cz, cy, cx = N / 2 + 0.3, 11.7, 12.2                      # centres off the voxel grid
    for az in (1.5, 2.5, 3.5, 5.0, 7.0):
        for axy in (2.5, 4.5):
            v = ((x - cx) / axy) ** 2 + ((y - cy) / (axy * 0.8)) ** 2 + ((z - cz) / az) ** 2 <= 1
            out.append(("ellipsoids", "a_z=%g a_xy=%g" % (az, axy), v))
    for r in (1.6, 2.6):
        for ell in (6.0, 10.0):
            for th in (0, 20, 40, 60):
                tr = np.deg2rad(th)
                u = np.array([np.cos(tr), 0.0, np.sin(tr)])
                P = np.stack([z - cz, y - cy, x - cx], -1)
                s = P @ u
                perp = np.linalg.norm(P - s[..., None] * u, axis=-1)
                v = (np.abs(s) <= ell / 2) & (perp <= r)
                fam = "aligned cylinders" if th == 0 else "oblique cylinders"
                out.append((fam, "r=%g l=%g theta=%d" % (r, ell, th), v))
    for ell in (8, 12, 16):
        for A in (1.0, 2.0, 3.0):
            v = np.zeros(z.shape, bool)
            z0 = int(cz - ell / 2)
            for zz in range(z0, z0 + ell + 1):
                ph = 2 * np.pi * (zz - z0) / ell
                ccx, ccy = cx + A * np.sin(ph), cy + 0.5 * A * np.cos(ph)
                v[zz] |= ((x[zz] - ccx) ** 2 + (y[zz] - ccy) ** 2) <= 1.8 ** 2
            out.append(("meandering tubes", "l=%d A=%g" % (ell, A), v))
    return out


def run(nphi=240, n_ph=24, connectivity=6, tau=0.10):
    tab = Tables(nphi)
    rows = []
    for fam, name, vol in shapes():
        if label(vol, connectivity)[1] != 1:
            continue
        les = runs_of(vol, 0, connectivity)[0]
        for t in T_SYN:
            mis, iso = [], []
            for ph in R.phases(t, n_ph):
                deg = R.degrade(vol, t, rule="nn", phase=float(ph), axis=0, spacing=1.0)
                iso.append(2.0 * (vol & deg).sum() / (vol.sum() + deg.sum()))
                per = match(vol, deg, connectivity=connectivity, tau=tau, axis=0)
                mis.append(per[0]["dice"] if per else 0.0)
            o, e, c = predict(les["columns"], les["offset"], t, tab)
            rows.append(dict(family=fam, shape=name, t=t, n_vox=int(vol.sum()), mis=float(np.mean(mis)),
                             iso=float(np.mean(iso)), old=o, exact=e, column=c))
    return rows


def table(rows):
    def line(name, g):
        err = lambda a, b: np.array([r[a] - r[b] for r in g])
        e2, pv, ind, pr = err("exact", "mis"), err("old", "mis"), err("exact", "column"), err("iso", "mis")
        return "%-18s %3d   %+.3f (%.3f)   %+.3f (%.3f)   %.3f   %.3f" % (
            name, len({r["shape"] for r in g}), e2.mean(), np.abs(e2).max(), pv.mean(), np.abs(pv).max(),
            np.abs(ind).max(), np.abs(pr).max())
    print("%-18s %6s   %-15s   %-15s   %-5s   %s" % ("", "shapes", "run-length", "previous form", "indep", "pairing"))
    for fam in FAMILIES:
        print(line(fam, [r for r in rows if r["family"] == fam]))
    mixed = [r for r in rows if r["family"] != "aligned cylinders"]
    for t in T_SYN:
        print(line("t=%g" % t, [r for r in mixed if r["t"] == t]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="also write the per-shape values to this CSV")
    a = ap.parse_args()
    rows = run()
    table(rows)
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
