"""Command line: ``slicelambda measure | fit | estimate``."""
import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np


def _floats(s):
    return tuple(float(x) for x in s.split(",")) if s else None


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def read_manifest(path):
    """CSV with columns scan_id, patient, mask and optionally axis (si, coarsest or
    an index). Relative mask paths are taken from the folder of the manifest."""
    base = os.path.dirname(os.path.abspath(path))
    items = []
    for r in _read_csv(path):
        for c in ("scan_id", "patient", "mask"):
            if not r.get(c):
                sys.exit("%s: column '%s' missing or empty" % (path, c))
        m = r["mask"] if os.path.isabs(r["mask"]) else os.path.join(base, r["mask"])
        items.append(dict(scan_id=r["scan_id"], patient=r["patient"], mask=m, axis=r.get("axis") or "si"))
    return items


def _print_fit(name, res):
    print("%-14s lambda = %.3f  [%.3f, %.3f]  c = %.3f  a = %.3f  R2 = %.3f  (%d points, %d resamples)"
          % (name, res["lam"], res["lo"], res["hi"], res["c"], res["a"], res["r2"], res["n"], res["n_boot"]))


def cmd_measure(a):
    from .experiment import T_MNI, run_manifest
    from .relation import Cohort, tail_depth
    items = read_manifest(a.manifest)
    T = _floats(a.T) or T_MNI
    os.makedirs(a.out, exist_ok=True)
    print("Re-slicing %d masks at %d thicknesses (%s rule, %d phases)" % (len(items), len(T), a.rule, a.phases))
    cr, lr = run_manifest(items, T, rule=a.rule, n_phases=a.phases, connectivity=a.connectivity,
                          tau=a.tau, jobs=a.jobs)
    _write_csv(os.path.join(a.out, "per_scan_curves.csv"), cr)
    _write_csv(os.path.join(a.out, "lesion_curves.csv.gz"), lr)
    T0 = min(T)
    spacing = a.spacing if a.spacing else float(np.median([r["spacing"] for r in cr]))
    tail = a.tail if a.tail else tail_depth(lr, T0)
    res = Cohort(cr, T0, spacing, tail, rule=a.rule, name=a.name).bootstrap(a.boot, a.seed)
    res.pop("p")
    res.update(name=a.name, T0=T0, spacing=spacing, tail=tail, rule=a.rule)
    json.dump(res, open(os.path.join(a.out, "lambda.json"), "w"), indent=2)
    _print_fit(a.name, res)
    print("Tail depth (class 9+): %.2f mm. Results in %s" % (tail, a.out))


def cmd_fit(a):
    from .relation import Cohort
    rows = _read_csv(a.curves)
    res = Cohort(rows, a.T0, a.spacing or a.T0, a.tail, rule=a.rule, name=a.name).bootstrap(a.boot, a.seed)
    _print_fit(a.name, res)


def cmd_estimate(a):
    import nibabel as nib
    from .experiment import resolve_axis
    from .lesions import FIT_CLASSES, depth_class
    from .runs import default_thicknesses, estimate_lambda, fit_pooled, lesion_predictions, measured_lambda
    if not a.manifest:
        if not a.masks:
            sys.exit("give one or more masks, or --manifest")
        for path in a.masks:
            img = nib.load(path)
            zooms = img.header.get_zooms()[:3]
            axis = resolve_axis(a.axis, img.affine, zooms)
            sp = float(zooms[axis])
            res = estimate_lambda(np.asarray(img.dataobj) > 0, axis, sp, _floats(a.T), nphi=a.nphi,
                                  connectivity=a.connectivity)
            print("%s  axis %d  T0 = %.2f mm  lesions (2+ slices) %d  lambda = %.3f  a = %.3f"
                  % (os.path.basename(path), axis, sp, res["n_lesions"], res["lam"], res["a"]))
        return
    # cohort: all lesions of all masks in one fit
    items = read_manifest(a.manifest)
    geo = []
    for it in items:
        img = nib.load(it["mask"])
        zooms = img.header.get_zooms()[:3]
        axis = resolve_axis(a.axis if a.axis != "si" else it["axis"], img.affine, zooms)
        geo.append((img, axis, float(zooms[axis])))
    sps = [round(g[2], 3) for g in geo]
    T0 = a.T0 if a.T0 else max(set(sps), key=sps.count)        # most common spacing
    rows, skipped = [], []
    for k, (it, (img, axis, sp)) in enumerate(zip(items, geo), 1):
        if abs(sp - T0) > 1e-3:
            skipped.append(it["scan_id"])
            continue
        Ts = [T for T in (_floats(a.T) or default_thicknesses(sp)) if T > sp + 1e-6]
        for r in lesion_predictions(np.asarray(img.dataobj) > 0, axis, sp, Ts, a.nphi, a.connectivity):
            rows.append(dict(r, scan_id=it["scan_id"]))
        print("  [%d/%d] %s" % (k, len(items), it["scan_id"]), flush=True)
    if skipped:
        print("Skipped (spacing differs from %.2f mm): %s" % (T0, ", ".join(skipped)))
    keep = [r for r in rows if depth_class([r["extent"]])[0] in FIT_CLASSES]
    a_p, lam_p = fit_pooled([r["pred"] for r in keep], [(r["T"] - T0) / r["d_mm"] for r in keep])
    n_les = len({(r["scan_id"], r["lesion"]) for r in keep})
    print("lambda predicted from runs = %.3f  a = %.3f  (%d lesions of 2+ slices)" % (lam_p, a_p, n_les))
    if a.lesion_curves:
        meas = [r for r in _read_csv(a.lesion_curves) if r["scan_id"] not in skipped
                and abs(float(r.get("spacing", T0)) - T0) < 1e-3 and r.get("rule", "nn") == "nn"]
        a_m, lam_m = measured_lambda(meas, T0)
        print("lambda measured         = %.3f  a = %.3f" % (lam_m, a_m))
        pred = {(r["scan_id"], r["lesion"], round(r["T"], 6)): r["pred"] for r in keep}
        err, cls_err = [], {}
        for r in meas:
            key = (r["scan_id"], int(r["lesion"]), round(float(r["T"]), 6))
            if key in pred:
                err.append(float(r["dice"]) - pred[key])
                cls_err.setdefault((depth_class([int(r["extent"])])[0], key[2]), []).append(float(r["dice"]))
        cm = [v - np.mean(g) for g in cls_err.values() for v in g]
        print("per-lesion error (RMS): run-length model %.3f, class mean %.3f"
              % (np.sqrt(np.mean(np.square(err))), np.sqrt(np.mean(np.square(cm)))))
    if a.out:
        _write_csv(a.out, rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slicelambda", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="re-slice a cohort of masks and fit lambda with a patient bootstrap")
    m.add_argument("manifest", help="CSV with scan_id, patient, mask[, axis]")
    m.add_argument("--out", default="slicelambda_out")
    m.add_argument("--name", default="cohort", help="cohort name (also seeds the bootstrap)")
    m.add_argument("--T", help="comma-separated thicknesses in mm, the first is T0 (default: 1 mm grid)")
    m.add_argument("--rule", default="nn", choices=["nn", "pv"])
    m.add_argument("--phases", type=int, default=6)
    m.add_argument("--spacing", type=float, help="native spacing for class depths (default: from masks)")
    m.add_argument("--tail", type=float, help="depth of class 9+ in mm (default: median of the cohort)")
    m.add_argument("--boot", type=int, default=1000)
    m.add_argument("--seed", type=int, default=20261024)
    m.add_argument("--connectivity", type=int, default=6, choices=[6, 18, 26])
    m.add_argument("--tau", type=float, default=0.10)
    m.add_argument("--jobs", type=int, default=1)
    m.set_defaults(fn=cmd_measure)

    f = sub.add_parser("fit", help="fit lambda on existing per-scan class curves")
    f.add_argument("curves", help="per_scan_curves.csv")
    f.add_argument("--T0", type=float, required=True)
    f.add_argument("--spacing", type=float)
    f.add_argument("--tail", type=float, required=True)
    f.add_argument("--name", default="cohort")
    f.add_argument("--rule", default="nn")
    f.add_argument("--boot", type=int, default=1000)
    f.add_argument("--seed", type=int, default=20261024)
    f.set_defaults(fn=cmd_fit)

    e = sub.add_parser("estimate", help="lambda predicted from the runs of the lesions alone")
    e.add_argument("masks", nargs="*", help="NIfTI lesion masks (one lambda per mask)")
    e.add_argument("--manifest", help="CSV as for measure: one lambda for all lesions of all masks")
    e.add_argument("--lesion-curves", help="lesion_curves.csv.gz from measure: also fit the measured lambda")
    e.add_argument("--out", help="write the per-lesion predictions to this CSV")
    e.add_argument("--axis", default="si", help="si, coarsest or an axis index (default: si)")
    e.add_argument("--T0", type=float, help="with --manifest: keep masks with this spacing "
                   "(default: the most common one)")
    e.add_argument("--T", help="comma-separated thicknesses in mm (default: 1 mm grid, or 0.8 mm grid for 0.8 mm masks)")
    e.add_argument("--nphi", type=int, default=240)
    e.add_argument("--connectivity", type=int, default=6, choices=[6, 18, 26])
    e.set_defaults(fn=cmd_estimate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
