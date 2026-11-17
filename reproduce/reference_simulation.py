#!/usr/bin/env python
"""Same prediction, two references (a simulation): hidden loss and Dice differences.

Input: ``data/refmethod_<method>.csv``, one row per patient and FLAIR thickness, with
the Dice of the method against the original reference (``dice_orig``) and against the
reference re-sliced to the same thickness (``dice_deg_nn``, ``dice_deg_pv``), the
lesion-wise F1 (``f1_*``) and the Dice of the thick reference against the original
(``gt_vs_gt_*``). These files are derived from the 30 Lesjak patients and need no images.

Slopes are per patient (least-squares line of Dice against thickness). Hidden loss
h = slope against thick reference - slope against original reference; hidden share
= median(h) / |median(original slope)|. Intervals: bootstrap of patients, 1000
resamples, seed 20261024. The order of the resamplings is that of the
published analysis, so the intervals are reproduced exactly.

    python reproduce/reference_simulation.py
"""
import argparse
import os

import numpy as np
import pandas as pd

METHODS = {"mssegdiff": "MSSegDiff", "lstai": "LST-AI"}
RULES = ("nn", "pv")
HERE = os.path.dirname(os.path.abspath(__file__))


def slopes(df, col):
    return pd.Series({p: np.polyfit(g.sort_values("T")["T"].values, g.sort_values("T")[col].values, 1)[0]
                      for p, g in df.groupby("patient")})


def boot_ci(frame, stat, n_boot, rng):
    idx = np.arange(len(frame))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        reps[b] = stat(frame.iloc[rng.choice(idx, size=len(idx), replace=True)])
    return np.percentile(reps, [2.5, 97.5])


def ci(est, c, nd=4):
    return "%+.*f [%+.*f, %+.*f]" % (nd, est, nd, c[0], nd, c[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261024)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    data = {m: pd.read_csv(os.path.join(a.dir, "refmethod_%s.csv" % m)) for m in METHODS}
    out = {m: {} for m in METHODS}

    for m in METHODS:
        d = data[m]
        s_orig = slopes(d, "dice_orig")
        o = out[m]
        o["gt"] = {r: slopes(d, "gt_vs_gt_%s" % r).median() for r in RULES}
        o["s_orig"] = s_orig.median()
        o["neg"] = int((s_orig < 0).sum())
        for r in RULES:
            s_thick = slopes(d, "dice_deg_%s" % r)
            pp = pd.DataFrame({"s_orig": s_orig, "h": s_thick - s_orig})
            share = lambda x: x["h"].median() / abs(x["s_orig"].median())
            boot_ci(pp, lambda x: x["h"].median(), a.boot, rng)               # kept for the random stream
            c_sh = boot_ci(pp, share, a.boot, rng)
            boot_ci(pp, lambda x: x["h"].median() / x["s_orig"].abs().median(), a.boot, rng)
            neg = pp[pp["s_orig"] < 0]
            q = (neg["h"] / neg["s_orig"].abs()).quantile([0.25, 0.5, 0.75]).values
            o[r] = dict(s_thick=s_thick.median(), h=pp["h"].median(), share=share(pp), share_ci=c_sh,
                        q=q, h_pos=int((pp["h"] > 0).sum()))

    for m in METHODS:
        d = data[m]
        for r in RULES:
            for T in (3.0, 4.2):
                g = d[np.isclose(d["T"], T)].set_index("patient")
                diff = (g["dice_deg_%s" % r] - g["dice_orig"]).to_frame("x")
                c = boot_ci(diff, lambda x: x["x"].mean(), a.boot, rng)
                boot_ci(diff, lambda x: x["x"].median(), a.boot, rng)              # kept for the random stream
                out[m]["diff_%s_%g" % (r, T)] = (diff["x"].mean(), c, int((diff["x"] > 0).sum()), len(diff))

    n = len(data["mssegdiff"]["patient"].unique())
    M, L = out["mssegdiff"], out["lstai"]
    rows = [("thick reference against original (nn / pv)",
             "%+.3f / %+.3f" % (M["gt"]["nn"], M["gt"]["pv"]), "(same references)"),
            ("method against original reference",
             "%+.3f (%d/%d negative)" % (M["s_orig"], M["neg"], n), "%+.3f (%d/%d)" % (L["s_orig"], L["neg"], n))]
    for r in RULES:
        rows.append(("method against thick reference, %s" % r, "%+.3f" % M[r]["s_thick"], "%+.3f" % L[r]["s_thick"]))
    rows.append(("hidden loss h, median, nn / pv", "%.3f / %.3f" % (M["nn"]["h"], M["pv"]["h"]),
                 "%.3f / %.3f" % (L["nn"]["h"], L["pv"]["h"])))
    sh = lambda o: " / ".join("%.0f %% [%.0f, %.0f]" % (100 * o[r]["share"], 100 * o[r]["share_ci"][0],
                                                       100 * o[r]["share_ci"][1]) for r in RULES)
    rows.append(("hidden share, nn / pv", sh(M), sh(L)))
    for r in RULES:
        q = lambda o: "%.0f %% (%.0f-%.0f %%)" % (100 * o[r]["q"][1], 100 * o[r]["q"][0], 100 * o[r]["q"][2])
        rows.append(("hidden share per patient, median (IQR), %s" % r, q(M), q(L)))
    rows.append(("patients with h > 0, nn / pv", "%d / %d" % (M["nn"]["h_pos"], M["pv"]["h_pos"]),
                 "%d / %d" % (L["nn"]["h_pos"], L["pv"]["h_pos"])))
    for T in (3.0, 4.2):
        for r in RULES:
            f = lambda o: "%s (%d/%d)" % (ci(o["diff_%s_%g" % (r, T)][0], o["diff_%s_%g" % (r, T)][1]),
                                          o["diff_%s_%g" % (r, T)][2], o["diff_%s_%g" % (r, T)][3])
            rows.append(("Dice, thick minus original, %g mm, %s" % (T, r), f(M), f(L)))

    w = max(len(r[0]) for r in rows)
    print("%-*s | %-40s | %s" % (w, "Same prediction, two references", "MSSegDiff", "LST-AI"))
    print("-" * (w + 90))
    for r in rows:
        print("%-*s | %-40s | %s" % (w, r[0], r[1], r[2]))


if __name__ == "__main__":
    main()
