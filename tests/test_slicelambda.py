import numpy as np
import pytest

from slicelambda import reslice as R
from slicelambda.lesions import depth_class, match, runs_of
from slicelambda.relation import Cohort, class_depth
from slicelambda.runs import Tables, estimate_lambda, fit_pooled, predict


def blob(shape=(24, 24, 32), centre=(11.7, 12.2, 15.3), radii=(3.5, 3.0, 4.5)):
    z = np.indices(shape).astype(float)
    return sum(((z[i] - centre[i]) / radii[i]) ** 2 for i in range(3)) <= 1


# ---------------------------------------------------------------- re-slicing

@pytest.mark.parametrize("rule", ["nn", "pv"])
def test_identity_at_native_thickness(rule):
    m = blob()
    for ph in (0.0, 0.2, 0.4):
        assert np.array_equal(R.degrade(m, 1.0, rule=rule, phase=ph, axis=2), m)


def test_degrade_keeps_in_plane_pattern():
    m = blob()
    d = R.degrade(m, 3.0, rule="nn", phase=0.7, axis=2)
    assert d.shape == m.shape
    # every degraded column is a copy of a native slice of the same column
    for z in range(m.shape[2]):
        assert any(np.array_equal(d[..., z], m[..., k]) for k in range(m.shape[2]))


def test_spacing_scales():
    m = blob()
    a = R.degrade(m, 3.0, rule="pv", phase=0.3, axis=2, spacing=1.0)
    b = R.degrade(m, 2.4, rule="pv", phase=0.24, axis=2, spacing=0.8)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- pairing

def test_lost_lesion_counts_zero_and_split_uses_union():
    ref = np.zeros((10, 10, 10), bool)
    ref[1:3, 1:3, 1:3] = True                 # lesion 1, lost
    ref[5:9, 5:9, 2:8] = True                 # lesion 2, split in two
    test = np.zeros_like(ref)
    test[5:9, 5:9, 2:4] = True
    test[5:9, 5:9, 6:8] = True
    out = match(ref, test, axis=2)
    assert out[0]["dice"] == 0 and not out[0]["detected"]
    assert out[1]["extent"] == 6
    assert out[1]["dice"] == pytest.approx(2 * 64 / (96 + 64))


def test_depth_classes():
    assert list(depth_class([1, 2, 5, 6, 8, 9, 30])) == ["1", "2", "5", "6-8", "6-8", "9+", "9+"]
    assert class_depth("6-8", 0.8, 9.0) == pytest.approx(5.6)
    assert class_depth("9+", 0.8, 9.0) == 9.0


# ---------------------------------------------------------------- run-length model

def _measured_iso(vol, t, n_ph=48):
    """Dice of the isolated lesion re-sliced with the nn rule, averaged over phases."""
    vals = []
    for ph in R.phases(t, n_ph):
        d = R.degrade(vol, t, rule="nn", phase=float(ph), axis=0)
        vals.append(2.0 * (vol & d).sum() / (vol.sum() + d.sum()))
    return float(np.mean(vals))


@pytest.mark.parametrize("t", [1.5, 2.0, 3.0, 5.0])
def test_eq2_exact_on_aligned_cylinder(t):
    z, y, x = np.mgrid[0:40, 0:16, 0:16].astype(float)
    vol = ((y - 7.6) ** 2 + (x - 8.2) ** 2 <= 2.6 ** 2) & (z >= 15) & (z < 23)
    les = runs_of(vol, 0)[0]
    _o, exact, _c = predict(les["columns"], les["offset"], t, Tables(240))
    assert exact == pytest.approx(_measured_iso(vol, t), abs=0.01)


@pytest.mark.parametrize("t", [1.25, 2.0, 3.0, 5.0])
def test_eq2_on_ellipsoid_mixed_runs(t):
    vol = np.moveaxis(blob(shape=(24, 24, 40), centre=(11.7, 12.2, 19.3), radii=(4.5, 3.6, 5.0)), 2, 0)
    les = runs_of(vol, 0)[0]
    assert len({L for c in les["columns"] for _s, L in c}) > 3      # runs of mixed lengths
    old, exact, column = predict(les["columns"], les["offset"], t, Tables(240))
    iso = _measured_iso(vol, t)
    assert exact == pytest.approx(iso, abs=0.011)
    assert column == pytest.approx(exact, abs=0.004)


def test_single_run_reduces_to_D():
    tab = Tables(240)
    _o, exact, _c = predict([[(0, 6)]], 0, 2.5, tab)
    assert exact == pytest.approx(tab.D(6, 2.5))


def test_fit_pooled_recovers_lambda():
    x = np.linspace(0.05, 3, 60)
    assert fit_pooled(0.94 * np.exp(-0.67 * x), x) == pytest.approx((0.94, 0.67), abs=1e-6)


def test_estimate_lambda_runs():
    m = np.zeros((40, 40, 40), bool)
    for k, c in enumerate([(8, 8, 10), (8, 28, 20), (28, 10, 15), (28, 28, 25)]):
        m |= blob(shape=m.shape, centre=(c[0] + .3, c[1] - .2, c[2] + .4), radii=(3, 2.5, 2 + k))
    res = estimate_lambda(m, axis=2, spacing=1.0)
    assert res["n_lesions"] == 4 and 0.2 < res["lam"] < 2.0


# ---------------------------------------------------------------- relation

def _synthetic_rows(c, a, lam, n_pat=12, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    T_grid = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    rows = []
    for p in range(n_pat):
        for s in range(2):
            for T in T_grid:
                row = dict(patient="P%d" % p, scan_id="P%d_T%d" % (p, s), T=T, rule="nn")
                for cls in ("1", "2", "3", "4", "5", "6-8", "9+"):
                    d = class_depth(cls, 1.0, 12.0)
                    y = 1.0 if T == 1.0 else c + a * np.exp(-lam * (T - 1.0) / d)
                    row["dice_all_e%s" % cls] = y + noise * rng.standard_normal()
                    row["n_e%s" % cls] = 5
                rows.append(row)
    return rows


def test_relation_recovers_lambda_and_bootstrap_is_deterministic():
    coh = Cohort(_synthetic_rows(0.05, 0.9, 0.7, noise=0.0), 1.0, 1.0, 12.0, name="toy")
    f = coh.fit()
    assert (f["c"], f["a"], f["lam"]) == pytest.approx((0.05, 0.9, 0.7), abs=1e-4)
    noisy = Cohort(_synthetic_rows(0.05, 0.9, 0.7), 1.0, 1.0, 12.0, name="toy")
    b1, b2 = noisy.bootstrap(100), noisy.bootstrap(100)
    assert (b1["lo"], b1["hi"]) == (b2["lo"], b2["hi"])
    assert b1["lo"] < 0.7 < b1["hi"]
