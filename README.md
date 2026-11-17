# slicelambda

**How much slice-axis resolution does a lesion ground truth hide?**

Code for the paper *The hidden slice-axis resolution of lesion ground truth: measurement,
mechanism and consequences for evaluation in multiple sclerosis* (F. Guarnera, A. Rondinella,
D. Ravì, S. Battiato; submitted to Medical Image Analysis).

A lesion mask drawn on thin slices is re-sliced at larger thicknesses *T*. Each lesion is paired
with its degraded pieces and its Dice is recorded. Grouped by depth *d* along the slice axis, the
per-lesion Dice follows

$$R(T) = c + a\,e^{-\lambda (T-T_0)/d}, \qquad T > T_0,$$

with one decay constant **λ** per dataset. `slicelambda` measures λ from the masks alone. It can
also predict λ, with no fitted parameter, from the runs that lesions draw along the slice axis
(the run-length model).

## Install

```bash
git clone https://github.com/guarnerafrancesco/slicelambda.git
cd slicelambda
pip install -e ".[test]"          # numpy, scipy, nibabel; pytest for the tests
pytest -q tests                   # about one second
```

Python 3.9 or later. `pandas` is needed only by `reproduce/reference_simulation.py`.

## Quick start

### Python

```python
import nibabel as nib, numpy as np
import slicelambda as sl

img = nib.load("mask.nii.gz")
mask = np.asarray(img.dataobj) > 0
axis = sl.reslice.si_axis(img.affine)                  # head-foot axis
T0 = float(img.header.get_zooms()[axis])

thick = sl.degrade(mask, T=3.0, rule="nn", phase=0.5, axis=axis, spacing=T0)
per_lesion = sl.match(mask, thick, axis=axis)          # Dice of every original lesion

res = sl.estimate_lambda(mask, axis, T0)               # lambda predicted from the runs alone
print(res["lam"])
```

### Command line

A cohort is described by a **manifest**, a CSV with columns `scan_id`, `patient` and `mask` (a
path to a NIfTI mask), plus an optional `axis` (`si`, `coarsest` or an index; default `si`).
`reproduce/make_manifest.py` writes one from a glob pattern.

```bash
# re-slice every mask (nn rule, 6 phases) and fit lambda with a bootstrap of patients
slicelambda measure cohort.csv --name mycohort --jobs 8 --out out/
#   -> out/per_scan_curves.csv, out/lesion_curves.csv.gz, out/lambda.json

# fit lambda again from existing class curves
slicelambda fit out/per_scan_curves.csv --T0 1.0 --tail 12.0 --name mycohort

# lambda predicted from runs: one per mask, or one for a whole cohort
slicelambda estimate mask1.nii.gz mask2.nii.gz
slicelambda estimate --manifest cohort.csv --lesion-curves out/lesion_curves.csv.gz
```

## Reproducing the results

### Without data (seconds)

| Command | What it computes |
| --- | --- |
| `python reproduce/synthetic_shapes.py` | Run-length model against re-slicing on 35 synthetic shapes with runs of mixed lengths |
| `python reproduce/reference_simulation.py` | Same prediction, two references: hidden loss and Dice differences |

`reproduce/data/refmethod_*.csv` holds the per-patient Dice used by `reference_simulation.py`:
MSSegDiff and LST-AI on the 30 Lesjak patients, against the original and the re-sliced reference.
The scripts print their tables; the values are exact, intervals included.

### With the datasets

The datasets are public (MSLesSeg, ISBI 2015, Lesjak) and are not redistributed here. Write one
manifest per dataset, then run:

```bash
# MSLesSeg, masks registered to MNI (1 mm), slice axis = head-foot
slicelambda measure mslesseg.csv --name mslesseg --tail 12.0 --jobs 8 --out out/mslesseg
# ISBI 2015, rater 1, registered to MNI
slicelambda measure isbi1.csv --name isbi1 --tail 14.0 --jobs 8 --out out/isbi1
# Lesjak, native space, along the acquired slice axis (0.8 mm): axis = coarsest in the manifest
slicelambda measure lesjak.csv --name lesjak_slice --tail 9.0 --spacing 0.8 \
    --T 0.8,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,8 --jobs 8 --out out/lesjak

# lambda measured on lesion curves and predicted from runs, per-lesion errors
slicelambda estimate --manifest mslesseg.csv --lesion-curves out/mslesseg/lesion_curves.csv.gz
```

Expected values:

| | MSLesSeg | ISBI, rater 1 | Lesjak |
| --- | --- | --- | --- |
| λ from class curves (`measure`) | 0.702 [0.686, 0.718] | 0.704 | 0.938 [0.905, 0.972] |
| λ measured on lesion curves (`estimate --lesion-curves`) | 0.672 | 0.680 | 0.927 |
| λ predicted from runs (`estimate`) | 0.650 | 0.654 | 0.846 |
| per-lesion error, run-length model / class mean | 0.062 / 0.082 | 0.063 / 0.080 | 0.085 / 0.099 |

Point estimates do not depend on random numbers. The bootstrap stream is seeded with
`[20261024, crc32(name)]`, and patients are resampled in sorted order of their identifiers. The
intervals are therefore reproduced exactly with the cohort names above and the identifiers we
used (for example `P1`, `P2`, …), and within bootstrap noise otherwise. ISBI has 5 patients,
too few for a meaningful interval. In `estimate --manifest`, the two Lesjak scans with a 0.9 mm slice spacing are left out automatically: masks whose spacing differs
from the most common one are skipped.

## What the code does

* **Re-slicing** (`reslice.degrade`). Along the slice axis, native voxels (spacing *T0*) are
  grouped into thick slices of thickness *T*. A thick slice is lesion by one of two rules: `nn`,
  if the native voxel at its centre is lesion; or `pv`, if lesion fills at least half of its
  thickness (box profile). Each native voxel then takes the label of the thick slice that contains
  its centre. Results are averaged over 6 equally spaced phases of the grid in `[0, T)`.
* **Pairing** (`lesions.match`). Lesions are 6-connected components. An original lesion and a
  degraded component are linked when they share at least 10 % of the smaller of the two; linked
  components form groups. The Dice of a lesion is computed against the union of the degraded
  components of its group. A lesion with no link is lost and counts as 0.
* **Depth classes**: extent along the slice axis of 1, 2, 3, 4, 5, 6–8 and 9+ slices. The depth
  *d* of a class is its number of slices × *T0*, 7 × *T0* for 6–8, and the median depth of the
  lesions of the cohort for 9+ (`--tail`).
* **Relation** (`relation.Cohort`). For each class and thickness, the per-scan class means are
  averaged over scans. *c*, *a* and λ are fitted on classes of 2 slices or more and on *T* > *T0*.
  The 95 % intervals come from a bootstrap of patients (1000 resamples).
* **Run-length model** (`runs`). Each run of a lesion (start *s_r*, length *L_r*) loses according
  to tabulated overlaps *I(L, t, ψ)* and degraded lengths *B(L, t, ψ)*, with ψ = (φ − s_r) mod *t*.
  The lesion's Dice is 2ΣI / Σ(L + B), averaged over 240 phases. `estimate_lambda` fits
  *R* = *a* e^(−λ(T−T0)/d) to the predicted curves of all lesions together, with *c* = 0.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## License

MIT, see `LICENSE`.
