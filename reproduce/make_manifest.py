#!/usr/bin/env python
"""Write a manifest (scan_id, patient, mask, axis) from a glob pattern.

The scan identifier is the file name without extension; the patient identifier is
the first group of ``--patient-regex`` applied to the path.

    python reproduce/make_manifest.py "MSLesSeg/**/*_MASK.nii.gz" --patient-regex "(P\\d+)_T\\d+" \\
        --out mslesseg.csv
"""
import argparse
import csv
import glob
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob pattern of the masks (quote it; ** is recursive)")
    ap.add_argument("--patient-regex", default=r"(.+)", help="regex with one group for the patient")
    ap.add_argument("--axis", default="si", help="si, coarsest or an axis index")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.pattern, recursive=True))
    if not paths:
        sys.exit("no file matches %s" % a.pattern)
    rx = re.compile(a.patient_regex)
    rows = []
    for p in paths:
        m = rx.search(p)
        if not m:
            sys.exit("patient regex does not match %s" % p)
        scan = os.path.basename(p).split(".nii")[0]
        rows.append(dict(scan_id=scan, patient=m.group(1), mask=os.path.abspath(p), axis=a.axis))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scan_id", "patient", "mask", "axis"])
        w.writeheader()
        w.writerows(rows)
    print("%d masks, %d patients -> %s" % (len(rows), len({r["patient"] for r in rows}), a.out))


if __name__ == "__main__":
    main()
