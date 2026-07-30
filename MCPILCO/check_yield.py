#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent verification of the total-yield numbers.

Deliberately re-derives everything from the raw CSVs rather than trusting the
analysis script: it checks the yield column by NAME (not by position), reports every
file individually, and flags anything that would silently distort a mean --
short/aborted runs, negative error_reward rows, unequal trajectory lengths.

    python check_yield.py
    python check_yield.py -v          # also print a per-phase breakdown
"""
import argparse
import csv
import glob
import os
import numpy as np

PENSIMENV = "/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensimenv"
PENSIM = "/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensim"

GROUPS = [("random (recipe)", os.path.join(PENSIMENV, "random_batch_*.csv")),
          ("gpei (BayesOpt)", os.path.join(PENSIMENV, "gpei_batch_*.csv")),
          ("CDIL unbounded",  os.path.join(PENSIM,    "cdil_batch_*.csv")),
          ("CDIL +/-10%",     os.path.join(PENSIM,    "cdil10pct_batch_*.csv"))]

YIELD_COL_NAME = "Yield Per Step"
TIME_COL_NAME = "Time Step"
ERROR_REWARD = -100.0
FULL_RUN_STEPS = 1150

ap = argparse.ArgumentParser()
ap.add_argument("-v", action="store_true", help="per-phase breakdown")
args = ap.parse_args()


def read(path):
    """Return (t, yield_col, n_rows, header) -- yield located BY NAME."""
    with open(path) as f:
        header = next(csv.reader(f))
    header = [h.strip() for h in header]
    if YIELD_COL_NAME not in header:
        raise KeyError(f"{os.path.basename(path)}: no '{YIELD_COL_NAME}' column; "
                       f"header = {header}")
    yi, ti = header.index(YIELD_COL_NAME), header.index(TIME_COL_NAME)
    d = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1))
    if d.size == 0:
        return np.array([]), np.array([]), 0, header
    return d[:, ti], d[:, yi], d.shape[0], header


print("=" * 92)
print("PER-FILE TOTAL YIELD   (sum of the ENTIRE 'Yield Per Step' column = one full batch)")
print("=" * 92)

results, warnings = {}, []
for name, pattern in GROUPS:
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(os.path.basename(p).split("_")[-1][:-4]))
    if not files:
        print(f"\n{name}: NO FILES matching {pattern}")
        continue
    print(f"\n{name}   ({len(files)} files found)")
    print(f"  {'file':26s}{'rows':>7}{'t_end(h)':>10}{'total yield':>14}"
          f"{'neg rows':>10}  note")
    good = []
    for p in files:
        t, y, n, hdr = read(p)
        if n == 0:
            print(f"  {os.path.basename(p):26s}{'EMPTY':>7}"); continue
        tot = float(y.sum())
        nneg = int((y <= ERROR_REWARD).sum())
        note = ""
        if n < FULL_RUN_STEPS:
            note = f"PARTIAL ({n}/{FULL_RUN_STEPS}) -- EXCLUDED"
            warnings.append(f"{os.path.basename(p)}: only {n} rows")
        elif nneg:
            note = f"{nneg} error_reward row(s)"
        else:
            good.append(tot)
        print(f"  {os.path.basename(p):26s}{n:>7}{t[-1]:>10.1f}{tot:>14.2f}"
              f"{nneg:>10}  {note}")
    if good:
        results[name] = np.array(good)

print("\n" + "=" * 92)
print("SUMMARY  (full-length runs only)")
print("=" * 92)
print(f"{'dataset':22s}{'n':>4}{'mean':>12}{'std':>10}{'min':>12}{'max':>12}{'% of gpei':>12}")
print("-" * 92)
ref = results.get("gpei (BayesOpt)")
ref_m = ref.mean() if ref is not None else None
for name, y in results.items():
    pct = f"{100*y.mean()/ref_m:11.1f}%" if ref_m else "          -"
    print(f"{name:22s}{len(y):>4}{y.mean():>12.1f}{y.std():>10.1f}"
          f"{y.min():>12.1f}{y.max():>12.1f}{pct}")

if args.v:
    print("\n" + "=" * 92)
    print("PER-PHASE YIELD  (mean over full-length runs; phases in hours)")
    print("=" * 92)
    print(f"{'dataset':22s}{'lag 0-35':>14}{'trans 35-51':>14}{'prod 51+':>14}{'total':>14}")
    print("-" * 92)
    for name, pattern in GROUPS:
        rows = []
        for p in sorted(glob.glob(pattern)):
            t, y, n, _ = read(p)
            if n < FULL_RUN_STEPS:
                continue
            rows.append([y[(t >= lo) & (t < hi)].sum()
                         for lo, hi in [(0, 35), (35, 51), (51, 1e9)]])
        if rows:
            m = np.mean(rows, axis=0)
            print(f"{name:22s}{m[0]:>14.1f}{m[1]:>14.1f}{m[2]:>14.1f}{m.sum():>14.1f}")

print("\n" + "=" * 92)
if warnings:
    print(f"WARNINGS -- {len(warnings)} file(s) excluded as partial runs:")
    for w in warnings[:20]:
        print(f"  {w}")
else:
    print("no partial runs found")
print("\nSANITY CHECKS")
print(f"  * yield located by column NAME ('{YIELD_COL_NAME}'), not by index")
print(f"  * only runs with the full {FULL_RUN_STEPS} steps are included in the summary")
print(f"  * rows at error_reward ({ERROR_REWARD}) are counted and reported")
print("  * each 'total yield' is the sum over an ENTIRE trajectory, not a timestep")

