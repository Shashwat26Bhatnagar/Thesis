#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse_ablation.py   (repo root)

Paired analysis of the one-model vs three-phase-model ablation.

    python analyse_ablation.py
    python analyse_ablation.py -res results_ablation

THE ENDPOINT is total yield collected in the REAL simulator, not the training loss.
Every objective tried in this project has moved its own loss without moving yield, so
the loss is not a proxy for the thing being claimed.

THE TEST IS PAIRED. Seed s gives the same policy initialisation, window draw and
particle sampling in both arms, so arm A and arm B at seed s are matched observations
rather than independent samples. A paired test removes the seed-to-seed variance, which
in this project has been large enough to swamp the differences between methods --
across configurations, yield has ranged 2728-3486 with no method reliably separated
from another. Both the parametric (paired t) and the distribution-free (Wilcoxon
signed-rank) results are reported; if they disagree, trust the latter, since n is small
and the differences need not be normal.

EPISODES THAT TERMINATE EARLY ARE NOT DROPPED. A policy that drains the vessel at
121 h has a low yield BECAUSE it failed, and excluding it would flatter the arm that
fails more often. Completion rate is reported alongside.
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-res", default="results_ablation")
ap.add_argument("-data",
                default=os.path.expanduser(
                    "~/Thesis/deps/smpl/smpl/configdata/pensim_abl"))
ap.add_argument("-gpei", type=float, default=3835.0, help="reference yield")
args = ap.parse_args()


def collect(arm):
    """{seed: [yield per episode]} from the written CSVs."""
    out = {}
    for f in sorted(glob.glob(os.path.join(args.data, f"{arm}_s*_e*_batch_*.csv"))):
        m = re.search(rf"{arm}_s(\d+)_e(\d+)_batch", os.path.basename(f))
        if not m:
            continue
        s = int(m.group(1))
        try:
            d = np.genfromtxt(f, delimiter=",", skip_header=1)
            d = d[~np.isnan(d).any(axis=1)]
            if len(d) < 10:
                continue
            out.setdefault(s, []).append(
                (float(d[:, -1].sum()), float(d[-1, 0]), len(d)))
        except Exception:
            continue
    return out


A, B = collect("A"), collect("B")
seeds = sorted(set(A) & set(B))
if not seeds:
    raise SystemExit(f"no paired seeds found in {args.data} -- have the jobs finished?")

print(f"paired seeds: {len(seeds)}   "
      f"(arm A only: {sorted(set(A)-set(B))}, arm B only: {sorted(set(B)-set(A))})")
print(f"\n{'seed':>5}{'A: 3 models':>14}{'B: 1 model':>13}{'A - B':>10}"
      f"{'A t_end':>10}{'B t_end':>10}")

dA, dB, diff = [], [], []
compA = compB = nA = nB = 0
for s in seeds:
    ya = np.mean([y for y, _, _ in A[s]]); yb = np.mean([y for y, _, _ in B[s]])
    ta = np.mean([t for _, t, _ in A[s]]); tb = np.mean([t for _, t, _ in B[s]])
    compA += sum(1 for _, t, _ in A[s] if t > 225); nA += len(A[s])
    compB += sum(1 for _, t, _ in B[s] if t > 225); nB += len(B[s])
    dA.append(ya); dB.append(yb); diff.append(ya - yb)
    print(f"{s:5d}{ya:14.1f}{yb:13.1f}{ya-yb:+10.1f}{ta:10.1f}{tb:10.1f}")

dA, dB, diff = map(np.array, (dA, dB, diff))
n = len(diff)
print(f"\n{'':22}{'arm A (3 models)':>18}{'arm B (1 model)':>18}")
print(f"{'mean yield':22}{dA.mean():18.1f}{dB.mean():18.1f}")
print(f"{'std across seeds':22}{dA.std(ddof=1):18.1f}{dB.std(ddof=1):18.1f}")
print(f"{'% of reference':22}{100*dA.mean()/args.gpei:17.1f}%{100*dB.mean()/args.gpei:17.1f}%")
print(f"{'episodes completed':22}{f'{compA}/{nA}':>18}{f'{compB}/{nB}':>18}")

md, sd = diff.mean(), diff.std(ddof=1)
se = sd / np.sqrt(n)
print(f"\npaired difference (A - B): mean {md:+.1f}, sd {sd:.1f}, se {se:.1f}")
print(f"  arm A better in {int((diff > 0).sum())}/{n} seeds")

if se > 0:
    t = md / se
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
        lo, hi = md - stats.t.ppf(0.975, n - 1) * se, md + stats.t.ppf(0.975, n - 1) * se
        w = stats.wilcoxon(dA, dB)
        print(f"  paired t({n-1}) = {t:+.3f},  p = {p:.4f}")
        print(f"  95% CI for the difference: [{lo:+.1f}, {hi:+.1f}]")
        print(f"  Wilcoxon signed-rank: W = {w.statistic:.1f}, p = {w.pvalue:.4f}")
        print(f"  Cohen's d (paired) = {md/sd:+.3f}")
        print()
        if p < 0.05:
            print(f"  -> the arms differ (p < 0.05). The three-model arm is "
                  f"{'better' if md > 0 else 'WORSE'} by {abs(md):.0f} yield on "
                  f"average, {100*abs(md)/dB.mean():.1f}% of the single-model arm.")
        else:
            print(f"  -> no detectable difference at n = {n}. The CI spans "
                  f"[{lo:+.0f}, {hi:+.0f}], so an effect larger than "
                  f"{max(abs(lo), abs(hi)):.0f} yield is ruled out but smaller ones "
                  f"are not. Note this is a NULL result, not evidence the arms are "
                  f"identical.")
    except ImportError:
        print(f"  paired t({n-1}) = {t:+.3f}   (scipy unavailable for p-values)")

print(f"\nfor reference: gpei {args.gpei:.0f}; every configuration tried in this "
      f"project has landed in 2728-3486.")
