#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import glob
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-root", default=os.path.expanduser(
    "~/Thesis/deps/smpl/smpl/configdata"))
ap.add_argument("-pattern", default="pensim_rlloop_s*",
                help="glob for the per-seed data folders")
ap.add_argument("-tag", default="rl_iter",
                help="CSV prefix; files are <tag><k>_batch_<n>.csv")
ap.add_argument("-out", default="figs/sample_efficiency_per_seed.html")
ap.add_argument("-csv", default="figs/yields_per_seed.csv")
ap.add_argument("-ref", type=float, default=3835.0,
                help="reference controller yield, drawn as a dashed line")
ap.add_argument("-ref_name", default="gpei reference")
args = ap.parse_args()


def read_yield(path):
    """Total yield and end time from one episode CSV.

    Appended STD / Mean summary rows and any NaN rows are dropped -- some files in
    this project carry them, and summing them would inflate the total.
    """
    try:
        d = np.genfromtxt(path, delimiter=",", skip_header=1)
    except Exception:
        return None
    if d.ndim != 2 or len(d) < 10:
        return None
    d = d[~np.isnan(d).any(axis=1)]
    if len(d) < 10:
        return None
    return float(d[:, -1].sum()), float(d[-1, 0]), len(d)


folders = sorted(glob.glob(os.path.join(args.root, args.pattern)))
if not folders:
    raise SystemExit(f"no folders match {args.pattern} under {args.root}")

SEEDS = {}
for fo in folders:
    m = re.search(r"_s(\d+)$", os.path.basename(fo))
    seed = int(m.group(1)) if m else os.path.basename(fo)
    rows = []
    for f in glob.glob(os.path.join(fo, f"{args.tag}*_batch_*.csv")):
        k = re.search(rf"{args.tag}(\d+)_batch_(\d+)", os.path.basename(f))
        r = read_yield(f)
        if k and r:
            rows.append((int(k.group(1)), int(k.group(2)), *r))
    if rows:
        rows.sort()
        SEEDS[seed] = rows

if not SEEDS:
    raise SystemExit(f"no '{args.tag}*_batch_*.csv' files found in {folders}")

print(f"{'seed':>6}{'batches':>9}{'first':>10}{'last':>10}{'mean':>10}"
      f"{'best':>10}{'complete':>10}")
for s in sorted(SEEDS):
    y = np.array([r[2] for r in SEEDS[s]])
    t = np.array([r[3] for r in SEEDS[s]])
    print(f"{s:6}{len(y):9d}{y[0]:10.1f}{y[-1]:10.1f}{y.mean():10.1f}"
          f"{y.max():10.1f}{f'{int((t>225).sum())}/{len(t)}':>10}")

os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
with open(args.csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["seed", "iteration", "batch", "batch_index", "yield",
                "t_end_h", "n_rows", "running_mean", "running_max"])
    for s in sorted(SEEDS):
        ys = []
        for i, (it, b, y, t, n) in enumerate(SEEDS[s]):
            ys.append(y)
            w.writerow([s, it, b, i + 1, f"{y:.4f}", f"{t:.1f}", n,
                        f"{np.mean(ys):.4f}", f"{np.max(ys):.4f}"])
print(f"\nsaved -> {args.csv}")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise SystemExit("plotly not installed:  pip install plotly --no-deps")

order = sorted(SEEDS)
fig = make_subplots(rows=len(order), cols=1, shared_xaxes=False,
                    vertical_spacing=0.06,
                    subplot_titles=[f"Seed {s:05d}" if isinstance(s, int)
                                    else f"Seed {s}" for s in order])

for r, s in enumerate(order, start=1):
    y = np.array([e[2] for e in SEEDS[s]])
    t = np.array([e[3] for e in SEEDS[s]])
    it = [e[0] for e in SEEDS[s]]
    x = np.arange(1, len(y) + 1)
    rmean = np.array([y[:i + 1].mean() for i in range(len(y))])
    rmax = np.maximum.accumulate(y)
    hover = [f"iter {a}<br>yield {b:.1f}<br>t_end {c:.1f} h"
             for a, b, c in zip(it, y, t)]
    show = (r == 1)

    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="batch yield",
                             marker=dict(size=7, color="#7F77DD"),
                             text=hover, hoverinfo="text",
                             legendgroup="b", showlegend=show), row=r, col=1)
    fig.add_trace(go.Scatter(x=x, y=rmean, mode="lines", name="average yield so far",
                             line=dict(width=2, color="#1D9E75"),
                             legendgroup="m", showlegend=show), row=r, col=1)
    fig.add_trace(go.Scatter(x=x, y=rmax, mode="lines", name="max yield so far",
                             line=dict(width=2, dash="dot", color="#D85A30"),
                             legendgroup="x", showlegend=show), row=r, col=1)
    if args.ref:
        fig.add_hline(y=args.ref, line=dict(width=1, dash="dash", color="#888780"),
                      row=r, col=1,
                      annotation_text=args.ref_name if show else None,
                      annotation_position="top right",
                      annotation_font=dict(size=11, color="#888780"))
    fig.update_xaxes(title_text="real batches collected", row=r, col=1)
    fig.update_yaxes(title_text="yield [kg]", row=r, col=1)

fig.update_layout(title_text="Per-seed learning curves",
                  height=300 * len(order), width=900,
                  template="plotly_white", hovermode="closest",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02,
                              xanchor="right", x=1))
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.write_html(args.out, include_plotlyjs="cdn")
print(f"saved -> {args.out}")

allmeans = np.array([np.mean([e[2] for e in SEEDS[s]]) for s in order])
print(f"\nacross seeds: mean {allmeans.mean():.1f}, "
      f"sd {allmeans.std(ddof=1) if len(allmeans) > 1 else 0:.1f}, "
      f"range {allmeans.min():.1f}-{allmeans.max():.1f}")
print("  if that spread is comparable to the differences between methods measured")
print("  earlier (2728-3486 across configurations), those differences were variance.")
