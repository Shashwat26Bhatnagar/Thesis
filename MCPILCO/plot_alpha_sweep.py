#!/usr/bin/env python3
import argparse
import glob
import re
import sys

VALUE_PAT = re.compile(r"Finished\s+(?:alpha|LR)\s*=\s*([\d.]+)")
CHANNEL_PAT = re.compile(
    r"^\s*(\w+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
    re.MULTILINE)
KNOWN_CHANNELS = {"discharge", "sugar", "soilbean", "aeration", "backpressure",
                  "waterinj"}
TIME_POINTS = [10, 50, 90, 130]


def parse_one(path):
    with open(path) as f:
        text = f.read()
    vm = VALUE_PAT.search(text)
    if vm is None:
        print(f"  [skip] {path}: no 'Finished alpha/LR = X' line found "
              f"(job may not have completed)", file=sys.stderr)
        return None
    alpha = float(vm.group(1))

    stds, series = {}, {}
    for m in CHANNEL_PAT.finditer(text):
        name = m.group(1)
        if name not in KNOWN_CHANNELS:
            continue
        vals = [float(m.group(i)) for i in (2, 3, 4, 5)]
        series[name] = vals
        stds[name] = float(m.group(6))

    if not stds:
        print(f"  [skip] {path}: alpha={alpha} found but no channel table "
              f"parsed", file=sys.stderr)
        return None
    return alpha, stds, series, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=["*.out"],
                    help="one or more glob patterns for .out log files")
    ap.add_argument("--out", default="alpha_sweep.png",
                    help="output image path")
    ap.add_argument("--csv", default="alpha_sweep_data.csv",
                    help="also write the parsed data to this CSV")
    args = ap.parse_args()

    paths = []
    for pat in args.glob:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))
    if not paths:
        print(f"No files matched: {args.glob}", file=sys.stderr)
        sys.exit(1)

    print(f"scanning {len(paths)} file(s):")
    rows = []
    for p in paths:
        r = parse_one(p)
        if r is not None:
            alpha, stds, series, src = r
            rows.append((alpha, stds, series, src))
            print(f"  [ok]   {p}: alpha={alpha}  "
                  f"discharge_std={stds.get('discharge', float('nan')):.4f}")

    if not rows:
        print("nothing parsed -- check that the jobs actually finished and "
              "printed the full closing table", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: r[0])

    by_alpha = {}
    for alpha, stds, series, src in rows:
        by_alpha[alpha] = (stds, series, src)
    alphas = sorted(by_alpha.keys())
    channels = sorted(KNOWN_CHANNELS)

    import csv as csvmod
    with open(args.csv, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(["alpha"] + channels + ["source_file"])
        for a in alphas:
            stds, series, src = by_alpha[a]
            w.writerow([a] + [stds.get(c, "") for c in channels] + [src])
    print(f"\nwrote {args.csv}")

    time_csv = args.csv.replace(".csv", "_by_time.csv")
    with open(time_csv, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(["alpha", "channel", "hour", "value"])
        for a in alphas:
            _, series, _ = by_alpha[a]
            for c in channels:
                for hr, v in zip(TIME_POINTS, series.get(c, [None]*4)):
                    w.writerow([a, c, hr, v])
    print(f"wrote {time_csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for c in channels:
        ys = [by_alpha[a][0].get(c, float("nan")) for a in alphas]
        marker = "o-" if c == "discharge" else ".--"
        lw = 2.5 if c == "discharge" else 1.2
        ax.plot(alphas, ys, marker, label=c, linewidth=lw,
                markersize=8 if c == "discharge" else 5)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (= LR, k fixed at 5, so alpha*k = 5*alpha)")
    ax.set_ylabel("cross-window action-spread std")
    ax.set_title("All channels vs alpha")
    ax.axvline(0.01, color="gray", linestyle=":", linewidth=1, label="repo default (0.01)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sane = [a for a in alphas if a <= 0.1]
    ys = [by_alpha[a][0].get("discharge", float("nan")) for a in sane]
    ax.plot(sane, ys, "o-", color="C0", linewidth=2.5, markersize=9)
    for a, y in zip(sane, ys):
        ax.annotate(f"{a:g}", (a, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.axvline(0.01, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("discharge cross-window std")
    ax.set_title("Discharge only (alpha <= 0.1)")
    ax.grid(alpha=0.3)

    fig.suptitle("Reptile alpha*k sweep (inner_k fixed at 5)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    sane_alphas = [a for a in alphas if a <= 0.1] or alphas
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    cmap = plt.get_cmap("viridis")
    for idx, c in enumerate(channels):
        ax = axes2[idx // 3, idx % 3]
        for j, a in enumerate(sane_alphas):
            _, series, _ = by_alpha[a]
            vals = series.get(c)
            if vals is None:
                continue
            color = cmap(j / max(len(sane_alphas) - 1, 1))
            ax.plot(TIME_POINTS, vals, "o-", color=color,
                    label=f"alpha={a:g}", linewidth=2, markersize=6)
        ax.set_title(c)
        ax.set_xlabel("time (h)")
        ax.set_ylabel("mean action (z-units)")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")
    fig2.suptitle("Action value vs time, per channel, one line per alpha "
                  "(flat line = no state/time-dependence at that alpha)",
                  fontsize=12)
    fig2.tight_layout()
    time_png = args.out.replace(".png", "_by_time.png")
    fig2.savefig(time_png, dpi=150)
    print(f"wrote {time_png}")

    best_a = max(alphas, key=lambda a: by_alpha[a][0].get("discharge", -1))
    print(f"\nbest discharge std: alpha={best_a}  "
          f"std={by_alpha[best_a][0]['discharge']:.4f}")


if __name__ == "__main__":
    main()
