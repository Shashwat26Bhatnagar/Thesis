"""
Per-seed sample efficiency: dots for each batch, running average, running max.
Nothing else.

    cd ~/Thesis/sindy-rl
    python -u latent_repr/latent_plotly_per_seed.py \
        --exp-dir ray_results/pensim_sindy_latent/dyna_pensim_latent_d4 \
        --out-dir analysis/figs \
        --filter 1a500
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise SystemExit('plotly is required: pip install plotly')

COL_REAL = 'traj_buffer/last_on_policy_ep_rew'
COL_NTRAJ = 'traj_buffer/n_traj_on_pi'

PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']


def load_per_seed(
    exp_dir: str,
    trial_filter: str | None = None
) -> dict[str, pd.DataFrame]:
    seeds = {}

    for path in sorted(
        glob.glob(
            os.path.join(
                os.path.expanduser(exp_dir),
                '*',
                'progress.csv'
            )
        )
    ):
        # If a filter was supplied, only load matching trial directories.
        if trial_filter is not None and trial_filter not in path:
            continue

        name = os.path.basename(os.path.dirname(path))

        parts = name.split('_')
        seed = parts[3] if len(parts) > 3 else name

        df = pd.read_csv(path)

        if COL_REAL not in df:
            continue

        # Extract every row where a real yield is reported.
        # The same yield is repeated across iterations until the next
        # collection, so detect transitions in n_traj_on_pi to find
        # the actual collection points.
        yields = []

        if COL_NTRAJ in df:
            prev_n = -1

            for idx, row in df.iterrows():
                n = row.get(COL_NTRAJ, np.nan)
                y = row.get(COL_REAL, np.nan)

                if pd.notna(n) and pd.notna(y) and n > prev_n:
                    yields.append(float(y))
                    prev_n = n

        else:
            # Fallback: take every non-NaN value and deduplicate
            # consecutive repeated yields.
            prev_y = None

            for y in df[COL_REAL]:
                if pd.notna(y) and y != prev_y:
                    yields.append(float(y))
                    prev_y = y

        if not yields:
            print(f'  seed {seed}: no real rollouts found')
            continue

        out = pd.DataFrame({
            'batch': np.arange(1, len(yields) + 1),
            'yield': yields,
        })

        out['avg_so_far'] = out['yield'].expanding().mean()
        out['max_so_far'] = out['yield'].cummax()

        seeds[seed] = out

        print(
            f'  seed {seed}: {len(yields)} batches, '
            f'mean {np.mean(yields):.1f}, '
            f'best {np.max(yields):.1f}'
        )

    if not seeds:
        if trial_filter is not None:
            raise FileNotFoundError(
                f'no real rollouts under {exp_dir} '
                f'with filter "{trial_filter}"'
            )
        raise FileNotFoundError(
            f'no real rollouts under {exp_dir}'
        )

    return seeds


def build_figure(seeds: dict[str, pd.DataFrame]) -> go.Figure:
    n = len(seeds)
    titles = [f'Seed {s}' for s in seeds.keys()]

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        subplot_titles=titles
    )

    for i, (seed, df) in enumerate(seeds.items(), start=1):
        col = PALETTE[(i - 1) % len(PALETTE)]

        # Dots for each batch.
        fig.add_trace(
            go.Scatter(
                x=df['batch'],
                y=df['yield'],
                mode='markers',
                marker=dict(
                    size=9,
                    color=col,
                    symbol='star',
                    line=dict(width=0.5, color='white')
                ),
                name='batch yield',
                showlegend=(i == 1),
                legendgroup='dots',
                hovertemplate=(
                    'batch %{x}<br>'
                    'yield %{y:.1f}'
                    '<extra></extra>'
                )
            ),
            row=i,
            col=1
        )

        # Running average.
        fig.add_trace(
            go.Scatter(
                x=df['batch'],
                y=df['avg_so_far'],
                mode='lines',
                line=dict(
                    color='darkcyan',
                    width=2.5,
                    dash='dash'
                ),
                name='average yield so far',
                showlegend=(i == 1),
                legendgroup='avg'
            ),
            row=i,
            col=1
        )

        # Running maximum.
        fig.add_trace(
            go.Scatter(
                x=df['batch'],
                y=df['max_so_far'],
                mode='lines',
                line=dict(
                    color='crimson',
                    width=2.5
                ),
                name='max yield so far',
                showlegend=(i == 1),
                legendgroup='max'
            ),
            row=i,
            col=1
        )

        fig.update_yaxes(
            title_text='yield [kg]',
            row=i,
            col=1
        )

        fig.update_xaxes(
            title_text='real batches collected',
            row=i,
            col=1
        )

    fig.update_layout(
        title='Per-seed learning curves',
        template='plotly_white',
        height=280 * n + 80,
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.08
        )
    )

    return fig


def write(fig, path):
    fig.write_html(
        path,
        include_plotlyjs='cdn'
    )

    print(f'  -> {path}')

    try:
        png = path.replace('.html', '.png')

        fig.write_image(
            png,
            width=1100,
            height=max(700, 280 * 4),
            scale=2
        )

        print(f'  -> {png}')

    except Exception:
        print('     (PNG skipped -- pip install kaleido)')


def main(argv=None):
    p = argparse.ArgumentParser(
        'per-seed sample efficiency'
    )

    p.add_argument(
        '--exp-dir',
        required=True
    )

    p.add_argument(
        '--out-dir',
        default='analysis/figs'
    )

    p.add_argument(
        '--filter',
        default=None,
        help='only use trial directories containing this string'
    )

    args = p.parse_args(argv)

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    print('loading:')

    if args.filter is not None:
        print(f'filter: {args.filter}')

    seeds = load_per_seed(
        args.exp_dir,
        trial_filter=args.filter
    )

    write(
        build_figure(seeds),
        os.path.join(
            args.out_dir,
            'sample_efficiency_per_seed.html'
        )
    )


if __name__ == '__main__':
    main()
