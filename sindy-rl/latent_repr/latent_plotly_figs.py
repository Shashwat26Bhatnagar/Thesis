"""
Plotly figures for the SINDy-RL results.

    cd ~/Thesis/sindy-rl
    python -u latent_repr/latent_plotly_figs.py \
        --exp-dir ray_results/pensim_sindy_latent/dyna_pensim_latent_d4 \
        --out-dir analysis/figs

Produces two interactive HTML figures (plus PNG if kaleido is installed):

  sample_efficiency.html
      Laid out like the Bayesian-Optimisation batch plot: yield against
      batch id, with running-best and running-average traces. The point of
      the figure is the x-axis -- BO needed ~1010 real batches, this needed
      the batches shown here. Use --bo-csv to overlay the BO run directly.

  extrapolation.html
      How far the learned world model can predict before it diverges from
      the truth. Rolls the model open-loop from a real initial state under
      the real action sequence and plots per-step error against a
      persistence baseline, with the ensemble spread as a band.

On the reference line: the shipped gpei_batch_*.csv files average 3729,
but the full BO run reaches ~4080. --reference sets which one is drawn, so
be explicit about which claim the figure supports.
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
    raise SystemExit('plotly is required:  pip install plotly --no-deps')

GPEI_SHIPPED = 3729.0
RECIPE = 3627.9
MCPILCO_BND = 3260.3

COL_REAL = 'traj_buffer/last_on_policy_ep_rew'
COL_NREAL = 'traj_buffer/n_total_real'
COL_NTRAJ = 'traj_buffer/n_traj_on_pi'
COL_REW = 'sampler_results/episode_reward_mean'
COL_LEN = 'sampler_results/episode_len_mean'


# ----------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------

def load_real_rollouts(exp_dir: str) -> pd.DataFrame:
    """Every distinct real on-policy batch, across all seeds."""
    rows = []
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(exp_dir),
                                              '*', 'progress.csv'))):
        name = os.path.basename(os.path.dirname(path))
        parts = name.split('_')
        seed = parts[3] if len(parts) > 3 else name
        df = pd.read_csv(path)
        if COL_REAL not in df:
            continue
        m = df[COL_REAL].notna()
        sub = df.loc[m, [COL_REAL] +
                     [c for c in (COL_NREAL, COL_NTRAJ) if c in df]].copy()
        # The same rollout is reported on every iteration until the next
        # collection, so keep one row per distinct (yield, n_traj) pair.
        key = [COL_REAL] + ([COL_NTRAJ] if COL_NTRAJ in sub else [])
        sub = sub.drop_duplicates(subset=key)
        sub['seed'] = seed
        sub['iteration'] = sub.index
        rows.append(sub)
    if not rows:
        raise FileNotFoundError(
            f'no real rollouts found under {exp_dir}/*/progress.csv')
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={COL_REAL: 'yield'})
    return out


# ----------------------------------------------------------------------
# figure 1 -- sample efficiency, laid out like the BO plot
# ----------------------------------------------------------------------

def fig_sample_efficiency(roll: pd.DataFrame,
                          reference: float,
                          reference_label: str,
                          bo_csv: str | None,
                          log_x: bool) -> go.Figure:
    roll = roll.sort_values(['seed', 'iteration']).reset_index(drop=True)
    roll['batch_id'] = np.arange(1, len(roll) + 1)
    roll['best_so_far'] = roll['yield'].cummax()
    roll['avg_so_far'] = roll['yield'].expanding().mean()

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['SINDy-RL: yield per real fermentation batch'])

    # Optional BO overlay -- the whole point of the comparison.
    if bo_csv and os.path.exists(os.path.expanduser(bo_csv)):
        bo = pd.read_csv(os.path.expanduser(bo_csv))
        ycol = next((c for c in bo.columns
                     if 'yield' in c.lower()), bo.columns[-1])
        bo_id = np.arange(1, len(bo) + 1)
        fig.add_trace(go.Scatter(
            x=bo_id, y=bo[ycol], mode='markers',
            marker=dict(size=3, color='rgba(70,130,220,0.35)',
                        symbol='star'),
            name=f'BO batches (n={len(bo)})',
            hovertemplate='batch %{x}<br>yield %{y:.1f}<extra></extra>'))
        fig.add_trace(go.Scatter(
            x=bo_id, y=pd.Series(bo[ycol]).cummax(), mode='lines',
            line=dict(color='rgba(200,60,60,0.8)', width=2),
            name='BO best so far'))

    # Per-seed markers.
    palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
               '#ff7f0e', '#8c564b']
    for i, (seed, g) in enumerate(roll.groupby('seed')):
        fig.add_trace(go.Scatter(
            x=g['batch_id'], y=g['yield'], mode='markers',
            marker=dict(size=13, color=palette[i % len(palette)],
                        symbol='star',
                        line=dict(width=1, color='white')),
            name=f'SINDy-RL seed {seed}',
            hovertemplate=('batch %{x}<br>yield %{y:.1f}'
                           '<br>iter %{customdata}<extra></extra>'),
            customdata=g['iteration']))

    fig.add_trace(go.Scatter(
        x=roll['batch_id'], y=roll['best_so_far'], mode='lines',
        line=dict(color='crimson', width=2.5),
        name='SINDy-RL best so far'))
    fig.add_trace(go.Scatter(
        x=roll['batch_id'], y=roll['avg_so_far'], mode='lines',
        line=dict(color='darkcyan', width=2, dash='dash'),
        name='SINDy-RL avg so far'))

    for y, lab, col in [
            (reference, reference_label, 'green'),
            (RECIPE, f'default recipe ({RECIPE:.0f})', 'royalblue'),
            (MCPILCO_BND, f'MC-PILCO ±10% ({MCPILCO_BND:.0f})', 'darkorange')]:
        fig.add_hline(y=y, line=dict(color=col, width=1.5, dash='dot'),
                      annotation_text=lab,
                      annotation_position='right',
                      annotation_font_size=10)

    n = len(roll)
    fig.update_layout(
        title=dict(
            text=('Sample efficiency: yield vs number of real batches<br>'
                  f'<sub>SINDy-RL used {n} real batches '
                  f'(plus 10 pre-existing offline batches). '
                  f'Bayesian Optimisation used ~1010.</sub>'),
            x=0.02),
        xaxis_title='real fermentation batch id',
        yaxis_title='total yield [kg]',
        template='plotly_white',
        height=560,
        hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=-0.28))
    if log_x:
        fig.update_xaxes(type='log',
                         title='real fermentation batch id (log scale)')
    return fig


# ----------------------------------------------------------------------
# figure 2 -- world-model extrapolation
# ----------------------------------------------------------------------

def fig_extrapolation(buffer_path: str,
                      config_path: str,
                      horizon: int,
                      n_members: int) -> go.Figure:
    """
    Roll the learned dynamics open-loop under the true action sequence and
    compare against the true latent trajectory.

    Persistence (predict z_{t+1} = z_t) is drawn alongside because over a
    12-minute step the state barely moves, so a model can post a small
    absolute error while having learned nothing. The point at which the
    model's error crosses persistence is where its predictions stop being
    worth anything.
    """
    import yaml
    from sindy_rl.dynamics import EnsembleSINDyDynamicsModel
    from sindy_rl.traj_buffer import BaseTrajectoryBuffer

    with open(os.path.expanduser(config_path)) as f:
        cfg = yaml.safe_load(f)

    buf = BaseTrajectoryBuffer()
    buf.load_data(os.path.expanduser(buffer_path))
    X, U, _ = buf.to_list()

    # Hold out the last trajectory: fitting and testing on the same data
    # would make the extrapolation look far better than it is.
    model = EnsembleSINDyDynamicsModel(cfg['dynamics_model']['config'])
    model.fit(X[:-1], U[:-1])
    x_true, u_true = X[-1], U[-1]
    H = int(min(horizon, len(x_true) - 1))

    def rollout(setter):
        setter()
        z, traj = x_true[0].copy(), [x_true[0].copy()]
        for t in range(H):
            try:
                z = model.predict(z, u_true[t])
            except Exception:
                break
            if not np.all(np.isfinite(z)) or np.abs(z).max() > 1e6:
                break
            traj.append(z.copy())
        return np.asarray(traj)

    med = rollout(model.set_median_coef_)

    members = []
    for idx in range(n_members):
        try:
            traj = rollout(lambda i=idx: model.set_idx_coef_(i))
        except Exception:
            break
        members.append(traj)

    steps = np.arange(H + 1)
    truth = x_true[:H + 1]
    persist = np.tile(x_true[0], (H + 1, 1))

    def err(a):
        n = min(len(a), len(truth))
        e = np.full(H + 1, np.nan)
        e[:n] = np.linalg.norm(a[:n] - truth[:n], axis=1)
        return e

    e_med, e_per = err(med), err(persist)
    e_mem = np.vstack([err(m) for m in members]) if members else None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=['Open-loop prediction error vs a persistence baseline',
                        'Ensemble disagreement (uncertainty)'])

    if e_mem is not None:
        lo = np.nanpercentile(e_mem, 10, axis=0)
        hi = np.nanpercentile(e_mem, 90, axis=0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill='toself', fillcolor='rgba(31,119,180,0.18)',
            line=dict(width=0), name='ensemble 10–90%',
            hoverinfo='skip'), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=e_med, mode='lines',
        line=dict(color='#1f77b4', width=2.5),
        name='SINDy median model'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=e_per, mode='lines',
        line=dict(color='grey', width=2, dash='dash'),
        name='persistence (z_{t+1}=z_t)'), row=1, col=1)

    # Where the model stops beating persistence.
    worse = np.where(e_med > e_per)[0]
    if len(worse):
        k = int(worse[0])
        fig.add_vline(x=k, line=dict(color='crimson', width=1.5, dash='dot'),
                      annotation_text=f'crosses persistence at step {k}',
                      annotation_position='top right',
                      annotation_font_size=10, row=1, col=1)

    if e_mem is not None:
        spread = np.nanstd(e_mem, axis=0)
        fig.add_trace(go.Scatter(
            x=steps, y=spread, mode='lines',
            line=dict(color='#d62728', width=2),
            name='std across ensemble members'), row=2, col=1)
        n_div = int(sum(1 for m in members if len(m) < H + 1))
        fig.add_annotation(
            text=f'{n_div} of {len(members)} members diverged before step {H}',
            xref='paper', yref='paper', x=0.02, y=-0.16, showarrow=False,
            font=dict(size=11, color='dimgrey'))

    fig.update_xaxes(title_text='open-loop prediction step (0.2 h each)',
                     row=2, col=1)
    fig.update_yaxes(title_text='‖predicted − true‖ (latent)', row=1, col=1)
    fig.update_yaxes(title_text='ensemble std', row=2, col=1)
    fig.update_layout(
        title=dict(text=('World-model extrapolation on a held-out batch<br>'
                         '<sub>How far the learned dynamics stay useful '
                         'before diverging</sub>'), x=0.02),
        template='plotly_white', height=720, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.22))
    return fig


# ----------------------------------------------------------------------

def write(fig, path_html):
    fig.write_html(path_html, include_plotlyjs='cdn')
    print(f'  -> {path_html}')
    try:
        png = path_html.replace('.html', '.png')
        fig.write_image(png, width=1400, height=700, scale=2)
        print(f'  -> {png}')
    except Exception:
        print('     (PNG skipped -- pip install kaleido for static export)')


def main(argv=None):
    p = argparse.ArgumentParser('plotly figures for the SINDy-RL run')
    p.add_argument('--exp-dir', required=True)
    p.add_argument('--out-dir', default='analysis/figs')
    p.add_argument('--reference', type=float, default=GPEI_SHIPPED,
                   help=f'reference yield line (default {GPEI_SHIPPED}, the '
                        f'mean of the shipped gpei CSVs; the full BO run '
                        f'reaches ~4080)')
    p.add_argument('--reference-label', default=None)
    p.add_argument('--bo-csv', default=None,
                   help='optional CSV of BO per-batch yields to overlay')
    p.add_argument('--log-x', action='store_true',
                   help='log x-axis, useful when overlaying 1010 BO batches')
    p.add_argument('--extrapolation', action='store_true',
                   help='also build the extrapolation figure (refits the model)')
    p.add_argument('--buffer', default='data/pensim_offpi_latent_d4.pkl')
    p.add_argument('--config',
                   default='sindy_rl/config_templates/dyna_pensim_latent.yml')
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--n-members', type=int, default=20)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    label = args.reference_label or f'reference ({args.reference:.0f})'

    roll = load_real_rollouts(args.exp_dir)
    print(f'{len(roll)} real batches across {roll["seed"].nunique()} seeds')
    print(f'  mean {roll["yield"].mean():.1f}  '
          f'best {roll["yield"].max():.1f}  '
          f'worst {roll["yield"].min():.1f}')

    print('\nsample efficiency figure:')
    write(fig_sample_efficiency(roll, args.reference, label,
                                args.bo_csv, args.log_x),
          os.path.join(args.out_dir, 'sample_efficiency.html'))

    if args.extrapolation:
        print('\nextrapolation figure (fitting the dynamics model):')
        write(fig_extrapolation(args.buffer, args.config,
                                args.horizon, args.n_members),
              os.path.join(args.out_dir, 'extrapolation.html'))

    roll.to_csv(os.path.join(args.out_dir, 'real_batches.csv'), index=False)
    print(f'\ndata -> {args.out_dir}/real_batches.csv')


if __name__ == '__main__':
    main()
