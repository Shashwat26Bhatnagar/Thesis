# FL-GFN denoiser over the dcFBA state space

Trains a Forward-Looking GFlowNet (Pan, Malkin, Zhang, Bengio 2023) whose
per-transition energy is the LOCAL DIVERGENCE of the dcFBA flow.

## The idea
A Gaussian p(s,0) over (G,X,V) flows under the dcFBA field f(s) via the
continuity equation  dp/dt + div(p f)=0.  Along a trajectory the log-density
obeys  d eta/dt = -div f  (eta=log p), so eta is ADDITIVE along the path.
That additivity is exactly FL-GFN's additive-energy assumption (Eq.6):
    E(s) = -eta(s),   reward R = e^{-E} = p,   E(s->s') = +div f(s) * dt.
The forward-looking flow F~ then carries dense per-step credit (the divergence),
instead of a single terminal reward.

## Files
- flgfn_denoiser.py   build field tables -> flow Gaussian -> train FL-DB -> roll out + save
- plot_denoiser.py    figure: Gaussian transport + additive credit
- flgfn_denoiser.pt   trained model (P_F drift = forward denoiser, P_B, logF~)
- field_tables.npz    mu(S), vglu(S) 1-D tables (fast field, no LP in the loop)
- denoiser_rollout.npz forward-denoiser trajectories + flow data
- flgfn_denoiser.png  the figure

## Run
    python3 flgfn_denoiser.py     # needs torch, numpy, scipy + 00_files/ (core model)
    python3 plot_denoiser.py

## Loss (Eq.11, FL-DB) + anchors
    L = ( logF~(s)+logP_F(s'|s) - logF~(s')-logP_B(s|s') + E(s->s') )^2
      + flow-matching( drift_F ~ s'-s,  drift_B ~ s-s' )    # anchors P_F as denoiser
      + 0.1*logF~(terminal)^2                               # F=R at terminal

## Honest scope / simplifications
- Continuous-state GFlowNet: P_F, P_B are Gaussian kernels (Lahlou 2023 style).
- Autonomous field snapshot (F=0.05, c=0.23); the true system is non-autonomous.
- Energy = local divergence * dt (first-order); eta-additivity verified to ~2%.
- The flow-matching anchor is what makes P_F a usable denoiser; pure FL-DB is
  under-determined with free F~ (residual collapses without learning the drift).
- Reward is proportional to terminal density up to the initial-density factor
  (E(s0):=0 convention).

## UPDATE: faithful on-policy version (flgfn_faithful.py)
The first version (flgfn_denoiser.py) produced a near-straight line. Two reasons:
  (1) it was OFF-POLICY with a flow-matching crutch -> P_F just copied the
      deterministic ODE flow (one path/start, near-affine field => straight).
  (2) E(s->s')=div*dt is only a state-function ALONG flow steps (violates
      Assumption 4.1 for arbitrary transitions).
flgfn_faithful.py fixes both, mirroring ling-pan/FL-GFN (learn_from_fl):
  * ON-POLICY: P_F samples the trajectories it trains on (no flow-matching).
  * STATE-FUNCTION energy E(s) = -log p_flow(s,t) from the flowed density, so
    E(s->s') = E(s')-E(s) is path-independent.
  * discrete time-layered (G,X) grid DAG (FL-GFN is inherently discrete).
  * trains FL-DB and plain DB; FL-DB matches the flowed terminal density
    (L1~0.8) while DB cannot assign credit over 10 steps (L1~1.7).
Caveats: single seed (DB is unstable run-to-run); V is deterministic (dV=F) so
the GFN samples in (G,X) with V=V(t); field frozen to one autonomous snapshot.
