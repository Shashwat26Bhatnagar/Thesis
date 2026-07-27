#!/usr/bin/env python3
"""Locate where the gradient dies: policy basis, squashing, or GP kernel.

Rolls the policy through the frozen GP and, at several points along the horizon,
measures separately:
    (A) state magnitude / distance to the policy's RBF centers
    (B) policy basis activation  -> tests "RBF center abandonment"
    (C) pre-squash vs post-squash output -> tests "squashing saturation"
    (D) d(action)/d(policy params)  -> is the POLICY still differentiable?
    (E) d(GP delta)/d(gp input)     -> is the GP still differentiable?

Run from the repo root:  python policy_learning/diagnose_vanishing.py
"""
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
import policy_learning.Policy as Policy

dtype, device = torch.float64, torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)
np.set_printoptions(precision=4, suppress=True, linewidth=130)

SD, ID = pdata.OBS_DIM, pdata.ACT_DIM
G = SD + ID
P = 64
T = 750
PROBES = [0, 50, 150, 300, 500, 749]

# ---------------- load frozen GP ----------------
c = torch.load(os.path.join(_REPO, "results_pensim", "rbf_model.pt"),
               map_location=device, weights_only=False)
init = dict(active_dims=np.arange(G), lengthscales_init=np.ones(G),
            flg_train_lengthscales=True, lambda_init=np.ones(1), flg_train_lambda=True,
            sigma_n_init=1e-2*np.ones(1), sigma_n_num=1e-4, flg_train_sigma_n=True,
            dtype=dtype, device=device)
model = ML.Model_learning_RBF(num_gp=SD, init_dict_list=[dict(init) for _ in range(SD)],
                              approximation_mode=None, dtype=dtype, device=device,
                              flg_norm=False)
model.load_state_dict(c["state_dict"])
for k in ("gp_inputs","gp_output_list","alpha_list","m_X_list","K_X_inv_list","gp_inputs_tr_list"):
    setattr(model, k, c[k])
model.num_samples = c["gp_inputs"].shape[0]; model.norm_list = [1.0]*SD
model.set_eval_mode()

Xtr = model.gp_inputs                       # (300, 14) training inputs
print(f"GP training inputs: {tuple(Xtr.shape)}")
print(f"  per-dim range: min={Xtr.min(0).values.numpy()}")
print(f"                 max={Xtr.max(0).values.numpy()}")
print(f"  GP lengthscales (GP0): {torch.exp(model.gp_list[0].log_lengthscales_par).detach().numpy()}")
print(f"  GP prior lambda per GP: "
      f"{[round(torch.exp(g.log_lambda_par).item(),4) for g in model.gp_list]}")

# ---------------- policy (same config as the driver) ----------------
nb = 200
policy = Policy.Sum_of_gaussians(
    state_dim=SD, input_dim=ID, num_basis=nb, u_max=3.0,
    flg_squash=True, flg_drop=True,
    centers_init=np.random.randn(nb, SD),
    lengthscales_init=np.ones(SD),
    weight_init=0.1*np.random.randn(ID, nb),
    dtype=dtype, device=device)

# find the centers / lengthscales tensors by name (naming varies)
named = dict(policy.named_parameters())
print("\npolicy parameters:", {k: tuple(v.shape) for k, v in named.items()})
ctr = None; ls = None
for k, v in named.items():
    if v.shape == (nb, SD):  ctr = v
    if v.numel() == SD and ls is None and v.shape != (nb, SD): ls = v
print(f"  centers range: [{ctr.min().item():.3f}, {ctr.max().item():.3f}]" if ctr is not None
      else "  (centers tensor not identified)")

# ---------------- rollout with probes ----------------
s = Xtr[:P, :SD].clone()
print(f"\n{'t':>5} {'|s|_max':>9} {'min dist':>9} {'basis max':>10} {'basis>1e-6':>11} "
      f"{'|a| max':>9} {'d a/d th':>10} {'d gp/d in':>11} {'GP var max':>11}")
print("-"*95)

for t in range(T):
    if t in PROBES:
        s_p = s.detach().clone().requires_grad_(True)
        a_p = policy(states=s_p, t=t, p_dropout=0.0)

        # (D) gradient of the action wrt POLICY parameters
        policy.zero_grad()
        a_p.abs().sum().backward(retain_graph=True)
        d_theta = max((p.grad.abs().max().item() for p in policy.parameters()
                       if p.grad is not None), default=0.0)

        with torch.no_grad():
            # (A) state magnitude and distance to nearest policy centre
            if ctr is not None:
                dist = torch.cdist(s_p.detach(), ctr.detach())           # (P, nb)
                min_dist = dist.min().item()
                # (B) basis activation exp(-0.5 * d^2 / l^2), l ~ 1
                basis = torch.exp(-0.5 * dist**2)
                bmax = basis.max().item()
                bcnt = int((basis > 1e-6).sum().item())
            else:
                min_dist, bmax, bcnt = float("nan"), float("nan"), -1
            smax = s_p.detach().abs().max().item()
            amax = a_p.detach().abs().max().item()

        # (E) gradient of the GP output wrt its INPUT
        gp_in = torch.cat([s_p.detach(), a_p.detach()], 1).requires_grad_(True)
        ml, vl = model.get_gp_estimate(gp_inputs=gp_in, gp_index_list=range(SD))
        dm = torch.cat(ml, 1); dv = torch.cat([v.reshape(-1,1) for v in vl], 1)
        dm.abs().sum().backward()
        d_gp = gp_in.grad.abs().max().item()

        print(f"{t:5d} {smax:9.3f} {min_dist:9.3f} {bmax:10.3e} {bcnt:11d} "
              f"{amax:9.3f} {d_theta:10.3e} {d_gp:11.3e} {dv.max().item():11.3e}")

    with torch.no_grad():
        a = policy(states=s, t=t, p_dropout=0.0)
        s, _, _ = model.get_next_state(current_state=s, current_input=a,
                                       particle_pred=True)

print("\nREADING THE TABLE")
print("  'basis>1e-6' collapsing to 0  -> RBF CENTER ABANDONMENT (policy basis dead)")
print("  '|a| max' pinned at u_max=3.0 -> SQUASH SATURATION")
print("  'd a/d th' -> 0               -> gradient dies in the POLICY")
print("  'd gp/d in' -> 0              -> gradient dies in the GP kernel")
print("  'GP var max' -> prior         -> GP has no information at that state")
