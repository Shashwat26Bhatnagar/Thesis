#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_test_cdil.py   (MC-PILCO repo root)

Verifies the CDIL integration in three independent parts:

  PART A  EXPERT (FL-GFN / PFQuery)
          Is the output really a Gaussian N(b, Sigma)? Checked EMPIRICALLY by
          drawing samples and comparing their moments to the claimed b and cov.

  PART B  MC-PILCO GP
          What does the GP actually return? Is P(s'|s,a) a Gaussian with the
          claimed mean/variance? Is it differentiable w.r.t. the input?

  PART C  INTEGRATION
          Rollout shapes, per-step loss, gradient flow, graph modes.

Run:  python smoke_test_cdil.py
"""
import os, sys, math
import numpy as np
import torch

EXPERT_REPO = os.path.expanduser("~/penicillin-dcfba")
if EXPERT_REPO not in sys.path:
    sys.path.insert(0, EXPERT_REPO)

dtype, device = torch.float64, torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

PASS, FAIL = "PASS", "FAIL"
_results = []
def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    return ok


# =====================================================================================
print("=" * 78)
print("PART A -- EXPERT (FL-GFN):  is the output a Gaussian with mean + covariance?")
print("=" * 78)
# =====================================================================================
from dcfba_pen.flgfn.pf_query import PFQuery

q = PFQuery(verbose=True)
d = q.next_state_distribution(t=75.0, source="traj")

a   = np.asarray(d["a"], dtype=float)
b   = np.asarray(d["b"], dtype=float)
cov = np.asarray(d["cov"], dtype=float)
sig = np.asarray(d["sigma"], dtype=float)

print(f"\n  a   (state at t)      = {a}")
print(f"  b   (Gaussian MEAN)   = {b}")
print(f"  drift = b - a         = {b - a}")
print(f"  sigma (marginal std)  = {sig}")
print(f"  cov (3x3, physical)   =\n{cov}")

check("expert mean b is 3-D", b.shape == (3,), f"shape={b.shape}")
check("expert cov is 3x3", cov.shape == (3, 3), f"shape={cov.shape}")
check("expert cov symmetric", np.allclose(cov, cov.T, atol=1e-12),
      f"max asym={np.abs(cov - cov.T).max():.2e}")
eig = np.linalg.eigvalsh(cov)
check("expert cov positive definite", eig.min() > 0, f"min eig={eig.min():.4e}")
off = cov - np.diag(np.diag(cov))
check("expert cov is FULL (off-diagonals non-zero)", np.abs(off).max() > 1e-12,
      f"max |off-diag|={np.abs(off).max():.4e}")
check("sigma == sqrt(diag(cov))", np.allclose(sig, np.sqrt(np.diag(cov))))

# --- EMPIRICAL: do samples actually follow N(b, cov)? ---
N = 200_000
S = np.asarray(d["sample"](N, seed=0), dtype=float)
emp_mu, emp_cov = S.mean(0), np.cov(S.T)
mu_err = np.abs(emp_mu - b) / (sig / math.sqrt(N) + 1e-12)          # in std errors
cov_rel = np.abs(emp_cov - cov) / (np.abs(cov) + 1e-12)
print(f"\n  empirical mean ({N} samples) = {emp_mu}")
print(f"  empirical cov =\n{emp_cov}")
check("samples match claimed MEAN", np.all(mu_err < 5.0),
      f"max deviation = {mu_err.max():.2f} std-errors")
check("samples match claimed COV", cov_rel.max() < 0.05,
      f"max rel err = {cov_rel.max():.3%}")

# --- density consistency: pdf peaks at the mean ---
p_at_b = float(np.asarray(d["pdf"](b)).ravel()[0])
p_off  = float(np.asarray(d["pdf"](b + 2 * sig)).ravel()[0])
check("pdf is maximal at the mean", p_at_b > p_off,
      f"pdf(b)={p_at_b:.4g} > pdf(b+2sig)={p_off:.4g}")
lp = float(np.asarray(d["logpdf"](b)).ravel()[0])
check("logpdf == log(pdf)", abs(lp - math.log(p_at_b)) < 1e-6)

# --- sanity vs the deterministic ground truth ---
b_true = np.asarray(d["b_true"], dtype=float)
rel = np.abs(b - b_true) / (np.abs(b_true) + 1e-9)
print(f"\n  b_true (a + field*DT) = {b_true}")
check("learned mean close to ground-truth next state", rel.max() < 0.5,
      f"max rel diff = {rel.max():.2%}")

# --- is the expert differentiable? (it should NOT be) ---
print("\n  NOTE: PFQuery runs under torch.no_grad and returns numpy -> the expert is")
print("        NON-differentiable by design. It is the target, not part of the graph.")


# =====================================================================================
print("\n" + "=" * 78)
print("PART B -- MC-PILCO GP:  what does it output for P(s'|s,a)?")
print("=" * 78)
# =====================================================================================
import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata

STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM
ckpt = torch.load("results_pensim/rbf_model.pt", map_location=device, weights_only=False)

init_dict = dict(active_dims=np.arange(0, GP_INPUT_DIM),
                 lengthscales_init=np.ones(GP_INPUT_DIM), flg_train_lengthscales=True,
                 lambda_init=np.ones(1), flg_train_lambda=True,
                 sigma_n_init=1e-2*np.ones(1), sigma_n_num=1e-4, flg_train_sigma_n=True,
                 dtype=dtype, device=device)
model = ML.Model_learning_RBF(num_gp=STATE_DIM,
                              init_dict_list=[dict(init_dict) for _ in range(STATE_DIM)],
                              approximation_mode=None, dtype=dtype, device=device,
                              flg_norm=False)
model.load_state_dict(ckpt["state_dict"])
for k in ("gp_inputs","gp_output_list","alpha_list","m_X_list","K_X_inv_list","gp_inputs_tr_list"):
    setattr(model, k, ckpt[k])
model.num_samples = ckpt["gp_inputs"].shape[0]
model.dim_state, model.dim_input = STATE_DIM, INPUT_DIM
model.norm_list = [1.0] * STATE_DIM
model.set_eval_mode()
print(f"  loaded: num_gp={model.num_gp}  train pts={model.gp_inputs.shape[0]}  "
      f"gp input dim={model.gp_inputs.shape[1]}")

P = 64
s = model.gp_inputs[:P, :STATE_DIM].clone()
act = model.gp_inputs[:P, STATE_DIM:].clone()

s_next, dmean, dvar = model.get_next_state(current_state=s, current_input=act,
                                           particle_pred=True)
mu = s + dmean          # mean of P(s'|s,a)
cov = dvar              # diagonal covariance

print(f"\n  get_next_state returns 3 tensors:")
print(f"    next_state  {tuple(s_next.shape)}")
print(f"    delta_mean  {tuple(dmean.shape)}   <- MEAN of the delta")
print(f"    delta_var   {tuple(dvar.shape)}   <- VARIANCE of the delta (DIAGONAL)")
check("mean shape (P, 8)", tuple(mu.shape) == (P, STATE_DIM))
check("cov shape (P, 8) diagonal", tuple(cov.shape) == (P, STATE_DIM))
check("variance strictly positive", bool((dvar > 0).all()),
      f"min var={dvar.min().item():.3e}")
print(f"\n  mean delta per channel : {dmean.mean(0).detach().numpy()}")
# Clamping dvar to ensure numerical domain stability before square root operation
print(f"  mean std  per channel : {torch.sqrt(torch.clamp(dvar, min=0.0)).mean(0).detach().numpy()}")

# --- deterministic branch returns exactly the mean ---
s_det, dmean2, dvar2 = model.get_next_state(current_state=s, current_input=act,
                                            particle_pred=False)
check("particle_pred=False gives exactly s + mean",
      torch.allclose(s_det, s + dmean2, atol=1e-12))

# --- EMPIRICAL: is the reparameterized sample really N(mu, cov)? ---
R = 20000
s1 = s[:1].expand(R, -1).contiguous()
a1 = act[:1].expand(R, -1).contiguous()
with torch.no_grad():
    sn, dm, dv = model.get_next_state(current_state=s1, current_input=a1,
                                      particle_pred=True)
emp_m = (sn - s1).mean(0)
emp_v = (sn - s1).var(0)
claim_m, claim_v = dm[0], torch.clamp(dv[0], min=0.0)
m_ok = torch.allclose(emp_m, claim_m, atol=5*torch.sqrt(claim_v/R).max().item()+1e-9)
v_rel = ((emp_v - claim_v).abs() / (claim_v + 1e-30)).max().item()
print(f"\n  EMPIRICAL check with {R} reparameterized samples from ONE (s,a):")
print(f"    claimed mean  = {claim_m.numpy()}")
print(f"    empirical mean= {emp_m.numpy()}")
print(f"    claimed var   = {claim_v.numpy()}")
print(f"    empirical var = {emp_v.numpy()}")
check("reparam samples match claimed MEAN", bool(m_ok))
check("reparam samples match claimed VARIANCE", v_rel < 0.10,
      f"max rel err = {v_rel:.2%}")

# --- covariance structure: independent GPs -> no cross-channel correlation ---
corr = np.corrcoef((sn - s1).detach().numpy().T)
offmax = np.abs(corr - np.eye(STATE_DIM)).max()
check("GP covariance is DIAGONAL (channels uncorrelated)", offmax < 0.05,
      f"max |off-diag corr| = {offmax:.3f}")
print("        -> 8 independent scalar GPs, so cov has NO cross terms by construction,")
print("           unlike the expert's FULL 3x3.")

# --- differentiability w.r.t. the INPUT (what the policy gradient needs) ---
s_g = s.clone().requires_grad_(True)
a_g = act.clone().requires_grad_(True)
sn_g, dm_g, dv_g = model.get_next_state(current_state=s_g, current_input=a_g,
                                        particle_pred=True)
sn_g.sum().backward()
check("d(next_state)/d(state) exists", s_g.grad is not None and s_g.grad.abs().max() > 0,
      f"max|grad|={s_g.grad.abs().max().item():.3e}")
check("d(next_state)/d(action) exists", a_g.grad is not None and a_g.grad.abs().max() > 0,
      f"max|grad|={a_g.grad.abs().max().item():.3e}")
check("GP hyperparameters are FROZEN",
      all(not p.requires_grad for p in model.parameters()))


# =====================================================================================
print("\n" + "=" * 78)
print("PART C -- INTEGRATION:  rollout, per-step loss, gradient flow")
print("=" * 78)
# =====================================================================================
import policy_learning.Policy as Policy
from policy_learning.gp_particle_rollout import (gp_rollout, sample_initial_particles,
                                                 cov_full)

num_basis = 50
policy = Policy.Sum_of_gaussians(
    state_dim=STATE_DIM, input_dim=INPUT_DIM, num_basis=num_basis,
    u_max=3.0, flg_squash=True, flg_drop=True,
    centers_init=np.random.randn(num_basis, STATE_DIM),
    lengthscales_init=np.ones(STATE_DIM),
    weight_init=0.1*np.random.randn(INPUT_DIM, num_basis),
    dtype=dtype, device=device)
check("policy input dim == state dim", policy.state_dim == STATE_DIM)
check("policy output dim == action dim", policy.input_dim == INPUT_DIM)

T, NP = 5, 16
s0 = sample_initial_particles(model.gp_inputs[:, :STATE_DIM], NP, dtype=dtype, device=device)

seen = {}
def probe_loss(t, s, a, mu, cov, s_next):
    seen[t] = dict(s=tuple(s.shape), a=tuple(a.shape),
                   mu=tuple(mu.shape), cov=tuple(cov.shape))
    return (mu ** 2).sum(dim=1).mean()

out = gp_rollout(model=model, policy=policy, s0=s0, T=T, p_dropout=0.0,
                 particle_pred=True, loss_fn=probe_loss, graph_mode="full")

print(f"\n  per-step tensors handed to loss_fn: {seen[0]}")
check("loss_fn called every step", len(seen) == T, f"{len(seen)}/{T}")
check("S shape (P, T+1, 8)", tuple(out["S"].shape) == (NP, T+1, STATE_DIM))
check("A shape (P, T, 6)", tuple(out["A"].shape) == (NP, T, INPUT_DIM))
check("Mu shape (P, T, 8)", tuple(out["Mu"].shape) == (NP, T, STATE_DIM))
check("Cov shape (P, T, 8)", tuple(out["Cov"].shape) == (NP, T, STATE_DIM))
check("Mu == S[:, :-1] + Dmean",
      torch.allclose(out["Mu"], out["S"][:, :-1] + out["Dmean"], atol=1e-12))
check("Cov == Dvar", torch.allclose(out["Cov"], out["Dvar"]))
check("cov_full -> (P, T, 8, 8)",
      tuple(cov_full(out["Cov"]).shape) == (NP, T, STATE_DIM, STATE_DIM))

check("step losses are scalars with grad_fn",
      all(l.dim() == 0 and l.grad_fn is not None for l in out["step_losses"]))
check("loss_total has grad_fn", out["loss_total"].grad_fn is not None)

# --- THE critical test: does the gradient reach the policy? ---
policy.zero_grad()
out["loss_total"].backward()
grads = [p.grad for p in policy.parameters() if p.grad is not None]
gmax = max(g.abs().max().item() for g in grads) if grads else 0.0
check("gradient reaches policy parameters", gmax > 0,
      f"max|grad| = {gmax:.3e}  over {len(grads)} tensors")
check("no NaN/Inf in gradients",
      all(torch.isfinite(g).all() for g in grads))

# --- graph modes ---
flushed = []
def flush_fn(seg):
    flushed.append(float(seg.detach()))
    policy.zero_grad(); seg.backward()

out_t = gp_rollout(model=model, policy=policy, s0=s0, T=T, loss_fn=probe_loss,
                   flush_fn=flush_fn, graph_mode="truncated", truncate_every=2)
check("truncated mode flushes at boundaries", len(flushed) == T // 2,
      f"{len(flushed)} flushes for T={T}, every 2")

flushed.clear()
out_p = gp_rollout(model=model, policy=policy, s0=s0, T=T, loss_fn=probe_loss,
                   flush_fn=flush_fn, graph_mode="per_step")
check("per_step mode flushes every step", len(flushed) == T, f"{len(flushed)}/{T}")

# --- expert-vs-model space mismatch, stated explicitly ---
print(f"\n  DIMENSION CHECK:")
print(f"    GP model state : {STATE_DIM}-D {pdata.OBS_NAMES}")
print(f"    expert state   : 3-D (S g/L, X g, V L)")
print(f"    GP cov         : DIAGONAL ({STATE_DIM},)   expert cov: FULL (3,3)")
check("model and expert live in DIFFERENT spaces (projection required)",
      STATE_DIM != 3, "-> project_to_expert() must be filled in")


# =====================================================================================
n_pass = sum(ok for _, ok in _results)
print("\n" + "=" * 78)
print(f"RESULT: {n_pass}/{len(_results)} checks passed")
for nm, ok in _results:
    if not ok:
        print(f"   FAILED: {nm}")
print("=" * 78)
