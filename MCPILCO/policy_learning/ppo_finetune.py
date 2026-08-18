#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/ppo_finetune.py

Fine-tune an imitation-trained policy for reward maximisation with PPO, rolling out
inside the LEARNED WORLD MODELS rather than the simulator.

    python policy_learning/ppo_finetune.py \\
        -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 \\
        -reward_model results_pensim/reward_model_base.pt \\
        -init_policy  results_clean/repro_bnd_rbf1.pt \\
        -out results_ppo/ppo_policy_iter0.pt

WHY THE WORLD MODELS AND NOT THE SIMULATOR
    One PenSim episode is 1150 ODE solves, ~2 minutes. PPO needs hundreds of episodes
    per update round, so on the simulator a single fine-tuning stage would run for
    hours; inside the GPs a full 1150-step episode takes seconds. The cost is that PPO
    optimises the MODEL's belief -- including its errors -- which is why the reward is
    taken as a LOWER confidence bound below.

THE ARCHITECTURE CHANGE PPO FORCES
    Sum_of_gaussians is deterministic (dropout aside) and has no log-probability, so
    it cannot be a PPO actor as it stands. A Gaussian head is wrapped around it: the
    imitation policy supplies the MEAN, and a learnable state-independent log-std
    supplies the spread. At log_std -> -inf the actor reduces exactly to the imitation
    policy, so initialisation is not a perturbation of it.

    The value function is a separate small MLP, trained from scratch -- the imitation
    run never learned one.

ACTION PENALTY ON PPO'S OWN REWARD: L1 ON DISCHARGE, L2 ON THE REST
    WorldModelEnv's reward used to be pure -LCB(reward GP), with nothing discouraging
    PPO from drifting discharge away from whatever sparsity the warm-start policy
    arrived with. An imitation-trained policy with L1 on discharge (exp_policy_l1.py)
    has a real reason to hold it near CLOSED; a policy trained with plain L2 on every
    channel (the earlier, buggy configuration) does not, and PPO fine-tuning from
    either warm start had no mechanism of its own to tell the two apart or to defend
    the good one.

    ActionPenaltyWrapper adds the SAME penalty shape as exp_policy_l1.py's imitation
    objective -- L1 on discharge measured from the CLOSED level (not from z=0, which
    is the dataset MEAN action, not "off"), L2 (additive) on the other five channels
    -- as a wrapper around WorldModelEnv rather than inside it, so the raw,
    unpenalised reward stays available for calibration and logging.

    -lam_l1 follows the same "measured, not guessed" convention used throughout this
    codebase: a short forward-only rollout of the CURRENT (warm-started) actor
    through a separate WorldModelEnv instance measures raw |reward|/step and raw
    l1_disch/step before any PPO training has happened, then
        lam_l1 = l1_ratio * |reward| / l1_disch
    puts the penalty on the reward's scale from the first training step, rather than
    starting from an arbitrary constant and hoping it's in the right regime the way
    the very first (unweighted, LAMBDA_L1=0.05) version of the imitation-stage fix
    did. Pass -lam_l1 explicitly, or -no_calib, to skip this.

WHAT LIMITS THIS
    The SMPL paper's own PPO on PenSim scores 2.5231 mean reward against the recipe
    baseline's 3.3071, i.e. PPO from scratch does WORSE than the recipe there. The
    imitation policies here already reach 2.82-3.03 per step. So the value of this
    stage rests entirely on the warm start; there is no evidence PPO finds a good
    PenSim policy unaided.

    Hyperparameters follow the stable-baselines3 defaults (n_steps 2048, batch 64,
    n_epochs 10, gamma 0.99, gae_lambda 0.95, clip 0.2, ent_coef 0.0, vf_coef 0.5,
    max_grad_norm 0.5, lr 3e-4) rather than being tuned here.

THE NUMPY/TORCH ABI BUG (and why torch.Tensor.numpy is patched below)
    SB3's collect_rollouts does `np.clip(actions, self.action_space.low,
    self.action_space.high)`, where `actions` comes from `torch.Tensor.numpy()`. In
    this env that call fails with:
        TypeError: no implementation found for 'numpy.clip' on types that implement
        __array_function__: [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>]
    This looks like a module-identity problem but isn't -- `numpy` is a single
    imported module everywhere here (verified: same id(), same __file__, same
    version, for numpy itself, gym.spaces.box.np, and
    stable_baselines3.common.on_policy_algorithm.np). The actual cause is lower
    level: this conda env's compiled `torch` and its pinned `numpy==1.23.5` disagree
    at the C-ABI layer, so a tensor's `.numpy()` view is a COMPILED TYPE that prints
    as "numpy.ndarray" but is not numpy's own registered ndarray type --
        id(type(torch.randn(6).numpy())) != id(type(np.array([0.0])))
    -- confirmed directly in this env.

    FIRST ATTEMPT (superseded): patch only the one `np.clip` call SB3 makes inside
    collect_rollouts, laundering the array back through `.tolist()` ->
    `np.array(...)` right there. This worked for that call site, but the poisoning
    resurfaced one step later inside RolloutBuffer.compute_returns_and_advantage's
    GAE arithmetic (self.rewards[step] + gamma*next_values*next_non_terminal -
    self.values[step]) -- because `values`/`rewards` were ALSO stored into the
    buffer straight from an unlaundered `.cpu().numpy()`, so the poisoned type was
    never actually confined to the one place it was first noticed. Chasing each
    downstream call site individually is not tractable -- there is no guarantee
    another one doesn't turn up in the GAE step, in logging, or in a later training
    epoch.

    FINAL FIX: patch `torch.Tensor.numpy` itself, once, at its single common
    source, so every torch->numpy conversion anywhere in SB3's internals -- actions,
    values, rewards, log_probs, whatever else -- is laundered from the moment it
    leaves torch. This trades the normal zero-copy `.numpy()` view for a
    `.tolist()`-based copy on every conversion; for the small (STATE_DIM=8,
    INPUT_DIM=6) arrays here that's not measurable, so it's the right tradeoff, but
    it would NOT be for large-array workloads -- don't reuse this pattern unmodified
    where arrays are big.

    ONE THING TO WATCH: any monkeypatch of an attribute reached via `np.<name>` in
    this codebase is patching THE SAME MODULE everywhere `np` is imported, not a
    scoped copy -- confirmed the hard way (the first version of the np.clip patch
    called `np.clip` from inside its own replacement and recursed until
    RecursionError, because `_opa.np.clip` and the module-global `np.clip` are the
    identical attribute). Always capture the original implementation into a
    plain local name BEFORE reassigning, and call that saved name, never the
    live `np.<name>` reference, from inside a replacement.
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import gym
from gym import spaces

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
from policy_learning.policy_variants import rebuild_policy

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
dtype, device = torch.float64, torch.device("cpu")

STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_IN = STATE_DIM + INPUT_DIM
HOURS_PER_STEP = 0.2
EPISODE_STEPS = 1150                 # the full 230 h batch
U_MAX = 3.0
KAPPA = 1.0                          # reward LCB: mu_r - KAPPA*sigma_r

_ap = argparse.ArgumentParser("PPO fine-tuning inside the world models")
_ap.add_argument("-phase_prefix", required=True)
_ap.add_argument("-reward_model", required=True)
_ap.add_argument("-init_policy", required=True, help="the imitation-trained policy")
_ap.add_argument("-out", required=True)
_ap.add_argument("-timesteps", type=int, default=50000)
_ap.add_argument("-kappa", type=float, default=None)
_ap.add_argument("-lr", type=float, default=3e-4)
_ap.add_argument("-init_log_std", type=float, default=-2.0,
                 help="start close to deterministic: exp(-2)=0.135 in z-units")
_ap.add_argument("-lam_l1", type=float, default=None,
                 help="L1 weight on discharge in PPO's reward, measured from the "
                      "closed level. If omitted, calibrated automatically (see "
                      "-l1_ratio); passing this explicitly disables calibration.")
_ap.add_argument("-lam_l2", type=float, default=0.01,
                 help="L2 weight on the five non-discharge action channels in PPO's "
                      "reward")
_ap.add_argument("-l1_ratio", type=float, default=1.0,
                 help="target ratio of (lam_l1 * l1_disch) to |raw reward| after "
                      "auto-calibration, same convention as exp_policy_l1.py")
_ap.add_argument("-calib_episodes", type=int, default=3)
_ap.add_argument("-calib_steps", type=int, default=200,
                 help="steps per calibration episode -- a fraction of the full "
                      "1150-step episode is enough to measure raw magnitudes")
_ap.add_argument("-no_calib", action="store_true",
                 help="skip lam_l1 auto-calibration; use -lam_l1 (or its default) "
                      "as given")
_args = _ap.parse_known_args()[0]
if _args.kappa is not None:
    KAPPA = _args.kappa
os.makedirs(os.path.dirname(_args.out) or ".", exist_ok=True)


# ================================================================== the models ===
def _load(path, n_gp):
    ck = torch.load(path, map_location=device, weights_only=False)
    init = dict(active_dims=np.arange(0, GP_IN), lengthscales_init=np.ones(GP_IN),
                flg_train_lengthscales=True, lambda_init=np.ones(1),
                flg_train_lambda=True, sigma_n_init=1e-2 * np.ones(1),
                sigma_n_num=1e-4, flg_train_sigma_n=True, dtype=dtype, device=device)
    m = ML.Model_learning_RBF(num_gp=n_gp,
                              init_dict_list=[dict(init) for _ in range(n_gp)],
                              approximation_mode=None, dtype=dtype, device=device,
                              flg_norm=False)
    m.load_state_dict(ck["state_dict"])
    for k in ("gp_inputs", "gp_output_list", "alpha_list", "m_X_list",
              "K_X_inv_list", "gp_inputs_tr_list"):
        setattr(m, k, ck[k])
    m.num_samples = ck["gp_inputs"].shape[0]
    m.dim_state, m.dim_input = STATE_DIM, INPUT_DIM
    m.norm_list = [1.0] * n_gp
    m.set_eval_mode()
    return m, ck


MODELS, CKS = {}, {}
for _p in (0, 1, 2):
    MODELS[_p], CKS[_p] = _load(f"{_args.phase_prefix}_phase{_p}.pt", STATE_DIM)
RMODEL, RCK = _load(_args.reward_model, 1)
R_MU, R_SD = float(RCK["reward_mu"]), float(RCK["reward_sd"])
STATS = {k: np.asarray(CKS[0][k]) for k in
         ("std_obs_mu", "std_obs_sd", "std_act_mu", "std_act_sd")}
POOL = torch.cat([MODELS[p].gp_inputs[:, :STATE_DIM] for p in (0, 1, 2)], 0)

print("world models:")
for _p in (0, 1, 2):
    lo, hi = pdata.PHASES[_p]
    print(f"  phase {_p}: [{lo:g},{'inf' if hi > 1e8 else f'{hi:g}'}) h  "
          f"train pts={MODELS[_p].gp_inputs.shape[0]}")
print(f"reward GP: held-out R^2={RCK.get('held_out_r2'):.4f}  "
      f"yield/step mu={R_MU:.4f} sd={R_SD:.4f}  -> LCB with kappa={KAPPA}")


def phase_of(t_h):
    for p in (0, 1, 2):
        lo, hi = pdata.PHASES[p]
        if lo <= t_h < hi:
            return p
    return 2


# ===================================================================== the env ===
class WorldModelEnv(gym.Env):
    """A Gym env whose dynamics are the three phase GPs and whose reward is the
    reward GP's lower confidence bound.

    Episodes run the full 1150 steps (230 h) from a start state drawn from the phase-0
    training inputs, so PPO optimises TOTAL production over the batch rather than the
    average yield of an isolated hour -- which is what the windowed imitation objective
    could not express.

    Terminates early if the state leaves the training range by more than
    OUT_OF_RANGE_MULT, because past that the GPs are extrapolating and their reward
    predictions are not meaningful.
    """

    OUT_OF_RANGE_MULT = 1.5

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=float("-inf"), high=float("inf"),
            shape=(STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-float(U_MAX), high=float(U_MAX),
            shape=(INPUT_DIM,), dtype=np.float32)
        self._pool0 = MODELS[0].gp_inputs[:, :STATE_DIM]
        self._lim = float(POOL.abs().max()) * self.OUT_OF_RANGE_MULT
        self._rng = np.random.default_rng(0)
        self.s = None
        self.t = 0

    def reset(self):
        i = int(self._rng.integers(self._pool0.shape[0]))
        self.s = self._pool0[i].clone()
        self.t = 0
        return np.asarray(self.s.tolist(), dtype=np.float32)

    def step(self, action):
        a = torch.tensor(np.asarray(action, dtype=np.float64).reshape(-1),
                         dtype=dtype, device=device)
        gp_in = torch.cat([self.s, a]).reshape(1, -1)
        t_h = self.t * HOURS_PER_STEP
        with torch.no_grad():
            mdl = MODELS[phase_of(t_h)]
            m_l, v_l = mdl.get_gp_estimate(gp_inputs=gp_in,
                                           gp_index_list=list(range(STATE_DIM)))
            delta = torch.cat([m_l[i].reshape(1) for i in range(STATE_DIM)])
            rm, rv = RMODEL.get_gp_estimate(gp_inputs=gp_in, gp_index_list=[0])
            # LOWER confidence bound: maximising the mean would send the policy where
            # the model is uncertain and optimistic
            r = float(rm[0].reshape(-1)[0]
                      - KAPPA * torch.sqrt(rv[0].reshape(-1)[0].clamp_min(1e-12)))
        self.s = self.s + delta
        self.t += 1
        out = bool(self.s.abs().max() > self._lim)
        done = out or self.t >= EPISODE_STEPS
        return (np.asarray(self.s.tolist(), dtype=np.float32), r, done,
                {"out_of_range": out, "t_hours": t_h})


# ============================================== discharge sparsity / L2 penalty ===
# Mirrors exp_policy_l1.py's imitation-stage objective exactly, applied to PPO's own
# reward instead: L1 on discharge (measured from CLOSED -- z=0 is the dataset MEAN
# action, not "off", so the offset matters), L2 (additive) on the other five
# channels. Without this, PPO has no structural reason to preserve whatever
# discharge sparsity the warm-start policy arrived with; its own gradient updates
# could just as easily drift discharge back toward the flat, "hands-full" band a
# plain-L2-trained imitation policy produces (see exp_policy_l1.py's own docstring
# for the underlying Nagahara et al. argument for why L1, not L2, is the right cost
# shape for a valve-like channel).
DISCHARGE_IDX = 0
DISCHARGE_OFF_PHYS = 0.0     # "closed"
_off_smpl = (2.0 * (DISCHARGE_OFF_PHYS - pdata.MIN_ACT[DISCHARGE_IDX])
             / (pdata.MAX_ACT[DISCHARGE_IDX] - pdata.MIN_ACT[DISCHARGE_IDX]) - 1.0)
A_OFF_Z = float((_off_smpl - STATS["std_act_mu"][DISCHARGE_IDX])
                / STATS["std_act_sd"][DISCHARGE_IDX])
_NON_DISCH = [i for i in range(INPUT_DIM) if i != DISCHARGE_IDX]
print(f"\ndischarge closed = {DISCHARGE_OFF_PHYS:.0f} phys = {A_OFF_Z:.3f} z "
      f"(the PPO action penalty is measured from here, not from z=0)")


class ActionPenaltyWrapper(gym.Wrapper):
    """Subtracts lam_l1*|a_disch - a_off| + lam_l2*||a_other||^2 from
    WorldModelEnv's raw LCB reward. Kept as a WRAPPER rather than folded into
    WorldModelEnv.step() itself so the raw, unpenalised reward stays available --
    both for the calibration pass below (which needs to measure the penalty's raw
    magnitude BEFORE it's applied) and for anyone inspecting `info["raw_reward"]`
    later without needing a second, unwrapped env.
    """

    def __init__(self, env, lambda_l1, lambda_l2):
        super().__init__(env)
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        l1 = abs(float(a[DISCHARGE_IDX] - A_OFF_Z))
        l2 = float(np.sum(a[_NON_DISCH] ** 2))
        info = dict(info, raw_reward=r, l1_disch=l1, l2_other=l2)
        return obs, r - self.lambda_l1 * l1 - self.lambda_l2 * l2, done, info


# ==================================================== the actor, warm-started ====
def build_actor():
    """Gaussian actor whose MEAN is the imitation policy."""
    ck = torch.load(_args.init_policy, map_location=device, weights_only=False)
    base = rebuild_policy(ck["policy_meta"], dtype=dtype, device=device)
    base.load_state_dict(ck["policy_state_dict"])
    print(f"\\nwarm start: {os.path.basename(_args.init_policy)}  "
          f"kind={ck['policy_meta']['kind']}  "
          f"params={sum(p.numel() for p in base.parameters())}")
    return base, ck


BASE, BASE_CK = build_actor()

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    raise SystemExit(
        "stable-baselines3 is not installed. Install it WITHOUT touching numpy:\\n"
        "    pip install 'stable-baselines3==1.8.0' --no-deps\\n"
        "    pip install 'gym==0.21.0' --no-deps   # if gym is missing\\n"
        "numpy is pinned at 1.23.5 here and the compiled extensions were built "
        "against the 1.x ABI, so a plain install that upgrades it breaks smpl, "
        "scipy and fastodeint.")


# --- launder torch-derived arrays before SB3's internal np.clip touches them -----
# See the module docstring for the full diagnosis. In short: `actions` inside
# collect_rollouts comes from torch.Tensor.numpy(), and in this env that produces a
# compiled type that is NOT numpy's own ndarray type despite printing as one --
# np.clip's __array_function__ dispatch then fails to reconcile it with
# self.action_space.low/high, which ARE native numpy arrays. Routing the array
# through `.tolist()` -> `np.array(...)` rebuilds it via numpy's own constructor
# from plain Python floats, so nothing of torch's compiled binary remains attached.
# --- launder EVERY torch->numpy conversion, process-wide, at the source ---------
# Patching np.clip alone (the first attempt) only covered ONE call site inside
# collect_rollouts. The identical poisoning then resurfaced inside
# RolloutBuffer.compute_returns_and_advantage's GAE arithmetic
# (self.rewards[step] + self.gamma*next_values*next_non_terminal - self.values[step]),
# confirming the poisoned type propagates through anything SB3 stores from a
# torch->numpy conversion -- values, rewards, whatever else -- not just the one call
# site that happened to be hit first. The TypeError there
# ("__array_wrap__() argument 1 must be numpy.ndarray, not numpy.ndarray") is numpy
# crashing while trying to PRINT the real error message for a ufunc type-resolution
# failure, which masks the underlying cause but is the same ABI mismatch: mixing a
# torch-derived array with a genuinely native one confuses both the
# __array_function__ protocol (np.clip, hit first) AND __array_ufunc__ (+/-/*, hit
# here) once enough of them get mixed together across buffer slots.
#
# Rather than chasing each downstream call site individually, patch
# torch.Tensor.numpy ONCE, at its single common source, so every conversion
# anywhere in SB3's internals is laundered from the moment it leaves torch. The
# np.clip-specific patch is no longer needed once this is in place and has been
# removed.
#
# Cost: every .numpy() call becomes a .tolist() round-trip copy instead of numpy's
# normal zero-copy view. For STATE_DIM=8 / INPUT_DIM=6 sized arrays over a
# short fine-tuning run this is not measurable; it would NOT be the right tradeoff
# for large-array workloads, so don't reuse this pattern unmodified somewhere the
# arrays are big.
# torch dtype -> numpy dtype, built directly from OUR OWN (genuinely native) np --
# used instead of ever reading `.dtype` off a torch-produced array, since that
# attribute access is itself suspect once the array's compiled type is foreign (see
# below: the first version of this patch read `arr.dtype` off the poisoned array
# and failed with a bare, message-less TypeError -- accessing an attribute of the
# mismatched type is not safe, not just calling functions on it).
_TORCH_TO_NP_DTYPE = {
    torch.float64: np.float64, torch.float32: np.float32, torch.float16: np.float16,
    torch.int64: np.int64, torch.int32: np.int32, torch.int16: np.int16,
    torch.int8: np.int8, torch.uint8: np.uint8, torch.bool: np.bool_,
}


def _laundered_tensor_numpy(self, *args, **kwargs):
    # Go straight from the TENSOR's own .tolist() -- a pure torch method that never
    # touches numpy's C-API -- to numpy's constructor. This never creates the
    # ABI-mismatched intermediate array at all, rather than creating it and trying
    # to clean it up afterward: the first version of this patch called torch's real
    # .numpy() first and then read `.tolist()`/`.dtype` off ITS result, which
    # crashed on the `.dtype` attribute read with a bare TypeError. `self` here is
    # the original torch.Tensor, never the poisoned array, so nothing about it is
    # suspect.
    np_dtype = _TORCH_TO_NP_DTYPE.get(self.dtype, np.float64)
    return np.array(self.detach().cpu().tolist(), dtype=np_dtype)


torch.Tensor.numpy = _laundered_tensor_numpy
print("patched torch.Tensor.numpy -> every torch->numpy conversion goes through "
      "tensor.tolist() instead of torch's real .numpy(), avoiding the ABI-mismatched "
      "intermediate array entirely (see 'THE NUMPY/TORCH ABI BUG' above)")


class ImitationMean(torch.nn.Module):
    """Wraps the imitation policy as SB3's mean network (float32 in, float32 out)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.latent_dim_pi = INPUT_DIM
        self.latent_dim_vf = 64
        self.vf = torch.nn.Sequential(
            torch.nn.Linear(STATE_DIM, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh())

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x):
        with torch.enable_grad():
            a = self.base(states=x.to(dtype), t=0, p_dropout=0.0)
        return a.to(torch.float32)

    def forward_critic(self, x):
        return self.vf(x.to(torch.float32))


class WarmStartPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = ImitationMean(BASE)


# ======================================================= calibrate lam_l1 ===
# Same convention as exp_policy_l1.py: measure the raw, unweighted magnitude of the
# penalty BEFORE any of its own pressure has shaped anything -- using the CURRENT
# (warm-started) actor, with no PPO noise/updates yet -- then set lam_l1 so the term
# starts on the reward's scale. A SEPARATE WorldModelEnv instance is used so
# calibration never touches the training env's own state.
LAMBDA_L1_PPO = _args.lam_l1 if _args.lam_l1 is not None else 0.05
_CALIBRATE_L1 = (_args.lam_l1 is None) and (not _args.no_calib)
if _CALIBRATE_L1:
    print(f"\ncalibrating lam_l1 over {_args.calib_episodes} episodes x "
          f"{_args.calib_steps} steps (warm-started actor, deterministic, no PPO "
          f"training yet) ...", flush=True)
    _calib_env = WorldModelEnv()
    _rewards, _l1s = [], []
    with torch.no_grad():
        for _ in range(_args.calib_episodes):
            obs = _calib_env.reset()
            for _ in range(_args.calib_steps):
                s = torch.tensor(obs, dtype=dtype, device=device).reshape(1, -1)
                a = BASE(states=s, t=0, p_dropout=0.0).reshape(-1)
                # .tolist() straight off the tensor, not .numpy() -- safe regardless
                # of whether the torch.Tensor.numpy patch below has run yet.
                a_np = np.array(a.detach().cpu().tolist(), dtype=np.float64)
                obs, r, done, _ = _calib_env.step(a_np)
                _rewards.append(r)
                _l1s.append(abs(float(a_np[DISCHARGE_IDX] - A_OFF_Z)))
                if done:
                    break
    _r_calib = float(np.mean(np.abs(_rewards))) if _rewards else float("nan")
    _l1_calib = float(np.mean(_l1s)) if _l1s else 0.0
    if _l1_calib > 1e-8:
        LAMBDA_L1_PPO = _args.l1_ratio * _r_calib / _l1_calib
        print(f"  measured |reward|/step={_r_calib:.4f}  raw l1_disch/step="
              f"{_l1_calib:.4f}  ->  lam_l1 = {_args.l1_ratio} * {_r_calib:.4f} / "
              f"{_l1_calib:.4f} = {LAMBDA_L1_PPO:.4f}", flush=True)
    else:
        print(f"  WARNING: raw l1_disch ~= 0 during calibration (warm start already "
              f"near CLOSED) -- keeping lam_l1={LAMBDA_L1_PPO}", flush=True)
else:
    print(f"\nlam_l1={LAMBDA_L1_PPO} (explicit -lam_l1 or -no_calib; not calibrated)",
          flush=True)

print(f"final action penalty: lam_l1={LAMBDA_L1_PPO:.4f} * |a_disch-{A_OFF_Z:.3f}| "
      f"(L1, from CLOSED)  +  lam_l2={_args.lam_l2} * ||a_other5||^2  (L2)")

env = Monitor(ActionPenaltyWrapper(WorldModelEnv(), LAMBDA_L1_PPO, _args.lam_l2))
model = PPO(
    WarmStartPolicy, env,
    # stable-baselines3 defaults, unmodified
    learning_rate=_args.lr, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(log_std_init=_args.init_log_std),
    verbose=1, device="cpu")

print(f"\\nPPO (stable-baselines3 defaults): n_steps=2048 batch=64 n_epochs=10 "
      f"gamma=0.99 gae_lambda=0.95 clip=0.2 lr={_args.lr}")
print(f"  log_std_init={_args.init_log_std} -> sigma={np.exp(_args.init_log_std):.4f} "
      f"z-units; at -inf the actor IS the imitation policy")
print(f"  episode = {EPISODE_STEPS} steps ({EPISODE_STEPS*HOURS_PER_STEP:.0f} h), "
      f"total timesteps = {_args.timesteps}")
print(f"  NOTE: the SMPL paper's own PPO on PenSim scores 2.5231 vs the recipe's "
      f"3.3071, so the value of this stage rests on the warm start\\n")

model.learn(total_timesteps=_args.timesteps, progress_bar=False)

# ------------------------------------------------ save in OUR checkpoint format ---
torch.save({"policy_state_dict": BASE.state_dict(),
            "policy_meta": BASE_CK["policy_meta"],
            "policy_kind": BASE_CK.get("policy_kind", "rbf"),
            "ppo_log_std": float(model.policy.log_std.detach().mean()),
            "ppo_timesteps": _args.timesteps, "kappa": KAPPA,
            "lam_l1_ppo": LAMBDA_L1_PPO, "lam_l2_ppo": _args.lam_l2,
            "init_policy": _args.init_policy,
            "reward_model": _args.reward_model,
            "phase_prefix": _args.phase_prefix,
            "std_obs_mu": STATS["std_obs_mu"].tolist(),
            "std_obs_sd": STATS["std_obs_sd"].tolist(),
            "std_act_mu": STATS["std_act_mu"].tolist(),
            "std_act_sd": STATS["std_act_sd"].tolist()}, _args.out)
model.save(_args.out.replace(".pt", "_sb3"))
print(f"\\nsaved -> {_args.out}   (and the SB3 archive alongside)")
print("The saved file is in the pipeline's own format, so explore_with_policy.py can")
print("load it directly; the mean network carries PPO's updates.")
