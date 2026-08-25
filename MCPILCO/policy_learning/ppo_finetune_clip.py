#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
EPISODE_STEPS = 1150
U_MAX = 3.0
KAPPA = 1.0

_ap = argparse.ArgumentParser("PPO fine-tuning inside the world models")
_ap.add_argument("-phase_prefix", required=True)
_ap.add_argument("-reward_model", required=True)
_ap.add_argument("-init_policy", required=True, help="the imitation-trained policy")
_ap.add_argument("-out", required=True)
_ap.add_argument("-timesteps", type=int, default=50000)
_ap.add_argument("-kappa", type=float, default=None)
_ap.add_argument("-lr", type=float, default=3e-4)
_ap.add_argument("-no_clip", action="store_true",
                 help="train WITHOUT the deployment clip (then the policy is again "
                      "optimising an action it will not deploy)")
_ap.add_argument("-init_log_std", type=float, default=-2.0,
                 help="start close to deterministic: exp(-2)=0.135 in z-units")
_args = _ap.parse_known_args()[0]
if _args.kappa is not None:
    KAPPA = _args.kappa
USE_CLIP = not _args.no_clip
os.makedirs(os.path.dirname(_args.out) or ".", exist_ok=True)


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
    RECIPE_FRAC = 0.10
    RECIPE_FLOOR = 0.05
    RECIPE_SMOOTH_H = 2.0

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)
        self.action_space = spaces.Box(-U_MAX, U_MAX, (INPUT_DIM,), np.float32)
        self._pool0 = MODELS[0].gp_inputs[:, :STATE_DIM]
        self._lim = float(POOL.abs().max()) * self.OUT_OF_RANGE_MULT
        self._rng = np.random.default_rng(0)
        self.s = None
        self.t = 0

        self._clip = None
        if USE_CLIP:
            from pensimpy.examples.recipe import Recipe, RecipeCombo
            from pensimpy.data.constants import (
                FS, FOIL, FG, PRES, DISCHARGE, WATER,
                FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
                PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
                WATER_DEFAULT_PROFILE)
            from policy_learning.chance_constraints import RecipeBounds
            _keys = [DISCHARGE, FS, FOIL, FG, PRES, WATER]
            _rc = RecipeCombo(recipe_dict={
                DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
                FS: Recipe(FS_DEFAULT_PROFILE, FS),
                FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
                FG: Recipe(FG_DEFAULT_PROFILE, FG),
                PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
                WATER: Recipe(WATER_DEFAULT_PROFILE, WATER)})
            self._clip = RecipeBounds(_rc, _keys, pdata.MIN_ACT, pdata.MAX_ACT,
                                      STATS["std_act_mu"], STATS["std_act_sd"],
                                      frac=self.RECIPE_FRAC,
                                      floor_frac=self.RECIPE_FLOOR,
                                      smooth_h=self.RECIPE_SMOOTH_H)
            self.n_clipped = 0
            self.n_steps_total = 0

    def reset(self):
        i = int(self._rng.integers(self._pool0.shape[0]))
        self.s = self._pool0[i].clone()
        self.t = 0
        return np.asarray(self.s.tolist(), dtype=np.float32)

    def step(self, action):
        a_raw = np.asarray(np.asarray(action).tolist(), dtype=np.float64).reshape(-1)
        t_h_now = self.t * HOURS_PER_STEP

        if self._clip is not None:
            lo_n, hi_n = self._clip.at(t_h_now)
            a_phys = ((a_raw * STATS["std_act_sd"] + STATS["std_act_mu"]) + 1.0) / 2.0 \
                * (pdata.MAX_ACT - pdata.MIN_ACT) + pdata.MIN_ACT
            a_cl = np.clip(a_phys, np.asarray(lo_n), np.asarray(hi_n))
            a_cl = np.clip(a_cl, pdata.MIN_ACT, pdata.MAX_ACT)
            self.n_steps_total += 1
            if not np.allclose(a_cl, a_phys):
                self.n_clipped += 1
            a_smpl = 2.0 * (a_cl - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
            a_raw = (a_smpl - STATS["std_act_mu"]) / STATS["std_act_sd"]

        a = torch.tensor(a_raw, dtype=dtype, device=device)
        gp_in = torch.cat([self.s, a]).reshape(1, -1)
        t_h = t_h_now
        with torch.no_grad():
            mdl = MODELS[phase_of(t_h)]
            m_l, v_l = mdl.get_gp_estimate(gp_inputs=gp_in,
                                           gp_index_list=list(range(STATE_DIM)))
            delta = torch.cat([m_l[i].reshape(1) for i in range(STATE_DIM)])
            rm, rv = RMODEL.get_gp_estimate(gp_inputs=gp_in, gp_index_list=[0])
            r = float(rm[0].reshape(-1)[0]
                      - KAPPA * torch.sqrt(rv[0].reshape(-1)[0].clamp_min(1e-12)))
        self.s = self.s + delta
        self.t += 1
        out = bool(self.s.abs().max() > self._lim)
        done = out or self.t >= EPISODE_STEPS
        return (np.asarray(self.s.tolist(), dtype=np.float32), r, done,
                {"out_of_range": out, "t_hours": t_h})


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


_TORCH_TO_NP_DTYPE = {
    torch.float64: np.float64, torch.float32: np.float32, torch.float16: np.float16,
    torch.int64: np.int64, torch.int32: np.int32, torch.int16: np.int16,
    torch.int8: np.int8, torch.uint8: np.uint8, torch.bool: np.bool_,
}


def _laundered_tensor_numpy(self, *args, **kwargs):
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


env = Monitor(WorldModelEnv())
model = PPO(
    WarmStartPolicy, env,
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
print(f"  deployment clip INSIDE the env: {'ON' if USE_CLIP else 'off'}"
      + ("  -- the policy is optimised against the action it will actually deploy"
         if USE_CLIP else "  -- the policy will deploy a DIFFERENT action than it "
                          "trains on"))
print(f"  NOTE: the SMPL paper's own PPO on PenSim scores 2.5231 vs the recipe's "
      f"3.3071, so the value of this stage rests on the warm start\\n")

model.learn(total_timesteps=_args.timesteps, progress_bar=False)

torch.save({"policy_state_dict": BASE.state_dict(),
            "policy_meta": BASE_CK["policy_meta"],
            "policy_kind": BASE_CK.get("policy_kind", "rbf"),
            "ppo_log_std": float(model.policy.log_std.detach().mean()),
            "ppo_timesteps": _args.timesteps, "kappa": KAPPA,
            "trained_with_clip": USE_CLIP,
            "init_policy": _args.init_policy,
            "reward_model": _args.reward_model,
            "phase_prefix": _args.phase_prefix,
            "std_obs_mu": STATS["std_obs_mu"].tolist(),
            "std_obs_sd": STATS["std_obs_sd"].tolist(),
            "std_act_mu": STATS["std_act_mu"].tolist(),
            "std_act_sd": STATS["std_act_sd"].tolist()}, _args.out)
_e = env.unwrapped if hasattr(env, "unwrapped") else env
if USE_CLIP and getattr(_e, "n_steps_total", 0):
    print(f"clip bound on {100*_e.n_clipped/_e.n_steps_total:.1f}% of "
          f"{_e.n_steps_total} env steps")
model.save(_args.out.replace(".pt", "_sb3"))
print(f"\\nsaved -> {_args.out}   (and the SB3 archive alongside)")
print("The saved file is in the pipeline's own format, so explore_with_policy.py can")
print("load it directly; the mean network carries PPO's updates.")
