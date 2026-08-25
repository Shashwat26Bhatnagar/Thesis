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
            r = float(rm[0].reshape(-1)[0]
                      - KAPPA * torch.sqrt(rv[0].reshape(-1)[0].clamp_min(1e-12)))
        self.s = self.s + delta
        self.t += 1
        out = bool(self.s.abs().max() > self._lim)
        done = out or self.t >= EPISODE_STEPS
        return (np.asarray(self.s.tolist(), dtype=np.float32), r, done,
                {"out_of_range": out, "t_hours": t_h})


DISCHARGE_IDX = 0
DISCHARGE_OFF_PHYS = 0.0
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
