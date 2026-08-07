"""
Run the MCPILCO call pattern and the wrapper's, side by side, on the same
first action. Whichever succeeds tells us what differs.

    cd ~/Thesis/sindy-rl
    python -u sindy_rl/scripts/compare_mcpilco.py
"""

import numpy as np
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)
from smpl.envs.pensimenv import PenSimEnvGym

RECIPE = {
    FS: Recipe(FS_DEFAULT_PROFILE, FS),
    FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
    FG: Recipe(FG_DEFAULT_PROFILE, FG),
    PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
    DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
    WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
    PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
}

# Row 0 of random_batch_0.csv -- the default recipe's first action.
RECIPE_ACTION = np.array([0.0, 7.361, 23.214, 31.255, 0.602, 0.0])


def make_env_mcpilco(seed):
    """Exactly as MCPILCO/policy_learning/explore_with_policy.py:202."""
    e = PenSimEnvGym(recipe_combo=RecipeCombo(recipe_dict=RECIPE),
                     normalize=False, random_seed=seed)
    if hasattr(e, 'seed'):
        e.seed(seed)
    return e


def trial(label, seed, action, n=3):
    print(f'\n--- {label} (seed={seed}) ---')
    e = make_env_mcpilco(seed)
    o = np.array(np.asarray(e.reset()).reshape(-1).tolist(), dtype=np.float64)
    print('  reset obs :', np.round(o, 3))
    print('  action    :', np.round(action, 4))
    lo = np.asarray(e.min_actions, dtype=np.float64)
    hi = np.asarray(e.max_actions, dtype=np.float64)
    a = np.clip(np.asarray(action, dtype=np.float64), lo, hi)
    for k in range(n):
        step = e.step(a)
        rew = float(step[1])
        info = step[3] if len(step) == 4 else step[4]
        ok = rew > -99
        print(f'  step {k}: reward={rew:10.4f}  '
              f'error={info.get("error_occurred")}  {"OK" if ok else "FAIL"}')
        if not ok:
            print('    obs returned:', np.round(np.asarray(step[0], float), 3))
            break
    return


def main():
    print('Recipe action (CSV row 0):', RECIPE_ACTION)

    # 1. The exact MCPILCO pattern.
    trial('MCPILCO pattern, recipe action', 0, RECIPE_ACTION)

    # 2. Different seeds -- sample_initial_state draws randint(3, 20000),
    #    so seed 0 may be a special case.
    for s in (3, 42, 1234):
        trial(f'MCPILCO pattern, seed {s}', s, RECIPE_ACTION)

    # 3. Does the action matter at all, or does every action fail?
    e = make_env_mcpilco(42)
    e.reset()
    mid = (np.asarray(e.min_actions, float) + np.asarray(e.max_actions, float)) / 2
    trial('midpoint action', 42, mid)

    # 4. What does the wrapper produce for a zero action?
    try:
        from sindy_rl.pensim_env import PenSimGymEnv
        w = PenSimGymEnv({
            'dataset_folder':
                '/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensim_sindy',
            'max_episode_steps': 3,
        })
        w.reset()
        u = w._to_physical_action(np.zeros(6))
        print('\n--- wrapper physical action for zeros ---')
        print('  ', np.round(u, 4))
        print('   recipe row 0 :', np.round(RECIPE_ACTION, 4))
        print('   match        :', np.allclose(u, RECIPE_ACTION, atol=1e-2))
        trial('wrapper action via MCPILCO pattern', 42, u)
    except Exception as exc:
        print('wrapper check failed:', exc)


if __name__ == '__main__':
    main()
