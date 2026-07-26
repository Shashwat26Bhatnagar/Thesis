import numpy as np, inspect
from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData

sig = inspect.signature(PenSimEnvGym.__init__)
for k in ("observation_dim","action_dim","max_observations","min_observations",
          "max_actions","min_actions","observation_name","action_name",
          "dense_reward","max_steps"):
    if k in sig.parameters:
        print(f"{k:20s} = {sig.parameters[k].default}")

print("\n--- PeniControlData ---")
try:
    d = PeniControlData(dataset_folder="examples/example_batches", normalize=True).get_dataset()
    if d is None:
        print("get_dataset() returned None -> check dataset_folder path")
    else:
        for k, v in d.items():
            print(f"  {k:22s} {getattr(v,'shape',type(v))}")
        o = np.asarray(d["observations"]); a = np.asarray(d["actions"])
        print("  obs min:", np.round(o.min(0),3))
        print("  obs max:", np.round(o.max(0),3))
        print("  act min:", np.round(a.min(0),3))
        print("  act max:", np.round(a.max(0),3))
        if "rewards" in d:
            r = np.asarray(d["rewards"]).reshape(-1)
            no = np.asarray(d.get("next_observations", o))
            for j in range(o.shape[1]):
                if np.allclose(r[:len(no)], no[:len(r), j], atol=1e-5):
                    print(f"  reward == observation index {j}")
            print("  reward stats:", r.min(), r.max(), r.mean())
except Exception as e:
    print("offline dataset failed:", repr(e))
