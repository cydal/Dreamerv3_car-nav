"""Exercise the environment with a random policy and assert the contract holds.

Run:  python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from car_env import CarNavConfig, CarNavEnv


def check_obs(obs, spec):
    assert set(obs) == set(spec), f"obs keys {set(obs)} != spec {set(spec)}"
    for k, s in spec.items():
        a = obs[k]
        assert a.shape == tuple(s["shape"]), f"{k}: shape {a.shape} != {s['shape']}"
        assert a.dtype == np.dtype(s["dtype"]), f"{k}: dtype {a.dtype} != {s['dtype']}"
        assert np.isfinite(a).all(), f"{k}: contains non-finite values"


def run(cfg, episodes, seed, label):
    env = CarNavEnv(cfg)
    spec = env.observation_spec
    rng = np.random.default_rng(seed)

    lengths, returns, outcomes = [], [], {"crash": 0, "goal": 0, "timeout": 0}
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        check_obs(obs, spec)
        assert info["distance_to_target"] > 0

        steps, ret, done = 0, 0.0, False
        while not done:
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            obs, r, terminated, truncated, info = env.step(action)
            check_obs(obs, spec)
            assert np.isfinite(r), "reward must be finite"
            assert not (terminated and truncated), "terminated and truncated both set"
            ret += r
            steps += 1
            done = terminated or truncated
            assert steps <= cfg.max_episode_steps, "ran past the step limit"

        if info["crashed"]:
            outcomes["crash"] += 1
        elif truncated:
            outcomes["timeout"] += 1
        else:
            outcomes["goal"] += 1
        lengths.append(steps)
        returns.append(ret)
        assert abs(ret - info["episode_reward"]) < 1e-3, "return bookkeeping drifted"

    print(f"\n{label}")
    print(f"  {env!r}")
    print(f"  obs spec: " + ", ".join(
        f"{k}{tuple(v['shape'])}:{v['dtype']}" for k, v in spec.items()))
    print(f"  episodes={episodes}  mean len={np.mean(lengths):6.1f}  "
          f"median len={np.median(lengths):5.1f}  max={max(lengths)}")
    print(f"  mean return={np.mean(returns):9.1f}")
    print(f"  outcomes: {outcomes}")
    return lengths, outcomes


def main():
    print("=" * 68)
    print("CarNavEnv smoke test - random policy")
    print("=" * 68)

    run(CarNavConfig(), 40, 0, "default (center collision, sensor_distance=10)")
    run(CarNavConfig(collision_mode="footprint"), 40, 0,
        "footprint collision (whole 28x16 car must be on road)")
    run(CarNavConfig(num_targets=3), 20, 0, "3 chained targets")
    run(CarNavConfig(use_image=False), 20, 0, "vector-only observation")
    run(CarNavConfig(use_vector=False), 20, 0, "image-only observation")

    # Determinism: same seed must reproduce the trajectory exactly.
    def rollout(seed):
        env = CarNavEnv(CarNavConfig())
        env.reset(seed=seed)
        rng = np.random.default_rng(123)
        out = []
        for _ in range(60):
            _, r, t, tr, info = env.step(rng.uniform(-1, 1, 2))
            out.append((round(info["x"], 6), round(info["y"], 6), round(r, 6)))
            if t or tr:
                break
        return out

    a, b, c = rollout(7), rollout(7), rollout(8)
    assert a == b, "same seed produced different trajectories"
    assert a != c, "different seeds produced identical trajectories"
    print("\ndeterminism: same seed reproduces exactly, different seed diverges  OK")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
