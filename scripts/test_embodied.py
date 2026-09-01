"""Conformance test for the embodied.Env adapter (car_env/embodied_env.py).

Drives CarNav through the exact wrapper stack dreamerv3/main.py:wrap_env
applies before training (NormalizeAction -> UnifyDtypes -> CheckSpaces ->
ClipAction), with a random policy, and lets CheckSpaces - the clone's own
contract checker - catch any dtype/shape/range mismatch with a real error
message. This is rung 1 of the verification ladder in
docs/dreamer-integration-plan.md: prove the adapter before spending any time
on a debug training run.

Run (needs the `dreamer` conda env - car_env plus elements/embodied):
    conda run -n dreamer python scripts/test_embodied.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# main.py needs to import as a package relative to the dreamerv3 clone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "dreamerv3"))

import numpy as np

import embodied
from car_env.embodied_env import CarNav, VARIANTS


def wrap(env):
    """Mirrors dreamerv3/main.py:wrap_env without importing main.py itself
    (main.py does argv parsing and process setup on import that we don't
    want here)."""
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def check_space_contract(space):
    assert space.shape is not None
    assert space.dtype is not None


def run_variant(task, steps=500, seed=0):
    raw = CarNav(task=task, seed=seed)
    env = wrap(raw)

    for key, space in env.obs_space.items():
        check_space_contract(space)
    for key, space in env.act_space.items():
        check_space_contract(space)

    assert set(env.obs_space) & set(env.act_space) == set(), (
        "obs/act key collision", env.obs_space.keys(), env.act_space.keys())
    assert "reward" in env.obs_space
    assert "is_first" in env.obs_space
    assert "is_last" in env.obs_space
    assert "is_terminal" in env.obs_space
    assert "reset" in env.act_space

    rng = np.random.default_rng(seed)
    act_space = env.act_space["action"]
    n_episodes = 0
    n_terminal = 0
    lengths = []
    length = 0

    action = {"reset": True, "action": np.zeros(act_space.shape, np.float32)}
    for i in range(steps):
        obs = env.step(action)

        # CheckSpaces already asserts every returned value is in-space; the
        # asserts below are about *our* contract, which is stricter than the
        # generic space check (e.g. no both-flags-set, log/ keys are scalar).
        assert not (bool(obs["is_last"]) and not bool(obs["is_first"])
                    and length == 0), "reset produced a mid-episode obs"
        assert not (bool(obs["is_terminal"]) and not bool(obs["is_last"])), (
            "is_terminal set without is_last")
        for key, value in obs.items():
            if key.startswith("log/"):
                value = np.asarray(value)
                assert value.ndim == 0, (
                    f"{key} must be scalar for run/train.py's episode logger, "
                    f"got shape {value.shape}")

        length += 1
        if obs["is_first"]:
            length = 1
        if obs["is_last"]:
            n_episodes += 1
            n_terminal += int(bool(obs["is_terminal"]))
            lengths.append(length)
            length = 0

        raw_action = rng.uniform(act_space.low, act_space.high, act_space.shape)
        action = {"reset": False, "action": raw_action.astype(np.float32)}

    print(f"  task={task!r}")
    print(f"    obs keys: {sorted(env.obs_space)}")
    print(f"    act keys: {sorted(env.act_space)}, action space: {act_space}")
    print(f"    {steps} steps -> {n_episodes} episodes "
          f"({n_terminal} terminal / {n_episodes - n_terminal} truncated), "
          f"mean len {np.mean(lengths) if lengths else float('nan'):.1f}")
    env.close()


def main():
    print("=" * 68)
    print("embodied.Env conformance: CarNav through NormalizeAction ->")
    print("UnifyDtypes -> CheckSpaces -> ClipAction")
    print("=" * 68)
    for task in VARIANTS:
        run_variant(task)
    print("\nAll variants passed CheckSpaces with a random policy.")


if __name__ == "__main__":
    main()
