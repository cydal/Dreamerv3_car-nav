"""Adapter from CarNavEnv to `embodied.Env`, the interface DreamerV3's
`embodied` runtime (github.com/danijar/dreamerv3) trains against.

Why this exists rather than reusing `embodied.envs.from_gym.FromGym`: that
adapter targets the old `gym` package, unpacks the 4-tuple step API, and
folds truncation into `is_terminal` via `info.get('is_terminal', done)`. Our
env returns the Gymnasium 5-tuple and deliberately keeps `terminated` and
`truncated` distinct (see car_env/env.py's docstring), so `FromGym` would
either crash on the tuple shape or silently erase that distinction. This
adapter is the ~80 lines it takes to do it natively instead.

`embodied.Env` has no `reset()`. `step(action)` receives a dict containing a
`reset` key and is responsible for resetting itself when that is set or when
the previous step ended the episode - see `dummy.py` / `crafter.py` in the
embodied clone for the pattern this follows.
"""

import numpy as np

from .config import CarNavConfig
from .env import CarNavEnv

# Named presets selectable via the task string, e.g. --task carnav_vector.
# Kept intentionally small; arbitrary overrides also flow through **kwargs
# from the `env.carnav` config block, so this is just for a few common shapes
# worth a short name.
VARIANTS = {
    "default": {},
    "vector": {"use_image": False},
    "image": {"use_vector": False},
    "multitarget": {"num_targets": 3},
    "footprint": {"collision_mode": "footprint"},
}


class CarNav:
    """embodied.Env over CarNavEnv. Registered in the dreamerv3 clone as
    `carnav`; task strings look like `carnav_default`, `carnav_vector`, etc.
    """

    def __init__(self, task="default", seed=None, **overrides):
        if task not in VARIANTS:
            raise ValueError(
                f"unknown carnav task {task!r}, expected one of {sorted(VARIANTS)}")
        cfg_kwargs = {**VARIANTS[task], **overrides}
        self._cfg = CarNavConfig(**cfg_kwargs)
        self._env = CarNavEnv(self._cfg)
        self._seed = seed
        self._done = True

    @property
    def env(self):
        return self._env

    @property
    def obs_space(self):
        import elements

        spaces = {}
        for key, spec in self._env.observation_spec.items():
            spaces[key] = elements.Space(
                np.dtype(spec["dtype"]), spec["shape"],
                spec.get("low"), spec.get("high"))
        spaces.update({
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            # Per-episode diagnostics, not consumed by the agent (see the
            # 'log/' convention in embodied/core/base.py). These are what
            # notes/concepts/episodes-and-rewards.md calls "the parts" -
            # watching them separately from total score is how you'd catch
            # the agent farming alignment/sensor bonus instead of the goal.
            "log/episode_reward": elements.Space(np.float32),
            "log/distance_to_target": elements.Space(np.float32),
            "log/crashed": elements.Space(bool),
            "log/reached_target": elements.Space(bool),
            "log/reward_step": elements.Space(np.float32),
            "log/reward_progress": elements.Space(np.float32),
            "log/reward_alignment": elements.Space(np.float32),
            "log/reward_clear_sensors": elements.Space(np.float32),
        })
        return spaces

    @property
    def act_space(self):
        import elements

        spec = self._env.action_spec
        return {
            "action": elements.Space(np.float32, spec["shape"], spec["low"], spec["high"]),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        if action["reset"] or self._done:
            obs, info = self._env.reset(seed=self._seed)
            self._seed = None  # only force a seed on the very first reset
            self._done = False
            return self._obs(obs, 0.0, info, is_first=True)

        act = np.asarray(action["action"], dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(act)
        self._done = terminated or truncated
        return self._obs(
            obs, reward, info, is_last=self._done, is_terminal=terminated)

    def _obs(self, obs, reward, info, is_first=False, is_last=False, is_terminal=False):
        parts = info.get("reward_parts", {})
        out = dict(obs)
        out.update({
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "log/episode_reward": np.float32(info.get("episode_reward", 0.0)),
            "log/distance_to_target": np.float32(info.get("distance_to_target", 0.0)),
            "log/crashed": bool(info.get("crashed", False)),
            # True only on the final step of an episode that ended by
            # reaching its last target - mirrors log/crashed so both can be
            # read the same way: per-episode 'sum' in embodied's episode
            # aggregator is exactly "did this happen" (0 or 1), and the
            # cross-episode average of that is the rate. Added after the
            # first real training run, where "did it reach the goal" had to
            # be inferred from 1 - crash_rate instead of measured directly -
            # see notes/journal.md, 2026-09-01.
            "log/reached_target": bool(is_last and parts.get("target", 0.0) > 0),
            "log/reward_step": np.float32(parts.get("step", 0.0)),
            "log/reward_progress": np.float32(parts.get("progress", 0.0)),
            "log/reward_alignment": np.float32(parts.get("alignment", 0.0)),
            "log/reward_clear_sensors": np.float32(parts.get("clear_sensors", 0.0)),
        })
        return out

    def render(self):
        from .render import render_topdown
        return render_topdown(self._env)

    def close(self):
        pass

    def __repr__(self):
        return f"CarNav({self._env!r})"
