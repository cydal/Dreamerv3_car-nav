"""CarNavEnv - the car navigation environment, with no learning algorithm in it.

Deliberately algorithm-agnostic and dependency-light (numpy + pillow). It knows
nothing about replay buffers, actors, critics, world models or Qt. Wrappers for
whichever DreamerV3 implementation we settle on go in a separate module.

API follows the Gymnasium 5-tuple convention:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

`terminated` means the episode ended for an in-world reason (crash, or the
final goal reached). `truncated` means the step limit ran out. Keeping these
distinct matters: bootstrapping the value of a truncated state is correct,
bootstrapping a crash is not.

Actions are normalised to [-1, 1] in both dimensions:
    action[0] -> steering, scaled to +/- max_steering_deg
    action[1] -> speed,    mapped onto [min_speed, max_speed]
"""

import numpy as np

from .citymap import CityMap
from .config import CarNavConfig
from .observations import (EgocentricRenderer, goal_features, sensor_distances)


class CarNavEnv:
    def __init__(self, config=None, **overrides):
        self.cfg = config or CarNavConfig(**overrides)
        if config is not None and overrides:
            raise ValueError("pass either a config object or keyword overrides, not both")

        self.map = CityMap(
            self.cfg.map_path,
            road_min_channel=self.cfg.road_min_channel,
            road_max_spread=self.cfg.road_max_spread,
        )

        self._renderer = EgocentricRenderer(
            self.cfg.image_size, self.cfg.crop_world_px,
            rotate=self.cfg.egocentric_rotation,
        ) if self.cfg.use_image else None

        # Car footprint corner offsets in the car's own frame, used for
        # "footprint" collision checking.
        hl, hw = self.cfg.car_length / 2.0, self.cfg.car_width / 2.0
        self._corners = np.array(
            [[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw],
             [hl, 0.0], [-hl, 0.0], [0.0, hw], [0.0, -hw]], dtype=np.float32)

        self._rng = np.random.default_rng()
        self.x = self.y = 0.0
        self._prev_x = self._prev_y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.targets = []
        self.target_idx = 0
        self.steps = 0
        self.episode_reward = 0.0
        self._prev_dist = None
        self._crashed = False
        self._last_parts = {}
        self._road_dist_field = None
        self._road_dist_stride = None
        self._prev_road_dist = None

    # -------------------------------------------------------------- specs
    @property
    def action_spec(self):
        return {"shape": (2,), "low": -1.0, "high": 1.0, "dtype": "float32",
                "names": ("steering", "speed")}

    @property
    def observation_spec(self):
        spec = {}
        if self.cfg.use_image:
            spec["image"] = {"shape": (self.cfg.image_size, self.cfg.image_size, 3),
                             "dtype": "uint8", "low": 0, "high": 255}
        if self.cfg.use_vector:
            spec["vector"] = {"shape": (self._vector_dim(),), "dtype": "float32"}
        return spec

    def _vector_dim(self):
        # sin(bearing), cos(bearing), norm_dist, norm_speed
        n = 4
        if self.cfg.use_sensors:
            n += len(self.cfg.sensor_angles_deg)
        if self.cfg.use_road_distance:
            n += 1
        return n

    # -------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.x, self.y = self.map.sample_road_point(
            self._rng, clearance=self.cfg.spawn_clearance,
            margin=int(self.cfg.crop_world_px // 4),
        ) if self.cfg.random_start else (self.map.width / 2, self.map.height / 2)

        self._prev_x, self._prev_y = self.x, self.y
        self.heading = (float(self._rng.uniform(0, 360))
                        if self.cfg.random_heading else 0.0)
        self.speed = self.cfg.min_speed

        self.targets = self._sample_targets()
        self.target_idx = 0
        self.steps = 0
        self.episode_reward = 0.0
        self._crashed = False
        self._last_parts = {}
        self._prev_dist = self._distance_to_target()
        if self.cfg.use_road_distance:
            self._refresh_road_distance_field()
            self._prev_road_dist = self._road_distance_to_target()

        return self._observe(), self._info()

    def _refresh_road_distance_field(self):
        tx, ty = self.targets[self.target_idx]
        self._road_dist_field, self._road_dist_stride = (
            self.map.road_distance_field(tx, ty, stride=self.cfg.road_distance_stride))

    def _road_distance_to_target(self):
        return self.map.road_distance_lookup(
            self._road_dist_field, self._road_dist_stride, self.x, self.y,
            fallback=self._distance_to_target())

    def _sample_targets(self):
        """Chain targets outward from the car, each a hop from the previous."""
        n = self.cfg.num_targets
        if self.cfg.randomize_num_targets:
            n = int(self._rng.integers(1, max(1, n) + 1))
        targets = []
        ax, ay = self.x, self.y
        for _ in range(max(1, n)):
            pt = self.map.road_points_within(
                ax, ay, self.cfg.target_min_dist, self.cfg.target_max_dist,
                self._rng, clearance=self.cfg.spawn_clearance)
            if pt is None:
                # Annulus empty (car in an isolated pocket): fall back to any
                # sufficiently clear road pixel so reset never fails.
                pt = self.map.sample_road_point(
                    self._rng, clearance=self.cfg.spawn_clearance)
            targets.append(pt)
            ax, ay = pt
        return targets

    # --------------------------------------------------------------- step
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError(f"expected action of shape (2,), got {action.shape}")
        a = np.clip(action, -1.0, 1.0)

        self.heading += float(a[0]) * self.cfg.max_steering_deg
        self.heading %= 360.0
        span = self.cfg.max_speed - self.cfg.min_speed
        self.speed = self.cfg.min_speed + (float(a[1]) + 1.0) * 0.5 * span

        self._prev_x, self._prev_y = self.x, self.y
        rad = np.radians(self.heading)
        self.x += float(np.cos(rad)) * self.speed
        self.y += float(np.sin(rad)) * self.speed
        self.steps += 1

        reward, terminated, parts = self._reward_and_termination()
        truncated = (not terminated) and self.steps >= self.cfg.max_episode_steps

        self.episode_reward += reward
        self._last_parts = parts
        return self._observe(), reward, terminated, truncated, self._info()

    def _reward_and_termination(self):
        cfg = self.cfg
        parts = {"step": cfg.reward_step, "clear_sensors": 0.0,
                 "progress": 0.0, "alignment": 0.0, "crash": 0.0,
                 "target": 0.0, "danger": 0.0, "caution": 0.0}

        if self._collided():
            self._crashed = True
            parts["crash"] = cfg.reward_crash
            # Crash overrides shaping entirely, as in the original.
            return cfg.reward_crash, True, parts

        dist = self._distance_to_target()
        if dist < cfg.target_radius:
            parts["target"] = cfg.reward_target
            if self.target_idx < len(self.targets) - 1:
                self.target_idx += 1
                self._prev_dist = self._distance_to_target()
                if cfg.use_road_distance:
                    self._refresh_road_distance_field()
                    self._prev_road_dist = self._road_distance_to_target()
                return cfg.reward_target, False, parts
            return cfg.reward_target, True, parts

        reward = cfg.reward_step
        if cfg.use_sensors:
            sensor_norm = sensor_distances(
                self.map, self.x, self.y, self.heading,
                cfg.sensor_angles_deg, cfg.sensor_max_range, cfg.sensor_step_px)
            parts["clear_sensors"] = float(sensor_norm.mean()) * cfg.reward_clearance_scale
            reward += parts["clear_sensors"]

            min_norm = float(sensor_norm.min())
            min_px = min_norm * cfg.sensor_max_range
            if min_px < cfg.danger_margin_px:
                parts["danger"] = -cfg.reward_danger_scale * (
                    (cfg.danger_margin_px - min_px) / cfg.danger_margin_px)
                reward += parts["danger"]
            if min_norm < cfg.caution_threshold:
                speed_span = cfg.max_speed - cfg.min_speed
                norm_speed = ((self.speed - cfg.min_speed) / speed_span
                              if speed_span > 0 else 0.0)
                parts["caution"] = cfg.reward_caution_scale * (1.0 - norm_speed)
                reward += parts["caution"]

        if cfg.use_road_distance:
            road_dist = self._road_distance_to_target()
            if self._prev_road_dist is not None:
                parts["progress"] = ((self._prev_road_dist - road_dist)
                                      * cfg.reward_progress_scale)
                reward += parts["progress"]
            self._prev_road_dist = road_dist
        elif self._prev_dist is not None:
            parts["progress"] = (self._prev_dist - dist) * cfg.reward_progress_scale
            reward += parts["progress"]
        self._prev_dist = dist

        sin_b, cos_b, _, _ = goal_features(
            self.x, self.y, self.heading, self.targets[self.target_idx])
        # cos(bearing) is +1 pointing straight at the goal, -1 directly away.
        parts["alignment"] = cos_b * cfg.reward_alignment_scale
        reward += parts["alignment"]

        return reward, False, parts

    # ---------------------------------------------------------- collisions
    def _footprint_clear(self, cx, cy):
        rad = np.radians(self.heading)
        c, s = np.cos(rad), np.sin(rad)
        # Rotate the footprint offsets into world space in one shot.
        wx = cx + self._corners[:, 0] * c - self._corners[:, 1] * s
        wy = cy + self._corners[:, 0] * s + self._corners[:, 1] * c
        return bool(self.map.is_road(wx, wy).all())

    def _collided(self):
        if self.cfg.collision_mode == "center":
            return not bool(self.map.is_road(self.x, self.y))
        # Swept check: also sample intermediate centres along this step's
        # straight-line displacement, not just the endpoint. A single step
        # can move up to max_speed (2.5px) and the tightest corridors leave
        # only ~6px of footprint clearance, so an endpoint-only check can
        # miss a corner clipped mid-step - this used the current (post-turn)
        # heading throughout, a reasonable approximation since heading
        # changes at most max_steering_deg (8deg) in one step.
        n_sub = 3
        for t in np.linspace(1.0 / n_sub, 1.0, n_sub):
            cx = self._prev_x + (self.x - self._prev_x) * t
            cy = self._prev_y + (self.y - self._prev_y) * t
            if not self._footprint_clear(cx, cy):
                return True
        return False

    # -------------------------------------------------------- observations
    def _distance_to_target(self):
        tx, ty = self.targets[self.target_idx]
        return float(np.hypot(tx - self.x, ty - self.y))

    def _observe(self):
        obs = {}
        target = self.targets[self.target_idx]
        if self.cfg.use_image:
            obs["image"] = self._renderer.render(
                self.map, self.x, self.y, self.heading, target=target,
                draw_target=self.cfg.draw_target_in_image)
        if self.cfg.use_vector:
            sin_b, cos_b, norm_d, _ = goal_features(
                self.x, self.y, self.heading, target)
            span = self.cfg.max_speed - self.cfg.min_speed
            norm_speed = ((self.speed - self.cfg.min_speed) / span) if span > 0 else 0.0
            vec = [sin_b, cos_b, norm_d, norm_speed]
            if self.cfg.use_sensors:
                vec.extend(sensor_distances(
                    self.map, self.x, self.y, self.heading,
                    self.cfg.sensor_angles_deg, self.cfg.sensor_max_range,
                    self.cfg.sensor_step_px).tolist())
            if self.cfg.use_road_distance:
                norm_road_d = min(self._road_distance_to_target()
                                   / self.cfg.road_distance_norm, 1.0)
                vec.append(norm_road_d)
            obs["vector"] = np.asarray(vec, dtype=np.float32)
        return obs

    def _info(self):
        return {
            "x": self.x, "y": self.y, "heading": self.heading, "speed": self.speed,
            "steps": self.steps, "episode_reward": self.episode_reward,
            "target_idx": self.target_idx, "num_targets": len(self.targets),
            "targets_reached": self.target_idx + (1 if self._reached_final() else 0),
            "distance_to_target": self._distance_to_target(),
            "crashed": self._crashed,
            "reward_parts": dict(self._last_parts),
        }

    def _reached_final(self):
        return (not self._crashed
                and self.target_idx == len(self.targets) - 1
                and self._distance_to_target() < self.cfg.target_radius)

    # ------------------------------------------------------------ plumbing
    @property
    def state(self):
        """Full sim state, for the debug renderer and for reproducing a step."""
        return {"x": self.x, "y": self.y, "heading": self.heading,
                "speed": self.speed, "targets": list(self.targets),
                "target_idx": self.target_idx}

    def __repr__(self):
        return (f"CarNavEnv(map={self.map.width}x{self.map.height}, "
                f"obs={list(self.observation_spec)}, "
                f"collision={self.cfg.collision_mode!r})")
