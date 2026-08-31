# Dreamerv3_car-nav

A standalone 2D car-navigation environment, extracted from the earlier T3D
(TD3) project so it can be driven by any learning algorithm. **This package
contains no learning code** — it is the environment only.

The car drives on the white roads of a city map image. It must reach a sequence
of goal points without leaving the road. Observations are an egocentric camera
crop plus a small goal/sensor vector; actions are continuous steering and speed.

## Why this exists separately

The original `citymap_t3d.py` entangled four things in one file: the simulator,
the TD3 agent, the replay buffer, and a PyQt6 GUI. That made the environment
impossible to reuse and impossible to test. Here they are split:

| | T3D version | this version |
|---|---|---|
| map lookups | one `QColor(img.pixel(x,y))` call per sensor per step | map decoded once into a NumPy RGB array + boolean road mask |
| dependencies | PyQt6 required to simulate at all | numpy + pillow; no Qt, no display |
| observation | 9 floats (angle, −angle, 7 binary sensors) | 64×64×3 egocentric crop + 11-float vector |
| goal encoding | raw normalised angle, discontinuous at ±180° | `sin`/`cos` of bearing, continuous everywhere |
| actions | rotation in degrees, then a convoluted speed remap | both in `[-1, 1]` |
| episode end | single `done` flag | Gymnasium `terminated` / `truncated`, kept distinct |
| config | module-level constants | `CarNavConfig` dataclass, validated |

Keeping `terminated` and `truncated` separate matters for correctness, not
tidiness: a crash is a true terminal state whose value is 0, whereas hitting
the step limit is an arbitrary cutoff and the value function must still
bootstrap from the final state. Collapsing them teaches the agent that running
out of time is as good as parking safely.

## Install

```bash
conda activate t3d
pip install -r requirements.txt   # numpy + pillow only
```

## Use

```python
from car_env import CarNavEnv, CarNavConfig

env = CarNavEnv(CarNavConfig(num_targets=2))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([0.0, 1.0])  # [steering, speed]
```

`obs` is a dict:

| key | shape | dtype | meaning |
|---|---|---|---|
| `image` | `(64, 64, 3)` | `uint8` | egocentric crop, car at centre facing row 0 |
| `vector` | `(11,)` | `float32` | `[sin bearing, cos bearing, norm dist, norm speed] + 7 sensors` |

Either can be switched off (`use_image=False` / `use_vector=False`). The image
alone is enough for a pixel-based world model, but the goal is often outside
the 96px crop, so the bearing/range vector is included by default — no world
model can recover a target the camera never sees.

`info` carries `reward_parts`, a per-term breakdown of the reward, which is the
fastest way to tell whether a policy is being driven by progress, alignment, or
the sensor bonus.

## Scripts

```bash
python scripts/smoke_test.py     # contract assertions + determinism check
python scripts/preview_obs.py    # orientation assertions, writes previews/*.png
python scripts/benchmark.py      # steps/sec
```

`preview_obs.py` is worth running after any change to the renderer. A
transposed or mirrored crop axis is invisible in reward curves but fatal to
learning, so it checks the convention numerically: a landmark placed at a known
bearing must land in the expected third of the crop, for 6 bearings × 5
headings, and the in-image goal marker must fall within 2px of where the goal
geometrically is.

## Throughput

Measured on this box (Tesla T4, 20k random steps), after replacing the
2-D fancy-index gather with a flat `np.take` and preallocating the coordinate
buffers:

| config | steps/s |
|---|---|
| vector only | 6,800 |
| image only | 4,500 |
| image + vector (default) | 3,600 |
| image 96×96 + vector | 2,800 |

DreamerV3 typically consumes a few hundred to a few thousand env steps/s, so
the environment is not the bottleneck.

## Known constraint: the car does not fit the roads

The map's roads have a median width of **14px** while the car is **28×16px**.
This is inherited from the original project and it dictates the collision
model. A car of length L and width W needs `hypot(L/2, W/2)` px of clearance to
occupy a point at any heading; eroding the road mask by that radius gives how
much of the map is legally drivable:

| car size | required clearance | road remaining | verdict |
|---|---|---|---|
| 28×16 (current) | 17px | **0.0%** | impossible |
| 20×12 | 12px | 0.1% | impossible |
| 14×8 | 9px | 0.7% | impossible |
| 10×6 | 6px | 12.8% | tight |
| 8×5 | 5px | 22.5% | workable |

So `collision_mode="footprint"` (the whole car must be on road) is unusable at
the current scale — every episode ends on step 1.

**Decision: stay on `collision_mode="center"`**, which treats the car as a
point, exactly as T3D did. The 28×16 sprite is cosmetic, so the car clips
buildings and can pass through gaps narrower than itself — but the task is
learnable, and this avoids retuning every distance in the config before there
is a baseline to compare against. This is also the same geometry problem behind
the 6-step median episodes in the original T3D run.

`footprint` is kept available for when the scale is revisited. Making it usable
would need the car shrunk to ~10×6, or the map upscaled 3–4× so roads are
~42–56px wide (which would also mean retuning `target_min_dist`,
`target_max_dist`, `sensor_distance` and `crop_world_px`).

## Layout

```
car_env/
  config.py        CarNavConfig dataclass, validated in __post_init__
  citymap.py       map image -> RGB array + road mask; erosion, spawn sampling
  observations.py  egocentric crop renderer, sensors, goal features
  env.py           CarNavEnv: reset/step, rewards, collisions, goal chaining
  render.py        top-down debug rendering (humans only; agent never sees it)
scripts/
  smoke_test.py  preview_obs.py  benchmark.py
assets/
  city_map.png  car.png
```

`car_env` never imports `render.py` at module level, so training processes stay
free of any drawing code.
