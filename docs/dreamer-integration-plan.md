# Integrating DreamerV3 with `car_env`

Plan for driving `car_env.CarNavEnv` with the official DreamerV3
implementation cloned at `world_models/dreamerv3`
(`danijar/dreamerv3`, commit `e3f0224`, version 3.3.1).

We are **not** reimplementing the algorithm. The work is an adapter plus
configuration, and the plan below is mostly about the constraints this
particular box and this particular task impose.

## 1. What the clone expects from an environment

The JAX/`embodied` rewrite does **not** use the Gymnasium API. From
`embodied/core/base.py` and the reference envs (`embodied/envs/crafter.py`):

- An env subclasses `embodied.Env` and exposes `obs_space` / `act_space` as
  dicts of `elements.Space`.
- There is **no `reset()`**. `step(action)` receives a dict containing a
  `reset` key, and resets itself when that is set or when the previous step
  ended the episode.
- `obs_space` must contain `reward`, `is_first`, `is_last`, `is_terminal`.
- `act_space` must contain `reset`.
- Keys prefixed `log/` are dropped before reaching the agent
  (`make_agent` filters them) and are logged instead.

### `from_gym.py` is not usable

The obvious shortcut — wrapping our env in `embodied.envs.from_gym.FromGym` —
does not work, for three separate reasons:

1. It imports `gym`, not `gymnasium`.
2. It uses the **4-tuple** step API (`obs, reward, done, info`) and expects
   `reset()` to return observations only. Our env returns the Gymnasium
   5-tuple and `(obs, info)`, so it would raise on unpacking.
3. It derives `is_terminal` from `info.get('is_terminal', done)`, collapsing
   truncation into termination — exactly the distinction we deliberately kept.

So: write a native `embodied.Env` adapter. It is ~60 lines.

### The termination mapping is the payoff

`embodied` splits episode-end into two flags, which lines up exactly with the
Gymnasium distinction we preserved when extracting the env:

| our env | `is_last` | `is_terminal` | meaning |
|---|---|---|---|
| `terminated=True` (crash) | `True` | `True` | true terminal, value 0 |
| `truncated=True` (step limit) | `True` | `False` | cutoff, keep bootstrapping |
| neither | `False` | `False` | mid-episode |

`is_terminal` feeds the continuation head (`conhead`) and the discount used in
imagination. Had we kept T3D's single `done` flag, we would be teaching the
world model that running out of time is a terminal state.

### Observation routing is automatic

`dreamerv3/agent.py:21`:

```python
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3
```

Any uint8 3-D space goes to the CNN, everything else to the MLP. So our
existing `image (64,64,3) uint8` and `vector (11,) float32` are routed
correctly with no configuration.

Image size is constrained by the encoder: `mults: [2,3,4,4]` means four
successive 2× max-pools, and `rssm.py` asserts the result is between 3 and 16
per side. **64 → 4×4 ✓** (and 96 → 6×6 ✓). Any multiple of 16 in 48–256 works.

## 2. Where the adapter should live

`make_env` in `dreamerv3/main.py` resolves the env class from a **hardcoded
dict** keyed by the task-name prefix. There is no plugin hook, so a zero-edit
integration is not possible. Two options:

- **A.** Copy our adapter into `embodied/envs/carnav.py` and register it.
  Puts our code inside the clone, so it drifts from our repo and is not
  covered by our tests.
- **B (recommended).** Keep the adapter in **our** package as
  `car_env/embodied_env.py`, and add a one-line registration in the clone
  pointing at it:

  ```python
  'carnav': 'car_env.embodied_env:CarNav',
  ```

B keeps the env code versioned and tested alongside the env it wraps, while
still getting all of `main.py`'s config plumbing (`--configs carnav`, flag
overrides, checkpointing) for free. The clone's diff stays at ~10 lines, on a
branch, easy to rebase onto upstream.

Requires `car_env` to be importable — `pip install -e dreamer-car-nav`.

## 3. Box constraints — these bite before anything else

Measured on this machine, not assumed.

### The default config will OOM immediately

`replay.size: 5e6` and the replay holds chunks **in RAM**
(`embodied/core/replay.py`: `self.chunks = {}`; `directory` is for
checkpointing, not paging). Our observation is 64·64·3 = 12,288 B per step,
so:

| `replay.size` | RAM for replay |
|---|---|
| 5e6 (default) | **~60 GB** |
| 1e6 | ~12 GB |
| 3e5 | ~3.7 GB |
| 2e5 | ~2.5 GB |

This box has **15 GB total RAM, ~11 GB available**. Start at
`replay.size: 2e5` and raise it only if headroom allows. This is the single
most likely cause of an early crash.

### 3a. cuDNN version drift breaks the first conv op (found by actually running it)

`jax-cuda12-plugin==0.4.33` declares `nvidia-cudnn-cu12<10.0,>=9.1` — a range
spanning roughly two years of cuDNN releases. Plain `pip install` resolves to
whatever is newest *at install time*, not whatever jaxlib 0.4.33 (~Sept 2024)
was actually built and tested against. On this box that was cuDNN 9.25.1.1,
and the first convolution (the encoder's first `Conv2D`, i.e. the first time
`image` touches the network) fails:

```
Unable to load any of {libcudnn_engines_runtime_compiled.so.9.25.1, ...}
RuntimeError: ... CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
```

No amount of reading the source would have surfaced this — it's a runtime
version-compatibility gap between two packages that individually declare
themselves compatible. Found by actually running a GPU op, not by reasoning
about it. Fixed by pinning `nvidia-cudnn-cu12==9.4.0.58` (contemporaneous
with jaxlib 0.4.33) in `requirements.txt`, verified first with a standalone
conv op (fast to iterate) before re-running the full pipeline.

### 3b. Intermittent native SIGSEGV during GPU training — worked around, not fixed

The first real (non-throughput-check) training run segfaulted at step 4,528,
98 seconds in — exit 139, no Python traceback, a native crash below JAX's
error handling. Not yet root-caused. Ruled out `agent.report()` as the
trigger (the natural first suspect, since every earlier successful run had
disabled it): `report_every` is a wall-clock threshold and the crash
happened well before it could have fired, confirmed by zero `report/` rows
in that run's `metrics.jsonl`. Leading suspect instead is the `cuDNN
heuristics didn't work, trying fallback algorithms` warning seen on every
run (§3a) — fallback conv kernels are less exercised than heuristic-selected
ones, and this box's driver (595.91.07) is far newer than the jaxlib
0.4.33 / cuDNN 9.4.0.58 pairing pinned in §3a.

Rather than spend more time isolating a native GPU crash with no
reliable repro, worked around it: `scripts/train_carnav.sh` retries the
identical `main.py` invocation on any nonzero exit. This needs no special
crash-recovery logic because `elements.Checkpoint.load_or_save` already
resumes from the last checkpoint in `--logdir` on construction — the
resume path already existed, it just wasn't being exercised across process
restarts. Use a low `--run.save_every` (120s used here) so a crash costs
minutes, not the whole run.

### bfloat16 is not native on this GPU

The GPU is a **Tesla T4, compute capability 7.5 (Turing)**. Hardware bf16
starts at Ampere (8.0). The default is `jax.compute_dtype: bfloat16`, which on
Turing is emulated and slow. Set `jax.compute_dtype: float32`.

Note the interaction flagged in `main.py`: prioritized replay is incompatible
with low-precision gradient scaling. We are on `fracs.uniform: 1.0`
(no prioritization), so this does not apply either way.

### The default model size is ~200M parameters

`defaults` uses `deter: 8192, units: 1024, depth: 64` — the `size200m` preset.
That is aimed at Atari/Minecraft, not a 64×64 navigation task, and is a poor
fit for 15 GB of VRAM alongside a float32 compute dtype. Start at **`size12m`**
(`deter: 2048, hidden: 256, classes: 16, depth: 16, units: 256`) and scale up
only if the world model underfits.

### `run.debug` defaults to `True`

`Driver(fns, parallel=not args.debug)`. With the shipped default, all 16 envs
run **in-process on threads**, GIL-bound. Real runs want
`--run.debug False`, which switches to one process per env (cloudpickled ctor
via `portal.Process`) — so the `make_env` callable must be picklable, i.e. a
module-level class, not a closure over local state.

16 env processes each load their own `CityMap` (~8 MB: RGB + road mask +
erosion caches + road coordinate list) ≈ 130 MB. Acceptable.

### The `numpy<2` pin is probably not binding on us

`requirements.txt` pins `numpy<2`, but the inline comment gives the reason:
`# DMLab: <2, MineRLv1.0: <1.24`. We use neither. JAX itself supports NumPy 2,
and `car_env` was built and verified on 2.2.6.

**Action:** try NumPy 2 first. If something in `embodied` breaks, fall back to
`<2` and re-run our smoke test to confirm `car_env` is clean on 1.x — it uses
only long-stable APIs (`asarray`, `rint(out=)`, `take`, `roll`, `default_rng`),
so this should be a non-event either way.

## 4. The real modelling risk: episodes are too short

Under a random policy, median episode length is **8 steps** (mean 11.8).
Meanwhile `batch_length: 64` and `replay_context: 1`, so each training
sequence is 65 steps and would span roughly **eight episode boundaries**.

Dreamer is correct in this regime — `is_first` resets the RSSM state — but it
is wasteful: the world model spends most of its capacity modelling resets, and
`imag_length: 15` means nearly every imagined rollout runs past a terminal.

Mitigations, in order of preference:

1. **Expect this to fix itself.** The reward already shapes progress and
   alignment, so episodes should lengthen quickly once the policy is better
   than random. Verify by watching `episode/length`.
2. **Lower `batch_length` to 16–32 initially**, so sequences sit closer to
   actual episode length.
3. If episodes stay short, revisit the task rather than the algorithm — this
   is the same geometry problem behind the 6-step median in the original T3D
   run (see the scale table in the README).

Worth stating plainly: this is a property of the task we inherited, and it is
the most likely reason a technically correct integration still fails to learn.

## 5. Verification ladder — status: rungs 1–4 done, all green

Each rung is cheap and isolates one failure mode. Do not skip to the last.
Results below are from actually running each rung, not from planning them.

1. **Adapter conformance** ✅ — `scripts/test_embodied.py` builds the
   adapter, runs it through `main.wrap_env` (`NormalizeAction`,
   `UnifyDtypes`, `CheckSpaces`, `ClipAction`) and drives random actions for
   500 steps across all 5 config variants (`default`, `vector`, `image`,
   `multitarget`, `footprint`). All passed `CheckSpaces` — the clone's own
   contract checker — on the first attempt. Also asserts `log/` values are
   scalars, per `run/train.py`'s `value.ndim == 0` check.
2. **CPU debug run** ✅ — `--configs carnav debug --run.steps 300` (plus
   explicit tiny `agent.dyn.rssm.*`/`enc.simple.depth`/`dec.simple.depth`
   overrides, since `debug`'s regex overrides don't touch `rssm.deter` or
   CNN `depth` — see the note under work item 5). Ran the **full** pipeline
   end to end: driver → replay → stream → `agent.train` → checkpoint →
   logger, with real losses on every head (`con`, `dyn`, `image`, `policy`,
   `rep`, `repval`, `rew`, `value`, `vector`). Produced `scores.jsonl`,
   `metrics.jsonl`, a checkpoint, and — unprompted — a `policy_image.mp4`
   scope artifact (see rung 5; it wasn't supposed to wait until rung 5).
3. **Random agent** ✅ — `--random_agent True --configs carnav debug`.
   Episode length 7, matching the standalone-env baseline. Confirms driver/
   replay/streams/logging all work with the agent removed.
4. **Short GPU run** ✅, after one real fix — `--configs carnav
   --run.steps 3000` (thread-based envs; process-per-env `--run.debug False`
   not yet tried). First attempt failed immediately with
   `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` — see the new §3a below, since
   this is a box/environment finding as significant as the RAM and dtype
   ones. After the fix: 3000 steps completed cleanly, peak **8.3 GB** VRAM
   (comfortably under 15 GB), **fps/train ≈ 900–1120**, GPU util 100% during
   training. `fps/policy` came in low (≈4) — see the train_ratio finding
   below, which is a tuning question, not a failure.
5. **Learning check** ✅ run, result: **not learning the actual task**. A
   60,000-step run (§5b) shows every naive signal improving (score, episode
   length, image-reconstruction loss) while the metric that actually
   matters — crash rate — stays at 88–100% throughout, and
   `distance_to_target` gets worse, not better. This is a reward-design
   finding, not an integration failure; see §5b.

Baselines to beat, from `scripts/benchmark.py` and `scripts/smoke_test.py`:
random policy median length 8, and the env sustains ~3,600 steps/s, so
DreamerV3 (a few hundred to a few thousand steps/s) will be the bottleneck,
not us.

### 5a. Resolved: lowered `train_ratio` from 256 to 32

The 3000-step GPU run's `fps/policy` (≈4) is not a simulator limit — the
standalone env does ~3,600 steps/s (`scripts/benchmark.py`). It was the
`carnav` config's `run.train_ratio: 256`, copied by analogy from
`atari100k` (a *data-starved* regime: a 100k-frame budget on an
expensive-to-step emulator, worth spending disproportionate compute per
frame — the opposite of our situation). Measured rather than assumed:
`fps/train` (GPU throughput) came out about the same at both ratios
(~900-1300 samples/s) — the GPU wasn't idle, the *ratio* was just spending
that same compute on fewer, more diverse env steps. Changed to **32**
(matching what `atari`/`dmlab` already use for the same real-time,
cheap-to-step situation) — `fps/policy` went to ~30-37, roughly 8x more
distinct experience per wall-clock second. See `notes/journal.md`,
2026-09-01, for the full reasoning.

### 5b. The learning-check run: score went up 40x, crash rate barely moved

60,000-step run, `carnav` preset with `train_ratio: 32`. Survived 3 native
segfaults via `scripts/train_carnav.sh`'s retry wrapper (root cause still
unknown — see §3a/§3b) and completed all 60,000 steps on attempt 4.

Naive read: episode length 12 → 90-236 (6-15x), score ~0 → +500..+2,240
(40x+), `train/loss/image` 1950 → 687. Every one of these says "it's
learning."

Correct read, from `scripts/analyze_run.py` (after fixing a metric-reading
bug — see its commit message): **crash rate 100% in the first third of the
run, 88.3% in the last third.** No real change.
`reward_progress` got *more* negative (-0.43 → -2.95/step),
`reward_alignment` too (-0.05 → -0.38/step), `distance_to_target` got worse
(220 → 266). Only `reward_clear_sensors` improved (6.9 → 11.8/step).
Cross-checked visually: 8 frames sampled across the last training video
(from `scope/epstats-policy_image.mp4`) all show the car centered in an
open, straight road corridor, goal marker in none of them.

**What happened:** the agent learned to survive longer in open corridors —
real behavior, not noise — but that increases score mechanically, not
because it's closer to solving the task. `reward_per_clear_sensor: 2.5`
across up to 7 sensors gives up to **+17.5 every step**, unbounded with
episode length; `reward_target: 100` is a **one-time** bonus. Past ~6 steps
of survival the dense term already exceeds the sparse one; at 200+ steps
it dwarfs it. DreamerV3 optimized the reward function exactly as written —
the reward function doesn't actually ask for the goal strongly enough
relative to loitering safely.

**Resolved: lowered `reward_per_clear_sensor` from 2.5 to 0.01.** Offered
three candidates (lower this value, cap total per-step shaping, raise the
progress/alignment scales); the surgical option was chosen. Derivation:
`7 * reward_per_clear_sensor` (max sensor income/step) should sit below
`|reward_step|` (0.1) so loitering nets negative, not positive — break-even
is `0.1/7 ≈ 0.0143`; 0.01 leaves margin. Also added `log/reached_target`
to `car_env/embodied_env.py`, closing the observability gap this
investigation ran into (goal-reaches had to be inferred from
`1 - crash_rate` rather than measured). Verified via `test_embodied.py`
before the fix, `smoke_test.py` after (random-policy mean return flips
+68.1 → -56.6, confirming existing is no longer profitable on its own).

**Re-ran the 60,000-step check with the fix.** Honest result, not a clean
win: crash rate barely moved (99.8% → 96.9%), but the hack is structurally
gone (`clear_sensors`/step: 0.03 → 0.04, negligible either way) and there's
real directional signal that wasn't visible before — `reward_progress`/step
flips -0.79 → **+0.16**, `reward_alignment`/step flips -0.08 → **+0.02**,
goal-reach rate 0.2% → 3.1% (15x, measured directly this time via
`log/reached_target`). The task remains almost entirely unsolved at this
training budget. This is the reward function working correctly, not the
integration failing — the old result was a false positive; this one is an
honest, unflattering, and still-improving-in-the-right-direction baseline.
Given the road/car geometry (14px median road, 28x16 car, `center`
collision already the easier setting — see the scale table in
`README.md`), a harder task than the reward bug's absence should have
been expected.

**500,000-step run (8.3x longer): unambiguous learning.** Same config,
1 crash (retry wrapper handled it), ~4h16m wall clock.

| | first third | last third |
|---|---|---|
| crash rate | 98.5% | **86.8%** |
| goal-reach rate | 1.5% | **13.2%** |
| episode length | 11.9 | 69.3 |
| `distance_to_target` | 218.7 | 206.8 |

Episode score climbs in a near-monotonic staircase across all 10 buckets
of the run (-101 → ... → +414), unlike the 60k run's noise around a flat
mean — the 60k run wasn't wrong, it was just too short to show a trend.
Goal-reach rate is a real 9x improvement, measured directly via
`log/reached_target`, not inferred. Visually confirmed: the goal marker
appears in 3 of 12 sampled frames from the final training video, versus
0 of 8 from the reward-hack run's video — the agent is visibly navigating
toward visible goals now, not cruising empty corridors.

Not a solved task (86.8% crash rate is still most episodes failing), but
this is the first run in the integration where the naive metrics (score,
length) and the metrics that actually matter (crash rate, goal-reach rate,
distance-to-target) finally agree, and both say "improving." The
remaining gap looks like training budget and task difficulty, not
anything broken in the pipeline itself.

## 6. Work items

| # | Item | Where | Status |
|---|---|---|---|
| 1 | conda env `dreamer` (py3.11) + `pip install -r dreamerv3/requirements.txt` | new env, `t3d` untouched | ✅ |
| 2 | `pip install -e .` so `car_env` is importable | our repo | ✅ |
| 3 | `CarNav(embodied.Env)` adapter | `car_env/embodied_env.py` | ✅ |
| 4 | `pyproject.toml` for our package | our repo | ✅ |
| 5 | Register `carnav` in the ctor dict; add `env.carnav` kwargs and a `carnav` named config | clone, branch `carnav-integration` | ✅ — caught a real bug pre-run: an early draft put `task:` inside `env.carnav`, which collides with `make_env`'s positional `task` argument (`TypeError: got multiple values for argument 'task'`) |
| 6 | Conformance test through `wrap_env` + `CheckSpaces` | `scripts/test_embodied.py` | ✅ — 5/5 variants clean |
| 7 | Walk the verification ladder | — | ✅ rungs 1–4, see §5 |
| 8 | Pin `nvidia-cudnn-cu12` (found while doing #7, not planned) | clone `requirements.txt`, same branch | ✅ — see §3a |

Items 1–2 are setup, 3–5 are the integration proper, 6–7 are verification.
Nothing here writes any part of the DreamerV3 algorithm.

## 7. 2026-09-02: switched to footprint collision, tried 4 tuning levers

Decision (made 2026-09-01, implemented 2026-09-02): the 500k-step run's
task — 28×16 car, `collision_mode="center"` — let the car visibly clip
buildings and pass through gaps narrower than itself. Switched to a
harder, more physically honest task: car shrunk to **10×6**,
`collision_mode="footprint"`. 12.8% of road survives erosion at that size
("tight" per the table in `README.md`, re-verified against the live
`CityMap` before committing to the number). The previous checkpoint
doesn't transfer — this is a fresh task.

Also fixed `scripts/watch_policy_live.py`'s view: it was showing
`render_topdown`'s car-centred crop, not the whole map the user expected
(the same view the T3D GUI showed). Added `render_fullmap()` to
`car_env/render.py`; it's now the default `topdown_mode`.

Four tuning levers tried, three kept, one didn't do what expected two did:

| lever | result |
|---|---|
| JAX compilation cache | ✅ real: ~67s cold → ~24s warm compile. Now default in `train_carnav.sh`. |
| `reward_progress_scale`/`reward_alignment_scale` (15→40, 3→8) | applied; untested until the next run's results come in |
| `size25m` | ❌ reverted — OOMs under `prealloc=False`; under `prealloc=True`, both `fps/policy` and `fps/train` come out *worse* than `size12m`. |
| `run.debug: False` | ❌ reverted — no improvement at steady state. `agent.policy()`/`agent.train()` run in the main process regardless of env-stepping mechanism, so this only parallelizes something that was never the bottleneck. |

Reasoning for both reversions kept inline in `dreamerv3/configs.yaml` so
they don't get retried blind. Relaunched training into
`/home/ubuntu/dreamer_runs/carnav_footprint/run1` — **not `/tmp`**, which
was wiped by a box restart overnight and cost us the previous 500k-step
checkpoint. `/home/ubuntu/dreamer_runs/` is now the persistent home for
logdirs and the JAX cache.

**Result: the best run yet, on a harder task.** Zero crashes across the
whole 500k steps (JAX compilation cache + zero native segfaults this
time). Crash rate 98.5%→91.4%, goal-reach rate 1.5%→**8.6%**,
`reward_progress`/step -0.00→**+1.02**, `reward_alignment`/step
0.01→**+0.14**. Score climbs in an almost perfectly monotonic staircase
across all 10 buckets — cleaner than the previous (easier-task) run.
Smaller absolute improvement than the center-collision run (which reached
13.2% goal-reach) is expected, not concerning: footprint collision on a
10×6 car is strictly harder than center collision on a 28×16 car.
Confirmed the full-map render fix applies to `watch_policy.py`'s saved
contact sheets too — see `notes/media/footprint_500k_*.png`.
