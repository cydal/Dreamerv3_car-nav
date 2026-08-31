"""Throughput benchmark. DreamerV3 trains on far fewer env steps than TD3, but
it still needs the env to not be the bottleneck, so measure it explicitly.

Run:  python scripts/benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from car_env import CarNavConfig, CarNavEnv


def timeit(fn, n):
    t0 = time.perf_counter()
    fn(n)
    return time.perf_counter() - t0


def bench(label, cfg, steps=20000):
    t0 = time.perf_counter()
    env = CarNavEnv(cfg)
    construct = time.perf_counter() - t0

    t0 = time.perf_counter()
    env.reset(seed=0)
    first_reset = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(steps, 2)).astype(np.float32)

    n_reset = 0
    t0 = time.perf_counter()
    for i in range(steps):
        _, _, term, trunc, _ = env.step(actions[i])
        if term or trunc:
            env.reset()
            n_reset += 1
    el = time.perf_counter() - t0

    print(f"{label}")
    print(f"   construct {construct*1000:7.1f} ms   first reset {first_reset*1000:7.1f} ms "
          f"(includes erosion)")
    print(f"   {steps:,} steps in {el:5.2f}s -> {steps/el:9,.0f} steps/s "
          f"({el/steps*1e6:5.1f} us/step), {n_reset} resets")
    return steps / el


def main():
    print("=" * 70)
    print("CarNavEnv throughput")
    print("=" * 70)
    r = {}
    r["image+vector"] = bench("image(64) + vector", CarNavConfig())
    r["vector only"] = bench("vector only (no crop rendering)",
                             CarNavConfig(use_image=False))
    r["image only"] = bench("image only", CarNavConfig(use_vector=False))
    r["image 96"] = bench("image(96) + vector", CarNavConfig(image_size=96))
    r["no rotation"] = bench("image(64), axis-aligned crop (no rotation)",
                             CarNavConfig(egocentric_rotation=False))
    r["footprint"] = bench("footprint collision", CarNavConfig(collision_mode="footprint"))

    print("\nsummary (steps/s):")
    for k, v in sorted(r.items(), key=lambda kv: -kv[1]):
        print(f"   {k:16s} {v:9,.0f}")
    print("\nFor reference, DreamerV3 on a single GPU typically consumes a few")
    print("hundred to a few thousand env steps/s, so the env has headroom.")


if __name__ == "__main__":
    main()
