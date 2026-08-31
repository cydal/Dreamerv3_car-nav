"""Look at what the agent actually sees, and assert the crop is oriented right.

The egocentric crop is the whole input to a pixel-based world model, so a
transposed or mirrored axis here would be invisible in the reward curves and
fatal to learning. This script checks the convention numerically (a landmark
placed ahead of the car must land near the top-centre of the crop) and also
writes PNGs so it can be confirmed by eye.

Run:  python scripts/preview_obs.py [--outdir DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from car_env import CarNavConfig, CarNavEnv, EgocentricRenderer
from car_env.render import render_topdown, save_png, tile


def brightest_cell(img, cells=3):
    """Which cell of a cells x cells grid holds the most non-black pixel mass."""
    S = img.shape[0]
    step = S / cells
    lum = img.astype(np.float64).sum(axis=2)
    best, best_v = None, -1.0
    for r in range(cells):
        for c in range(cells):
            block = lum[int(r * step):int((r + 1) * step),
                        int(c * step):int((c + 1) * step)]
            v = block.sum()
            if v > best_v:
                best_v, best = v, (r, c)
    return best


def check_orientation():
    """Place a bright landmark at a known bearing and find it in the crop.

    Uses a synthetic all-black map with one white blob, so the only bright
    thing in the crop is the landmark.
    """
    class FakeMap:
        def __init__(self, size=400):
            self.width = self.height = size
            self.rgb = np.zeros((size, size, 3), dtype=np.uint8)
            self.rgb_flat = self.rgb.reshape(-1, 3)

        def blob(self, x, y, r=6):
            self.rgb[...] = 0
            yy, xx = np.mgrid[0:self.height, 0:self.width]
            self.rgb[(xx - x) ** 2 + (yy - y) ** 2 <= r * r] = 255

    fm = FakeMap()
    rend = EgocentricRenderer(image_size=64, crop_world_px=96.0, rotate=True)
    cx = cy = 200.0
    d = 30.0   # well inside the 48px half-crop

    # (label, bearing relative to the car, expected (row, col) cell)
    cases = [
        ("ahead",       0.0,   (0, 1)),
        ("behind",    180.0,   (2, 1)),
        ("right",      90.0,   (1, 2)),
        ("left",     -90.0,    (1, 0)),
        ("ahead-right", 45.0,  (0, 2)),
        ("ahead-left", -45.0,  (0, 0)),
    ]

    print("orientation check (car at crop centre, expected to face row 0)")
    ok = True
    # Repeat for several car headings; the result must not depend on heading.
    for heading in (0.0, 37.0, 90.0, 180.0, 271.0):
        for label, rel, expect in cases:
            ang = np.radians(heading + rel)
            fm.blob(cx + np.cos(ang) * d, cy + np.sin(ang) * d)
            img = rend.render(fm, cx, cy, heading, target=None, draw_target=False)
            got = brightest_cell(img)
            good = got == expect
            ok &= good
            if not good:
                print(f"   FAIL heading={heading:5.0f} {label:12s} "
                      f"expected cell {expect} got {got}")
    print(f"   {'OK' if ok else 'FAILED'}: landmark lands in the expected cell "
          f"for all 6 bearings x 5 headings")
    assert ok, "egocentric crop orientation is wrong"


def check_target_stamp():
    """The in-image goal marker must land where the goal geometrically is."""
    cfg = CarNavConfig(target_min_dist=40.0, target_max_dist=45.0)
    env = CarNavEnv(cfg)
    print("\ntarget-stamp check (cyan marker vs true goal bearing)")
    hits = misses = 0
    for seed in range(300):
        obs, info = env.reset(seed=seed)
        tx, ty = env.targets[env.target_idx]
        dx, dy = tx - env.x, ty - env.y
        h = np.radians(env.heading)
        u = dx * -np.sin(h) + dy * np.cos(h)
        v = -(dx * np.cos(h) + dy * np.sin(h))
        half = cfg.crop_world_px / 2.0
        if not (abs(u) < half - 4 and abs(v) < half - 4):
            continue   # goal out of frame, nothing to check
        scale = cfg.image_size / cfg.crop_world_px
        exp_col, exp_row = (u + half) * scale, (v + half) * scale
        cyan = np.argwhere((obs["image"][:, :, 0] == 0) &
                           (obs["image"][:, :, 1] == 255) &
                           (obs["image"][:, :, 2] == 255))
        if len(cyan) == 0:
            misses += 1
            continue
        row, col = cyan.mean(axis=0)
        if abs(row - exp_row) <= 2 and abs(col - exp_col) <= 2:
            hits += 1
        else:
            misses += 1
            if misses <= 3:
                print(f"   FAIL seed={seed} expected (row,col)=({exp_row:.1f},"
                      f"{exp_col:.1f}) got ({row:.1f},{col:.1f})")
    print(f"   {hits} in-frame goals located correctly, {misses} wrong")
    assert misses == 0 and hits > 0, "target marker is drawn in the wrong place"


def write_previews(outdir):
    from PIL import Image

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Same place, eight headings: shows the crop rotating with the car.
    env = CarNavEnv(CarNavConfig())
    env.reset(seed=3)
    x, y = env.x, env.y
    crops, tops, labels = [], [], []
    for hdeg in range(0, 360, 45):
        env.heading = float(hdeg)
        obs = env._observe()
        crops.append(np.asarray(Image.fromarray(obs["image"]).resize(
            (192, 192), Image.NEAREST)))
        tops.append(render_topdown(env, view_px=140, scale=1))
        labels.append(f"h={hdeg}")
    save_png(tile(crops, cols=4, labels=labels), outdir / "crops_by_heading.png")
    save_png(tile(tops, cols=4, labels=labels), outdir / "topdown_by_heading.png")

    # A rollout: top-down beside the agent's view, so the two can be compared.
    env = CarNavEnv(CarNavConfig())
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(0)
    trail, pairs, labels2 = [(env.x, env.y)], [], []
    from PIL import Image
    for i in range(8):
        top = render_topdown(env, view_px=140, scale=1)
        crop = np.asarray(Image.fromarray(obs["image"]).resize(
            (top.shape[1], top.shape[0]), Image.NEAREST))
        pairs.append(np.concatenate([top, crop], axis=1))
        labels2.append(f"t={env.steps}")
        obs, r, term, trunc, info = env.step(
            np.array([rng.uniform(-0.4, 0.4), 1.0], dtype=np.float32))
        trail.append((env.x, env.y))
        if term or trunc:
            break
    save_png(tile(pairs, cols=2, labels=labels2), outdir / "rollout_pairs.png")

    print(f"\nwrote previews to {outdir}/")
    for p in sorted(outdir.glob("*.png")):
        print(f"   {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="previews")
    args = ap.parse_args()

    print("=" * 68)
    print("Observation preview + orientation assertions")
    print("=" * 68)
    check_orientation()
    check_target_stamp()
    write_previews(args.outdir)


if __name__ == "__main__":
    main()
