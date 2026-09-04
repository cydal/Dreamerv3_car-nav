"""Live top-down view of a trained checkpoint driving CarNavEnv, in a real
window - the DreamerV3-side equivalent of the T3D PyQt6 GUI.

Before each episode (fullmap view only), the car is placed and the view
pauses: left-click the map to add one or more target waypoints - the same
"click a sequence" workflow T3D's GUI used - then right-click or press Enter
to start driving. Press C to clear pending clicks, Q to quit. If you don't
click anything, Enter/right-click on an empty selection is ignored - you
have to place at least one target. This is the main way to check whether a
checkpoint trained with num_targets > 1 has actually learned to continue
past its first target rather than just stopping there.

Needs the persistent display stack from T3D-car-navigation/start_display.sh
(Xvfb on :1, x11vnc, noVNC) already running, and an SSH tunnel to view it
remotely - same connection instructions that setup prints. This script only
needs DISPLAY set; it doesn't touch the display stack itself.

Runs on CPU - eval doesn't need the GPU, and this sidesteps the intermittent
native GPU crash documented in docs/dreamer-integration-plan.md SS3b for a
task that doesn't need it. Renders at roughly human-watchable speed
(--fps, default 8) rather than the env's native throughput, since the point
is for a person to follow the car, not to benchmark anything.

Run (needs the `dreamer` conda env, and DISPLAY set to the Xvfb display):
    DISPLAY=:1 python scripts/watch_policy_live.py \
        --checkpoint /path/to/logdir/ckpt
Press Q or close the window to quit.
"""

import argparse
from pathlib import Path

from _dreamer_common import load_agent  # sets up sys.path; import first

import embodied
import numpy as np
import pygame
from PIL import Image, ImageDraw

from car_env.embodied_env import CarNav
from car_env.render import TARGET_COLOR

PANEL_W = 260
SURVIVAL_COLOR = (60, 220, 60)
DANGER_COLOR = (230, 60, 60)
PANEL_BG = (30, 33, 41)


def _draw_pending(frame, targets, scale):
    """Overlay clicked-but-not-yet-started waypoint markers on a frame."""
    img = Image.fromarray(frame.copy())
    d = ImageDraw.Draw(img)
    for i, (wx, wy) in enumerate(targets):
        cx, cy = wx * scale, wy * scale
        r = 6
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=TARGET_COLOR, width=2)
        d.text((cx + r + 2, cy - r), str(i + 1), fill=TARGET_COLOR)
        if i > 0:
            px, py = targets[i - 1]
            d.line([(px * scale, py * scale), (cx, cy)], fill=TARGET_COLOR, width=1)
    return np.asarray(img)


def _lerp_color(t, hi=SURVIVAL_COLOR, lo=DANGER_COLOR):
    t = max(0.0, min(1.0, t))
    return tuple(int(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))


def imagine_ghosts(agent, carry, cfg, x0, y0, heading0, n_samples, length):
    """Roll the trained world model forward `n_samples` times from the
    car's current belief state, with no real observations - genuine
    imagination, not a scripted preview. Each sample's imagined actions are
    replayed through the same deterministic kinematics CarNavEnv.step uses,
    starting from the car's real current pose, to get a precise path to
    draw - the alternative (decoding position from the vector head's
    predicted bearing/distance) carries reconstruction error the actions
    themselves don't. What's genuinely "imagined" here - and what varies
    seed to seed - is the action sequence and the predicted reward/value/
    continue-probability alongside it, all straight from agent.imagine().

    Returns (paths, values) where each path is (xs, ys, continue_prob) and
    values is the imagined value estimate at the first imagined step.
    """
    paths, values = [], []
    for _ in range(n_samples):
        _, out = agent.imagine(carry, length)
        actions = np.asarray(out["action"]["action"])[0]
        contp = np.asarray(out["continue_prob"])[0]
        values.append(float(np.asarray(out["value"])[0, 0]))
        xs, ys = [x0], [y0]
        x, y, heading = x0, y0, heading0
        for a in actions:
            steer, spd = float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1))
            heading = (heading + steer * cfg.max_steering_deg) % 360.0
            speed = cfg.min_speed + (spd + 1.0) * 0.5 * (cfg.max_speed - cfg.min_speed)
            rad = np.radians(heading)
            x += np.cos(rad) * speed
            y += np.sin(rad) * speed
            xs.append(x)
            ys.append(y)
        paths.append((xs, ys, contp))
    return paths, values


def draw_ghosts(frame, paths, scale):
    """Blend translucent ghost paths onto a frame, faded to red where the
    model's own continue-probability head expects trouble."""
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for xs, ys, contp in paths:
        pts = [(x * scale, y * scale) for x, y in zip(xs, ys)]
        for i in range(len(pts) - 1):
            color = _lerp_color(contp[i]) + (150,)
            d.line([pts[i], pts[i + 1]], fill=color, width=3)
    img = Image.alpha_composite(img, overlay)
    return np.asarray(img.convert("RGB"))


def render_panel(height, values, paths, n_samples, length):
    """The side panel: a big predicted-value readout plus a colour strip
    per imagined sample showing its survival probability decaying (or not)
    over the horizon - a direct readout of "how long the model thinks this
    future lasts", stacked so multiple imagined futures are visible at once.
    """
    img = Image.new("RGB", (PANEL_W, height), PANEL_BG)
    d = ImageDraw.Draw(img)
    d.text((10, 10), "IMAGINATION", fill=(230, 230, 230))
    d.text((10, 32), f"{n_samples} futures x {length} steps", fill=(150, 155, 165))
    mean_value = float(np.mean(values)) if values else 0.0
    d.text((10, 60), "predicted value", fill=(150, 155, 165))
    d.text((10, 78), f"{mean_value:+.0f}", fill=(230, 230, 230))

    top = 120
    row_h = max(10, (height - top - 10) // max(1, n_samples))
    cell_w = max(2, (PANEL_W - 20) // max(1, length))
    for row, (xs, ys, contp) in enumerate(paths):
        survival = np.cumprod(contp)
        y0 = top + row * row_h
        for i, s in enumerate(survival):
            x0 = 10 + i * cell_w
            d.rectangle([x0, y0, x0 + cell_w - 1, y0 + row_h - 2], fill=_lerp_color(s))
    d.text((10, height - 22), "green = model expects to survive", fill=(110, 115, 125))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fps", type=float, default=20.0,
                     help="frames shown per second. The driver+policy+render "
                          "pipeline sustains ~70/s on CPU with this model "
                          "size, so this is a display pacing choice, not a "
                          "compute limit - raise it if it still feels slow "
                          "(and check whether you're on native VNC vs "
                          "browser noVNC, which has ~5x transport overhead)")
    ap.add_argument("--view", choices=["fullmap", "crop"], default="fullmap",
                     help="fullmap: whole map fixed in frame, what the T3D "
                          "GUI showed, and the only view click-to-target "
                          "works in (crop is car-centred and scrolls, so a "
                          "click has no stable world position). crop: "
                          "car-centred, follows the car, random targets.")
    ap.add_argument("--task", default=None,
                     help="task name if not carnav_default, e.g. 'fast'")
    ap.add_argument("--units", type=int, default=None,
                     help="agent MLP width if not the default 256")
    ap.add_argument("--auto", action="store_true",
                     help="skip the click-to-target pause - just drive "
                          "continuously, auto-resetting on every episode "
                          "end, using whatever target sampling the task "
                          "config already does (e.g. carnav_fast randomises "
                          "num_targets 1-3 on its own, no clicking needed)")
    ap.add_argument("--imagine", action="store_true",
                     help="show the world model's own imagined futures: a "
                          "fan of translucent ghost paths on the map (fullmap "
                          "view only) plus a side panel, both driven by "
                          "agent.imagine() rolling the trained RSSM forward "
                          "with no real observations - not a scripted replay")
    ap.add_argument("--imagine-samples", type=int, default=5,
                     help="number of independent imagined rollouts per "
                          "refresh - imagination is stochastic, so each one "
                          "can genuinely differ")
    ap.add_argument("--imagine-length", type=int, default=20,
                     help="imagined steps per rollout")
    ap.add_argument("--imagine-every", type=int, default=3,
                     help="real steps between imagination refreshes - "
                          "agent.imagine() isn't free, so this isn't redone "
                          "every single frame")
    args = ap.parse_args()
    click_targets = (args.view == "fullmap") and not args.auto

    extra_config = {}
    if args.task:
        extra_config["task"] = f"carnav_{args.task}"
    if args.units:
        for head in ("enc.simple", "dec.simple", "rewhead", "conhead",
                     "policy", "value"):
            extra_config[f"agent.{head}.units"] = args.units

    agent, config, dm3main = load_agent(args.checkpoint, extra_config=extra_config)
    env = dm3main.make_env(config, 0, log_topdown=True, topdown_mode=args.view)
    driver = embodied.Driver([lambda: env], parallel=False)

    # `env` here is several embodied.wrappers layers (NormalizeAction,
    # UnifyDtypes, CheckSpaces, ClipAction) around the CarNav adapter, which
    # itself wraps the raw CarNavEnv. Walk `.env` down to the CarNav
    # instance by type rather than a hardcoded hop count (fragile if the
    # wrapper chain shape ever changes) or hasattr (embodied's
    # Wrapper.__getattr__ raises ValueError, not AttributeError, for a
    # missing name, which breaks hasattr's normal AttributeError contract).
    wrapped = env
    while not isinstance(wrapped, CarNav):
        wrapped = wrapped.env
    raw_env = wrapped.env
    scale = (env.obs_space["log/topdown"].shape[1] / raw_env.map.width
             if args.view == "fullmap" else None)

    out_h, out_w = env.obs_space["log/topdown"].shape[:2]
    show_ghosts = args.imagine and args.view == "fullmap"
    panel_w = PANEL_W if args.imagine else 0

    pygame.init()
    pygame.display.set_caption("CarNav - DreamerV3 checkpoint (live)")
    screen = pygame.display.set_mode((out_w + panel_w, out_h))
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()
    ghosts = {"paths": [], "values": [], "since": 0}

    state = {"episodes": 0, "length": 0, "need_bootstrap": False}
    outcomes = {"crash": 0, "goal": 0, "timeout": 0}
    latest = {"frame": np.zeros((out_h, out_w, 3), np.uint8), "hud": ""}

    def on_step(tran, worker):
        state["length"] += 1
        latest["frame"] = tran["log/topdown"]
        if tran["is_last"]:
            outcome = ("crash" if tran["log/crashed"] else
                       ("goal" if tran["log/reached_target"] else "timeout"))
            outcomes[outcome] += 1
            state["episodes"] += 1
            latest["hud"] = f"episode {state['episodes']}: {outcome} at t={state['length']}"
            state["length"] = 0
            state["need_bootstrap"] = True
        elif tran["is_first"]:
            latest["hud"] = ("click target(s), then Enter/right-click to drive"
                              if click_targets else "driving...")
        else:
            latest["hud"] = f"episode {state['episodes'] + 1}, t={state['length']}"

    driver.on_step(on_step)
    driver.reset(agent.init_policy)
    policy = lambda *a: agent.policy(*a, mode="eval")

    def finalize_targets(targets):
        raw_env.targets = list(targets)
        raw_env.target_idx = 0
        raw_env._prev_dist = raw_env._distance_to_target()
        if raw_env.cfg.use_road_distance:
            raw_env._refresh_road_distance_field()
            raw_env._prev_road_dist = raw_env._road_distance_to_target()

    driver(policy, steps=1)  # bootstrap: perform the first reset
    mode = "select" if click_targets else "drive"
    pending = []

    if args.imagine:
        print("Warming up agent.imagine() (first call triggers a JIT "
              "compile, a few seconds)...")
        imagine_ghosts(agent, driver.carry, raw_env.cfg, 0.0, 0.0, 0.0, 1, 4)
        print("Ready.")

    print(f"Live. Window is {out_w + panel_w}x{out_h} ({args.view} view"
          + (" + imagination panel" if args.imagine else "") + "). "
          + ("Click to place targets, Enter/right-click to drive, C to "
             "clear. " if click_targets else "")
          + "Q or close the window to quit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif mode == "select":
                    if event.key == pygame.K_RETURN and pending:
                        finalize_targets(pending)
                        mode = "drive"
                    elif event.key == pygame.K_c:
                        pending = []
            elif event.type == pygame.MOUSEBUTTONDOWN and mode == "select":
                if event.button == 1:
                    mx, my = event.pos
                    wx, wy = raw_env.map.nearest_road_point(mx / scale, my / scale)
                    pending.append((wx, wy))
                elif event.button == 3 and pending:
                    finalize_targets(pending)
                    mode = "drive"
        if not running:
            break

        if mode == "drive":
            driver(policy, steps=1)
            if state["need_bootstrap"]:
                driver(policy, steps=1)  # consume the pending auto-reset
                state["need_bootstrap"] = False
                pending = []
                mode = "select" if click_targets else "drive"
                ghosts["paths"], ghosts["values"] = [], []
            elif args.imagine:
                ghosts["since"] += 1
                if ghosts["since"] >= args.imagine_every:
                    ghosts["since"] = 0
                    ghosts["paths"], ghosts["values"] = imagine_ghosts(
                        agent, driver.carry, raw_env.cfg,
                        raw_env.x, raw_env.y, raw_env.heading,
                        args.imagine_samples, args.imagine_length)

        frame = latest["frame"]
        if mode == "select" and pending:
            frame = _draw_pending(frame, pending, scale)
        elif show_ghosts and ghosts["paths"]:
            frame = draw_ghosts(frame, ghosts["paths"], scale)
        frame = np.transpose(frame, (1, 0, 2))  # pygame is (w,h)
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        totals = sum(outcomes.values())
        rate = (f"crash {outcomes['crash']/totals:.0%} / "
                f"goal {outcomes['goal']/totals:.0%}" if totals else "no episodes yet")
        hud = font.render(f"{latest['hud']}   ({rate})", True, (255, 255, 0))
        screen.blit(hud, (6, 6))

        if args.imagine:
            panel = render_panel(out_h, ghosts["values"], ghosts["paths"],
                                  args.imagine_samples, args.imagine_length)
            panel_surf = pygame.surfarray.make_surface(np.transpose(panel, (1, 0, 2)))
            screen.blit(panel_surf, (out_w, 0))

        pygame.display.flip()

        clock.tick(args.fps)

    pygame.quit()
    env.close()
    print(f"\n{state['episodes']} episodes: {outcomes}")


if __name__ == "__main__":
    main()
