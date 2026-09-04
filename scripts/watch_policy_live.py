"""Live top-down view of a trained checkpoint driving CarNavEnv, in a real
window - the DreamerV3-side equivalent of the T3D PyQt6 GUI.

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
                          "GUI showed. crop: car-centred, follows the car.")
    ap.add_argument("--task", default=None,
                     help="task name if not carnav_default, e.g. 'fast'")
    ap.add_argument("--units", type=int, default=None,
                     help="agent MLP width if not the default 256")
    args = ap.parse_args()

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

    out_h, out_w = env.obs_space["log/topdown"].shape[:2]

    pygame.init()
    pygame.display.set_caption("CarNav - DreamerV3 checkpoint (live)")
    screen = pygame.display.set_mode((out_w, out_h))
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    state = {"episodes": 0, "length": 0}
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
        else:
            latest["hud"] = f"episode {state['episodes'] + 1}, t={state['length']}"

    driver.on_step(on_step)
    driver.reset(agent.init_policy)
    policy = lambda *a: agent.policy(*a, mode="eval")

    print(f"Live. Window is {out_w}x{out_h} ({args.view} view). Ctrl+C or "
          f"close the window to quit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
        if not running:
            break

        driver(policy, steps=1)

        frame = np.transpose(latest["frame"], (1, 0, 2))  # pygame is (w,h)
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        totals = sum(outcomes.values())
        rate = (f"crash {outcomes['crash']/totals:.0%} / "
                f"goal {outcomes['goal']/totals:.0%}" if totals else "no episodes yet")
        hud = font.render(f"{latest['hud']}   ({rate})", True, (255, 255, 0))
        screen.blit(hud, (6, 6))
        pygame.display.flip()

        clock.tick(args.fps)

    pygame.quit()
    env.close()
    print(f"\n{state['episodes']} episodes: {outcomes}")


if __name__ == "__main__":
    main()
