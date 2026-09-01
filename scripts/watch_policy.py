"""Drive CarNavEnv with a trained checkpoint and save top-down contact
sheets of what actually happened, one per episode.

This exists because the training-time video (scope/epstats-policy_image.mp4)
only shows the small 64x64 egocentric crop the agent receives - useful for
sanity-checking observations, but hard to read as "is the car navigating
well". This renders the same rollout top-down instead (car, road, goal
marker, sensors - see car_env/render.py), which is directly comparable to
what the old T3D PyQt6 GUI showed.

Runs on CPU by default - eval is cheap (no gradient steps), and this avoids
the intermittent native GPU crash documented in
docs/dreamer-integration-plan.md SS3b for a task that doesn't need the GPU.

Note: this codebase's Agent.policy() samples from the learned distribution
regardless of the mode='train'/'eval' argument - there is no separate
greedy/deterministic eval mode to opt into. What you see is the same
stochastic policy training used, not a cleaned-up "best behavior" mode.

Run (needs the `dreamer` conda env):
    python scripts/watch_policy.py --checkpoint /path/to/logdir/ckpt/<timestamp>
"""

import argparse
from pathlib import Path

from _dreamer_common import load_agent  # sets up sys.path; import first

import embodied
import numpy as np

from car_env.render import save_png, tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                     help="path to a specific ckpt/<timestamp> directory, "
                          "or a ckpt/ directory containing a 'latest' pointer")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--outdir", default="eval_frames")
    ap.add_argument("--frames-per-episode", type=int, default=6)
    ap.add_argument("--max-episodes-saved", type=int, default=8)
    ap.add_argument("--cpu", action="store_true", default=True)
    args = ap.parse_args()

    agent, config, dm3main = load_agent(args.checkpoint)

    # log_topdown isn't a configs.yaml key, so it can't go through
    # config.update() (elements.Config.update requires the key to already
    # exist) - make_env forwards **overrides straight to the env
    # constructor, which is the intended way to pass one-off kwargs like
    # this without touching configs.yaml.
    env = dm3main.make_env(config, 0, log_topdown=True)
    driver = embodied.Driver([lambda: env], parallel=False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    state = {"episodes": 0, "saved": 0, "frames": [], "length": 0}
    outcomes = {"crash": 0, "goal": 0, "timeout": 0}

    def on_step(tran, worker):
        state["frames"].append(tran["log/topdown"])
        state["length"] += 1
        if tran["is_last"]:
            crashed = bool(tran["log/crashed"])
            reached = bool(tran["log/reached_target"])
            outcome = "crash" if crashed else ("goal" if reached else "timeout")
            outcomes[outcome] += 1
            state["episodes"] += 1
            if state["saved"] < args.max_episodes_saved:
                frames = state["frames"]
                n = min(args.frames_per_episode, len(frames))
                idxs = np.linspace(0, len(frames) - 1, n).astype(int)
                sheet = tile([frames[i] for i in idxs], cols=3,
                             labels=[f"t={i}" for i in idxs])
                save_png(sheet, outdir / (
                    f"ep{state['episodes']:02d}_len{state['length']}"
                    f"_{outcome}.png"))
                state["saved"] += 1
            state["frames"] = []
            state["length"] = 0

    driver.on_step(on_step)
    driver.reset(agent.init_policy)
    policy = lambda *a: agent.policy(*a, mode="eval")

    print(f"Running {args.episodes} episodes...")
    while state["episodes"] < args.episodes:
        driver(policy, steps=10)

    print(f"\n{state['episodes']} episodes: {outcomes}")
    print(f"crash rate: {outcomes['crash'] / state['episodes']:.1%}  "
          f"goal-reach rate: {outcomes['goal'] / state['episodes']:.1%}  "
          f"timeout rate: {outcomes['timeout'] / state['episodes']:.1%}")
    print(f"wrote {state['saved']} contact sheets to {outdir}/")
    env.close()


if __name__ == "__main__":
    main()
