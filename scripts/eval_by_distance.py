"""Stratify goal-reach rate by how far the target was at episode start.

analyze_run.py's training-health section tells you whether the actor-critic
is still learning at all. This tells you *what it's still bad at* - if
goal-reach stays low specifically for far/indirect targets while close ones
succeed, the remaining gap is a navigation/planning problem (turns, lookahead)
rather than a general policy-quality one. Used 2026-09-03 to confirm the
reward-shaping fix helped but didn't close that gap (30.8% close vs
12.5-16.4% far) - see notes/journal.md.

Run (needs the `dreamer` conda env):
    python scripts/eval_by_distance.py --checkpoint /path/to/logdir/ckpt
"""

import argparse

from _dreamer_common import load_agent  # sets up sys.path; import first

import embodied
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--buckets", type=float, nargs="+",
                     default=[0, 150, 200, 250, 300, 1000],
                     help="init-distance bucket edges, in world px")
    args = ap.parse_args()

    agent, config, dm3main = load_agent(args.checkpoint)
    env = dm3main.make_env(config, 0, log_topdown=False)
    driver = embodied.Driver([lambda: env], parallel=False)

    results = []
    state = {"init_dist": None}

    def on_step(tran, worker):
        if tran["is_first"]:
            state["init_dist"] = float(tran["log/distance_to_target"])
        if tran["is_last"]:
            outcome = ("crash" if tran["log/crashed"] else
                       ("goal" if tran["log/reached_target"] else "timeout"))
            results.append((state["init_dist"], outcome))

    driver.on_step(on_step)
    driver.reset(agent.init_policy)
    policy = lambda *a: agent.policy(*a, mode="eval")

    print(f"Running {args.episodes} episodes...")
    while len(results) < args.episodes:
        driver(policy, steps=10)

    edges = args.buckets
    print(f"\n{len(results)} episodes, stratified by initial target distance:")
    print(f"  {'init_dist':>14}  {'n':>4}  {'goal_rate':>10}  {'crash_rate':>10}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [o for d, o in results if lo <= d < hi]
        if not sub:
            continue
        goal = sum(1 for o in sub if o == "goal")
        crash = sum(1 for o in sub if o == "crash")
        print(f"  [{lo:>5.0f},{hi:>5.0f})  {len(sub):>4}  "
              f"{goal/len(sub):>9.1%}  {crash/len(sub):>9.1%}")

    overall_goal = sum(1 for _, o in results if o == "goal") / len(results)
    print(f"\noverall goal-reach rate: {overall_goal:.1%}")
    env.close()


if __name__ == "__main__":
    main()
