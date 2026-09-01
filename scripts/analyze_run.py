"""Summarize a dreamerv3 training run against the known random-policy
baselines from scripts/smoke_test.py (median episode length 8), so "is it
learning" has numbers to compare against rather than a vibe.

Reads metrics.jsonl, which interleaves two row shapes: per-episode rows
(episode/length, episode/score, step) and periodic aggregate rows (fps,
train losses, epstats/* - the reward_parts breakdown from
notes/concepts/episodes-and-rewards.md).

Run:  python scripts/analyze_run.py /path/to/logdir/run1
"""

import json
import sys
from pathlib import Path

import numpy as np

RANDOM_MEDIAN_LENGTH = 8.0  # scripts/smoke_test.py, default config


def load_episodes(logdir):
    path = Path(logdir) / "metrics.jsonl"
    steps, lengths, scores = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "episode/length" in row:
            steps.append(row["step"])
            lengths.append(row["episode/length"])
            scores.append(row["episode/score"])
    return np.array(steps), np.array(lengths), np.array(scores)


def load_epstats(logdir):
    path = Path(logdir) / "metrics.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if any(k.startswith("epstats/") for k in row):
            rows.append(row)
    return rows


def bucket(steps, values, n_buckets=10):
    edges = np.linspace(steps.min(), steps.max() + 1, n_buckets + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (steps >= lo) & (steps < hi)
        if mask.sum() == 0:
            continue
        out.append((int(lo), int(hi), int(mask.sum()),
                    float(values[mask].mean())))
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    logdir = Path(sys.argv[1])

    steps, lengths, scores = load_episodes(logdir)
    n = len(steps)
    print(f"{n} episodes over env steps {steps.min():,}..{steps.max():,}\n")

    print("episode length over training (bucketed by env step):")
    print(f"  {'env steps':>20}  {'n_eps':>6}  {'mean len':>9}")
    for lo, hi, cnt, mean in bucket(steps, lengths):
        marker = " *" if mean > RANDOM_MEDIAN_LENGTH else ""
        print(f"    {lo:>8,} - {hi:>8,}  {cnt:>6}  {mean:>9.1f}{marker}")
    print(f"  (* above the random-policy median of {RANDOM_MEDIAN_LENGTH:.0f})")

    print("\nepisode score over training:")
    print(f"  {'env steps':>20}  {'n_eps':>6}  {'mean score':>11}")
    for lo, hi, cnt, mean in bucket(steps, scores):
        print(f"    {lo:>8,} - {hi:>8,}  {cnt:>6}  {mean:>11.1f}")

    k = max(1, n // 10)
    print(f"\nfirst {k} episodes: mean length {lengths[:k].mean():6.1f}  "
          f"mean score {scores[:k].mean():8.1f}")
    print(f"last  {k} episodes: mean length {lengths[-k:].mean():6.1f}  "
          f"mean score {scores[-k:].mean():8.1f}")

    epstats = load_epstats(logdir)
    if epstats:
        # log/crashed is 0/1, true only on an episode's final step, so its
        # per-episode 'sum' (or equivalently 'max') is exactly "did this
        # episode crash" - and epstats' own aggregation across episodes in
        # the window is an average, so epstats/log/crashed/sum is actually
        # the *crash rate*, not a sum. Reading '/avg' instead (the per-step
        # average, diluted by episode length) looks similar in magnitude but
        # means something different and is the wrong number for "did it
        # learn to stop crashing" - it drops as episodes get longer even if
        # every single one of them still ends in a crash. Learned this the
        # hard way; kept the comment so it doesn't happen twice.
        n = len(epstats)
        early, late = epstats[: max(1, n // 3)], epstats[-max(1, n // 3):]

        def avg_key(rows, key):
            vals = [r[key] for r in rows if key in r]
            return sum(vals) / len(vals) if vals else float("nan")

        crash_early = avg_key(early, "epstats/log/crashed/sum")
        crash_late = avg_key(late, "epstats/log/crashed/sum")
        print(f"\ncrash rate (fraction of episodes ending in collision):")
        print(f"  first third of run: {crash_early:.1%}")
        print(f"  last  third of run: {crash_late:.1%}")

        reach_key = "epstats/log/reached_target/sum"
        if any(reach_key in r for r in epstats):
            reach_early = avg_key(early, reach_key)
            reach_late = avg_key(late, reach_key)
            print(f"\ngoal-reach rate (fraction of episodes ending at the "
                  f"final target):")
            print(f"  first third of run: {reach_early:.1%}")
            print(f"  last  third of run: {reach_late:.1%}")
        else:
            print(f"  (remainder ends via goal-reach or the {1000}-step "
                  f"timeout - no log/reached_target in this run, added "
                  f"2026-09-01, so this has to stay an inference)")

        print("\nreward_parts breakdown (per-step average, first vs last "
              "third of run):")
        keys = ["log/reward_step", "log/reward_progress",
                "log/reward_alignment", "log/reward_clear_sensors",
                "log/distance_to_target"]
        for key in keys:
            k_avg = f"epstats/{key}/avg"
            e, l = avg_key(early, k_avg), avg_key(late, k_avg)
            print(f"    {key:28s} {e:9.2f} -> {l:9.2f}")


if __name__ == "__main__":
    main()
