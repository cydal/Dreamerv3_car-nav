"""Shared checkpoint-loading helper for scripts/watch_policy.py and
scripts/watch_policy_live.py. Not part of car_env - depends on the
dreamerv3 clone, so it stays out of the dependency-light package.
"""

import sys
from functools import partial as bind
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "dreamerv3"))

import elements
import ruamel.yaml as yaml


def resolve_checkpoint(path):
    """Accept either a specific ckpt/<timestamp> dir or its ckpt/ parent
    (which holds a 'latest' pointer file)."""
    path = Path(path)
    if (path / "latest").exists():
        path = path / (path / "latest").read_text().strip()
    return path


def load_agent(checkpoint, extra_config=None, regex=".*", skip_weights=False):
    """Build the carnav agent and load weights from a checkpoint.

    skip_weights=True builds the agent with its freshly-initialized
    (random) parameters and never touches `checkpoint` at all - for
    showing what an untrained/never-trained agent looks like, e.g. as a
    before/after contrast. Task/config still come from `extra_config`,
    same as normal.

    regex is forwarded to Agent.load(regex=...) - a partial load, matching
    checkpoint param keys against this pattern and leaving anything not
    present in the checkpoint (e.g. a newly added Normalize submodule's
    stats, added by an agent config change like advnorm.impl) at its
    fresh-initialized value instead of raising a shape-mismatch error. The
    default '.*' just means "load everything that exists in the
    checkpoint" - safe even when nothing changed. Uses the same
    elements.checkpoint.load(path, dict(agent=bind(agent.load, regex=...)))
    free-function path dreamerv3/main.py's train() uses for
    --run.from_checkpoint, rather than elements.Checkpoint's class
    interface, which doesn't expose regex - found the hard way when this
    module's earlier plain `cp.load(..., keys=['agent'])` call raised a
    hard shape-mismatch trying to load a checkpoint saved before an
    advnorm config change. See notes/journal.md, 2026-09-03.

    Returns (agent, config, dm3main) - dm3main is the imported
    dreamerv3.main module, handed back so callers can reuse make_env
    without re-importing it under a different sys.path state.
    """
    from dreamerv3 import main as dm3main

    folder = Path(dm3main.__file__).resolve().parent
    raw = elements.Path(folder / "configs.yaml").read()
    configs = yaml.YAML(typ="safe").load(raw)
    config = elements.Config(configs["defaults"])
    config = config.update(configs["carnav"])
    config = config.update(logdir="/tmp/watch_policy_scratch")
    config = config.update({"jax.platform": "cpu", "jax.prealloc": False})
    if extra_config:
        config = config.update(extra_config)

    print("Loading agent (network sizes from the 'carnav' config preset)...")
    agent = dm3main.make_agent(config)
    if skip_weights:
        print("Skipping checkpoint weights - agent is freshly initialized "
              "(random), never trained.")
    else:
        checkpoint = resolve_checkpoint(checkpoint)
        elements.checkpoint.load(str(checkpoint), dict(
            agent=bind(agent.load, regex=regex)))
        print(f"Loaded checkpoint: {checkpoint}")
    return agent, config, dm3main
