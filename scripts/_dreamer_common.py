"""Shared checkpoint-loading helper for scripts/watch_policy.py and
scripts/watch_policy_live.py. Not part of car_env - depends on the
dreamerv3 clone, so it stays out of the dependency-light package.
"""

import sys
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


def load_agent(checkpoint, extra_config=None):
    """Build the carnav agent and load weights from a checkpoint.

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

    checkpoint = resolve_checkpoint(checkpoint)
    print("Loading agent (network sizes from the 'carnav' config preset)...")
    agent = dm3main.make_agent(config)
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(str(checkpoint), keys=["agent"])
    print(f"Loaded checkpoint: {checkpoint}")
    return agent, config, dm3main
