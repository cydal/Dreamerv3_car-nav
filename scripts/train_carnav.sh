#!/usr/bin/env bash
# Resilient launcher for dreamerv3/main.py against the carnav task.
#
# Works around an intermittent native SIGSEGV observed a few tens of seconds
# to a couple minutes into GPU training on this box (T4, driver 595.91.07,
# cuDNN 9.4.0.58 - see notes/journal.md, 2026-09-01, and the cuDNN pin
# finding in docs/dreamer-integration-plan.md). Not yet root-caused; the
# leading theory is a fallback-kernel instability, since the driver is far
# newer than the jaxlib/cuDNN build. `elements.Checkpoint` already resumes
# from the last save in --logdir automatically (Checkpoint.load_or_save), so
# retrying the same command just picks up where it left off rather than
# restarting from step 0. Pass a low --run.save_every so a crash doesn't
# cost much progress.
#
# Also sets a persistent JAX compilation cache (JAX_COMPILATION_CACHE_DIR),
# since every retry above otherwise repeats a full XLA recompile - measured
# at ~67s cold vs ~24s warm for this model, and a crash-triggered retry is
# exactly the case where a warm cache pays off most. Override the location
# with JAX_CACHE_DIR if needed; default is out of /tmp on purpose (/tmp is
# not persistent across a box restart on this machine - lost a checkpoint
# to that once, see notes/journal.md, 2026-09-02).
#
# Usage: scripts/train_carnav.sh --logdir /path/to/run --configs carnav ...
#        (all arguments are forwarded to dreamerv3/main.py verbatim)
set -uo pipefail

DREAMERV3="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../dreamerv3" && pwd)"
PY=/home/ubuntu/miniconda3/envs/dreamer/bin/python
MAX_RETRIES="${MAX_RETRIES:-20}"
JAX_CACHE_DIR="${JAX_CACHE_DIR:-$HOME/dreamer_runs/jax_cache}"
mkdir -p "$JAX_CACHE_DIR"
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"

attempt=0
while (( attempt < MAX_RETRIES )); do
  attempt=$((attempt + 1))
  echo "[train_carnav] attempt $attempt/$MAX_RETRIES: $*"
  "$PY" "$DREAMERV3/dreamerv3/main.py" "$@"
  code=$?
  if (( code == 0 )); then
    echo "[train_carnav] finished cleanly (exit 0) after $attempt attempt(s)"
    exit 0
  fi
  echo "[train_carnav] exited with code $code (attempt $attempt), resuming from checkpoint in 5s..."
  sleep 5
done

echo "[train_carnav] gave up after $MAX_RETRIES attempts"
exit 1
