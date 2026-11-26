"""Shared settings for the sarflood DAGs.

Airflow only orchestrates: every task runs the `sarflood` CLI from the *project's*
virtualenv, so Airflow's environment never has to hold rasterio / scikit-learn /
MLflow / PyTorch (whose pins can conflict with Airflow's own).

Environment variables (set where the scheduler / workers run):
    SARFLOOD_REPO       repo root            (default: parent of this dags/ folder)
    SARFLOOD_BIN        sarflood executable  (default: $SARFLOOD_REPO/.venv/bin/sarflood)
    SARFLOOD_DATA_ROOT  data folder          (default: $SARFLOOD_REPO/data)
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pendulum

REPO = os.environ.get("SARFLOOD_REPO", str(Path(__file__).resolve().parents[1]))
SARFLOOD_BIN = os.environ.get("SARFLOOD_BIN", f"{REPO}/.venv/bin/sarflood")
DATA_ROOT = os.environ.get("SARFLOOD_DATA_ROOT", f"{REPO}/data")
START = pendulum.datetime(2026, 1, 1, tz="UTC")

# Keyword arguments for every @task.bash: run from the repo with our data root.
BASH = {
    "cwd": REPO,
    "env": {"DATA_ROOT": DATA_ROOT, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    "append_env": True,   # keep PATH, HOME, ... from the worker
}

SNAP_POOL = "snap"        # create with: airflow pools set snap 1 "one SNAP gpt job at a time"
REJECTED_EXIT = 99        # BashOperator's default skip_on_exit_code: task shows as 'skipped'


def sarflood(*args: object) -> str:
    """Shell-quoted `sarflood ...` command line."""
    return " ".join([shlex.quote(SARFLOOD_BIN), *(shlex.quote(str(a)) for a in args)])
