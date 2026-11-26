"""Structure checks for the Airflow DAGs.

Skipped unless Airflow is importable (it lives in its own env, not the project
venv). Run from the Airflow env:

    AIRFLOW_HOME=/tmp/af-test python -m pytest tests/test_dags.py
"""
from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import pytest

pytest.importorskip("airflow")

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def bag():
    # The scheduler puts the DAGs folder on sys.path; do the same so
    # `from _sarflood_common import ...` resolves.
    sys.path.insert(0, str(DAGS))
    from airflow.dag_processing.dagbag import DagBag

    return DagBag(str(DAGS))


def _edges(dag) -> set[tuple[str, str]]:
    return {(u, t.task_id) for t in dag.tasks for u in t.upstream_task_ids}


def test_no_import_errors(bag):
    assert bag.import_errors == {}
    assert {"s1_catalog", "sarflood_process_event", "sarflood_train_and_register",
            "sarflood_rollback_champion"} <= set(bag.dag_ids)


def test_process_event_chain_and_snap_pool(bag):
    dag = bag.dags["sarflood_process_event"]
    chain = ["pairs", "download", "orbits", "terrain_correct", "change", "evaluate"]
    ids = [f"per_aoi.{s}" for s in chain]
    assert set(ids) | {"aoi_list"} == set(dag.task_ids)
    for a, b in pairwise(ids):
        assert (a, b) in _edges(dag)
    tc = dag.get_task("per_aoi.terrain_correct")
    assert tc.pool == "snap"
    assert tc.retries == 0                    # a SNAP failure is not transient
    for net in ("pairs", "download", "orbits"):
        assert dag.get_task(f"per_aoi.{net}").retries >= 1
    assert dag.schedule is None               # triggered per event, not on a clock


def test_train_and_register_chain_and_skip(bag):
    dag = bag.dags["sarflood_train_and_register"]
    chain = ["train_pixels", "mlflow_log", "fit_final", "promote"]
    assert set(dag.task_ids) == set(chain)
    assert _edges(dag) == set(pairwise(chain))
    assert 99 in dag.get_task("promote").skip_on_exit_code
    assert set(dag.params) >= {"aois", "event", "flood_pass", "smooth", "method",
                               "samples", "min_delta"}


def test_rollback_chain(bag):
    dag = bag.dags["sarflood_rollback_champion"]
    assert _edges(dag) == {("rollback", "show")}


def test_promote_maps_rejection_to_skip():
    """`registry promote` exits 1 on rejection; the task turns that into 99 (skip)."""
    sys.path.insert(0, str(DAGS))
    import _sarflood_common as c

    assert c.REJECTED_EXIT == 99
    assert c.sarflood("registry", "show").endswith("registry show")
    assert "'a b'" in c.sarflood("x", "a b")    # arguments are shell-quoted
