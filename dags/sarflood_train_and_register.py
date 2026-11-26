"""Train, track, register, and (gated) promote the pixel flood model.

    train_pixels -> mlflow_log -> fit_final -> promote

promote applies the registry gate: the new version becomes 'champion' only if its
mean held-out IoU with a 300-label cut beats the current champion's (+ min_delta).
A rejection is not an error: the task ends as SKIPPED ('not promoted'), so the run
stays green and the UI shows what happened.
"""

from __future__ import annotations

from _sarflood_common import BASH, DATA_ROOT, REJECTED_EXIT, START, sarflood
from airflow.sdk import Param, dag, get_current_context, task


def _p() -> dict:
    return get_current_context()["params"]


def _report(p: dict) -> str:
    """The train-pixels report, under the same DATA_ROOT the tasks run with."""
    return f"{DATA_ROOT}/ml/pixels/{p['flood_pass']}_s{p['smooth']}_loao.json"


def _common(p: dict) -> list[str]:
    args = []
    for a in p["aois"]:
        args += ["--aoi", a]
    return args + ["--event", p["event"], "--flood-pass", p["flood_pass"]]


@dag(
    dag_id="sarflood_train_and_register",
    schedule=None,
    start_date=START,
    catchup=False,
    max_active_runs=1,
    params={
        "aois": Param(["larkana", "shikarpur", "jacobabad"], type="array"),
        "event": Param("pakistan-2022", type="string"),
        "flood_pass": Param("S1A_044775_0558C6", type="string"),
        "smooth": Param(3, type="integer", minimum=1),
        "method": Param("hgb_bal", enum=["hgb_bal", "hgb", "logreg_bal", "logreg"]),
        "samples": Param(200_000, type="integer", minimum=100),
        "min_delta": Param(0.0, type="number", minimum=0),
    },
    tags=["sarflood", "ml"],
)
def sarflood_train_and_register():

    @task.bash(**BASH)
    def train_pixels() -> str:
        p = _p()
        return sarflood("train-pixels", *_common(p), "--smooth", p["smooth"],
                        "--samples", p["samples"], "--no-importance")

    @task.bash(**BASH)
    def mlflow_log() -> str:
        p = _p()
        return sarflood("mlflow-log", _report(p), "--tag", "source=airflow")

    @task.bash(**BASH)
    def fit_final() -> str:
        p = _p()
        return sarflood("fit-final", "--method", p["method"],
                        "--report", _report(p),
                        *_common(p), "--smooth", p["smooth"], "--samples", p["samples"])

    @task.bash(**BASH, skip_on_exit_code=REJECTED_EXIT)
    def promote() -> str:
        # registry promote exits 1 on a gate rejection -> map to 99 -> task SKIPPED.
        cmd = sarflood("registry", "promote", "--latest", "--min-delta", _p()["min_delta"])
        return f"{cmd}; rc=$?; if [ $rc -eq 1 ]; then exit {REJECTED_EXIT}; fi; exit $rc"

    train_pixels() >> mlflow_log() >> fit_final() >> promote()


sarflood_train_and_register()
