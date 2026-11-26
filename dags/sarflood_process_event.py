"""Process one flood event for several AOIs: Sentinel-1 data -> change map -> scores.

Per AOI (dynamically mapped task group):
    pairs -> download -> orbits -> terrain_correct -> change -> evaluate

terrain_correct runs SNAP gpt and is limited to the 'snap' pool (1 slot): one SNAP
job at a time, because a slice needs most of the machine's memory. Network steps
retry; SNAP does not (a failure there needs a look, not a blind rerun).

Trigger with params, e.g.:
    airflow dags trigger sarflood_process_event --conf '{"aois": ["larkana"],
        "event": "pakistan-2022", "flood_pass": "S1A_044775_0558C6",
        "ems_zips": {"larkana": "EMSR629_AOI01_DEL_PRODUCT_r1_RTP01_v2_vector.zip"}}'
"""

from __future__ import annotations

import pendulum
from _sarflood_common import BASH, DATA_ROOT, SNAP_POOL, START, sarflood
from airflow.sdk import Param, dag, get_current_context, task, task_group

DEFAULT_ZIPS = {
    "larkana": "EMSR629_AOI01_DEL_PRODUCT_r1_RTP01_v2_vector.zip",
    "shikarpur": "EMSR629_AOI02_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
    "jacobabad": "EMSR629_AOI03_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
}
NET_RETRIES = {"retries": 3, "retry_delay": pendulum.duration(minutes=2),
               "retry_exponential_backoff": True}


def _p() -> dict:
    return get_current_context()["params"]


@dag(
    dag_id="sarflood_process_event",
    schedule=None,
    start_date=START,
    catchup=False,
    max_active_runs=1,
    params={
        "aois": Param(["larkana", "shikarpur", "jacobabad"], type="array",
                      description="AOI names from configs/aois.yaml"),
        "event": Param("pakistan-2022", type="string"),
        "flood_pass": Param("S1A_044775_0558C6", type="string",
                            description="flood pass key from `sarflood pairs`"),
        "smooth": Param(3, type="integer", minimum=1),
        "ems_zips": Param(DEFAULT_ZIPS, type="object",
                          description="AOI -> EMS DEL zip in $DATA_ROOT/ems/zips (for evaluate)"),
    },
    tags=["sarflood", "processing"],
)
def sarflood_process_event():

    @task
    def aoi_list() -> list[str]:
        return list(_p()["aois"])

    @task_group(group_id="per_aoi")
    def per_aoi(aoi: str):

        @task.bash(**BASH, **NET_RETRIES)
        def pairs(aoi: str) -> str:
            return sarflood("pairs", "--aoi", aoi, "--event", _p()["event"])

        @task.bash(**BASH, **NET_RETRIES)
        def download(aoi: str) -> str:
            p = _p()
            return sarflood("download", "--aoi", aoi, "--event", p["event"],
                            "--flood-pass", p["flood_pass"])

        @task.bash(**BASH, **NET_RETRIES)
        def orbits(aoi: str) -> str:
            p = _p()
            return sarflood("orbits", "--aoi", aoi, "--event", p["event"],
                            "--flood-pass", p["flood_pass"])

        @task.bash(**BASH, pool=SNAP_POOL, retries=0,
                   execution_timeout=pendulum.duration(hours=2))
        def terrain_correct(aoi: str) -> str:
            p = _p()
            return sarflood("terrain-correct", "--aoi", aoi, "--event", p["event"],
                            "--flood-pass", p["flood_pass"])

        @task.bash(**BASH)
        def change(aoi: str) -> str:
            p = _p()
            return sarflood("change", "--aoi", aoi, "--event", p["event"],
                            "--flood-pass", p["flood_pass"], "--smooth", p["smooth"])

        @task.bash(**BASH)
        def evaluate(aoi: str) -> str:
            p = _p()
            zip_name = p["ems_zips"].get(aoi)
            if not zip_name:
                return f"echo 'no EMS zip configured for {aoi}; skipping evaluate'"
            return sarflood("evaluate", "--aoi", aoi, "--event", p["event"],
                            "--flood-pass", p["flood_pass"], "--smooth", p["smooth"],
                            "--ems-zip", f"{DATA_ROOT}/ems/zips/{zip_name}",
                            "--rule", "A", "--rule", "B", "--rule", "C", "--rule", "D")

        (pairs(aoi) >> download(aoi) >> orbits(aoi) >> terrain_correct(aoi)
         >> change(aoi) >> evaluate(aoi))

    per_aoi.expand(aoi=aoi_list())


sarflood_process_event()
