"""Daily Sentinel-1 catalog query, one mapped task per AOI.

Each run covers its data interval [data_interval_start, data_interval_end) and
writes data/manifests/<aoi>/<start>_<end>.json. Because the file name is fixed
for a given window, reruns and backfills overwrite rather than duplicate.

Backfill a flood event (the Airflow 3 scheduler must be running):
    airflow backfill create --dag-id s1_catalog \
        --from-date 2022-08-15 --to-date 2022-09-30

Why the explicit timetable: in Airflow 3, schedule="@daily" defaults to
CronTriggerTimetable, where data_interval_start == data_interval_end. That
would make every query window zero-length. CronDataIntervalTimetable gives
real one-day intervals.
"""

from __future__ import annotations

import pendulum
from airflow.sdk import dag, get_current_context, task
from airflow.timetables.interval import CronDataIntervalTimetable


@dag(
    dag_id="s1_catalog",
    schedule=CronDataIntervalTimetable("0 0 * * *", timezone="UTC"),
    start_date=pendulum.datetime(2017, 1, 1, tz="UTC"),  # early enough to backfill Harvey
    catchup=False,  # scheduled runs only for new days; history comes from backfills
    max_active_runs=4,  # be polite to the CDSE API during backfills
    default_args={"retries": 3, "retry_delay": pendulum.duration(minutes=2),
                  "retry_exponential_backoff": True},
    params={"aois": []},  # empty list means all AOIs in configs/aois.yaml
    tags=["sentinel-1", "ingest"],
)
def s1_catalog():

    @task
    def select_aois() -> list[str]:
        from sarflood.aoi import load_aois

        requested = get_current_context()["params"].get("aois") or []
        available = load_aois()
        unknown = set(requested) - set(available)
        if unknown:
            raise ValueError(f"unknown AOIs: {sorted(unknown)}")
        return list(requested or available)

    @task
    def query_aoi(aoi_name: str) -> dict:
        from sarflood.aoi import load_aois
        from sarflood.catalog import DEFAULT_PRODUCT_TYPE, search_scenes
        from sarflood.manifest import write_manifest

        ctx = get_current_context()
        start, end = ctx["data_interval_start"], ctx["data_interval_end"]
        aoi = load_aois()[aoi_name]
        scenes = search_scenes(aoi.to_wkt(), start, end, DEFAULT_PRODUCT_TYPE)
        path = write_manifest(aoi_name, start, end, scenes, DEFAULT_PRODUCT_TYPE)
        return {"aoi": aoi_name, "count": len(scenes), "manifest": str(path)}

    query_aoi.expand(aoi_name=select_aois())


s1_catalog()
