"""Manual one-click undo: make the 'previous' model version champion again.

    rollback -> show

Never scheduled. Trigger from the UI or:  airflow dags trigger sarflood_rollback_champion
"""

from __future__ import annotations

from _sarflood_common import BASH, START, sarflood
from airflow.sdk import dag, task


@dag(dag_id="sarflood_rollback_champion", schedule=None, start_date=START, catchup=False,
     max_active_runs=1, tags=["sarflood", "ml", "ops"])
def sarflood_rollback_champion():

    @task.bash(**BASH)
    def rollback() -> str:
        return sarflood("registry", "rollback")

    @task.bash(**BASH)
    def show() -> str:
        return sarflood("registry", "show")

    rollback() >> show()


sarflood_rollback_champion()
