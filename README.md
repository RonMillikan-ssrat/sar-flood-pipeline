# sar-flood-pipeline

Sentinel-1 SAR flood mapping, built to measure and catch **domain shift**.

A segmentation model is trained on open floodplains with plentiful labels
(Bangladesh, Pakistan), then evaluated on regions it has never seen: Houston
(an urban area, the hard case for SAR) and the Red River of the North (rural,
USGS-gauged). The pipeline should show three things: the model degrades on new
terrain, a drift check flags it, and the model registry refuses to promote it.

## Architecture (planned)

| Stage | Tool | Status |
|---|---|---|
| Catalog query: Sentinel-1 GRD scenes per AOI per day | Airflow DAG `s1_catalog` + CDSE OData | **done** |
| Download + preprocess (calibration, speckle filter, tiling) | Airflow / Cloud Batch | todo |
| Labels: Copernicus EMS delineations, Sen1Floods11 | | todo |
| Baseline: Otsu threshold on VV backscatter | | todo |
| Train U-Net | GEAP Pipelines (Kubeflow) | todo |
| Track, register, gate, roll back | MLflow | todo |
| Evaluate on holdout AOIs + drift checks | | todo |

## Areas of interest

Defined in `configs/aois.yaml`. Each one has a `role` of `train` or `holdout`,
plus flood-event windows used for backfills. The bounding boxes are approximate.

## Layout

```
configs/    AOIs and events
dags/       Airflow DAGs (thin: they call into src/)
pipelines/  Kubeflow pipeline definitions (later)
src/sarflood/
  aoi.py       config loading, bbox -> WKT
  catalog.py   CDSE OData search with pagination
  manifest.py  idempotent JSON manifests under $DATA_ROOT
  cli.py       run the same logic without Airflow
tests/      pytest, no network needed
```

## Quickstart (WSL2 Ubuntu)

Core package and tests (no Airflow):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
sarflood list-aois
sarflood query --aoi sindh --start 2022-08-28 --end 2022-09-01 --dry-run   # prints $filter
sarflood -v query --aoi sindh --start 2022-08-28 --end 2022-09-01          # live query
```

Airflow 3.4, installed with its constraints file so dependency versions match
what Airflow was tested against:

```bash
uv pip install "apache-airflow==3.4.0" \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt

export AIRFLOW_HOME=$PWD/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export DATA_ROOT=$PWD/data

airflow db migrate
airflow dags list-import-errors
airflow dags test s1_catalog 2022-08-29     # one run, in-process; queries 2022-08-28 -> 08-29
```

Backfill a flood event (run `airflow standalone` in another terminal first;
Airflow 3 backfills are run by the scheduler):

```bash
airflow backfill create --dag-id s1_catalog --from-date 2022-08-15 --to-date 2022-09-30
```

## Design notes

- **Explicit interval timetable.** In Airflow 3, `schedule="@daily"` defaults to a
  trigger timetable where `data_interval_start == data_interval_end`, which would
  make every query window zero-length. The DAG uses `CronDataIntervalTimetable`.
- **Idempotent writes.** The manifest file name comes from the AOI and window,
  so reruns overwrite. Writes are atomic (temp file + rename).
- **Thin DAGs.** All logic lives in `src/` and is testable without Airflow;
  the CLI runs the same code paths.
- **Catalog search needs no token.** Downloads will (CDSE account, later step).
- **Coverage gap.** Only one satellite was operating from 2022 to 2024, after
  Sentinel-1B failed, so revisit was lower. S1C (Dec 2024) and S1D (data from
  Apr 2026) restore the two-satellite cycle.
