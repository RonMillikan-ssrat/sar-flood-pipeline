# Databricks lakehouse layer

Acquisition, SNAP terrain correction, change maps and EMS labels stay on the Ubuntu
box (Free Edition can't reach CDSE). Their products, one change/labels GeoTIFF pair
per AOI and event, land in a Unity Catalog Volume, and Databricks builds the
medallion tables from there:

| Layer | Table | Content |
|---|---|---|
| landing | `/Volumes/<catalog>/<schema>/raw/pairs/<aoi>/<stem>/` | change.tif, labels.tif, manifest.json (written last) |
| bronze | `<schema>.bronze_pairs` | one row per manifest; grid mismatches quarantined |
| silver | `<schema>.silver_pixels` | one row per scored pixel: `features.FEATURES`, y, frac (from `load_aoi`) |
| gold | `<schema>.gold_*` | pure/balanced LOAO training samples, per-AOI feature distributions |

## Phase 0: spike

```bash
uv add --optional databricks databricks-sdk     # once
databricks auth login --host https://<your-workspace>.cloud.databricks.com
databricks catalogs list                        # confirm the catalog name ("workspace"?)

uv build --wheel
python scripts/upload_to_volume.py --dry-run
python scripts/upload_to_volume.py --aoi sindh --wheel dist/sarflood-0.1.0-py3-none-any.whl

cd databricks
databricks bundle validate
databricks bundle deploy
databricks bundle run sarflood_phase0_spike
```

The run ends with a PASS/FAIL/INFO table. Its results decide whether silver
runs `load_aoi` on serverless (expected) or on Ubuntu with Parquet uploads
(fallback), and which `tracking.py` settings become configurable.

## Phase 1: medallion tables and the training loop

Spike results (2026-11-27) that shaped it: serverless (Python 3.12, Spark 4.2, aarch64,
16 GB) installs the wheel and runs `load_aoi` straight from the Volume (Houston: 12.8M
pixels in 12 s, 3.6 GB peak); CDSE token endpoint is blocked (acquisition stays local);
MLflow needs absolute experiment paths, no `file://` artifact locations, and a model
signature for the Unity Catalog registry; UC aliases work.

| Task | Notebook | Writes |
|---|---|---|
| bronze | `01_bronze.py` | `bronze_pairs`: one row per manifest, MERGE on (aoi, stem); bad pairs `status='quarantine'` with the reason |
| silver | `02_silver.py` | `silver_pixels` (partitioned by aoi): `load_aoi` -> one row per scored pixel; an AOI with unchanged checksums is skipped. Timings -> `silver_loads` |
| gold | `03_gold.py` | `gold_feature_stats` (per AOI/feature summaries); `gold_drift`: label-free PSI of every AOI vs the pooled training AOIs |
| train | `04_train.py` | LOAO from silver -> MLflow report -> final model in UC `pixel_flood` -> gated promotion -> holdout scores (`gold_holdout_scores`, `sarflood-holdout` experiment) |

`tracking.py` switches backends with `SARFLOOD_MLFLOW=databricks` (set by `04_train.py`);
locally nothing changes. Promotion semantics match Airflow: REJECTED is a normal
outcome, REFUSED (e.g. different `labels_version`) fails the job after the holdouts are
logged.

```bash
uv build --wheel && python scripts/upload_to_volume.py --wheel dist/sarflood-0.1.0-py3-none-any.whl
cd databricks
databricks bundle validate && databricks bundle deploy
databricks bundle run sarflood_write_benchmark      # once: parquet vs arrow on Houston
databricks bundle run sarflood_lakehouse
```

After the benchmark: `SELECT * FROM workspace.sarflood.silver_loads` compares the two
write paths; `DROP TABLE workspace.sarflood.silver_bench` removes the scratch table.
The spike's `spike_pixels` table and `spike_pixel_model` can be dropped too.
