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
| Group slices into passes; pair each flood pass with a same-track dry reference | `passes.py`, `sarflood pairs` | **done** |
| Download raw `.SAFE` for one flood/reference pair | `download.py`, `sarflood download` | **done** |
| Thermal-noise removal + σ⁰ calibration, radar geometry | `calibrate.py`, `sarflood calibrate` | **done** |
| Same slice through SNAP `gpt`; compare σ⁰ pixel for pixel | `snap/graphs/tnr_cal.xml`, `compare.py`, `sarflood compare` | **done**: VV/VH 100% within 0.1 dB |
| Precise orbit files from CDSE into SNAP's auxdata folder (step.esa.int fallback) | `orbits.py`, `sarflood orbits` | **done** |
| Terrain correction: per-slice TNR (no clip) → slice assembly → precise orbit → σ⁰ → Copernicus 30 m DEM, UTM, fixed grid → AOI clip | `snap.py`, `sarflood terrain-correct` | **done** (Sindh 78D flood pass) |
| Server-side processed backscatter for scale | Sentinel Hub / openEO | todo |
| Change detection on the shared grid: VV/VH dB, change, masks, quick-look, threshold preview | `change.py`, `sarflood change` | **built** |
| EMS activation check: AOI extents vs ours, product imagery (circularity check) | `ems.py`, `sarflood ems-info` | **built** |
| Labels: EMS delineation (flood fraction via sub-pixel rasterization, AOI, not-analysed, permanent water) on the change grid; IoU/F1/precision/recall; outcome map | `labels.py`, `sarflood evaluate` | **built** |
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
  passes.py    slices -> passes (datatake), dedupe, same-track reference pairing
  auth.py      CDSE access tokens (password grant + refresh)
  download.py  .SAFE zip download: manual redirects, size check, skip-if-complete
  calibrate.py DN -> sigma0 from the calibration/noise LUT XMLs (radar geometry)
  compare.py   our sigma0 vs SNAP's, pixel for pixel, in dB
  snap.py      generates SNAP gpt graphs (any slice count) and runs gpt
  orbits.py    finds/downloads AUX_POEORB orbit files for a pass from CDSE
  change.py    flood vs reference on the shared grid: dB, change, masks, quick-look
  ems.py       Copernicus EMS Rapid Mapping API: AOIs, products, imagery
  labels.py    EMS vector package -> label rasters on the change grid; scoring
snap/graphs/  SNAP gpt graphs used as references
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
sarflood pairs --aoi sindh --event pakistan-2022                            # flood/reference pairs
sarflood download --aoi sindh --event pakistan-2022 \
    --flood-pass S1A_044775_0558C6 --dry-run                                # list files + sizes
```

Downloading needs a free Copernicus Data Space account. Put the credentials in
`~/.netrc` (`chmod 600`), never in the repo:

```
machine identity.dataspace.copernicus.eu login you@example.com password ...
```

```bash
sarflood -v download --aoi sindh --event pakistan-2022 --flood-pass S1A_044775_0558C6
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
- **Negative power after noise removal.** Over calm water the thermal noise can
  exceed DN², especially in VH (13% of pixels in a Sindh 2022 test window,
  vs 0.35% for VV). With clipping on, intensity ≤ 0 is set to 1e-5 (DN² units)
  before calibration. That matches SNAP 14 `ThermalNoiseRemoval`
  (`clipNegativeValues=true`), verified as SNAP σ⁰ × A² = 1e-5 at every such
  pixel, and keeps dB values finite (about −105 dB). Clipping before averaging or
  resampling biases dark targets upward (MPC-0392), so the training pipeline
  uses `--no-clip` until after terrain correction.
- **Validated against SNAP 14.** VV σ⁰ on a 2048 × 2048 window: max
  |difference| 0.0029 dB, 100% of pixels within 0.01 dB, no mirroring.
- **Shared pixel grid.** Terrain correction uses the AOI's UTM zone, a fixed
  pixel spacing and `alignToStandardGrid` with origin (0, 0), so every pass over
  an AOI lands on the same grid. Flood and reference images can then be
  differenced pixel for pixel without resampling again.
- **Labels are independent of the input.** EMSR629 AOI01 (Larkana) was delineated
  from SPOT-6/7 optical imagery on 2022-08-30 05:44 UTC, 4 h 19 min after the
  Sentinel-1 pass we classify. Scored region: inside the EMS AOI, assessed, valid
  in the change map, excluding permanent water (Lake/River; "Land Subject to
  Inundation" is floodable land and stays in).
- **Pairing rule.** A reference must match the flood pass's relative orbit, orbit
  direction and polarisation, fall in the event's dry-season `reference` window,
  and cover at least 50% of the AOI. The latest qualifying pass wins (least
  unrelated change). S1A/S1B/S1C/S1D on the same track are interchangeable
  unless `--same-platform` is given.
- **Catalog search needs no token; downloads do.** The download URL redirects, and
  `requests` drops the `Authorization` header on a cross-host redirect, so the
  downloader follows redirects itself and re-sends the token only to
  `*.copernicus.eu`. Files land as `.zip.part` and are renamed only after the
  byte count matches the catalog's `ContentLength`.
- **Coverage gap.** Only one satellite was operating from 2022 to 2024, after
  Sentinel-1B failed, so revisit was lower. S1C (Dec 2024) and S1D (data from
  Apr 2026) restore the two-satellite cycle.
