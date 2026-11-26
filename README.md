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
| Change detection on the shared grid: optional N×N speckle averaging (linear, unclipped), per-polarization validity, VV/VH dB and change, quick-look, threshold preview | `change.py`, `sarflood change` | **done** |
| EMS activation check: AOI extents vs ours, product imagery (circularity check) | `ems.py`, `sarflood ems-info` | **built** |
| Labels: EMS delineation (flood fraction via sub-pixel rasterization, AOI, not-analysed, permanent water) on the change grid; IoU/F1/precision/recall; outcome map; threshold sweep incl. no-change rule; class-conditional distributions + best single cut per variable | `labels.py`, `sarflood evaluate` | **done** |
| Fixed rules for held-out testing (presets A/B/C), incl. label-free Otsu baseline | `rules.py`, `evaluate --rule` | **built** |
| AOI entries straight from EMS vector packages, 2022 and 2017 layouts | `ems_zip.py`, `ems_legacy.py`, `sarflood ems-aoi` | **built** |
| Held-out test: EMSR629 Shikarpur (AOI02), Jacobabad (AOI03) | `evaluate --rule` | **done** (table below) |
| Pixel classifiers (logistic regression, gradient boosting), leave-one-AOI-out vs rule A | `features.py`, `pixels.py`, `sarflood train-pixels` | **built** |
| U-Net (plain PyTorch, ~7.8M params), leave-one-AOI-out vs rule A; fp16 on CUDA | `unet.py`, `sarflood train-unet` | **built** |
| GEAP Pipelines (Kubeflow) training | | todo |
| Track (ingest LOAO reports), register final pixel model, gated promotion, rollback | `tracking.py`, `sarflood mlflow-log / fit-final / registry` (MLflow 3) | **built** |
| Orchestration: per-event processing (mapped per AOI, SNAP pool of 1), train → log → register → gated promote, rollback | Airflow DAGs `sarflood_process_event`, `sarflood_train_and_register`, `sarflood_rollback_champion` | **built** |
| Evaluate on holdout AOIs + drift checks | | todo |

## Results so far

Flood pass S1A 2022-08-30 01:25 UTC (track 78 descending), reference 2022-05-26,
3x3 averaging, 20 m. Labels: Copernicus EMS EMSR629 delineations traced from
SPOT-6/7 at 05:44-05:45 UTC the same day (independent sensor, ~4 h later).
Rules were fixed on Larkana before scoring Shikarpur and Jacobabad. IoU:

| Rule | Larkana (in-sample) | Shikarpur (held out) | Jacobabad (held out) |
|---|---|---|---|
| A: VV <= -12 dB | 0.778 | 0.730 | 0.691 |
| B: dVV <= -1 dB and VV <= -12 dB | 0.738 | 0.662 | 0.693 |
| C: Otsu on VV (no labels) | 0.702 | 0.547 | 0.598 |

A single threshold errs in opposite directions on the two held-out areas
(Shikarpur P 0.95 / R 0.76: too strict; Jacobabad P 0.76 / R 0.88: too loose),
which is what motivates learned, context-aware models.

Pixel classifiers (11 features incl. 7x7 context), leave-one-AOI-out: IoU
0.787 / 0.695 / 0.669 vs rule A 0.778 / 0.730 / 0.691. Threshold-free, they rank
only slightly better than VV (AUC +0.005 to +0.013); the ideal VV cut varies
~3 dB between AOIs (-11.3 / -9.7 / -12.6 dB). Label-free adaptive cuts (Otsu,
2-component GMM) land 2.5-5 dB stricter than the ideal: they find open water,
while EMS "flooded area" also includes brighter flooded ground.

U-Net (7.8M params, 20 epochs): 0.757 / 0.642 / 0.686, no better than rule A or
the pixel models out of area (AUC equal or lower). Choosing the cut from ~300
random labeled pixels of the target area recovers nearly the ceiling everywhere;
with that, hgb_bal reaches 0.783 / 0.815 / 0.699 (mean 0.766 vs 0.733 for rule A
at -12 dB).

## Model registry (MLflow)

```bash
uv pip install -e ".[tracking]"
sarflood mlflow-log data/ml/pixels/*_loao.json data/ml/unet/*_loao.json
sarflood fit-final --method hgb_bal --report data/ml/pixels/<...>_loao.json \
    --aoi larkana --aoi shikarpur --aoi jacobabad --event pakistan-2022 --flood-pass <key>
sarflood registry show
sarflood registry promote --version 1        # gate: beat the champion's mean.iou_n300
sarflood registry rollback                   # champion <-> previous
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db
```

The gate metric is the mean held-out IoU with the cut chosen from 300 local
labels (`mean.iou_n300`), the operational setting the results support.
MLflow 3 stores scikit-learn models in the skops format; only scikit-learn's
own types are declared trusted, and the list is recorded on the run.

## Holdout scoring and label versions

`sarflood score-holdout --aoi <aoi> --event <e> --flood-pass <pass>` applies a registered
model (default: the `champion` alias) to an AOI it was not trained on and logs the result
to the MLflow experiment `sarflood-holdout`. Every score is reported next to the
**all-flood baseline** (IoU of a map that calls everything flood = the flood share), since
a high flood share inflates IoU: on Sanghar (72% flooded) the all-flood map scores 0.721
and rule A 0.378. AUC and AP are reported because they depend on neither a cut nor the
flood share.

Each labels GeoTIFF carries a `labels_version` tag (`labels.LABELS_VERSION`; untagged = 1).
Train-pixels reports and registered versions record it, `fit-final` refuses a report built
on other labels than those on disk, and `registry promote` refuses to compare gate values
across label versions; a person decides with `--force`.

## Orchestration (Airflow)

| DAG | Tasks | Notes |
|---|---|---|
| `sarflood_process_event` | `aoi_list` → per AOI (mapped): `pairs` → `download` → `orbits` → `terrain_correct` → `change` → `evaluate` | Network steps retry 3×. `terrain_correct` runs in pool `snap` (1 slot: one SNAP job at a time, it takes ~9 GB heap) with no retries and a 2 h timeout. Needs SNAP and CDSE credentials. |
| `sarflood_train_and_register` | `train_pixels` → `mlflow_log` → `fit_final` → `promote` | A gate rejection shows as **skipped** `promote`, not failed: `registry promote` exits 1 on rejection, the task maps that to 99, and Airflow's bash tasks treat exit 99 as skip. |
| `sarflood_rollback_champion` | `rollback` → `show` | Swaps `champion` and `previous` aliases. |

All three are manual-trigger (`schedule=None`) and take run parameters
(`-c '{...}'` or the trigger form in the UI). The tasks shell out to the
project venv's `sarflood`, so Airflow and its pinned dependencies live in a
separate venv and the DAGs import nothing from `src/`.

| Variable | Default | Purpose |
|---|---|---|
| `SARFLOOD_REPO` | parent of `dags/` | working directory for tasks |
| `SARFLOOD_BIN` | `$SARFLOOD_REPO/.venv/bin/sarflood` | CLI the tasks run |
| `SARFLOOD_DATA_ROOT` | `$SARFLOOD_REPO/data` | passed to tasks as `DATA_ROOT`; report and EMS zip paths are built from it |

Where it runs: `process_event` needs SNAP and the raw `.SAFE` data (laptop);
`train_and_register` and `rollback_champion` need only the change/label rasters
and the MLflow store (desktop).

```bash
airflow pools set snap 1 "one SNAP gpt job at a time"
airflow dags test sarflood_process_event -c '{"aois": ["larkana"]}'
airflow dags test sarflood_train_and_register                         # first run: promotes v1
airflow dags test sarflood_train_and_register -c '{"min_delta": 0.01}' # same model: promote skipped
airflow dags test sarflood_rollback_champion
```

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
  rules.py     classification rules (A/B/C presets, Otsu), fixed before held-out tests
  ems_zip.py   AOI outline + source records from an EMS package; flags suspect flood sources
  ems_legacy.py  older (2017) shapefile-only EMS packages -> same label roles
  features.py  per-pixel features (backscatter, change, ratios, 7x7 context)
  pixels.py    leave-one-AOI-out training/scoring of pixel classifiers
  unet.py      U-Net: tiles, masked class-balanced loss, overlapping-tile prediction
  tracking.py  MLflow: report ingestion, final-model registration, champion gate, rollback
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

Airflow 3.4 in its own venv (outside the repo), installed with its constraints
file so dependency versions match what Airflow was tested against:

```bash
uv venv --python 3.12 ~/venvs/airflow && source ~/venvs/airflow/bin/activate
uv pip install "apache-airflow==3.4.0" \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.4.0/constraints-3.12.txt
uv pip install pytest
uv pip install -e . --no-deps     # s1_catalog imports sarflood (needs only requests + pyyaml, which Airflow has)
rm ~/venvs/airflow/bin/sarflood   # that CLI lacks its dependencies; tasks use $SARFLOOD_BIN

export AIRFLOW_HOME=$PWD/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export DATA_ROOT=$PWD/data SARFLOOD_DATA_ROOT=$PWD/data

airflow db migrate
airflow dags list-import-errors
python -m pytest -q tests/test_dags.py      # DAG structure checks (skipped in the project venv)
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
- **First score (Larkana, EMSR629 AOI01).** Threshold preview ΔVV ≤ −3 dB and
  VV ≤ −18 dB, no smoothing: IoU 0.486, precision 0.93, recall 0.50. Two causes
  identified from the outcome map: speckle splits uniform flood across the
  threshold (→ `--smooth`), and VV-only predictions were wrongly dropped where
  VH was negative (→ per-polarization validity). Thresholds tuned on Larkana
  are to be tested on other EMSR629 AOIs, not reported from Larkana itself.
- **Larkana findings (in-sample).** Flood-pass VV alone separates best (single cut
  ≈ −11.3 dB after 3×3, IoU 0.78); VH 0.74; VV change 0.72; reference VV 0.59.
  The change condition adds nothing here: differencing two speckled images adds
  variance, and the area has few dark dry surfaces. ~10% of EMS-flooded pixels
  have VV > −12 dB, which caps darkness rules. Rules A/B/C were fixed from this
  before any held-out scoring.
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
