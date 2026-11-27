# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 0 spike: can Free Edition serverless run `sarflood`?
# MAGIC
# MAGIC One question decides the Databricks design: can serverless install the `sarflood`
# MAGIC wheel (rasterio + GDAL, pyshp, scikit-learn) and run `features.load_aoi()` on a
# MAGIC change/labels pair stored in a Volume?
# MAGIC
# MAGIC | # | Check | If it fails |
# MAGIC |---|---|---|
# MAGIC | 1 | Environment: Python, CPU arch, memory, preinstalled numpy/pandas/pyarrow | - |
# MAGIC | 2 | Imports after `%pip install` of the wheel; GDAL version | Fallback: run `load_aoi` on Ubuntu, upload Parquet |
# MAGIC | 3 | Outbound network (PyPI, CDSE, Earth Engine) | Expected for CDSE/EE: ingest stays local |
# MAGIC | 4 | `load_aoi()` reading straight from the Volume (then from a local copy) | Copy-to-local in silver |
# MAGIC | 5 | Write a pixel sample to a Delta table | - |
# MAGIC | 6 | MLflow: the three `tracking.py` snags + UC registry and alias | Decides the `tracking.py` changes |
# MAGIC
# MAGIC **Before running:** upload a pair and the wheel from Ubuntu:
# MAGIC ```
# MAGIC uv build --wheel
# MAGIC python scripts/upload_to_volume.py --aoi sindh --wheel dist/sarflood-0.1.0-py3-none-any.whl
# MAGIC ```
# MAGIC If you changed catalog/schema/volume, edit the wheel path in the next cell too
# MAGIC (`%pip` lines can't read widgets).

# COMMAND ----------

# MAGIC %pip install "sarflood[tracking] @ file:///Volumes/workspace/sarflood/raw/wheels/sarflood-0.1.0-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821  (dbutils/spark/display are notebook globals)

# COMMAND ----------

import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

dbutils.widgets.text("catalog", "workspace")  # noqa: F821
dbutils.widgets.text("schema", "sarflood")  # noqa: F821
dbutils.widgets.text("volume", "raw")  # noqa: F821
dbutils.widgets.text("aoi", "")  # noqa: F821  (blank: first pair found)
dbutils.widgets.text("sample_rows", "200000")  # noqa: F821

CATALOG = dbutils.widgets.get("catalog")  # noqa: F821
SCHEMA = dbutils.widgets.get("schema")  # noqa: F821
VOLUME = dbutils.widgets.get("volume")  # noqa: F821
AOI = dbutils.widgets.get("aoi").strip()  # noqa: F821
SAMPLE = int(dbutils.widgets.get("sample_rows"))  # noqa: F821
VOL = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")

results: list[tuple[str, str, str]] = []  # (check, PASS|FAIL|INFO, detail)


def record(check: str, status: str, detail: object = "") -> None:
    results.append((check, status, str(detail)))
    print(f"{status:<5} {check}: {detail}")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: kB

# COMMAND ----------

# MAGIC %md ## 1. Environment

# COMMAND ----------

with open("/proc/meminfo") as f:
    mem_gb = int(f.readline().split()[1]) / 1e6
record("python", "INFO", sys.version.split()[0])
record("machine", "INFO", f"{platform.machine()}, {os.cpu_count()} CPUs, {mem_gb:.1f} GB RAM")
record("spark", "INFO", spark.version)  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 2. Imports
# MAGIC Watch numpy/pandas/pyarrow: if `%pip` upgraded a preinstalled package, Spark's
# MAGIC pandas conversion can break. The versions printed here tell us.

# COMMAND ----------

import importlib

for mod in ("numpy", "pandas", "pyarrow", "rasterio", "shapefile", "shapely", "sklearn",
            "mlflow", "sarflood"):
    try:
        m = importlib.import_module(mod)
        record(f"import {mod}", "PASS", getattr(m, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        record(f"import {mod}", "FAIL", repr(e))

try:
    import rasterio

    record("gdal", "PASS", f"{rasterio.__gdal_version__}, drivers: "
                           f"GTiff={'GTiff' in rasterio.drivers.raster_driver_extensions().values()}")
except Exception as e:  # noqa: BLE001
    record("gdal", "FAIL", repr(e))

# COMMAND ----------

# MAGIC %md ## 3. Outbound network
# MAGIC Any HTTP status means the host is reachable; an exception means it is blocked.

# COMMAND ----------

import requests

for url in ("https://pypi.org/simple/rasterio/",
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$top=1",
            "https://identity.dataspace.copernicus.eu/",
            "https://earthengine.googleapis.com/"):
    host = url.split("/")[2]
    try:
        r = requests.get(url, timeout=8)
        record(f"net {host}", "INFO", f"reachable (HTTP {r.status_code})")
    except Exception as e:  # noqa: BLE001
        record(f"net {host}", "INFO", f"blocked ({type(e).__name__})")

# COMMAND ----------

# MAGIC %md ## 4. `load_aoi()` on a pair in the Volume

# COMMAND ----------

from sarflood.features import FEATURES, load_aoi

manifests = sorted(VOL.glob("pairs/*/*/manifest.json"))
if AOI:
    manifests = [m for m in manifests if m.parent.parent.name == AOI]
record("pairs in volume", "PASS" if manifests else "FAIL",
       f"{len(manifests)} under {VOL}/pairs" + (f" for aoi={AOI}" if AOI else ""))

data = None
if manifests:
    man_path = manifests[0]
    man = json.loads(man_path.read_text())
    ctif = man_path.parent / man["files"]["change"]["name"]
    ltif = man_path.parent / man["files"]["labels"]["name"]
    record("pair", "INFO", f"{man['aoi']}/{man['stem']}  labels_version={man['labels_version']}  "
                           f"grid_match={man['grid_match']}  "
                           f"{sum(f['bytes'] for f in man['files'].values()) / 1e6:,.1f} MB")

    # 4a: read directly through the /Volumes FUSE mount
    t0 = time.perf_counter()
    try:
        data = load_aoi(man["aoi"], ctif, ltif)
        record("load_aoi from Volume", "PASS", f"{time.perf_counter() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        record("load_aoi from Volume", "FAIL", repr(e))

    # 4b: copy to local disk first (fallback if FUSE reads are slow or fail)
    tmp = Path(tempfile.mkdtemp())
    t0 = time.perf_counter()
    try:
        lc, ll = shutil.copy(ctif, tmp), shutil.copy(ltif, tmp)
        t_copy = time.perf_counter() - t0
        data_local = load_aoi(man["aoi"], Path(lc), Path(ll))
        record("load_aoi from local copy", "PASS",
               f"copy {t_copy:.1f}s + load {time.perf_counter() - t0 - t_copy:.1f}s")
        data = data or data_local
    except Exception as e:  # noqa: BLE001
        record("load_aoi from local copy", "FAIL", repr(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if data is not None:
    record("scored pixels", "INFO",
           f"{data.X.shape[0]:,} x {data.X.shape[1]} features, X = {data.X.nbytes / 1e9:.2f} GB, "
           f"flood share {data.y.mean():.3f}, peak RSS {peak_rss_gb():.1f} GB")

# COMMAND ----------

# MAGIC %md ## 5. Pixel sample to Delta

# COMMAND ----------

import numpy as np
import pandas as pd

pdf = None
if data is not None:
    n = min(SAMPLE, len(data.y))
    sel = np.sort(np.random.default_rng(0).choice(len(data.y), n, replace=False))
    pdf = pd.DataFrame(data.X[sel], columns=FEATURES)
    pdf.insert(0, "aoi", data.name)
    pdf.insert(1, "idx", data.idx[sel].astype("int64"))
    pdf["y"] = data.y[sel]
    pdf["frac"] = data.frac[sel].astype("float32")
    pdf["labels_version"] = data.labels_version

    table = f"{CATALOG}.{SCHEMA}.spike_pixels"
    t0 = time.perf_counter()
    try:
        spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable(table)  # noqa: F821
        cnt = spark.table(table).count()  # noqa: F821
        record("delta write", "PASS" if cnt == n else "FAIL",
               f"{cnt:,} rows -> {table} in {time.perf_counter() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        record("delta write", "FAIL", repr(e))

# COMMAND ----------

# MAGIC %md ## 6. MLflow on Databricks
# MAGIC Snags 1 and 2 are probes: INFO either way, the detail says which behaviour we got.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
user = spark.sql("select current_user()").first()[0]  # noqa: F821

# Snag 1: tracking.setup() passes a local file:// artifact_location
try:
    mlflow.create_experiment(f"/Users/{user}/sarflood-spike-localart-{int(time.time())}",
                             artifact_location="file:///tmp/mlartifacts")
    record("snag 1: local artifact_location", "INFO", "accepted")
except Exception as e:  # noqa: BLE001
    record("snag 1: local artifact_location", "INFO", f"rejected: {type(e).__name__}: {e}"[:200])

# Snag 2: bare experiment name, as in tracking.EXPERIMENT
try:
    mlflow.set_experiment("sarflood-loao")
    record("snag 2: bare experiment name", "INFO", "accepted")
except Exception as e:  # noqa: BLE001
    record("snag 2: bare experiment name", "INFO", f"rejected: {type(e).__name__}: {e}"[:200])

# Snag 3: UC registry needs catalog.schema.model and a signature; aliases carry over
model_name = f"{CATALOG}.{SCHEMA}.spike_pixel_model"
if pdf is not None:
    try:
        mlflow.set_experiment(f"/Users/{user}/sarflood-spike")
        X, y = pdf[FEATURES].to_numpy(), pdf["y"].to_numpy()
        clf = LogisticRegression(max_iter=300).fit(X, y)
        with mlflow.start_run(run_name="phase0-spike"):
            mlflow.log_metric("train_accuracy", clf.score(X, y))
            mlflow.sklearn.log_model(clf, name="model",
                                     signature=infer_signature(X[:100], clf.predict(X[:100])),
                                     registered_model_name=model_name)
        c = MlflowClient()
        v = max(int(m.version) for m in c.search_model_versions(f"name='{model_name}'"))
        c.set_registered_model_alias(model_name, "champion", v)
        got = c.get_model_version_by_alias(model_name, "champion").version
        record("snag 3: UC registry + alias", "PASS", f"{model_name} v{got} = champion")
    except Exception as e:  # noqa: BLE001
        record("snag 3: UC registry + alias", "FAIL", f"{type(e).__name__}: {e}"[:300])

# COMMAND ----------

# MAGIC %md ## Summary (paste this back)

# COMMAND ----------

width = max(len(r[0]) for r in results)
print("\n".join(f"{s:<5} {c:<{width}}  {d}" for c, s, d in results))
n_fail = sum(s == "FAIL" for _, s, _ in results)
print(f"\n{n_fail} FAIL, peak RSS {peak_rss_gb():.1f} GB")
dbutils.notebook.exit(json.dumps({"fail": n_fail, "results": results}))  # noqa: F821
