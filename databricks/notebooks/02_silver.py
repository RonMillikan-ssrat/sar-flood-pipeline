# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: one row per scored pixel
# MAGIC
# MAGIC For each `ok` pair in `bronze_pairs`: run `features.load_aoi()` straight from the
# MAGIC Volume (the Phase 0 spike showed serverless handles Houston, 12.8M pixels, in 12 s
# MAGIC and 3.6 GB), convert with `lakehouse.aoi_to_frame()` and replace that AOI's rows in
# MAGIC `silver_pixels`. An AOI whose (stem, checksums) are already in silver is skipped
# MAGIC unless `force`. Every write is timed into `silver_loads`, which is also where the
# MAGIC arrow-vs-parquet write comparison is read from.

# COMMAND ----------

# MAGIC %pip install "sarflood[tracking] @ file:///Volumes/workspace/sarflood/raw/wheels/sarflood-0.1.0-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sarflood import dbx
from sarflood.features import load_aoi
from sarflood.lakehouse import aoi_to_frame

for k, v in {"catalog": "workspace", "schema": "sarflood", "volume": "raw",
             "aois": "", "write_method": "parquet", "table": "silver_pixels",
             "force": "false"}.items():
    dbutils.widgets.text(k, v)  # noqa: F821
W = {k: dbutils.widgets.get(k) for k in  # noqa: F821
     ("catalog", "schema", "volume", "aois", "write_method", "table", "force")}
CS = f"{W['catalog']}.{W['schema']}"
VOL = Path(f"/Volumes/{W['catalog']}/{W['schema']}/{W['volume']}")
TABLE = f"{CS}.{W['table']}"
METHOD = W["write_method"]
FORCE = W["force"].lower() == "true"
ONLY = [a.strip() for a in W["aois"].split(",") if a.strip()]
if METHOD not in dbx.WRITERS:
    raise ValueError(f"write_method must be one of {sorted(dbx.WRITERS)}")

# COMMAND ----------

bronze = spark.table(f"{CS}.bronze_pairs").where("status = 'ok'").toPandas()  # noqa: F821
if ONLY:
    bronze = bronze[bronze["aoi"].isin(ONLY)]
loads = []
for b in bronze.sort_values("aoi").itertuples():
    key = (b.stem, b.change_sha256, b.labels_sha256)
    if not FORCE and dbx.loaded_keys(spark, TABLE, b.aoi) == [key]:  # noqa: F821
        print(f"same   {b.aoi:<10} {b.stem}")
        continue
    folder = Path(b.manifest_path).parent
    t0 = time.perf_counter()
    d = load_aoi(b.aoi, folder / b.change_name, folder / b.labels_name)
    if d.labels_version != b.labels_version:
        raise RuntimeError(f"{b.aoi}: labels file says v{d.labels_version}, "
                           f"bronze says v{b.labels_version}")
    pdf = aoi_to_frame(d, b.stem, b.change_sha256, b.labels_sha256)
    t_load = time.perf_counter() - t0
    if METHOD == "parquet":
        t_write = dbx.write_parquet(spark, pdf, TABLE, b.aoi, VOL / "_staging" / W["table"])  # noqa: F821
    else:
        t_write = dbx.write_arrow(spark, pdf, TABLE, b.aoi)  # noqa: F821
    n = len(pdf)
    loads.append({"aoi": b.aoi, "stem": b.stem, "table": TABLE, "rows": n, "method": METHOD,
                  "load_s": round(t_load, 1), "write_s": round(t_write, 1),
                  "rows_per_s": int(n / t_write) if t_write else None,
                  "labels_version": d.labels_version,
                  "loaded_at": datetime.now(UTC).replace(tzinfo=None)})
    print(f"put    {b.aoi:<10} {n:>12,} rows  load {t_load:6.1f}s  "
          f"{METHOD} write {t_write:6.1f}s  ({n / t_write:,.0f} rows/s)")
    del d, pdf
    gc.collect()

if loads:
    (spark.createDataFrame(pd.DataFrame(loads)).write.format("delta")  # noqa: F821
     .mode("append").saveAsTable(f"{CS}.silver_loads"))

# COMMAND ----------

counts = spark.sql(  # noqa: F821
    f"SELECT aoi, labels_version, count(*) AS rows, round(avg(cast(y AS INT)), 3) AS flood_share "
    f"FROM {TABLE} GROUP BY aoi, labels_version ORDER BY aoi").toPandas()
print(counts.to_string(index=False))
dbutils.notebook.exit(json.dumps({"written": [x["aoi"] for x in loads],  # noqa: F821
                                  "rows": int(counts["rows"].sum()), "method": METHOD}))
