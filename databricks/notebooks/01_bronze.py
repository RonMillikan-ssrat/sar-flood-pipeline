# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: one row per uploaded pair
# MAGIC
# MAGIC Reads every `pairs/<aoi>/<stem>/manifest.json` in the Volume (a manifest is written
# MAGIC last, so its presence means a complete pair) and MERGEs one row per (aoi, stem) into
# MAGIC `bronze_pairs`. Pairs that fail a check (grid mismatch, file missing, unknown
# MAGIC manifest schema) get `status = 'quarantine'` with the reason: recorded and visible,
# MAGIC never silently dropped, and silver skips them. Re-running changes nothing unless a
# MAGIC pair's checksums or status changed.

# COMMAND ----------

# MAGIC %pip install "sarflood[tracking] @ file:///Volumes/workspace/sarflood/raw/wheels/sarflood-0.1.0-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821  (dbutils/spark are notebook globals)

# COMMAND ----------

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sarflood.lakehouse import BRONZE_COLUMNS, ROLES, bronze_row

dbutils.widgets.text("catalog", "workspace")  # noqa: F821
dbutils.widgets.text("schema", "sarflood")  # noqa: F821
dbutils.widgets.text("volume", "raw")  # noqa: F821
CATALOG, SCHEMA, VOLUME = (dbutils.widgets.get(k) for k in ("catalog", "schema", "volume"))  # noqa: F821
VOL = Path(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")
TABLE = f"{CATALOG}.{SCHEMA}.bronze_pairs"

# COMMAND ----------

rows = []
for m in sorted(VOL.glob("pairs/*/*/manifest.json")):
    man = json.loads(m.read_text())
    present = {r: (m.parent / man["files"][r]["name"]).exists() for r in ROLES}
    rows.append(bronze_row(man, str(m), present))
if not rows:
    raise RuntimeError(f"no manifests under {VOL}/pairs: run scripts/upload_to_volume.py")

pdf = pd.DataFrame(rows, columns=BRONZE_COLUMNS)
pdf["ingested_at"] = datetime.now(UTC).replace(tzinfo=None)
spark.createDataFrame(pdf).createOrReplaceTempView("incoming")  # noqa: F821

if not spark.catalog.tableExists(TABLE):  # noqa: F821
    spark.table("incoming").write.format("delta").saveAsTable(TABLE)  # noqa: F821
    print(f"created {TABLE}")
else:
    merge_sql = f"""
        MERGE INTO {TABLE} t USING incoming s
        ON t.aoi = s.aoi AND t.stem = s.stem
        WHEN MATCHED AND (t.change_sha256 <> s.change_sha256 OR t.labels_sha256 <> s.labels_sha256
                          OR t.status <> s.status OR t.reason <> s.reason)
          THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """
    spark.sql(merge_sql)  # noqa: F821

# COMMAND ----------

summary = spark.table(TABLE).select("aoi", "stem", "labels_version", "status", "reason",  # noqa: F821
                                    "change_bytes", "labels_bytes").orderBy("aoi").toPandas()
print(summary.to_string(index=False))
n_ok = int((summary["status"] == "ok").sum())
n_q = int((summary["status"] == "quarantine").sum())
if n_ok == 0:
    raise RuntimeError("no pairs passed bronze checks")
dbutils.notebook.exit(json.dumps({"ok": n_ok, "quarantine": n_q,  # noqa: F821
                                  "aois": sorted(summary["aoi"].tolist())}))
