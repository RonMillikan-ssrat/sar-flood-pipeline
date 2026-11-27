# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: feature distributions and label-free drift
# MAGIC
# MAGIC * `gold_feature_stats`: per AOI and feature, count, mean, std, p01/p50/p99, and the
# MAGIC   AOI's flood share.
# MAGIC * `gold_drift`: population stability index (PSI) of each feature for each AOI against
# MAGIC   the pooled training AOIs. PSI uses **no labels**: it compares where the inputs fall,
# MAGIC   so it can flag a new scene (e.g. Bangladesh, or Houston's urban landscape) before
# MAGIC   anyone labels it. Conventional reading: < 0.10 stable, 0.10-0.25 moderate,
# MAGIC   >= 0.25 major shift. Training AOIs are scored too, as a within-region baseline.

# COMMAND ----------

# MAGIC %pip install "sarflood[tracking] @ file:///Volumes/workspace/sarflood/raw/wheels/sarflood-0.1.0-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import json

import pandas as pd
from pyspark.sql import functions as F

from sarflood.features import FEATURES
from sarflood.lakehouse import psi, psi_edges, psi_level

for k, v in {"catalog": "workspace", "schema": "sarflood", "train_aois":
             "larkana,shikarpur,jacobabad", "sample_rows": "300000", "seed": "0"}.items():
    dbutils.widgets.text(k, v)  # noqa: F821
W = {k: dbutils.widgets.get(k) for k in  # noqa: F821
     ("catalog", "schema", "train_aois", "sample_rows", "seed")}
CS = f"{W['catalog']}.{W['schema']}"
SILVER = spark.table(f"{CS}.silver_pixels")  # noqa: F821
TRAIN = [a.strip() for a in W["train_aois"].split(",") if a.strip()]
N, SEED = int(W["sample_rows"]), int(W["seed"])

# COMMAND ----------

aggs = [F.count("*").alias("n"), F.avg(F.col("y").cast("double")).alias("flood_share")]
for f in FEATURES:
    aggs += [F.avg(f).alias(f"{f}__mean"), F.stddev(f).alias(f"{f}__std"),
             F.percentile_approx(f, [0.01, 0.5, 0.99], 10000).alias(f"{f}__q")]
wide = SILVER.groupBy("aoi", "labels_version").agg(*aggs).toPandas()
stats = pd.DataFrame([
    {"aoi": r["aoi"], "labels_version": r["labels_version"], "feature": f, "n": int(r["n"]),
     "flood_share": float(r["flood_share"]), "mean": float(r[f"{f}__mean"]),
     "std": float(r[f"{f}__std"]), "p01": float(r[f"{f}__q"][0]),
     "p50": float(r[f"{f}__q"][1]), "p99": float(r[f"{f}__q"][2])}
    for _, r in wide.iterrows() for f in FEATURES])
(spark.createDataFrame(stats).write.format("delta").mode("overwrite")  # noqa: F821
 .option("overwriteSchema", "true").saveAsTable(f"{CS}.gold_feature_stats"))

# COMMAND ----------


def sample(aois: list[str]) -> pd.DataFrame:
    sub = SILVER.where(F.col("aoi").isin(aois))
    n = sub.count()
    frac = min(1.0, N / max(n, 1))
    return sub.sample(withReplacement=False, fraction=frac, seed=SEED).select(*FEATURES).toPandas()


ref = sample(TRAIN)
edges = {f: psi_edges(ref[f].to_numpy()) for f in FEATURES}
drift = []
for aoi in sorted(wide["aoi"]):
    cur = sample([aoi])
    for f in FEATURES:
        v = psi(ref[f].to_numpy(), cur[f].to_numpy(), edges[f])
        drift.append({"aoi": aoi, "role": "train" if aoi in TRAIN else "holdout", "feature": f,
                      "psi": round(v, 4), "level": psi_level(v),
                      "n_reference": len(ref), "n_aoi": len(cur)})
drift = pd.DataFrame(drift)
(spark.createDataFrame(drift).write.format("delta").mode("overwrite")  # noqa: F821
 .option("overwriteSchema", "true").saveAsTable(f"{CS}.gold_drift"))

# COMMAND ----------

by_aoi = (drift.groupby(["aoi", "role"])
          .agg(max_psi=("psi", "max"), mean_psi=("psi", "mean"),
               n_major=("level", lambda s: int((s == "major").sum())))
          .reset_index().sort_values("max_psi"))
print(by_aoi.to_string(index=False))
print()
print(drift.pivot(index="feature", columns="aoi", values="psi").round(3).to_string())
dbutils.notebook.exit(json.dumps(by_aoi.round(4).to_dict(orient="records")))  # noqa: F821
