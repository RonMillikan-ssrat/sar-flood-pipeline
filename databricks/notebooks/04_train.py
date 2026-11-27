# Databricks notebook source
# MAGIC %md
# MAGIC # Train, register, gate, score holdouts: the Airflow `train_and_register` loop on Databricks
# MAGIC
# MAGIC Same code as the Ubuntu box, fed from `silver_pixels` instead of GeoTIFFs:
# MAGIC
# MAGIC 1. Leave-one-AOI-out over the training AOIs (`pixels.leave_one_aoi_out`). Silver rows
# MAGIC    are re-sorted into `load_aoi` order, so with the same seed the same pixels are
# MAGIC    sampled as on Ubuntu; differences come only from library versions (serverless
# MAGIC    scikit-learn vs. the local one).
# MAGIC 2. Log the report to MLflow (`/Users/<you>/sarflood-loao`), fit the final model,
# MAGIC    register it in Unity Catalog (`<catalog>.<schema>.pixel_flood`) with its gate
# MAGIC    value and `labels_version`.
# MAGIC 3. Gated promotion, same rules as `sarflood registry promote`: REJECTED (did not
# MAGIC    beat the champion) is a normal outcome; REFUSED (gate values not comparable,
# MAGIC    e.g. different label versions) fails the job after the holdouts are scored,
# MAGIC    because a person has to decide.
# MAGIC 4. Score the new version on the holdout AOIs (`sarflood-holdout` experiment and
# MAGIC    the `gold_holdout_scores` table).

# COMMAND ----------

# MAGIC %pip install "sarflood[tracking] @ file:///Volumes/workspace/sarflood/raw/wheels/sarflood-0.1.0-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sarflood import dbx
from sarflood.features import FEATURES
from sarflood.lakehouse import flood_pass_of, frame_to_aoi

defaults = {"catalog": "workspace", "schema": "sarflood", "volume": "raw",
            "train_aois": "larkana,shikarpur,jacobabad",
            "holdout_aois": "sanghar,khipro,houston",
            "method": "hgb_bal", "samples": "200000", "seed": "0", "min_delta": "0.0"}
for k, v in defaults.items():
    dbutils.widgets.text(k, v)  # noqa: F821
W = {k: dbutils.widgets.get(k) for k in defaults}  # noqa: F821
CS = f"{W['catalog']}.{W['schema']}"
VOL = Path(f"/Volumes/{W['catalog']}/{W['schema']}/{W['volume']}")
SILVER = f"{CS}.silver_pixels"
TRAIN = [a.strip() for a in W["train_aois"].split(",") if a.strip()]
HOLDOUT = [a.strip() for a in W["holdout_aois"].split(",") if a.strip()]
METHOD, N_TRAIN, SEED = W["method"], int(W["samples"]), int(W["seed"])
MIN_DELTA = float(W["min_delta"])

USER = dbx.current_user(spark)  # noqa: F821
os.environ["SARFLOOD_MLFLOW"] = "databricks"
os.environ["SARFLOOD_EXPERIMENT_PREFIX"] = f"/Users/{USER}/"
os.environ["SARFLOOD_MODEL_NAME"] = f"{CS}.pixel_flood"

from sarflood import tracking
from sarflood.pixels import leave_one_aoi_out

COLS = ["aoi", "stem", "idx", *FEATURES, "y", "frac", "labels_version"]


def load(aoi: str):
    pdf = dbx.silver_aoi(spark, SILVER, aoi, COLS)  # noqa: F821
    if pdf.empty:
        raise RuntimeError(f"{aoi}: no rows in {SILVER}; run the silver task")
    return frame_to_aoi(pdf), str(pdf["stem"].iloc[0])

# COMMAND ----------

# MAGIC %md ## 1. Leave-one-AOI-out from silver

# COMMAND ----------

train, stems = zip(*(load(a) for a in TRAIN), strict=True)
train = list(train)
for d, s in zip(train, stems, strict=True):
    print(f"{d.name:<10} {len(d.y):>11,} px  flood share {d.y.mean():.3f}  labels v{d.labels_version}  {s}")

res = leave_one_aoi_out(train, n_train=N_TRAIN, seed=SEED, importance=False)
for fold in res["folds"].values():
    for e in fold["models"].values():
        e.pop("_prob", None)
res["labels_versions"] = {d.name: d.labels_version for d in train}
res["config"] = {"source": "databricks", "silver": SILVER, "seed": SEED}
tag = f"{flood_pass_of(stems[0])}_{stems[0].rsplit('_', 1)[-1]}"     # e.g. S1A_..._s3
report = VOL / "ml" / "pixels" / f"{tag}_loao.json"
report.parent.mkdir(parents=True, exist_ok=True)
report.write_text(json.dumps(res, indent=2))

for aoi, f in res["folds"].items():
    m = f["models"][METHOD]
    print(f"  held out {aoi:<10} rule A IoU {f['rule_A']['iou']:.3f}   {METHOD} IoU@.5 "
          f"{m['metrics']['iou']:.3f}  AUC {m['auc']:.3f}  n300 {f['fewshot'][METHOD]['300']['mean']:.3f}")

# COMMAND ----------

# MAGIC %md ## 2. MLflow: log the report, fit and register the final model

# COMMAND ----------

tags = {"source": "databricks"}
parent = tracking.log_report(report, tags=tags)
print(f"report run: {parent or 'already logged (same SHA-256)'}")
gate = tracking.gate_value_from_report(report, METHOD)
version, run_id = tracking.fit_final(train, METHOD, N_TRAIN, SEED, gate, report.name, tags)
print(f"registered {tracking.model_name()} v{version} ({METHOD}); "
      f"gate {tracking.GATE_METRIC} = {gate:.4f}")

# COMMAND ----------

# MAGIC %md ## 3. Gated promotion

# COMMAND ----------

refused = None
try:
    decision = tracking.promote(version, MIN_DELTA)
except ValueError as e:
    refused = str(e)
    decision = f"REFUSED: {e}"
print(decision)
for r in tracking.registry_table():
    print(f"  v{r['version']:<3} {r['method'] or '?':<11} gate {r['gate_value'] or '-':<9} "
          f"labels v{r['labels_version']:<8} {' '.join(r['aliases'])}")

# COMMAND ----------

# MAGIC %md ## 4. Holdout scoring of the new version

# COMMAND ----------

model, _, mtags = tracking.load_version(version)
scores = []
for aoi in HOLDOUT:
    d, stem = load(aoi)
    prob = model.predict_proba(d.X)[:, 1]
    r = tracking.holdout_metrics(prob, d.y, d.X[:, FEATURES.index("vv")], seed=SEED)
    r.update(flood_pass=flood_pass_of(stem), labels_version=d.labels_version)
    run = tracking.log_holdout(aoi, version, mtags, r, {"source": "databricks"})
    for who in ("model", "rule_A"):
        x = r[who]
        scores.append({"aoi": aoi, "model_version": version, "who": f"v{version}" if who == "model"
                       else "rule_A", "flood_share": r["flood_share"], "iou": x["iou"],
                       "gain": x["iou"] - r["flood_share"], "precision": x["precision"],
                       "recall": x["recall"], "auc": x["auc"], "ap": x["ap"],
                       "iou_n300": x["iou_n300"], "ceiling": x["best_iou"],
                       "labels_version": d.labels_version, "mlflow_run": run or "",
                       "scored_at": datetime.now(UTC).replace(tzinfo=None)})
    del d, prob

scores = pd.DataFrame(scores)
(spark.createDataFrame(scores).write.format("delta").mode("append")  # noqa: F821
 .saveAsTable(f"{CS}.gold_holdout_scores"))
print(scores.drop(columns=["scored_at", "mlflow_run"]).round(3).to_string(index=False))

# COMMAND ----------

summary = {"version": version, "gate": round(gate, 6), "decision": decision,
           "holdout": scores[["aoi", "who", "iou", "auc", "ap"]].round(4).to_dict(orient="records")}
if refused:
    # Fail after the holdouts are logged: the evidence a person needs is already there.
    raise RuntimeError(f"promotion refused, needs a human decision: {refused}")
dbutils.notebook.exit(json.dumps(summary))  # noqa: F821
