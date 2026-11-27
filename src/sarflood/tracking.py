"""MLflow tracking and model registry for sarflood.

Storage (local, no server needed): SQLite backend + artifact folder under
$DATA_ROOT/mlflow/. Set MLFLOW_TRACKING_URI to use a server instead.

1. log_report(): ingest a leave-one-AOI-out report (train-pixels or train-unet JSON)
   as one parent run with a nested run per method (rule_A, logreg, hgb, ..., unet).
   Per method: per-fold metrics '<aoi>.<metric>' and their means 'mean.<metric>'.
   The report JSON and its PNGs are logged as artifacts. Re-ingesting the same
   report (same SHA-256) is skipped.

2. fit_final(): train a pixel model on all AOIs, log it (mlflow.sklearn) and register
   a version of model_name(). Its gate metric (mean held-out IoU with ~300 local
   labels for the cut, from the LOAO report) is stored as a version tag.

3. promote() / rollback(): alias 'champion' moves to a candidate only if the
   candidate's gate metric beats the champion's by at least min_delta (or force).
   The displaced champion gets alias 'previous'; rollback swaps them back.
   Versions whose gate values were computed on different label definitions
   (tag labels_version; untagged = "1") are not compared: promote refuses, and a
   person decides with --force after looking at both.

4. score_holdout(): apply a registered version to an AOI it never saw and log
   AUC/AP/IoU (with the all-flood baseline) to the 'sarflood-holdout' experiment.

Backends (env SARFLOOD_MLFLOW):
    local (default)  SQLite + artifact folder under $DATA_ROOT/mlflow; experiments by
                     bare name; model 'sarflood-pixel-flood' in the workspace registry.
    databricks       tracking URI 'databricks', registry 'databricks-uc'. Databricks
                     rejects bare experiment names and file:// artifact locations
                     (Phase 0 spike), so experiments live under
                     SARFLOOD_EXPERIMENT_PREFIX (e.g. /Users/<you>/) with the default
                     artifact store, and the model is a Unity Catalog name
                     (SARFLOOD_MODEL_NAME, default workspace.sarflood.pixel_flood).
                     UC also requires a model signature, which fit_final always logs.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT = "sarflood-loao"
REGISTERED_MODEL = "sarflood-pixel-flood"          # local default; see model_name()
UC_MODEL_DEFAULT = "workspace.sarflood.pixel_flood"
GATE_METRIC = "mean.iou_n300"   # mean held-out IoU, cut chosen from 300 local labels
HOLDOUT_EXPERIMENT = "sarflood-holdout"


def combined_labels_version(versions) -> str:
    """One version string for a set of AOIs: the shared version, or 'mixed:a,b'."""
    vs = sorted(set(versions))
    return vs[0] if len(vs) == 1 else "mixed:" + ",".join(vs)


def labels_version_of_report(report: Path) -> str:
    rep = json.loads(report.read_text())
    lv = rep.get("labels_versions")
    return combined_labels_version(lv.values()) if lv else "1"


def backend() -> str:
    b = os.environ.get("SARFLOOD_MLFLOW", "local").strip().lower()
    if b not in ("local", "databricks"):
        raise ValueError(f"SARFLOOD_MLFLOW must be 'local' or 'databricks', not {b!r}")
    return b


def experiment_name(name: str) -> str:
    """Bare name locally; an absolute workspace path on Databricks."""
    if backend() == "local":
        return name
    prefix = os.environ.get("SARFLOOD_EXPERIMENT_PREFIX", "").strip()
    if not prefix.startswith("/"):
        raise ValueError("SARFLOOD_EXPERIMENT_PREFIX must be an absolute workspace path such "
                         "as /Users/<you>/ when SARFLOOD_MLFLOW=databricks")
    return prefix.rstrip("/") + "/" + name


def model_name() -> str:
    if os.environ.get("SARFLOOD_MODEL_NAME"):
        return os.environ["SARFLOOD_MODEL_NAME"]
    return UC_MODEL_DEFAULT if backend() == "databricks" else REGISTERED_MODEL


def mlflow_dir() -> Path:
    from .manifest import data_root

    return data_root() / "mlflow"


def setup(experiment: str = EXPERIMENT):
    """Point MLflow at the backend and select the experiment (created if missing).
    Local: SQLite store (unless MLFLOW_TRACKING_URI is set) and an explicit artifact
    location. Databricks: workspace tracking, Unity Catalog registry, default artifacts."""
    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if backend() == "databricks":
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(experiment_name(experiment))
        return mlflow
    d = mlflow_dir()
    d.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(d / 'mlflow.db').resolve()}")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        mlflow.create_experiment(experiment,
                                 artifact_location=(d / "artifacts").resolve().as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


# ---------------------------------------------------------------- reports -> runs

def methods_from_report(rep: dict) -> tuple[str, dict[str, dict[str, dict]]]:
    """Normalize a LOAO report to (kind, {method: {aoi: {metric: value}}})."""
    out: dict[str, dict[str, dict]] = {}

    def put(method, aoi, m: dict, fs: dict | None):
        row = {k: m[k] for k in ("iou", "precision", "recall", "f1", "auc", "ap", "best_iou")
               if k in m and m[k] is not None}
        for n in ("100", "300", "1000"):
            if fs and n in fs:
                row[f"iou_n{n}"] = fs[n]["mean"]
        out.setdefault(method, {})[aoi] = row

    folds = rep["folds"]
    first = next(iter(folds.values()))
    kind = "unet" if "unet" in first else "pixels"
    for aoi, f in folds.items():
        fs = f.get("fewshot", {})
        put("rule_A", aoi, f["rule_A"], fs.get("vv"))
        if kind == "unet":
            put("unet", aoi, f["unet"], fs.get("unet"))
        else:
            for m, e in f["models"].items():
                put(m, aoi, {**e["metrics"], **e}, fs.get(m))
    return kind, out


def _flatten_params(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_params(v, f"{key}."))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
    return out


def log_report(report: Path, tags: dict | None = None, force: bool = False) -> str | None:
    """Log a LOAO report. Returns the parent run id, or None if already logged."""
    mlflow = setup()
    raw = report.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if not force:
        hit = mlflow.search_runs(filter_string=f"tags.report_sha256 = '{sha}'",
                                 output_format="list")
        if hit:
            return None
    rep = json.loads(raw)
    kind, methods = methods_from_report(rep)
    params = {"kind": kind, "report": report.name, "aois": ",".join(rep["folds"])}
    if "config" in rep:
        params.update(_flatten_params(rep["config"], "config."))
    if "n_train_per_aoi" in rep:
        params["n_train_per_aoi"] = rep["n_train_per_aoi"]
    with mlflow.start_run(run_name=f"{kind}:{report.stem}") as parent:
        mlflow.set_tags({"report_sha256": sha, "kind": kind, **(tags or {})})
        mlflow.log_params(params)
        mlflow.log_artifact(str(report))
        prefix = report.stem.removesuffix("_loao")
        for png in sorted(report.parent.glob(f"{prefix}_*.png")):
            mlflow.log_artifact(str(png), artifact_path="maps")
        for method, folds in methods.items():
            with mlflow.start_run(run_name=method, nested=True):
                mlflow.set_tags({"method": method, "kind": kind, **(tags or {})})
                metrics = {}
                for aoi, row in folds.items():
                    metrics.update({f"{aoi}.{k}": v for k, v in row.items()})
                for k in sorted({k for row in folds.values() for k in row}):
                    vals = [row[k] for row in folds.values() if k in row]
                    if len(vals) == len(folds):
                        metrics[f"mean.{k}"] = sum(vals) / len(vals)
                mlflow.log_metrics(metrics)
        return parent.info.run_id


def gate_value_from_report(report: Path, method: str, metric: str = GATE_METRIC) -> float:
    _, methods = methods_from_report(json.loads(report.read_text()))
    if method not in methods:
        raise ValueError(f"method {method!r} not in {report.name}: {sorted(methods)}")
    key = metric.removeprefix("mean.")
    vals = [row[key] for row in methods[method].values()]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------- final model + registry

def skops_trusted_types(model) -> list[str]:
    """Types in `model` that skops (MLflow 3's default sklearn format) won't load
    without being declared trusted. We trust them only if every one is scikit-learn's
    own: anything else is an error rather than silently trusted."""
    import skops.io as sio

    types = sorted(sio.get_untrusted_types(data=sio.dumps(model)))
    foreign = [t for t in types if not t.startswith("sklearn.")]
    if foreign:
        raise ValueError(f"model contains non-scikit-learn types; refusing to trust: {foreign}")
    return types


def fit_final(data: list, method: str, n_train: int, seed: int, gate_value: float,
              gate_source: str, tags: dict) -> tuple[int, str]:
    """Train `method` (logreg/hgb, optionally '_bal') on all AOIs, log and register it.
    Returns (version, run_id)."""
    import numpy as np

    from .features import FEATURES
    from .pixels import make_models, sample_pure

    base = method.removesuffix("_bal")
    balanced = method.endswith("_bal")
    models = make_models(seed)
    if base not in models:
        raise ValueError(f"unknown method {method!r}; use one of "
                         f"{sorted(models) + [m + '_bal' for m in models]}")
    rng = np.random.default_rng(seed)
    parts = [sample_pure(d, n_train, rng, balanced) for d in data]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    model = models[base].fit(X, y)

    trusted = skops_trusted_types(model)
    mlflow = setup()
    with mlflow.start_run(run_name=f"final:{method}") as run:
        mlflow.set_tags({"stage": "final", "method": method, **tags})
        mlflow.log_params({"method": method, "n_train_per_aoi": n_train, "seed": seed,
                           "train_aois": ",".join(d.name for d in data),
                           "features": ",".join(FEATURES), "n_train": int(y.size),
                           "train_flood_share": round(float(y.mean()), 4)})
        mlflow.log_metric("gate." + GATE_METRIC, gate_value)
        mlflow.set_tag("skops_trusted_types", ",".join(trusted))
        from mlflow.models import infer_signature

        sig = infer_signature(X[:100], model.predict_proba(X[:100])[:, 1])
        info = mlflow.sklearn.log_model(model, name="model", input_example=X[:5],
                                        signature=sig, skops_trusted_types=trusted)
    mv = mlflow.register_model(info.model_uri, model_name(), tags={
        "method": method, "gate_metric": GATE_METRIC, "gate_value": f"{gate_value:.6f}",
        "gate_source": gate_source, "features": ",".join(FEATURES),
        "train_aois": ",".join(d.name for d in data),
        "labels_version": combined_labels_version(d.labels_version for d in data), **tags})
    return int(mv.version), run.info.run_id


def _client():
    from mlflow import MlflowClient

    setup()
    return MlflowClient()


def _alias_version(client, alias: str) -> int | None:
    """Version holding `alias`, or None if the alias (or the model) doesn't exist."""
    from mlflow.exceptions import MlflowException

    try:
        return int(client.get_model_version_by_alias(model_name(), alias).version)
    except MlflowException:
        return None


def registry_table() -> list[dict]:
    client = _client()
    champ, prev = _alias_version(client, "champion"), _alias_version(client, "previous")
    rows = []
    for mv in client.search_model_versions(f"name = '{model_name()}'"):
        v = int(mv.version)
        rows.append({"version": v, "method": mv.tags.get("method"),
                     "gate_value": mv.tags.get("gate_value"),
                     "labels_version": mv.tags.get("labels_version", "1"),
                     "aliases": [a for a, x in (("champion", champ), ("previous", prev)) if x == v],
                     "created": datetime.fromtimestamp(mv.creation_timestamp / 1000, UTC)
                     .strftime("%Y-%m-%d %H:%M")})
    return sorted(rows, key=lambda r: r["version"])


def promote(version: int, min_delta: float = 0.0, force: bool = False) -> str:
    client = _client()
    cand = client.get_model_version(model_name(), str(version))
    champ_v = _alias_version(client, "champion")
    if champ_v == version:
        return f"version {version} is already champion"
    if champ_v is not None and not force:
        champ = client.get_model_version(model_name(), str(champ_v))
        if cand.tags.get("gate_metric") != champ.tags.get("gate_metric"):
            raise ValueError("gate metrics differ between candidate and champion; use --force")
        lc, lch = cand.tags.get("labels_version", "1"), champ.tags.get("labels_version", "1")
        if lc != lch:
            raise ValueError(f"labels_version differs (candidate v{version}: {lc}, champion "
                             f"v{champ_v}: {lch}): the gate values were computed on different "
                             "label definitions and are not comparable. Review both, then "
                             "promote with --force if the candidate should replace the champion")
        c, ch = float(cand.tags["gate_value"]), float(champ.tags["gate_value"])
        if c < ch + min_delta:
            return (f"REJECTED: v{version} {GATE_METRIC}={c:.4f} does not beat champion "
                    f"v{champ_v} ({ch:.4f}) by >= {min_delta}")
    if champ_v is not None:
        client.set_registered_model_alias(model_name(), "previous", str(champ_v))
    client.set_registered_model_alias(model_name(), "champion", str(version))
    client.set_model_version_tag(model_name(), str(version), "promoted_at",
                                 datetime.now(UTC).isoformat(timespec="seconds"))
    return f"PROMOTED: v{version} is champion" + (f" (v{champ_v} -> previous)" if champ_v else "")


def rollback() -> str:
    client = _client()
    champ_v, prev_v = _alias_version(client, "champion"), _alias_version(client, "previous")
    if prev_v is None:
        return "nothing to roll back to (no 'previous' alias)"
    client.set_registered_model_alias(model_name(), "champion", str(prev_v))
    if champ_v is not None:
        client.set_registered_model_alias(model_name(), "previous", str(champ_v))
        client.set_model_version_tag(model_name(), str(champ_v), "rolled_back_at",
                                     datetime.now(UTC).isoformat(timespec="seconds"))
    return f"ROLLED BACK: v{prev_v} is champion again" + (f" (v{champ_v} -> previous)" if champ_v else "")


# ---------------------------------------------------------------- holdout scoring

def load_version(version: int | None = None, alias: str = "champion"):
    """(model, version, tags) for a registered version, or the one holding `alias`."""
    import mlflow.sklearn

    client = _client()
    v = version if version is not None else _alias_version(client, alias)
    if v is None:
        raise ValueError(f"no version holds alias {alias!r}")
    mv = client.get_model_version(model_name(), str(v))
    model = mlflow.sklearn.load_model(f"models:/{model_name()}/{v}")
    return model, int(v), dict(mv.tags)


def holdout_metrics(prob, y, vv, seed: int = 0) -> dict:
    """Scores of a model probability map and of rule A on one AOI's scored pixels."""
    import numpy as np

    from .pixels import RULE_A_DB, fewshot_iou, metrics, threshold_free

    rng = np.random.default_rng(seed)
    share = float(y.mean())
    model = {**metrics(prob >= 0.5, y), **threshold_free(prob, y),
             "iou_n300": fewshot_iou(-prob, y, rng, ns=(300,))["300"]["mean"]}
    rule = {**metrics(vv <= RULE_A_DB, y), **threshold_free(-vv, y),
            "iou_n300": fewshot_iou(vv, y, rng, ns=(300,))["300"]["mean"]}
    rule["best_cut"] = -rule["best_cut"]
    return {"flood_share": share, "iou_all_flood": share, "n": int(y.size),
            "model": model, "rule_A": rule}


def log_holdout(aoi: str, version: int, tags: dict, result: dict, run_tags: dict,
                force: bool = False) -> str | None:
    """Log one holdout score. A run with the same AOI, flood pass, model version and
    labels version already logged is not logged again (returns None) unless force."""
    mlflow = setup(HOLDOUT_EXPERIMENT)
    key = {"aoi": aoi, "flood_pass": str(result.get("flood_pass", "")),
           "model_version": str(version), "labels_version": str(result.get("labels_version", "1"))}
    if not force:
        filt = " and ".join(f"tags.{k} = '{v}'" for k, v in key.items())
        if mlflow.search_runs(experiment_names=[experiment_name(HOLDOUT_EXPERIMENT)],
                              filter_string=filt,
                              output_format="list", max_results=1):
            return None
    with mlflow.start_run(run_name=f"v{version}:{aoi}") as run:
        mlflow.set_tags({**key,
                         "model_method": tags.get("method", "?"),
                         "model_labels_version": tags.get("labels_version", "1"), **run_tags})
        mlflow.log_metrics({"flood_share": result["flood_share"],
                            "iou_all_flood": result["iou_all_flood"]})
        for who in ("model", "rule_A"):
            r = result[who]
            mlflow.log_metrics({f"{who}.{k}": float(r[k]) for k in
                                ("iou", "precision", "recall", "auc", "ap", "best_iou", "iou_n300")})
        return run.info.run_id
