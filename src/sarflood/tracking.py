"""MLflow tracking and model registry for sarflood.

Storage (local, no server needed): SQLite backend + artifact folder under
$DATA_ROOT/mlflow/. Set MLFLOW_TRACKING_URI to use a server instead.

1. log_report(): ingest a leave-one-AOI-out report (train-pixels or train-unet JSON)
   as one parent run with a nested run per method (rule_A, logreg, hgb, ..., unet).
   Per method: per-fold metrics '<aoi>.<metric>' and their means 'mean.<metric>'.
   The report JSON and its PNGs are logged as artifacts. Re-ingesting the same
   report (same SHA-256) is skipped.

2. fit_final(): train a pixel model on all AOIs, log it (mlflow.sklearn) and register
   a version of REGISTERED_MODEL. Its gate metric (mean held-out IoU with ~300 local
   labels for the cut, from the LOAO report) is stored as a version tag.

3. promote() / rollback(): alias 'champion' moves to a candidate only if the
   candidate's gate metric beats the champion's by at least min_delta (or force).
   The displaced champion gets alias 'previous'; rollback swaps them back.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT = "sarflood-loao"
REGISTERED_MODEL = "sarflood-pixel-flood"
GATE_METRIC = "mean.iou_n300"   # mean held-out IoU, cut chosen from 300 local labels


def mlflow_dir() -> Path:
    from .manifest import data_root

    return data_root() / "mlflow"


def setup(experiment: str = EXPERIMENT):
    """Point MLflow at the local store (unless MLFLOW_TRACKING_URI is set) and make
    sure the experiment exists with an explicit artifact location."""
    import mlflow

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
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
        info = mlflow.sklearn.log_model(model, name="model", input_example=X[:5],
                                        skops_trusted_types=trusted)
    mv = mlflow.register_model(info.model_uri, REGISTERED_MODEL, tags={
        "method": method, "gate_metric": GATE_METRIC, "gate_value": f"{gate_value:.6f}",
        "gate_source": gate_source, "features": ",".join(FEATURES), **tags})
    return int(mv.version), run.info.run_id


def _client():
    from mlflow import MlflowClient

    setup()
    return MlflowClient()


def _alias_version(client, alias: str) -> int | None:
    """Version holding `alias`, or None if the alias (or the model) doesn't exist."""
    from mlflow.exceptions import MlflowException

    try:
        return int(client.get_model_version_by_alias(REGISTERED_MODEL, alias).version)
    except MlflowException:
        return None


def registry_table() -> list[dict]:
    client = _client()
    champ, prev = _alias_version(client, "champion"), _alias_version(client, "previous")
    rows = []
    for mv in client.search_model_versions(f"name = '{REGISTERED_MODEL}'"):
        v = int(mv.version)
        rows.append({"version": v, "method": mv.tags.get("method"),
                     "gate_value": mv.tags.get("gate_value"),
                     "aliases": [a for a, x in (("champion", champ), ("previous", prev)) if x == v],
                     "created": datetime.fromtimestamp(mv.creation_timestamp / 1000, UTC)
                     .strftime("%Y-%m-%d %H:%M")})
    return sorted(rows, key=lambda r: r["version"])


def promote(version: int, min_delta: float = 0.0, force: bool = False) -> str:
    client = _client()
    cand = client.get_model_version(REGISTERED_MODEL, str(version))
    champ_v = _alias_version(client, "champion")
    if champ_v == version:
        return f"version {version} is already champion"
    if champ_v is not None and not force:
        champ = client.get_model_version(REGISTERED_MODEL, str(champ_v))
        if cand.tags.get("gate_metric") != champ.tags.get("gate_metric"):
            raise ValueError("gate metrics differ between candidate and champion; use --force")
        c, ch = float(cand.tags["gate_value"]), float(champ.tags["gate_value"])
        if c < ch + min_delta:
            return (f"REJECTED: v{version} {GATE_METRIC}={c:.4f} does not beat champion "
                    f"v{champ_v} ({ch:.4f}) by >= {min_delta}")
    if champ_v is not None:
        client.set_registered_model_alias(REGISTERED_MODEL, "previous", str(champ_v))
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", str(version))
    client.set_model_version_tag(REGISTERED_MODEL, str(version), "promoted_at",
                                 datetime.now(UTC).isoformat(timespec="seconds"))
    return f"PROMOTED: v{version} is champion" + (f" (v{champ_v} -> previous)" if champ_v else "")


def rollback() -> str:
    client = _client()
    champ_v, prev_v = _alias_version(client, "champion"), _alias_version(client, "previous")
    if prev_v is None:
        return "nothing to roll back to (no 'previous' alias)"
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", str(prev_v))
    if champ_v is not None:
        client.set_registered_model_alias(REGISTERED_MODEL, "previous", str(champ_v))
        client.set_model_version_tag(REGISTERED_MODEL, str(champ_v), "rolled_back_at",
                                     datetime.now(UTC).isoformat(timespec="seconds"))
    return f"ROLLED BACK: v{prev_v} is champion again" + (f" (v{champ_v} -> previous)" if champ_v else "")
