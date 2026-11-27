import json

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")

from sarflood import tracking

PIX = {"n_train_per_aoi": 1000, "folds": {
    a: {"rule_A": {"iou": iou, "precision": .9, "recall": .8, "f1": .85, "auc": .9, "ap": .9,
                   "best_iou": iou + .01},
        "fewshot": {"vv": {"100": {"mean": iou + .005}, "300": {"mean": iou + .008}},
                    "hgb_bal": {"100": {"mean": iou + .02}, "300": {"mean": iou + .03}}},
        "models": {"hgb_bal": {"metrics": {"iou": iou - .01, "precision": .9, "recall": .7,
                                           "f1": .8}, "auc": .92, "ap": .91, "best_iou": iou + .04,
                               "perm_importance_iou": {"vv": .1}}}}
    for a, iou in [("larkana", .78), ("shikarpur", .73)]}}
UNET = {"config": {"tile": 256, "epochs": 20}, "folds": {
    "larkana": {"rule_A": {"iou": .78, "auc": .91}, "unet": {"iou": .76, "auc": .90},
                "fewshot": {"vv": {"300": {"mean": .776}}, "unet": {"300": {"mean": .770}}}}}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return tmp_path


def test_methods_from_report_pixels_and_unet():
    kind, m = tracking.methods_from_report(PIX)
    assert kind == "pixels" and set(m) == {"rule_A", "hgb_bal"}
    assert m["hgb_bal"]["shikarpur"]["iou_n300"] == pytest.approx(.76)
    assert "perm_importance_iou" not in m["hgb_bal"]["larkana"]
    kind, m = tracking.methods_from_report(UNET)
    assert kind == "unet" and set(m) == {"rule_A", "unet"}


def test_log_report_runs_and_dedup(store):
    rep = store / "ml" / "pixels" / "X_s3_loao.json"
    rep.parent.mkdir(parents=True)
    rep.write_text(json.dumps(PIX))
    (rep.parent / "X_s3_larkana_hgb_confusion.png").write_bytes(b"\x89PNG fake")
    run = tracking.log_report(rep, tags={"note": "t"})
    assert run is not None
    assert tracking.log_report(rep) is None                     # same SHA: skipped
    assert tracking.log_report(rep, force=True) is not None
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT], output_format="list",
                              filter_string=f"tags.mlflow.parentRunId = '{run}'")
    child = next(r for r in runs if r.info.run_name == "hgb_bal")
    assert child.data.metrics["mean.iou_n300"] == pytest.approx((.81 + .76) / 2)
    assert child.data.tags["note"] == "t"
    arts = [a.path for a in mlflow.MlflowClient().list_artifacts(run, "maps")]
    assert arts == ["maps/X_s3_larkana_hgb_confusion.png"]
    assert tracking.gate_value_from_report(rep, "hgb_bal") == pytest.approx((.81 + .76) / 2)
    with pytest.raises(ValueError, match="not in"):
        tracking.gate_value_from_report(rep, "unet")


def test_fit_register_gate_promote_rollback(store, tmp_path):
    from tests.test_pixels import make_aoi

    data = [make_aoi(tmp_path, n, s) for n, s in [("a", 1), ("b", 2)]]
    v1, _ = tracking.fit_final(data, "hgb_bal", 2000, 0, 0.80, "r1.json", {"event": "e"})
    v2, _ = tracking.fit_final(data, "logreg", 2000, 0, 0.75, "r2.json", {"event": "e"})
    assert (v1, v2) == (1, 2)
    assert tracking.promote(v1).startswith("PROMOTED")
    assert tracking.promote(v2).startswith("REJECTED")            # 0.75 < 0.80
    assert tracking.promote(v2, force=True).startswith("PROMOTED")
    rows = {r["version"]: r for r in tracking.registry_table()}
    assert rows[2]["aliases"] == ["champion"] and rows[1]["aliases"] == ["previous"]
    assert tracking.rollback().startswith("ROLLED BACK: v1")
    rows = {r["version"]: r for r in tracking.registry_table()}
    assert rows[1]["aliases"] == ["champion"] and rows[2]["aliases"] == ["previous"]
    model = mlflow.sklearn.load_model(f"models:/{tracking.REGISTERED_MODEL}@champion")
    assert type(model).__name__ == "HistGradientBoostingClassifier"    # v1, after rollback
    p = model.predict_proba(data[0].X[:10])[:, 1]
    assert p.shape == (10,) and np.all((p >= 0) & (p <= 1))
    with pytest.raises(ValueError, match="unknown method"):
        tracking.fit_final(data, "svm", 100, 0, .5, "x", {})


def test_promote_min_delta(store, tmp_path):
    from tests.test_pixels import make_aoi

    data = [make_aoi(tmp_path, "a", 1)]
    v1, _ = tracking.fit_final(data, "logreg", 1000, 0, 0.80, "r", {})
    v2, _ = tracking.fit_final(data, "logreg", 1000, 1, 0.805, "r", {})
    tracking.promote(v1)
    assert tracking.promote(v2, min_delta=0.01).startswith("REJECTED")   # +0.005 < 0.01
    assert tracking.promote(v2, min_delta=0.001).startswith("PROMOTED")


def test_rollback_without_previous(store):
    assert "nothing to roll back" in tracking.rollback()


def test_cli_registry_and_mlflow_log(store, capsys):
    from sarflood import cli

    rep = store / "r_loao.json"
    rep.write_text(json.dumps(UNET))
    assert cli.main(["mlflow-log", str(rep), "--tag", "machine=desktop"]) == 0
    assert cli.main(["mlflow-log", str(rep)]) == 0
    out = capsys.readouterr().out
    assert "logged run" in out and "already logged" in out
    assert cli.main(["registry", "show"]) == 0
    assert "no registered versions" in capsys.readouterr().out


def test_skops_trusted_types_only_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.random.default_rng(0).random((200, 3))
    m = HistGradientBoostingClassifier(max_iter=5).fit(X, X[:, 0] > .5)
    types = tracking.skops_trusted_types(m)
    assert types and all(t.startswith("sklearn.") for t in types)


def test_promote_refuses_across_labels_versions(store, tmp_path):
    from tests.test_pixels import make_aoi

    data = [make_aoi(tmp_path, "a", 1)]
    assert data[0].labels_version == "1"                       # untagged labels file
    v1, _ = tracking.fit_final(data, "logreg", 1000, 0, 0.70, "r", {})
    data[0].labels_version = "2"
    v2, _ = tracking.fit_final(data, "logreg", 1000, 1, 0.90, "r", {})
    tracking.promote(v1)
    with pytest.raises(ValueError, match="labels_version differs"):
        tracking.promote(v2)                                   # better gate, other labels
    from sarflood import cli

    assert cli.main(["registry", "promote", "--version", str(v2)]) == 2   # fails, not skip
    assert tracking.promote(v2, force=True).startswith("PROMOTED")
    rows = {r["version"]: r for r in tracking.registry_table()}
    assert (rows[1]["labels_version"], rows[2]["labels_version"]) == ("1", "2")
    assert tracking.combined_labels_version(["2", "1", "2"]) == "mixed:1,2"


def test_holdout_scoring_and_logging(store, tmp_path):
    from tests.test_pixels import make_aoi

    train = [make_aoi(tmp_path, n, s) for n, s in [("a", 1), ("b", 2)]]
    v, _ = tracking.fit_final(train, "hgb_bal", 2000, 0, 0.8, "r", {})
    tracking.promote(v)
    model, got, tags = tracking.load_version()
    assert got == v and tags["train_aois"] == "a,b"
    test = make_aoi(tmp_path, "c", 3)
    vv = test.X[:, 0]
    res = tracking.holdout_metrics(model.predict_proba(test.X)[:, 1], test.y, vv)
    assert res["flood_share"] == pytest.approx(test.y.mean())
    assert res["model"]["auc"] > 0.95 and res["rule_A"]["best_cut"] < 0   # cut in VV dB
    assert res["model"]["iou"] > res["flood_share"]                     # skill over all-flood
    res.update(flood_pass="P1", labels_version="3")
    run = tracking.log_holdout("c", v, tags, res, {"event": "e"})
    r = mlflow.get_run(run)
    assert r.data.tags["model_version"] == str(v) and r.data.tags["labels_version"] == "3"
    assert r.data.metrics["model.auc"] == pytest.approx(res["model"]["auc"])
    assert tracking.log_holdout("c", v, tags, res, {}) is None          # same key: skipped
    assert tracking.log_holdout("c", v, tags, res, {}, force=True) is not None
    res["labels_version"] = "3-rb60"
    assert tracking.log_holdout("c", v, tags, res, {}) is not None      # new labels: logged
