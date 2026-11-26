import json
import warnings

import numpy as np
import pytest
from affine import Affine

from sarflood.change import OUT_BANDS
from sarflood.features import FEATURES, VH_FILL_DB, feature_stack, load_aoi
from sarflood.labels import LABEL_BANDS
from sarflood.pixels import leave_one_aoi_out, metrics

H, W = 60, 80
T = Affine(20.0, 0, 421000.0, 0, -20.0, 3045000.0)


def synth(seed: int):
    """Three classes: dry land, open flood water (dark VV), flooded vegetation
    (bright VV, but VH - VV much higher than dry land). Blocks keep context realistic."""
    rng = np.random.default_rng(seed)
    cls = np.zeros((H, W), int)
    cls[5:30, 5:40] = 1                    # open water
    cls[35:55, 45:75] = 2                  # flooded vegetation
    vv = np.choose(cls, [-9.0, -20.0, -7.0]) + rng.normal(0, 1.0, (H, W))
    vh = np.choose(cls, [-17.0, -27.0, -10.0]) + rng.normal(0, 1.0, (H, W))
    ref_vv, ref_vh = np.full((H, W), -9.0), np.full((H, W), -17.0)
    bands = {"flood_vv_db": vv, "flood_vh_db": vh, "ref_vv_db": ref_vv, "ref_vh_db": ref_vh,
             "dvv_db": vv - ref_vv, "dvh_db": vh - ref_vh, "valid": np.ones((H, W)),
             "valid_vh": np.ones((H, W)), "candidate": np.zeros((H, W))}
    flood = (cls > 0).astype(np.float32)
    labels = {"flood_frac": flood, "flood": flood, "aoi": np.ones((H, W)),
              "not_analysed": np.zeros((H, W)), "perm_water": np.zeros((H, W))}
    return bands, labels


def write(path, names, arrays):
    import rasterio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=len(names),
                           width=W, height=H, crs="EPSG:32642", transform=T) as dst:
            for i, n in enumerate(names, 1):
                dst.write(np.asarray(arrays[n], np.float32), i)
                dst.set_band_description(i, n)


def make_aoi(tmp_path, name, seed):
    bands, labels = synth(seed)
    c, lab = tmp_path / f"{name}_change.tif", tmp_path / f"{name}_labels.tif"
    write(c, OUT_BANDS, bands)
    write(lab, LABEL_BANDS, labels)
    return load_aoi(name, c, lab)


def test_metrics():
    m = metrics(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["iou"] == pytest.approx(1 / 3)


def test_feature_stack_vh_fill_and_context():
    bands, _ = synth(0)
    bands["valid_vh"][0, 0] = 0
    bands["flood_vh_db"][0, 0] = np.nan
    f = feature_stack(bands)
    assert f.shape == (len(FEATURES), H, W)
    vh, flag = f[FEATURES.index("vh")], f[FEATURES.index("vh_valid")]
    assert vh[0, 0] == VH_FILL_DB and flag[0, 0] == 0 and flag[1, 1] == 1
    # 7x7 mean deep inside the open-water block is close to -20 dB
    assert f[FEATURES.index("vv_mean7")][15, 20] == pytest.approx(-20, abs=1.0)


def test_load_aoi_region_and_grid_check(tmp_path):
    d = make_aoi(tmp_path, "a", 0)
    assert d.X.shape[1] == len(FEATURES)
    assert d.X.shape[0] == d.y.size == d.idx.size <= H * W
    assert 0.2 < d.y.mean() < 0.5


def test_loao_model_beats_rule_a_on_flooded_vegetation(tmp_path):
    data = [make_aoi(tmp_path, n, s) for n, s in [("a", 1), ("b", 2), ("c", 3)]]
    res = leave_one_aoi_out(data, n_train=5000, seed=0)
    for name, fold in res["folds"].items():
        a = fold["rule_A"]
        hgb = fold["models"]["hgb"]["metrics"]
        assert {"hgb", "hgb_bal", "logreg", "logreg_bal"} <= set(fold["models"])
        assert fold["models"]["hgb_bal"]["train_flood_share"] == pytest.approx(0.5, abs=0.01)
        assert fold["models"]["hgb"]["auc"] > a["auc"]        # VH - VV ranks the vegetation
        assert 0 <= a["best_iou"] <= 1 and np.isfinite(a["best_cut"])
        assert a["best_cut"] < 0          # reported as a VV threshold in dB
        assert {"vv", "hgb", "hgb_bal", "logreg", "logreg_bal"} <= set(fold["fewshot"])
        # Rule A misses the bright flooded vegetation; the model catches it from VH - VV.
        assert a["recall"] < 0.7, name
        assert hgb["iou"] > 0.95 and hgb["iou"] > a["iou"] + 0.2, name
        assert fold["train_aois"] == sorted({"a", "b", "c"} - {name})
        imp = fold["models"]["hgb"]["perm_importance_iou"]
        assert set(imp) == set(FEATURES)
        assert max(imp, key=imp.get) in {"vh", "vh_minus_vv", "vv", "vv_mean7", "dvv",
                                         "dvv_mean7", "dvh"}
        assert "std_coefficients" in fold["models"]["logreg"]


def test_cli_train_pixels(tmp_path, monkeypatch, capsys):
    from sarflood import cli

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    stem = "F__R_20m_s3"
    for name, seed in [("larkana", 1), ("shikarpur", 2)]:
        bands, labels = synth(seed)
        (tmp_path / "change" / name).mkdir(parents=True)
        (tmp_path / "eval" / name).mkdir(parents=True)
        write(tmp_path / "change" / name / f"{stem}_change.tif", OUT_BANDS, bands)
        write(tmp_path / "eval" / name / f"{stem}_EMSR999_AOI01_DEL_v1_vector_labels.tif",
              LABEL_BANDS, labels)
        pf = tmp_path / "pairs" / name / "pakistan-2022.json"
        pf.parent.mkdir(parents=True)
        pf.write_text(json.dumps({"pairs": [{"flood": {"key": "F"}, "reference": {"key": "R"},
                                             "reason": "ok"}]}))
    rc = cli.main(["train-pixels", "--aoi", "larkana", "--aoi", "shikarpur", "--event",
                   "pakistan-2022", "--flood-pass", "F", "--samples", "3000"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rule A" in out and "hgb" in out and "top features" in out
    rep = json.loads((tmp_path / "ml" / "pixels" / "F_s3_loao.json").read_text())
    assert set(rep["folds"]) == {"larkana", "shikarpur"}
    assert "_prob" not in json.dumps(rep)
    assert len(list((tmp_path / "ml" / "pixels").glob("*_hgb_confusion.png"))) == 2
    assert len(list((tmp_path / "ml" / "pixels").glob("*_hgb_bal_confusion.png"))) == 2
    assert "bestIoU" in out


def test_sample_pure_balanced():
    from sarflood.features import AoiData
    from sarflood.pixels import sample_pure

    n = 1000
    y = np.zeros(n, bool)
    y[:100] = True                                   # 10% flood
    frac = y.astype(np.float32)
    frac[100:150] = 0.5                              # mixed: never sampled
    d = AoiData("x", np.arange(n, dtype=np.float32)[:, None], y, frac, np.arange(n), (1, n),
                {}, np.ones(n, bool), 400.0)
    rng = np.random.default_rng(0)
    X, yy = sample_pure(d, 120, rng, balanced=True)
    assert yy.sum() == 60 and (~yy).sum() == 60
    assert not np.isin(X[:, 0], np.arange(100, 150)).any()
    X, yy = sample_pure(d, 400, rng, balanced=True)  # only 100 flood exist -> 100 + 100
    assert yy.sum() == 100 and (~yy).sum() == 100


def test_threshold_free_perfect_and_reversed():
    from sarflood.pixels import threshold_free

    truth = np.array([1, 1, 0, 0], bool)
    good = threshold_free(np.array([0.9, 0.8, 0.2, 0.1]), truth)
    assert good["auc"] == 1.0 and good["best_iou"] == 1.0
    bad = threshold_free(np.array([0.1, 0.2, 0.8, 0.9]), truth)
    assert bad["auc"] == 0.0


def test_best_cut_exact():
    from sarflood.pixels import best_cut_exact

    x = np.array([-22.0, -20.0, -15.0, -8.0, -6.0])
    y = np.array([1, 1, 1, 0, 0], bool)
    assert best_cut_exact(x, y) == -15.0
    assert best_cut_exact(x, np.zeros(5, bool)) == float("-inf")
    assert best_cut_exact(x, np.ones(5, bool)) == float("inf")
    ties = np.array([-10.0, -10.0, -5.0])          # a cut must not split tied values
    assert best_cut_exact(ties, np.array([1, 0, 0], bool)) == -10.0


def test_fewshot_approaches_ceiling():
    from sarflood.pixels import fewshot_iou

    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.random(n) < 0.5
    x = np.where(y, rng.normal(-18, 2, n), rng.normal(-9, 2, n))   # separable-ish
    fs = fewshot_iou(x, y, rng, ns=(25, 1000), reps=20)
    ceil = max((y & (x <= t)).sum() / (y | (x <= t)).sum() for t in np.linspace(-16, -11, 101))
    assert fs["1000"]["mean"] > ceil - 0.01
    assert fs["25"]["mean"] <= fs["1000"]["mean"] + 1e-9
    assert fs["25"]["p10"] <= fs["25"]["mean"] <= fs["25"]["p90"]
