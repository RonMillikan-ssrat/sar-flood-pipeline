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
