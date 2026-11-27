from pathlib import Path

import numpy as np
import pytest

from sarflood.lakehouse import (
    RasterPair,
    build_manifest,
    find_pairs,
    pair_dir,
    same_content,
    volume_root,
)

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

TRANSFORM = from_origin(600000, 2900000, 20, 20)


def write_tif(path: Path, bands: list[str], transform=TRANSFORM, shape=(4, 5), tags=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=len(bands), dtype="float32", crs="EPSG:32642",
                       transform=transform) as dst:
        for i, name in enumerate(bands, 1):
            dst.write(np.full(shape, i, np.float32), i)
            dst.set_band_description(i, name)
        if tags:
            dst.update_tags(**tags)


@pytest.fixture
def root(tmp_path):
    # good pair
    write_tif(tmp_path / "change/sindh/s1_A_change.tif", ["flood_vv_db", "valid"])
    write_tif(tmp_path / "eval/sindh/s1_A_EMSR629_labels.tif", ["flood", "aoi"],
              tags={"labels_version": "2"})
    # change map with no labels
    write_tif(tmp_path / "change/jamuna/j_B_change.tif", ["flood_vv_db"])
    # change map with two labels files
    write_tif(tmp_path / "change/khipro/k_C_change.tif", ["flood_vv_db"])
    write_tif(tmp_path / "eval/khipro/k_C_EMSR1_labels.tif", ["flood"])
    write_tif(tmp_path / "eval/khipro/k_C_EMSR2_labels.tif", ["flood"])
    return tmp_path


def test_find_pairs_reports_missing_and_ambiguous_labels(root):
    pairs, problems = find_pairs(root)
    assert [(p.aoi, p.stem) for p in pairs] == [("sindh", "s1_A")]
    assert pairs[0].labels_tif.name == "s1_A_EMSR629_labels.tif"
    assert len(problems) == 2
    assert any("jamuna" in m and "0 labels" in m for m in problems)
    assert any("khipro" in m and "2 labels" in m for m in problems)


def test_find_pairs_aoi_filter(root):
    pairs, problems = find_pairs(root, aois=["jamuna"])
    assert pairs == [] and len(problems) == 1


def test_manifest_contents(root):
    (pair,), _ = find_pairs(root)
    m = build_manifest(pair)
    assert m["aoi"] == "sindh" and m["stem"] == "s1_A"
    assert m["labels_version"] == "2"
    assert m["grid_match"] is True
    assert m["files"]["change"]["bands"] == ["flood_vv_db", "valid"]
    assert m["files"]["change"]["crs"] == "EPSG:32642"
    assert len(m["files"]["labels"]["sha256"]) == 64


def test_grid_mismatch_flagged(tmp_path):
    c = tmp_path / "c_change.tif"
    lab = tmp_path / "c_EMSR1_labels.tif"
    write_tif(c, ["flood_vv_db"])
    write_tif(lab, ["flood"], transform=from_origin(600010, 2900000, 20, 20))
    m = build_manifest(RasterPair("x", "c", c, lab))
    assert m["grid_match"] is False
    assert m["labels_version"] == "1"  # untagged labels default to version 1


def test_same_content(root):
    (pair,), _ = find_pairs(root)
    m = build_manifest(pair)
    assert same_content(m, m)
    assert not same_content(m, None)
    other = {"files": {"change": m["files"]["change"], "labels": {"sha256": "0" * 64}}}
    assert not same_content(m, other)


def test_paths():
    vol = volume_root("workspace", "sarflood", "raw")
    assert vol == "/Volumes/workspace/sarflood/raw"
    p = RasterPair("sindh", "s1_A", Path("a"), Path("b"))
    assert pair_dir(vol + "/", p) == "/Volumes/workspace/sarflood/raw/pairs/sindh/s1_A"


# ---------------------------------------------------------------- Phase 1 helpers

from sarflood.lakehouse import (
    aoi_to_frame,
    bronze_row,
    flood_pass_of,
    frame_to_aoi,
    psi,
    psi_edges,
    psi_level,
    silver_columns,
)


def test_bronze_row_ok_and_quarantine(root, tmp_path):
    (pair,), _ = find_pairs(root)
    m = build_manifest(pair)
    ok = bronze_row(m, "/Volumes/x/manifest.json", {"change": True, "labels": True})
    assert ok["status"] == "ok" and ok["reason"] == "" and ok["labels_version"] == "2"
    assert ok["flood_pass"] == "s1_A" and (ok["height"], ok["width"]) == (4, 5)
    bad = bronze_row({**m, "grid_match": False}, "p", {"change": True, "labels": False})
    assert bad["status"] == "quarantine"
    assert "grids differ" in bad["reason"] and "labels file missing" in bad["reason"]
    assert flood_pass_of("S1A_044775_0558C6__S1A_043375_052E02_20m_s3") == "S1A_044775_0558C6"


def test_silver_frame_roundtrip_restores_load_order(tmp_path):
    from tests.test_pixels import make_aoi

    d = make_aoi(tmp_path, "larkana", 1)
    d.labels_version = "3"
    df = aoi_to_frame(d, "S_P__S_R_20m_s3", "c" * 64, "l" * 64)
    assert list(df.columns) == silver_columns()
    assert df["vv"].dtype == np.float32 and df["idx"].dtype == np.int64 and df["y"].dtype == bool
    back = frame_to_aoi(df.sample(frac=1.0, random_state=0))      # shuffled, as from Spark
    assert back.name == "larkana" and back.labels_version == "3"
    np.testing.assert_array_equal(back.idx, d.idx)
    np.testing.assert_array_equal(back.X, d.X)
    np.testing.assert_array_equal(back.y, d.y)
    np.testing.assert_array_equal(back.frac, d.frac.astype(np.float32))


def test_silver_roundtrip_gives_identical_loao(tmp_path):
    """Training from silver must be the same computation as training from GeoTIFFs."""
    from sarflood.pixels import leave_one_aoi_out
    from tests.test_pixels import make_aoi

    data = [make_aoi(tmp_path, n, s) for n, s in (("a", 1), ("b", 2))]
    via = [frame_to_aoi(aoi_to_frame(d, "s", "c", "l").iloc[::-1]) for d in data]
    r1 = leave_one_aoi_out(data, n_train=3000, importance=False)
    r2 = leave_one_aoi_out(via, n_train=3000, importance=False)
    for aoi in ("a", "b"):
        m1, m2 = r1["folds"][aoi]["models"], r2["folds"][aoi]["models"]
        assert {k: v["metrics"]["iou"] for k, v in m1.items()} == \
               {k: v["metrics"]["iou"] for k, v in m2.items()}


def test_frame_to_aoi_rejects_mixed():
    import pandas as pd

    df = pd.DataFrame({"aoi": ["a", "b"], "stem": ["s", "s"], "labels_version": ["3", "3"],
                       "idx": [0, 1]})
    with pytest.raises(ValueError, match="exactly one"):
        frame_to_aoi(df)


def test_psi():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 50_000)
    edges = psi_edges(ref)
    assert len(edges) == 9
    same = psi(ref, rng.normal(0, 1, 50_000), edges)
    shifted = psi(ref, rng.normal(1.0, 1, 50_000), edges)
    assert same < 0.01 and shifted > 0.25
    assert (psi_level(same), psi_level(0.15), psi_level(shifted)) == ("stable", "moderate", "major")
    flags = np.r_[np.ones(900), np.zeros(100)]                   # discrete feature
    assert len(psi_edges(flags)) <= 2 and psi(flags, flags, psi_edges(flags)) == 0.0
