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
