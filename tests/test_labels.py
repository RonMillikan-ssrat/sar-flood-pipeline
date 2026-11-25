import json
import warnings
import zipfile

import numpy as np
import pytest
from affine import Affine

from sarflood.labels import _burn, rasterize_ems, score

CRS = "EPSG:32642"
X0, Y0, RES = 421000.0, 3045000.0, 20.0
H, W = 50, 60
T = Affine(RES, 0, X0, 0, -RES, Y0)


def utm_box(c0, r0, c1, r1):
    """Polygon in UTM covering pixel cols [c0, c1) and rows [r0, r1)."""
    x0, x1 = X0 + c0 * RES, X0 + c1 * RES
    y0, y1 = Y0 - r1 * RES, Y0 - r0 * RES
    return {"type": "Polygon", "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]}


def to_lonlat(g):
    from rasterio.warp import transform_geom

    return transform_geom(CRS, "EPSG:4326", g)


def fc(features):
    return json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": p, "geometry": to_lonlat(g)} for g, p in features]})


def make_zip(tmp_path):
    p = "EMSR999_AOI01_DEL_PRODUCT_{}_r1_v1.json"
    z = tmp_path / "EMSR999_AOI01_DEL_PRODUCT_r1_RTP01_v1_vector.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(p.format("areaOfInterestA"), fc([(utm_box(5, 5, 55, 45), {"locality": "X"})]))
        zf.writestr(p.format("observedEventA"), fc([
            (utm_box(10, 10, 30, 30), {"notation": "Flooded area"}),
            (utm_box(40, 10, 41, 11), {"notation": "Flooded area"}),     # one exact pixel
        ]))
        zf.writestr(p.format("imageFootprintA"), fc([
            (utm_box(0, 0, W, H), {"obj_type": "Image Footprint"}),
            (utm_box(45, 35, 50, 40), {"obj_type": "Not Analysed"}),
        ]))
        zf.writestr(p.format("hydrographyA"), fc([
            (utm_box(32, 5, 36, 45), {"obj_type": "BH140-River"}),
            (utm_box(5, 40, 55, 45), {"obj_type": "BH090-Land Subject to Inundation"}),
        ]))
    return z


def test_burn_fraction_half_pixel():
    # Polygon covering the left half of pixel (0, 0) -> fraction 0.5 with even supersampling.
    g = {"type": "Polygon", "coordinates": [[(X0, Y0 - RES), (X0 + RES / 2, Y0 - RES),
                                             (X0 + RES / 2, Y0), (X0, Y0), (X0, Y0 - RES)]]}
    f = _burn([g], (2, 2), T, supersample=4)
    assert f[0, 0] == pytest.approx(0.5)
    assert f[0, 1] == 0 and f[1, 0] == 0


def test_rasterize_ems_layers(tmp_path):
    lab = rasterize_ems(make_zip(tmp_path), (H, W), T, CRS)
    # Interior pixels: away from edges, so the lon/lat round trip can't matter.
    assert lab["flood"][15:25, 15:25].all() and not lab["flood"][35:40, 15:25].any()
    assert lab["flood"][10, 40] == 1 and lab["flood"][10, 41] == 0
    assert lab["aoi"][20, 20] == 1 and lab["aoi"][2, 2] == 0
    assert lab["not_analysed"][37, 47] == 1 and lab["not_analysed"][20, 20] == 0
    assert lab["perm_water"][20, 34] == 1
    # Land Subject to Inundation is floodable land, not permanent water.
    assert lab["perm_water"][42, 20] == 0


def test_score_counts_and_exclusions(tmp_path):
    lab = rasterize_ems(make_zip(tmp_path), (H, W), T, CRS)
    valid = np.ones((H, W), bool)
    valid[12, 12] = False                      # e.g. radar shadow inside the flood
    pred = lab["flood"] > 0                    # perfect prediction...
    pred = pred.copy()
    pred[20, 34] = True                        # ...plus permanent water (not scored)
    pred[37, 47] = True                        # ...plus not-analysed area (not scored)
    pred[42, 50] = True                        # ...plus one real false alarm
    s = score(pred, lab, valid, RES * RES)
    flood_px = int(lab["flood"].sum()) - 1     # minus the invalid pixel
    assert (s.tp, s.fp, s.fn) == (flood_px, 1, 0)
    assert s.precision == pytest.approx(flood_px / (flood_px + 1))
    assert s.recall == 1.0
    assert s.iou == pytest.approx(flood_px / (flood_px + 1))


def test_mixed_pixels_excluded():
    frac = np.array([[0.0, 0.3, 0.6, 1.0]], np.float32)
    lab = {"flood_frac": frac, "flood": (frac >= 0.5).astype(np.float32),
           "aoi": np.ones_like(frac), "not_analysed": np.zeros_like(frac),
           "perm_water": np.zeros_like(frac)}
    pred = np.array([[False, True, False, True]])
    valid = np.ones_like(pred)
    assert score(pred, lab, valid, 400).fp == 1
    s = score(pred, lab, valid, 400, mixed_lo=0.2, mixed_hi=0.8)
    assert (s.tp, s.fp, s.fn, s.tn, s.excluded_mixed) == (1, 0, 0, 1, 2)


def test_cli_evaluate_end_to_end(tmp_path, monkeypatch, capsys):
    import rasterio

    from sarflood import cli
    from sarflood.change import OUT_BANDS

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    z = make_zip(tmp_path)
    lab = rasterize_ems(z, (H, W), T, CRS)
    flood = lab["flood"] > 0
    dvv = np.where(flood, -10.0, 0.0).astype(np.float32)
    fvv = np.where(flood, -22.0, -10.0).astype(np.float32)
    bands = {"ref_vv_db": np.full((H, W), -10.0), "flood_vv_db": fvv, "dvv_db": dvv,
             "ref_vh_db": np.full((H, W), -17.0), "flood_vh_db": np.full((H, W), -17.0),
             "dvh_db": np.zeros((H, W)), "valid": np.ones((H, W)),
             "candidate": ((dvv <= -3) & (fvv <= -18)).astype(float)}
    cdir = tmp_path / "change" / "larkana"
    cdir.mkdir(parents=True)
    ctif = cdir / "F_KEY__R_KEY_20m_change.tif"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(ctif, "w", driver="GTiff", dtype="float32", count=len(OUT_BANDS),
                           width=W, height=H, crs=CRS, transform=T) as dst:
            for i, n in enumerate(OUT_BANDS, 1):
                dst.write(np.asarray(bands[n], np.float32), i)
                dst.set_band_description(i, n)
            dst.update_tags(change_db="-3.0", water_db="-18.0")
    pf = tmp_path / "pairs" / "larkana" / "pakistan-2022.json"
    pf.parent.mkdir(parents=True)
    pf.write_text(json.dumps({"pairs": [{"flood": {"key": "F_KEY"}, "reference": {"key": "R_KEY"},
                                         "reason": "ok"}]}))
    rc = cli.main(["evaluate", "--aoi", "larkana", "--event", "pakistan-2022",
                   "--flood-pass", "F_KEY", "--ems-zip", str(z), "--sweep"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "IoU 1.000" in out
    rep = json.loads(next((tmp_path / "eval" / "larkana").glob("*_score.json")).read_text())
    assert rep["all"]["iou"] == 1.0 and len(rep["sweep"]) == 30
    assert next((tmp_path / "eval" / "larkana").glob("*_confusion.png")).stat().st_size > 1000
