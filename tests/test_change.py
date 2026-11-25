import warnings

import numpy as np
import pytest

from sarflood.change import box_mean, compute_change, overlap, quicklook, run_change, to_db

RES = 20.0


def write_tc(path, vh, vv, mask, x0, y0, crs="EPSG:32642"):
    """Write a 4-band raster shaped like SNAP's TC output (VH, VV, LIA, mask)."""
    import rasterio
    from rasterio.transform import from_origin

    h, w = vv.shape
    lia = np.full((h, w), 38.5, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path, "w", driver="GTiff", dtype="float32", count=4, width=w,
                           height=h, crs=crs, transform=from_origin(x0, y0, RES, RES)) as dst:
            for i, a in enumerate([vh, vv, lia, mask.astype(np.float32)], 1):
                dst.write(a.astype(np.float32), i)
    return path


def db(x):
    return 10 ** (x / 10)


def test_to_db():
    out = to_db(np.array([1.0, 0.1, 0.0, -1e-3, np.nan], dtype=np.float32))
    np.testing.assert_allclose(out[:2], [0.0, -10.0], atol=1e-6)
    assert np.isnan(out[2:]).all()


def test_compute_change_values_and_masks():
    ones = np.ones((2, 3), np.float32)
    ref = {"vv": db(-10) * ones, "vh": db(-17) * ones, "mask": np.zeros((2, 3))}
    flood = {"vv": db(-10) * ones, "vh": db(-17) * ones, "mask": np.zeros((2, 3))}
    flood["vv"][0, 0] = db(-22)      # flooded: -12 dB change, below -18 dB
    flood["vv"][0, 1] = db(-14)      # darker by 4 dB but not water-dark -> not candidate
    flood["vh"][0, 2] = -1e-4        # VH negative (unclipped): VH invalid, VV still valid
    flood["vv"][1, 2] = -1e-4        # VV negative: VV invalid
    ref["mask"][1, 0] = 2            # shadow in the reference -> invalid in both
    bands, st = compute_change(flood, ref, pixel_area_m2=400.0)
    assert bands["dvv_db"][0, 0] == pytest.approx(-12.0, abs=1e-4)
    assert bands["dvv_db"][0, 1] == pytest.approx(-4.0, abs=1e-4)
    assert bands["candidate"].tolist() == [[1, 0, 0], [0, 0, 0]]
    assert bands["valid"].tolist() == [[1, 1, 1], [0, 1, 0]]      # [1,0] shadow, [1,2] VV<0
    assert bands["valid_vh"].tolist() == [[1, 1, 0], [0, 1, 1]]   # [1,0] shadow, [0,2] VH<0
    assert np.isfinite(bands["dvv_db"][0, 2]) and np.isnan(bands["dvh_db"][0, 2])
    assert (st.valid, st.valid_vh, st.masked_layover_shadow) == (4, 4, 1)
    assert (st.masked_nonpositive, st.masked_nonpositive_vh) == (1, 1)
    assert st.candidate_km2 == pytest.approx(400 / 1e6)


def brute_box_mean(x, n):
    r = n // 2
    out = np.full(x.shape, np.nan)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            w = x[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
            ok = np.isfinite(w)
            if ok.sum() > n * n / 2:
                out[i, j] = w[ok].mean()
    return out


@pytest.mark.parametrize("n", [3, 5])
def test_box_mean_matches_brute_force(n):
    rng = np.random.default_rng(3)
    x = rng.normal(0.05, 0.03, (17, 23))   # includes negative values, as unclipped data does
    x[rng.random(x.shape) < 0.15] = np.nan
    np.testing.assert_allclose(box_mean(x, n), brute_box_mean(x, n), rtol=1e-5, equal_nan=True)


def test_box_mean_identity_and_even():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(box_mean(x, 1), x)
    with pytest.raises(ValueError, match="odd"):
        box_mean(x, 4)


def test_smoothing_recovers_speckled_flood():
    """Speckle splits a uniform flood across a hard threshold; averaging fixes it."""
    rng = np.random.default_rng(7)
    shape = (60, 60)
    ref_vv = db(-10) * rng.gamma(4.4, 1 / 4.4, shape)      # about 4.4 looks, like GRDH
    flood_vv = db(-20) * rng.gamma(4.4, 1 / 4.4, shape)
    vh = np.full(shape, db(-17))
    zeros = np.zeros(shape)
    f = {"vv": flood_vv, "vh": vh, "mask": zeros}
    r = {"vv": ref_vv, "vh": vh, "mask": zeros}
    raw, _ = compute_change(f, r, 400.0, change_db=-3, water_db=-18, smooth=1)
    sm, st = compute_change(f, r, 400.0, change_db=-3, water_db=-18, smooth=5)
    inner = (slice(5, -5), slice(5, -5))
    assert raw["candidate"][inner].mean() < 0.9
    assert sm["candidate"][inner].mean() > 0.99
    assert st.smooth == 5


def test_smoothing_skips_layover_shadow():
    vv = np.full((5, 5), db(-10), np.float32)
    vv[2, 2] = 1e6                      # absurd value in a shadow pixel
    mask = np.zeros((5, 5))
    mask[2, 2] = 2
    d = {"vv": vv, "vh": vv.copy(), "mask": mask}
    bands, _ = compute_change(d, d, 400.0, smooth=3)
    assert bands["ref_vv_db"][1, 1] == pytest.approx(-10.0, abs=1e-4)   # not polluted
    assert np.isnan(bands["ref_vv_db"][2, 2])                          # still masked


def test_overlap_with_whole_pixel_offset(tmp_path):
    import rasterio

    # Reference starts 2 px east and 1 px south of the flood raster.
    f = write_tc(tmp_path / "f.tif", np.ones((10, 12)), np.ones((10, 12)), np.zeros((10, 12)),
                 349870.0, 3042730.0)
    r = write_tc(tmp_path / "r.tif", np.ones((10, 12)), np.ones((10, 12)), np.zeros((10, 12)),
                 349870.0 + 2 * RES, 3042730.0 - 1 * RES)
    with rasterio.open(f) as fs, rasterio.open(r) as rs:
        ov = overlap(fs, rs)
    assert ov.flood == (1, 2, 9, 10)   # row0, col0, height, width in the flood raster
    assert ov.ref == (0, 0, 9, 10)
    assert (ov.transform.c, ov.transform.f) == (349870.0 + 40, 3042730.0 - 20)


def test_overlap_rejects_half_pixel_offset(tmp_path):
    import rasterio

    a = np.ones((4, 4))
    f = write_tc(tmp_path / "f.tif", a, a, np.zeros((4, 4)), 349870.0, 3042730.0)
    r = write_tc(tmp_path / "r.tif", a, a, np.zeros((4, 4)), 349880.0, 3042730.0)  # +10 m
    with rasterio.open(f) as fs, rasterio.open(r) as rs, pytest.raises(ValueError, match="aligned"):
        overlap(fs, rs)


def test_overlap_rejects_crs_mismatch(tmp_path):
    import rasterio

    a = np.ones((4, 4))
    f = write_tc(tmp_path / "f.tif", a, a, np.zeros((4, 4)), 0.0, 0.0)
    r = write_tc(tmp_path / "r.tif", a, a, np.zeros((4, 4)), 0.0, 0.0, crs="EPSG:32615")
    with rasterio.open(f) as fs, rasterio.open(r) as rs, pytest.raises(ValueError, match="CRS"):
        overlap(fs, rs)


def test_run_change_pairs_the_right_pixels(tmp_path):
    import rasterio

    # Make each pixel's VV encode its map position, so a misregistration would show.
    h, w = 10, 12
    rows, cols = np.mgrid[0:h, 0:w]
    fvv = db(-10.0 - 0.1 * (rows * w + cols))
    f = write_tc(tmp_path / "f.tif", np.full((h, w), db(-17)), fvv, np.zeros((h, w)),
                 349870.0, 3042730.0)
    # Reference offset by (+2 cols, +1 row); same map positions -> same value -> zero change.
    rvv = db(-10.0 - 0.1 * ((rows + 1) * w + (cols + 2)))
    r = write_tc(tmp_path / "r.tif", np.full((h, w), db(-17)), rvv, np.zeros((h, w)),
                 349870.0 + 40, 3042730.0 - 20)
    out = tmp_path / "out" / "change.tif"
    st = run_change(f, r, out)
    with rasterio.open(out) as src:
        names = list(src.descriptions)
        d = src.read(names.index("dvv_db") + 1)
        assert src.shape == (9, 10)
    np.testing.assert_allclose(d, 0.0, atol=1e-4)
    assert st.valid == 90 and st.candidate_pixels == 0

    png = quicklook(out, tmp_path / "out" / "ql.png")
    assert png.exists() and png.stat().st_size > 1000


def test_cli_change_end_to_end(tmp_path, monkeypatch, capsys):
    import json

    from sarflood import cli

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    tc = tmp_path / "tc" / "sindh"
    tc.mkdir(parents=True)
    h, w = 8, 8
    ref_vv = np.full((h, w), db(-10))
    flood_vv = ref_vv.copy()
    flood_vv[:4, :] = db(-22)  # top half floods
    vh = np.full((h, w), db(-17))
    write_tc(tc / "S1A_044775_0558C6_tc_20m_noclip_epsg32642.tif", vh, flood_vv,
             np.zeros((h, w)), 349870.0, 3042730.0)
    write_tc(tc / "S1A_043375_052E02_tc_20m_noclip_epsg32642.tif", vh, ref_vv,
             np.zeros((h, w)), 349870.0, 3042730.0)
    pf = tmp_path / "pairs" / "sindh" / "pakistan-2022.json"
    pf.parent.mkdir(parents=True)
    pf.write_text(json.dumps({"pairs": [{
        "flood": {"key": "S1A_044775_0558C6", "slices": [], "ids": []},
        "reference": {"key": "S1A_043375_052E02", "slices": [], "ids": []}, "reason": "ok"}]}))
    rc = cli.main(["change", "--aoi", "sindh", "--event", "pakistan-2022",
                   "--flood-pass", "S1A_044775_0558C6"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "32 px = 0.0 km2" in out  # 32 px x 400 m2 = 0.0128 km2
    stats = json.loads((tmp_path / "change" / "sindh" /
                        "S1A_044775_0558C6__S1A_043375_052E02_20m_stats.json").read_text())
    assert stats["candidate_pixels"] == 32 and stats["valid"] == 64
