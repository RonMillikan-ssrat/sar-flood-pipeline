import warnings
import zipfile

import numpy as np
import pytest

from sarflood.calibrate import (
    AzimuthBlock,
    SparseGrid,
    azimuth_noise,
    calibrate_safe,
    find_members,
    parse_calibration,
    parse_noise,
    sigma0_block,
)

H, W = 40, 60  # tiny synthetic image


def cal_xml(lines, pixels, sigma, const=1.0) -> bytes:
    vecs = "".join(
        f"<calibrationVector><line>{ln}</line>"
        f"<pixel count='{len(pixels)}'>{' '.join(map(str, pixels))}</pixel>"
        f"<sigmaNought count='{len(pixels)}'>{' '.join(map(str, s))}</sigmaNought>"
        f"<betaNought count='{len(pixels)}'>{' '.join(['474'] * len(pixels))}</betaNought>"
        "</calibrationVector>"
        for ln, s in zip(lines, sigma)
    )
    return (f"<calibration><calibrationInformation><absoluteCalibrationConstant>{const}"
            f"</absoluteCalibrationConstant></calibrationInformation>"
            f"<calibrationVectorList count='{len(lines)}'>{vecs}</calibrationVectorList>"
            "</calibration>").encode()


def noise_xml(lines, pixels, lut, blocks) -> bytes:
    rv = "".join(
        f"<noiseRangeVector><line>{ln}</line>"
        f"<pixel count='{len(pixels)}'>{' '.join(map(str, pixels))}</pixel>"
        f"<noiseRangeLut count='{len(pixels)}'>{' '.join(map(str, v))}</noiseRangeLut>"
        "</noiseRangeVector>"
        for ln, v in zip(lines, lut)
    )
    av = "".join(
        f"<noiseAzimuthVector><swath>{b['swath']}</swath>"
        f"<firstAzimuthLine>{b['fl']}</firstAzimuthLine><firstRangeSample>{b['fs']}</firstRangeSample>"
        f"<lastAzimuthLine>{b['ll']}</lastAzimuthLine><lastRangeSample>{b['ls']}</lastRangeSample>"
        f"<line count='{len(b['lines'])}'>{' '.join(map(str, b['lines']))}</line>"
        f"<noiseAzimuthLut count='{len(b['lut'])}'>{' '.join(map(str, b['lut']))}</noiseAzimuthLut>"
        "</noiseAzimuthVector>"
        for b in blocks
    )
    return (f"<noise><noiseRangeVectorList count='{len(lines)}'>{rv}</noiseRangeVectorList>"
            f"<noiseAzimuthVectorList count='{len(blocks)}'>{av}</noiseAzimuthVectorList></noise>"
            ).encode()


def test_vectorized_sample_matches_per_column_interp():
    rng = np.random.default_rng(0)
    lines = np.array([0.0, 7.0, 19.0, 40.0])
    pixels = [np.array([0.0, 20.0, 40.0, 60.0])] * 4
    values = [rng.uniform(400, 700, 4) for _ in range(4)]
    g = SparseGrid(lines, pixels, values)
    rows = np.arange(-3, 45, dtype=float)  # includes clamped rows beyond both ends
    cols = np.arange(0, 60, dtype=float)
    got = g.sample(rows, cols)
    across = np.stack([np.interp(cols, p, v) for p, v in zip(pixels, values)])
    want = np.stack([np.interp(rows, lines, across[:, j]) for j in range(cols.size)], axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_vectors_with_different_pixel_grids():
    g = SparseGrid(np.array([0.0, 10.0]),
                   [np.array([0.0, 10.0]), np.array([0.0, 5.0, 10.0])],
                   [np.array([1.0, 3.0]), np.array([1.0, 1.0, 1.0])])
    out = g.sample(np.array([0.0, 5.0, 10.0]), np.array([10.0]))
    np.testing.assert_allclose(out[:, 0], [3.0, 2.0, 1.0])


def test_parse_calibration_and_constant():
    g = parse_calibration(cal_xml([0, 40], [0, 60], [[600, 700], [600, 700]], const=2.0))
    np.testing.assert_allclose(g.values[0], [300, 350])  # divided by the constant
    assert parse_calibration(cal_xml([0], [0], [[1]]), lut="betaNought").values[0][0] == 474


def test_azimuth_blocks_cover_their_samples_only():
    blocks = [AzimuthBlock("IW1", 0, 39, 0, 29, np.array([0.0, 39.0]), np.array([1.0, 2.0])),
              AzimuthBlock("IW2", 0, 39, 30, 59, np.array([0.0, 39.0]), np.array([3.0, 3.0]))]
    out = azimuth_noise(blocks, np.array([0.0, 39.0]), np.array([0.0, 29.0, 30.0, 59.0]))
    np.testing.assert_allclose(out, [[1, 1, 3, 3], [2, 2, 3, 3]])


def test_sigma0_formula_nodata_and_negative_clip():
    cal = SparseGrid(np.array([0.0, 10.0]), [np.array([0.0, 10.0])] * 2,
                     [np.array([500.0, 500.0])] * 2)
    noise = (SparseGrid(np.array([0.0, 10.0]), [np.array([0.0, 10.0])] * 2,
                        [np.array([100.0, 100.0])] * 2), [])
    dn = np.array([[0, 5, 100]], dtype=np.uint16)  # nodata, noise > DN^2, normal
    s0, st = sigma0_block(dn, np.array([0.0]), np.array([0.0, 1.0, 2.0]), cal, noise)
    assert np.isnan(s0[0, 0])
    assert s0[0, 1] == pytest.approx(1e-5 / 500**2)  # 25 - 100 < 0 -> SNAP floor
    assert s0[0, 2] == pytest.approx((100**2 - 100) / 500**2)
    assert (st.valid, st.negative) == (2, 1)


def make_safe(tmp_path):
    """Write a tiny but structurally real .SAFE zip: tiff with GCPs + both XMLs."""
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.crs import CRS

    rng = np.random.default_rng(1)
    dn = rng.integers(50, 400, size=(H, W), dtype=np.uint16)
    dn[:, :3] = 0  # no-data strip at the swath edge
    tif = tmp_path / "vv.tiff"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no georeferencing until GCPs are set below
        dst = rasterio.open(tif, "w", driver="GTiff", dtype="uint16", count=1,
                            width=W, height=H)
    with dst:
        dst.write(dn, 1)
        dst.gcps = ([GroundControlPoint(row=r, col=c, x=68 + c / 100, y=27 - r / 100)
                     for r in (0, H) for c in (0, W)], CRS.from_epsg(4326))
    stem = "S1A_IW_GRDH_1SDV_X.SAFE"
    z = tmp_path / "S1A_IW_GRDH_1SDV_X.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(tif, f"{stem}/measurement/s1a-iw-grd-vv-x-001.tiff")
        zf.writestr(f"{stem}/annotation/calibration/calibration-s1a-iw-grd-vv-x-001.xml",
                    cal_xml([0, H], [0, 30, W], [[600, 650, 700]] * 2))
        zf.writestr(f"{stem}/annotation/calibration/noise-s1a-iw-grd-vv-x-001.xml",
                    noise_xml([0, H], [0, W], [[1000, 2000]] * 2,
                              [{"swath": "IW1", "fl": 0, "ll": H - 1, "fs": 0, "ls": W - 1,
                                "lines": [0, H - 1], "lut": [1.0, 1.0]}]))
    return z, dn


def test_find_members(tmp_path):
    z, _ = make_safe(tmp_path)
    with zipfile.ZipFile(z) as zf:
        m = find_members(zf, "VV")
        assert m.measurement.endswith("-001.tiff")
        with pytest.raises(ValueError):
            find_members(zf, "vh")


def test_calibrate_safe_end_to_end(tmp_path):
    import rasterio

    z, dn = make_safe(tmp_path)
    out = tmp_path / "out" / "s0.tif"
    st = calibrate_safe(z, "vv", out, block_rows=16)  # several blocks, last one partial
    with rasterio.open(out) as src:
        s0 = src.read(1)
        assert (src.width, src.height) == (W, H)
        assert len(src.gcps[0]) == 4
        assert src.tags()["quantity"] == "sigma0_linear"

    # Independent brute-force reference for every pixel.
    cols = np.arange(W, dtype=float)
    a = np.interp(cols, [0, 30, W], [600, 650, 700])[None, :]
    noise = np.interp(cols, [0, W], [1000, 2000])[None, :]
    power = dn.astype(float) ** 2 - noise
    want = np.where(power <= 0, 1e-5, power) / a**2
    want[dn == 0] = np.nan
    np.testing.assert_allclose(s0, want.astype(np.float32), rtol=1e-6, equal_nan=True)
    assert st.valid == int((dn > 0).sum())
    assert st.negative == int(((dn.astype(float) ** 2 - noise) < 0)[dn > 0].sum())


def test_window_shifts_gcps(tmp_path):
    import rasterio

    z, _ = make_safe(tmp_path)
    out = tmp_path / "win.tif"
    calibrate_safe(z, "vv", out, window=(10, 30, 20, 50))
    with rasterio.open(out) as src:
        assert (src.height, src.width) == (20, 30)
        g = src.gcps[0][0]
        assert (g.row, g.col) == (-10, -20)  # original (0, 0) in the window's frame


def test_window_out_of_bounds(tmp_path):
    z, _ = make_safe(tmp_path)
    with pytest.raises(ValueError, match="outside image"):
        calibrate_safe(z, "vv", tmp_path / "x.tif", window=(0, H + 1, 0, W))


def test_parse_noise_real_layout():
    rng, blocks = parse_noise(noise_xml([0, 10], [0, 40], [[2048.9, 2022.4]] * 2,
                                        [{"swath": "IW1", "fl": 0, "ll": 16719, "fs": 0,
                                          "ls": 8523, "lines": [0, 10], "lut": [1.0579, 1.0560]}]))
    assert rng.values[0][0] == pytest.approx(2048.9)
    assert (blocks[0].swath, blocks[0].last_sample) == ("IW1", 8523)


def test_no_clip_keeps_negative_power():
    cal = SparseGrid(np.array([0.0, 10.0]), [np.array([0.0, 10.0])] * 2,
                     [np.array([500.0, 500.0])] * 2)
    noise = (SparseGrid(np.array([0.0, 10.0]), [np.array([0.0, 10.0])] * 2,
                        [np.array([100.0, 100.0])] * 2), [])
    dn = np.array([[5]], dtype=np.uint16)
    s0, st = sigma0_block(dn, np.array([0.0]), np.array([0.0]), cal, noise, clip_negative=False)
    assert s0[0, 0] == pytest.approx((25 - 100) / 500**2)
    assert st.negative == 1


def test_cli_compare_window_against_full_is_exact(tmp_path, capsys):
    import json

    from sarflood import cli

    z, _ = make_safe(tmp_path)
    full = tmp_path / "full.tif"   # stands in for SNAP's full-slice output
    win = tmp_path / "win.tif"
    calibrate_safe(z, "vv", full)
    calibrate_safe(z, "vv", win, window=(5, 35, 10, 50))
    assert cli.main(["compare", str(win), str(full)]) == 0
    rep = json.loads(win.with_suffix(".compare_b1.json").read_text())
    assert rep["window"] == [5, 35, 10, 50]
    assert rep["max_abs_db"] == 0.0 and rep["nodata_mismatch"] == 0
    assert rep["corr_log"] > rep["corr_log_flipped_lr"]


def test_cli_compare_missing_file(tmp_path, capsys):
    from sarflood import cli

    assert cli.main(["compare", str(tmp_path / "nope.tif"), str(tmp_path / "snap.tif")]) == 2
    assert "not found" in capsys.readouterr().err
