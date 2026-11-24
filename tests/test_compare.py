import numpy as np
import pytest

from sarflood.compare import compare_arrays


def field(seed=0, shape=(64, 80)):
    rng = np.random.default_rng(seed)
    # Spatially structured so a mirrored copy correlates worse than the aligned one.
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    base = 10 ** ((-15 + 8 * np.sin(x / 9.0) + 0.05 * y) / 10)
    return base * rng.gamma(4.0, 0.25, size=shape)  # speckle-like


def test_identical_arrays():
    a = field()
    r = compare_arrays(a, a.copy())
    assert r.max_abs_db == 0 and r.within_001_db == 1.0
    assert r.corr_log == pytest.approx(1.0)
    assert r.corr_log > r.corr_log_flipped_lr


def test_constant_gain_shows_as_mean_offset():
    a = field()
    r = compare_arrays(a * 10 ** (0.2 / 10), a)  # ours 0.2 dB hotter
    assert r.mean_db == pytest.approx(0.2, abs=1e-9)
    assert r.std_db == pytest.approx(0, abs=1e-9)
    assert r.within_01_db == 0.0


def test_mirrored_grid_is_detected():
    a = field()
    r = compare_arrays(a, a[:, ::-1])
    assert r.corr_log_flipped_lr > r.corr_log


def test_nodata_and_zero_accounting():
    a = field()
    ours, snap = a.copy(), a.copy()
    ours[0, :5] = np.nan
    snap[0, :5] = 0.0          # SNAP no-data where ours is NaN -> agreement
    ours[1, :3] = 0.0
    snap[1, :3] = 0.0          # both clipped -> agreement
    snap[2, 0] = 0.0           # only SNAP clipped -> zero mismatch
    r = compare_arrays(ours, snap)
    assert (r.nodata_ours, r.nodata_snap, r.nodata_mismatch) == (5, 5, 0)
    assert (r.zero_ours, r.zero_snap, r.zero_mismatch) == (3, 4, 1)
    assert r.n_compared == a.size - 5 - 4


def test_negative_counts_when_unclipped():
    a = field()
    ours, snap = a.copy(), a.copy()
    ours[3, :2] = -1e-4
    snap[3, :1] = -1e-4
    r = compare_arrays(ours, snap)
    assert (r.negative_ours, r.negative_snap) == (2, 1)


def test_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        compare_arrays(np.ones((2, 2)), np.ones((2, 3)))
