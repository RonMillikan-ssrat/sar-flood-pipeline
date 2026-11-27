"""Compare our sigma0 window against SNAP's full-slice sigma0, pixel for pixel.

Both are in radar geometry on the same grid (SNAP's TNR + Calibration does not
resample), so pixel (r, c) of our window corresponds to (r + row0, c + col0) in
SNAP's output.

Reported:
  - no-data agreement (ours NaN vs SNAP 0 or NaN)
  - zero agreement (pixels clipped to 0 by each tool)
  - for pixels positive in both: the difference in dB, 10*log10(ours / snap):
    mean, median, std, percentiles, max |diff|, and the share within 0.01 / 0.1 dB
  - a geometry sanity check: the correlation of log values, and whether a flipped
    alignment would correlate better (it would mean the grids are mirrored)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Comparison:
    n_pixels: int
    nodata_ours: int
    nodata_snap: int
    nodata_mismatch: int
    zero_ours: int
    zero_snap: int
    zero_mismatch: int
    negative_ours: int
    negative_snap: int
    n_compared: int
    mean_db: float
    median_db: float
    std_db: float
    p01_db: float
    p99_db: float
    max_abs_db: float
    within_001_db: float
    within_01_db: float
    corr_log: float
    corr_log_flipped_lr: float
    corr_log_flipped_ud: float

    def to_dict(self) -> dict:
        return asdict(self)


def _corr_log(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 10:
        return float("nan")
    la, lb = np.log10(a[m]), np.log10(b[m])
    return float(np.corrcoef(la, lb)[0, 1])


def compare_arrays(ours: np.ndarray, snap: np.ndarray, snap_nodata: float | None = 0.0) -> Comparison:
    if ours.shape != snap.shape:
        raise ValueError(f"shape mismatch: ours {ours.shape} vs snap {snap.shape}")
    ours = ours.astype(np.float64)
    snap = snap.astype(np.float64)

    nd_o = ~np.isfinite(ours)
    # SNAP marks no-data with its band no-data value (typically 0) or NaN. A real 0
    # after clipping is indistinguishable from no-data in SNAP's output, so where our
    # pixel is valid, SNAP 0 counts as "zero", not as no-data.
    nd_s = ~np.isfinite(snap)
    if snap_nodata is not None:
        nd_s |= (snap == snap_nodata) & nd_o

    valid = ~nd_o & ~nd_s
    zero_o = valid & (ours == 0)
    zero_s = valid & (snap == 0)
    both_pos = valid & (ours > 0) & (snap > 0)

    d = 10 * np.log10(ours[both_pos] / snap[both_pos])
    if d.size:
        ad = np.abs(d)
        stats = {
            "mean_db": float(d.mean()), "median_db": float(np.median(d)), "std_db": float(d.std()),
            "p01_db": float(np.percentile(d, 1)), "p99_db": float(np.percentile(d, 99)),
            "max_abs_db": float(ad.max()),
            "within_001_db": float((ad <= 0.01).mean()), "within_01_db": float((ad <= 0.1).mean()),
        }
    else:
        stats = dict.fromkeys(("mean_db", "median_db", "std_db", "p01_db", "p99_db",
                               "max_abs_db", "within_001_db", "within_01_db"), float("nan"))

    return Comparison(
        n_pixels=int(ours.size),
        nodata_ours=int(nd_o.sum()), nodata_snap=int(nd_s.sum()),
        nodata_mismatch=int((nd_o ^ nd_s).sum()),
        zero_ours=int(zero_o.sum()), zero_snap=int(zero_s.sum()),
        zero_mismatch=int((zero_o ^ zero_s).sum()),
        negative_ours=int((valid & (ours < 0)).sum()),
        negative_snap=int((valid & (snap < 0)).sum()),
        n_compared=int(both_pos.sum()),
        corr_log=_corr_log(ours, snap),
        corr_log_flipped_lr=_corr_log(ours, snap[:, ::-1]),
        corr_log_flipped_ud=_corr_log(ours, snap[::-1, :]),
        **stats,
    )


def read_window(path: str, band: int | str, window: tuple[int, int, int, int]) -> np.ndarray:
    """band may be a 1-based index or a band description such as 'Sigma0_VV'."""
    import rasterio
    from rasterio.windows import Window

    r0, r1, c0, c1 = window
    with rasterio.open(path) as src:
        if isinstance(band, str):
            names = list(src.descriptions)
            if band not in names:
                raise ValueError(f"band {band!r} not in {path}; bands are {names}")
            band = names.index(band) + 1
        return src.read(band, window=Window(c0, r0, c1 - c0, r1 - r0))


def describe(path: str) -> list[str]:
    import rasterio

    with rasterio.open(path) as src:
        head = (f"{src.width} x {src.height}, {src.count} band(s), dtype {src.dtypes[0]}, "
                f"nodata {src.nodata}")
        return [head] + [f"  band {i + 1}: {d}" for i, d in enumerate(src.descriptions)]
