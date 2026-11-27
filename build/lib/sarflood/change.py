"""Change detection between a terrain-corrected flood pass and its dry reference.

Inputs are two outputs of `sarflood terrain-correct` for the same AOI. SNAP
writes them without band names, so the order is fixed here and was identified
from the statistics of the first Sindh run:

    1 sigma0 VH (linear)   2 sigma0 VV (linear)
    3 local incidence angle (deg)   4 layover/shadow mask (0 = clear)

Both rasters must sit on the same pixel grid (alignToStandardGrid in the SNAP
graph). Their extents can still differ by a pixel at the edges, so we work on
the intersection, and refuse to run when the grids are not aligned rather than
resampling silently.

Per pixel, on the overlap:
    smoothing  optional N x N mean of linear sigma0 in each pass before dB (N=1: off).
               Averages speckle; negative (unclipped) values take part, which is
               why noise removal is not clipped upstream. NaN / layover-shadow pixels
               are left out of the mean; a pixel needs > half its window valid.
    valid      (VV) both passes > 0 and finite in VV, layover/shadow mask 0 in both.
    valid_vh   the same for VH. Kept separate: VH goes negative mostly over calm
               water, and requiring it for VV products would drop likely flood pixels.
    *_db       10 log10(sigma0)
    dvv, dvh   flood dB - reference dB  (negative = darker in the flood pass)
    candidate  valid & dvv <= change_db & flood VV <= water_db
               A simple threshold preview for looking at the data, NOT a validated
               flood map. The defaults are starting values to tune against the EMS
               polygons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

BAND_VH, BAND_VV, BAND_LIA, BAND_MASK = 1, 2, 3, 4
OUT_BANDS = ["ref_vv_db", "flood_vv_db", "dvv_db", "ref_vh_db", "flood_vh_db", "dvh_db",
             "valid", "valid_vh", "candidate"]


def box_mean(x: np.ndarray, n: int) -> np.ndarray:
    """NaN-aware N x N moving mean (N odd) via integral images. A pixel gets a value
    only if more than half of its window is finite; otherwise NaN."""
    if n <= 1:
        return x.astype(np.float32)
    if n % 2 == 0:
        raise ValueError("smoothing window must be odd")
    ok = np.isfinite(x)
    vals = np.where(ok, x, 0.0).astype(np.float64)
    r = n // 2

    def box_sum(a: np.ndarray) -> np.ndarray:
        p = np.pad(a, r + 1, mode="constant")[:-1, :-1]  # leading zero row/col for the cumsum
        c = p.cumsum(0).cumsum(1)
        return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]

    s, cnt = box_sum(vals), box_sum(ok.astype(np.float64))
    out = np.full(x.shape, np.nan, np.float32)
    good = cnt > (n * n) / 2
    out[good] = (s[good] / cnt[good]).astype(np.float32)
    return out


@dataclass
class Overlap:
    """Pixel windows (row0, col0, height, width) of the shared area in each raster."""
    flood: tuple[int, int, int, int]
    ref: tuple[int, int, int, int]
    transform: object  # affine.Affine of the overlap


def overlap(flood_src, ref_src, tol: float = 1e-6) -> Overlap:
    if flood_src.crs != ref_src.crs:
        raise ValueError(f"CRS differ: {flood_src.crs} vs {ref_src.crs}")
    ft, rt = flood_src.transform, ref_src.transform
    if ft.b != 0 or ft.d != 0 or rt.b != 0 or rt.d != 0:
        raise ValueError("rotated rasters are not supported")
    if abs(ft.a - rt.a) > tol or abs(ft.e - rt.e) > tol:
        raise ValueError(f"pixel sizes differ: {ft.a, ft.e} vs {rt.a, rt.e}")
    res_x, res_y = ft.a, -ft.e
    # Offset of the reference origin from the flood origin, in pixels; must be whole.
    dx = (rt.c - ft.c) / res_x
    dy = (ft.f - rt.f) / res_y
    if abs(dx - round(dx)) > 1e-3 or abs(dy - round(dy)) > 1e-3:
        raise ValueError(f"grids not aligned: reference is offset by ({dx:.4f}, {dy:.4f}) px. "
                         "Check alignToStandardGrid / pixel spacing in the SNAP graph")
    dx, dy = round(dx), round(dy)
    # Overlap in flood pixel coordinates.
    c0, r0 = max(0, dx), max(0, dy)
    c1 = min(flood_src.width, dx + ref_src.width)
    r1 = min(flood_src.height, dy + ref_src.height)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("rasters do not overlap")
    h, w = r1 - r0, c1 - c0
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform

    return Overlap(flood=(r0, c0, h, w), ref=(r0 - dy, c0 - dx, h, w),
                   transform=win_transform(Window(c0, r0, w, h), ft))


def _read(src, band: int, win: tuple[int, int, int, int]) -> np.ndarray:
    from rasterio.windows import Window

    r0, c0, h, w = win
    return src.read(band, window=Window(c0, r0, w, h)).astype(np.float32)


def to_db(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan, dtype=np.float32)
    pos = np.isfinite(x) & (x > 0)
    out[pos] = 10 * np.log10(x[pos])
    return out


@dataclass
class ChangeStats:
    pixels: int
    valid: int
    valid_vh: int
    masked_layover_shadow: int
    masked_nonpositive: int      # VV
    masked_nonpositive_vh: int
    smooth: int
    dvv_p05: float
    dvv_p50: float
    dvv_p95: float
    candidate_pixels: int
    candidate_km2: float
    change_db: float
    water_db: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_change(flood: dict[str, np.ndarray], ref: dict[str, np.ndarray],
                   pixel_area_m2: float, change_db: float = -3.0,
                   water_db: float = -18.0, smooth: int = 1) -> tuple[dict[str, np.ndarray], ChangeStats]:
    """flood/ref: dicts with linear 'vv', 'vh' and 'mask' arrays of equal shape."""
    ls = (flood["mask"] != 0) | (ref["mask"] != 0)

    def prep(d: dict, pol: str) -> np.ndarray:
        x = d[pol].astype(np.float32).copy()
        x[d["mask"] != 0] = np.nan          # layover/shadow out of the window means
        return box_mean(x, smooth)

    fvv, rvv = to_db(prep(flood, "vv")), to_db(prep(ref, "vv"))
    fvh, rvh = to_db(prep(flood, "vh")), to_db(prep(ref, "vh"))
    finite_vv = np.isfinite(fvv) & np.isfinite(rvv)
    finite_vh = np.isfinite(fvh) & np.isfinite(rvh)
    valid, valid_vh = finite_vv & ~ls, finite_vh & ~ls

    def masked(a, m):
        out = a.copy()
        out[~m] = np.nan
        return out

    dvv = masked(fvv - rvv, valid)
    dvh = masked(fvh - rvh, valid_vh)
    cand = valid & (dvv <= change_db) & (fvv <= water_db)
    d = dvv[valid]
    pct = np.percentile(d, [5, 50, 95]) if d.size else [np.nan] * 3
    stats = ChangeStats(
        pixels=int(valid.size), valid=int(valid.sum()), valid_vh=int(valid_vh.sum()),
        masked_layover_shadow=int(ls.sum()),
        masked_nonpositive=int((~finite_vv & ~ls).sum()),
        masked_nonpositive_vh=int((~finite_vh & ~ls).sum()),
        smooth=smooth,
        dvv_p05=float(pct[0]), dvv_p50=float(pct[1]), dvv_p95=float(pct[2]),
        candidate_pixels=int(cand.sum()),
        candidate_km2=float(cand.sum() * pixel_area_m2 / 1e6),
        change_db=change_db, water_db=water_db,
    )
    bands = {
        "ref_vv_db": masked(rvv, valid), "flood_vv_db": masked(fvv, valid), "dvv_db": dvv,
        "ref_vh_db": masked(rvh, valid_vh), "flood_vh_db": masked(fvh, valid_vh), "dvh_db": dvh,
        "valid": valid.astype(np.float32), "valid_vh": valid_vh.astype(np.float32),
        "candidate": cand.astype(np.float32),
    }
    return bands, stats


def run_change(flood_tif: Path, ref_tif: Path, out_tif: Path,
               change_db: float = -3.0, water_db: float = -18.0, smooth: int = 1) -> ChangeStats:
    import rasterio

    with rasterio.open(flood_tif) as fs, rasterio.open(ref_tif) as rs:
        ov = overlap(fs, rs)
        flood = {"vh": _read(fs, BAND_VH, ov.flood), "vv": _read(fs, BAND_VV, ov.flood),
                 "mask": _read(fs, BAND_MASK, ov.flood)}
        ref = {"vh": _read(rs, BAND_VH, ov.ref), "vv": _read(rs, BAND_VV, ov.ref),
               "mask": _read(rs, BAND_MASK, ov.ref)}
        crs = fs.crs
        pixel_area = abs(fs.transform.a * fs.transform.e)
    bands, stats = compute_change(flood, ref, pixel_area, change_db, water_db, smooth)

    h, w = bands["dvv_db"].shape
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "dtype": "float32", "count": len(OUT_BANDS),
               "width": w, "height": h, "crs": crs, "transform": ov.transform,
               "nodata": float("nan"), "tiled": True, "blockxsize": 512, "blockysize": 512,
               "compress": "deflate", "predictor": 3, "BIGTIFF": "IF_SAFER"}
    with rasterio.open(out_tif, "w", **profile) as dst:
        for i, name in enumerate(OUT_BANDS, 1):
            dst.write(bands[name], i)
            dst.set_band_description(i, name)
        dst.update_tags(flood=Path(flood_tif).name, reference=Path(ref_tif).name,
                        change_db=str(change_db), water_db=str(water_db), smooth=str(smooth),
                        note="candidate = simple threshold preview, not a validated flood map")
    return stats


def quicklook(change_tif: Path, png: Path, max_px: int = 1600) -> Path:
    """Reference VV | flood VV | VV change, decimated for display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    with rasterio.open(change_tif) as src:
        step = max(1, int(np.ceil(max(src.width, src.height) / max_px)))
        shape = (src.height // step, src.width // step)
        names = list(src.descriptions)
        get = lambda n: src.read(names.index(n) + 1, out_shape=shape)
        ref, fl, d = get("ref_vv_db"), get("flood_vv_db"), get("dvv_db")
        tags = src.tags()
    fig, ax = plt.subplots(1, 3, figsize=(15, 8), constrained_layout=True)
    for a, img, title in [(ax[0], ref, "reference VV (dB)"), (ax[1], fl, "flood VV (dB)")]:
        im = a.imshow(img, cmap="gray", vmin=-25, vmax=0)
        a.set_title(title)
        fig.colorbar(im, ax=a, shrink=0.6)
    im = ax[2].imshow(d, cmap="RdBu", vmin=-10, vmax=10)
    ax[2].set_title("VV change, flood - reference (dB)\nblue = brighter, red = darker")
    fig.colorbar(im, ax=ax[2], shrink=0.6)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{tags.get('flood', '')}  vs  {tags.get('reference', '')}", fontsize=9)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png
