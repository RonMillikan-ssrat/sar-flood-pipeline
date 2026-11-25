"""Rasterize Copernicus EMS delineation layers onto a change grid, and score a flood map.

Layers used from the EMS vector package (GeoJSON copies, EPSG:4326), as found in
EMSR629 AOI01 DEL v2:
    observedEventA   flood polygons (notation 'Flooded area')
    areaOfInterestA  AOI outline
    imageFootprintA  obj_type 'Not Analysed' = areas EMS did not assess
    hydrographyA     obj_type BH080-Lake / BH140-River = permanent water.
                     BH090-Land Subject to Inundation is floodable land, not water,
                     so it is NOT part of the permanent-water mask.

Flood polygons are traced from 6 m optical imagery; our grid is 20 m. Each pixel's
flood *fraction* is estimated by rasterizing on a finer grid (supersample x
supersample sub-pixels per pixel). label = fraction >= 0.5. Pixels with a fraction
strictly between `mixed_lo` and `mixed_hi` can be left out of scoring as mixed.

Scoring region = inside the EMS AOI, assessed (not 'Not Analysed'), valid in the
change map (no layover/shadow, positive backscatter in both passes), and not
permanent water (whether EMS counts normal water as flood varies, so it's excluded
rather than guessed).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PERMANENT_WATER = ("BH080", "BH140")  # lake, river (obj_type prefixes)
LABEL_BANDS = ["flood_frac", "flood", "aoi", "not_analysed", "perm_water"]


def read_layer(zf: zipfile.ZipFile, tag: str) -> list[dict]:
    names = [n for n in zf.namelist() if f"_{tag}_" in n and n.endswith(".json")]
    if not names:
        return []
    return json.loads(zf.read(names[0]))["features"]


def _to_crs(features: list[dict], dst_crs) -> list[dict]:
    from rasterio.warp import transform_geom

    return [transform_geom("EPSG:4326", dst_crs, f["geometry"]) for f in features]


def _burn(geoms: list[dict], shape: tuple[int, int], transform, supersample: int = 1) -> np.ndarray:
    """Fraction of each pixel covered by geoms (0..1), via sub-pixel rasterization."""
    from affine import Affine
    from rasterio.features import rasterize

    h, w = shape
    if not geoms:
        return np.zeros(shape, np.float32)
    if supersample == 1:
        return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                         fill=0, dtype="uint8").astype(np.float32)
    fine_t = transform @ Affine.scale(1 / supersample)
    fine = rasterize(((g, 1) for g in geoms), out_shape=(h * supersample, w * supersample),
                     transform=fine_t, fill=0, dtype="uint8")
    return fine.reshape(h, supersample, w, supersample).mean(axis=(1, 3)).astype(np.float32)


def rasterize_ems(zip_path: Path, shape: tuple[int, int], transform, crs,
                  supersample: int = 5) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(zip_path) as zf:
        flood = read_layer(zf, "observedEventA")
        aoi = read_layer(zf, "areaOfInterestA")
        foot = read_layer(zf, "imageFootprintA")
        hydro = read_layer(zf, "hydrographyA")
    na = [f for f in foot if f["properties"].get("obj_type") == "Not Analysed"]
    perm = [f for f in hydro if str(f["properties"].get("obj_type", "")).startswith(PERMANENT_WATER)]
    frac = _burn(_to_crs(flood, crs), shape, transform, supersample)
    return {
        "flood_frac": frac,
        "flood": (frac >= 0.5).astype(np.float32),
        "aoi": (_burn(_to_crs(aoi, crs), shape, transform, supersample) >= 0.5).astype(np.float32),
        "not_analysed": (_burn(_to_crs(na, crs), shape, transform, supersample) > 0).astype(np.float32),
        "perm_water": (_burn(_to_crs(perm, crs), shape, transform, supersample) >= 0.5).astype(np.float32),
    }


@dataclass
class Score:
    n_eval: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    iou: float
    label_km2: float
    pred_km2: float
    excluded_mixed: int

    def to_dict(self) -> dict:
        return asdict(self)


def score(pred: np.ndarray, labels: dict[str, np.ndarray], valid: np.ndarray,
          pixel_area_m2: float, mixed_lo: float | None = None,
          mixed_hi: float | None = None) -> Score:
    """pred, valid: boolean arrays on the label grid."""
    region = (labels["aoi"] > 0) & (labels["not_analysed"] == 0) & (labels["perm_water"] == 0) & valid
    excluded = 0
    if mixed_lo is not None and mixed_hi is not None:
        mixed = (labels["flood_frac"] > mixed_lo) & (labels["flood_frac"] < mixed_hi)
        excluded = int((region & mixed).sum())
        region &= ~mixed
    truth = labels["flood"] > 0
    p, t = pred[region], truth[region]
    tp, fp = int((p & t).sum()), int((p & ~t).sum())
    fn, tn = int((~p & t).sum()), int((~p & ~t).sum())

    def div(a, b):
        return float(a / b) if b else float("nan")

    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return Score(
        n_eval=int(region.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        precision=prec, recall=rec, f1=div(2 * tp, 2 * tp + fp + fn), iou=div(tp, tp + fp + fn),
        label_km2=(tp + fn) * pixel_area_m2 / 1e6, pred_km2=(tp + fp) * pixel_area_m2 / 1e6,
        excluded_mixed=excluded,
    )


def threshold_sweep(dvv: np.ndarray, fvv: np.ndarray, labels: dict[str, np.ndarray],
                    valid: np.ndarray, pixel_area_m2: float,
                    change_grid=(None, 3.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0),
                    water_grid=(-8.0, -10.0, -12.0, -14.0, -16.0, -18.0, -20.0)) -> list[dict]:
    """IoU for each (change_db, water_db) pair. change_db None = no change condition
    (a single-image threshold on flood VV only). Tuning on the evaluation labels is
    optimistic; use it to understand the data, not as a reported accuracy."""
    rows = []
    for c in change_grid:
        for w in water_grid:
            pred = valid & (fvv <= w)
            if c is not None:
                pred &= dvv <= c
            s = score(pred, labels, valid, pixel_area_m2)
            rows.append({"change_db": c, "water_db": w, "iou": s.iou, "f1": s.f1,
                         "precision": s.precision, "recall": s.recall})
    return rows


def best_single_threshold(x: np.ndarray, truth: np.ndarray, n: int = 200) -> dict:
    """Best IoU of the rule 'x <= t' (darker = flood) over n candidate thresholds
    taken from the percentiles of x. Shows how well one variable alone separates."""
    ok = np.isfinite(x)
    x, truth = x[ok], truth[ok]
    if x.size == 0 or not truth.any():
        return {"t": float("nan"), "iou": float("nan")}
    ts = np.unique(np.percentile(x, np.linspace(0.5, 99.5, n)))
    n_true = truth.sum()
    best = (float("nan"), -1.0)
    for t in ts:
        p = x <= t
        tp = int((p & truth).sum())
        iou = tp / (n_true + p.sum() - tp)
        if iou > best[1]:
            best = (float(t), float(iou))
    return {"t": best[0], "iou": best[1]}


HIST_VARS = [  # (band name, label, needs VH validity)
    ("flood_vv_db", "flood-pass VV (dB)", False),
    ("ref_vv_db", "reference VV (dB)", False),
    ("dvv_db", "VV change, flood - ref (dB)", False),
    ("flood_vh_db", "flood-pass VH (dB)", True),
]


def class_histograms(bands: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                     valid: np.ndarray, valid_vh: np.ndarray, png: Path,
                     title: str = "") -> dict:
    """Per variable: distributions for EMS flood vs EMS dry pixels in the scoring
    region, class percentiles, and the best single-threshold IoU."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = (labels["aoi"] > 0) & (labels["not_analysed"] == 0) & (labels["perm_water"] == 0)
    truth = labels["flood"] > 0
    summary = {}
    fig, axes = plt.subplots(1, len(HIST_VARS), figsize=(5 * len(HIST_VARS), 4),
                             constrained_layout=True)
    for ax, (name, label, vh) in zip(axes, HIST_VARS):
        region = base & (valid_vh if vh else valid)
        v = bands[name]
        f, d = v[region & truth], v[region & ~truth]
        f, d = f[np.isfinite(f)], d[np.isfinite(d)]
        lo, hi = np.percentile(np.concatenate([f, d]), [0.5, 99.5]) if f.size + d.size else (0, 1)
        if hi <= lo:  # degenerate (constant) data: give the histogram a usable range
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 120)
        ax.hist(d, bins=bins, density=True, alpha=0.55, color="#b8860b", label=f"EMS dry (n={d.size:,})")
        ax.hist(f, bins=bins, density=True, alpha=0.55, color="#2166ac", label=f"EMS flood (n={f.size:,})")
        best = best_single_threshold(np.where(region, v, np.nan), truth & region)
        ax.axvline(best["t"], color="k", lw=1, ls="--")
        ax.set_title(f"{label}\nbest single cut {best['t']:.1f}: IoU {best['iou']:.3f}", fontsize=9)
        ax.legend(fontsize=7)
        pct = lambda a: [float(q) for q in np.percentile(a, [10, 50, 90])] if a.size else []
        summary[name] = {"flood_p10_p50_p90": pct(f), "dry_p10_p50_p90": pct(d),
                         "best_cut": best["t"], "best_cut_iou": best["iou"]}
    fig.suptitle(title, fontsize=9)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return summary


def confusion_quicklook(pred: np.ndarray, labels: dict[str, np.ndarray], valid: np.ndarray,
                        png: Path, title: str = "", max_px: int = 1600) -> Path:
    """Map of outcomes: hit (TP), false alarm (FP), miss (FN), correct dry (TN), excluded."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    region = (labels["aoi"] > 0) & (labels["not_analysed"] == 0) & (labels["perm_water"] == 0) & valid
    truth = labels["flood"] > 0
    cat = np.zeros(pred.shape, np.uint8)            # 0 outside AOI
    cat[(labels["aoi"] > 0) & ~region] = 1          # excluded inside AOI
    cat[region & ~pred & ~truth] = 2                # TN
    cat[region & pred & truth] = 3                  # TP
    cat[region & pred & ~truth] = 4                 # FP
    cat[region & ~pred & truth] = 5                 # FN
    cat[labels["perm_water"] > 0] = 6               # permanent water (shown, not scored)
    step = max(1, int(np.ceil(max(cat.shape) / max_px)))
    colors = ["#ffffff", "#bdbdbd", "#f2efe6", "#2166ac", "#d6604d", "#f4a582", "#053061"]
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.imshow(cat[::step, ::step], cmap=ListedColormap(colors), vmin=0, vmax=6, interpolation="nearest")
    names = ["outside AOI", "excluded", "dry, correct", "flood, hit", "false alarm", "missed flood",
             "permanent water"]
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors[1:], names[1:])],
              loc="lower right", fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png
