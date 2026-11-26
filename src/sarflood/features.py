"""Per-pixel features for flood classification, from a change map (+ its labels).

Features (all from the change GeoTIFF written by `sarflood change`):
    vv, vh             flood-pass sigma0 (dB)
    ref_vv, ref_vh     reference-pass sigma0 (dB)
    dvv, dvh           change, flood - reference (dB)
    vh_minus_vv        cross-pol ratio (dB); vegetation raises VH relative to VV
    vv_mean7, dvv_mean7  7x7 neighbourhood means (spatial context)
    vv_std7            7x7 standard deviation of VV (texture)
    vh_valid           1 if VH was valid in both passes, else 0

Missing VH (noise-floor pixels, mostly calm water) is filled with VH_FILL_DB and
flagged by vh_valid rather than dropped: dropping would remove mostly water,
which biases both training and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .change import box_mean

FEATURES = ["vv", "vh", "ref_vv", "ref_vh", "dvv", "dvh", "vh_minus_vv",
            "vv_mean7", "dvv_mean7", "vv_std7", "vh_valid"]
VH_FILL_DB = -35.0
CONTEXT = 7


def feature_stack(b: dict[str, np.ndarray]) -> np.ndarray:
    """b: change-map bands by name. Returns (len(FEATURES), H, W) float32."""
    valid_vh = b["valid_vh"] > 0
    vv, ref_vv, dvv = b["flood_vv_db"], b["ref_vv_db"], b["dvv_db"]
    vh = np.where(valid_vh, b["flood_vh_db"], VH_FILL_DB)
    ref_vh = np.where(valid_vh, b["ref_vh_db"], VH_FILL_DB)
    dvh = np.where(valid_vh, b["dvh_db"], 0.0)
    mean_vv = box_mean(vv, CONTEXT)
    var_vv = box_mean(vv * vv, CONTEXT) - mean_vv * mean_vv
    feats = {
        "vv": vv, "vh": vh, "ref_vv": ref_vv, "ref_vh": ref_vh, "dvv": dvv, "dvh": dvh,
        "vh_minus_vv": vh - vv, "vv_mean7": mean_vv, "dvv_mean7": box_mean(dvv, CONTEXT),
        "vv_std7": np.sqrt(np.clip(var_vv, 0, None)), "vh_valid": valid_vh.astype(np.float32),
    }
    return np.stack([np.asarray(feats[n], np.float32) for n in FEATURES])


@dataclass
class AoiData:
    name: str
    X: np.ndarray          # (n, F) features of scored pixels
    y: np.ndarray          # (n,) bool, EMS flood (fraction >= 0.5)
    frac: np.ndarray       # (n,) EMS flood fraction
    idx: np.ndarray        # (n,) flat indices into the grid, to rebuild maps
    shape: tuple[int, int]
    labels: dict           # full-grid label bands (for outcome maps)
    valid: np.ndarray      # full-grid VV validity
    pixel_area: float


def load_aoi(name: str, change_tif: Path, labels_tif: Path) -> AoiData:
    import rasterio

    with rasterio.open(change_tif) as src:
        names = list(src.descriptions)
        b = {n: src.read(names.index(n) + 1) for n in
             ("flood_vv_db", "ref_vv_db", "dvv_db", "flood_vh_db", "ref_vh_db", "dvh_db",
              "valid", "valid_vh")}
        shape, px = src.shape, abs(src.transform.a * src.transform.e)
        transform = src.transform
    with rasterio.open(labels_tif) as src:
        if src.shape != shape or src.transform != transform:
            raise ValueError(f"{name}: labels grid differs from change grid")
        ln = list(src.descriptions)
        labels = {n: src.read(ln.index(n) + 1) for n in ln}
    valid = b["valid"] > 0
    region = (labels["aoi"] > 0) & (labels["not_analysed"] == 0) & (labels["perm_water"] == 0) & valid
    stack = feature_stack(b)
    region &= np.isfinite(stack).all(axis=0)  # context features need a valid neighbourhood
    idx = np.flatnonzero(region)
    X = stack.reshape(len(FEATURES), -1)[:, idx].T.copy()
    return AoiData(name=name, X=X, y=labels["flood"].ravel()[idx] > 0,
                   frac=labels["flood_frac"].ravel()[idx], idx=idx, shape=shape,
                   labels=labels, valid=valid, pixel_area=px)
