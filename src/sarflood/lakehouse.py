"""Find change/labels raster pairs under DATA_ROOT and describe them for upload to a
Databricks Unity Catalog Volume (the landing zone that bronze ingests from).

Local layout read (written by `sarflood change` and `sarflood evaluate`):
    $DATA_ROOT/change/<aoi>/<stem>_change.tif
    $DATA_ROOT/eval/<aoi>/<stem>_EMSR*_labels.tif      exactly one per stem

Volume layout written by scripts/upload_to_volume.py:
    <volume>/pairs/<aoi>/<stem>/<stem>_change.tif
    <volume>/pairs/<aoi>/<stem>/<stem>_EMSR..._labels.tif
    <volume>/pairs/<aoi>/<stem>/manifest.json
    <volume>/wheels/sarflood-<version>-py3-none-any.whl

The manifest is uploaded last, so a folder with a manifest holds a complete pair.
Bronze keys on manifests, never on bare rasters. The manifest carries SHA-256 of
both files, so a re-upload of unchanged files is skipped (idempotent), and a grid
check matching the one in features.load_aoi, so bronze can quarantine a bad pair
without opening the rasters.

No Databricks imports here: this module is plain Python and unit-tested offline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .manifest import data_root

MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
ROLES = ("change", "labels")


@dataclass(frozen=True)
class RasterPair:
    aoi: str
    stem: str
    change_tif: Path
    labels_tif: Path


def find_pairs(root: Path | None = None,
               aois: list[str] | None = None) -> tuple[list[RasterPair], list[str]]:
    """Pairs ready to upload, plus one message per change map that has no unique labels."""
    root = Path(root) if root is not None else data_root()
    pairs: list[RasterPair] = []
    problems: list[str] = []
    for ctif in sorted((root / "change").glob("*/*_change.tif")):
        aoi = ctif.parent.name
        if aois and aoi not in aois:
            continue
        stem = ctif.name[: -len("_change.tif")]
        labs = sorted((root / "eval" / aoi).glob(f"{stem}_EMSR*_labels.tif"))
        if len(labs) != 1:
            problems.append(f"{aoi}/{ctif.name}: {len(labs)} labels files (need exactly 1)")
            continue
        pairs.append(RasterPair(aoi=aoi, stem=stem, change_tif=ctif, labels_tif=labs[0]))
    return pairs, problems


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def raster_info(path: Path) -> dict:
    """Grid and band metadata, enough to check a pair without reading pixels."""
    import rasterio

    with rasterio.open(path) as src:
        return {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": [float(v) for v in list(src.transform)[:6]],
            "shape": [int(src.height), int(src.width)],
            "count": int(src.count),
            "bands": [d or "" for d in src.descriptions],
            "tags": dict(src.tags()),
        }


def build_manifest(pair: RasterPair) -> dict:
    files = {}
    for role, path in zip(ROLES, (pair.change_tif, pair.labels_tif), strict=True):
        files[role] = {"name": path.name, "bytes": path.stat().st_size,
                       "sha256": sha256_file(path), **raster_info(path)}
    c, lab = files["change"], files["labels"]
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "aoi": pair.aoi,
        "stem": pair.stem,
        "labels_version": lab["tags"].get("labels_version", "1"),
        # Same test as features.load_aoi: shapes and transforms must be identical.
        "grid_match": c["shape"] == lab["shape"] and c["transform"] == lab["transform"],
        "files": files,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def volume_root(catalog: str, schema: str, volume: str) -> str:
    return f"/Volumes/{catalog}/{schema}/{volume}"


def pair_dir(vol_root: str, pair: RasterPair) -> str:
    return f"{vol_root.rstrip('/')}/pairs/{pair.aoi}/{pair.stem}"


def same_content(local: dict, remote: dict | None) -> bool:
    """True if a remote manifest describes byte-identical files."""
    if not remote:
        return False
    rf = remote.get("files", {})
    return all(local["files"][r]["sha256"] == rf.get(r, {}).get("sha256") for r in ROLES)


# ---------------------------------------------------------------- bronze

BRONZE_COLUMNS = ["aoi", "stem", "flood_pass", "labels_version", "grid_match", "status",
                  "reason", "change_name", "labels_name", "change_sha256", "labels_sha256",
                  "change_bytes", "labels_bytes", "crs", "height", "width",
                  "manifest_path", "manifest_created_at"]


def flood_pass_of(stem: str) -> str:
    """'S1A_044775_0558C6__S1A_043375_052E02_20m_s3' -> 'S1A_044775_0558C6'."""
    return stem.split("__", 1)[0]


def bronze_row(man: dict, manifest_path: str, files_present: dict[str, bool]) -> dict:
    """One bronze row per manifest. status 'ok' or 'quarantine' with the reasons, so a
    bad pair is recorded (and visible) instead of silently dropped or crashing silver."""
    reasons = []
    if man.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        reasons.append(f"manifest schema {man.get('manifest_schema_version')} "
                       f"!= {MANIFEST_SCHEMA_VERSION}")
    if not man.get("grid_match"):
        reasons.append("change/labels grids differ")
    for role in ROLES:
        if not files_present.get(role):
            reasons.append(f"{role} file missing")
    c, lab = man["files"]["change"], man["files"]["labels"]
    return {
        "aoi": man["aoi"], "stem": man["stem"], "flood_pass": flood_pass_of(man["stem"]),
        "labels_version": str(man.get("labels_version", "1")),
        "grid_match": bool(man.get("grid_match")),
        "status": "quarantine" if reasons else "ok", "reason": "; ".join(reasons),
        "change_name": c["name"], "labels_name": lab["name"],
        "change_sha256": c["sha256"], "labels_sha256": lab["sha256"],
        "change_bytes": int(c["bytes"]), "labels_bytes": int(lab["bytes"]),
        "crs": c.get("crs"), "height": int(c["shape"][0]), "width": int(c["shape"][1]),
        "manifest_path": manifest_path, "manifest_created_at": man.get("created_at"),
    }


# ---------------------------------------------------------------- silver <-> AoiData

SILVER_KEY = ["aoi", "stem", "change_sha256", "labels_sha256"]


def silver_columns() -> list[str]:
    from .features import FEATURES

    return ["aoi", "stem", "idx", *FEATURES, "y", "frac", "labels_version",
            "change_sha256", "labels_sha256"]


def aoi_to_frame(d, stem: str, change_sha256: str, labels_sha256: str):
    """Scored pixels of one AOI as a pandas frame (one row per pixel), dtypes kept:
    features and frac float32, idx int64, y bool. Round-trips with frame_to_aoi."""
    import pandas as pd

    from .features import FEATURES

    df = pd.DataFrame(d.X, columns=FEATURES)          # float32 preserved
    df.insert(0, "aoi", d.name)
    df.insert(1, "stem", stem)
    df.insert(2, "idx", d.idx.astype("int64"))
    df["y"] = d.y.astype(bool)
    df["frac"] = d.frac.astype("float32")
    df["labels_version"] = d.labels_version
    df["change_sha256"] = change_sha256
    df["labels_sha256"] = labels_sha256
    # Constant string columns as categoricals: one dictionary entry instead of 12.8M
    # copies (Houston) when pyarrow converts the frame; they arrive in Delta as strings.
    for c in ("aoi", "stem", "labels_version", "change_sha256", "labels_sha256"):
        df[c] = df[c].astype("category")
    return df[silver_columns()]


def frame_to_aoi(df):
    """AoiData for training/scoring from one AOI's silver rows. Rows are re-sorted by
    pixel index, the order load_aoi produces, so seeded sampling picks the same pixels
    as on the Ubuntu box. Full-grid fields (labels, valid, shape) are not in silver:
    LOAO training and holdout scoring don't use them; outcome maps need load_aoi."""
    import numpy as np

    from .features import FEATURES, AoiData

    names, lvs, stems = df["aoi"].unique(), df["labels_version"].unique(), df["stem"].unique()
    if len(names) != 1 or len(lvs) != 1 or len(stems) != 1:
        raise ValueError(f"frame_to_aoi needs exactly one aoi/stem/labels_version, got "
                         f"{list(names)} / {list(stems)} / {list(lvs)}")
    df = df.sort_values("idx", kind="stable")
    return AoiData(name=str(names[0]), X=df[FEATURES].to_numpy(np.float32),
                   y=df["y"].to_numpy(bool), frac=df["frac"].to_numpy(np.float32),
                   idx=df["idx"].to_numpy(np.int64), shape=(0, 0), labels={},
                   valid=np.zeros(0, bool), pixel_area=float("nan"),
                   labels_version=str(lvs[0]))


# ---------------------------------------------------------------- gold: label-free drift

PSI_BINS = 10
PSI_EPS = 1e-4
PSI_LEVELS = ((0.10, "stable"), (0.25, "moderate"))   # >= 0.25: "major"


def psi_edges(reference, n_bins: int = PSI_BINS):
    """Interior bin edges at the reference distribution's quantiles (deduplicated, so a
    constant or discrete feature such as vh_valid gets fewer bins)."""
    import numpy as np

    qs = np.quantile(np.asarray(reference, float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(qs)


def psi(reference, actual, edges) -> float:
    """Population stability index: sum (a - e) ln(a / e) over bins, with bin shares
    floored at PSI_EPS so empty bins don't blow up. Uses no labels: it compares where
    the inputs fall, so it can flag a new scene before anyone has labelled it."""
    import numpy as np

    def shares(v):
        counts = np.bincount(np.searchsorted(edges, np.asarray(v, float), side="right"),
                             minlength=len(edges) + 1)
        return np.maximum(counts / max(counts.sum(), 1), PSI_EPS)

    e, a = shares(reference), shares(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_level(value: float) -> str:
    for cut, name in PSI_LEVELS:
        if value < cut:
            return name
    return "major"
