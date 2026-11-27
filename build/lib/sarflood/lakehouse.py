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
