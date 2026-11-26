"""Write scene manifests: one JSON file per AOI per query window.

Local disk for now (DATA_ROOT, default ./data). This later becomes a gs:// prefix.
The file name is deterministic for a given AOI and window, so rerunning a
backfill overwrites rather than duplicates. That makes the task idempotent.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .catalog import Scene


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "data"))


def raw_store() -> Path:
    """Shared store for raw Sentinel-1 zips. Slices don't depend on the AOI, so one
    folder serves all AOIs (no per-AOI copies or symlinks)."""
    return data_root() / "raw" / "s1"


def find_raw(zip_name: str) -> Path | None:
    """Existing zip in the shared store, else in a legacy per-AOI folder
    (data/raw/<aoi>/, possibly a symlink), else None."""
    p = raw_store() / zip_name
    if p.is_file():
        return p
    raw = data_root() / "raw"
    if raw.is_dir():
        for d in sorted(raw.iterdir()):
            q = d / zip_name
            if d.name != "s1" and d.is_dir() and q.is_file():
                return q
    return None


def manifest_path(aoi: str, start: datetime, end: datetime, root: Path | None = None) -> Path:
    root = root or data_root()
    fname = f"{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json"
    return root / "manifests" / aoi / fname


def write_manifest(aoi: str, start: datetime, end: datetime, scenes: list[Scene],
                   product_type: str, root: Path | None = None) -> Path:
    path = manifest_path(aoi, start, end, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "aoi": aoi,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "product_type": product_type,
        "queried_at": datetime.now(UTC).isoformat(),
        "count": len(scenes),
        "scenes": [s.to_dict() for s in scenes],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)  # atomic rename, so readers never see a half-written file
    return path


def load_manifest(path: Path | str) -> list[Scene]:
    doc = json.loads(Path(path).read_text())
    return [Scene(**s) for s in doc["scenes"]]


def write_json(path: Path, doc: dict) -> Path:
    """Atomic JSON write for other outputs (e.g. pairs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)
    return path
