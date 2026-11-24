"""Load areas of interest (AOIs) from configs/aois.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "aois.yaml"
VALID_ROLES = {"train", "holdout"}


@dataclass(frozen=True)
class Event:
    name: str
    start: date
    end: date
    notes: str = ""


@dataclass(frozen=True)
class AOI:
    name: str
    role: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    description: str = ""
    events: tuple[Event, ...] = field(default_factory=tuple)

    def to_wkt(self) -> str:
        """Closed WKT polygon (first point repeated last), as the OData API requires."""
        x0, y0, x1, y1 = self.bbox
        ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        return "POLYGON((" + ",".join(f"{x} {y}" for x, y in ring) + "))"


def _parse_bbox(name: str, raw) -> tuple[float, float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"{name}: bbox must be [min_lon, min_lat, max_lon, max_lat]")
    x0, y0, x1, y1 = (float(v) for v in raw)
    if not (-180 <= x0 < x1 <= 180):
        raise ValueError(f"{name}: bad longitude range {x0}..{x1}")
    if not (-90 <= y0 < y1 <= 90):
        raise ValueError(f"{name}: bad latitude range {y0}..{y1}")
    return (x0, y0, x1, y1)


def _parse_event(aoi_name: str, raw: dict) -> Event:
    start = date.fromisoformat(str(raw["start"]))
    end = date.fromisoformat(str(raw["end"]))
    if end <= start:
        raise ValueError(f"{aoi_name}/{raw['name']}: end must be after start")
    return Event(name=raw["name"], start=start, end=end, notes=raw.get("notes", ""))


def load_aois(path: Path | str = DEFAULT_CONFIG) -> dict[str, AOI]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    aois = {}
    for name, raw in (doc.get("aois") or {}).items():
        role = raw.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"{name}: role must be one of {sorted(VALID_ROLES)}, got {role!r}")
        events = tuple(_parse_event(name, e) for e in (raw.get("events") or []))
        aois[name] = AOI(
            name=name,
            role=role,
            bbox=_parse_bbox(name, raw.get("bbox")),
            description=raw.get("description", ""),
            events=events,
        )
    return aois
