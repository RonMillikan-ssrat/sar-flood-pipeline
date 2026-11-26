"""Read metadata from a Copernicus EMS Rapid Mapping vector package (.zip).

Older shapefile-only packages (2017 era) are handled by `ems_legacy`.

Used to define AOIs from the packages themselves, since the EMS API requires a
login. Reads:
    areaOfInterestA (GeoJSON)  outline -> bounds; emsr_id, area_id, locality
    source (.dbf, via pyshp)   image records
    observedEventA (.dbf)      dmg_src_id per flood polygon -> which source record(s)
                               the polygons were actually traced from

The flood source is found through dmg_src_id, not the 'eventphase' tag: in
EMSR629 AOI03 the 30 Aug 2022 SPOT-6/7 image the flood was mapped from is tagged
'Pre-event', an error in the package metadata.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from datetime import date, datetime
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union


def _member(zf: zipfile.ZipFile, tag: str, ext: str) -> str | None:
    hits = [n for n in zf.namelist() if f"_{tag}_" in n and n.endswith(ext)]
    return hits[0] if hits else None


def source_records(zf: zipfile.ZipFile) -> list[dict]:
    name = _member(zf, "source", ".dbf")
    if name is None:
        return []
    import shapefile  # pyshp

    r = shapefile.Reader(dbf=io.BytesIO(zf.read(name)))
    return [rec.as_dict() for rec in r.records()]


def flood_source_ids(zf: zipfile.ZipFile) -> set[str]:
    """dmg_src_id values used by the flood polygons (observedEventA attribute table)."""
    name = _member(zf, "observedEventA", ".dbf")
    if name is None:
        return set()
    import shapefile  # pyshp

    r = shapefile.Reader(dbf=io.BytesIO(zf.read(name)))
    fields = [f[0] for f in r.fields[1:]]
    if "dmg_src_id" not in fields:
        return set()
    i = fields.index("dmg_src_id")
    return {str(rec[i]).strip() for rec in r.iterRecords()}


def parse_src_date(s: str) -> date | None:
    """EMS source dates: 'dd/mm/yyyy' (2022 packages), 'yyyy-mm-dd' (2017), or a bare year."""
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def source_warnings(flood_src: list[dict], recs: list[dict], max_gap_days: int = 60) -> list[str]:
    """Things about the flood source a person should check before trusting the labels."""
    out = []
    for r in flood_src:
        if "sentinel-1" in str(r.get("source_nam", "")).lower().replace(" ", "-"):
            out.append(f"flood source {r.get('source_nam')} is Sentinel-1: labels are not "
                       "independent of our input (circular)")
    # Imagery records carry an acquisition time; map/land-cover records don't.
    imgs = [(parse_src_date(r.get("src_date")), r) for r in recs
            if str(r.get("source_tm", "")).startswith("T")]
    imgs = [(d, r) for d, r in imgs if d]
    if imgs:
        latest, lr = max(imgs, key=lambda t: t[0])
        for r in flood_src:
            d = parse_src_date(r.get("src_date"))
            if d and (latest - d).days > max_gap_days:
                out.append(f"flood polygons point (dmg_src_id) to {r.get('source_nam')} "
                           f"{r.get('src_date')}, {(latest - d).days} days before the latest image "
                           f"({lr.get('source_nam')} {lr.get('src_date')} {lr.get('source_tm')}, "
                           f"{lr.get('eventphase')}); likely a metadata error, check the map PDF")
    return out


def aoi_from_zip(path: Path, margin: float = 0.01) -> dict:
    from . import ems_legacy

    with zipfile.ZipFile(path) as zf:
        if ems_legacy.is_legacy(zf):
            leg = ems_legacy.aoi_info(zf, path.name)
            feats, recs, used = leg["features"], leg["flood_sources"], None
        else:
            name = _member(zf, "areaOfInterestA", ".json")
            if name is None:
                raise ValueError(f"{path.name}: no areaOfInterestA GeoJSON")
            feats = json.loads(zf.read(name))["features"]
            recs = source_records(zf)
            used = flood_source_ids(zf)
            leg = None
    geom = unary_union([shape(f["geometry"]) for f in feats])
    x0, y0, x1, y1 = geom.bounds
    # Round outward to 2 decimals after adding the margin, so the box contains the AOI.
    # round(..., 6) first: 68.29 * 100 is 6828.999... in binary floating point.
    def down(v):
        return math.floor(round(v * 100, 6)) / 100

    def up(v):
        return math.ceil(round(v * 100, 6)) / 100

    bbox = [down(x0 - margin), down(y0 - margin), up(x1 + margin), up(y1 + margin)]
    if leg is not None:
        return {"bounds": geom.bounds, "bbox": bbox, "emsr_id": leg["emsr_id"],
                "area_id": leg["area_id"], "locality": leg["locality"],
                "flood_sources": recs, "warnings": source_warnings(recs, []),
                "zip": path.name}
    props = feats[0]["properties"]
    flood_src = [r for r in recs if str(r.get("src_id")).strip() in used]
    if not flood_src:  # no dmg_src_id: fall back to the eventphase tag
        flood_src = [r for r in recs if str(r.get("eventphase", "")).lower().startswith("post")]
    return {"bounds": geom.bounds, "bbox": bbox, "emsr_id": props.get("emsr_id"),
            "area_id": props.get("area_id"), "locality": props.get("locality"),
            "flood_sources": flood_src, "warnings": source_warnings(flood_src, recs),
            "zip": path.name}


def yaml_entry(name: str, info: dict, like, role: str = "train") -> str:
    """Config text for a new AOI, copying event and reference windows from `like`."""
    src = "; ".join(f"{r.get('source_nam')} {r.get('src_date')} {r.get('source_tm')}".strip()
                    for r in info["flood_sources"]) or "flood source unknown"
    desc = f"{info['emsr_id']} AOI{info['area_id']} {info['locality']} (labels = EMS DEL from {src})"
    lines = ["", f"  {name}:", f"    role: {role}", f"    description: {desc}",
             f"    bbox: [{', '.join(f'{v:.2f}' for v in info['bbox'])}]", "    events:"]
    for e in like.events:
        lines += [f"      - name: {e.name}", f'        start: "{e.start}"', f'        end: "{e.end}"',
                  f'        notes: "{info["zip"]}"']
        if e.reference_start:
            lines.append(f'        reference: {{start: "{e.reference_start}", '
                         f'end: "{e.reference_end}"}}  # copied from {like.name}; verify')
    return "\n".join(lines) + "\n"
