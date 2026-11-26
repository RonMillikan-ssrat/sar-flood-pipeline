"""Read metadata from a Copernicus EMS Rapid Mapping vector package (.zip).

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


def aoi_from_zip(path: Path, margin: float = 0.01) -> dict:
    with zipfile.ZipFile(path) as zf:
        name = _member(zf, "areaOfInterestA", ".json")
        if name is None:
            raise ValueError(f"{path.name}: no areaOfInterestA GeoJSON")
        feats = json.loads(zf.read(name))["features"]
        recs = source_records(zf)
        used = flood_source_ids(zf)
    geom = unary_union([shape(f["geometry"]) for f in feats])
    x0, y0, x1, y1 = geom.bounds
    # Round outward to 2 decimals after adding the margin, so the box contains the AOI.
    # round(..., 6) first: 68.29 * 100 is 6828.999... in binary floating point.
    def down(v):
        return math.floor(round(v * 100, 6)) / 100

    def up(v):
        return math.ceil(round(v * 100, 6)) / 100

    bbox = [down(x0 - margin), down(y0 - margin), up(x1 + margin), up(y1 + margin)]
    props = feats[0]["properties"]
    flood_src = [r for r in recs if str(r.get("src_id")).strip() in used]
    if not flood_src:  # no dmg_src_id: fall back to the eventphase tag
        flood_src = [r for r in recs if str(r.get("eventphase", "")).lower().startswith("post")]
    return {"bounds": geom.bounds, "bbox": bbox, "emsr_id": props.get("emsr_id"),
            "area_id": props.get("area_id"), "locality": props.get("locality"),
            "flood_sources": flood_src, "zip": path.name}


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
