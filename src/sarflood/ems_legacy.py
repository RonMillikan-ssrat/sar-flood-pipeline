"""Read the older (2017-era) EMS Rapid Mapping vector package layout.

Older packages (e.g. EMSR229 Hurricane Harvey) hold ESRI shapefiles only, with no
GeoJSON copies, no `source` table and different layer names. Layout as found in
EMSR229 01HOUSTON 01DELINEATION v2 (and MONIT01/02):

    ..._area_of_interest.shp          AOI outline; name, act_id
    ..._crisis_information_poly.shp   flood polygons; interpret 'Flooded Area',
                                      subtype 'EM009 - Flood'; src_info / src_date
                                      name the imagery the flood was mapped from
    ..._general_information_poly.shp  subtype 'Satellite Footprint' (source_nam =
                                      the image) and 'ZD020 - Void Collection Area'
                                      (reliabilit 'Missing data' = not assessed)
    ..._hydrography_poly.shp          subtype 'BH080 - Lake', 'BA040 - Open Water'
    ..._hydrography_line.shp          rivers as lines: no area, not used

Members may sit at the zip root or under VECTOR/. Each layer's .prj gives its CRS;
features are returned as GeoJSON-like dicts reprojected to EPSG:4326, the same
form `labels.read_layer` returns for the newer packages, so the rasterizing and
scoring code is shared.
"""

from __future__ import annotations

import io
import re
import zipfile


def is_legacy(zf: zipfile.ZipFile) -> bool:
    names = [n.lower() for n in zf.namelist()]
    return (any(n.endswith("_crisis_information_poly.shp") for n in names)
            and not any("_observedeventa_" in n for n in names))


def _members(zf: zipfile.ZipFile, tag: str) -> dict[str, str]:
    """{'.shp': name, '.shx': ..., '.dbf': ..., '.prj': ...} for one layer."""
    out = {}
    for n in zf.namelist():
        low = n.lower()
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            if low.endswith(f"_{tag}{ext}"):
                out.setdefault(ext, n)
    return out


def read_shapes(zf: zipfile.ZipFile, tag: str) -> list[dict]:
    """Features of one layer as {'type','properties','geometry'} in EPSG:4326."""
    import shapefile  # pyshp

    m = _members(zf, tag)
    if ".shp" not in m or ".dbf" not in m:
        return []
    r = shapefile.Reader(shp=io.BytesIO(zf.read(m[".shp"])),
                         shx=io.BytesIO(zf.read(m[".shx"])) if ".shx" in m else None,
                         dbf=io.BytesIO(zf.read(m[".dbf"])),
                         encoding="utf-8", encodingErrors="replace")
    feats = [{"type": "Feature", "properties": sr.record.as_dict(),
              "geometry": sr.shape.__geo_interface__}
             for sr in r.iterShapeRecords() if sr.shape.shapeType != shapefile.NULL]
    if ".prj" in m and feats:
        from rasterio.crs import CRS
        from rasterio.warp import transform_geom

        src = CRS.from_wkt(zf.read(m[".prj"]).decode("utf-8", "replace"))
        if src != CRS.from_epsg(4326):
            for f in feats:
                f["geometry"] = transform_geom(src, "EPSG:4326", f["geometry"])
    return feats


def _s(props: dict, key: str) -> str:
    return str(props.get(key, "")).strip()


def label_layers(zf: zipfile.ZipFile, permanent: tuple[str, ...]) -> dict[str, list[dict]]:
    crisis = read_shapes(zf, "crisis_information_poly")
    flood = [f for f in crisis
             if "flood" in (_s(f["properties"], "interpret") or _s(f["properties"], "subtype")).lower()]
    general = read_shapes(zf, "general_information_poly")
    void = [f for f in general if _s(f["properties"], "subtype").startswith("ZD020")]
    hydro = read_shapes(zf, "hydrography_poly")
    perm = [f for f in hydro if _s(f["properties"], "subtype").replace(" ", "").startswith(permanent)]
    return {"flood": flood, "aoi": read_shapes(zf, "area_of_interest"),
            "not_analysed": void, "perm_water": perm}


def aoi_info(zf: zipfile.ZipFile, zip_name: str) -> dict:
    """Outline features plus the metadata `ems_zip.aoi_from_zip` reports."""
    aoi = read_shapes(zf, "area_of_interest")
    if not aoi:
        raise ValueError(f"{zip_name}: no area_of_interest layer")
    p = aoi[0]["properties"]
    m = re.search(r"EMSR(\d+)_(\d+)([A-Z]+)", zip_name.upper())
    srcs: list[dict] = []
    for f in read_shapes(zf, "general_information_poly"):
        q = f["properties"]
        if _s(q, "subtype") == "Satellite Footprint":
            srcs.append({"source_nam": _s(q, "source_nam"), "src_date": _s(q, "src_date"),
                         "source_tm": ""})
    if not srcs:  # fall back to what the flood polygons say
        seen = set()
        for f in read_shapes(zf, "crisis_information_poly"):
            k = (_s(f["properties"], "src_info"), _s(f["properties"], "src_date"))
            if k not in seen:
                seen.add(k)
                srcs.append({"source_nam": k[0], "src_date": k[1], "source_tm": ""})
    return {"features": aoi, "emsr_id": _s(p, "act_id").replace("-", "") or (m and f"EMSR{m[1]}"),
            "area_id": m[2] if m else "", "locality": _s(p, "name") or (m[3].title() if m else ""),
            "flood_sources": srcs}
