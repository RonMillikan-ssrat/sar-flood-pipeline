"""Copernicus EMS Rapid Mapping: activation metadata, AOI overlap, product imagery.

Public JSON API, no key:
    https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code=EMSR629

Response (as documented by IFRC's API notes, to be checked against a live response):
    results[0].aois[i]  -> name, number, extent (WKT POLYGON), products[]
    product             -> type (REF/FEP/DEL/GRA), version{number, deliveryTime},
                           images[]{sensorType, sensorName, acquisitionTime, fileName},
                           downloadPath (zip), extent (WKT)

Why check before using EMS as labels:
- An EMS AOI may not overlap ours at all; they map where they were asked to.
- If a product was made from the same Sentinel-1 pass we process, scoring against
  it is partly circular. The image list says which data they used.

Attribution: Copernicus Emergency Management Service (c) European Union.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests
from shapely import wkt as shp_wkt
from shapely.geometry.base import BaseGeometry

API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"


@dataclass
class EmsImage:
    sensor_type: str | None
    sensor_name: str | None
    acquisition_time: str | None
    file_name: str | None


@dataclass
class EmsProduct:
    type: str
    version: int | None
    delivery_time: str | None
    download: str | None
    images: list[EmsImage] = field(default_factory=list)


@dataclass
class EmsAoi:
    number: int
    name: str
    extent: BaseGeometry | None
    products: list[EmsProduct] = field(default_factory=list)


def _geom(raw) -> BaseGeometry | None:
    if not raw:
        return None
    s = str(raw)
    if s.upper().startswith("SRID="):
        s = s.split(";", 1)[1]
    return shp_wkt.loads(s)


def parse_activation(doc: dict) -> tuple[dict, list[EmsAoi]]:
    results = doc.get("results") or []
    if not results:
        raise ValueError("activation not found (empty results)")
    act = results[0]
    aois = []
    for a in act.get("aois") or []:
        prods = []
        for p in a.get("products") or []:
            ver = p.get("version") or {}
            prods.append(EmsProduct(
                type=p.get("type", "?"),
                version=ver.get("number"),
                delivery_time=ver.get("deliveryTime"),
                download=p.get("downloadPath"),
                images=[EmsImage(i.get("sensorType"), i.get("sensorName"),
                                 i.get("acquisitionTime"), i.get("fileName"))
                        for i in p.get("images") or []],
            ))
        aois.append(EmsAoi(number=int(a.get("number", 0)), name=a.get("name", ""),
                           extent=_geom(a.get("extent")), products=prods))
    meta = {k: act.get(k) for k in ("code", "name", "eventTime", "activationTime", "countries")}
    return meta, sorted(aois, key=lambda x: x.number)


def fetch_activation(code: str, session: requests.Session | None = None) -> dict:
    sess = session or requests.Session()
    r = sess.get(API, params={"code": code}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"EMS API: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def overlap_fractions(ems_extent: BaseGeometry, aoi_geom: BaseGeometry) -> tuple[float, float]:
    """(share of the EMS AOI inside ours, share of our AOI covered by EMS), in degrees²."""
    if ems_extent is None or ems_extent.area == 0 or aoi_geom.area == 0:
        return 0.0, 0.0
    inter = ems_extent.intersection(aoi_geom).area
    return inter / ems_extent.area, inter / aoi_geom.area


S1_TOKENS = ("sentinel-1", "sentinel 1", "sentinel1", "s1a", "s1b", "s1c", "s1d")


def is_sentinel1(img: EmsImage) -> bool:
    text = f"{img.sensor_name or ''} {img.file_name or ''}".lower()
    return any(t in text for t in S1_TOKENS)
