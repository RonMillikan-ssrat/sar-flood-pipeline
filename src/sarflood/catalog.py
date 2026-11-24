"""Query the Copernicus Data Space Ecosystem (CDSE) OData catalog for Sentinel-1 scenes.

Searching the catalog does not need an access token; downloading products does
(that comes in a later step).

Reference: https://documentation.dataspace.copernicus.eu/APIs/OData.html
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import requests

log = logging.getLogger(__name__)

ODATA_PRODUCTS_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DEFAULT_PRODUCT_TYPE = "IW_GRDH_1S"  # Interferometric Wide swath, GRD high-res
MAX_TOP = 1000  # API limit for $top
DEFAULT_TIMEOUT_S = 60


@dataclass(frozen=True)
class Scene:
    id: str
    name: str
    platform: str  # S1A / S1B / S1C / S1D, taken from the product name
    polarisation: str  # e.g. "DV" = VV+VH, "SV" = VV only
    start: str
    end: str
    orbit_direction: str | None
    relative_orbit: int | None
    online: bool | None
    size_bytes: int | None
    s3_path: str | None
    footprint: str | None
    published: str | None = None  # PublicationDate: newer = reprocessed copy

    def to_dict(self) -> dict:
        return asdict(self)


def _iso_z(dt: datetime) -> str:
    """Format as the API expects: 2022-05-20T00:00:00.000Z (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def build_filter(wkt: str, start: datetime, end: datetime,
                 product_type: str = DEFAULT_PRODUCT_TYPE) -> str:
    """OData $filter for Sentinel-1 scenes of one product type intersecting `wkt`
    with sensing start in the half-open window [start, end)."""
    if end <= start:
        raise ValueError("end must be after start")
    return " and ".join([
        "Collection/Name eq 'SENTINEL-1'",
        (
            "Attributes/OData.CSC.StringAttribute/any("
            f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        ),
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
        f"ContentDate/Start ge {_iso_z(start)}",
        f"ContentDate/Start lt {_iso_z(end)}",
    ])


def _attr(product: dict, name: str):
    for a in product.get("Attributes") or []:
        if a.get("Name") == name:
            return a.get("Value")
    return None


def parse_scene(p: dict) -> Scene:
    name = p.get("Name", "")
    # Product names look like S1A_IW_GRDH_1SDV_20220828T005123_..., so the
    # platform is the first 3 characters and the polarisation code is 2 chars
    # after "_1S".
    platform = name[:3] if name.startswith("S1") else "?"
    pol = ""
    idx = name.find("_1S")
    if idx != -1:
        pol = name[idx + 3: idx + 5]
    content = p.get("ContentDate") or {}
    footprint = p.get("Footprint") or p.get("GeoFootprint")
    rel_orbit = _attr(p, "relativeOrbitNumber")
    return Scene(
        id=p["Id"],
        name=name,
        platform=platform,
        polarisation=pol,
        start=content.get("Start", ""),
        end=content.get("End", ""),
        orbit_direction=_attr(p, "orbitDirection"),
        relative_orbit=int(rel_orbit) if rel_orbit is not None else None,
        online=p.get("Online"),
        size_bytes=p.get("ContentLength"),
        s3_path=p.get("S3Path"),
        footprint=footprint if isinstance(footprint, str) else None,
        published=p.get("PublicationDate"),
    )


def _next_link(body: dict) -> str | None:
    # The docs show "@OData.nextLink"; the OData standard spelling is "@odata.nextLink".
    return body.get("@odata.nextLink") or body.get("@OData.nextLink")


def search_scenes(wkt: str, start: datetime, end: datetime,
                  product_type: str = DEFAULT_PRODUCT_TYPE,
                  page_size: int = 200, max_pages: int = 20,
                  session: requests.Session | None = None) -> list[Scene]:
    """Return every matching scene, following next links up to `max_pages`."""
    if not 1 <= page_size <= MAX_TOP:
        raise ValueError(f"page_size must be 1..{MAX_TOP}")
    sess = session or requests.Session()
    params = {
        "$filter": build_filter(wkt, start, end, product_type),
        "$orderby": "ContentDate/Start asc",
        "$top": str(page_size),
        "$expand": "Attributes",
    }
    url: str | None = ODATA_PRODUCTS_URL
    scenes: list[Scene] = []
    pages = 0
    while url:
        if pages >= max_pages:
            log.warning("Stopped after %d pages; narrow the date window", max_pages)
            break
        # The next link already carries the query string, so send params only on page 1.
        resp = sess.get(url, params=params if pages == 0 else None, timeout=DEFAULT_TIMEOUT_S)
        if resp.status_code != 200:
            raise RuntimeError(f"CDSE query failed: HTTP {resp.status_code}: {resp.text[:500]}")
        body = resp.json()
        batch = [parse_scene(p) for p in body.get("value", [])]
        scenes.extend(batch)
        pages += 1
        log.info("page %d: %d scenes (total %d)", pages, len(batch), len(scenes))
        url = _next_link(body)
    return scenes
