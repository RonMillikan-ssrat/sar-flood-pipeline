"""Fetch Sentinel-1 precise orbit files (AUX_POEORB) from CDSE into SNAP's orbit folder.

SNAP's Apply-Orbit-File normally downloads orbits from step.esa.int. When that
server is unreachable, SNAP looks for files already placed in
    ~/.snap/auxdata/Orbits/Sentinel-1/POEORB/<platform>/<YYYY>/<MM>/
(its own error message names this folder). This module puts them there from
CDSE, using the same token and downloader as the products.

Orbit file names encode their validity window:
    S1A_OPER_AUX_POEORB_OPOD_20220919T081753_V20220829T225942_20220831T005942.EOF
                             production        valid from       valid to
A POEORB file covers about 26 h, so one file covers any single pass. If several
match, the newest production wins, since a later file may supersede an earlier one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from .catalog import ODATA_PRODUCTS_URL, _iso_z

ORBIT_MARGIN = timedelta(minutes=1)  # validity must extend this far beyond the pass


@dataclass(frozen=True)
class OrbitFile:
    id: str
    name: str
    valid_start: datetime
    valid_end: datetime
    size: int | None

    @property
    def production(self) -> str:
        # S1A_OPER_AUX_POEORB_OPOD_<production>_V<start>_<end>.EOF
        return self.name.split("_")[5]


def snap_orbit_dir(platform: str, when: datetime, kind: str = "POEORB") -> Path:
    base = Path(os.environ.get("SNAP_AUXDATA", Path.home() / ".snap" / "auxdata"))
    return base / "Orbits" / "Sentinel-1" / kind / platform / f"{when:%Y}" / f"{when:%m}"


def pass_window(slice_names: list[str]) -> tuple[datetime, datetime]:
    """First start and last stop time from Sentinel-1 product names (fields 4 and 5)."""
    starts, stops = [], []
    for n in slice_names:
        parts = n.removesuffix(".SAFE").split("_")
        starts.append(datetime.strptime(parts[4], "%Y%m%dT%H%M%S").replace(tzinfo=UTC))
        stops.append(datetime.strptime(parts[5], "%Y%m%dT%H%M%S").replace(tzinfo=UTC))
    return min(starts), max(stops)


def build_orbit_filter(platform: str, start: datetime, end: datetime,
                       product_type: str = "AUX_POEORB") -> str:
    return " and ".join([
        "Collection/Name eq 'SENTINEL-1'",
        ("Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
         f"att/OData.CSC.StringAttribute/Value eq '{product_type}')"),
        f"startswith(Name,'{platform}')",
        f"ContentDate/Start lt {_iso_z(start - ORBIT_MARGIN)}",
        f"ContentDate/End gt {_iso_z(end + ORBIT_MARGIN)}",
    ])


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def find_orbit(platform: str, start: datetime, end: datetime,
               session: requests.Session | None = None) -> OrbitFile | None:
    sess = session or requests.Session()
    r = sess.get(ODATA_PRODUCTS_URL, timeout=60, params={
        "$filter": build_orbit_filter(platform, start, end), "$top": "20"})
    if r.status_code != 200:
        raise RuntimeError(f"orbit query failed: HTTP {r.status_code}: {r.text[:300]}")
    found = [
        OrbitFile(id=p["Id"], name=p["Name"], valid_start=_ts(p["ContentDate"]["Start"]),
                  valid_end=_ts(p["ContentDate"]["End"]), size=p.get("ContentLength"))
        for p in r.json().get("value", [])
    ]
    return max(found, key=lambda o: o.production) if found else None
