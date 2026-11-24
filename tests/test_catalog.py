import json
from datetime import UTC, datetime

import pytest

from sarflood.catalog import _iso_z, build_filter, parse_scene, search_scenes
from sarflood.manifest import manifest_path, write_manifest

WKT = "POLYGON((0 0,1 0,1 1,0 1,0 0))"

PRODUCT = {
    "Id": "abc-123",
    "Name": "S1C_IW_GRDH_1SDV_20250601T001122_20250601T001147_002345_004567_ABCD.SAFE",
    "ContentDate": {"Start": "2025-06-01T00:11:22.000Z", "End": "2025-06-01T00:11:47.000Z"},
    "Online": True,
    "ContentLength": 1700000000,
    "S3Path": "/eodata/Sentinel-1/SAR/IW_GRDH_1S/2025/06/01/x.SAFE",
    "Footprint": "geography'SRID=4326;POLYGON((0 0,1 0,1 1,0 1,0 0))'",
    "Attributes": [
        {"Name": "orbitDirection", "Value": "ASCENDING"},
        {"Name": "relativeOrbitNumber", "Value": 41},
    ],
}


def test_iso_z_format():
    assert _iso_z(datetime(2022, 5, 20, tzinfo=UTC)) == "2022-05-20T00:00:00.000Z"


def test_filter_contains_all_clauses():
    f = build_filter(WKT, datetime(2022, 8, 28, tzinfo=UTC), datetime(2022, 8, 29, tzinfo=UTC))
    assert "Collection/Name eq 'SENTINEL-1'" in f
    assert "Value eq 'IW_GRDH_1S'" in f
    assert f"SRID=4326;{WKT}" in f
    assert "ContentDate/Start ge 2022-08-28T00:00:00.000Z" in f
    assert "ContentDate/Start lt 2022-08-29T00:00:00.000Z" in f


def test_filter_rejects_empty_window():
    t = datetime(2022, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        build_filter(WKT, t, t)


def test_parse_scene():
    s = parse_scene(PRODUCT)
    assert (s.platform, s.polarisation) == ("S1C", "DV")
    assert (s.orbit_direction, s.relative_orbit) == ("ASCENDING", 41)
    assert s.size_bytes == 1700000000


def test_parse_scene_missing_attributes():
    s = parse_scene({"Id": "x", "Name": "S1A_IW_GRDH_1SSV_x"})
    assert s.orbit_direction is None and s.relative_orbit is None
    assert s.polarisation == "SV"


class FakeResp:
    def __init__(self, body, status=200):
        self._body, self.status_code, self.text = body, status, json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    """Returns canned pages and records what was requested."""

    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.pages.pop(0)


def test_search_follows_next_link_both_spellings():
    p2 = dict(PRODUCT, Id="def-456")
    p3 = dict(PRODUCT, Id="ghi-789")
    sess = FakeSession([
        FakeResp({"value": [PRODUCT], "@odata.nextLink": "https://next/1"}),
        FakeResp({"value": [p2], "@OData.nextLink": "https://next/2"}),
        FakeResp({"value": [p3]}),
    ])
    t0, t1 = datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 2, tzinfo=UTC)
    scenes = search_scenes(WKT, t0, t1, session=sess)
    assert [s.id for s in scenes] == ["abc-123", "def-456", "ghi-789"]
    assert sess.calls[0][1]["$top"] == "200"
    assert sess.calls[1] == ("https://next/1", None)  # no params on later pages


def test_search_stops_at_max_pages():
    sess = FakeSession([FakeResp({"value": [PRODUCT], "@odata.nextLink": "u"})] * 5)
    t0, t1 = datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 2, tzinfo=UTC)
    assert len(search_scenes(WKT, t0, t1, max_pages=2, session=sess)) == 2


def test_search_raises_on_http_error():
    sess = FakeSession([FakeResp({"error": "bad filter"}, status=400)])
    t0, t1 = datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 2, tzinfo=UTC)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        search_scenes(WKT, t0, t1, session=sess)


def test_manifest_is_idempotent(tmp_path):
    t0, t1 = datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 2, tzinfo=UTC)
    s = [parse_scene(PRODUCT)]
    p1 = write_manifest("sylhet", t0, t1, s, "IW_GRDH_1S", root=tmp_path)
    p2 = write_manifest("sylhet", t0, t1, s, "IW_GRDH_1S", root=tmp_path)
    assert p1 == p2 == manifest_path("sylhet", t0, t1, tmp_path)
    assert len(list(p1.parent.iterdir())) == 1
    assert json.loads(p1.read_text())["count"] == 1
