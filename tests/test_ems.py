import pytest
from shapely.geometry import box

from sarflood.ems import EmsImage, is_sentinel1, overlap_fractions, parse_activation

DOC = {"count": 1, "results": [{
    "code": "EMSR629", "name": "Floods in Pakistan", "eventTime": "2022-08-25T19:00:00",
    "activationTime": "2022-08-29T10:50:00", "countries": ["Pakistan"],
    "aois": [
        {"number": 2, "name": "Shikarpur", "extent": "POLYGON((68.4 27.8,68.8 27.8,68.8 28.1,68.4 28.1,68.4 27.8))",
         "products": []},
        {"number": 1, "name": "Larkana",
         "extent": "SRID=4326;POLYGON((68.0 27.3,68.4 27.3,68.4 27.7,68.0 27.7,68.0 27.3))",
         "products": [{"type": "DEL", "version": {"number": 2, "deliveryTime": "2022-08-30T17:34:32"},
                       "downloadPath": "https://example/EMSR629_AOI01_DEL.zip",
                       "images": [{"sensorType": "sar", "sensorName": "Sentinel-1A",
                                   "acquisitionTime": "2022-08-30T01:25:51", "fileName": None}]}]},
    ]}]}


def test_parse_activation_sorted_and_fields():
    meta, aois = parse_activation(DOC)
    assert meta["code"] == "EMSR629"
    assert [a.number for a in aois] == [1, 2]
    p = aois[0].products[0]
    assert (p.type, p.version, p.download) == ("DEL", 2, "https://example/EMSR629_AOI01_DEL.zip")
    assert aois[0].extent.bounds == (68.0, 27.3, 68.4, 27.7)   # SRID prefix stripped
    assert is_sentinel1(p.images[0])


def test_parse_activation_empty():
    with pytest.raises(ValueError, match="not found"):
        parse_activation({"results": []})


def test_overlap_fractions():
    sindh = box(67.5, 26.0, 68.5, 27.5)
    larkana = box(68.0, 27.3, 68.4, 27.7)          # half of it below 27.5
    in_ours, covers = overlap_fractions(larkana, sindh)
    assert in_ours == pytest.approx(0.5)
    assert covers == pytest.approx(0.4 * 0.2 / 1.5)
    assert overlap_fractions(box(70, 30, 71, 31), sindh) == (0.0, 0.0)


@pytest.mark.parametrize("name,fn,want", [
    ("Sentinel-1B", None, True), ("ICEYE", None, False), (None, "S1A_IW_GRDH_x", True),
    ("PlanetScope", "ps_scene.tif", False),
])
def test_is_sentinel1(name, fn, want):
    assert is_sentinel1(EmsImage("sar", name, None, fn)) is want


def test_cli_ems_info(tmp_path, monkeypatch, capsys):
    from sarflood import cli, ems

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(ems, "fetch_activation", lambda code: DOC)
    assert cli.main(["ems-info", "--code", "EMSR629"]) == 0
    out = capsys.readouterr().out
    assert "AOI01 Larkana" in out and "overlaps sindh" in out and "<-- Sentinel-1" in out
    assert (tmp_path / "ems" / "EMSR629.json").exists()
