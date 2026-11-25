import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sarflood.snap import build_tc_graph, utm_epsg

WKT = "POLYGON((67.5 26.0,68.5 26.0,68.5 27.5,67.5 27.5,67.5 26.0))"


@pytest.mark.parametrize("lon,lat,epsg", [
    (68.0, 26.75, 32642),    # Sindh
    (89.7, 24.65, 32645),    # Jamuna
    (-95.4, 29.8, 32615),    # Houston
    (-96.85, 47.35, 32614),  # Red River
    (18.4, -33.9, 32734),    # southern hemisphere
    (180.0, 0.0, 32660),     # antimeridian edge stays in zone 60
])
def test_utm_epsg(lon, lat, epsg):
    assert utm_epsg(lon, lat) == epsg


def nodes(xml):
    root = ET.fromstring(xml)
    return {n.get("id"): n for n in root.findall("node")}


def chain_order(xml):
    """Follow first sources back from Write to a Read."""
    ns, order, cur = nodes(xml), [], "Write"
    while cur:
        order.append(cur)
        src = ns[cur].find("sources")
        cur = src[0].get("refid") if src is not None and len(src) else None
    return order[::-1]


def test_single_slice_has_no_slice_assembly():
    g = build_tc_graph([Path("/d/a.zip")], Path("/o/x"), WKT, 32642)
    assert "SliceAssembly" not in nodes(g)
    assert chain_order(g) == ["Read1", "ThermalNoiseRemoval1", "Apply-Orbit-File", "Calibration",
                              "Terrain-Correction", "Subset", "Write"]


def test_three_slices_denoised_before_slice_assembly():
    zips = [Path(f"/d/s{i}.zip") for i in (1, 2, 3)]
    g = build_tc_graph(zips, Path("/o/x"), WKT, 32642)
    ns = nodes(g)
    srcs = list(ns["SliceAssembly"].find("sources"))
    assert [s.tag for s in srcs] == ["sourceProduct", "sourceProduct.1", "sourceProduct.2"]
    assert [s.get("refid") for s in srcs] == [f"ThermalNoiseRemoval{i}" for i in (1, 2, 3)]
    for i in (1, 2, 3):  # each TNR reads its own slice
        assert ns[f"ThermalNoiseRemoval{i}"].find("sources")[0].get("refid") == f"Read{i}"
    assert [ns[f"Read{i}"].findtext("parameters/file") for i in (1, 2, 3)] == [str(z) for z in zips]
    assert chain_order(g)[:4] == ["Read1", "ThermalNoiseRemoval1", "SliceAssembly",
                                  "Apply-Orbit-File"]
    assert "ThermalNoiseRemoval" not in ns  # no TNR after assembly


def test_key_parameters():
    g = build_tc_graph([Path("/d/a.zip"), Path("/d/b.zip")], Path("/o/x"), WKT, 32642, spacing_m=10)
    ns = nodes(g)
    for i in (1, 2):
        tnr = ns[f"ThermalNoiseRemoval{i}"].find("parameters")
        assert tnr.findtext("clipNegativeValues") == "false"  # default: no clipping before TC
    tc = ns["Terrain-Correction"].find("parameters")
    assert tc.findtext("mapProjection") == "EPSG:32642"
    assert tc.findtext("pixelSpacingInMeter") == "10"
    assert tc.findtext("alignToStandardGrid") == "true"
    assert tc.findtext("saveLayoverShadowMask") == "true"
    assert ns["Subset"].findtext("parameters/geoRegion") == WKT
    assert ns["Write"].findtext("parameters/formatName") == "GeoTIFF-BigTIFF"


def test_paths_are_xml_escaped():
    g = build_tc_graph([Path("/d/a&b.zip")], Path("/o/x"), WKT, 32642)
    assert nodes(g)["Read1"].findtext("parameters/file") == "/d/a&b.zip"


def test_cli_dry_run_writes_graphs(tmp_path, monkeypatch, capsys):
    import json

    from sarflood import cli

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = tmp_path / "raw" / "sindh"
    raw.mkdir(parents=True)
    flood = ["F1.SAFE", "F2.SAFE"]
    ref = ["R1.SAFE", "R2.SAFE"]
    for n in flood + ref:
        (raw / n.replace(".SAFE", ".zip")).write_bytes(b"x")
    pf = tmp_path / "pairs" / "sindh" / "pakistan-2022.json"
    pf.parent.mkdir(parents=True)
    pf.write_text(json.dumps({"pairs": [{
        "flood": {"key": "S1A_044775_0558C6", "slices": flood, "ids": ["a", "b"]},
        "reference": {"key": "S1A_043375_052E02", "slices": ref, "ids": ["c", "d"]},
        "reason": "ok"}]}))
    rc = cli.main(["terrain-correct", "--aoi", "sindh", "--event", "pakistan-2022",
                   "--flood-pass", "S1A_044775_0558C6", "--dry-run"])
    assert rc == 0
    graphs = sorted((tmp_path / "tc" / "sindh").glob("*.graph.xml"))
    assert [g.name for g in graphs] == [
        "S1A_043375_052E02_tc_20m_noclip_epsg32642.graph.xml",
        "S1A_044775_0558C6_tc_20m_noclip_epsg32642.graph.xml",
    ]
    assert "EPSG:32642" in capsys.readouterr().out


def test_cli_missing_slice(tmp_path, monkeypatch, capsys):
    import json

    from sarflood import cli

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    pf = tmp_path / "pairs" / "sindh" / "pakistan-2022.json"
    pf.parent.mkdir(parents=True)
    pf.write_text(json.dumps({"pairs": [{
        "flood": {"key": "K", "slices": ["F1.SAFE"], "ids": ["a"]},
        "reference": None, "reason": "none"}]}))
    rc = cli.main(["terrain-correct", "--aoi", "sindh", "--event", "pakistan-2022",
                   "--flood-pass", "K", "--role", "flood", "--dry-run"])
    assert rc == 2 and "missing" in capsys.readouterr().err
