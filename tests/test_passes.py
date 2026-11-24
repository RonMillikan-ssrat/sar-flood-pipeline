from datetime import UTC, datetime, timedelta

import pytest
from shapely.geometry import box

from sarflood.aoi import load_aois
from sarflood.catalog import Scene
from sarflood.passes import (
    dedupe,
    group_passes,
    parse_footprint,
    parse_name,
    select_reference,
)

AOI = box(0, 0, 1, 1)
FULL = "geography'SRID=4326;POLYGON((-1 -1,2 -1,2 2,-1 2,-1 -1))'"  # covers AOI
HALF = "geography'SRID=4326;POLYGON((0.5 -1,2 -1,2 2,0.5 2,0.5 -1))'"  # covers 50%


def scene(start: str, *, plat="S1A", orbit=44775, dt="0558C6", pid="AAAA", rel=78,
          direction="DESCENDING", pol="DV", dur=25, footprint=FULL, published=None) -> Scene:
    t0 = datetime.fromisoformat(start).replace(tzinfo=UTC)
    t1 = t0 + timedelta(seconds=dur)
    name = (f"{plat}_IW_GRDH_1S{pol}_{t0:%Y%m%dT%H%M%S}_{t1:%Y%m%dT%H%M%S}"
            f"_{orbit:06d}_{dt}_{pid}.SAFE")
    return Scene(id=f"{name}-id", name=name, platform=plat, polarisation=pol,
                 start=t0.isoformat().replace("+00:00", "Z"),
                 end=t1.isoformat().replace("+00:00", "Z"),
                 orbit_direction=direction, relative_orbit=rel, online=True,
                 size_bytes=1, s3_path=None, footprint=footprint, published=published)


def d(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def test_parse_name_real_product():
    f = parse_name("S1A_IW_GRDH_1SDV_20220830T012551_20220830T012616_044775_0558C6_E8B2.SAFE")
    assert (f.platform, f.absolute_orbit, f.datatake, f.product_id) == ("S1A", 44775, "0558C6", "E8B2")
    assert f.slice_key.endswith("_044775_0558C6")


def test_parse_name_rejects_garbage():
    with pytest.raises(ValueError):
        parse_name("S2A_MSIL2A_20220830T053641_N0400_R005_T42RTN_20220830T082011.SAFE")


def test_parse_footprint_strips_wrapper():
    g = parse_footprint(FULL)
    assert g.contains(AOI)
    assert parse_footprint(None) is None


def test_dedupe_keeps_newest_publication():
    old = scene("2022-08-30T01:25:51", pid="AAAA", published="2022-08-30T05:00:00Z")
    new = scene("2022-08-30T01:25:51", pid="BBBB", published="2023-02-01T00:00:00Z")
    assert [s.name for s in dedupe([old, new])] == [new.name]


def test_two_contiguous_slices_make_one_pass():
    # Mirrors the live Sindh result: one datatake, 25 s slices back to back.
    a = scene("2022-08-30T01:25:51", pid="E8B2", footprint=HALF)
    b = scene("2022-08-30T01:26:16", pid="CC82",
              footprint="geography'SRID=4326;POLYGON((-1 -1,0.5 -1,0.5 2,-1 2,-1 -1))'")
    (p,) = group_passes([b, a])  # input order shouldn't matter
    assert p.key == "S1A_044775_0558C6"
    assert [s.name for s in p.slices] == [a.name, b.name]
    assert p.contiguous
    assert p.coverage(AOI) == pytest.approx(1.0)  # union of the two halves


def test_gap_between_slices_is_flagged():
    a = scene("2022-08-30T01:25:51")
    c = scene("2022-08-30T01:27:00", pid="CCCC")  # skips a slice
    (p,) = group_passes([a, c])
    assert not p.contiguous


def test_different_datatakes_are_different_passes():
    a = scene("2022-08-18T01:25:51", orbit=44600, dt="055000")
    b = scene("2022-08-30T01:25:51")
    assert len(group_passes([a, b])) == 2


def test_disagreeing_track_metadata_raises():
    a = scene("2022-08-30T01:25:51", rel=78)
    b = scene("2022-08-30T01:26:16", pid="CC82", rel=79)
    with pytest.raises(ValueError, match="disagree"):
        group_passes([a, b])


WIN = (d("2022-03-01"), d("2022-06-01"))


def test_reference_must_be_same_track():
    (flood,) = group_passes([scene("2022-08-30T01:25:51")])
    other = group_passes([scene("2022-05-01T13:00:00", orbit=43000, dt="050000",
                                rel=5, direction="ASCENDING")])
    pr = select_reference(flood, other, AOI, *WIN)
    assert pr.reference is None and "no candidate on track" in pr.reason


def test_reference_picks_latest_in_window():
    (flood,) = group_passes([scene("2022-08-30T01:25:51")])
    cands = group_passes([
        scene("2022-04-07T01:25:51", orbit=43375, dt="052D00"),
        scene("2022-05-25T01:25:51", orbit=44075, dt="054000"),
        scene("2022-07-12T01:25:51", orbit=44075 + 700, dt="056000"),  # outside window
    ])
    pr = select_reference(flood, cands, AOI, *WIN)
    assert pr.reference.start == d("2022-05-25T01:25:51")
    assert "97 days before" in pr.reason


def test_reference_needs_coverage():
    (flood,) = group_passes([scene("2022-08-30T01:25:51")])
    cands = group_passes([scene("2022-05-25T01:25:51", orbit=44075, dt="054000",
                                footprint=HALF)])
    pr = select_reference(flood, cands, AOI, *WIN, min_coverage=0.8)
    assert pr.reference is None and "0.50" in pr.reason
    assert select_reference(flood, cands, AOI, *WIN, min_coverage=0.4).reference is not None


def test_cross_platform_allowed_unless_requested():
    # S1A and S1B on the same relative orbit share geometry, so by default either is OK.
    (flood,) = group_passes([scene("2021-08-30T01:25:51", orbit=39000, dt="049000")])
    cands = group_passes([scene("2021-05-25T01:26:00", plat="S1B", orbit=27000, dt="033000")])
    win = (d("2021-03-01"), d("2021-06-01"))
    assert select_reference(flood, cands, AOI, *win).reference is not None
    pr = select_reference(flood, cands, AOI, *win, same_platform=True)
    assert pr.reference is None and "same-platform" in pr.reason


def test_config_reference_windows_precede_events():
    for a in load_aois().values():
        for e in a.events:
            if e.reference_start:
                assert e.reference_end <= e.start


def test_cli_pairs_end_to_end(tmp_path, monkeypatch, capsys):
    import json

    from sarflood import cli

    sindh = "geography'SRID=4326;POLYGON((66 25,70 25,70 29,66 29,66 25))'"
    flood = [scene("2022-08-30T01:25:51", pid="E8B2", footprint=sindh),
             scene("2022-08-30T01:26:16", pid="CC82", footprint=sindh)]
    refs = [scene("2022-05-25T01:25:51", orbit=44075, dt="054000", footprint=sindh)]

    def fake_search(wkt, start, end, product_type):
        return flood if start.month == 8 else refs

    monkeypatch.setattr(cli, "search_scenes", fake_search)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert cli.main(["pairs", "--aoi", "sindh", "--event", "pakistan-2022"]) == 0
    out = capsys.readouterr().out
    assert "1/1 flood passes paired" in out
    doc = json.loads((tmp_path / "pairs" / "sindh" / "pakistan-2022.json").read_text())
    assert doc["pairs"][0]["flood"]["n_slices"] == 2
    assert doc["pairs"][0]["reference"]["key"] == "S1A_044075_054000"
