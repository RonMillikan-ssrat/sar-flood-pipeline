from datetime import date

import pytest

from sarflood.aoi import AOI, load_aois


def test_repo_config_loads():
    aois = load_aois()
    assert {"sindh", "houston"} <= set(aois)
    assert {a.role for a in aois.values()} == {"train", "holdout"}


def test_wkt_is_closed_ring():
    a = AOI(name="x", role="train", bbox=(1.0, 2.0, 3.0, 4.0))
    wkt = a.to_wkt()
    assert wkt == "POLYGON((1.0 2.0,3.0 2.0,3.0 4.0,1.0 4.0,1.0 2.0))"


def _write(tmp_path, body: str):
    p = tmp_path / "aois.yaml"
    p.write_text(body)
    return p


def test_rejects_bad_role(tmp_path):
    p = _write(tmp_path, "aois:\n  a:\n    role: test\n    bbox: [0, 0, 1, 1]\n")
    with pytest.raises(ValueError, match="role"):
        load_aois(p)


def test_rejects_inverted_bbox(tmp_path):
    p = _write(tmp_path, "aois:\n  a:\n    role: train\n    bbox: [5, 0, 1, 1]\n")
    with pytest.raises(ValueError, match="longitude"):
        load_aois(p)


def test_parses_events(tmp_path):
    p = _write(tmp_path, (
        "aois:\n  a:\n    role: holdout\n    bbox: [0, 0, 1, 1]\n"
        "    events:\n      - {name: e1, start: '2020-01-01', end: '2020-01-05'}\n"
    ))
    ev = load_aois(p)["a"].events[0]
    assert (ev.start, ev.end) == (date(2020, 1, 1), date(2020, 1, 5))


def test_rejects_backwards_event(tmp_path):
    p = _write(tmp_path, (
        "aois:\n  a:\n    role: train\n    bbox: [0, 0, 1, 1]\n"
        "    events:\n      - {name: e1, start: '2020-01-05', end: '2020-01-01'}\n"
    ))
    with pytest.raises(ValueError, match="end must be after start"):
        load_aois(p)
