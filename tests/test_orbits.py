from datetime import UTC, datetime

from sarflood.orbits import build_orbit_filter, find_orbit, pass_window, snap_orbit_dir

FLOOD = ["S1A_IW_GRDH_1SDV_20220830T012551_20220830T012616_044775_0558C6_E8B2.SAFE",
         "S1A_IW_GRDH_1SDV_20220830T012616_20220830T012641_044775_0558C6_CC82.SAFE"]
REAL = "S1A_OPER_AUX_POEORB_OPOD_20220919T081753_V20220829T225942_20220831T005942.EOF"


def test_pass_window_spans_all_slices():
    s, e = pass_window(list(reversed(FLOOD)))
    assert s == datetime(2022, 8, 30, 1, 25, 51, tzinfo=UTC)
    assert e == datetime(2022, 8, 30, 1, 26, 41, tzinfo=UTC)


def test_filter_has_margin_and_platform():
    s, e = pass_window(FLOOD)
    f = build_orbit_filter("S1A", s, e)
    assert "Value eq 'AUX_POEORB'" in f
    assert "startswith(Name,'S1A')" in f
    assert "ContentDate/Start lt 2022-08-30T01:24:51.000Z" in f  # 1 min before
    assert "ContentDate/End gt 2022-08-30T01:27:41.000Z" in f    # 1 min after


def test_snap_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SNAP_AUXDATA", str(tmp_path))
    d = snap_orbit_dir("S1A", datetime(2022, 8, 30, tzinfo=UTC))
    assert d == tmp_path / "Orbits" / "Sentinel-1" / "POEORB" / "S1A" / "2022" / "08"


class Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code, self.text = body, status, str(body)

    def json(self):
        return self._b


class Sess:
    def __init__(self, body):
        self.body = body

    def get(self, url, params=None, timeout=None):
        return Resp(self.body)


def product(name, pid):
    return {"Id": pid, "Name": name, "ContentLength": 4409612,
            "ContentDate": {"Start": "2022-08-29T22:59:42.000000Z",
                            "End": "2022-08-31T00:59:42.000000Z"}}


def test_find_orbit_picks_newest_production():
    older = REAL.replace("20220919T081753", "20220918T000000")
    o = find_orbit("S1A", *pass_window(FLOOD),
                   session=Sess({"value": [product(older, "old"), product(REAL, "new")]}))
    assert o.id == "new" and o.production == "20220919T081753"
    assert o.valid_start == datetime(2022, 8, 29, 22, 59, 42, tzinfo=UTC)


def test_find_orbit_none():
    assert find_orbit("S1A", *pass_window(FLOOD), session=Sess({"value": []})) is None


def test_download_keeps_eof_name(tmp_path):
    from sarflood.download import download_product
    from tests.test_download import DownloadSession, make_auth
    from tests.test_download import Resp as DResp

    auth, _, _ = make_auth()
    sess = DownloadSession([DResp(200, content=b"<Earth_Explorer_File/>")])
    out = download_product("x", REAL, tmp_path, auth, expected_size=22, session=sess, out_name=REAL)
    assert out.name == REAL and out.read_bytes().startswith(b"<Earth")
