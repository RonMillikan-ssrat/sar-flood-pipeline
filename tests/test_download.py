import pytest

from sarflood.auth import CDSEAuth, credentials
from sarflood.download import DOWNLOAD_URL, download_product


class Resp:
    def __init__(self, status=200, body=None, headers=None, content=b""):
        self.status_code, self._body = status, body or {}
        self.headers, self._content = headers or {}, content
        self.text = str(self._body)

    def json(self):
        return self._body

    def iter_content(self, n):
        for i in range(0, len(self._content), 3):  # small chunks to exercise the loop
            yield self._content[i:i + 3]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TokenSession:
    """Fake identity server: records grant types, hands out numbered tokens."""

    def __init__(self, fail_refresh=False):
        self.grants, self.n, self.fail_refresh = [], 0, fail_refresh

    def post(self, url, data=None, timeout=None):
        self.grants.append(data["grant_type"])
        if data["grant_type"] == "refresh_token" and self.fail_refresh:
            return Resp(400, {"error": "invalid_grant"})
        self.n += 1
        return Resp(200, {"access_token": f"tok{self.n}", "expires_in": 600,
                          "refresh_token": f"ref{self.n}", "refresh_expires_in": 3600})


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_auth(**kw):
    sess, clock = TokenSession(**kw), Clock()
    return CDSEAuth("u", "p", session=sess, clock=clock), sess, clock


def test_token_cached_until_near_expiry():
    auth, sess, clock = make_auth()
    assert auth.token() == "tok1"
    clock.t = 500  # inside 600 - 60 s margin
    assert auth.token() == "tok1"
    assert sess.grants == ["password"]


def test_expired_access_uses_refresh_token():
    auth, sess, clock = make_auth()
    auth.token()
    clock.t = 560  # past 600 - 60
    assert auth.token() == "tok2"
    assert sess.grants == ["password", "refresh_token"]


def test_rejected_refresh_falls_back_to_login():
    auth, sess, clock = make_auth(fail_refresh=True)
    auth.token()
    clock.t = 560
    assert auth.token() == "tok2"
    assert sess.grants == ["password", "refresh_token", "password"]


def test_credentials_from_env(monkeypatch):
    monkeypatch.setenv("CDSE_USERNAME", "me")
    monkeypatch.setenv("CDSE_PASSWORD", "secret")
    assert credentials() == ("me", "secret")


def test_missing_credentials_message(monkeypatch, tmp_path):
    monkeypatch.delenv("CDSE_USERNAME", raising=False)
    monkeypatch.delenv("CDSE_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.netrc
    with pytest.raises(RuntimeError, match="No CDSE credentials"):
        credentials()


class DownloadSession:
    """Scripted GET responses; records (url, auth header) for each call."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, headers=None, allow_redirects=True, stream=False, timeout=None):
        assert allow_redirects is False  # we must follow redirects ourselves
        self.calls.append((url, (headers or {}).get("Authorization")))
        return self.responses.pop(0)


DATA = b"0123456789abcdef"
NAME = "S1A_IW_GRDH_1SDV_20220830T012551_20220830T012616_044775_0558C6_E8B2.SAFE"


def test_redirect_keeps_token_and_writes_file(tmp_path):
    auth, _, _ = make_auth()
    sess = DownloadSession([
        Resp(302, headers={"Location": "https://zipper.dataspace.copernicus.eu/x/abc"}),
        Resp(200, content=DATA),
    ])
    out = download_product("abc", NAME, tmp_path, auth, expected_size=len(DATA), session=sess)
    assert out.name == NAME.removesuffix(".SAFE") + ".zip"
    assert out.read_bytes() == DATA
    assert sess.calls[0] == (DOWNLOAD_URL.format(id="abc"), "Bearer tok1")
    assert sess.calls[1] == ("https://zipper.dataspace.copernicus.eu/x/abc", "Bearer tok1")
    assert not list(tmp_path.glob("*.part"))


def test_refuses_to_send_token_off_copernicus(tmp_path):
    auth, _, _ = make_auth()
    sess = DownloadSession([Resp(302, headers={"Location": "https://evil.example.com/x"})])
    with pytest.raises(RuntimeError, match="untrusted host"):
        download_product("abc", NAME, tmp_path, auth, session=sess)
    assert len(sess.calls) == 1


def test_401_refreshes_once(tmp_path):
    auth, tsess, _ = make_auth()
    sess = DownloadSession([Resp(401), Resp(200, content=DATA)])
    download_product("abc", NAME, tmp_path, auth, expected_size=len(DATA), session=sess)
    assert [c[1] for c in sess.calls] == ["Bearer tok1", "Bearer tok2"]
    assert tsess.grants == ["password", "refresh_token"]


def test_size_mismatch_leaves_part_file(tmp_path):
    auth, _, _ = make_auth()
    sess = DownloadSession([Resp(200, content=DATA[:10])])
    with pytest.raises(RuntimeError, match="catalog says 16"):
        download_product("abc", NAME, tmp_path, auth, expected_size=len(DATA), session=sess)
    assert not list(tmp_path.glob("*.zip"))
    assert len(list(tmp_path.glob("*.zip.part"))) == 1


def test_complete_file_is_skipped(tmp_path):
    auth, _, _ = make_auth()
    (tmp_path / (NAME.removesuffix(".SAFE") + ".zip")).write_bytes(DATA)
    sess = DownloadSession([])  # any GET would fail
    download_product("abc", NAME, tmp_path, auth, expected_size=len(DATA), session=sess)
    assert sess.calls == []


def test_cli_download_dry_run(tmp_path, monkeypatch, capsys):
    import json

    from sarflood import cli, download

    pairs = {"pairs": [{
        "flood": {"key": "S1A_044775_0558C6", "ids": ["f1", "f2"], "slices": ["F1.SAFE", "F2.SAFE"]},
        "reference": {"key": "S1A_043375_052E02", "ids": ["r1", "r2"], "slices": ["R1.SAFE", "R2.SAFE"]},
        "reason": "ok",
    }]}
    f = tmp_path / "pairs" / "sindh" / "pakistan-2022.json"
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps(pairs))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(download, "product_size", lambda pid: 1_700_000_000)

    def no_auth(*a, **k):
        raise AssertionError("dry run must not authenticate")

    monkeypatch.setattr("sarflood.auth.CDSEAuth", no_auth)
    rc = cli.main(["download", "--aoi", "sindh", "--event", "pakistan-2022",
                   "--flood-pass", "S1A_044775_0558C6", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "4 products, 6.80 GB total" in out
    assert out.count("reference") == 2
