"""CDSE access tokens (OpenID Connect password grant + refresh token).

Credentials come from the environment or ~/.netrc. They are never logged or
written anywhere by this module:

    export CDSE_USERNAME=you@example.com CDSE_PASSWORD=...
or, in ~/.netrc (chmod 600):
    machine identity.dataspace.copernicus.eu login you@example.com password ...

Reference: https://documentation.dataspace.copernicus.eu/APIs/OData.html (download section)
"""

from __future__ import annotations

import netrc
import os
import time

import requests

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
TOKEN_HOST = "identity.dataspace.copernicus.eu"
CLIENT_ID = "cdse-public"
EXPIRY_MARGIN_S = 60  # refresh this long before the server-stated expiry


def credentials() -> tuple[str, str]:
    user, pw = os.environ.get("CDSE_USERNAME"), os.environ.get("CDSE_PASSWORD")
    if user and pw:
        return user, pw
    try:
        entry = netrc.netrc().authenticators(TOKEN_HOST)
    except (FileNotFoundError, netrc.NetrcParseError):
        entry = None
    if entry and entry[0] and entry[2]:
        return entry[0], entry[2]
    raise RuntimeError(
        "No CDSE credentials: set CDSE_USERNAME/CDSE_PASSWORD or add "
        f"'machine {TOKEN_HOST}' to ~/.netrc"
    )


class CDSEAuth:
    """Hands out a valid access token, refreshing or re-logging in as needed."""

    def __init__(self, username: str | None = None, password: str | None = None,
                 session: requests.Session | None = None, clock=time.monotonic):
        if username is None or password is None:
            username, password = credentials()
        self._user, self._pw = username, password
        self._sess = session or requests.Session()
        self._clock = clock
        self._access: str | None = None
        self._refresh: str | None = None
        self._access_exp = 0.0
        self._refresh_exp = 0.0

    def _store(self, body: dict) -> None:
        now = self._clock()
        self._access = body["access_token"]
        # expires_in / refresh_expires_in are standard OIDC fields; fall back to
        # conservative guesses if the server omits them.
        self._access_exp = now + float(body.get("expires_in", 300)) - EXPIRY_MARGIN_S
        self._refresh = body.get("refresh_token")
        self._refresh_exp = now + float(body.get("refresh_expires_in", 0)) - EXPIRY_MARGIN_S

    def _post(self, data: dict) -> dict:
        r = self._sess.post(TOKEN_URL, data={"client_id": CLIENT_ID, **data}, timeout=30)
        if r.status_code != 200:
            # Don't echo the request (it holds the password); the response body is safe.
            raise RuntimeError(f"CDSE token request failed: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def token(self) -> str:
        now = self._clock()
        if self._access and now < self._access_exp:
            return self._access
        if self._refresh and now < self._refresh_exp:
            try:
                self._store(self._post({"grant_type": "refresh_token",
                                        "refresh_token": self._refresh}))
                return self._access
            except RuntimeError:
                pass  # refresh rejected: fall through to a fresh login
        self._store(self._post({"grant_type": "password",
                                "username": self._user, "password": self._pw}))
        return self._access

    def invalidate(self) -> None:
        """Force the next token() call to refresh (used after an HTTP 401)."""
        self._access_exp = 0.0
