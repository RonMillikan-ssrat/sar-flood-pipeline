"""Download Sentinel-1 products (zipped .SAFE) from CDSE.

Mechanics worth knowing:
- The download URL redirects. requests drops the Authorization header when a
  redirect changes host, so redirects are followed here by hand, and the token
  is re-sent only to *.copernicus.eu hosts so it can't leak elsewhere.
- Files stream to <name>.zip.part and are renamed only after the byte count
  matches the catalog's ContentLength. A half-finished file never looks complete.
- A finished file of the right size is skipped, so reruns are cheap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from .auth import CDSEAuth
from .catalog import ODATA_PRODUCTS_URL

log = logging.getLogger(__name__)

DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"
TRUSTED_SUFFIX = ".copernicus.eu"
CHUNK = 8 * 1024 * 1024
REDIRECT_CODES = {301, 302, 303, 307, 308}


def product_size(product_id: str, session: requests.Session | None = None) -> int | None:
    """ContentLength from the catalog. Needs no token."""
    sess = session or requests.Session()
    r = sess.get(f"{ODATA_PRODUCTS_URL}({product_id})", timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"catalog lookup failed for {product_id}: HTTP {r.status_code}")
    return r.json().get("ContentLength")


def _open_stream(url: str, auth: CDSEAuth, session: requests.Session,
                 max_redirects: int = 5) -> requests.Response:
    retried_401 = False
    for _ in range(max_redirects + 2):
        host = urlparse(url).hostname or ""
        if not host.endswith(TRUSTED_SUFFIX):
            raise RuntimeError(f"refusing to send CDSE token to untrusted host {host!r}")
        r = session.get(url, headers={"Authorization": f"Bearer {auth.token()}"},
                        allow_redirects=False, stream=True, timeout=(30, 300))
        if r.status_code in REDIRECT_CODES:
            url = urljoin(url, r.headers["Location"])
            log.debug("redirect -> %s", urlparse(url).hostname)
            r.close()
            continue
        if r.status_code == 401 and not retried_401:
            r.close()
            auth.invalidate()
            retried_401 = True
            continue
        if r.status_code != 200:
            body = r.text[:300]
            r.close()
            raise RuntimeError(f"download failed: HTTP {r.status_code}: {body}")
        return r
    raise RuntimeError(f"too many redirects for {url}")


def download_product(product_id: str, name: str, dest_dir: Path, auth: CDSEAuth,
                     expected_size: int | None = None,
                     session: requests.Session | None = None,
                     out_name: str | None = None) -> Path:
    """out_name defaults to <name without .SAFE>.zip (Sentinel-1 products arrive zipped).
    Pass it for products delivered as-is, e.g. orbit .EOF files."""
    sess = session or requests.Session()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / (out_name or (name.removesuffix(".SAFE") + ".zip"))
    if out.exists() and (expected_size is None or out.stat().st_size == expected_size):
        log.info("skip (complete): %s", out.name)
        return out

    part = out.with_name(out.name + ".part")
    written = 0
    with _open_stream(DOWNLOAD_URL.format(id=product_id), auth, sess) as r, open(part, "wb") as f:
        for chunk in r.iter_content(CHUNK):
            f.write(chunk)
            written += len(chunk)
    if expected_size is not None and written != expected_size:
        raise RuntimeError(f"{name}: got {written} bytes, catalog says {expected_size}; "
                           f"partial file left at {part}")
    part.replace(out)
    log.info("done: %s (%.3f GB)", out.name, written / 1e9)
    return out
