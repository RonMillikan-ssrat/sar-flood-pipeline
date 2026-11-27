#!/usr/bin/env python
"""Upload change/labels raster pairs (and optionally the sarflood wheel) to a Unity
Catalog Volume. Runs on the Ubuntu box, not in Databricks: Free Edition can't reach
CDSE / Earth Engine, so acquisition and SNAP stay local and only products move.

Order per pair: change.tif, labels.tif, then manifest.json. A pair is complete only
once its manifest exists, so an interrupted upload is never ingested half-done.
Unchanged pairs (same SHA-256 in the remote manifest) are skipped unless --force.

Auth: the databricks-sdk default chain. Simplest:
    databricks auth login --host https://<your-workspace>.cloud.databricks.com
then pass --profile <name> if it isn't DEFAULT.

Examples:
    python scripts/upload_to_volume.py --dry-run
    uv build --wheel
    python scripts/upload_to_volume.py --wheel dist/sarflood-0.1.0-py3-none-any.whl
    python scripts/upload_to_volume.py --aoi sindh jamuna --force
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from sarflood.lakehouse import (
    MANIFEST_NAME,
    build_manifest,
    find_pairs,
    pair_dir,
    same_content,
    volume_root,
)


def ensure_volume(w, catalog: str, schema: str, volume: str) -> None:
    """Create schema and managed volume if missing. The catalog must already exist."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.catalog import VolumeType

    try:
        w.schemas.get(f"{catalog}.{schema}")
    except NotFound:
        w.schemas.create(name=schema, catalog_name=catalog)
        print(f"created schema {catalog}.{schema}")
    try:
        w.volumes.read(f"{catalog}.{schema}.{volume}")
    except NotFound:
        w.volumes.create(catalog_name=catalog, schema_name=schema, name=volume,
                         volume_type=VolumeType.MANAGED)
        print(f"created volume {catalog}.{schema}.{volume}")


def remote_manifest(w, path: str) -> dict | None:
    from databricks.sdk.errors import NotFound

    try:
        return json.loads(w.files.download(path).contents.read())
    except NotFound:
        return None


def put_file(w, local: Path, remote: str) -> None:
    with open(local, "rb") as f:
        w.files.upload(remote, f, overwrite=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--catalog", default="workspace")
    p.add_argument("--schema", default="sarflood")
    p.add_argument("--volume", default="raw")
    p.add_argument("--aoi", nargs="*", help="only these AOIs (default: all found)")
    p.add_argument("--data-root", type=Path, help="default: $DATA_ROOT or ./data")
    p.add_argument("--wheel", type=Path, help="also upload this wheel to <volume>/wheels/")
    p.add_argument("--profile", help="~/.databrickscfg profile (default: SDK default chain)")
    p.add_argument("--force", action="store_true", help="re-upload unchanged pairs")
    p.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    args = p.parse_args(argv)

    pairs, problems = find_pairs(args.data_root, args.aoi)
    for msg in problems:
        print(f"skip   {msg}", file=sys.stderr)
    if not pairs and not args.wheel:
        print("nothing to upload", file=sys.stderr)
        return 1

    vol = volume_root(args.catalog, args.schema, args.volume)
    if args.dry_run:
        for pr in pairs:
            mb = (pr.change_tif.stat().st_size + pr.labels_tif.stat().st_size) / 1e6
            print(f"would  {pr.aoi:<10} {pr.stem}  ({mb:,.1f} MB) -> {pair_dir(vol, pr)}/")
        if args.wheel:
            print(f"would  wheel {args.wheel.name} -> {vol}/wheels/")
        return 0

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    ensure_volume(w, args.catalog, args.schema, args.volume)

    if args.wheel:
        put_file(w, args.wheel, f"{vol}/wheels/{args.wheel.name}")
        print(f"wheel  {args.wheel.name} -> {vol}/wheels/")

    n_up = n_skip = 0
    for pr in pairs:
        man = build_manifest(pr)
        dest = pair_dir(vol, pr)
        if not man["grid_match"]:
            print(f"WARN   {pr.aoi}/{pr.stem}: change/labels grids differ "
                  "(uploading anyway; bronze will quarantine it)", file=sys.stderr)
        if not args.force and same_content(man, remote_manifest(w, f"{dest}/{MANIFEST_NAME}")):
            print(f"same   {pr.aoi:<10} {pr.stem}")
            n_skip += 1
            continue
        put_file(w, pr.change_tif, f"{dest}/{pr.change_tif.name}")
        put_file(w, pr.labels_tif, f"{dest}/{pr.labels_tif.name}")
        w.files.upload(f"{dest}/{MANIFEST_NAME}",
                       io.BytesIO(json.dumps(man, indent=2).encode()), overwrite=True)
        mb = sum(f["bytes"] for f in man["files"].values()) / 1e6
        print(f"put    {pr.aoi:<10} {pr.stem}  ({mb:,.1f} MB)")
        n_up += 1
    print(f"done: {n_up} uploaded, {n_skip} unchanged, {len(problems)} skipped -> {vol}/pairs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
