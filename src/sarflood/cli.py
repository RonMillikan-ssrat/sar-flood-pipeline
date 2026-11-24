"""Command-line entry point. Lets you run the same logic as the DAG without Airflow.

Examples:
    sarflood list-aois
    sarflood query --aoi sindh --start 2022-08-28 --end 2022-09-01
    sarflood query --aoi houston --event harvey-2017 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import UTC, datetime, time

from .aoi import load_aois
from .catalog import DEFAULT_PRODUCT_TYPE, build_filter, search_scenes
from .manifest import write_manifest


def _day(s: str) -> datetime:
    return datetime.combine(datetime.fromisoformat(s).date(), time(0), tzinfo=UTC)


def cmd_list(args) -> int:
    for a in load_aois(args.config).values():
        print(f"{a.name:<10} {a.role:<8} {a.bbox}  {a.description}")
        for e in a.events:
            print(f"{'':<20}event {e.name}: {e.start} -> {e.end}  {e.notes}")
    return 0


def cmd_query(args) -> int:
    aois = load_aois(args.config)
    if args.aoi not in aois:
        print(f"unknown AOI {args.aoi!r}; choose from {sorted(aois)}", file=sys.stderr)
        return 2
    aoi = aois[args.aoi]
    if args.event:
        ev = {e.name: e for e in aoi.events}.get(args.event)
        if ev is None:
            print(f"unknown event {args.event!r} for {aoi.name}", file=sys.stderr)
            return 2
        start, end = _day(ev.start.isoformat()), _day(ev.end.isoformat())
    elif args.start and args.end:
        start, end = _day(args.start), _day(args.end)
    else:
        print("give --event, or both --start and --end", file=sys.stderr)
        return 2

    if args.dry_run:
        print(build_filter(aoi.to_wkt(), start, end, args.product_type))
        return 0

    scenes = search_scenes(aoi.to_wkt(), start, end, args.product_type)
    path = write_manifest(aoi.name, start, end, scenes, args.product_type)
    print(f"{len(scenes)} scenes -> {path}")
    print("by platform:", dict(Counter(s.platform for s in scenes)))
    print("by orbit:   ", dict(Counter(s.orbit_direction for s in scenes)))
    for s in scenes[: args.show]:
        print(f"  {s.start[:19]}  {s.platform} {s.orbit_direction or '?':<10} "
              f"rel={s.relative_orbit}  {s.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sarflood")
    p.add_argument("--config", default=None, help="path to aois.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-aois").set_defaults(func=cmd_list)

    q = sub.add_parser("query", help="search the CDSE catalog and write a manifest")
    q.add_argument("--aoi", required=True)
    q.add_argument("--event")
    q.add_argument("--start", help="YYYY-MM-DD (inclusive)")
    q.add_argument("--end", help="YYYY-MM-DD (exclusive)")
    q.add_argument("--product-type", default=DEFAULT_PRODUCT_TYPE)
    q.add_argument("--dry-run", action="store_true", help="print the $filter and exit")
    q.add_argument("--show", type=int, default=10, help="scenes to print")
    q.set_defaults(func=cmd_query)

    args = p.parse_args(argv)
    if args.config is None:
        from .aoi import DEFAULT_CONFIG
        args.config = DEFAULT_CONFIG
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
