"""Command-line entry point. Lets you run the same logic as the DAG without Airflow.

Examples:
    sarflood list-aois
    sarflood query --aoi sindh --start 2022-08-28 --end 2022-09-01
    sarflood query --aoi houston --event harvey-2017 --dry-run
    sarflood pairs --aoi sindh --event pakistan-2022
    sarflood download --aoi sindh --event pakistan-2022 --flood-pass S1A_044775_0558C6 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime, time

from .aoi import load_aois
from .catalog import DEFAULT_PRODUCT_TYPE, build_filter, search_scenes
from .manifest import data_root, write_json, write_manifest
from .passes import group_passes, select_reference


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


def cmd_pairs(args) -> int:
    aois = load_aois(args.config)
    aoi = aois.get(args.aoi)
    if aoi is None:
        print(f"unknown AOI {args.aoi!r}; choose from {sorted(aois)}", file=sys.stderr)
        return 2
    ev = {e.name: e for e in aoi.events}.get(args.event)
    if ev is None:
        print(f"unknown event {args.event!r} for {aoi.name}", file=sys.stderr)
        return 2
    if ev.reference_start is None:
        print(f"{aoi.name}/{ev.name} has no reference window in the config", file=sys.stderr)
        return 2

    wkt, geom = aoi.to_wkt(), aoi.geometry()
    ev_start, ev_end = _day(ev.start.isoformat()), _day(ev.end.isoformat())
    ref_start, ref_end = _day(ev.reference_start.isoformat()), _day(ev.reference_end.isoformat())

    flood_passes = group_passes(search_scenes(wkt, ev_start, ev_end, args.product_type))
    ref_passes = group_passes(search_scenes(wkt, ref_start, ref_end, args.product_type))
    print(f"{aoi.name}/{ev.name}: {len(flood_passes)} flood passes "
          f"({ev.start}..{ev.end}), {len(ref_passes)} reference candidates "
          f"({ev.reference_start}..{ev.reference_end})")
    print("reference tracks:",
          dict(Counter(f"{p.relative_orbit}{(p.orbit_direction or '?')[0]}" for p in ref_passes)))

    pairings = []
    for fp in flood_passes:
        pr = select_reference(fp, ref_passes, geom, ref_start, ref_end,
                              min_coverage=args.min_coverage, same_platform=args.same_platform)
        pairings.append(pr)
        ref = pr.reference.key if pr.reference else "-"
        print(f"  {fp.start:%Y-%m-%d %H:%M} {fp.key}  track {fp.relative_orbit:>3}"
              f"{(fp.orbit_direction or '?')[0]}  slices={len(fp.slices)}"
              f"{'' if fp.contiguous else '(GAP)'}  cov={fp.coverage(geom):.3f}"
              f"  ref={ref}  [{pr.reason}]")

    paired = sum(p.reference is not None for p in pairings)
    out = write_json(data_root() / "pairs" / aoi.name / f"{ev.name}.json", {
        "aoi": aoi.name, "event": ev.name,
        "event_window": [ev.start.isoformat(), ev.end.isoformat()],
        "reference_window": [ev.reference_start.isoformat(), ev.reference_end.isoformat()],
        "min_coverage": args.min_coverage, "same_platform": args.same_platform,
        "pairs": [{
            "flood": pr.flood.summary(geom),
            "reference": pr.reference.summary(geom) if pr.reference else None,
            "reason": pr.reason,
        } for pr in pairings],
    })
    print(f"{paired}/{len(pairings)} flood passes paired -> {out}")
    return 0


def cmd_download(args) -> int:
    from .auth import CDSEAuth
    from .download import download_product, product_size

    pairs_file = data_root() / "pairs" / args.aoi / f"{args.event}.json"
    if not pairs_file.exists():
        print(f"{pairs_file} not found; run 'sarflood pairs' first", file=sys.stderr)
        return 2
    doc = json.loads(pairs_file.read_text())
    match = [p for p in doc["pairs"] if p["flood"]["key"] == args.flood_pass]
    if not match:
        keys = [p["flood"]["key"] for p in doc["pairs"]]
        print(f"no flood pass {args.flood_pass!r} in {pairs_file}; have {keys}", file=sys.stderr)
        return 2
    pair = match[0]
    if pair["reference"] is None:
        print(f"{args.flood_pass} has no reference ({pair['reason']})", file=sys.stderr)
        return 2

    jobs = []  # (role, pass key, product id, name)
    for role in ("flood", "reference"):
        ps = pair[role]
        jobs += [(role, ps["key"], pid, name) for pid, name in zip(ps["ids"], ps["slices"])]

    sizes = {pid: product_size(pid) for _, _, pid, _ in jobs}
    total = sum(v or 0 for v in sizes.values())
    for role, key, pid, name in jobs:
        sz = sizes[pid]
        print(f"  {role:<9} {key}  {name}  {sz / 1e9 if sz else float('nan'):.2f} GB")
    print(f"{len(jobs)} products, {total / 1e9:.2f} GB total")
    if args.dry_run:
        return 0

    dest = data_root() / "raw" / args.aoi
    auth = CDSEAuth()
    for role, key, pid, name in jobs:
        path = download_product(pid, name, dest, auth, expected_size=sizes[pid])
        print(f"  ok {path}")
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

    pr = sub.add_parser("pairs", help="pair flood passes with same-track dry references")
    pr.add_argument("--aoi", required=True)
    pr.add_argument("--event", required=True)
    pr.add_argument("--product-type", default=DEFAULT_PRODUCT_TYPE)
    pr.add_argument("--min-coverage", type=float, default=0.5,
                    help="fraction of the AOI a pass must cover (default 0.5)")
    pr.add_argument("--same-platform", action="store_true",
                    help="require the reference to come from the same satellite")
    pr.set_defaults(func=cmd_pairs)

    dl = sub.add_parser("download", help="download one flood pass and its reference (.SAFE zips)")
    dl.add_argument("--aoi", required=True)
    dl.add_argument("--event", required=True)
    dl.add_argument("--flood-pass", required=True, help="pass key, e.g. S1A_044775_0558C6")
    dl.add_argument("--dry-run", action="store_true", help="list products and sizes only")
    dl.set_defaults(func=cmd_download)

    args = p.parse_args(argv)
    if args.config is None:
        from .aoi import DEFAULT_CONFIG
        args.config = DEFAULT_CONFIG
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
