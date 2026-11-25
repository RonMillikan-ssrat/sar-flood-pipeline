"""Command-line entry point. Lets you run the same logic as the DAG without Airflow.

Examples:
    sarflood list-aois
    sarflood query --aoi sindh --start 2022-08-28 --end 2022-09-01
    sarflood query --aoi houston --event harvey-2017 --dry-run
    sarflood pairs --aoi sindh --event pakistan-2022
    sarflood download --aoi sindh --event pakistan-2022 --flood-pass S1A_044775_0558C6 --dry-run
    sarflood calibrate data/raw/sindh/S1A_..._E8B2.zip --pol vv --window 8000 10048 12000 14048
    sarflood compare data/sigma0/sindh/..._vv_nr_win8000-10048_12000-14048.tif \
        data/snap/sindh/..._tnr_cal.tif --snap-band Sigma0_VV
    sarflood orbits --aoi sindh --event pakistan-2022 --flood-pass S1A_044775_0558C6
    sarflood terrain-correct --aoi sindh --event pakistan-2022 --flood-pass S1A_044775_0558C6 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime, time
from pathlib import Path

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


def cmd_calibrate(args) -> int:
    from pathlib import Path

    from .calibrate import calibrate_safe

    z = Path(args.zip)
    if not z.exists():
        print(f"{z} not found", file=sys.stderr)
        return 2
    suffix = "" if args.window is None else "_win{}-{}_{}-{}".format(*args.window)
    noise = "" if args.no_noise else ("_nr" if not args.no_clip else "_nr_noclip")
    out = Path(args.out) if args.out else (
        data_root() / "sigma0" / z.parent.name / f"{z.stem}_{args.pol.lower()}{noise}{suffix}.tif")
    st = calibrate_safe(z, args.pol, out, window=tuple(args.window) if args.window else None,
                        remove_noise=not args.no_noise, clip_negative=not args.no_clip)
    print(f"{out}")
    print(f"valid px {st.valid:,}  negative after noise removal {st.negative:,} "
          f"({100 * st.negative / max(st.valid, 1):.2f}%)")
    return 0


def cmd_compare(args) -> int:
    import rasterio

    from .compare import compare_arrays, describe, read_window

    for f in (args.ours, args.snap):
        if not Path(f).exists():
            print(f"{f} not found", file=sys.stderr)
            return 2
    print("SNAP file:", *describe(args.snap), sep="\n  ")
    with rasterio.open(args.ours) as src:
        tags = src.tags()
        ours = src.read(1)
    win = tags.get("window")
    if not win:
        print(f"{args.ours} has no 'window' tag; was it written by sarflood calibrate?",
              file=sys.stderr)
        return 2
    rows, cols = win.split(",")
    r0, r1 = map(int, rows.split(":"))
    c0, c1 = map(int, cols.split(":"))
    band = int(args.snap_band) if args.snap_band.isdigit() else args.snap_band
    snap = read_window(args.snap, band, (r0, r1, c0, c1))

    res = compare_arrays(ours, snap)
    print(f"window rows {r0}:{r1} cols {c0}:{c1}  ({res.n_pixels:,} px)  "
          f"ours: {tags.get('polarisation')} clip={tags.get('negatives_clipped')}")
    print(f"  no-data   ours {res.nodata_ours:,}  snap {res.nodata_snap:,}  "
          f"mismatch {res.nodata_mismatch:,}")
    print(f"  zeros     ours {res.zero_ours:,}  snap {res.zero_snap:,}  "
          f"mismatch {res.zero_mismatch:,}")
    print(f"  negative  ours {res.negative_ours:,}  snap {res.negative_snap:,}")
    print(f"  compared  {res.n_compared:,} px positive in both")
    print(f"  diff dB   mean {res.mean_db:+.4f}  median {res.median_db:+.4f}  "
          f"std {res.std_db:.4f}  p1 {res.p01_db:+.4f}  p99 {res.p99_db:+.4f}  "
          f"max|d| {res.max_abs_db:.4f}")
    print(f"  within    0.01 dB {100 * res.within_001_db:.2f}%   "
          f"0.1 dB {100 * res.within_01_db:.2f}%")
    print(f"  corr(log) aligned {res.corr_log:.5f}  flipped L-R {res.corr_log_flipped_lr:.5f}  "
          f"flipped U-D {res.corr_log_flipped_ud:.5f}")
    tag = "".join(ch if ch.isalnum() else "_" for ch in args.snap_band)
    out = write_json(Path(args.ours).with_suffix(f".compare_b{tag}.json"),
                     {"ours": args.ours, "snap": args.snap, "snap_band": args.snap_band,
                      "window": [r0, r1, c0, c1], **res.to_dict()})
    print(f"-> {out}")
    return 0


def _load_pair(aoi: str, event: str, flood_pass: str):
    pairs_file = data_root() / "pairs" / aoi / f"{event}.json"
    if not pairs_file.exists():
        print(f"{pairs_file} not found; run 'sarflood pairs' first", file=sys.stderr)
        return None
    doc = json.loads(pairs_file.read_text())
    match = [p for p in doc["pairs"] if p["flood"]["key"] == flood_pass]
    if not match:
        print(f"no flood pass {flood_pass!r} in {pairs_file}", file=sys.stderr)
        return None
    return match[0]


def cmd_orbits(args) -> int:
    from .auth import CDSEAuth
    from .download import download_product
    from .orbits import find_orbit, pass_window, snap_orbit_dir

    pair = _load_pair(args.aoi, args.event, args.flood_pass)
    if pair is None:
        return 2
    roles = ["flood", "reference"] if args.role == "both" else [args.role]
    auth = None
    for role in roles:
        ps = pair[role]
        if ps is None:
            print(f"{args.flood_pass}: no {role} pass", file=sys.stderr)
            return 2
        platform = ps["key"].split("_")[0]
        start, end = pass_window(ps["slices"])
        orb = find_orbit(platform, start, end)
        if orb is None:
            print(f"{role} {ps['key']}: no AUX_POEORB covering {start:%Y-%m-%d %H:%M:%S}-"
                  f"{end:%H:%M:%S}. Too recent? Precise orbits appear about 3 weeks after "
                  f"acquisition", file=sys.stderr)
            return 1
        dest = snap_orbit_dir(platform, start)
        print(f"{role:<9} {ps['key']}  {start:%Y-%m-%d %H:%M:%S}-{end:%H:%M:%S}\n"
              f"  orbit  {orb.name}  (valid {orb.valid_start:%m-%d %H:%M} -> "
              f"{orb.valid_end:%m-%d %H:%M})\n  dest   {dest}")
        if args.dry_run:
            continue
        auth = auth or CDSEAuth()
        path = download_product(orb.id, orb.name, dest, auth, expected_size=orb.size,
                                out_name=orb.name)
        print(f"  ok     {path}")
    return 0


def cmd_terrain_correct(args) -> int:
    from .snap import build_tc_graph, gpt_path, run_gpt, tc_stem, utm_epsg

    aois = load_aois(args.config)
    aoi = aois.get(args.aoi)
    if aoi is None:
        print(f"unknown AOI {args.aoi!r}", file=sys.stderr)
        return 2
    pairs_file = data_root() / "pairs" / args.aoi / f"{args.event}.json"
    if not pairs_file.exists():
        print(f"{pairs_file} not found; run 'sarflood pairs' first", file=sys.stderr)
        return 2
    doc = json.loads(pairs_file.read_text())
    match = [p for p in doc["pairs"] if p["flood"]["key"] == args.flood_pass]
    if not match:
        print(f"no flood pass {args.flood_pass!r} in {pairs_file}", file=sys.stderr)
        return 2
    pair = match[0]
    roles = ["flood", "reference"] if args.role == "both" else [args.role]

    x0, y0, x1, y1 = aoi.bbox
    epsg = args.epsg or utm_epsg((x0 + x1) / 2, (y0 + y1) / 2)
    raw = data_root() / "raw" / args.aoi
    out_dir = data_root() / "tc" / args.aoi
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    for role in roles:
        ps = pair[role]
        if ps is None:
            print(f"{args.flood_pass}: no {role} pass ({pair['reason']})", file=sys.stderr)
            return 2
        zips = [(raw / (n.removesuffix(".SAFE") + ".zip")).resolve() for n in ps["slices"]]
        missing = [z for z in zips if not z.exists()]
        if missing:
            print(f"{role} {ps['key']}: missing {[z.name for z in missing]}; run 'sarflood download'",
                  file=sys.stderr)
            return 2
        stem = tc_stem(ps["key"], args.spacing, args.clip, epsg)
        out = (out_dir / stem).resolve()  # absolute for gpt; the GeoTIFF writer adds .tif
        graph = build_tc_graph(zips, out, aoi.to_wkt(), epsg, args.spacing, args.clip)
        gfile = out_dir / f"{stem}.graph.xml"
        gfile.write_text(graph)
        print(f"{role:<9} {ps['key']}  {len(zips)} slice(s)  EPSG:{epsg}  {args.spacing:g} m  "
              f"clip={args.clip}\n  graph  {gfile}\n  output {out}.tif")
        if args.dry_run:
            print(f"  run: {gpt_path()} {gfile}")
            continue
        rc = run_gpt(gfile)
        if rc != 0:
            print(f"gpt failed with exit code {rc}", file=sys.stderr)
            return rc
    return rc


def cmd_change(args) -> int:
    from .change import quicklook, run_change
    from .snap import tc_stem, utm_epsg

    aois = load_aois(args.config)
    aoi = aois.get(args.aoi)
    if aoi is None:
        print(f"unknown AOI {args.aoi!r}", file=sys.stderr)
        return 2
    pair = _load_pair(args.aoi, args.event, args.flood_pass)
    if pair is None:
        return 2
    if pair["reference"] is None:
        print(f"{args.flood_pass} has no reference ({pair['reason']})", file=sys.stderr)
        return 2
    x0, y0, x1, y1 = aoi.bbox
    epsg = args.epsg or utm_epsg((x0 + x1) / 2, (y0 + y1) / 2)
    tc = data_root() / "tc" / args.aoi
    f_tif = tc / (tc_stem(pair["flood"]["key"], args.spacing, args.clip, epsg) + ".tif")
    r_tif = tc / (tc_stem(pair["reference"]["key"], args.spacing, args.clip, epsg) + ".tif")
    missing = [p.name for p in (f_tif, r_tif) if not p.exists()]
    if missing:
        print(f"missing {missing}; run 'sarflood terrain-correct' first", file=sys.stderr)
        return 2

    out_dir = data_root() / "change" / args.aoi
    stem = f"{pair['flood']['key']}__{pair['reference']['key']}_{int(args.spacing)}m"
    out_tif = out_dir / f"{stem}_change.tif"
    st = run_change(f_tif, r_tif, out_tif, args.change_db, args.water_db)
    png = quicklook(out_tif, out_dir / f"{stem}_quicklook.png")
    write_json(out_dir / f"{stem}_stats.json", {"flood": f_tif.name, "reference": r_tif.name,
                                                **st.to_dict()})
    print(f"overlap {st.pixels:,} px   valid {st.valid:,} ({100 * st.valid / st.pixels:.1f}%)   "
          f"masked: layover/shadow {st.masked_layover_shadow:,}, "
          f"non-positive {st.masked_nonpositive:,}")
    print(f"VV change dB  p5 {st.dvv_p05:+.2f}  median {st.dvv_p50:+.2f}  p95 {st.dvv_p95:+.2f}")
    print(f"candidate (dVV <= {st.change_db} dB and flood VV <= {st.water_db} dB): "
          f"{st.candidate_pixels:,} px = {st.candidate_km2:,.1f} km2   [threshold preview]")
    print(f"-> {out_tif}\n-> {png}")
    return 0


def cmd_ems_info(args) -> int:
    from .ems import fetch_activation, is_sentinel1, overlap_fractions, parse_activation

    aois = load_aois(args.config)
    rc = 0
    for code in args.code:
        doc = fetch_activation(code)
        raw = write_json(data_root() / "ems" / f"{code}.json", doc)
        meta, ems_aois = parse_activation(doc)
        print(f"== {meta['code']}  {meta['name']}  event {meta['eventTime']}  "
              f"activated {meta['activationTime']}   (raw -> {raw})")
        for a in ems_aois:
            bb = ", ".join(f"{v:.3f}" for v in a.extent.bounds) if a.extent else "no extent"
            print(f"  AOI{a.number:02d} {a.name:<14} bbox [{bb}]")
            for name, ours in aois.items():
                in_ours, covers = overlap_fractions(a.extent, ours.geometry())
                if in_ours > 0:
                    print(f"      overlaps {name:<10}: {100 * in_ours:5.1f}% of the EMS AOI lies in "
                          f"ours; covers {100 * covers:5.1f}% of ours")
            for pr in a.products:
                print(f"      {pr.type} v{pr.version}  delivered {pr.delivery_time}")
                for im in pr.images:
                    flag = "   <-- Sentinel-1" if is_sentinel1(im) else ""
                    print(f"          image {im.sensor_type}/{im.sensor_name}  "
                          f"{im.acquisition_time}  {im.file_name or ''}{flag}")
                if pr.download:
                    print(f"          zip {pr.download}")
    return rc


def cmd_evaluate(args) -> int:
    import numpy as np
    import rasterio

    from .labels import LABEL_BANDS, confusion_quicklook, rasterize_ems, score, threshold_sweep

    pair = _load_pair(args.aoi, args.event, args.flood_pass)
    if pair is None or pair["reference"] is None:
        print("no usable pair (run 'sarflood pairs')", file=sys.stderr)
        return 2
    cdir = data_root() / "change" / args.aoi
    stem = f"{pair['flood']['key']}__{pair['reference']['key']}_{int(args.spacing)}m"
    change_tif = cdir / f"{stem}_change.tif"
    if not change_tif.exists():
        print(f"{change_tif} not found; run 'sarflood change' first", file=sys.stderr)
        return 2
    z = Path(args.ems_zip)
    if not z.exists():
        print(f"{z} not found", file=sys.stderr)
        return 2

    with rasterio.open(change_tif) as src:
        names = list(src.descriptions)
        band = lambda n: src.read(names.index(n) + 1)
        dvv, fvv = band("dvv_db"), band("flood_vv_db")
        valid, cand = band("valid") > 0, band("candidate") > 0
        profile, tags = src.profile, src.tags()
        transform, crs, shape = src.transform, src.crs, src.shape
    px_area = abs(transform.a * transform.e)

    labels = rasterize_ems(z, shape, transform, crs, supersample=args.supersample)
    out_dir = data_root() / "eval" / args.aoi
    out_dir.mkdir(parents=True, exist_ok=True)
    lab_tif = out_dir / f"{stem}_{z.stem}_labels.tif"
    profile.update(count=len(LABEL_BANDS), dtype="float32")
    with rasterio.open(lab_tif, "w", **profile) as dst:
        for i, n in enumerate(LABEL_BANDS, 1):
            dst.write(labels[n].astype(np.float32), i)
            dst.set_band_description(i, n)
        dst.update_tags(ems_zip=z.name, change=change_tif.name)

    aoi_px = int(labels["aoi"].sum())
    print(f"labels -> {lab_tif}")
    print(f"EMS AOI {aoi_px:,} px ({aoi_px * px_area / 1e6:.1f} km2): flooded "
          f"{100 * labels['flood'][labels['aoi'] > 0].mean():.1f}%, not analysed "
          f"{int(labels['not_analysed'].sum()):,} px, permanent water {int(labels['perm_water'].sum()):,} px")
    cd, wd = tags.get("change_db"), tags.get("water_db")
    results = {}
    for name, lo, hi in [("all", None, None), ("pure_only", 0.2, 0.8)]:
        s = score(cand, labels, valid, px_area, lo, hi)
        results[name] = s.to_dict()
        print(f"[{name:<9}] threshold preview (dVV <= {cd}, VV <= {wd}): IoU {s.iou:.3f}  "
              f"F1 {s.f1:.3f}  precision {s.precision:.3f}  recall {s.recall:.3f}   "
              f"label {s.label_km2:.1f} km2, predicted {s.pred_km2:.1f} km2, "
              f"eval {s.n_eval:,} px" + (f", {s.excluded_mixed:,} mixed excluded" if lo else ""))
    if args.sweep:
        rows = threshold_sweep(dvv, fvv, labels, valid, px_area)
        results["sweep"] = rows
        best = max(rows, key=lambda r: r["iou"] if r["iou"] == r["iou"] else -1)
        print("sweep (IoU; tuned on these labels, so optimistic):")
        waters = sorted({r["water_db"] for r in rows}, reverse=True)
        print("  dVV\\VV  " + "".join(f"{w:>8.0f}" for w in waters))
        for c in sorted({r["change_db"] for r in rows}, reverse=True):
            row = {r["water_db"]: r["iou"] for r in rows if r["change_db"] == c}
            print(f"  {c:>6.0f}  " + "".join(f"{row[w]:>8.3f}" for w in waters))
        print(f"  best: dVV <= {best['change_db']}, VV <= {best['water_db']}: IoU {best['iou']:.3f}")
    png = confusion_quicklook(cand, labels, valid, out_dir / f"{stem}_{z.stem}_confusion.png",
                              title=f"{args.aoi}: threshold preview vs {z.stem}")
    out = write_json(out_dir / f"{stem}_{z.stem}_score.json",
                     {"change": change_tif.name, "ems_zip": z.name, **results})
    print(f"-> {out}\n-> {png}")
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

    ca = sub.add_parser("calibrate", help="thermal-noise removal + sigma0 calibration (radar geometry)")
    ca.add_argument("zip", help="downloaded .SAFE zip")
    ca.add_argument("--pol", default="vv", choices=["vv", "vh", "VV", "VH"])
    ca.add_argument("--window", type=int, nargs=4, metavar=("ROW0", "ROW1", "COL0", "COL1"),
                    help="half-open pixel window for quick tests")
    ca.add_argument("--no-noise", action="store_true", help="skip thermal-noise removal")
    ca.add_argument("--no-clip", action="store_true",
                    help="keep negative values after noise removal (use before resampling)")
    ca.add_argument("--out", help="output GeoTIFF (default under $DATA_ROOT/sigma0/)")
    ca.set_defaults(func=cmd_calibrate)

    co = sub.add_parser("compare", help="compare our sigma0 window with SNAP's output")
    co.add_argument("ours", help="GeoTIFF from 'sarflood calibrate'")
    co.add_argument("snap", help="SNAP GeoTIFF (full slice)")
    co.add_argument("--snap-band", default="1",
                    help="band name (e.g. Sigma0_VV) or 1-based index (default 1)")
    co.set_defaults(func=cmd_compare)

    tc = sub.add_parser("terrain-correct",
                        help="SNAP: assemble, orbit, TNR, calibrate, terrain-correct, clip to AOI")
    tc.add_argument("--aoi", required=True)
    tc.add_argument("--event", required=True)
    tc.add_argument("--flood-pass", required=True)
    tc.add_argument("--role", choices=["flood", "reference", "both"], default="both")
    tc.add_argument("--spacing", type=float, default=20.0, help="pixel spacing in m (default 20)")
    tc.add_argument("--epsg", type=int, help="output CRS (default: UTM zone of the AOI centre)")
    tc.add_argument("--clip", action="store_true",
                    help="clip negatives in TNR (default off: TC resamples, see MPC-0392)")
    tc.add_argument("--dry-run", action="store_true", help="write the graph(s) and print, no gpt run")
    tc.set_defaults(func=cmd_terrain_correct)

    ob = sub.add_parser("orbits", help="fetch precise orbit files from CDSE into SNAP's folder")
    ob.add_argument("--aoi", required=True)
    ob.add_argument("--event", required=True)
    ob.add_argument("--flood-pass", required=True)
    ob.add_argument("--role", choices=["flood", "reference", "both"], default="both")
    ob.add_argument("--dry-run", action="store_true", help="find the files, don't download")
    ob.set_defaults(func=cmd_orbits)

    ch = sub.add_parser("change", help="flood-vs-reference change on the shared grid + quick-look")
    ch.add_argument("--aoi", required=True)
    ch.add_argument("--event", required=True)
    ch.add_argument("--flood-pass", required=True)
    ch.add_argument("--spacing", type=float, default=20.0)
    ch.add_argument("--epsg", type=int)
    ch.add_argument("--clip", action="store_true", help="use the clipped TC outputs")
    ch.add_argument("--change-db", type=float, default=-3.0,
                    help="candidate needs VV change <= this (default -3)")
    ch.add_argument("--water-db", type=float, default=-18.0,
                    help="candidate needs flood VV <= this (default -18)")
    ch.set_defaults(func=cmd_change)

    ei = sub.add_parser("ems-info", help="EMS activation AOIs, overlap with our AOIs, product imagery")
    ei.add_argument("--code", action="append", required=True, help="e.g. EMSR629 (repeatable)")
    ei.set_defaults(func=cmd_ems_info)

    ev = sub.add_parser("evaluate", help="rasterize EMS labels onto the change grid and score")
    ev.add_argument("--aoi", required=True)
    ev.add_argument("--event", required=True)
    ev.add_argument("--flood-pass", required=True)
    ev.add_argument("--ems-zip", required=True, help="EMS DEL vector package (.zip)")
    ev.add_argument("--spacing", type=float, default=20.0)
    ev.add_argument("--supersample", type=int, default=5, help="sub-pixels per side (default 5)")
    ev.add_argument("--sweep", action="store_true", help="IoU over a grid of thresholds")
    ev.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    if args.config is None:
        from .aoi import DEFAULT_CONFIG
        args.config = DEFAULT_CONFIG
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
