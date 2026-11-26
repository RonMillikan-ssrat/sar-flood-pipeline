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

    from .manifest import find_raw, raw_store

    auth = None
    for role, key, pid, name in jobs:
        zip_name = name.removesuffix(".SAFE") + ".zip"
        existing = find_raw(zip_name)
        # Already downloaded (shared store or a legacy per-AOI folder): let download_product
        # check its size there and skip; new files go to the shared store.
        dest = existing.parent if existing else raw_store()
        if existing is None or (sizes[pid] and existing.stat().st_size != sizes[pid]):
            auth = auth or CDSEAuth()
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
    from .manifest import find_raw, raw_store

    out_dir = data_root() / "tc" / args.aoi
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    for role in roles:
        ps = pair[role]
        if ps is None:
            print(f"{args.flood_pass}: no {role} pass ({pair['reason']})", file=sys.stderr)
            return 2
        names = [n.removesuffix(".SAFE") + ".zip" for n in ps["slices"]]
        zips = [(find_raw(n) or raw_store() / n).resolve() for n in names]
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


def _change_stem(pair: dict, spacing: float, smooth: int) -> str:
    s = f"{pair['flood']['key']}__{pair['reference']['key']}_{int(spacing)}m"
    return s if smooth <= 1 else f"{s}_s{smooth}"


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
    stem = _change_stem(pair, args.spacing, args.smooth)
    out_tif = out_dir / f"{stem}_change.tif"
    st = run_change(f_tif, r_tif, out_tif, args.change_db, args.water_db, args.smooth)
    png = quicklook(out_tif, out_dir / f"{stem}_quicklook.png")
    write_json(out_dir / f"{stem}_stats.json", {"flood": f_tif.name, "reference": r_tif.name,
                                                **st.to_dict()})
    print(f"overlap {st.pixels:,} px   smooth {st.smooth}x{st.smooth}   "
          f"valid VV {st.valid:,} ({100 * st.valid / st.pixels:.1f}%), "
          f"VH {st.valid_vh:,} ({100 * st.valid_vh / st.pixels:.1f}%)")
    print(f"masked: layover/shadow {st.masked_layover_shadow:,}, non-positive VV "
          f"{st.masked_nonpositive:,}, VH {st.masked_nonpositive_vh:,}")
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

    from .labels import (
        LABEL_BANDS,
        class_histograms,
        confusion_quicklook,
        rasterize_ems,
        score,
        threshold_sweep,
    )

    pair = _load_pair(args.aoi, args.event, args.flood_pass)
    if pair is None or pair["reference"] is None:
        print("no usable pair (run 'sarflood pairs')", file=sys.stderr)
        return 2
    cdir = data_root() / "change" / args.aoi
    stem = _change_stem(pair, args.spacing, args.smooth)
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
        need = {"flood_vv_db", "ref_vv_db", "dvv_db", "flood_vh_db", "dvh_db"} \
            if (args.hist or args.rule) else set()
        hist_bands = {n: band(n) for n in need}
        valid_vh = band("valid_vh") > 0 if "valid_vh" in names else valid
        profile, tags = src.profile, src.tags()
        transform, crs, shape = src.transform, src.crs, src.shape
    px_area = abs(transform.a * transform.e)

    labels = rasterize_ems(z, shape, transform, crs, supersample=args.supersample)
    out_dir = data_root() / "eval" / args.aoi
    out_dir.mkdir(parents=True, exist_ok=True)
    lab_tif = out_dir / f"{stem}_{z.stem}_labels.tif"
    profile.update(count=len(LABEL_BANDS), dtype="float32", tiled=True,
                   blockxsize=256, blockysize=256, compress="deflate")
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
        print("sweep (IoU; tuned on these labels, so optimistic; 'none' = no change condition):")
        waters = sorted({r["water_db"] for r in rows}, reverse=True)
        print("  dVV\\VV  " + "".join(f"{w:>8.0f}" for w in waters))
        changes = [None] + sorted({r["change_db"] for r in rows} - {None}, reverse=True)
        for ch in changes:
            row = {r["water_db"]: r["iou"] for r in rows if r["change_db"] == ch}
            lab = "none" if ch is None else f"{ch:.0f}"
            print(f"  {lab:>6}  " + "".join(f"{row[w]:>8.3f}" for w in waters))
        bc = "none" if best["change_db"] is None else best["change_db"]
        print(f"  best: dVV <= {bc}, VV <= {best['water_db']}: IoU {best['iou']:.3f}"
              f"  (P {best['precision']:.3f}, R {best['recall']:.3f})")
        nums = [ch for ch in changes if ch is not None]
        edge_c = best["change_db"] is not None and best["change_db"] in (nums[0], nums[-1])
        if edge_c or best["water_db"] in (waters[0], waters[-1]):
            print("  note: best is on the edge of the grid; the optimum may lie beyond it")
    if args.rule:
        from .rules import apply_rule, parse_rule

        results["rules"] = {}
        print("rules (fixed in advance; otsu/gmm thresholds come from the change map, not labels):")
        for spec in args.rule:
            rule = parse_rule(spec)
            pred = apply_rule(rule, hist_bands, valid, valid_vh)
            entry = {"rule": rule.describe()}
            for name, lo, hi in [("all", None, None), ("pure_only", 0.2, 0.8)]:
                s = score(pred, labels, valid, px_area, lo, hi)
                entry[name] = s.to_dict()
            results["rules"][spec] = entry
            a, p = entry["all"], entry["pure_only"]
            print(f"  {spec:<18} [{rule.describe()}]  IoU {a['iou']:.3f}  F1 {a['f1']:.3f}  "
                  f"P {a['precision']:.3f}  R {a['recall']:.3f}  | pure-only IoU {p['iou']:.3f}")
            tag = "".join(ch if ch.isalnum() else "_" for ch in spec)
            confusion_quicklook(pred, labels, valid, out_dir / f"{stem}_{z.stem}_rule_{tag}.png",
                                title=f"{args.aoi}: rule {spec} [{rule.describe()}] vs {z.stem}")
    if args.hist:
        hpng = out_dir / f"{stem}_{z.stem}_hist.png"
        summ = class_histograms(hist_bands, labels, valid, valid_vh, hpng,
                                title=f"{args.aoi} {stem}: EMS flood vs dry (scored region)")
        results["hist"] = summ
        print("distributions in the scored region (p10 / p50 / p90), best single cut 'x <= t':")
        for n, s in summ.items():
            fl, dr = s["flood_p10_p50_p90"], s["dry_p10_p50_p90"]
            print(f"  {n:<12} flood {fl[0]:7.2f} {fl[1]:7.2f} {fl[2]:7.2f}   "
                  f"dry {dr[0]:7.2f} {dr[1]:7.2f} {dr[2]:7.2f}   "
                  f"cut {s['best_cut']:7.2f} -> IoU {s['best_cut_iou']:.3f}")
        print(f"-> {hpng}")
    png = confusion_quicklook(cand, labels, valid, out_dir / f"{stem}_{z.stem}_confusion.png",
                              title=f"{args.aoi}: threshold preview vs {z.stem}")
    out = write_json(out_dir / f"{stem}_{z.stem}_score.json",
                     {"change": change_tif.name, "ems_zip": z.name, **results})
    print(f"-> {out}\n-> {png}")
    return 0


def cmd_ems_aoi(args) -> int:
    from .ems_zip import aoi_from_zip, yaml_entry

    aois = load_aois(args.config)
    like = aois.get(args.like)
    if like is None:
        print(f"--like {args.like!r} is not a configured AOI", file=sys.stderr)
        return 2
    if args.name in aois:
        print(f"AOI {args.name!r} already exists in the config", file=sys.stderr)
        return 2
    info = aoi_from_zip(Path(args.zip), margin=args.margin)
    text = yaml_entry(args.name, info, like, role=args.role)
    print(f"EMS AOI bounds {[round(v, 4) for v in info['bounds']]} -> bbox {info['bbox']}")
    print(text)
    if args.append:
        cfg = Path(args.config)
        cfg.write_text(cfg.read_text().rstrip("\n") + "\n" + text)
        load_aois(cfg)  # re-validate
        print(f"appended to {cfg}")
    return 0


def _print_fewshot(name: str, fold: dict, fixed: dict) -> None:
    """fixed: score name -> (IoU at its fixed cut, best-cut IoU ceiling)."""
    fs = fold["fewshot"]
    ns = list(next(iter(fs.values())).keys())
    print(f"  {name:<10} few-label cut (mean IoU over 50 draws; [p10-p90] at n=100)")
    print(f"  {'':<10} {'score':<11} {'fixed':>6} " + "".join(f"{'n=' + n:>8}" for n in ns)
          + f" {'p10-p90@100':>13} {'ceiling':>8}")
    for score, d in fs.items():
        fx, ceil = fixed.get(score, (float("nan"), float("nan")))
        band = d.get("100")
        rng_s = f"{band['p10']:.3f}-{band['p90']:.3f}" if band else ""
        print(f"  {'':<10} {score:<11} {fx:6.3f} " + "".join(f"{d[n]['mean']:8.3f}" for n in ns)
              + f" {rng_s:>13} {ceil:8.3f}")


def cmd_train_pixels(args) -> int:
    import numpy as np

    from .features import load_aoi
    from .labels import confusion_quicklook
    from .pixels import leave_one_aoi_out

    if len(args.aoi) < 2:
        print("need at least 2 AOIs for leave-one-out", file=sys.stderr)
        return 2
    data = []
    for a in args.aoi:
        pair = _load_pair(a, args.event, args.flood_pass)
        if pair is None or pair["reference"] is None:
            return 2
        stem = _change_stem(pair, args.spacing, args.smooth)
        ctif = data_root() / "change" / a / f"{stem}_change.tif"
        labs = sorted((data_root() / "eval" / a).glob(f"{stem}_EMSR*_labels.tif"))
        if not ctif.exists() or len(labs) != 1:
            print(f"{a}: need {ctif.name} and exactly one labels file ({len(labs)} found); "
                  f"run 'sarflood change' and 'sarflood evaluate' with --smooth {args.smooth}",
                  file=sys.stderr)
            return 2
        d = load_aoi(a, ctif, labs[0])
        print(f"{a:<10} {d.X.shape[0]:>10,} scored px   flood share {d.y.mean():.3f}   "
              f"labels {labs[0].name[len(stem) + 1:]}")
        data.append(d)

    res = leave_one_aoi_out(data, n_train=args.samples, seed=args.seed,
                            importance=not args.no_importance)
    out_dir = data_root() / "ml" / "pixels"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.flood_pass}_s{args.smooth}"
    print(f"\nleave-one-AOI-out (train: pure-pixel sample, {args.samples:,}/AOI; "
          f"test: all scored px of the held-out AOI; cut 0.5; '_bal' = class-balanced training)")
    print("best-IoU uses the test labels to pick the cut: a diagnostic ceiling, not a result")
    print(f"  {'held out':<10} {'model':<11} {'trainFS':>7} {'IoU@.5':>7} {'P':>6} {'R':>6} "
          f"{'AUC':>6} {'AP':>6} {'bestIoU':>8} {'cut':>7}")
    for test in data:
        fold = res["folds"][test.name]
        a = fold["rule_A"]
        print(f"  {test.name:<10} {'rule A':<11} {'':>7} {a['iou']:7.3f} {a['precision']:6.3f} "
              f"{a['recall']:6.3f} {a['auc']:6.3f} {a['ap']:6.3f} {a['best_iou']:8.3f} "
              f"{a['best_cut']:6.1f}dB")
        for m, e in fold["models"].items():
            s = e["metrics"]
            print(f"  {'':<10} {m:<11} {e['train_flood_share']:7.2f} {s['iou']:7.3f} "
                  f"{s['precision']:6.3f} {s['recall']:6.3f} {e['auc']:6.3f} {e['ap']:6.3f} "
                  f"{e['best_iou']:8.3f} {e['best_cut']:7.2f}")
        print(f"  {'':<10} (test flood share {fold['test_flood_share']:.2f})")
        for m, e in fold["models"].items():
            prob = e.pop("_prob")
            if m in ("hgb", "hgb_bal"):
                grid = np.zeros(test.shape[0] * test.shape[1], bool)
                grid[test.idx] = prob >= 0.5
                confusion_quicklook(grid.reshape(test.shape), test.labels, test.valid,
                                    out_dir / f"{tag}_{test.name}_{m}_confusion.png",
                                    title=f"{test.name} held out: {m} "
                                          f"(trained on {', '.join(fold['train_aois'])})")
        fixed = {"vv": (a["iou"], a["best_iou"])}
        fixed.update({m: (e["metrics"]["iou"], e["best_iou"]) for m, e in fold["models"].items()})
        _print_fewshot(test.name, fold, fixed)
        for m in ("hgb", "hgb_bal"):
            imp = fold["models"].get(m, {}).get("perm_importance_iou")
            if imp:
                top = sorted(imp.items(), key=lambda kv: -kv[1])[:5]
                print(f"  {'':<10} {m} top features (IoU drop when shuffled): "
                      + ", ".join(f"{k} {v:+.3f}" for k, v in top))
    out = write_json(out_dir / f"{tag}_loao.json", res)
    print(f"-> {out}")
    return 0


def cmd_train_unet(args) -> int:
    import numpy as np
    import rasterio

    from .features import load_aoi
    from .labels import confusion_quicklook
    from .unet import TrainConfig, leave_one_aoi_out_unet, load_grid

    if len(args.aoi) < 2:
        print("need at least 2 AOIs for leave-one-out", file=sys.stderr)
        return 2
    grids, evals, ctifs = [], [], []
    for a in args.aoi:
        pair = _load_pair(a, args.event, args.flood_pass)
        if pair is None or pair["reference"] is None:
            return 2
        stem = _change_stem(pair, args.spacing, args.smooth)
        ctif = data_root() / "change" / a / f"{stem}_change.tif"
        labs = sorted((data_root() / "eval" / a).glob(f"{stem}_EMSR*_labels.tif"))
        if not ctif.exists() or len(labs) != 1:
            print(f"{a}: need {ctif.name} and exactly one labels file; run 'change' and "
                  f"'evaluate' with --smooth {args.smooth}", file=sys.stderr)
            return 2
        g = load_grid(a, ctif, labs[0])
        ev = load_aoi(a, ctif, labs[0])
        print(f"{a:<10} grid {g.target.shape[0]}x{g.target.shape[1]}   loss px "
              f"{int(g.loss_mask.sum()):,}   scored px {ev.idx.size:,}")
        grids.append(g)
        evals.append(ev)
        ctifs.append(ctif)

    cfg = TrainConfig(tile=args.tile, batch=args.batch, epochs=args.epochs,
                      tiles_per_epoch=args.tiles_per_epoch, lr=args.lr, base=args.base,
                      seed=args.seed, device=args.device)
    out_dir = data_root() / "ml" / "unet"
    tag = f"{args.flood_pass}_s{args.smooth}_b{args.base}_e{args.epochs}"
    res = leave_one_aoi_out_unet(grids, evals, cfg, save_dir=out_dir / f"{tag}_models")

    print(f"\nU-Net leave-one-AOI-out ({args.epochs} epochs x {args.tiles_per_epoch} tiles "
          f"of {args.tile}px, base {args.base}); best-IoU picks the cut with test labels "
          f"(a ceiling, not a result)")
    print(f"  {'held out':<10} {'model':<7} {'IoU@.5':>7} {'P':>6} {'R':>6} {'AUC':>6} "
          f"{'AP':>6} {'bestIoU':>8} {'cut':>8}")
    for (g, ctif) in zip(grids, ctifs):
        f = res["folds"][g.name]
        for name, s, cut in [("rule A", f["rule_A"], f"{f['rule_A']['best_cut']:.1f}dB"),
                             ("unet", f["unet"], f"{f['unet']['best_cut']:.2f}")]:
            print(f"  {g.name:<10} {name:<7} {s['iou']:7.3f} {s['precision']:6.3f} "
                  f"{s['recall']:6.3f} {s['auc']:6.3f} {s['ap']:6.3f} {s['best_iou']:8.3f} "
                  f"{cut:>8}")
        ep = f["epochs"]
        print(f"  {'':<10} params {f['params']:,}  train {f['train_seconds']:.0f}s  "
              f"loss {ep[0]['loss']:.3f} -> {ep[-1]['loss']:.3f}")
        _print_fewshot(g.name, f, {"vv": (f["rule_A"]["iou"], f["rule_A"]["best_iou"]),
                                   "unet": (f["unet"]["iou"], f["unet"]["best_iou"])})
        prob = f.pop("_prob_grid")
        with rasterio.open(ctif) as src:
            prof = src.profile
        prof.update(count=1, dtype="float32", nodata=None, tiled=True, blockxsize=256,
                    blockysize=256, compress="deflate")
        ptif = out_dir / f"{tag}_{g.name}_prob.tif"
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(ptif, "w", **prof) as dst:
            dst.write(prob.astype(np.float32), 1)
            dst.set_band_description(1, "flood_probability")
        ev = evals[[e.name for e in evals].index(g.name)]
        confusion_quicklook(prob >= 0.5, ev.labels, ev.valid,
                            out_dir / f"{tag}_{g.name}_confusion.png",
                            title=f"{g.name} held out: U-Net (trained on "
                                  f"{', '.join(f['train_aois'])})")
    out = write_json(out_dir / f"{tag}_loao.json", res)
    print(f"-> {out}")
    return 0


def _kv(pairs: list[str]) -> dict:
    out = {}
    for s in pairs:
        if "=" not in s:
            raise SystemExit(f"--tag expects key=value, got {s!r}")
        k, v = s.split("=", 1)
        out[k] = v
    return out


def cmd_mlflow_log(args) -> int:
    from .tracking import log_report

    rc = 0
    for r in args.report:
        path = Path(r)
        if not path.exists():
            print(f"{path} not found", file=sys.stderr)
            rc = 2
            continue
        run = log_report(path, tags=_kv(args.tag), force=args.force)
        print(f"{path.name}: " + (f"logged run {run}" if run else "already logged (same SHA-256); "
                                  "use --force to log again"))
    return rc


def cmd_fit_final(args) -> int:
    from .features import load_aoi
    from .tracking import fit_final, gate_value_from_report

    report = Path(args.report)
    if not report.exists():
        print(f"{report} not found", file=sys.stderr)
        return 2
    gate = gate_value_from_report(report, args.method)
    data = []
    for a in args.aoi:
        pair = _load_pair(a, args.event, args.flood_pass)
        if pair is None or pair["reference"] is None:
            return 2
        stem = _change_stem(pair, args.spacing, args.smooth)
        ctif = data_root() / "change" / a / f"{stem}_change.tif"
        labs = sorted((data_root() / "eval" / a).glob(f"{stem}_EMSR*_labels.tif"))
        if not ctif.exists() or len(labs) != 1:
            print(f"{a}: missing change map or labels for --smooth {args.smooth}", file=sys.stderr)
            return 2
        data.append(load_aoi(a, ctif, labs[0]))
    v, run = fit_final(data, args.method, args.samples, args.seed, gate, report.name,
                       {"flood_pass": args.flood_pass, "event": args.event,
                        "smooth": str(args.smooth)})
    print(f"registered sarflood-pixel-flood v{v} ({args.method}, run {run}); "
          f"gate mean.iou_n300 = {gate:.4f} from {report.name}")
    return 0


def cmd_registry(args) -> int:
    from .tracking import promote, registry_table, rollback

    if args.action == "show":
        rows = registry_table()
        if not rows:
            print("no registered versions")
        for r in rows:
            print(f"  v{r['version']:<3} {r['method'] or '?':<11} gate {r['gate_value'] or '-':<9} "
                  f"{r['created']}  {' '.join(r['aliases'])}")
    elif args.action == "promote":
        version = args.version
        if args.latest:
            rows = registry_table()
            version = rows[-1]["version"] if rows else None
        if version is None:
            print("promote needs --version N or --latest (and a registered version)", file=sys.stderr)
            return 2
        msg = promote(version, args.min_delta, args.force)
        print(msg)
        return 1 if msg.startswith("REJECTED") else 0
    else:
        print(rollback())
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
    ch.add_argument("--smooth", type=int, default=1,
                    help="N x N mean of linear sigma0 before dB, N odd (default 1 = off)")
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
    ev.add_argument("--smooth", type=int, default=1,
                    help="evaluate the change map made with this --smooth (default 1)")
    ev.add_argument("--hist", action="store_true",
                    help="class-conditional distributions (EMS flood vs dry) + best single cuts")
    ev.add_argument("--rule", action="append", default=[],
                    help="rule to score, repeatable: preset A-D or e.g. 'dvv<=-1,vv<=-12', 'vv<=otsu', 'vv<=gmm'")
    ev.set_defaults(func=cmd_evaluate)

    ea = sub.add_parser("ems-aoi", help="make an AOI config entry from an EMS vector package")
    ea.add_argument("--zip", required=True)
    ea.add_argument("--name", required=True, help="new AOI name, e.g. shikarpur")
    ea.add_argument("--like", required=True, help="copy event/reference windows from this AOI")
    ea.add_argument("--role", default="train", choices=["train", "holdout"])
    ea.add_argument("--margin", type=float, default=0.01, help="degrees added around the EMS AOI")
    ea.add_argument("--append", action="store_true", help="append to the config (else just print)")
    ea.set_defaults(func=cmd_ems_aoi)

    tp = sub.add_parser("train-pixels", help="leave-one-AOI-out pixel classifiers vs rule A")
    tp.add_argument("--aoi", action="append", required=True, help="repeat; >= 2 AOIs")
    tp.add_argument("--event", required=True)
    tp.add_argument("--flood-pass", required=True)
    tp.add_argument("--smooth", type=int, default=3)
    tp.add_argument("--spacing", type=float, default=20.0)
    tp.add_argument("--samples", type=int, default=200_000, help="training pixels per AOI")
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--no-importance", action="store_true", help="skip permutation importance")
    tp.set_defaults(func=cmd_train_pixels)

    tu = sub.add_parser("train-unet", help="leave-one-AOI-out U-Net vs rule A (GPU if available)")
    tu.add_argument("--aoi", action="append", required=True, help="repeat; >= 2 AOIs")
    tu.add_argument("--event", required=True)
    tu.add_argument("--flood-pass", required=True)
    tu.add_argument("--smooth", type=int, default=3)
    tu.add_argument("--spacing", type=float, default=20.0)
    tu.add_argument("--tile", type=int, default=256)
    tu.add_argument("--batch", type=int, default=16)
    tu.add_argument("--epochs", type=int, default=20)
    tu.add_argument("--tiles-per-epoch", type=int, default=400)
    tu.add_argument("--lr", type=float, default=1e-3)
    tu.add_argument("--base", type=int, default=32, help="channels at the top level (32 -> ~7.8M params)")
    tu.add_argument("--seed", type=int, default=0)
    tu.add_argument("--device", default="auto", help="auto, cuda or cpu")
    tu.set_defaults(func=cmd_train_unet)

    ml = sub.add_parser("mlflow-log", help="ingest LOAO report(s) into MLflow")
    ml.add_argument("report", nargs="+", help="data/ml/.../*_loao.json")
    ml.add_argument("--tag", action="append", default=[], help="key=value, repeatable")
    ml.add_argument("--force", action="store_true", help="log again even if already logged")
    ml.set_defaults(func=cmd_mlflow_log)

    ff = sub.add_parser("fit-final", help="train a pixel model on all AOIs and register it")
    ff.add_argument("--method", required=True, help="logreg, hgb, logreg_bal or hgb_bal")
    ff.add_argument("--report", required=True, help="pixels LOAO JSON that supplies the gate metric")
    ff.add_argument("--aoi", action="append", required=True)
    ff.add_argument("--event", required=True)
    ff.add_argument("--flood-pass", required=True)
    ff.add_argument("--smooth", type=int, default=3)
    ff.add_argument("--spacing", type=float, default=20.0)
    ff.add_argument("--samples", type=int, default=200_000)
    ff.add_argument("--seed", type=int, default=0)
    ff.set_defaults(func=cmd_fit_final)

    rg = sub.add_parser("registry", help="model registry: show, promote (gated), rollback")
    rg.add_argument("action", choices=["show", "promote", "rollback"])
    rg.add_argument("--version", type=int)
    rg.add_argument("--latest", action="store_true", help="promote: use the newest version")
    rg.add_argument("--min-delta", type=float, default=0.0,
                    help="candidate must beat the champion's gate metric by this much")
    rg.add_argument("--force", action="store_true", help="promote without the gate")
    rg.set_defaults(func=cmd_registry)

    args = p.parse_args(argv)
    if args.config is None:
        from .aoi import DEFAULT_CONFIG
        args.config = DEFAULT_CONFIG
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
