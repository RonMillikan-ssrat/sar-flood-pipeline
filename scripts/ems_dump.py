"""Dump the attribute tables of an EMS vector package, for working out its layout.

    python scripts/ems_dump.py data/ems/zips/<package>.zip [--layer crisis]

For every .dbf in the zip: record count, field names, and per field either the
value counts (<= 25 distinct values) or a few example values. Also prints any
.shp.xml metadata, which in 2017 packages may name the source imagery.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from collections import Counter

import shapefile  # pyshp


def dump_dbf(zf: zipfile.ZipFile, name: str, max_distinct: int = 25) -> None:
    r = shapefile.Reader(dbf=io.BytesIO(zf.read(name)), encoding="utf-8",
                         encodingErrors="replace")
    fields = [f[0] for f in r.fields[1:]]
    recs = r.records()
    print(f"\n### {name.rsplit('/', 1)[-1]}   {len(recs)} records")
    if len(recs) <= 10:  # small tables (e.g. source): print rows, so columns line up
        for rec in recs:
            print("  | " + " | ".join(f"{f}={str(v).strip()}" for f, v in zip(fields, rec)))
        return
    for i, f in enumerate(fields):
        vals = Counter(str(rec[i]).strip() for rec in recs)
        if len(vals) <= max_distinct:
            shown = ", ".join(f"{v!r}:{n}" for v, n in vals.most_common())
            print(f"  {f:<12} {shown}")
        else:
            ex = ", ".join(repr(v) for v, _ in vals.most_common(3))
            print(f"  {f:<12} {len(vals)} distinct, e.g. {ex}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--layer", default="", help="only members whose name contains this")
    a = ap.parse_args()
    with zipfile.ZipFile(a.zip) as zf:
        names = sorted(n for n in zf.namelist() if a.layer in n)
        for n in names:
            if n.lower().endswith(".dbf"):
                dump_dbf(zf, n)
        for n in names:
            if n.lower().endswith(".shp.xml"):
                print(f"\n### {n.rsplit('/', 1)[-1]}")
                print(zf.read(n).decode("utf-8", "replace")[:1500])
        for n in names:
            if n.lower().endswith((".json", ".geojson")):
                print(f"(geojson member: {n.rsplit('/', 1)[-1]}, {zf.getinfo(n).file_size} B)")


if __name__ == "__main__":
    main()
