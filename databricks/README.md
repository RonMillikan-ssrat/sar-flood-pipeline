# Databricks lakehouse layer

Acquisition, SNAP terrain correction, change maps and EMS labels stay on the Ubuntu
box (Free Edition can't reach CDSE). Their products, one change/labels GeoTIFF pair
per AOI and event, land in a Unity Catalog Volume, and Databricks builds the
medallion tables from there:

| Layer | Table | Content |
|---|---|---|
| landing | `/Volumes/<catalog>/<schema>/raw/pairs/<aoi>/<stem>/` | change.tif, labels.tif, manifest.json (written last) |
| bronze | `<schema>.bronze_pairs` | one row per manifest; grid mismatches quarantined |
| silver | `<schema>.silver_pixels` | one row per scored pixel: `features.FEATURES`, y, frac (from `load_aoi`) |
| gold | `<schema>.gold_*` | pure/balanced LOAO training samples, per-AOI feature distributions |

## Phase 0: spike

```bash
uv add --optional databricks databricks-sdk     # once
databricks auth login --host https://<your-workspace>.cloud.databricks.com
databricks catalogs list                        # confirm the catalog name ("workspace"?)

uv build --wheel
python scripts/upload_to_volume.py --dry-run
python scripts/upload_to_volume.py --aoi sindh --wheel dist/sarflood-0.1.0-py3-none-any.whl

cd databricks
databricks bundle validate
databricks bundle deploy
databricks bundle run sarflood_phase0_spike
```

The run ends with a PASS/FAIL/INFO table. Its results decide whether silver
runs `load_aoi` on serverless (expected) or on Ubuntu with Parquet uploads
(fallback), and which `tracking.py` settings become configurable.
