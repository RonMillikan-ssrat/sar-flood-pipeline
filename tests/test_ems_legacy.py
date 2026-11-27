"""2017-era EMS packages (shapefiles only), mirroring EMSR229 01HOUSTON 01DELINEATION v2."""
import io
import zipfile

import numpy as np
import pytest
import shapefile

from sarflood.ems_zip import aoi_from_zip, source_warnings
from sarflood.labels import rasterize_ems, read_package
from tests.test_labels import CRS, H, T, W, to_lonlat, utm_box

WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,'
             '298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
STEM = "EMSR229_01HOUSTON_01DELINEATION_v2_440000_{}"


def layer(zf, tag, fields, feats, prj=WGS84_PRJ, crs_of_geoms=CRS, prefix="VECTOR/"):
    """Write one polygon shapefile layer into the zip. Geometries are given in UTM
    (test grid) and stored in the layer's own CRS (WGS84 unless prj says otherwise)."""
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON)
    for f in fields:
        w.field(f, "C", 60)
    for g, props in feats:
        if prj == WGS84_PRJ:
            g = to_lonlat(g)
        w.poly(g["coordinates"])
        w.record(*[props.get(f, "") for f in fields])
    w.close()
    base = prefix + STEM.format(tag)
    zf.writestr(base + ".shp", shp.getvalue())
    zf.writestr(base + ".shx", shx.getvalue())
    zf.writestr(base + ".dbf", dbf.getvalue())
    zf.writestr(base + ".prj", prj)


def make_legacy_zip(tmp_path, prj=WGS84_PRJ):
    z = tmp_path / "EMSR229_01HOUSTON_01DELINEATION_MAP_v2_vector.zip"
    with zipfile.ZipFile(z, "w") as zf:
        layer(zf, "area_of_interest", ["name", "act_id"],
              [(utm_box(5, 5, 55, 45), {"name": "Houston", "act_id": "EMSR229"})], prj)
        layer(zf, "crisis_information_poly", ["interpret", "subtype", "src_info", "src_date"], [
            (utm_box(10, 10, 30, 30), {"interpret": "Flooded Area", "subtype": "EM009 - Flood",
                                       "src_info": "Cosmo-Skymed 100 m", "src_date": "2017-08-28"}),
            (utm_box(40, 10, 41, 11), {"interpret": "Flooded Area", "subtype": "EM009 - Flood"}),
            (utm_box(40, 20, 44, 24), {"interpret": "Damaged road", "subtype": "XX - Other"}),
        ], prj)
        layer(zf, "general_information_poly", ["subtype", "source_nam", "src_date", "reliabilit"], [
            (utm_box(0, 0, W, H), {"subtype": "Satellite Footprint", "src_date": "2017-08-28",
                                   "source_nam": "Cosmo-Skymed 3-15m Stripmap imagery"}),
            (utm_box(45, 35, 50, 40), {"subtype": "ZD020 - Void Collection Area",
                                       "reliabilit": "Missing data"}),
        ], prj)
        layer(zf, "hydrography_poly", ["subtype"], [
            (utm_box(32, 5, 36, 45), {"subtype": "BH080 - Lake"}),
            (utm_box(5, 40, 55, 45), {"subtype": "BH090 - Land Subject to Inundation"}),
        ], prj)
    return z


@pytest.mark.parametrize("prj", ["wgs84", "utm"])
def test_legacy_rasterize_same_roles_as_2022(tmp_path, prj):
    from rasterio.crs import CRS as RCRS

    z = make_legacy_zip(tmp_path, WGS84_PRJ if prj == "wgs84" else RCRS.from_string(CRS).to_wkt())
    lab = rasterize_ems(z, (H, W), T, CRS)
    assert lab["flood"][10:30, 10:30].all() and lab["flood"][10, 40] == 1
    assert lab["flood"].sum() == 20 * 20 + 1            # the 'Damaged road' polygon is not flood
    assert lab["aoi"][5:45, 5:55].all() and lab["aoi"].sum() == 50 * 40
    assert lab["not_analysed"][35:40, 45:50].all() and lab["not_analysed"].sum() == 25
    assert lab["perm_water"][5:45, 32:36].all()
    assert lab["perm_water"].sum() == 40 * 4            # BH090 is floodable land, not water


def test_legacy_aoi_info_and_source(tmp_path):
    info = aoi_from_zip(make_legacy_zip(tmp_path))
    assert (info["emsr_id"], info["area_id"], info["locality"]) == ("EMSR229", "01", "Houston")
    assert [r["source_nam"] for r in info["flood_sources"]] == ["Cosmo-Skymed 3-15m Stripmap imagery"]
    assert info["flood_sources"][0]["src_date"] == "2017-08-28"
    assert info["warnings"] == []
    lon0, lat0, lon1, lat1 = info["bounds"]
    assert lon0 < lon1 and lat0 < lat1 and -180 < lon0 < 180


def test_root_level_members_and_reservoir(tmp_path):
    """MONIT packages put the shapefiles at the zip root; BH130 reservoirs are water."""
    z = tmp_path / "EMSR229_01HOUSTON_01DELINEATION_MONIT01_v1_vector.zip"
    with zipfile.ZipFile(z, "w") as zf:
        layer(zf, "area_of_interest", ["name"], [(utm_box(5, 5, 55, 45), {"name": "Houston"})],
              prefix="")
        layer(zf, "crisis_information_poly", ["interpret"],
              [(utm_box(10, 10, 12, 12), {"interpret": "Flooded Area"})], prefix="")
        layer(zf, "hydrography_poly", ["subtype"],
              [(utm_box(20, 20, 22, 22), {"subtype": "BH130 - Reservoir"})], prefix="")
    lay = read_package(z)
    assert len(lay["flood"]) == 1 and len(lay["perm_water"]) == 1 and lay["not_analysed"] == []
    lab = rasterize_ems(z, (H, W), T, CRS)
    assert lab["flood"].sum() == 4 and lab["perm_water"].sum() == 4


def test_not_a_package(tmp_path):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", "hi")
    with pytest.raises(ValueError, match="not a recognised"):
        read_package(z)


def test_source_warning_for_stale_dmg_source():
    """Mirrors EMSR631 AOI01: polygons point to a 2021 pre-event WorldView-2 image."""
    recs = [
        {"source_nam": "WorldView-2", "src_date": "02/05/2021", "source_tm": "T06:16:33Z",
         "eventphase": "Pre-event", "src_id": 1},
        {"source_nam": "PlanetScope", "src_date": "09/09/2022", "source_tm": "T05:36:42Z",
         "eventphase": "Post-event", "src_id": 2},
        {"source_nam": "Open Street Map", "src_date": "10/09/2022", "source_tm": "Not Applicable",
         "eventphase": "Pre-event", "src_id": 994},
    ]
    w = source_warnings([recs[0]], recs)
    assert len(w) == 1 and "PlanetScope 09/09/2022" in w[0] and "WorldView-2" in w[0]
    assert source_warnings([recs[1]], recs) == []           # flood source is the latest image
    s1 = [{"source_nam": "Sentinel-1", "src_date": "30/08/2022", "source_tm": "T01:25:00Z"}]
    assert "circular" in source_warnings(s1, s1)[0]


def test_flood_frac_is_float_mask(tmp_path):
    lab = rasterize_ems(make_legacy_zip(tmp_path), (H, W), T, CRS)
    assert lab["flood_frac"].dtype == np.float32 and lab["flood_frac"].max() == pytest.approx(1.0)


def _line_layer(zf, tag, feats, prefix="VECTOR/"):
    """Polyline layer in WGS84; feats: [(utm (x, y) list, subtype)]."""
    from rasterio.warp import transform

    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYLINE)
    w.field("subtype", "C", 40)
    for pts, sub in feats:
        lon, lat = transform(CRS, "EPSG:4326", [p[0] for p in pts], [p[1] for p in pts])
        w.line([list(zip(lon, lat))])
        w.record(sub)
    w.close()
    base = prefix + STEM.format(tag)
    for ext, b in ((".shp", shp), (".shx", shx), (".dbf", dbf)):
        zf.writestr(base + ext, b.getvalue())
    zf.writestr(base + ".prj", WGS84_PRJ)


def test_river_lines_buffered_into_permanent_water(tmp_path):
    from sarflood.labels import labels_version
    from tests.test_labels import RES, X0, Y0

    z = make_legacy_zip(tmp_path)
    y = Y0 - 20.5 * RES                                  # horizontal line through row 20
    with zipfile.ZipFile(z, "a") as zf:
        _line_layer(zf, "hydrography_line", [
            ([(X0 + 2 * RES, y), (X0 + 58 * RES, y)], "BH140 - River"),
            ([(X0 + 2 * RES, Y0 - 2.5 * RES), (X0 + 58 * RES, Y0 - 2.5 * RES)], "BI020 - Dam"),
        ])
    base = rasterize_ems(z, (H, W), T, CRS, river_buffer_m=0)
    lab = rasterize_ems(z, (H, W), T, CRS)                         # default 30 m
    added = (lab["perm_water"] > 0) & (base["perm_water"] == 0)
    rows = np.flatnonzero(added.any(axis=1))
    assert set(rows) <= {18, 19, 20, 21, 22} and {19, 20, 21} <= set(rows)   # +-30 m = +-1.5 px
    assert not added[2].any()                                       # dams are not water
    assert labels_version(30.0) == "3" and labels_version(60.0) == "3-rb60"


def test_2022_hydrographyL_streams_buffered(tmp_path):
    import json

    from tests.test_labels import RES, X0, Y0, make_zip

    z = make_zip(tmp_path)
    y = Y0 - 42.5 * RES
    line = {"type": "LineString", "coordinates": [(X0 + 5 * RES, y), (X0 + 55 * RES, y)]}
    from tests.test_labels import to_lonlat
    with zipfile.ZipFile(z, "a") as zf:
        zf.writestr("EMSR999_AOI01_DEL_PRODUCT_hydrographyL_r1_v1.json", json.dumps({
            "type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {"obj_type": "BH141-Stream"},
                 "geometry": to_lonlat(line)}]}))
    base = rasterize_ems(z, (H, W), T, CRS, river_buffer_m=0)
    lab = rasterize_ems(z, (H, W), T, CRS)
    assert lab["perm_water"].sum() > base["perm_water"].sum()
    assert lab["perm_water"][42, 10] == 1 and base["perm_water"][42, 10] == 0
