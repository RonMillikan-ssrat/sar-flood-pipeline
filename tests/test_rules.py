import json
import zipfile

import numpy as np
import pytest

from sarflood.rules import PRESETS, apply_rule, otsu_threshold, parse_rule


def test_otsu_bimodal():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-20, 1.5, 4000), rng.normal(-9, 1.5, 6000), [np.nan] * 5])
    t = otsu_threshold(x)
    assert -16.5 < t < -12.5            # between the two modes


def test_otsu_needs_data():
    with pytest.raises(ValueError):
        otsu_threshold(np.array([np.nan]))


def test_parse_presets_and_specs():
    assert PRESETS["A"] == "vv<=-12"
    r = parse_rule("B")
    assert [(c.var, c.threshold, c.otsu) for c in r.conditions] == [("dvv", -1.0, False),
                                                                     ("vv", -12.0, False)]
    r = parse_rule("vv <= otsu")
    assert r.conditions[0].otsu and r.conditions[0].threshold is None
    with pytest.raises(ValueError, match="bad condition"):
        parse_rule("vv<-12")
    with pytest.raises(ValueError):
        parse_rule("foo<=1")


def test_apply_rule_and_resolve_otsu():
    fvv = np.array([[-22.0, -20.0, -8.0, -9.0]])
    dvv = np.array([[-5.0, 2.0, 0.0, -6.0]])
    bands = {"flood_vv_db": fvv, "dvv_db": dvv, "flood_vh_db": fvv - 7, "dvh_db": dvv}
    valid = np.array([[True, True, True, False]])
    assert apply_rule(parse_rule("A"), bands, valid).tolist() == [[True, True, False, False]]
    assert apply_rule(parse_rule("B"), bands, valid).tolist() == [[True, False, False, False]]
    r = parse_rule("C")
    pred = apply_rule(r, bands, valid)
    assert -20 <= r.conditions[0].threshold < -8          # resolved from the data
    assert pred.tolist() == [[True, True, False, False]]
    assert "(otsu)" in r.describe()


def test_ems_aoi_from_zip_and_yaml(tmp_path):
    from datetime import date

    from sarflood.aoi import AOI, Event
    from sarflood.ems_zip import aoi_from_zip, yaml_entry

    aoi = {"type": "FeatureCollection", "features": [{"type": "Feature",
           "properties": {"emsr_id": "EMSR629", "area_id": "02", "locality": "Shikarpur"},
           "geometry": {"type": "Polygon", "coordinates": [[[68.4512, 27.8123], [68.8034, 27.8123],
                        [68.8034, 28.0991], [68.4512, 28.0991], [68.4512, 27.8123]]]}}]}
    z = tmp_path / "EMSR629_AOI02_DEL_PRODUCT_r1_RTP01_v1_vector.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("EMSR629_AOI02_DEL_PRODUCT_areaOfInterestA_r1_v1.json", json.dumps(aoi))
    info = aoi_from_zip(z)
    assert info["bbox"] == [68.44, 27.80, 68.82, 28.11]   # +0.01 margin, rounded outward
    like = AOI("larkana", "train", (68.05, 27.44, 68.46, 27.72), events=(
        Event("pakistan-2022", date(2022, 8, 15), date(2022, 9, 30),
              reference_start=date(2022, 3, 1), reference_end=date(2022, 6, 1)),))
    text = yaml_entry("shikarpur", info, like)
    assert "  shikarpur:\n    role: train" in text
    assert "bbox: [68.44, 27.80, 68.82, 28.11]" in text
    assert 'reference: {start: "2022-03-01", end: "2022-06-01"}' in text

    # The generated text must load through the real config parser.
    from sarflood.aoi import load_aois
    cfg = tmp_path / "aois.yaml"
    cfg.write_text("aois:\n" + text)
    loaded = load_aois(cfg)["shikarpur"]
    assert loaded.bbox == (68.44, 27.80, 68.82, 28.11)
    assert loaded.events[0].reference_end == date(2022, 6, 1)


def test_cli_ems_aoi_append(tmp_path, capsys):
    import shutil

    from sarflood import cli
    from sarflood.aoi import DEFAULT_CONFIG, load_aois

    cfg = tmp_path / "aois.yaml"
    shutil.copy(DEFAULT_CONFIG, cfg)
    aoi = {"type": "FeatureCollection", "features": [{"type": "Feature",
           "properties": {"emsr_id": "EMSR629", "area_id": "03", "locality": "Jacobabad"},
           "geometry": {"type": "Polygon", "coordinates": [[[68.3, 28.2], [68.6, 28.2],
                        [68.6, 28.4], [68.3, 28.4], [68.3, 28.2]]]}}]}
    z = tmp_path / "EMSR629_AOI03_DEL_PRODUCT_r1_RTP01_v1_vector.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("EMSR629_AOI03_DEL_PRODUCT_areaOfInterestA_r1_v1.json", json.dumps(aoi))
    name = "test_aoi_zz"   # must not collide with anything in the live config
    assert name not in load_aois(cfg)
    args = ["--config", str(cfg), "ems-aoi", "--zip", str(z), "--name", name,
            "--like", "larkana"]
    assert cli.main(args) == 0 and name not in load_aois(cfg)              # print only
    assert cli.main(args + ["--append"]) == 0
    assert load_aois(cfg)[name].bbox == (68.29, 28.19, 68.61, 28.41)
    assert cli.main(args + ["--append"]) == 2                              # no duplicates
    assert "already exists" in capsys.readouterr().err


def test_flood_source_via_dmg_src_id_not_eventphase(tmp_path):
    """Mirrors EMSR629 AOI03: the flood image is tagged 'Pre-event' by mistake."""
    import io

    import shapefile

    from sarflood.ems_zip import aoi_from_zip

    def dbf(fields, rows):
        b = io.BytesIO()
        w = shapefile.Writer(dbf=b)
        for f in fields:
            w.field(*f)
        for r in rows:
            w.record(*r)
        w.close()
        return b.getvalue()

    src = dbf([("source_nam", "C", 40), ("src_date", "C", 12), ("source_tm", "C", 12),
               ("eventphase", "C", 16), ("src_id", "N", 6, 0)],
              [("Sentinel-2", "24/03/2022", "T06:11:23Z", "Pre-event", 1),
               ("SPOT-6-7", "30/08/2022", "T05:44:37Z", "Pre-event", 2)])
    obs = dbf([("notation", "C", 20), ("dmg_src_id", "C", 4)],
              [("Flooded area", "2"), ("Flooded area", "2")])
    aoi = {"type": "FeatureCollection", "features": [{"type": "Feature",
           "properties": {"emsr_id": "EMSR629", "area_id": "03", "locality": "Jacobabad"},
           "geometry": {"type": "Polygon", "coordinates": [[[68.3, 28.2], [68.6, 28.2],
                        [68.6, 28.4], [68.3, 28.4], [68.3, 28.2]]]}}]}
    z = tmp_path / "EMSR629_AOI03_DEL_PRODUCT_r1_RTP01_v1_vector.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("EMSR629_AOI03_DEL_PRODUCT_areaOfInterestA_r1_v1.json", json.dumps(aoi))
        zf.writestr("EMSR629_AOI03_DEL_PRODUCT_source_r1_v1.dbf", src)
        zf.writestr("EMSR629_AOI03_DEL_PRODUCT_observedEventA_r1_v1.dbf", obs)
    info = aoi_from_zip(z)
    assert [r["source_nam"] for r in info["flood_sources"]] == ["SPOT-6-7"]
