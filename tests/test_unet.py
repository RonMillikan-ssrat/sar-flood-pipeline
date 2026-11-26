import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sarflood.unet import (
    CHANNELS,
    Grid,
    TrainConfig,
    build_unet,
    count_params,
    fit_norm,
    make_input,
    predict,
    sample_tiles,
    train,
)


def grid(h=64, w=80, seed=0, name="g"):
    """Open water (dark VV) in a block; target = that block. VH partly invalid."""
    rng = np.random.default_rng(seed)
    flood = np.zeros((h, w), bool)
    flood[h // 4: 3 * h // 4, w // 5: w // 2] = True
    vv = np.where(flood, -20.0, -9.0) + rng.normal(0, 1.0, (h, w))
    vh = np.where(flood, -27.0, -17.0) + rng.normal(0, 1.0, (h, w))
    raw = np.stack([vv, vh, np.full((h, w), -9.0), np.full((h, w), -17.0),
                    vv + 9.0, vh + 17.0]).astype(np.float32)
    valid_vh = np.ones((h, w), bool)
    valid_vh[:5, :5] = False
    raw[[1, 3, 5], :5, :5] = np.nan
    return Grid(name, raw, np.ones((h, w), bool), valid_vh, flood.astype(np.float32),
                np.ones((h, w), bool))


def test_param_count_base32():
    n = count_params(build_unet(base=32))
    assert 7.0e6 < n < 8.5e6


def test_forward_shape():
    m = build_unet(base=4).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, len(CHANNELS), 64, 96))
    assert out.shape == (2, 64, 96)


def test_make_input_fills_and_flags():
    g = grid()
    x = make_input(g, fit_norm([g]))
    assert x.shape == (len(CHANNELS), 64, 80) and np.isfinite(x).all()
    assert x[CHANNELS.index("vh"), 0, 0] == 0.0 and x[CHANNELS.index("valid_vh"), 0, 0] == 0.0
    assert x[CHANNELS.index("valid_vh"), 10, 10] == 1.0


def test_sample_tiles_shapes_and_min_labeled():
    g = grid()
    g.loss_mask[:, 40:] = False                  # right half unlabeled
    x = make_input(g, fit_norm([g]))
    X, Y, M = sample_tiles([x], [g], 12, 32, 0.9, np.random.default_rng(0))
    assert X.shape == (12, len(CHANNELS), 32, 32) and Y.shape == M.shape == (12, 32, 32)
    assert (M.mean(axis=(1, 2)) >= 0.9).all()


class Const(torch.nn.Module):
    """Fake model: logit 0 everywhere -> probability 0.5."""

    def forward(self, x):
        return torch.zeros(x.shape[0], x.shape[2], x.shape[3])


@pytest.mark.parametrize("h,w,tile", [(64, 80, 32), (70, 45, 32), (20, 25, 32)])
def test_predict_stitching_covers_grid(h, w, tile):
    g = grid(h, w)
    p = predict(Const(), g, fit_norm([g]), tile=tile, batch=4, device="cpu")
    assert p.shape == (h, w)
    np.testing.assert_allclose(p, 0.5, atol=1e-6)   # weighted average of identical tiles


def test_training_learns_the_block():
    gs = [grid(seed=s, name=f"g{s}") for s in range(2)]
    cfg = TrainConfig(tile=32, batch=8, epochs=6, tiles_per_epoch=64, lr=5e-3, base=8,
                      min_labeled=0.5, device="cpu")
    model, norm, pw = train(gs, cfg)
    assert cfg.log[-1]["loss"] < 0.5 * cfg.log[0]["loss"]
    test = grid(seed=9)
    p = predict(model, test, norm, tile=32, batch=8, device="cpu")
    pred, truth = p >= 0.5, test.target > 0
    iou = (pred & truth).sum() / (pred | truth).sum()
    assert iou > 0.8
    share = np.mean([g.target[g.loss_mask].mean() for g in gs])   # 768 / 5120 = 0.15
    assert pw == pytest.approx((1 - share) / share, rel=1e-4)      # n_dry / n_flood


def test_cli_train_unet(tmp_path, monkeypatch, capsys):
    from sarflood import cli
    from sarflood.change import OUT_BANDS
    from sarflood.labels import LABEL_BANDS
    from tests.test_pixels import synth, write

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    stem = "F__R_20m_s3"
    for name, seed in [("larkana", 1), ("shikarpur", 2)]:
        bands, labels = synth(seed)
        (tmp_path / "change" / name).mkdir(parents=True)
        (tmp_path / "eval" / name).mkdir(parents=True)
        write(tmp_path / "change" / name / f"{stem}_change.tif", OUT_BANDS, bands)
        write(tmp_path / "eval" / name / f"{stem}_EMSR999_AOI01_DEL_v1_vector_labels.tif",
              LABEL_BANDS, labels)
        pf = tmp_path / "pairs" / name / "pakistan-2022.json"
        pf.parent.mkdir(parents=True)
        pf.write_text(json.dumps({"pairs": [{"flood": {"key": "F"}, "reference": {"key": "R"},
                                             "reason": "ok"}]}))
    rc = cli.main(["train-unet", "--aoi", "larkana", "--aoi", "shikarpur", "--event",
                   "pakistan-2022", "--flood-pass", "F", "--tile", "32", "--batch", "8",
                   "--epochs", "2", "--tiles-per-epoch", "16", "--base", "4", "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rule A" in out and "unet" in out and "params" in out
    d = tmp_path / "ml" / "unet"
    rep = json.loads((d / "F_s3_b4_e2_loao.json").read_text())
    assert set(rep["folds"]) == {"larkana", "shikarpur"}
    assert "_prob_grid" not in json.dumps(rep)
    f = rep["folds"]["larkana"]
    assert f["train_aois"] == ["shikarpur"] and len(f["epochs"]) == 2
    assert len(list(d.glob("*_prob.tif"))) == 2 and len(list(d.glob("*_confusion.png"))) == 2
    assert len(list((d / "F_s3_b4_e2_models").glob("unet_heldout_*.pt"))) == 2
