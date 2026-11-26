"""A small U-Net for flood segmentation, trained leave-one-AOI-out.

Inputs (8 channels, from the change map written by `sarflood change`):
    vv, vh, ref_vv, ref_vh, dvv, dvh   dB, standardized with training-AOI statistics;
                                       invalid pixels set to 0 after standardizing
    valid_vv, valid_vh                 0/1 flags, so the network knows what was filled
Target: EMS flood (fraction >= 0.5).
Loss mask: inside the EMS AOI, assessed, not permanent water, VV-valid, and not a
mixed pixel (0.2 < flood fraction < 0.8). Everything else is ignored by the loss.
Loss: binary cross-entropy, flood class weighted by n_dry / n_flood of the
training pixels (class balance).

Architecture: 4 down / 4 up levels, two 3x3 conv + BatchNorm + ReLU per level,
channels base*(1, 2, 4, 8, 16); base=32 gives about 7.8 M parameters.
Prediction: overlapping tiles (stride = 3/4 tile), probabilities averaged with a
smooth window so tile edges don't show.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

IN_BANDS = ["flood_vv_db", "flood_vh_db", "ref_vv_db", "ref_vh_db", "dvv_db", "dvh_db"]
CHANNELS = ["vv", "vh", "ref_vv", "ref_vh", "dvv", "dvh", "valid_vv", "valid_vh"]


@dataclass
class Grid:
    """One AOI on its change-map grid."""
    name: str
    raw: np.ndarray        # (6, H, W) dB, NaN where invalid
    valid_vv: np.ndarray   # (H, W) bool
    valid_vh: np.ndarray   # (H, W) bool
    target: np.ndarray     # (H, W) float32 0/1
    loss_mask: np.ndarray  # (H, W) bool: pixels that count in the loss


def load_grid(name: str, change_tif, labels_tif) -> Grid:
    import rasterio

    with rasterio.open(change_tif) as src:
        names = list(src.descriptions)
        raw = np.stack([src.read(names.index(b) + 1) for b in IN_BANDS]).astype(np.float32)
        valid_vv = src.read(names.index("valid") + 1) > 0
        valid_vh = src.read(names.index("valid_vh") + 1) > 0
        shape, transform = src.shape, src.transform
    with rasterio.open(labels_tif) as src:
        if src.shape != shape or src.transform != transform:
            raise ValueError(f"{name}: labels grid differs from change grid")
        ln = list(src.descriptions)
        lab = {n: src.read(ln.index(n) + 1) for n in ln}
    frac = lab["flood_frac"]
    mask = ((lab["aoi"] > 0) & (lab["not_analysed"] == 0) & (lab["perm_water"] == 0)
            & valid_vv & ((frac <= 0.2) | (frac >= 0.8)))
    return Grid(name, raw, valid_vv, valid_vh, (lab["flood"] > 0).astype(np.float32), mask)


@dataclass
class Norm:
    mean: np.ndarray   # (6,)
    std: np.ndarray    # (6,)

    def to_dict(self) -> dict:
        return {"bands": IN_BANDS, "mean": self.mean.tolist(), "std": self.std.tolist()}


def fit_norm(grids: list[Grid]) -> Norm:
    """Per-band mean/std over training pixels that count in the loss (finite only)."""
    means, stds = [], []
    for b in range(len(IN_BANDS)):
        v = np.concatenate([g.raw[b][g.loss_mask] for g in grids])
        v = v[np.isfinite(v)]
        means.append(v.mean())
        stds.append(max(v.std(), 1e-6))
    return Norm(np.array(means, np.float32), np.array(stds, np.float32))


def make_input(g: Grid, norm: Norm) -> np.ndarray:
    """(8, H, W) float32 network input."""
    x = (g.raw - norm.mean[:, None, None]) / norm.std[:, None, None]
    valid = np.isfinite(x)
    x = np.where(valid, x, 0.0)
    return np.concatenate([x, g.valid_vv[None].astype(np.float32),
                           g.valid_vh[None].astype(np.float32)]).astype(np.float32)


# ---------------------------------------------------------------- model

def build_unet(in_ch: int = len(CHANNELS), base: int = 32):
    import torch
    from torch import nn

    def block(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = [base * m for m in (1, 2, 4, 8, 16)]
            self.enc = nn.ModuleList([block(in_ch, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
            self.pool = nn.MaxPool2d(2)
            self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2)
                                     for i in reversed(range(4))])
            self.dec = nn.ModuleList([block(2 * c[i], c[i]) for i in reversed(range(4))])
            self.head = nn.Conv2d(c[0], 1, 1)

        def forward(self, x):
            skips = []
            for i, enc in enumerate(self.enc):
                x = enc(x if i == 0 else self.pool(x))
                skips.append(x)
            x = skips.pop()
            for up, dec in zip(self.up, self.dec):
                x = up(x)
                x = dec(torch.cat([x, skips.pop()], dim=1))
            return self.head(x)[:, 0]          # (N, H, W) logits

    return UNet()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- training

@dataclass
class TrainConfig:
    tile: int = 256
    batch: int = 16
    epochs: int = 20
    tiles_per_epoch: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    base: int = 32
    min_labeled: float = 0.25   # a training tile needs this share of loss pixels
    seed: int = 0
    device: str = "auto"
    log: list = field(default_factory=list)


def _pick_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sample_tiles(inputs: list[np.ndarray], grids: list[Grid], n: int, tile: int,
                 min_labeled: float, rng: np.random.Generator):
    """n random tiles (with flips / 90-degree rotations) from the training AOIs,
    each with at least min_labeled of its pixels in the loss mask."""
    X = np.empty((n, inputs[0].shape[0], tile, tile), np.float32)
    Y = np.empty((n, tile, tile), np.float32)
    M = np.empty((n, tile, tile), np.float32)
    k, tries = 0, 0
    while k < n:
        tries += 1
        if tries > 200 * n:
            raise RuntimeError("could not find enough labeled tiles; lower min_labeled or tile")
        i = int(rng.integers(len(grids)))
        g, x = grids[i], inputs[i]
        h, w = g.target.shape
        if h < tile or w < tile:
            raise ValueError(f"{g.name}: grid {h}x{w} smaller than tile {tile}")
        r, c = int(rng.integers(h - tile + 1)), int(rng.integers(w - tile + 1))
        m = g.loss_mask[r:r + tile, c:c + tile]
        if m.mean() < min_labeled:
            continue
        xt, yt, mt = x[:, r:r + tile, c:c + tile], g.target[r:r + tile, c:c + tile], m
        rot = int(rng.integers(4))
        flip = bool(rng.integers(2))
        xt, yt, mt = (np.rot90(a, rot, axes=(-2, -1)) for a in (xt, yt, mt))
        if flip:
            xt, yt, mt = xt[..., ::-1], yt[..., ::-1], mt[..., ::-1]
        X[k], Y[k], M[k] = xt, yt, mt
        k += 1
    return X, Y, M


def train(train_grids: list[Grid], cfg: TrainConfig):
    import torch

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = _pick_device(cfg.device)
    norm = fit_norm(train_grids)
    inputs = [make_input(g, norm) for g in train_grids]
    n_pos = sum(float(g.target[g.loss_mask].sum()) for g in train_grids)
    n_all = sum(float(g.loss_mask.sum()) for g in train_grids)
    pos_weight = torch.tensor((n_all - n_pos) / max(n_pos, 1.0), device=device)

    model = build_unet(base=cfg.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(cfg.tiles_per_epoch / cfg.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    use_amp = device.type == "cuda"   # fp16 autocast; Turing (RTX 20xx) has no bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    model.train()
    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        X, Y, M = sample_tiles(inputs, train_grids, cfg.tiles_per_epoch, cfg.tile,
                               cfg.min_labeled, rng)
        total, count = 0.0, 0.0
        for s in range(0, len(X), cfg.batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[s:s + cfg.batch])).to(device)
            yb = torch.from_numpy(np.ascontiguousarray(Y[s:s + cfg.batch])).to(device)
            mb = torch.from_numpy(np.ascontiguousarray(M[s:s + cfg.batch])).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
            loss_px = bce(logits.float(), yb)
            loss = (loss_px * mb).sum() / mb.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * float(mb.sum())
            count += float(mb.sum())
        cfg.log.append({"epoch": epoch + 1, "loss": total / max(count, 1.0),
                        "seconds": time.perf_counter() - t0})
    return model, norm, float(pos_weight)


def _window(tile: int) -> np.ndarray:
    """2-D weight that is ~1 in the middle and tapers toward tile edges."""
    w1 = np.hanning(tile + 2)[1:-1]
    return np.outer(w1, w1).astype(np.float32) + 1e-3


def predict(model, g: Grid, norm: Norm, tile: int = 256, batch: int = 16,
            device: str = "auto") -> np.ndarray:
    """Flood probability for every pixel of the grid, from overlapping tiles."""
    import torch

    dev = _pick_device(device)
    x = make_input(g, norm)
    _, h, w = x.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    H, W = x.shape[1:]
    stride = max(1, tile * 3 // 4)
    rows = list(range(0, H - tile + 1, stride))
    cols = list(range(0, W - tile + 1, stride))
    if rows[-1] != H - tile:
        rows.append(H - tile)
    if cols[-1] != W - tile:
        cols.append(W - tile)
    coords = [(r, c) for r in rows for c in cols]
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    win = _window(tile)
    model.eval()
    use_amp = dev.type == "cuda"
    with torch.no_grad():
        for s in range(0, len(coords), batch):
            chunk = coords[s:s + batch]
            xb = torch.from_numpy(np.stack([x[:, r:r + tile, c:c + tile] for r, c in chunk])).to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                p = torch.sigmoid(model(xb).float()).cpu().numpy()
            for (r, c), pt in zip(chunk, p):
                acc[r:r + tile, c:c + tile] += pt * win
                wsum[r:r + tile, c:c + tile] += win
    return (acc / wsum)[:h, :w]


# ---------------------------------------------------------------- evaluation

def leave_one_aoi_out_unet(grids: list[Grid], evals: list, cfg: TrainConfig,
                           save_dir=None) -> dict:
    """grids[i] and evals[i] (features.AoiData) describe the same AOI. Each fold trains
    on the other AOIs and scores the held-out one on exactly evals[i]'s pixels, the
    same pixels used for rule A and the pixel classifiers."""
    import dataclasses

    import torch

    from .features import FEATURES
    from .pixels import RULE_A_DB, fewshot_iou, metrics, threshold_free

    vv_col = FEATURES.index("vv")
    res = {"config": {k: v for k, v in dataclasses.asdict(cfg).items() if k != "log"},
           "channels": CHANNELS, "folds": {}}
    for i, (g, ev) in enumerate(zip(grids, evals)):
        train_g = [x for j, x in enumerate(grids) if j != i]
        fold_cfg = dataclasses.replace(cfg, log=[])
        t0 = time.perf_counter()
        model, norm, pos_weight = train(train_g, fold_cfg)
        train_s = time.perf_counter() - t0
        prob_grid = predict(model, g, norm, cfg.tile, cfg.batch, cfg.device)
        prob = prob_grid.ravel()[ev.idx]
        vv = ev.X[:, vv_col]
        rule_a = {**metrics(vv <= RULE_A_DB, ev.y), **threshold_free(-vv, ev.y)}
        rule_a["best_cut"] = -rule_a["best_cut"]
        res["folds"][g.name] = {
            "train_aois": [x.name for x in train_g], "params": count_params(model),
            "pos_weight": pos_weight, "norm": norm.to_dict(), "train_seconds": train_s,
            "epochs": fold_cfg.log, "rule_A": rule_a,
            "unet": {**metrics(prob >= 0.5, ev.y), **threshold_free(prob, ev.y)},
            "fewshot": {"vv": fewshot_iou(vv, ev.y, np.random.default_rng(cfg.seed)),
                        "unet": fewshot_iou(-prob, ev.y, np.random.default_rng(cfg.seed + 1))},
            "_prob_grid": prob_grid,
        }
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "norm": norm.to_dict(),
                        "base": cfg.base, "channels": CHANNELS},
                       save_dir / f"unet_heldout_{g.name}.pt")
    return res
