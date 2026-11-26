"""Leave-one-AOI-out evaluation of per-pixel flood classifiers vs rule A.

For each AOI in turn: train on a sample of 'pure' pixels (EMS flood fraction
<= 0.2 or >= 0.8) from the other AOIs, then score on *all* scored pixels of the
held-out AOI with a fixed 0.5 probability cut. Rule A (VV <= -12 dB) is scored on
exactly the same pixels for a like-for-like comparison.

Holding out whole AOIs (~960 km2 each) matters: random pixel splits would put
near-duplicate neighbours in train and test and inflate the score.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURES, AoiData

RULE_A_DB = -12.0


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())

    def div(a, b):
        return float(a / b) if b else float("nan")

    return {"iou": div(tp, tp + fp + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "tp": tp, "fp": fp, "fn": fn, "n": int(pred.size)}


def sample_pure(d: AoiData, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    pure = np.flatnonzero((d.frac <= 0.2) | (d.frac >= 0.8))
    take = pure if pure.size <= n else rng.choice(pure, n, replace=False)
    return d.X[take], d.y[take]


def make_models(seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              early_stopping=True, random_state=seed),
    }


def permutation_importance_iou(model, X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                               n: int = 200_000) -> dict[str, float]:
    """Drop in IoU when one feature's values are shuffled, on a held-out subsample."""
    if X.shape[0] > n:
        take = rng.choice(X.shape[0], n, replace=False)
        X, y = X[take], y[take]
    base = metrics(model.predict_proba(X)[:, 1] >= 0.5, y)["iou"]
    out = {}
    for j, name in enumerate(FEATURES):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        out[name] = base - metrics(model.predict_proba(Xp)[:, 1] >= 0.5, y)["iou"]
    return out


def leave_one_aoi_out(data: list[AoiData], n_train: int = 200_000, seed: int = 0,
                      importance: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    vv_col = FEATURES.index("vv")
    results = {"features": FEATURES, "n_train_per_aoi": n_train, "folds": {}}
    for test in data:
        train = [d for d in data if d is not test]
        parts = [sample_pure(d, n_train, rng) for d in train]
        Xtr = np.concatenate([p[0] for p in parts])
        ytr = np.concatenate([p[1] for p in parts])
        fold = {"train_aois": [d.name for d in train], "n_train": int(ytr.size),
                "train_flood_share": float(ytr.mean()),
                "rule_A": metrics(test.X[:, vv_col] <= RULE_A_DB, test.y), "models": {}}
        for name, model in make_models(seed).items():
            model.fit(Xtr, ytr)
            prob = model.predict_proba(test.X)[:, 1]
            entry = {"metrics": metrics(prob >= 0.5, test.y)}
            if importance and name == "hgb":
                entry["perm_importance_iou"] = permutation_importance_iou(model, test.X, test.y, rng)
            if name == "logreg":
                lr = model[-1]
                entry["std_coefficients"] = dict(zip(FEATURES, map(float, lr.coef_[0])))
            entry["_prob"] = prob      # removed before JSON; used for maps
            fold["models"][name] = entry
        results["folds"][test.name] = fold
    return results
