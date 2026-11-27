"""Flood classification rules, fixed *before* scoring on held-out areas.

A rule is a comma-separated list of conditions, all of which must hold:

    vv<=-12              flood-pass VV dB <= -12
    dvv<=-1,vv<=-12      also VV change (flood - reference) <= -1 dB
    vh<=-20              flood-pass VH dB <= -20
    vv<=otsu             threshold chosen by Otsu's method on flood-pass VV
    vv<=gmm              threshold from a 2-component Gaussian mixture on flood-pass VV:
                         where the darker component's posterior crosses 0.5 (the Bayes
                         boundary between the fitted classes). Unlike Otsu it models
                         unequal class sizes and spreads.

Otsu and GMM thresholds are computed from the change map's own valid pixels (the
whole raster), never from the labels, so these rules are label-free baselines.

Presets decided on Larkana (EMSR629 AOI01) before the held-out test:
    A = vv<=-12                  best simple rule on Larkana (3x3 smoothing)
    B = dvv<=-1,vv<=-12          does a change condition generalize better?
    C = vv<=otsu                 label-free baseline
Added after the pixel-classifier results (ideal cut varies ~3 dB between AOIs):
    D = vv<=gmm                  label-free, adapts the cut per scene
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

PRESETS = {"A": "vv<=-12", "B": "dvv<=-1,vv<=-12", "C": "vv<=otsu", "D": "vv<=gmm"}
VARS = {"vv": "flood_vv_db", "vh": "flood_vh_db", "dvv": "dvv_db", "dvh": "dvh_db"}
_COND = re.compile(r"^(vv|vh|dvv|dvh)<=(otsu|gmm|[+-]?\d+(?:\.\d+)?)$")


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold: maximizes between-class variance of a 1-D sample.
    The range is clipped to the 0.5-99.5 percentiles so outliers don't squeeze the bins."""
    v = x[np.isfinite(x)]
    if v.size < 2:
        raise ValueError("not enough finite values for Otsu")
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return float(lo)
    hist, edges = np.histogram(np.clip(v, lo, hi), bins=bins, range=(lo, hi))
    p = hist.astype(np.float64) / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)                       # weight of class "<= t"
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1
    k = int(np.argmax(between))
    return float(edges[k + 1])              # upper edge of the last bin in the low class


def gmm_threshold(x: np.ndarray, n_sample: int = 300_000, seed: int = 0) -> float:
    """Fit a 2-component Gaussian mixture to x and return the value between the two
    means where the lower-mean component's posterior equals 0.5."""
    from sklearn.mixture import GaussianMixture

    v = x[np.isfinite(x)]
    if v.size < 10:
        raise ValueError("not enough finite values for a GMM")
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = v[(v >= lo) & (v <= hi)]
    rng = np.random.default_rng(seed)
    if v.size > n_sample:
        v = rng.choice(v, n_sample, replace=False)
    gm = GaussianMixture(n_components=2, random_state=seed).fit(v.reshape(-1, 1))
    means = gm.means_.ravel()
    low = int(np.argmin(means))
    grid = np.linspace(means.min(), means.max(), 2001).reshape(-1, 1)
    post = gm.predict_proba(grid)[:, low]
    k = int(np.argmax(post < 0.5))          # first point where 'low' stops dominating
    if post[0] < 0.5 or k == 0:             # degenerate fit: fall back to the midpoint
        return float(means.mean())
    return float(grid[k, 0])


@dataclass
class Condition:
    var: str            # key in VARS
    threshold: float | None   # None until resolved (otsu / gmm)
    otsu: bool
    gmm: bool = False


@dataclass
class Rule:
    spec: str
    conditions: list[Condition]

    def describe(self) -> str:
        parts = []
        for c in self.conditions:
            t = "?" if c.threshold is None else f"{c.threshold:.2f}"
            parts.append(f"{c.var}<={t}" + (" (otsu)" if c.otsu else " (gmm)" if c.gmm else ""))
        return ", ".join(parts)


def parse_rule(spec: str) -> Rule:
    s = PRESETS.get(spec.strip(), spec).replace(" ", "")
    conds = []
    for part in s.split(","):
        m = _COND.match(part)
        if not m:
            raise ValueError(f"bad condition {part!r} in rule {spec!r} "
                             "(use e.g. vv<=-12, dvv<=-1, vv<=otsu, vv<=gmm, or a preset A-D)")
        var, t = m.groups()
        auto = t in ("otsu", "gmm")
        conds.append(Condition(var, None if auto else float(t), t == "otsu", t == "gmm"))
    return Rule(spec=spec, conditions=conds)


def apply_rule(rule: Rule, bands: dict[str, np.ndarray], valid: np.ndarray,
               valid_vh: np.ndarray | None = None) -> np.ndarray:
    """Resolve any Otsu/GMM thresholds from the bands (label-free), then predict.
    Fills in rule.conditions[i].threshold for those conditions."""
    pred = valid.copy()
    for c in rule.conditions:
        v = bands[VARS[c.var]]
        m = valid if c.var in ("vv", "dvv") or valid_vh is None else valid & valid_vh
        if c.otsu:
            c.threshold = otsu_threshold(v[m])
        elif c.gmm:
            c.threshold = gmm_threshold(v[m])
        pred &= m & (v <= c.threshold)
    return pred
