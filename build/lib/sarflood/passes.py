"""Group Sentinel-1 slices into passes, and pair flood passes with dry references.

A *pass* is every slice from one datatake: same platform, absolute orbit and
datatake ID. The slices are contiguous pieces of one strip and get assembled
before terrain correction.

A *reference* for a flood pass must share its viewing geometry: same relative
orbit (track), orbit direction and polarisation. Then a change in backscatter
comes from the surface (water), not from a different incidence angle.

Product name layout (fields split on "_"):
    S1A _ IW _ GRDH _ 1SDV _ 20220830T012551 _ 20220830T012616 _ 044775 _ 0558C6 _ E8B2.SAFE
    0     1    2      3      4                 5                 6        7        8
    plat  mode type   lvl+pol start             stop              abs orb  datatake product id
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from shapely import wkt as shp_wkt
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .catalog import Scene


@dataclass(frozen=True)
class NameFields:
    platform: str
    absolute_orbit: int
    datatake: str
    product_id: str
    slice_key: str  # the name without the product ID: identical for reprocessed copies


def parse_name(name: str) -> NameFields:
    stem = name.removesuffix(".SAFE")
    parts = stem.split("_")
    if len(parts) != 9 or not parts[0].startswith("S1"):
        raise ValueError(f"unexpected Sentinel-1 product name: {name}")
    return NameFields(
        platform=parts[0],
        absolute_orbit=int(parts[6]),
        datatake=parts[7].upper(),
        product_id=parts[8],
        slice_key="_".join(parts[:8]),
    )


def parse_footprint(raw: str | None) -> BaseGeometry | None:
    """CDSE returns footprints like  geography'SRID=4326;POLYGON((...))'  ."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("geography'"):
        s = s[len("geography'"):].rstrip("'")
    if s.upper().startswith("SRID="):
        s = s.split(";", 1)[1]
    return shp_wkt.loads(s)


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso)  # 3.11+ parses a trailing "Z"


def dedupe(scenes: list[Scene]) -> list[Scene]:
    """Keep one copy per slice. Reprocessed copies share the name up to the
    product ID; keep the most recently published (ties: highest product ID)."""
    best: dict[str, Scene] = {}
    for s in scenes:
        key = parse_name(s.name).slice_key
        cur = best.get(key)
        rank = (s.published or "", parse_name(s.name).product_id)
        if cur is None or rank > (cur.published or "", parse_name(cur.name).product_id):
            best[key] = s
    return list(best.values())


@dataclass(frozen=True)
class Pass:
    platform: str
    absolute_orbit: int
    datatake: str
    relative_orbit: int | None
    orbit_direction: str | None
    polarisation: str
    start: datetime
    end: datetime
    slices: tuple[Scene, ...]
    contiguous: bool  # every slice starts where the previous one stopped (1 s tolerance)
    footprint: BaseGeometry | None

    @property
    def key(self) -> str:
        return f"{self.platform}_{self.absolute_orbit:06d}_{self.datatake}"

    @property
    def track(self) -> tuple:
        """What must match between a flood pass and its reference."""
        return (self.relative_orbit, self.orbit_direction, self.polarisation)

    def coverage(self, aoi: BaseGeometry) -> float:
        """Fraction of the AOI inside this pass (in degrees²; fine as a ratio
        for small AOIs, not for exact areas)."""
        if self.footprint is None or aoi.area == 0:
            return 0.0
        return self.footprint.intersection(aoi).area / aoi.area

    def summary(self, aoi: BaseGeometry | None = None) -> dict:
        d = {
            "key": self.key,
            "start": self.start.isoformat(),
            "relative_orbit": self.relative_orbit,
            "orbit_direction": self.orbit_direction,
            "polarisation": self.polarisation,
            "n_slices": len(self.slices),
            "contiguous": self.contiguous,
            "slices": [s.name for s in self.slices],
            "ids": [s.id for s in self.slices],
        }
        if aoi is not None:
            d["aoi_coverage"] = round(self.coverage(aoi), 3)
        return d


def group_passes(scenes: list[Scene], gap_tolerance_s: float = 1.0) -> list[Pass]:
    groups: dict[tuple, list[Scene]] = {}
    for s in dedupe(scenes):
        f = parse_name(s.name)
        groups.setdefault((f.platform, f.absolute_orbit, f.datatake), []).append(s)

    passes = []
    for (platform, abs_orb, dt), slices in groups.items():
        slices.sort(key=lambda s: s.start)
        contiguous = all(
            abs((_ts(b.start) - _ts(a.end)).total_seconds()) <= gap_tolerance_s
            for a, b in pairwise(slices)
        )
        # Metadata should agree across slices of one datatake; flag it if not.
        tracks = {(s.relative_orbit, s.orbit_direction, s.polarisation) for s in slices}
        if len(tracks) != 1:
            raise ValueError(f"{platform}_{abs_orb}_{dt}: slices disagree on track: {tracks}")
        rel, direction, pol = tracks.pop()
        geoms = [g for g in (parse_footprint(s.footprint) for s in slices) if g is not None]
        passes.append(Pass(
            platform=platform, absolute_orbit=abs_orb, datatake=dt,
            relative_orbit=rel, orbit_direction=direction, polarisation=pol,
            start=_ts(slices[0].start), end=_ts(slices[-1].end),
            slices=tuple(slices), contiguous=contiguous,
            footprint=unary_union(geoms) if geoms else None,
        ))
    return sorted(passes, key=lambda p: p.start)


@dataclass(frozen=True)
class Pairing:
    flood: Pass
    reference: Pass | None
    reason: str  # why the reference was chosen, or why none was


def select_reference(flood: Pass, candidates: list[Pass], aoi: BaseGeometry,
                     window_start: datetime, window_end: datetime,
                     min_coverage: float = 0.5, same_platform: bool = False) -> Pairing:
    """Pick the reference for one flood pass.

    Rules, in order: same track (relative orbit, direction, polarisation);
    sensed inside the reference window and before the flood pass; covers at
    least `min_coverage` of the AOI; optionally same satellite. Among those,
    take the latest. It is closest in time to the flood, so it has the least
    unrelated change (crops, construction)."""
    same_track = [c for c in candidates if c.track == flood.track and c.key != flood.key]
    if not same_track:
        return Pairing(flood, None, f"no candidate on track {flood.track}")
    in_window = [c for c in same_track
                 if window_start <= c.start < window_end and c.start < flood.start]
    if not in_window:
        return Pairing(flood, None, f"{len(same_track)} on track, none in reference window")
    if same_platform:
        in_window = [c for c in in_window if c.platform == flood.platform]
        if not in_window:
            return Pairing(flood, None, "no same-platform candidate in window")
    covering = [c for c in in_window if c.coverage(aoi) >= min_coverage]
    if not covering:
        best = max(c.coverage(aoi) for c in in_window)
        return Pairing(flood, None, f"best coverage {best:.2f} < {min_coverage}")
    ref = max(covering, key=lambda c: c.start)
    days = (flood.start - ref.start).days
    return Pairing(flood, ref, f"latest of {len(covering)} qualifying; {days} days before flood")
