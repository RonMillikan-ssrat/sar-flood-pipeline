"""Build and run SNAP gpt graphs for terrain-corrected sigma0 of one pass.

Chain (per slice: Read -> ThermalNoiseRemoval; SliceAssembly only for 2+ slices):

    Read_i -> ThermalNoiseRemoval_i --> SliceAssembly -> Apply-Orbit-File
           -> Calibration (sigma0, linear) -> Terrain-Correction -> Subset (AOI) -> Write

Noise removal must come before slice assembly. SNAP 14 refuses otherwise
("Noise removal should be applied prior to slice assembly"), because each
slice has its own noise annotation.

Choices that matter (operator parameters as printed by SNAP 14's `gpt <op> -h`):
- clipNegativeValues=false by default: Terrain-Correction resamples (averages),
  and clipping first biases dark water upward (MPC-0392).
- UTM projection of the AOI centre, fixed pixel spacing, and alignToStandardGrid
  with origin (0, 0): every pass over an AOI lands on the *same* pixel grid, so
  flood and reference images can be differenced without resampling again.
- Layover/shadow mask and local incidence angle are saved: radar shadow is dark
  like water and must be masked, and incidence angle is a useful model input.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

DEFAULT_GPT = Path.home() / "esa-snap" / "bin" / "gpt"


def gpt_path() -> Path:
    return Path(os.environ.get("SNAP_GPT", DEFAULT_GPT))


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing (lon, lat)."""
    if not (-180 <= lon <= 180 and -80 <= lat <= 84):
        raise ValueError(f"UTM undefined for lon={lon}, lat={lat}")
    zone = min(int((lon + 180) // 6) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


def _node(node_id: str, operator: str, sources: list[str], params: dict[str, object]) -> str:
    src = "".join(f'<sourceProduct{"" if i == 0 else f".{i}"} refid="{s}"/>'
                  for i, s in enumerate(sources))
    par = "".join(
        f"<{k}>{escape(str(v).lower() if isinstance(v, bool) else str(v))}</{k}>"
        for k, v in params.items()
    )
    srcs = f"<sources>{src}</sources>" if sources else ""
    return (f'  <node id="{node_id}">\n    <operator>{operator}</operator>\n'
            f"    {srcs}\n    <parameters>{par}</parameters>\n  </node>\n")


def build_tc_graph(slices: list[Path], output: Path, aoi_wkt: str, epsg: int,
                   spacing_m: float = 20.0, clip: bool = False,
                   polarisations: str = "VV,VH") -> str:
    if not slices:
        raise ValueError("need at least one slice")
    nodes = []
    denoised = []
    for i, z in enumerate(slices, 1):
        nodes.append(_node(f"Read{i}", "Read", [], {"file": str(z)}))
        nodes.append(_node(f"ThermalNoiseRemoval{i}", "ThermalNoiseRemoval", [f"Read{i}"], {
            "selectedPolarisations": polarisations,
            "removeThermalNoise": True, "clipNegativeValues": clip}))
        denoised.append(f"ThermalNoiseRemoval{i}")
    prev = denoised[0]
    if len(denoised) > 1:
        nodes.append(_node("SliceAssembly", "SliceAssembly", denoised,
                           {"selectedPolarisations": polarisations}))
        prev = "SliceAssembly"
    chain = [
        ("Apply-Orbit-File", {"orbitType": "Sentinel Precise (Auto Download)",
                              "continueOnFail": False}),
        ("Calibration", {"outputSigmaBand": True, "outputImageScaleInDb": False}),
        ("Terrain-Correction", {
            "demName": "Copernicus 30m Global DEM",
            "demResamplingMethod": "BILINEAR_INTERPOLATION",
            "imgResamplingMethod": "BILINEAR_INTERPOLATION",
            "mapProjection": f"EPSG:{epsg}",
            "pixelSpacingInMeter": spacing_m,
            "alignToStandardGrid": True,
            "standardGridOriginX": 0.0,
            "standardGridOriginY": 0.0,
            "nodataValueAtSea": False,  # inland AOIs; don't let a coastline mask eat data
            "saveSelectedSourceBand": True,
            "saveLayoverShadowMask": True,
            "saveLocalIncidenceAngle": True,
        }),
        ("Subset", {"geoRegion": aoi_wkt, "copyMetadata": True}),
    ]
    for op, params in chain:
        nodes.append(_node(op, op, [prev], params))
        prev = op
    nodes.append(_node("Write", "Write", [prev],
                       {"file": str(output), "formatName": "GeoTIFF-BigTIFF"}))
    return '<graph id="terrain_correct">\n  <version>1.0</version>\n' + "".join(nodes) + "</graph>\n"


def run_gpt(graph_file: Path, extra: list[str] | None = None) -> int:
    """Run gpt on a graph file, streaming its output. Returns the exit code."""
    cmd = [str(gpt_path()), str(graph_file), *(extra or [])]
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode
