"""
EarthPulse — raster ingest, common-grid warping and spectral indices.

Key idea: every epoch is read directly onto ONE shared EPSG:4326 grid using
rasterio's WarpedVRT. Because all epochs land on the identical grid definition,
pixel (i, j) in 2022 and pixel (i, j) in 2026 are the same patch of ground by
construction. That removes the classic "misaligned images produce fake change"
failure mode without a separate resampling step.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import settings, SCL_INVALID, SCL_DEGRADED, SCL_LABELS  # noqa: F401  (sets GDAL env)

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

WGS84 = CRS.from_epsg(4326)

# How many source pixels per output pixel to keep before dropping to a coarser
# COG overview. Higher = more faithful (and more memory). At 4x, AOIs up to
# roughly 0.2 deg still read at full resolution, so all previously-tuned
# small-area results are unchanged; only very large AOIs decimate.
OVERSAMPLE = float(os.getenv("EP_OVERSAMPLE", "4"))

# Hard ceiling on source pixels materialised per band read. 6 Mpx of uint16 is
# ~12 MB per band; with 6 bands x 2 tiles read concurrently that stays inside
# the container. This is the OOM guard for country-scale AOIs.
#
# 6 Mpx is chosen so every AOI the app was previously tuned against (Kokapet,
# Terai, Sambhar 5.3, Wayanad 5.6, Amaravati) still reads at FULL resolution
# and returns byte-identical numbers, while country-scale AOIs (Nepal and
# Bangladesh, ~29 Mpx) are forced onto an overview.
MAX_SRC_MPX = float(os.getenv("EP_MAX_SRC_MPX", "6"))

# Reflectance scaling for Sentinel-2 L2A COGs (DN -> reflectance).
S2_SCALE = 1.0 / 10000.0

# --- Radiometric offset handling -------------------------------------------
# ESA processing baseline 04.00 (2022-01-25 onward) introduced a +1000 DN
# pedestal, nominally removed via a -0.1 reflectance offset.
#
# HOWEVER: the AWS `sentinel-cogs` archive is MIXED. Most items are already
# harmonised back to the pre-04.00 convention even though their STAC metadata
# still advertises `offset: -0.1`; a minority (mostly 2022, baseline 04.00)
# genuinely carry the pedestal. Trusting either the date or the metadata alone
# produces a hard error:
#
#   * trusting metadata  -> deep water SWIR becomes -0.09 reflectance
#                           (physically impossible; inverts MNDWI/NDBI)
#   * ignoring it        -> 2022-era scenes sit ~0.1 above every neighbour,
#                           creating a fake step change in every index
#
# So we DETECT the pedestal from the pixels themselves. Dark targets (deep
# water, terrain shadow, dense conifer) always exist somewhere in a 10-50 km
# AOI and have near-zero SWIR reflectance. If the low percentile of SWIR DN
# sits near 1000 instead of near 0, the pedestal is present.
S2_OFFSET_DN = 1000.0
PEDESTAL_TEST_DN = 700.0     # p01(SWIR) above this => pedestal present
PEDESTAL_TEST_PCT = 1.0


@dataclass
class Grid:
    """The common analysis grid shared by every epoch."""

    bounds: tuple[float, float, float, float]   # west, south, east, north (EPSG:4326)
    width: int
    height: int

    @property
    def transform(self):
        return transform_from_bounds(*self.bounds, self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": list(self.bounds),
            "width": self.width,
            "height": self.height,
            "crs": "EPSG:4326",
        }

    def pixel_area_ha(self) -> float:
        """Approximate ground area of one pixel, in hectares."""
        w, s, e, n = self.bounds
        mid_lat = np.deg2rad((s + n) / 2.0)
        m_per_deg_lat = 111_132.92 - 559.82 * np.cos(2 * mid_lat) + 1.175 * np.cos(4 * mid_lat)
        m_per_deg_lon = 111_412.84 * np.cos(mid_lat) - 93.5 * np.cos(3 * mid_lat)
        px_h_m = abs(n - s) / self.height * m_per_deg_lat
        px_w_m = abs(e - w) / self.width * abs(m_per_deg_lon)
        return float(px_w_m * px_h_m / 10_000.0)

    def xy(self, row: float, col: float) -> tuple[float, float]:
        """Grid (row, col) -> (lon, lat)."""
        w, s, e, n = self.bounds
        lon = w + (col + 0.5) / self.width * (e - w)
        lat = n - (row + 0.5) / self.height * (n - s)
        return float(lon), float(lat)


def build_grid(bbox: tuple[float, float, float, float], target_px: int | None = None) -> Grid:
    """Build a square-ish analysis grid that preserves the AOI aspect ratio."""
    target_px = target_px or settings.GRID_SIZE
    w, s, e, n = bbox
    dx, dy = abs(e - w), abs(n - s)
    if dx <= 0 or dy <= 0:
        raise ValueError("Degenerate bounding box")
    if dx >= dy:
        width = target_px
        height = max(64, int(round(target_px * dy / dx)))
    else:
        height = target_px
        width = max(64, int(round(target_px * dx / dy)))
    return Grid(bounds=(w, s, e, n), width=width, height=height)


@dataclass
class EpochStack:
    """All reflectance bands + SCL for one epoch, on the common grid."""

    scene_id: str
    date: str
    year: int
    cloud_cover: float
    bands: dict[str, np.ndarray]        # 'red','green','blue','nir','swir16' -> float32 reflectance
    scl: np.ndarray                     # uint8
    valid: np.ndarray                   # bool — usable pixels
    degraded_fraction: float = 0.0
    valid_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)


def _pick_overview_level(src, grid: Grid) -> int:
    """
    Choose the COG overview decimation factor that most closely matches the
    output grid resolution, without going below it.

    WHY THIS EXISTS
    ---------------
    `WarpedVRT` warps from the *full-resolution* source unless told otherwise.
    For a small AOI that is harmless, but the cost scales with the SOURCE
    window, not the output size:

        India  (0.065 deg AOI) ->    666 x  406 src px =  0.3 Mpx -> +78 MB
        Nepal  (0.50  deg AOI) ->   5046 x 5671 src px = 28.6 Mpx -> +591 MB

    Both produce a <=480x480 output, but Nepal allocated 95x more source
    pixels per band. With 6 bands x 2 tiles x 5 epochs that exhausted the
    container and the server was OOM-killed (exit 137) on the next request.

    Reading from an overview level instead makes cost depend on the OUTPUT
    resolution, so memory is bounded no matter how large the AOI is.
    """
    overviews = src.overviews(1) or [1]
    if 1 not in overviews:
        overviews = [1] + list(overviews)

    # Source pixels that the AOI spans, in source CRS units.
    try:
        left, bottom, right, top = transform_bounds(WGS84, src.crs, *grid.bounds, densify_pts=21)
    except Exception:
        return 1
    res_x, res_y = src.res
    if not res_x or not res_y:
        return 1
    src_w = abs(right - left) / abs(res_x)
    src_h = abs(top - bottom) / abs(res_y)

    # Decimation needed so the source window is about the size of the output.
    need = min(src_w / max(grid.width, 1), src_h / max(grid.height, 1))

    # Two independent budgets, whichever demands more decimation wins.
    #
    # (a) Quality budget: keep OVERSAMPLE source pixels per output pixel so the
    #     bilinear warp still averages properly. This leaves small and medium
    #     AOIs at full resolution (level 1), so existing tuned results are
    #     unchanged.
    quality_budget = need / OVERSAMPLE

    # (b) Hard memory budget: never materialise more than MAX_SRC_MPX source
    #     pixels for a single band read, regardless of AOI size. This is the
    #     actual OOM guard — a country-sized AOI would otherwise pull ~28 Mpx
    #     per band (x6 bands x2 tiles x5 epochs) and exhaust the container.
    total_mpx = (src_w * src_h) / 1e6
    if total_mpx > MAX_SRC_MPX:
        # Decimation is per-axis, so area scales with the square.
        mem_budget = (total_mpx / MAX_SRC_MPX) ** 0.5
    else:
        mem_budget = 1.0

    levels = sorted(overviews)

    # Quality: largest overview that still leaves OVERSAMPLE redundancy.
    q_usable = [f for f in levels if f <= quality_budget]
    level_q = q_usable[-1] if q_usable else 1

    # Memory: smallest overview that brings the read under the cap. Rounded UP,
    # because this is a hard limit rather than a preference.
    m_usable = [f for f in levels if f >= mem_budget]
    level_m = m_usable[0] if m_usable else levels[-1]
    if mem_budget <= 1:
        level_m = 1

    # The memory cap always wins if it demands more decimation.
    return max(level_q, level_m, 1)


def _read_band_on_grid(href: str, grid: Grid, resampling: Resampling) -> np.ndarray:
    """
    Read one COG band, warped onto the shared grid.

    Uses an overview level matched to the output resolution so peak memory
    depends on the analysis grid rather than on the size of the AOI.
    """
    with rasterio.open(href) as src:
        level = _pick_overview_level(src, grid)
        # `open(overview_level=N)` is 0-based over the overview list, so map
        # the decimation factor back to its index.
        if level > 1:
            factors = sorted(src.overviews(1) or [])
            try:
                idx = factors.index(level)
            except ValueError:
                idx = None
            if idx is not None:
                src.close()
                with rasterio.open(href, overview_level=idx) as ov:
                    with WarpedVRT(
                        ov,
                        crs=WGS84,
                        transform=grid.transform,
                        width=grid.width,
                        height=grid.height,
                        resampling=resampling,
                        src_nodata=ov.nodata if ov.nodata is not None else 0,
                        nodata=0,
                        warp_mem_limit=64,
                    ) as vrt:
                        return vrt.read(1)

        with WarpedVRT(
            src,
            crs=WGS84,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
            src_nodata=src.nodata if src.nodata is not None else 0,
            nodata=0,
            warp_mem_limit=64,
        ) as vrt:
            return vrt.read(1)


def load_epoch(scene, grid: Grid, *, workers: int | None = None) -> EpochStack:
    """
    Load one Sentinel-2 scene onto the common grid.

    Bands are fetched in parallel; only the windowed overview levels that
    intersect the AOI are transferred, so this is typically < 1 s per epoch.
    """
    workers = workers or settings.READ_WORKERS
    band_keys = ["red", "green", "blue", "nir", "swir16"]

    # Sources: the primary scene first, then any same-day neighbouring tiles.
    sources = [scene] + list(getattr(scene, "companions", []) or [])

    def job(args):
        src_idx, key = args
        resampling = Resampling.nearest if key == "scl" else Resampling.bilinear
        try:
            arr = _read_band_on_grid(sources[src_idx].assets[key], grid, resampling)
        except Exception as exc:  # noqa: BLE001
            # A single unreadable asset (404, truncated COG, transient S3 error)
            # must not abort the whole epoch. Return zeros; the SCL/no-data
            # logic below will mark those pixels invalid, and the quality gate
            # decides whether the epoch is still usable.
            arr = np.zeros((grid.height, grid.width),
                           dtype=np.uint8 if key == "scl" else np.uint16)
            _READ_FAILURES.append(f"{key}@{getattr(sources[src_idx], 'id', '?')}: {exc}")
        return src_idx, key, arr

    tasks = [(i, k) for i in range(len(sources)) for k in band_keys + ["scl"]]
    raw: dict[tuple[int, str], np.ndarray] = {}
    _READ_FAILURES: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for src_idx, key, arr in ex.map(job, tasks):
            raw[(src_idx, key)] = arr

    # Guarantee every required band exists, even if a read failed entirely.
    for k in band_keys + ["scl"]:
        if (0, k) not in raw:
            raw[(0, k)] = np.zeros((grid.height, grid.width),
                                   dtype=np.uint8 if k == "scl" else np.uint16)

    # Mosaic: start from the primary tile, fill holes from companions.
    results: dict[str, np.ndarray] = {k: raw[(0, k)] for k in band_keys + ["scl"]}
    mosaic_note = None
    if len(sources) > 1:
        gap = results["swir16"] == 0
        filled_total = 0
        for i in range(1, len(sources)):
            if not gap.any():
                break
            donor_valid = raw[(i, "swir16")] > 0
            fill = gap & donor_valid
            n_fill = int(fill.sum())
            if n_fill == 0:
                continue
            filled_total += n_fill
            for k in band_keys + ["scl"]:
                results[k] = np.where(fill, raw[(i, k)], results[k])
            gap &= ~fill
        if filled_total:
            pct = filled_total / results["swir16"].size * 100
            mosaic_note = (
                f"{pct:.0f}% of this epoch was filled from {len(sources) - 1} neighbouring "
                f"Sentinel-2 tile(s) acquired the same day, giving full coverage of the area."
            )

    scl = results.pop("scl").astype(np.uint8)

    # --- Data-driven radiometric offset detection (see module header) ------
    swir_raw = results["swir16"].astype(np.float32)
    dark_sample = swir_raw[swir_raw > 0]
    if dark_sample.size > 64:
        p_low = float(np.percentile(dark_sample, PEDESTAL_TEST_PCT))
    else:
        p_low = 0.0
    pedestal_present = p_low >= PEDESTAL_TEST_DN
    offset = S2_OFFSET_DN if pedestal_present else 0.0

    notes: list[str] = []
    if mosaic_note:
        notes.append(mosaic_note)
    if _READ_FAILURES:
        notes.append(
            f"{len(_READ_FAILURES)} band read(s) failed for this date and were treated as "
            f"no-data: {_READ_FAILURES[0][:120]}"
        )
    if pedestal_present:
        notes.append(
            f"Baseline-04.00 radiometric pedestal detected (SWIR p1 = {p_low:.0f} DN); "
            f"a -1000 DN offset was applied to match the other epochs."
        )

    bands: dict[str, np.ndarray] = {}
    raw_zero = None
    for key in band_keys:
        raw = results[key].astype(np.float32)
        if raw_zero is None:
            raw_zero = raw == 0
        else:
            raw_zero &= raw == 0
        refl = (raw - offset) * S2_SCALE
        # Physically impossible reflectance -> clamp, keeps indices well-behaved.
        bands[key] = np.clip(refl, 0.0, 1.6).astype(np.float32)

    invalid_scl = np.isin(scl, list(SCL_INVALID))
    degraded_scl = np.isin(scl, list(SCL_DEGRADED))
    valid = ~invalid_scl & ~(raw_zero if raw_zero is not None else False)

    total = float(valid.size)
    return EpochStack(
        scene_id=scene.id,
        date=scene.date,
        year=scene.year,
        cloud_cover=scene.cloud_cover,
        bands=bands,
        scl=scl,
        valid=valid,
        valid_fraction=float(valid.sum() / total),
        degraded_fraction=float(degraded_scl.sum() / total),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------
def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Normalised difference (a - b) / (a + b), hardened against bad inputs.

    Guarantees a finite float32 result in [-1, 1] with no NaN or infinity, so
    downstream statistics (median, MAD, percentiles) can never be poisoned by
    a single bad pixel. Handles:
      * division by zero (a + b == 0, common over deep shadow / no-data)
      * NaN or infinite reflectance from a corrupt or partially-read tile
      * out-of-domain values from an unexpected scaling/offset
    """
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    denom = a + b
    out = np.zeros_like(denom, dtype=np.float32)
    ok = np.abs(denom) > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = ((a - b)[ok] / denom[ok]).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -1.0, 1.0)


def ndvi(st: EpochStack) -> np.ndarray:
    """Vegetation vigour. (NIR - Red) / (NIR + Red)."""
    return _safe_ratio(st.bands["nir"], st.bands["red"])


def ndbi(st: EpochStack) -> np.ndarray:
    """Built-up / impervious surface. (SWIR - NIR) / (SWIR + NIR)."""
    return _safe_ratio(st.bands["swir16"], st.bands["nir"])


def mndwi(st: EpochStack) -> np.ndarray:
    """Open water. (Green - SWIR) / (Green + SWIR). Better than NDWI in cities."""
    return _safe_ratio(st.bands["green"], st.bands["swir16"])


def nbr(st: EpochStack) -> np.ndarray:
    """Burn severity. (NIR - SWIR) / (NIR + SWIR). Drops sharply after fire."""
    return _safe_ratio(st.bands["nir"], st.bands["swir16"])


INDEX_FN = {"NDVI": ndvi, "NDBI": ndbi, "MNDWI": mndwi, "NBR": nbr}

INDEX_META = {
    "NDVI": {
        "name": "Normalised Difference Vegetation Index",
        "formula": "(NIR - Red) / (NIR + Red)",
        "bands": "B08, B04",
        "meaning": "Higher values mean denser, healthier vegetation.",
    },
    "NDBI": {
        "name": "Normalised Difference Built-up Index",
        "formula": "(SWIR1 - NIR) / (SWIR1 + NIR)",
        "bands": "B11, B08",
        "meaning": "Higher values indicate impervious surfaces such as concrete, rooftops and roads.",
    },
    "MNDWI": {
        "name": "Modified Normalised Difference Water Index",
        "formula": "(Green - SWIR1) / (Green + SWIR1)",
        "bands": "B03, B11",
        "meaning": "Positive values indicate open water; robust against built-up noise.",
    },
    "NBR": {
        "name": "Normalised Burn Ratio",
        "formula": "(NIR - SWIR2) / (NIR + SWIR2)",
        "bands": "B08, B11",
        "meaning": "Falls sharply where vegetation has been consumed by fire.",
    },
}


def compute_index(st: EpochStack, index: str) -> np.ndarray:
    fn = INDEX_FN.get(index.upper())
    if fn is None:
        raise ValueError(f"Unknown index: {index}")
    return fn(st)


def rgb_preview(st: EpochStack, gamma: float = 1.05) -> np.ndarray:
    """
    Natural-colour composite as uint8 RGB, with a robust per-band stretch.

    Percentile clipping is computed on valid pixels only so a cloud edge or a
    black no-data margin cannot wash out the whole scene.
    """
    chans = []
    valid = st.valid
    for key in ("red", "green", "blue"):
        b = st.bands[key].astype(np.float32)
        sample = b[valid] if valid.any() else b.ravel()
        if sample.size == 0:
            chans.append(np.zeros(b.shape, dtype=np.uint8))
            continue
        lo, hi = np.percentile(sample, [2.0, 96.0])
        if hi - lo < 1e-4:
            hi = lo + 1e-4
        x = np.clip((b - lo) / (hi - lo), 0.0, 1.0)
        x = np.power(x, 1.0 / max(gamma, 1e-3))
        chans.append((x * 255.0).astype(np.uint8))
    rgb = np.dstack(chans)
    # Paint invalid pixels a neutral dark grey rather than pure black.
    rgb[~valid] = 28
    return rgb
