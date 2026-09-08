"""
Country-scale AOI resolution.

WHY THIS MODULE EXISTS
----------------------
`geocode._clamp_bbox` caps any AOI at `MAX_AOI_DEG` (0.5 deg) by shrinking the
box around its CENTROID. For a city that is harmless. For a country it is
actively wrong:

    Nepal true bbox : 80.06 E .. 88.20 E , 26.35 N .. 30.45 N  (798 x 453 km)
    centroid        : 84.13 E , 28.40 N
    clamped bbox    : 83.88..84.38 E , 28.15..28.65 N  (49 x 55 km)

That centroid box sits in the Annapurna/Manaslu massif at 2200-4100 m. The
pipeline then ran happily and reported "water expansion in Nepal" from snow and
glacial melt in the High Himalaya, while Nepal's actual flood risk is in the
Terai lowlands along the southern border (60-200 m).

So for country-scale requests we must not just shrink the box — we must CHOOSE
a representative study area. This module picks the most flood-prone window
that lies inside the country's real polygon, using a public DEM.

No mock data is used: the boundary comes from OSM/Nominatim and the elevation
from the Open-Meteo DEM API (Copernicus DEM GLO-90).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import settings

# A bbox wider than this in either axis is treated as country/region scale.
COUNTRY_SPAN_DEG = float(__import__("os").getenv("EP_COUNTRY_SPAN_DEG", "2.0"))

# Size of the representative study window carved out of a large country.
STUDY_WINDOW_DEG = float(__import__("os").getenv("EP_STUDY_WINDOW_DEG", "0.5"))

# Candidate grid resolution when scanning a country for flood-prone terrain.
SCAN_NX = int(__import__("os").getenv("EP_SCAN_NX", "9"))
SCAN_NY = int(__import__("os").getenv("EP_SCAN_NY", "7"))

ELEVATION_ENDPOINT = "https://api.open-meteo.com/v1/elevation"

# Minimum elevation for a candidate to count as LAND rather than open water.
# Bangladesh's southern sample cells sit in the Bay of Bengal at 0 m; picking
# those gives a scene with no land and therefore no detectable change.
MIN_LAND_ELEV_M = float(__import__("os").getenv("EP_MIN_LAND_ELEV_M", "5"))


@dataclass
class StudyArea:
    """A concrete, processable AOI derived from a larger region."""
    bbox: tuple[float, float, float, float]
    lon: float
    lat: float
    elevation_m: float | None = None
    reduced: bool = False
    parent_bbox: tuple[float, float, float, float] | None = None
    method: str = "direct"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 6) for v in self.bbox],
            "centre": [round(self.lon, 6), round(self.lat, 6)],
            "elevation_m": self.elevation_m,
            "reduced": self.reduced,
            "parent_bbox": ([round(v, 6) for v in self.parent_bbox]
                            if self.parent_bbox else None),
            "method": self.method,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# geometry helpers (no shapely dependency)
# --------------------------------------------------------------------------
def _rings(geom: dict | None) -> list[list[list[float]]]:
    """Flatten Polygon / MultiPolygon / GeometryCollection to exterior rings."""
    if not geom:
        return []
    t = geom.get("type")
    try:
        if t == "Polygon":
            return [geom["coordinates"][0]]
        if t == "MultiPolygon":
            return [p[0] for p in geom["coordinates"] if p]
        if t == "GeometryCollection":
            out: list[list[list[float]]] = []
            for g in geom.get("geometries", []):
                out.extend(_rings(g))
            return out
    except Exception:
        return []
    return []


def point_in_rings(lon: float, lat: float, rings: list[list[list[float]]]) -> bool:
    """Ray-casting point-in-polygon over a set of exterior rings."""
    inside = False
    for ring in rings:
        n = len(ring)
        if n < 4:
            continue
        j = n - 1
        for i in range(n):
            try:
                xi, yi = ring[i][0], ring[i][1]
                xj, yj = ring[j][0], ring[j][1]
            except (TypeError, IndexError):
                j = i
                continue
            if (yi > lat) != (yj > lat):
                denom = (yj - yi) or 1e-12
                if lon < (xj - xi) * (lat - yi) / denom + xi:
                    inside = not inside
            j = i
    return inside


_BOUNDARY_CACHE: dict[str, dict] = {}
_ELEV_CACHE: dict[tuple, list] = {}


async def fetch_boundary(name: str, *, retries: int = 2) -> dict | None:
    """
    Fetch a real administrative polygon from Nominatim. None on any failure.

    Cached and retried: a single transient Nominatim timeout would otherwise
    silently downgrade a country query back to the wrong centroid box, making
    results non-deterministic between runs.
    """
    key = name.strip().lower()
    if key in _BOUNDARY_CACHE:
        return _BOUNDARY_CACHE[key]

    data = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=25.0, headers={"User-Agent": settings.USER_AGENT}
            ) as client:
                r = await client.get(
                    settings.GEOCODER_ENDPOINT,
                    params={"q": name, "format": "json", "limit": 1,
                            "polygon_geojson": 1, "addressdetails": 0},
                )
                r.raise_for_status()
                data = r.json()
            break
        except Exception:
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))
            else:
                return None
    if not data:
        return None
    item = data[0]
    bb = item.get("boundingbox")
    out: dict[str, Any] = {"geojson": item.get("geojson"),
                           "label": item.get("display_name", name),
                           "class": item.get("class"), "type": item.get("type")}
    if bb and len(bb) == 4:
        s, n, w, e = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        out["bbox"] = (w, s, e, n)
    _BOUNDARY_CACHE[key] = out
    return out


async def fetch_elevations(points: list[tuple[float, float]]) -> list[float | None]:
    """Batch DEM lookup. Returns None per point on failure (never raises)."""
    if not points:
        return []
    ckey = tuple(round(v, 4) for p in points for v in p)
    if ckey in _ELEV_CACHE:
        return list(_ELEV_CACHE[ckey])
    lats = ",".join(f"{lat:.4f}" for _, lat in points)
    lons = ",".join(f"{lon:.4f}" for lon, _ in points)
    try:
        async with httpx.AsyncClient(
            timeout=25.0, headers={"User-Agent": settings.USER_AGENT}
        ) as client:
            r = await client.get(ELEVATION_ENDPOINT,
                                 params={"latitude": lats, "longitude": lons})
            r.raise_for_status()
            vals = r.json().get("elevation") or []
    except Exception:
        return [None] * len(points)
    out: list[float | None] = []
    for i in range(len(points)):
        try:
            out.append(float(vals[i]))
        except (IndexError, TypeError, ValueError):
            out.append(None)
    if any(v is not None for v in out):
        _ELEV_CACHE[ckey] = list(out)
    return out


def _span(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    w, s, e, n = bbox
    return abs(e - w), abs(n - s)


def is_country_scale(bbox: tuple[float, float, float, float]) -> bool:
    dx, dy = _span(bbox)
    return dx >= COUNTRY_SPAN_DEG or dy >= COUNTRY_SPAN_DEG


async def resolve_study_area(
    name: str,
    fallback_bbox: tuple[float, float, float, float],
    fallback_lon: float,
    fallback_lat: float,
    *,
    prefer: str = "water",
) -> StudyArea:
    """
    Turn a possibly country-sized place into one concrete, processable AOI.

    For sub-country places this is a no-op passthrough. For country-scale
    places it scans a coarse lat/lon grid, keeps only candidate points that
    fall INSIDE the real boundary polygon (so a Nepal request can never drift
    into India), and picks the lowest-elevation candidate — the floodplain —
    when the query is about water.

    Defensive throughout: if the boundary or DEM lookup fails for any reason,
    it degrades to the caller's already-clamped bbox rather than raising.
    """
    notes: list[str] = []
    boundary = await fetch_boundary(name)

    true_bbox = (boundary or {}).get("bbox") or fallback_bbox
    if not is_country_scale(true_bbox):
        return StudyArea(bbox=fallback_bbox, lon=fallback_lon, lat=fallback_lat,
                         reduced=False, method="direct")

    rings = _rings((boundary or {}).get("geojson"))
    w, s, e, n = true_bbox
    half = STUDY_WINDOW_DEG / 2.0

    # Candidate centres on a coarse grid, inset so the window stays in-country.
    cands: list[tuple[float, float]] = []
    for iy in range(SCAN_NY):
        for ix in range(SCAN_NX):
            lon = w + (e - w) * (ix + 0.5) / SCAN_NX
            lat = s + (n - s) * (iy + 0.5) / SCAN_NY
            if rings and not point_in_rings(lon, lat, rings):
                continue
            cands.append((lon, lat))

    if not cands:
        notes.append("Could not sample inside the national boundary; "
                     "using the centre of the region.")
        return StudyArea(bbox=fallback_bbox, lon=fallback_lon, lat=fallback_lat,
                         elevation_m=None, reduced=True, parent_bbox=true_bbox,
                         method="centroid-fallback", notes=notes)

    elevs = await fetch_elevations(cands)
    scored = [(lon, lat, ev) for (lon, lat), ev in zip(cands, elevs) if ev is not None]

    if not scored:
        lon, lat = cands[len(cands) // 2]
        notes.append("Elevation service unavailable; used a central in-country window.")
        method = "in-country-centre"
        elev = None
    else:
        if prefer == "water":
            # Lowest LAND = floodplain. Nepal -> Terai; Bangladesh -> delta.
            #
            # Elevations at or below sea level are open water (Bangladesh's
            # southern grid cells fall in the Bay of Bengal), which yields a
            # scene with no usable land and no detectable change. Require a
            # little relief so the window is genuine floodplain, not ocean.
            land = [t for t in scored if t[2] >= MIN_LAND_ELEV_M]
            pool = land or scored
            lon, lat, elev = min(pool, key=lambda t: t[2])
            method = ("lowest-land-in-country" if land
                      else "lowest-elevation-in-country")
        else:
            scored.sort(key=lambda t: t[2])
            lon, lat, elev = scored[len(scored) // 2]
            method = "median-elevation-in-country"

    # Keep the window inside the country's own bbox.
    lon = min(max(lon, w + half), e - half) if (e - w) > STUDY_WINDOW_DEG else (w + e) / 2
    lat = min(max(lat, s + half), n - half) if (n - s) > STUDY_WINDOW_DEG else (s + n) / 2

    # The lowest point often sits ON the international border (Nepal's Terai
    # runs along the Indian frontier), so a window centred there would straddle
    # two countries and silently analyse the neighbour. Nudge the window until
    # as much of it as possible lies inside the real boundary.
    if rings:
        def coverage(clon: float, clat: float) -> float:
            pts = [(clon + dx * half, clat + dy * half)
                   for dx in (-0.9, -0.45, 0.0, 0.45, 0.9)
                   for dy in (-0.9, -0.45, 0.0, 0.45, 0.9)]
            hits = sum(1 for x, y in pts if point_in_rings(x, y, rings))
            return hits / len(pts)

        best = (coverage(lon, lat), lon, lat)
        if best[0] < 1.0:
            # Search a small neighbourhood, preferring minimal displacement.
            for step_mult in (0.25, 0.5, 0.75, 1.0):
                for dx in (0.0, -1, 1):
                    for dy in (0.0, 1, -1):
                        if dx == 0 and dy == 0:
                            continue
                        clon = lon + dx * half * 2 * step_mult
                        clat = lat + dy * half * 2 * step_mult
                        if not (w + half <= clon <= e - half):
                            continue
                        if not (s + half <= clat <= n - half):
                            continue
                        cov = coverage(clon, clat)
                        if cov > best[0]:
                            best = (cov, clon, clat)
                if best[0] >= 1.0:
                    break
            if best[1] != lon or best[2] != lat:
                lon, lat = best[1], best[2]
                revised = await fetch_elevations([(lon, lat)])
                if revised and revised[0] is not None:
                    elev = revised[0]
        notes.append(f"Study window is {best[0] * 100:.0f}% inside the national boundary.")

    bbox = (round(lon - half, 6), round(lat - half, 6),
            round(lon + half, 6), round(lat + half, 6))

    notes.append(
        f"{name} is a country-scale area ({_span(true_bbox)[0]:.1f}\u00b0 \u00d7 "
        f"{_span(true_bbox)[1]:.1f}\u00b0). A representative "
        f"{STUDY_WINDOW_DEG:.1f}\u00b0 study window was selected"
        + (f" at the lowest-lying in-country location ({elev:.0f} m), where flood risk "
           f"is concentrated." if elev is not None else ".")
    )
    notes.append("For a specific area, name a province, district or municipality "
                 "(e.g. \u201cChitwan, Nepal\u201d or \u201cKoshi Province, Nepal\u201d).")

    return StudyArea(bbox=bbox, lon=lon, lat=lat, elevation_m=elev,
                     reduced=True, parent_bbox=true_bbox, method=method, notes=notes)
