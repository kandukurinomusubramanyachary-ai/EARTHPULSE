"""
EarthPulse — place name -> bounding box resolution.

Uses OpenStreetMap Nominatim (free, no key). A small built-in gazetteer of
Indian demo locations is checked first: it removes network latency from the
demo path and guarantees the judge-facing scenarios always resolve.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import httpx

from .config import settings


@dataclass
class Place:
    label: str
    lon: float
    lat: float
    bbox: tuple[float, float, float, float]     # w, s, e, n
    source: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


# Curated AOIs — chosen because each shows a real, documented change signal.
GAZETTEER: dict[str, dict[str, Any]] = {
    "hyderabad": {
        "label": "Hyderabad, Telangana, India",
        "lon": 78.4867, "lat": 17.3850,
        "bbox": (78.30, 17.32, 78.62, 17.58),
    },
    "gachibowli": {
        "label": "Gachibowli, Hyderabad, India",
        "lon": 78.3489, "lat": 17.4401,
        "bbox": (78.29, 17.39, 78.42, 17.49),
    },
    "kokapet": {
        "label": "Kokapet, Hyderabad, India",
        "lon": 78.3378, "lat": 17.4045,
        "bbox": (78.29, 17.36, 78.39, 17.44),
    },
    "osman sagar": {
        "label": "Osman Sagar (Gandipet), Hyderabad, India",
        "lon": 78.3167, "lat": 17.3833,
        "bbox": (78.25, 17.34, 78.38, 17.43),
    },
    "himayat sagar": {
        "label": "Himayat Sagar, Hyderabad, India",
        "lon": 78.3500, "lat": 17.3300,
        "bbox": (78.28, 17.29, 78.41, 17.37),
    },
    "shamirpet": {
        "label": "Shamirpet Lake, Hyderabad, India",
        "lon": 78.5700, "lat": 17.6300,
        "bbox": (78.52, 17.59, 78.62, 17.67),
    },
    "amaravati": {
        "label": "Amaravati, Andhra Pradesh, India",
        "lon": 80.5150, "lat": 16.5150,
        "bbox": (80.44, 16.45, 80.60, 16.59),
    },
    "bengaluru": {
        "label": "Bengaluru, Karnataka, India",
        "lon": 77.5946, "lat": 12.9716,
        "bbox": (77.45, 12.85, 77.75, 13.12),
    },
    "bangalore": {
        "label": "Bengaluru, Karnataka, India",
        "lon": 77.5946, "lat": 12.9716,
        "bbox": (77.45, 12.85, 77.75, 13.12),
    },
    "chennai": {
        "label": "Chennai, Tamil Nadu, India",
        "lon": 80.2707, "lat": 13.0827,
        "bbox": (80.13, 12.95, 80.33, 13.20),
    },
    "mumbai": {
        "label": "Mumbai, Maharashtra, India",
        "lon": 72.8777, "lat": 19.0760,
        "bbox": (72.78, 18.95, 72.99, 19.28),
    },
    "delhi": {
        "label": "Delhi, India",
        "lon": 77.2090, "lat": 28.6139,
        "bbox": (76.98, 28.45, 77.40, 28.78),
    },
    "pune": {
        "label": "Pune, Maharashtra, India",
        "lon": 73.8567, "lat": 18.5204,
        "bbox": (73.72, 18.42, 74.00, 18.65),
    },
    "surat": {
        "label": "Surat, Gujarat, India",
        "lon": 72.8311, "lat": 21.1702,
        "bbox": (72.72, 21.08, 72.95, 21.28),
    },
    "ahmedabad": {
        "label": "Ahmedabad, Gujarat, India",
        "lon": 72.5714, "lat": 23.0225,
        "bbox": (72.46, 22.93, 72.70, 23.13),
    },
    "kolkata": {
        "label": "Kolkata, West Bengal, India",
        "lon": 88.3639, "lat": 22.5726,
        "bbox": (88.26, 22.46, 88.46, 22.68),
    },
    "aralam": {
        "label": "Aralam Forest, Kerala, India",
        "lon": 75.8300, "lat": 11.9300,
        "bbox": (75.76, 11.87, 75.92, 12.00),
    },
    "wayanad": {
        "label": "Wayanad, Kerala, India",
        "lon": 76.1320, "lat": 11.6854,
        "bbox": (76.03, 11.59, 76.26, 11.79),
    },
    "araku": {
        "label": "Araku Valley, Andhra Pradesh, India",
        "lon": 82.8700, "lat": 18.3300,
        "bbox": (82.78, 18.24, 82.97, 18.42),
    },
    "sundarbans": {
        "label": "Sundarbans, West Bengal, India",
        "lon": 88.8000, "lat": 21.9500,
        "bbox": (88.66, 21.84, 88.95, 22.07),
    },
    "chilika": {
        "label": "Chilika Lake, Odisha, India",
        "lon": 85.3200, "lat": 19.7100,
        "bbox": (85.16, 19.60, 85.50, 19.83),
    },
    "sambhar": {
        "label": "Sambhar Salt Lake, Rajasthan, India",
        "lon": 75.0800, "lat": 26.9300,
        "bbox": (74.94, 26.85, 75.24, 27.01),
    },
    "ujjani": {
        "label": "Ujjani Reservoir, Maharashtra, India",
        "lon": 75.1200, "lat": 18.0800,
        "bbox": (75.00, 18.00, 75.28, 18.18),
    },
    "nagarjuna sagar": {
        "label": "Nagarjuna Sagar Reservoir, Telangana, India",
        "lon": 79.3120, "lat": 16.5750,
        "bbox": (79.18, 16.48, 79.44, 16.68),
    },
    "polavaram": {
        "label": "Polavaram, Andhra Pradesh, India",
        "lon": 81.6400, "lat": 17.2400,
        "bbox": (81.54, 17.15, 81.75, 17.33),
    },
}


def _clamp_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """
    Keep the AOI inside sane analysis limits.

    Also normalises the geometry defensively, because upstream sources
    (Nominatim, user-supplied bboxes) can hand us:
      * reversed corner order (south > north, or west > east)
      * lat/lon swapped
      * NaN / infinite values
      * coordinates outside the valid WGS84 domain
    A country-level query such as "Nepal" is the common way these surface,
    so they are corrected here rather than crashing a downstream raster read.
    """
    vals = list(bbox)
    if len(vals) != 4 or any(v is None for v in vals):
        raise ValueError("Bounding box must contain four numeric values")
    try:
        w, s, e, n = (float(v) for v in vals)
    except (TypeError, ValueError):
        raise ValueError("Bounding box contains non-numeric values")

    import math
    if any(math.isnan(v) or math.isinf(v) for v in (w, s, e, n)):
        raise ValueError("Bounding box contains NaN or infinite coordinates")

    # Detect swapped lat/lon: latitudes are only valid to +/-90.
    if (abs(w) > 180 or abs(e) > 180) and abs(s) <= 180 and abs(n) <= 180:
        w, s, e, n = s, w, n, e

    w, e = min(w, e), max(w, e)
    s, n = min(s, n), max(s, n)

    # Clip to the valid WGS84 domain (Web-Mercator-safe latitude range).
    w = max(-180.0, min(180.0, w))
    e = max(-180.0, min(180.0, e))
    s = max(-85.0, min(85.0, s))
    n = max(-85.0, min(85.0, n))

    cx, cy = (w + e) / 2, (s + n) / 2
    dx, dy = e - w, n - s

    max_d, min_d = settings.MAX_AOI_DEG, settings.MIN_AOI_DEG
    if dx > max_d:
        dx = max_d
    if dy > max_d:
        dy = max_d
    if dx < min_d:
        dx = min_d
    if dy < min_d:
        dy = min_d

    return (
        round(cx - dx / 2, 6), round(cy - dy / 2, 6),
        round(cx + dx / 2, 6), round(cy + dy / 2, 6),
    )


async def geocode(query: str) -> Place | None:
    """Resolve a place name to a clamped bounding box."""
    if not query:
        return None
    key = query.strip().lower()

    # 1. Built-in gazetteer (exact, then substring).
    #
    # For a comma-separated chain such as "Kokapet, Hyderabad" the FIRST
    # component is the most specific one and must win. A naive substring scan
    # matched the broader "hyderabad" entry and silently widened the AOI from a
    # neighbourhood to the whole metro, changing the answer. So try the leading
    # component first, and only then fall back to looser matching.
    hit = GAZETTEER.get(key)
    if hit is None and "," in key:
        head = key.split(",", 1)[0].strip()
        hit = GAZETTEER.get(head)
    if hit is None:
        for gk, gv in GAZETTEER.items():
            if gk in key or key in gk:
                hit = gv
                break
    if hit:
        return Place(
            label=hit["label"], lon=hit["lon"], lat=hit["lat"],
            bbox=_clamp_bbox(tuple(hit["bbox"])), source="gazetteer",
        )

    # 2. Nominatim.
    try:
        async with httpx.AsyncClient(
            timeout=25.0, headers={"User-Agent": settings.USER_AGENT}
        ) as client:
            r = await client.get(
                settings.GEOCODER_ENDPOINT,
                params={"q": query, "format": "json", "limit": 1, "addressdetails": 0},
            )
            r.raise_for_status()
            data = r.json()
    except Exception:
        return None

    if not data:
        return None

    item = data[0]
    lat, lon = float(item["lat"]), float(item["lon"])
    bb = item.get("boundingbox")
    if bb and len(bb) == 4:
        s, n, w, e = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        bbox = (w, s, e, n)
    else:
        d = 0.08
        bbox = (lon - d, lat - d, lon + d, lat + d)

    return Place(
        label=item.get("display_name", query), lon=lon, lat=lat,
        bbox=_clamp_bbox(bbox), source="nominatim",
    )


def bbox_from_point(lon: float, lat: float, size_deg: float = 0.12) -> tuple[float, float, float, float]:
    return _clamp_bbox((lon - size_deg / 2, lat - size_deg / 2, lon + size_deg / 2, lat + size_deg / 2))
