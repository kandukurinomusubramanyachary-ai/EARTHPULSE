"""
EarthPulse — satellite catalog search and epoch selection.

Talks to the Element84 Earth Search STAC API (free, anonymous) to find
Sentinel-2 L2A scenes, then picks ONE best scene per epoch (year).

The selection strategy is the part that matters scientifically:

  * Season locking — every epoch is drawn from the same +/- N-day window around
    a common day-of-year. Comparing an April image to an April image removes
    the single largest source of false change in multi-temporal analysis
    (phenology). This directly implements Feature 6.
  * Cloud ranking — within the seasonal window, scenes are ranked by scene-level
    cloud cover, and ties are broken by closeness to the anchor day-of-year.
  * Tile locking — all epochs are forced onto the same MGRS tile where possible
    so we compare identical viewing geometry.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import settings


@dataclass
class Scene:
    """A single selected Sentinel-2 acquisition."""

    id: str
    datetime: str                    # ISO
    date: str                        # YYYY-MM-DD
    year: int
    doy: int
    cloud_cover: float
    epsg: int | None
    mgrs: str | None
    assets: dict[str, str]           # band key -> href
    platform: str | None = None
    # Same-day acquisitions on neighbouring tiles, used to fill AOI gaps.
    companions: list["Scene"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "date": self.date,
            "year": self.year,
            "cloud_cover": round(self.cloud_cover, 2),
            "mgrs": self.mgrs,
            "epsg": self.epsg,
            "platform": self.platform,
        }


@dataclass
class CatalogResult:
    scenes: list[Scene]
    anchor_doy: int
    total_candidates: int
    notes: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)


# Band keys we need from the STAC item.
REQUIRED_ASSETS = ["red", "green", "blue", "nir", "swir16", "scl"]


def _doy(d: _dt.date) -> int:
    return d.timetuple().tm_yday


def _circular_doy_delta(a: int, b: int) -> int:
    """Shortest distance between two days-of-year on a 365-day circle."""
    diff = abs(a - b) % 365
    return min(diff, 365 - diff)


async def search_scenes(
    bbox: tuple[float, float, float, float],
    start_year: int,
    end_year: int,
    *,
    max_cloud: float | None = None,
) -> list[dict[str, Any]]:
    """Query the STAC API for every candidate scene in the window."""
    max_cloud = max_cloud if max_cloud is not None else settings.MAX_CLOUD_COVER

    # IMPORTANT: query YEAR BY YEAR rather than paginating one long date range.
    #
    # A busy AOI can match >1500 scenes across a 6-year window. Paginating a
    # single ascending-by-date query exhausts the page budget inside the first
    # two or three years, so the most recent epochs are never seen and the
    # analysis silently ends early. Per-year queries bound the result set and
    # guarantee every epoch in the window gets a fair chance.
    #
    # Each year is also sorted by cloud cover, so the first page already holds
    # the best candidates and one page per year is normally enough.
    years = list(range(start_year, end_year + 1))
    features: list[dict[str, Any]] = []

    async def fetch_year(client: httpx.AsyncClient, year: int) -> list[dict[str, Any]]:
        body = {
            "collections": [settings.STAC_COLLECTION],
            "bbox": list(bbox),
            "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
            "limit": settings.STAC_PAGE_LIMIT,
            "query": {"eo:cloud_cover": {"lt": max_cloud}},
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        }
        out: list[dict[str, Any]] = []
        url: str | None = settings.STAC_ENDPOINT
        payload: dict[str, Any] | None = body
        pages = 0
        while url and pages < 3:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except Exception:
                break
            data = resp.json()
            out.extend(data.get("features", []))
            pages += 1
            nxt = next((l for l in data.get("links", []) if l.get("rel") == "next"), None)
            if not nxt or nxt.get("method", "POST").upper() != "POST":
                break
            url, payload = nxt.get("href"), nxt.get("body", payload)
        return out

    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
        import asyncio
        results = await asyncio.gather(
            *(fetch_year(client, y) for y in years), return_exceptions=True
        )
    for r in results:
        if isinstance(r, list):
            features.extend(r)

    return features


def _to_scene(f: dict[str, Any]) -> Scene | None:
    props = f.get("properties", {})
    assets = f.get("assets", {})
    hrefs: dict[str, str] = {}
    for key in REQUIRED_ASSETS:
        a = assets.get(key)
        if not a or not a.get("href"):
            return None
        hrefs[key] = a["href"]

    dt_raw = props.get("datetime") or props.get("start_datetime")
    if not dt_raw:
        return None
    try:
        d = _dt.datetime.fromisoformat(dt_raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None

    return Scene(
        id=f.get("id", "unknown"),
        datetime=dt_raw,
        date=d.isoformat(),
        year=d.year,
        doy=_doy(d),
        cloud_cover=float(props.get("eo:cloud_cover", 100.0)),
        epsg=props.get("proj:epsg"),
        mgrs=props.get("grid:code") or props.get("s2:mgrs_tile"),
        assets=hrefs,
        platform=props.get("platform"),
    )


def select_epochs(
    features: list[dict[str, Any]],
    start_year: int,
    end_year: int,
    *,
    season_window: int | None = None,
) -> CatalogResult:
    """
    Choose one best scene per year using season-locked, cloud-ranked selection.
    """
    season_window = season_window or settings.SEASON_WINDOW_DAYS
    notes: list[str] = []
    rejected: list[dict[str, Any]] = []

    scenes = [s for s in (_to_scene(f) for f in features) if s is not None]
    if not scenes:
        return CatalogResult([], 0, 0, ["No Sentinel-2 scenes with a complete band set were found."], [])

    # ---- 1. Pick the anchor day-of-year -----------------------------------
    # Choose the DOY whose +/- window contains usable scenes in the most years
    # (and, as a tiebreak, the lowest median cloud). This automatically finds
    # the local dry season rather than assuming one.
    years_needed = list(range(start_year, end_year + 1))
    best_anchor, best_score = None, (-1, 1e9)
    for cand in range(15, 366, 10):
        covered, clouds = 0, []
        for y in years_needed:
            pool = [
                s for s in scenes
                if s.year == y and _circular_doy_delta(s.doy, cand) <= season_window
            ]
            if pool:
                covered += 1
                clouds.append(min(p.cloud_cover for p in pool))
        if not clouds:
            continue
        med = sorted(clouds)[len(clouds) // 2]
        score = (covered, -med)
        if score > (best_score[0], -best_score[1]):
            best_anchor, best_score = cand, (covered, med)

    if best_anchor is None:
        best_anchor = scenes[0].doy
        notes.append("Could not find a consistent seasonal window; using the first available acquisition date.")
    else:
        anchor_date = (_dt.date(2024, 1, 1) + _dt.timedelta(days=best_anchor - 1))
        notes.append(
            f"Season-locked to ~{anchor_date.strftime('%d %B')} "
            f"(±{season_window} days) so all epochs share comparable phenology and sun angle."
        )

    # ---- 2. Prefer a single MGRS tile, but only if it actually covers the
    #         whole time range. An AOI that straddles a tile boundary (or sits
    #         in a swath-overlap zone) would otherwise silently lose epochs.
    tile_counts: dict[str, int] = {}
    tile_years: dict[str, set[int]] = {}
    for s in scenes:
        if not s.mgrs:
            continue
        tile_counts[s.mgrs] = tile_counts.get(s.mgrs, 0) + 1
        tile_years.setdefault(s.mgrs, set()).add(s.year)

    years_all = set(years_needed)
    # Years that ANY tile can serve — the realistic ceiling.
    coverable = {y for s in scenes for y in [s.year] if y in years_all}

    preferred_tile = None
    if tile_counts:
        # Rank tiles by how many needed years they cover, then by scene count.
        ranked = sorted(
            tile_counts,
            key=lambda t: (len(tile_years[t] & coverable), tile_counts[t]),
            reverse=True,
        )
        best = ranked[0]
        if len(tile_years[best] & coverable) >= len(coverable):
            preferred_tile = best
        else:
            missing = sorted(coverable - tile_years[best])
            notes.append(
                "The area straddles more than one Sentinel-2 tile, so epochs were drawn from "
                f"whichever tile was clearest each year (a single tile would have lost {len(missing)} "
                f"year(s): {', '.join(map(str, missing))}). All epochs are still resampled onto one "
                "common grid before comparison."
            )

    # ---- 3. Best scene per year -------------------------------------------
    selected: list[Scene] = []
    for y in years_needed:
        pool = [
            s for s in scenes
            if s.year == y and _circular_doy_delta(s.doy, best_anchor) <= season_window
        ]
        if not pool:
            # Relax the season window before giving up on the year.
            pool = [
                s for s in scenes
                if s.year == y and _circular_doy_delta(s.doy, best_anchor) <= season_window * 2
            ]
            if pool:
                rejected.append({
                    "year": y,
                    "reason": f"No scene inside the ±{season_window}-day seasonal window; "
                              f"window relaxed to ±{season_window * 2} days for this epoch.",
                })
        if not pool:
            rejected.append({"year": y, "reason": "No low-cloud Sentinel-2 scene available for this year."})
            continue

        same_tile = [s for s in pool if preferred_tile and s.mgrs == preferred_tile]
        candidates = same_tile or pool

        candidates.sort(
            key=lambda s: (
                round(s.cloud_cover, 1),
                _circular_doy_delta(s.doy, best_anchor),
            )
        )
        chosen = candidates[0]

        # Always gather same-day acquisitions from OTHER tiles. Even when a
        # single tile is available for every year, that tile may only cover
        # part of the AOI (Sentinel-2 tiles are 110 km and AOIs straddle them).
        # The raster loader mosaics these in only where the primary has no data,
        # so this is free when the primary already covers everything.
        same_day = [
            s for s in scenes
            if s.date == chosen.date and s.mgrs != chosen.mgrs and s.id != chosen.id
        ]
        seen: set[str] = set()
        for s in sorted(same_day, key=lambda x: x.cloud_cover):
            if s.mgrs and s.mgrs not in seen:
                seen.add(s.mgrs)
                chosen.companions.append(s)

        selected.append(chosen)

    selected.sort(key=lambda s: s.date)
    if preferred_tile and selected:
        tiles = {s.mgrs for s in selected}
        if len(tiles) == 1:
            notes.append(f"All epochs come from the same Sentinel-2 tile ({preferred_tile}), giving identical viewing geometry.")
        else:
            notes.append(f"Epochs span multiple tiles ({', '.join(sorted(t for t in tiles if t))}); resampled onto a common grid.")

    return CatalogResult(
        scenes=selected,
        anchor_doy=best_anchor,
        total_candidates=len(scenes),
        notes=notes,
        rejected=rejected,
    )
