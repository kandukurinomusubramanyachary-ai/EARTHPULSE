"""
EarthPulse — Feature 12: semantic similarity search.

We build a compact, physically interpretable embedding for any analysed AOI
and compare it against a library of previously analysed areas plus a set of
reference signatures.

Why not a deep model? An embedding whose dimensions are *named quantities*
(built-up fraction, change rate, water fraction, fragmentation ...) can be
explained to a user — "these areas match because both gained ~4% built-up
surface with similar patch fragmentation" — which is exactly what an
evidence-first product needs. The TRD documents the swap-in path for a
learned geospatial encoder (e.g. Clay / Prithvi embeddings) behind the same
interface.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .config import settings
from .raster import EpochStack, compute_index

STORE_PATH = os.path.join(settings.CACHE_DIR, "signatures.json")

FEATURE_NAMES = [
    "veg_fraction_start", "veg_fraction_end", "veg_delta",
    "built_fraction_start", "built_fraction_end", "built_delta",
    "water_fraction_start", "water_fraction_end", "water_delta",
    "change_rate", "cluster_density", "fragmentation", "mean_patch_ha",
]


@dataclass
class Signature:
    key: str
    label: str
    lon: float
    lat: float
    period: str
    mode: str
    vector: list[float]
    headline: str
    area_ha: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_signature(
    epochs: list[EpochStack],
    change_mask: np.ndarray,
    clusters: list[Any],
    stats: dict[str, Any],
) -> list[float]:
    """Compute the interpretable descriptor for an analysed AOI."""
    first, last = epochs[0], epochs[-1]
    valid = first.valid & last.valid
    n = max(int(valid.sum()), 1)

    def frac(arr: np.ndarray, thr: float, above: bool = True) -> float:
        sel = (arr > thr) if above else (arr < thr)
        return float((sel & valid).sum() / n)

    ndvi_a, ndvi_b = compute_index(first, "NDVI"), compute_index(last, "NDVI")
    ndbi_a, ndbi_b = compute_index(first, "NDBI"), compute_index(last, "NDBI")
    mndwi_a, mndwi_b = compute_index(first, "MNDWI"), compute_index(last, "MNDWI")

    veg_a, veg_b = frac(ndvi_a, 0.30), frac(ndvi_b, 0.30)
    blt_a, blt_b = frac(ndbi_a, 0.00), frac(ndbi_b, 0.00)
    wat_a, wat_b = frac(mndwi_a, 0.05), frac(mndwi_b, 0.05)

    change_rate = float(change_mask.sum() / n)
    px_ha = stats.get("pixel_area_ha", 0.04)
    ncl = len(clusters)
    aoi_ha = max(n * px_ha, 1e-6)
    cluster_density = ncl / (aoi_ha / 100.0)                # clusters per km²
    total_ch = float(change_mask.sum())
    fragmentation = (ncl / total_ch) if total_ch > 0 else 0.0
    mean_patch = (total_ch * px_ha / ncl) if ncl else 0.0

    return [
        veg_a, veg_b, veg_b - veg_a,
        blt_a, blt_b, blt_b - blt_a,
        wat_a, wat_b, wat_b - wat_a,
        change_rate,
        min(cluster_density / 10.0, 1.0),
        min(fragmentation * 50.0, 1.0),
        min(mean_patch / 50.0, 1.0),
    ]


# Reference signatures — archetypal patterns so similarity search returns
# something meaningful on the very first query of a session.
REFERENCE: list[dict[str, Any]] = [
    {
        "key": "ref-periurban-sprawl",
        "label": "Peri-urban sprawl archetype",
        "lon": 78.3489, "lat": 17.4401, "period": "reference", "mode": "urban_expansion",
        "headline": "Fragmented built-up growth on a former agricultural fringe: many medium patches, moderate vegetation decline.",
        "area_ha": 0.0,
        "vector": [0.42, 0.31, -0.11, 0.18, 0.29, 0.11, 0.03, 0.03, 0.00, 0.09, 0.55, 0.30, 0.35],
    },
    {
        "key": "ref-planned-township",
        "label": "Planned township archetype",
        "lon": 80.5150, "lat": 16.5150, "period": "reference", "mode": "urban_expansion",
        "headline": "Large contiguous built-up blocks appearing at once: few, very large patches with sharp boundaries.",
        "area_ha": 0.0,
        "vector": [0.55, 0.34, -0.21, 0.10, 0.27, 0.17, 0.02, 0.02, 0.00, 0.14, 0.15, 0.06, 0.85],
    },
    {
        "key": "ref-reservoir-drawdown",
        "label": "Reservoir drawdown archetype",
        "lon": 79.3120, "lat": 16.5750, "period": "reference", "mode": "water_loss",
        "headline": "Water perimeter recedes uniformly, exposing bare shoreline: one large ring-shaped change zone.",
        "area_ha": 0.0,
        "vector": [0.22, 0.26, 0.04, 0.09, 0.14, 0.05, 0.34, 0.19, -0.15, 0.13, 0.10, 0.04, 0.90],
    },
    {
        "key": "ref-forest-clearing",
        "label": "Forest clearing archetype",
        "lon": 76.1320, "lat": 11.6854, "period": "reference", "mode": "vegetation_loss",
        "headline": "Compact clearings inside dense canopy: high starting vegetation, sharp localised losses.",
        "area_ha": 0.0,
        "vector": [0.86, 0.74, -0.12, 0.04, 0.07, 0.03, 0.01, 0.01, 0.00, 0.08, 0.40, 0.22, 0.40],
    },
    {
        "key": "ref-flood-inundation",
        "label": "Flood inundation archetype",
        "lon": 88.8000, "lat": 21.9500, "period": "reference", "mode": "water_gain",
        "headline": "Water spreads across low-lying land following terrain: sprawling, irregular, high-area change.",
        "area_ha": 0.0,
        "vector": [0.51, 0.28, -0.23, 0.07, 0.06, -0.01, 0.09, 0.38, 0.29, 0.31, 0.20, 0.08, 0.80],
    },
]

# Dimensions that carry the most discriminative signal get more weight.
_WEIGHTS = np.array([
    0.7, 0.7, 1.4,
    0.7, 0.7, 1.4,
    0.7, 0.7, 1.4,
    1.2, 0.9, 0.9, 0.8,
], dtype=np.float64)


def _load_store() -> list[dict[str, Any]]:
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def _save_store(items: list[dict[str, Any]]) -> None:
    try:
        with open(STORE_PATH, "w", encoding="utf-8") as fh:
            json.dump(items[-200:], fh)
    except Exception:
        pass


def remember(sig: Signature) -> None:
    """Persist an analysed AOI so later queries can match against it."""
    items = _load_store()
    items = [i for i in items if i.get("key") != sig.key]
    items.append(sig.to_dict())
    _save_store(items)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b) * _WEIGHTS
    return float(np.sqrt(np.sum(d * d)))


def _explain_match(a: list[float], b: list[float]) -> str:
    """Name the two dimensions that agree most closely."""
    diffs = [(abs(a[i] - b[i]) * float(_WEIGHTS[i]), FEATURE_NAMES[i], a[i], b[i])
             for i in range(len(FEATURE_NAMES))]
    diffs.sort(key=lambda t: t[0])
    pretty = {
        "veg_delta": "change in vegetation cover",
        "built_delta": "change in built-up cover",
        "water_delta": "change in water extent",
        "change_rate": "proportion of area changed",
        "fragmentation": "how fragmented the change patches are",
        "mean_patch_ha": "typical patch size",
        "cluster_density": "density of change clusters",
        "veg_fraction_start": "starting vegetation cover",
        "built_fraction_start": "starting built-up cover",
        "water_fraction_start": "starting water cover",
        "veg_fraction_end": "final vegetation cover",
        "built_fraction_end": "final built-up cover",
        "water_fraction_end": "final water cover",
    }
    top = [d for d in diffs if d[1] in ("veg_delta", "built_delta", "water_delta",
                                        "change_rate", "fragmentation", "mean_patch_ha")][:2]
    if not top:
        top = diffs[:2]
    bits = [f"{pretty.get(name, name)} ({av:+.2f} vs {bv:+.2f})" for _, name, av, bv in top]
    return "Closest agreement on " + " and ".join(bits) + "."


def find_similar(vector: list[float], *, exclude_key: str | None = None, k: int = 4) -> list[dict[str, Any]]:
    """Rank reference archetypes and past analyses against this AOI."""
    v = np.array(vector, dtype=np.float64)
    pool = REFERENCE + [i for i in _load_store() if i.get("key") != exclude_key]

    scored: list[dict[str, Any]] = []
    for item in pool:
        try:
            w = np.array(item["vector"], dtype=np.float64)
        except Exception:
            continue
        if w.shape != v.shape:
            continue
        dist = _distance(v, w)
        sim = 1.0 / (1.0 + dist)
        scored.append({
            "key": item["key"],
            "label": item["label"],
            "lon": item["lon"],
            "lat": item["lat"],
            "period": item.get("period", ""),
            "mode": item.get("mode", ""),
            "headline": item.get("headline", ""),
            "area_ha": item.get("area_ha", 0.0),
            "similarity": round(sim, 4),
            "similarity_percent": round(sim * 100),
            "distance": round(dist, 4),
            "why": _explain_match(vector, item["vector"]),
            "is_reference": item["key"].startswith("ref-"),
        })

    scored.sort(key=lambda s: -s["similarity"])
    return scored[:k]
